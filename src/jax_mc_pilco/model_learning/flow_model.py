### Equinox Module that acts as a world model given actions."""
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.bijections import AbstractBijection, Chain, Identity, RationalQuadraticSpline, Sigmoid, Stack
from flowjax.distributions import AbstractDistribution, Affine, Chain, Transformed
from flowjax.flows import masked_autoregressive_flow
from paramax import non_trainable

EPSILON = 1e-5

class CircularRationalQuadraticSpline(AbstractBijection):
    """
    Circular Rational Quadratic Spline Bijection mapping [-pi, pi] to [-pi, pi].
    """
    shape = ()
    cond_shape = None
    shift_value: float
    spline: RationalQuadraticSpline

    def __init__(self, knots: int = 8, *, shift_value: float = 0.0) -> None:
        self.shift_value = shift_value
        # We model the periodic box specifically bound tightly on [-pi, pi]
        self.spline = RationalQuadraticSpline(
            knots=knots,
            interval=float(jnp.pi),
        )

    def shift(self, x: jtp.ArrayLike) -> jtp.Array:
        return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    def transform_and_log_det(
        self,
        x: jtp.ArrayLike,
        condition: jtp.ArrayLike | None = None,
    ) -> tuple[jtp.Array, jtp.Array]:
        # Enforce periodic boundary constraints wrapping natively around the circular domain
        x_shifted = self.shift(x)

        # Forward pass through the base RationalQuadraticSpline transformer
        y_shifted, log_det_jacobian = self.spline.transform_and_log_det(
            x_shifted, condition
        )
        # Undo shift wrapping safely
        y = self.shift(y_shifted)
        return y, log_det_jacobian

    def inverse_and_log_det(
        self, y: jtp.ArrayLike, condition: jtp.ArrayLike | None = None
    ) -> tuple[jtp.Array, jtp.Array]:
        y_shifted = self.shift(y)
        x_shifted, log_det_jacobian = self.spline.inverse_and_log_det(
            y_shifted, condition
        )
        x = self.shift(x_shifted)
        return x, log_det_jacobian


class ConditionalMVN(AbstractDistribution):
    """Multivariate normal base distribution whose loc and covariance
    (via its Cholesky factor) are produced from the conditioning variable.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    tril_rows: jtp.Int[jtp.Array, " n_offdiag"]
    tril_cols: jtp.Int[jtp.Array, " n_offdiag"]

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        dim: int,
        cond_dim: int,
        width_size: int = 32,
        depth: int = 2,
    ) -> None:
        self.shape = (dim,)
        self.cond_shape = (cond_dim,)
        self.dim = dim

        n_offdiag = dim * (dim - 1) // 2
        out_size = 2 * dim + n_offdiag  # loc, chol-diag, chol-off-diag

        self.mlp = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

        rows, cols = jnp.tril_indices(dim, k=-1)
        self.tril_rows = rows
        self.tril_cols = cols

    def _loc_and_chol(
        self,
        condition: jtp.Float[jtp.Array, " cond_dim"] | None,
    ) -> tuple[jtp.Float[jtp.Array, " dim"], jtp.Float[jtp.Array, "dim dim"]]:

        out: jtp.Float[jtp.Array, " out_size"] = self.mlp(typing.cast(jtp.Array, condition))

        # Static split points (self.dim is a static Python int), so this
        # is ordinary shape-level splitting, not a dynamic gather.
        loc, diag_raw, off_diag = jnp.split(out, (self.dim, 2 * self.dim))

        diag: jtp.Float[jtp.Array, " dim"] = jax.nn.softplus(diag_raw) + 1e-4

        L: jtp.Float[jtp.Array, "dim dim"] = jnp.zeros((self.dim, self.dim))
        L = L.at[self.tril_rows, self.tril_cols].set(off_diag)
        L = L + jnp.diag(diag)

        return loc, L

    def _log_prob(
        self,
        x: jtp.Float[jtp.Array, " dim"],
        condition: jtp.Float[jtp.Array, " cond_dim"] | None = None,
    ) -> jtp.Float[jtp.Array, ""]:
        loc, L = self._loc_and_chol(condition)
        y = jax.scipy.linalg.solve_triangular(L, x - loc, lower=True)
        log_det = jnp.sum(jnp.log(jnp.diag(L)))
        quad = jnp.sum(y**2)
        return -0.5 * quad - log_det - 0.5 * self.dim * jnp.log(2 * jnp.pi)

    def _sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        condition: jtp.Float[jtp.Array, " cond_dim"] | None = None,
    ) -> jtp.Float[jtp.Array, " dim"]:
        loc, L = self._loc_and_chol(condition)
        z = jax.random.normal(key, (self.dim,))
        return loc + L @ z


class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model with GRUCell for recurrent memory.
    """

    flow: Transformed
    memory: eqx.nn.GRUCell
    state_high: jax.Array
    state_low: jax.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        state_low: jax.Array,
        state_high: jax.Array,
        deter_dim: int = 64,
        flow_layers: int = 4,
        *,
        base_flow: Transformed | None = None,
    ):
        # Context consists of current state and taken action
        cond_dim = state_dim + action_dim

        self.state_low = jnp.broadcast_to(state_low, (state_dim,))
        self.state_high = jnp.broadcast_to(state_high, (state_dim,))

        key, base_key, flow_key = jax.random.split(key, 3)

        self.memory = eqx.nn.GRUCell(input_size=cond_dim, hidden_size=deter_dim, key=key)

        if base_flow is None:
            # Define a base distribution matching the state delta dimension
            base_dist = ConditionalMVN(base_key, dim=state_dim, cond_dim=deter_dim)
            base_flow = masked_autoregressive_flow(
                key=flow_key,
                base_dist=base_dist,
                cond_dim=deter_dim,
                transformer=RationalQuadraticSpline(
                    knots=8,
                    interval=(min(self.state_low).item(), max(self.state_high).item()),
                ),
                nn_width=256,
                nn_depth=2,
                flow_layers=flow_layers,
            )
        mixed_bijectors = [
            RationalQuadraticSpline(knots=8, interval=3.0),      # Position (bounded domain spline)
            CircularRationalQuadraticSpline(knots=8),            # Angle 1 (your custom periodic spline)
            CircularRationalQuadraticSpline(knots=8),            # Angle 2 (your custom periodic spline)
            Identity(),                                          # Velocity pos (unconstrained)
            Identity(),                                          # Velocity angle 1 (unconstrained)
            Identity(),                                          # Velocity angle 2 (unconstrained)
        ]

        # This creates a single bijection acting element-wise across the dimensions
        final_constraints = Stack(mixed_bijectors)
        squash = Chain([
            Sigmoid(shape=(state_dim,)),  # type: ignore  # R -> (0, 1)  # noqa: PGH003
            Affine(
                loc=self.state_low-EPSILON,
                scale=2*EPSILON + self.state_high - self.state_low,
            ),  # (0, 1) -> (low, high)
        ])

        self.flow = Transformed(base_flow, Chain([final_constraints, non_trainable(squash)]))


    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition state."""
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        # Update hidden state
        next_hidden = self.memory(context, prev_hidden)
        next_state = self.flow.sample(key, condition=next_hidden)
        return jnp.clip(next_state, self.state_low, self.state_high), next_hidden

    def log_prob(
        self,
        state: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state given the context."""
        eps = EPSILON * (self.state_high - self.state_low)
        clipped_state = jnp.clip(state, self.state_low + eps, self.state_high - eps)
        return self.flow.log_prob(clipped_state, condition=context)

    def predict_reward(self, state: jax.Array) -> jax.Array:
        # Returns a scalar value for the given latent state representation
        x = state[0]
        y = jnp.arctan2(state[2], state[4])
        v1 = state[5]
        v2 = state[6]

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        # Condition: y <= 1 -> invert it so True means "alive" (y > 1)
        is_alive = y > 1.0

        # jax.lax.cond(pred, true_fn, false_fn, *operands)
        alive_bonus = jax.lax.cond(
            is_alive,
            lambda _: 10.0,  # If True (y > 1), return 10.0
            lambda _: 0.0,  # If False (y <= 1), return 0.0
            operand=None,
        )
        return jnp.array(alive_bonus - dist_penalty - vel_penalty)
