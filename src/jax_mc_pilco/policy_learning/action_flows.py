### Equinox Module that acts as a world model given actions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import paramax
from beartype import beartype
from flowjax.bijections import Affine, Chain, RationalQuadraticSpline, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import masked_autoregressive_flow

EPSILON = 1e-4


class ActionFlow(eqx.Module):
    """State-conditioned behavior model used to sample imagined actions."""

    flow: Transformed
    action_low: jtp.Array
    action_high: jtp.Array
    state_mean: jtp.Array
    state_std: jtp.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        flow_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ):
        cond_dim = state_dim + hidden_dim

        action_low = jnp.full((action_dim,), -jnp.pi / 2.0)
        action_high = jnp.full((action_dim,), jnp.pi / 2.0)
        self.action_low = paramax.non_trainable(action_low)
        self.action_high = paramax.non_trainable(action_high)

        if state_mean is None:
            state_mean = jnp.zeros((state_dim,))
        if state_std is None:
            state_std = jnp.ones((state_dim,))

        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(jnp.where(state_std < 1e-6, 1.0, state_std))

        if base_flow is None:
            unbounded_flow = masked_autoregressive_flow(
                key=key,
                base_dist=MultivariateNormal(loc=jnp.zeros(action_dim), covariance=jnp.eye(action_dim)),
                cond_dim=cond_dim,
                flow_layers=flow_layers,
                invert=True,
                nn_depth=2,
                transformer=RationalQuadraticSpline(knots=12, interval=8),
            )
            eps = 1e-3

            # Pad bounds by eps so true [-pi/2, pi/2] values stay strictly inside (0, 1)
            padded_low = action_low - eps
            padded_scale = (action_high - action_low) + 2.0 * eps
            bound_bijection = Chain([
                Sigmoid(shape=(action_dim,)), # type: ignore[call-arg]
                Affine(loc=padded_low, scale=padded_scale),
            ])
            base_flow = Transformed(
                unbounded_flow,
                paramax.non_trainable(bound_bijection),
            )

        self.flow = base_flow

    @jtp.jaxtyped(typechecker=beartype)
    def normalize_state(self, state: jax.Array) -> jax.Array:
        ret_array: jax.Array = (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(self.state_std)
        return ret_array

    @jtp.jaxtyped(typechecker=beartype)
    def _build_context(self, state: jax.Array, hidden: jax.Array) -> jax.Array:
        norm_state = self.normalize_state(state)
        return jnp.concatenate([norm_state, hidden], axis=-1)

    @jtp.jaxtyped(typechecker=beartype)
    def _clip_action(self, raw_sample: jax.Array) -> jax.Array:
        clipped = jnp.clip(
            raw_sample,
            paramax.unwrap(self.action_low),
            paramax.unwrap(self.action_high),
        )
        return raw_sample + jax.lax.stop_gradient(clipped - raw_sample)

    @jtp.jaxtyped(typechecker=beartype)
    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Sample a bounded action conditioned on the current context."""
        context = self._build_context(state, hidden)
        raw_sample = self.flow.sample(key, (), context)
        return self._clip_action(raw_sample)

    @jtp.jaxtyped(typechecker=beartype)
    def sample_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample both a bounded action and its log density."""
        context = self._build_context(state, hidden)
        raw_sample, log_prob = self.flow.sample_and_log_prob(key, (), context)
        action = self._clip_action(raw_sample)
        return action, log_prob

    @jtp.jaxtyped(typechecker=beartype)
    def log_prob(
        self,
        action: jax.Array,
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Return the behavior-model log density of an observed action."""
        context = self._build_context(state, hidden)
        safe_action = jnp.clip(
            action,
            paramax.unwrap(self.action_low),
            paramax.unwrap(self.action_high),
        )
        return self.flow.log_prob(safe_action, context)


class Critic(eqx.Module):
    network: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int = 256,
        *,
        key: jtp.Key[jtp.Array, ""],
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        in_dim = state_dim + rnn_hidden_dim

        # Zero-initialize output layer (DreamerV3 convention) so initial V(s, h) ≈ 0
        out_layer = eqx.nn.Linear(mlp_hidden_dim, 1, key=k3)
        assert out_layer.bias is not None  # noqa: S101

        out_layer = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            out_layer,
            (jnp.zeros_like(out_layer.weight), jnp.zeros_like(out_layer.bias)),
        )

        self.network = eqx.nn.Sequential((
            eqx.nn.Linear(in_dim, mlp_hidden_dim, key=k1),
            eqx.nn.LayerNorm(mlp_hidden_dim),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=k2),
            eqx.nn.LayerNorm(mlp_hidden_dim),
            eqx.nn.Lambda(jax.nn.silu),
            out_layer,
        ))

    def __call__(
        self,
        norm_state: jtp.Float[jtp.Array, " state_dim"],
        hidden: jtp.Float[jtp.Array, " rnn_hidden_dim"],
    ) -> jtp.Float[jtp.Array, ""]:
        context = jnp.concatenate([norm_state, hidden], axis=-1)
        raw_value = jnp.squeeze(self.network(context), axis=-1)
        return -jax.nn.softplus(raw_value)  # Strictly <= 0, smooth gradients everywhere
