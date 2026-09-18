### Equinox Module that acts as a world model given actions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow


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

        self.memory = eqx.nn.GRUCell(input_size=cond_dim, hidden_size=deter_dim, key=key)

        if base_flow is None:
            # Define a base distribution matching the state delta dimension
            base_dist = MultivariateNormal(
                loc=jnp.zeros(state_dim, dtype=float),
                covariance=jnp.eye(state_dim, dtype=float),
            )
            base_flow = coupling_flow(
                key=key,
                base_dist=base_dist,
                cond_dim=deter_dim,
                nn_width=256,
                nn_depth=2,
                flow_layers=flow_layers,
            )
        self.flow = base_flow

        # TODO: utilize a chained sigmoid and affine to constrain the flow
        # (somehow) so deltas are within bounds.

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition delta and adds it to s_t."""
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        # Update hidden state
        next_hidden = self.memory(context, prev_hidden)
        delta_s = self.flow.sample(key, condition=next_hidden)
        return jnp.clip(prev_state + delta_s, self.state_low, self.state_high), next_hidden

    def log_prob(
        self,
        delta_s: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state delta given the context."""
        return self.flow.log_prob(delta_s, condition=context)

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
