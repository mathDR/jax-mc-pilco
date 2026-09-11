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
    base_flow: Transformed
    state_dim: int
    action_dim: int
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
        self.state_dim = state_dim
        self.action_dim = action_dim

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

            self.flow = coupling_flow(
                key=key,
                base_dist=base_dist,
                cond_dim=cond_dim,
                nn_width=256,
                nn_depth=2,
                flow_layers=flow_layers,
            )
        else:
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
