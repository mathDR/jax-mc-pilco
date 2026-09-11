import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp

from jax_mc_pilco.model_learning.flow_model import FlowDynamics

# from jax_mc_pilco.policy_learning.action_flows import FlowActor

def single_trajectory_loss(
    world_model: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"], 
    actions: jtp.Float[jtp.Array, " seq_len action_dim"]
) -> jtp.Float[jtp.Array, ""]:
    """Computes NLL loss for a single trajectory episode."""

    # Initialize hidden state to zeros for the start of the sequence
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    def scan_fn(
        current_hidden: jax.Array,
        data: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        prev_state, prev_action = data

        # Mirroring your internal predict_next_state_and_hidden context definition
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        next_hidden = world_model.memory(context, current_hidden)

        # Pass next_hidden forward, but also record it as the step output
        return next_hidden, next_hidden
    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    _, contexts = jax.lax.scan(scan_fn, init_hidden, (states, actions))

    delta_s_targets = jnp.diff(states, axis=0)

    nll_losses = -world_model.log_prob(delta_s_targets, context=contexts[:-1])

    return jnp.mean(nll_losses)
