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

    def scan_fn(current_hidden: jax.Array, data: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        prev_state, prev_action = data

        # Mirroring your internal predict_next_state_and_hidden context definition
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        next_hidden = world_model.memory(context, current_hidden)

        # Pass next_hidden forward, but also record it as the step output
        return next_hidden, next_hidden

    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    _, contexts = jax.lax.scan(scan_fn, init_hidden, (states, actions))

    actual_next_states = states[1:]
    current_states = states[:-1]
    delta_s_targets = actual_next_states - current_states

    nll_losses = -world_model.log_prob(delta_s_targets, context=contexts)

    return jnp.mean(nll_losses)

@eqx.filter_value_and_grad
def compute_batch_loss(
    world_model: FlowDynamics,
    batch_states: jtp.Float[jtp.Array, " batch_size seq_len state_dim"], 
    batch_actions: jtp.Float[jtp.Array, " batch_size seq_len action_dim"]
) -> jtp.Float[jtp.Array, ""]:
    """Batches the trajectory loss across a full training batch using vmap."""
    vmap_loss = jax.vmap(single_trajectory_loss, in_axes=(None, 0, 0))
    losses = vmap_loss(world_model, batch_states, batch_actions)
    return jnp.mean(losses)
