import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp

from jax_mc_pilco.model_learning.flow_model import FlowDynamics


def single_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len action_dim"],
) -> jtp.Float[jtp.Array, ""]:
    """Computes NLL loss for a single trajectory episode."""

    world_model = eqx.combine(world_model_params, world_model_static)
    # Initialize hidden state to zeros for the start of the sequence
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    # Important alignment: We have T steps of states/actions.
    # We can only compute T-1 deltas.
    # Therefore, we only run the scan on the first T-1 steps.
    input_states = states[:-1]
    input_actions = actions[:-1]

    def scan_fn(current_hidden: jax.Array, data: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        prev_state, prev_action = data

        # Mirroring your internal predict_next_state_and_hidden context definition
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        next_hidden = world_model.memory(context, current_hidden)

        # Pass next_hidden forward, but also record it as the step output
        return next_hidden, next_hidden

    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    _, contexts = jax.lax.scan(scan_fn, init_hidden, (input_states, input_actions))

    delta_s_targets = jnp.diff(states, axis=0)

    nll_losses = -world_model.log_prob(delta_s_targets, context=contexts)

    return jnp.mean(nll_losses)


def batched_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len action_dim"],
) -> jtp.Float[jtp.Array, ""]:
    loss = jax.vmap(single_trajectory_loss, in_axes=(None, None, 0, 0))(
        world_model_params,
        world_model_static,
        states,
        actions,
    )
    return jnp.mean(loss)


def imagine_trajectory_with_true_actions(
    world_model: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    init_state: jax.Array,
    true_next_states: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Rolls out the policy inside the world model with known actions."""

    init_hidden = jnp.zeros(world_model.memory.hidden_size)

    def step(carry: tuple, action: jax.Array) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        state, hidden, key = carry
        # Move world model forward
        key, wm_key = jax.random.split(key)
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
            wm_key,
            state,
            action,
            hidden,
        )

        return (next_state, next_hidden, key), next_state
    _, pred_next_states = jax.lax.scan(step, (init_state, init_hidden, key), actions)
    return jnp.mean(jnp.linalg.norm(pred_next_states-true_next_states))


def batched_rollout_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    init_states: jax.Array,
    next_states: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    world_model = eqx.combine(world_model_params, world_model_static)
    keys = jax.random.split(key, next_states.shape[0])
    losses = jax.vmap(imagine_trajectory_with_true_actions, in_axes=(None,0,0,0,0))(
        world_model,
        keys,
        init_states,
        next_states,
        actions,
    )
    return jnp.mean(losses)
