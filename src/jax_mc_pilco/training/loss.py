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
    next_states: jtp.Float[jtp.Array, " seq_len action_dim"],
) -> jtp.Float[jtp.Array, ""]:
    """Computes NLL loss for a single trajectory episode."""

    world_model = eqx.combine(world_model_params, world_model_static)

    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    contexts, _ = world_model.memory.scan_sequence(states, actions)

    delta_s_targets = next_states - states

    nll_losses = -world_model.log_prob(delta_s_targets, context=contexts)

    return jnp.mean(nll_losses)


def batched_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    states: jtp.Float[jtp.Array, " batch_dim seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " batch_dim seq_len action_dim"],
    next_states: jtp.Float[jtp.Array, " batch_dim seq_len action_dim"],
) -> jtp.Float[jtp.Array, ""]:
    loss = jax.vmap(single_trajectory_loss, in_axes=(None, None, 0, 0, 0))(
        world_model_params,
        world_model_static,
        states,
        actions,
        next_states,
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

    _, init_fast_weights = world_model.memory.init_state()

    def step(carry: tuple, action: jax.Array) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        state, hidden, key = carry
        # Move world model forward
        key, wm_key = jax.random.split(key)
        next_state, next_fast_weights = world_model.predict_next_state_and_hidden(
            wm_key,
            state,
            action,
            hidden,
        )

        return (next_state, next_fast_weights, key), next_state

    _, pred_next_states = jax.lax.scan(step, (init_state, init_fast_weights, key), actions)
    return jnp.mean(jnp.linalg.norm(pred_next_states - true_next_states))


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
    losses = jax.vmap(imagine_trajectory_with_true_actions, in_axes=(None, 0, 0, 0, 0))(
        world_model,
        keys,
        init_states,
        next_states,
        actions,
    )
    return jnp.mean(losses)
