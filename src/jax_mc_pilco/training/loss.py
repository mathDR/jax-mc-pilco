import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax

from jax_mc_pilco.model_learning.flow_model import FlowDynamics


def single_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    obs: jtp.Float[jtp.Array, " seq_len state_dim"],
    next_obs: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len action_dim"],
    key: jtp.Key[jtp.Array, ""],
) -> jtp.Float[jtp.Array, ""]:
    """Computes Hybrid loss for a single trajectory episode.
    This consists of a combination of image reconstruction
    loss and the liklihood loss from the conditional normalizing flow.
    """

    world_model = eqx.combine(world_model_params, world_model_static)
    # Initialize hidden state to zeros for the start of the sequence
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    keys = jax.random.split(key, actions.shape[0])

    # Important alignment: We have T steps of states/actions.
    # We can only compute T-1 deltas.
    # Therefore, we only run the scan on the first T-1 steps.

    def scan_fn(
        prev_hidden: jax.Array, data: tuple[jtp.Key[jtp.Array, ""], jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        subkey, prev_state, prev_action = data
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
            subkey,
            prev_state,
            prev_action,
            prev_hidden,
        )

        # Pass next_hidden forward, but record the normalized for use in log_prob
        return next_hidden, (next_hidden, next_state)

    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    states = eqx.filter_vmap(world_model.encoder)(obs.astype(jnp.float32) / 255.0)
    _, (contexts, next_states) = jax.lax.scan(scan_fn, init_hidden, (keys, states, actions))

    nll_losses = -eqx.filter_vmap(world_model.log_prob, in_axes=(0, 0))(next_states - states, contexts)
    reconstruction_losses = optax.losses.huber_loss(eqx.filter_vmap(world_model.decoder)(next_states), next_obs)

    # Balance the losses by scaling to per pixel reconstruction and per dimension log likelihood.
    return (1.0 / (64.0 * 64.0 * 3.0)) * jnp.mean(reconstruction_losses) + 1.0 * jnp.mean(
        nll_losses / (1.0 * world_model.latent_state_dim)
    )


def batched_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    next_states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len action_dim"],
    key: jtp.Key[jtp.Array, ""],
) -> jtp.Float[jtp.Array, ""]:
    """Compute the vmapped loss over all the trajectories."""
    keys = jax.random.split(key, actions.shape[0])
    loss = jax.vmap(single_trajectory_loss, in_axes=(None, None, 0, 0, 0, 0))(
        world_model_params,
        world_model_static,
        states,
        next_states,
        actions,
        keys,
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

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], action: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
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
    return pred_next_states


def imagine_trajectory_loss_with_true_actions(
    world_model: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    init_state: jax.Array,
    true_next_states: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Rolls out the policy inside the world model with known actions."""

    init_hidden = jnp.zeros(world_model.memory.hidden_size)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], action: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
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
    losses = jax.vmap(imagine_trajectory_loss_with_true_actions, in_axes=(None, 0, 0, 0, 0))(
        world_model,
        keys,
        init_states,
        next_states,
        actions,
    )
    return jnp.mean(losses)
