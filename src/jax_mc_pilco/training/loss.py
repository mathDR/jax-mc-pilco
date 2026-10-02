import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.policy_learning.critic import DreamerCritic


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

    delta_s_targets = eqx.filter_vmap(world_model.encoder)(next_states) - eqx.filter_vmap(world_model.encoder)(states)

    nll_losses = -world_model.log_prob(delta_s_targets, context=contexts)

    return jnp.mean(nll_losses)


<<<<<<< HEAD
<<<<<<< HEAD
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
=======
def imagine_trajectory(
>>>>>>> parent of b821211 (getting rollout loss to work)
=======
def imagine_trajectory(
>>>>>>> parent of b821211 (getting rollout loss to work)
    key: jtp.Key[jtp.Array, ""],
    world_model: FlowDynamics,
    actor: FlowActor,
    init_state: jax.Array,
    horizon: int,
) -> tuple[jax.Array, jax.Array, jax.Array, float]:
    """Rolls out the policy inside the world model for a fixed horizon."""

    _, init_fast_weights = world_model.memory.init_state()

    def step(carry, key):
        state, hidden = carry
        combined_latent = jnp.concatenate([state, hidden], axis=-1)

        # Propose action using the Flow Actor
        act_key, wm_key = jax.random.split(key)
        action, log_prob = actor.sample_action_and_log_prob(act_key, context=combined_latent)

<<<<<<< HEAD
        # Move world model forward
<<<<<<< HEAD
        key, wm_key = jax.random.split(key)
        next_state, next_fast_weights = world_model.predict_next_state_and_hidden(
=======
=======
    def step(carry, key):
        state, hidden = carry
        combined_latent = jnp.concatenate([state, hidden], axis=-1)

        # Propose action using the Flow Actor
        act_key, wm_key = jax.random.split(key)
        action, log_prob = actor.sample_action_and_log_prob(act_key, context=combined_latent)

        # Move world model forward
>>>>>>> parent of b821211 (getting rollout loss to work)
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
>>>>>>> parent of b821211 (getting rollout loss to work)
            wm_key,
            state,
            action,
            hidden,
        )

<<<<<<< HEAD
<<<<<<< HEAD
        return (next_state, next_fast_weights, key), next_state

    _, pred_next_states = jax.lax.scan(step, (init_state, init_fast_weights, key), actions)
    return jnp.mean(jnp.linalg.norm(pred_next_states - true_next_states))
=======
=======
>>>>>>> parent of b821211 (getting rollout loss to work)
        # We also need to predict rewards inside the world model to optimize the actor.
        # Assuming your world model has a reward predictor module:
        reward = world_model.predict_reward(state)

        return (next_state, next_hidden), (combined_latent, action, log_prob, reward)

    keys = jax.random.split(key, horizon)
    _, (latents, actions, log_probs, rewards) = jax.lax.scan(step, (init_state, init_hidden), keys)
    return latents, actions, log_probs, rewards
<<<<<<< HEAD
>>>>>>> parent of b821211 (getting rollout loss to work)
=======
>>>>>>> parent of b821211 (getting rollout loss to work)


def actor_loss(
    actor: jtp.PyTree,
    static: FlowActor,
    world_model: FlowDynamics,
    init_state: jax.Array,
    horizon: int,
    ent_coef: float,
    key: jtp.Key[jtp.Array, ""],
) -> jax.Array:
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> parent of b821211 (getting rollout loss to work)
    """Computes the actor BPTT loss combining imagined values and flow entropy."""
    actor_model = eqx.combine(actor, static)
    traj_key, _ = jax.random.split(key)
    _, _, log_probs, rewards = imagine_trajectory(traj_key, world_model, actor_model, init_state, horizon)

    # 1. Estimate values of imagined states

    # 2. Dreamer-v2/v3 objective: Maximize a mix of rewards and values.
    # For simplicity, we use the value of the next step as the bootstrap target.
    # In a full setup, you would compute lambda-returns across the horizon.
    actor_objective = rewards

    # 3. Entropy Regularization
    # Since entropy H = E[-log_prob], maximizing entropy means minimizing log_prob
    entropy_approx = -log_probs

    # Total objective to maximize
    total_objective = actor_objective + ent_coef * entropy_approx

    # Return negative mean for gradient descent minimization
    return -jnp.mean(total_objective)


# def critic_loss(
#     critic: jtp.PyTree,
#     static: DreamerCritic,
#     latents: jax.Array,
#     target_returns: jax.Array,
# ) -> jax.Array:
#     """Fits the critic to target returns (e.g., lambda-returns calculated from imagination)."""
#     critic_model = eqx.combine(critic, static)
#     predicted_values = jax.vmap(critic_model, in_axes=0)(latents)
#     return jnp.mean(jnp.square(predicted_values - target_returns))
<<<<<<< HEAD
>>>>>>> parent of b821211 (getting rollout loss to work)
=======
>>>>>>> parent of b821211 (getting rollout loss to work)
