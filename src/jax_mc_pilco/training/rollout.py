"""Multi-step imagination rollout."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from typeguard import typechecked as typechecker

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor

NUM_BURN_IN_STEPS = 10

@typechecker
def functional_rollout(
    world_model: FlowDynamics,
    action_model: FlowActor,
    init_state: jtp.Float[jtp.Array, " state_dim"],
    init_hidden: jtp.Float[jtp.Array, " hidden_dim"],
    horizon: int,
    key: jtp.Key[jtp.Array, ""],
) -> tuple[jtp.Float[jtp.Array, "horizon state_dim"], jtp.Float[jtp.Array, "horizon action_dim"]]:
    """Imagines a trajectory using independent world and action model PyTrees.

    Args:
        world_model: FlowRSSM instance (Equinox module PyTree).
        action_model: eqx.nn.MLP instance (Equinox module PyTree).
        init_state: Starting raw state array.
        init_hidden: Starting GRU hidden state array.
        horizon: Number of steps to imagine forward.
        key: PRNGKey for the trajectory.

    Returns:
        imagined_states: Logged states over the horizon steps.
        imagined_actions: Logged actions over the horizon steps.
    """
    # Generate an array of unique random keys for every horizon step
    keys: jtp.Key[jtp.Array, ""] = jax.random.split(key, horizon)

    # Define the single-step pure transition function for jax.lax.scan
    def scan_fn(
        carry: tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " hidden_dim"]],
        step_key: jtp.Key[jtp.Array, ""],
        action_key: jtp.Key[jtp.Array, ""],
    ) -> tuple[
        tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " hidden_dim"]],
        tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " action_dim"]],
    ]:
        current_state, current_hidden = carry

        # Action model decides the action based on state and hidden context
        action = action_model.sample_action(action_key, current_state)

        # World model predicts the next state using its internal logic
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
            current_state, action, current_hidden, step_key
        )

        next_carry = (next_state, next_hidden)
        step_outputs = (next_state, action)

        return next_carry, step_outputs

    # 3. Initial loop carry state
    init_carry = (init_state, init_hidden)

    # 4. Run the fast scanned loop across the horizon steps
    _, (imagined_states, imagined_actions) = jax.lax.scan(scan_fn, init_carry, keys, length=horizon)

    return imagined_states, imagined_actions

def single_wm_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    ss_prob: float,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len state_dim"],
    next_states: jtp.Float[jtp.Array, " seq_len state_dim"],
) -> jtp.Float[jtp.Array, ""]:
    """Computes NLL loss for a single trajectory episode. Recall
    that episodes are all the same length, padded by nans. So
    we need to take this into consideration.
    """

    world_model = eqx.combine(world_model_params, world_model_static)
    # Initialize hidden state to zeros for the start of the sequence
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    prev_state_mask = ~jnp.isnan(states)
    next_state_mask = ~jnp.isnan(next_states)
    state_mask = jnp.logical_and(prev_state_mask, next_state_mask)

    # Assumption is that the single mask works for all inputs (including actions).
    valid_steps = jnp.all(state_mask, axis=-1)

    # Replace padded values before passing them into the model.
    safe_states = jnp.where(valid_steps[:, None], states, 0.0)
    safe_actions = jnp.where(valid_steps[:, None], actions, 0.0)
    safe_next_states = jnp.where(valid_steps[:, None], next_states, 0.0)

    def run_scan(
        init_carry: tuple[jax.Array, jax.Array, jtp.Key[jtp.Array, ""]],
        xs: tuple[jax.Array, jax.Array, jax.Array],
        bernoulli_prob: float,
    ) -> tuple[tuple[jax.Array, jax.Array, jtp.Key[jtp.Array, ""]], jax.Array]:

        def step(
            carry: tuple[jax.Array, jax.Array, jtp.Key[jtp.Array, ""]],
            data: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array, jtp.Key[jtp.Array, ""]], jax.Array]:
            current_hidden, s_prev_hat, step_key = carry
            prev_state, prev_action, is_valid = data

            next_step_key, scheduled_sampling_key, sample_key = jax.random.split(
                step_key, 3
            )
            use_pred = jax.random.bernoulli(scheduled_sampling_key, bernoulli_prob)
            context = jnp.where(
                use_pred,
                jax.lax.stop_gradient(s_prev_hat),
                prev_state,
            )

            s_next_hat, next_hidden = jax.lax.cond(
                is_valid,
                # True branch: executes the world model memory update
                lambda: world_model.predict_next_state_and_hidden(
                    sample_key,
                    context,
                    prev_action,
                    current_hidden,
                ),
                # False branch: returns the current hidden state unchanged
                lambda: (s_prev_hat, current_hidden),
            )
            # Pass next_hidden forward, but also record it as the step output
            return (next_hidden, s_next_hat, next_step_key), next_hidden

        return jax.lax.scan(step, init_carry, xs)

    last_carry, _ = run_scan(
        (init_hidden, safe_states[0], key),
        (
            safe_states[:NUM_BURN_IN_STEPS],
            safe_actions[:NUM_BURN_IN_STEPS],
            valid_steps[:NUM_BURN_IN_STEPS],
        ),
        bernoulli_prob=0.0,
    )

    last_hidden, _, rollout_key = last_carry
    _, hiddens = run_scan(
        (last_hidden, safe_states[NUM_BURN_IN_STEPS], rollout_key),
        (
            safe_states[NUM_BURN_IN_STEPS:],
            safe_actions[NUM_BURN_IN_STEPS:],
            valid_steps[NUM_BURN_IN_STEPS:],
        ),
        bernoulli_prob=ss_prob,
    )

    # delta_s = safe_next_states - safe_states
    nll_losses = -eqx.filter_vmap(world_model.log_prob, in_axes=(0, 0, 0))(
        safe_next_states[NUM_BURN_IN_STEPS:], safe_states[NUM_BURN_IN_STEPS:], hiddens
    )

    loss_mask = valid_steps[NUM_BURN_IN_STEPS:]
    return jnp.mean(nll_losses, where=loss_mask)


def batched_wm_trajectory_loss(
    world_model_params: FlowDynamics,
    world_model_static: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    ss_prob: float,
    states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    next_states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
) -> jtp.Float[jtp.Array, ""]:
    keys = jax.random.split(key, states.shape[0])
    loss = eqx.filter_vmap(
        single_wm_trajectory_loss, in_axes=(None, None, 0, None, 0, 0, 0)
    )(
        world_model_params,
        world_model_static,
        keys,
        ss_prob,
        states,
        actions,
        next_states,
    )
    return jnp.mean(loss)
