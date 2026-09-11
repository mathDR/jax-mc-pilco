"""Methods to train Flows and generate data."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax

from jax_mc_pilco.model_learning.flow_model import FlowDynamics

# from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.training.loss import single_trajectory_loss

jax.config.update("jax_enable_x64", True)


def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    dataset_states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    dataset_actions: jtp.Float[jtp.Array, " num_episodes seq_len action_dim"],
    world_model: FlowDynamics | None = None,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 3e-4,
) -> tuple[FlowDynamics, jax.Array]:
    """Standard training loop for the world model."""

    # Should split states and actions into training and validation so we can stop
    # under a patience threshold like in fit_to_data
    key, subkey = jax.random.split(key)

    if world_model is None:
        world_model = FlowDynamics(
            key=subkey,
            state_dim=dataset_states.shape[-1],
            action_dim=dataset_actions.shape[-1],
            state_low=dataset_states.min(axis=0),
            state_high=dataset_states.max(axis=0),
        )

    # Setup Optax optimizer
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

    opt_state = optimizer.init(eqx.filter(world_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def world_train_step(
        world_model: FlowDynamics,
        _opt_state: jtp.PyTree,
        _states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""
        loss_value, grads = eqx.filter_value_and_grad(single_trajectory_loss)(world_model, _states, _actions)
        updates, opt_state = optimizer.update(grads, _opt_state, eqx.filter(world_model, eqx.is_inexact_array))
        # Apply updates via optax
        world_model = eqx.apply_updates(world_model, updates)
        return world_model, opt_state, loss_value

    num_samples = dataset_states.shape[0]
    steps_per_epoch = num_samples // batch_size
    losses = []

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        # Shuffle dataset indices each epoch
        shuffled_idx = jax.random.permutation(subkey, num_samples)

        epoch_losses = []
        for step in range(steps_per_epoch):
            # Batch slicing
            batch_idx = shuffled_idx[step * batch_size : (step + 1) * batch_size]
            b_states = dataset_states[batch_idx]
            b_actions = dataset_actions[batch_idx]

            # Step update
            world_model, opt_state, loss_val = world_train_step(world_model, opt_state, b_states, b_actions)
            epoch_losses.append(loss_val)

        print(f"Epoch {epoch + 1:02d} | Avg NLL Loss: {jnp.mean(jnp.array(epoch_losses)):.4f}")
        losses.extend(epoch_losses)

    return world_model, jnp.array(losses)
