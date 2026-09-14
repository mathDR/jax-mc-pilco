"""Methods to train Flows and generate data."""
from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax

from jax_mc_pilco.model_learning.flow_model import FlowDynamics

# from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.training.loss import single_trajectory_loss
from jax_mc_pilco.training.training_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)

jax.config.update("jax_enable_x64", True)


def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    dataset_states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    dataset_actions: jtp.Float[jtp.Array, " num_episodes seq_len action_dim"],
    world_model: FlowDynamics | None = None,
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    val_pct: float = 0.1,
) -> tuple[FlowDynamics, jax.Array, jax.Array]:
    """Standard training loop for the world model."""

    # Should split states and actions into training and validation so we can stop
    # under a patience threshold like in fit_to_data
    key, subkey = jax.random.split(key)

    num_samples = dataset_states.shape[0]
    steps_per_epoch = num_samples // batch_size

    n_val = int(num_samples * val_pct)

    # Shuffle indices
    permuted_indices = jax.random.permutation(subkey, num_samples)

    # Shuffle the arrays
    shuffled_states = dataset_states[permuted_indices]
    shuffled_actions = dataset_actions[permuted_indices]

    # Split into train and validation sets
    val_states, train_states = jnp.split(shuffled_states, [n_val])
    val_actions, train_actions = jnp.split(shuffled_actions, [n_val])

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
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""], jax.Array]:
        """Performs a single functional gradient step update."""

        (loss_value, final_hidden), grads = eqx.filter_value_and_grad(single_trajectory_loss)(world_model, _states, _actions)
        updates, opt_state = optimizer.update(grads, _opt_state, eqx.filter(world_model, eqx.is_inexact_array))

        world_model = eqx.apply_updates(world_model, updates)
        return world_model, opt_state, loss_value, final_hidden

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
            world_model, opt_state, loss_val, final_hidden = world_train_step(world_model, opt_state, b_states, b_actions)
            epoch_losses.append(loss_val)

        print(f"Epoch {epoch + 1:02d} | Avg NLL Loss: {jnp.mean(jnp.array(epoch_losses)):.4f}")
        losses.extend(epoch_losses)

    return world_model, final_hidden, jnp.array(losses)

def fit_to_data(
    key: jtp.Key[jtp.Array, ""],
    model: jtp.PyTree,  # Custom losses may support broader types than AbstractDistribution
    data: jtp.ArrayLike | Iterable[jtp.ArrayLike],
    loss_fn: Callable,
    *,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[jtp.PyTree, jax.Array]:
    r"""Train a PyTree (e.g. a distribution) to samples from the target.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        data: An array or tuple of arrays passed as positional arguments to the
            loss function (see documentation for ``loss_fn``). Commonly this is a
            single array for unconditional density estimation, or two arrays
            ``(target, condition)`` for conditional density estimation.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        loss_fn: Loss function. The signature should be of the form
            ``(params, static, *arrays, key)``. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    data = (data,) if isinstance(data, jtp.ArrayLike) else data
    data = tuple(jnp.asarray(a) for a in data)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            params, opt_state, loss_i = step(
                params,
                static,
                *batch,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=subkey,
            )
            batch_losses.append(loss_i)
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = eqx.filter_jit(loss_fn)(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    model = eqx.combine(params, static)
    return model, losses