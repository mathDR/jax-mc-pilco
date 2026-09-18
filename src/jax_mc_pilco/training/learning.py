"""Methods to train Flows and generate data."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
import tqdm

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.training.loss import batched_rollout_loss, batched_trajectory_loss
from jax_mc_pilco.training.training_utils import (
    count_fruitless,
    get_batches,
    train_val_split,
)

jax.config.update("jax_enable_x64", True)


def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " num_episodes seq_len action_dim"],
    world_model: FlowDynamics | None = None,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 512,
    max_epochs: int = 1000,
    learning_rate: float = 3e-4,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[FlowDynamics, dict[str, list]]:
    """Standard training loop for the world model."""

    key, subkey = jax.random.split(key)

    if world_model is None:
        world_model = FlowDynamics(
            key=subkey,
            state_dim=states.shape[-1],
            action_dim=actions.shape[-1],
            state_low=states.min(axis=tuple(range(states.ndim - 1))),
            state_high=states.max(axis=tuple(range(states.ndim - 1))),
        )

    # Setup Optax optimizer
    if optimizer is None:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    params, static = eqx.partition(
        world_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    data = (states, actions)

    @eqx.filter_jit
    def world_train_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        _opt_state: jtp.PyTree,
        _states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""

        loss_value, grads = eqx.filter_value_and_grad(batched_trajectory_loss)(
            params,
            static,
            _states,
            _actions,
        )
        updates, opt_state = optimizer.update(grads, _opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_pct)
    losses: dict[str, list] = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jax.random.split(key, 3)
        train_data = [jax.random.permutation(subkeys[0], a) for a in train_data]
        val_data = [jax.random.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            params, opt_state, loss_i = world_train_step(params, static, opt_state, *batch)
            batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jax.random.split(key)
            loss_i = eqx.filter_jit(batched_trajectory_loss)(params, static, *batch)
            batch_losses.append(loss_i)
        losses["val"].append(jnp.mean(jnp.array(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    model = eqx.combine(params, static)
    return model, losses


def rollout_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    next_states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " num_episodes seq_len action_dim"],
    world_model: FlowDynamics | None = None,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 512,
    max_epochs: int = 1000,
    learning_rate: float = 3e-4,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[FlowDynamics, dict[str, list]]:
    """Standard training loop for the world model."""

    key, subkey = jax.random.split(key)

    if world_model is None:
        world_model = FlowDynamics(
            key=subkey,
            state_dim=states.shape[-1],
            action_dim=actions.shape[-1],
            state_low=states.min(axis=tuple(range(states.ndim - 1))),
            state_high=states.max(axis=tuple(range(states.ndim - 1))),
        )

    # Setup Optax optimizer
    if optimizer is None:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    params, static = eqx.partition(
        world_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    data = (states[:,0,:], next_states, actions)

    @eqx.filter_jit
    def rollout_train_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        _opt_state: jtp.PyTree,
        _init_states: jtp.Float[jtp.Array, "batch_size state_dim"],
        _next_states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""

        loss_value, grads = eqx.filter_value_and_grad(batched_rollout_loss)(
            params,
            static,
            key,
            _init_states,
            _next_states,
            _actions,
        )
        updates, opt_state = optimizer.update(grads, _opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_pct)
    losses: dict[str, list] = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jax.random.split(key, 3)
        train_data = [jax.random.permutation(subkeys[0], a) for a in train_data]
        val_data = [jax.random.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            params, opt_state, loss_i = rollout_train_step(params, static, opt_state, *batch)
            batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jax.random.split(key)
            loss_i = eqx.filter_jit(batched_rollout_loss)(params, static, subkey, *batch)
            batch_losses.append(loss_i)
        losses["val"].append(jnp.mean(jnp.array(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    model = eqx.combine(params, static)
    return model, losses
