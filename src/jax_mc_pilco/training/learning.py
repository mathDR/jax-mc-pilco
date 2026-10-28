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


def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    datadict: dict,
    world_model: FlowDynamics,
    *,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 64,
    max_epochs: int = 1000,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[FlowDynamics, dict[str, list]]:
    """Standard training loop for the world model."""

    key, subkey = jax.random.split(key)

    # Setup Optax optimizer
    if optimizer is None:
        schedule = optax.sgdr_schedule([
            {"init_value": 1e-3, "peak_value": 1e-3, "warmup_steps": 0, "decay_steps": 5000, "end_value": 1e-5},
        ])
        optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),  # Clip gradients at 5.0 global norm
            optax.adam(learning_rate=schedule),  # Apply Adam using our schedule
        )
    params, static = eqx.partition(
        world_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    data = (datadict["obs"][:, :-1, ...], datadict["obs"][:, 1:, ...], datadict["action"][:, :-1, ...])

    @eqx.filter_jit
    def world_train_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        _opt_state: jtp.PyTree,
        _key: jtp.Key[jtp.Array, ""],
        _states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _next_states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""

        loss_value, grads = eqx.filter_value_and_grad(batched_trajectory_loss)(
            params,
            static,
            _states,
            _next_states,
            _actions,
            _key,
        )
        updates, opt_state = optimizer.update(grads, _opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_pct)
    losses: dict[str, list] = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)
    jit_batched_trajectory_loss = eqx.filter_jit(batched_trajectory_loss)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jax.random.split(key, 3)
        train_data = [jax.random.permutation(subkeys[0], a) for a in train_data]
        val_data = [jax.random.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jax.random.split(key)
            params, opt_state, loss_i = world_train_step(params, static, opt_state, subkey, *batch)
            batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jax.random.split(key)
            loss_i = jit_batched_trajectory_loss(params, static, subkey, *batch)
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
    world_model: FlowDynamics,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 512,
    accumulation_steps: int = 4,
    max_epochs: int = 1000,
    learning_rate: float = 3e-4,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[FlowDynamics, dict[str, list]]:
    """Standard training loop for the world model."""

    # Ensure batch_size can be cleanly divided by accumulation_steps
    assert batch_size % accumulation_steps == 0, (
        f"batch_size ({batch_size}) must be divisible by accumulation_steps ({accumulation_steps})"
    )
    micro_batch_size = batch_size // accumulation_steps

    key, subkey = jax.random.split(key)

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

    data = (states[:, 0, :], next_states, actions)

    @eqx.filter_jit
    def rollout_train_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        _opt_state: jtp.PyTree,
        _init_states: jtp.Float[jtp.Array, "batch_size state_dim"],
        _next_states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> tuple[jtp.PyTree, optax.OptState, jtp.Float[jtp.Array, ""]]:

        # Reshape the full batch into a stack of micro-batches
        micro_init_states = jnp.reshape(_init_states, (accumulation_steps, micro_batch_size, -1))
        micro_next_states = jnp.reshape(_next_states, (accumulation_steps, micro_batch_size, _next_states.shape[1], -1))
        micro_actions = jnp.reshape(_actions, (accumulation_steps, micro_batch_size, _actions.shape[1], -1))

        # Sequential loop over micro-batches using jax.lax.scan (low VRAM overhead)
        def scan_fn(carry, micro_batch):
            _m_init, _m_next, _m_act = micro_batch

            # Compute loss and gradients for a single micro-batch
            loss_value, grads = eqx.filter_value_and_grad(batched_rollout_loss)(
                params, static, key, _m_init, _m_next, _m_act
            )

            # Accumulate grads and losses across steps
            acc_grads, acc_loss = carry
            new_acc_grads = jax.tree.map(lambda g, ag: ag + g / accumulation_steps, grads, acc_grads)
            new_acc_loss = acc_loss + loss_value / accumulation_steps

            return (new_acc_grads, new_acc_loss), None

        # Initialize accumulation structures matching PyTree shapes
        init_grads = jax.tree.map(jnp.zeros_like, params)
        init_loss = jnp.array(0.0)

        # Perform sequential loop over micro-batches
        (total_grads, mean_loss), _ = jax.lax.scan(
            scan_fn, (init_grads, init_loss), (micro_init_states, micro_next_states, micro_actions)
        )

        # 3. Apply optimizer update once using the accumulated gradients
        updates, opt_state = optimizer.update(total_grads, _opt_state, params=params)
        params = eqx.apply_updates(params, updates)

        return params, opt_state, mean_loss

    @eqx.filter_jit
    def rollout_val_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        _subkey: jtp.Key[jtp.Array, ""],
        _init_states: jtp.Float[jtp.Array, "batch_size state_dim"],
        _next_states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len action_dim"],
    ) -> jtp.Float[jtp.Array, ""]:
        return batched_rollout_loss(params, static, _subkey, _init_states, _next_states, _actions)

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
            loss_i = rollout_val_step(params, static, subkey, *batch)
            # Convert to float scalar
            batch_losses.append(loss_i.item())
        losses["val"].append(float(jnp.mean(jnp.array(batch_losses)).item()))

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    model = eqx.combine(params, static)
    return model, losses
