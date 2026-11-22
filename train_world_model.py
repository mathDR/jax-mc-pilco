"""Threat Model Supervised Training Script (Interactive Version).

Loads saved trajectory data, partitions into train/val splits,
trains the learning.LearnedController's MLP on ECEF acceleration labels,
applies early stopping, and visualizes the loss curves and 3D rollouts.
"""

# ruff: noqa: E402
import argparse
import logging
import os
import pathlib
import warnings
from collections.abc import Sequence

# Parse CLI flags (including --x64/--no-x64) and configure JAX precision
# before any JAX arrays, operations, or JAX-dependent libraries (e.g., equinox, optax)
# are imported or initialized.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--x64",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable 64-bit JAX precision (use --no-x64 for 32-bit). Defaults to x64.",
)
_precision_args, _ = _parser.parse_known_args()

import jax

jax.config.update("jax_enable_x64", _precision_args.x64)

logger = logging.getLogger(__name__)

import hypersonic
from hypersonic import (
    data,
    hyperlog,
    script_tools,
)

SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIRECTORY / "config" / "scripts.default.yaml"


### Equinox Module that acts as a world model given actions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
import tqdm
from flowjax.bijections import Affine, Chain, RationalQuadraticSpline, Sigmoid
from flowjax.distributions import (
    MultivariateNormal,
    Transformed,
)
from flowjax.flows import masked_autoregressive_flow
from flowjax.train.train_utils import (
    count_fruitless,
    train_val_split,
)

NUM_BURN_IN_STEPS = 10


class Critic(eqx.Module):
    network: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        rnn_hidden_dim: int,
        mlp_hidden_dim: int = 256,
        *,
        key: jtp.Key[jtp.Array, ""],
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        in_dim = state_dim + rnn_hidden_dim

        # Zero-initialize output layer (DreamerV3 convention) so initial V(s, h) ≈ 0
        out_layer = eqx.nn.Linear(mlp_hidden_dim, 1, key=k3)
        out_layer = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            out_layer,
            (jnp.zeros_like(out_layer.weight), jnp.zeros_like(out_layer.bias)),
        )

        self.network = eqx.nn.Sequential(
            (
                eqx.nn.Linear(in_dim, mlp_hidden_dim, key=k1),
                eqx.nn.LayerNorm(mlp_hidden_dim),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, key=k2),
                eqx.nn.LayerNorm(mlp_hidden_dim),
                eqx.nn.Lambda(jax.nn.silu),
                out_layer,
            )
        )

    def __call__(
        self,
        norm_state: jtp.Float[jtp.Array, "state_dim"],
        hidden: jtp.Float[jtp.Array, "rnn_hidden_dim"],
    ) -> jtp.Float[jtp.Array, ""]:
        context = jnp.concatenate([norm_state, hidden], axis=-1)
        raw_value = jnp.squeeze(self.network(context), axis=-1)
        return -jax.nn.softplus(raw_value)  # Strictly <= 0, smooth gradients everywhere


def compute_lambda_returns(
    rewards: jtp.Float[jtp.Array, "B H"],
    values: jtp.Float[jtp.Array, "B H_plus_1"],
    discounts: jtp.Float[jtp.Array, "B H"],
    lambda_: float = 0.95,
) -> jtp.Float[jtp.Array, "B H"]:
    """Compute TD(lambda) targets backward across imagined horizon H."""
    # Scan backward over time axis (transpose to [H, B] for jax.lax.scan)
    rewards_T = jnp.swapaxes(rewards, 0, 1)
    discounts_T = jnp.swapaxes(discounts, 0, 1)
    next_values_T = jnp.swapaxes(values[:, 1:], 0, 1)

    def backward_step(
        next_return: jtp.Float[jtp.Array, "B"],
        inputs: tuple[
            jtp.Float[jtp.Array, "B"],
            jtp.Float[jtp.Array, "B"],
            jtp.Float[jtp.Array, "B"],
        ],
    ) -> tuple[jtp.Float[jtp.Array, "B"], jtp.Float[jtp.Array, "B"]]:
        r_t, gamma_t, v_next = inputs
        target_t = r_t + gamma_t * ((1.0 - lambda_) * v_next + lambda_ * next_return)
        return target_t, target_t

    _, returns_T = jax.lax.scan(
        backward_step,
        init=values[:, -1],  # Bootstrap from V(s_H, h_H)
        xs=(rewards_T, discounts_T, next_values_T),
        reverse=True,
    )
    return jnp.swapaxes(returns_T, 0, 1)


def get_batches(
    arrays: Sequence[jtp.Array],
    batch_size: int,
    traj_size: int,
    key: jtp.Key[jtp.Array, ""] | None = None,
) -> tuple[jtp.Array, ...]:
    """Split padded trajectories into shuffled batches of valid chunks.

    Each returned array has shape:
        (n_batches, batch_size, traj_size, *array.shape[2:])

    A chunk is retained only if its first `NUM_BURN_IN_STEPS + 1` timesteps
    are finite across all input arrays (since padding is at the tail).
    """
    if not arrays:
        return ()
    if batch_size <= 0 or traj_size <= 0:
        raise ValueError("batch_size and traj_size must be positive.")

    min_valid_steps = NUM_BURN_IN_STEPS + 1
    if traj_size < min_valid_steps:
        raise ValueError(
            f"traj_size ({traj_size}) must be >= NUM_BURN_IN_STEPS + 1 ({min_valid_steps})."
        )

    B, T = arrays[0].shape[:2]
    if any(arr.shape[:2] != (B, T) for arr in arrays):
        raise ValueError("All arrays must share leading (B, T) dimensions.")

    chunks_per_traj = T // traj_size
    if chunks_per_traj == 0:
        raise ValueError(
            f"Sequence length T ({T}) is shorter than traj_size ({traj_size})."
        )

    # 1. Reshape each (B, T, ...) array into (B * chunks_per_traj, traj_size, ...)
    usable_T = chunks_per_traj * traj_size
    chunks = tuple(
        arr[:, :usable_T].reshape(-1, traj_size, *arr.shape[2:]) for arr in arrays
    )

    # 2. A chunk is valid iff every array is finite across all feature dims for at least `min_valid_steps`
    per_array_valid = [
        jnp.all(jnp.isfinite(c), axis=tuple(range(2, c.ndim))) for c in chunks
    ]
    valid_timesteps = jnp.logical_and.reduce(jnp.stack(per_array_valid, axis=0), axis=0)
    valid_chunk_mask = jnp.sum(valid_timesteps, axis=1) >= min_valid_steps

    valid_chunks = tuple(c[valid_chunk_mask] for c in chunks)
    n_valid = valid_chunks[0].shape[0]

    # 3. Optionally shuffle valid chunks before batching
    if key is not None and n_valid > 0:
        perm = jax.random.permutation(key, n_valid)
        valid_chunks = tuple(c[perm] for c in valid_chunks)

    # 4. Drop remainder and reshape into (n_batches, batch_size, traj_size, ...)
    n_batches = n_valid // batch_size
    usable = n_batches * batch_size
    return tuple(
        c[:usable].reshape(n_batches, batch_size, traj_size, *c.shape[2:])
        for c in valid_chunks
    )


class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model with GRUCell for recurrent memory.
    """

    flow: Transformed
    memory: eqx.nn.GRUCell
    state_high: jtp.Array
    state_low: jtp.Array
    state_mean: jtp.Array
    state_std: jtp.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        state_low: jax.Array,
        state_high: jax.Array,
        hidden_dim: int = 128,
        flow_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ):
        # Context consists of current normalized state and taken action
        cond_dim = state_dim + action_dim

        self.state_low = paramax.non_trainable(
            jnp.broadcast_to(state_low, (state_dim,))
        )
        self.state_high = paramax.non_trainable(
            jnp.broadcast_to(state_high, (state_dim,))
        )

        if state_mean is None:
            state_mean = jnp.zeros((state_dim,))
        if state_std is None:
            state_std = jnp.ones((state_dim,))

        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(
            jnp.where(state_std < 1e-6, 1.0, state_std)
        )

        memory_key, norm_flow_key = jax.random.split(key, 2)

        self.memory = eqx.nn.GRUCell(
            input_size=cond_dim, hidden_size=hidden_dim, key=memory_key
        )
        if base_flow is None:
            base_flow = masked_autoregressive_flow(
                key=norm_flow_key,
                base_dist=MultivariateNormal(
                    loc=jnp.zeros(state_dim), covariance=jnp.eye(state_dim)
                ),
                cond_dim=hidden_dim,
                flow_layers=flow_layers,
                invert=True,
                nn_depth=2,
            )
        self.flow = base_flow

    def normalize_state(self, state: jax.Array) -> jax.Array:
        return (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(
            self.state_std
        )

    def unnormalize_state(self, norm_state: jax.Array) -> jax.Array:
        return norm_state * paramax.unwrap(self.state_std) + paramax.unwrap(
            self.state_mean
        )

    def step_hidden(
        self,
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        """Advance recurrent hidden state using normalized state and action."""
        norm_state = self.normalize_state(prev_state)
        context = jnp.concatenate([norm_state, prev_action])
        return self.memory(context, prev_hidden)

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition delta and adds it to s_t."""
        norm_state = self.normalize_state(prev_state)
        next_hidden = self.memory(
            jnp.concatenate([norm_state, prev_action]), prev_hidden
        )

        # No stop_gradient here: gradients flow through the world model's inputs
        # back to `prev_action` (and thus `ActionFlow`), while world_model
        # weights remain frozen because they are not passed to the optimizer.
        norm_state_delta = self.flow.sample(key, (), next_hidden)

        next_state = jnp.clip(
            self.unnormalize_state(norm_state + norm_state_delta),
            paramax.unwrap(self.state_low),
            paramax.unwrap(self.state_high),
        )
        return next_state, next_hidden

    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        next_state, _ = self.predict_next_state_and_hidden(
            key, prev_state, prev_action, prev_hidden
        )
        return next_state

    def log_prob(
        self,
        prev_state: jax.Array,
        next_state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the normalized state delta given hidden."""
        norm_state_delta = self.normalize_state(next_state) - self.normalize_state(
            prev_state
        )
        return self.flow.log_prob(norm_state_delta, hidden)


class ActionFlow(eqx.Module):
    """State-conditioned behavior model used to sample imagined actions."""

    flow: Transformed
    action_low: jtp.Array
    action_high: jtp.Array
    state_mean: jtp.Array
    state_std: jtp.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        flow_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ):
        cond_dim = state_dim + hidden_dim

        action_low = jnp.full((action_dim,), -jnp.pi / 2.0)
        action_high = jnp.full((action_dim,), jnp.pi / 2.0)
        self.action_low = paramax.non_trainable(action_low)
        self.action_high = paramax.non_trainable(action_high)

        if state_mean is None:
            state_mean = jnp.zeros((state_dim,))
        if state_std is None:
            state_std = jnp.ones((state_dim,))

        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(
            jnp.where(state_std < 1e-6, 1.0, state_std)
        )

        if base_flow is None:
            unbounded_flow = masked_autoregressive_flow(
                key=key,
                base_dist=MultivariateNormal(
                    loc=jnp.zeros(action_dim), covariance=jnp.eye(action_dim)
                ),
                cond_dim=cond_dim,
                flow_layers=flow_layers,
                invert=True,
                nn_depth=2,
                transformer=RationalQuadraticSpline(knots=12, interval=8),
            )
            eps = 1e-3

            # Pad bounds by eps so true [-pi/2, pi/2] values stay strictly inside (0, 1)
            padded_low = action_low - eps
            padded_scale = (action_high - action_low) + 2.0 * eps
            bound_bijection = Chain(
                [
                    Sigmoid(shape=(action_dim,)),
                    Affine(loc=padded_low, scale=padded_scale),
                ]
            )
            base_flow = Transformed(
                unbounded_flow,
                paramax.non_trainable(bound_bijection),
            )

        self.flow = base_flow

    def normalize_state(self, state: jax.Array) -> jax.Array:
        return (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(
            self.state_std
        )

    def _build_context(self, state: jax.Array, hidden: jax.Array) -> jax.Array:
        norm_state = self.normalize_state(state)
        return jnp.concatenate([norm_state, hidden], axis=-1)

    def _clip_action(self, raw_sample: jax.Array) -> jax.Array:
        clipped = jnp.clip(
            raw_sample,
            paramax.unwrap(self.action_low),
            paramax.unwrap(self.action_high),
        )
        return raw_sample + jax.lax.stop_gradient(clipped - raw_sample)

    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Sample a bounded action conditioned on the current context."""
        context = self._build_context(state, hidden)
        raw_sample = self.flow.sample(key, (), context)
        return self._clip_action(raw_sample)

    def sample_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample both a bounded action and its log density."""
        context = self._build_context(state, hidden)
        raw_sample, log_prob = self.flow.sample_and_log_prob(key, (), context)
        action = self._clip_action(raw_sample)
        return action, log_prob

    def log_prob(
        self,
        action: jax.Array,
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Return the behavior-model log density of an observed action."""
        context = self._build_context(state, hidden)
        return self.flow.log_prob(action, context)


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

    def run_scan(init_carry, xs, bernoulli_prob: float):

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


def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    next_states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    world_model: FlowDynamics,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 64,
    val_batch_size: int = 16,
    traj_size: int = 100,
    max_epochs: int = 1000,
    learning_rate: optax.ScalarOrSchedule = 3e-4,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[FlowDynamics, dict[str, list]]:
    """Standard training loop for the world model."""

    key, subkey = jax.random.split(key)

    # Setup Optax optimizer
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
        )
    teacher_forcing_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.0),
            optax.linear_schedule(init_value=0.0, end_value=0.99, transition_steps=100),
        ],
        boundaries=[100],
    )
    params, static = eqx.partition(
        world_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    data = (states, actions, next_states)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def world_train_step(
        params: jtp.PyTree,
        static: FlowDynamics,
        key: jtp.Key[jtp.Array, ""],
        _opt_state: jtp.PyTree,
        ss_prob: float,
        _states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _next_states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
    ) -> tuple[FlowDynamics, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""
        loss_value, grads = eqx.filter_value_and_grad(batched_wm_trajectory_loss)(
            params,
            static,
            key,
            ss_prob,
            _states,
            _actions,
            _next_states,
        )
        updates, opt_state = optimizer.update(grads, _opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_pct)
    losses: dict[str, list] = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)
    jitted_batched_trajectory_loss = eqx.filter_jit(batched_wm_trajectory_loss)

    global_step: int = 0

    for _ in loop:
        # Shuffle data
        key, *subkeys = jax.random.split(key, 3)
        train_data = [jax.random.permutation(subkeys[0], a) for a in train_data]
        val_data = [jax.random.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        train_batch_losses = []
        for batch in zip(
            *get_batches(train_data, batch_size, traj_size + NUM_BURN_IN_STEPS),
            strict=True,
        ):
            key, subkey = jax.random.split(key)
            ss_prob = teacher_forcing_schedule(global_step)
            params, opt_state, loss_i = world_train_step(
                params, static, subkey, opt_state, ss_prob, *batch
            )
            train_batch_losses.append(loss_i)
            global_step += 1
        losses["train"].append(jnp.mean(jnp.array(train_batch_losses)).item())

        # Val epoch
        val_batch_losses = []
        for batch in zip(
            *get_batches(val_data, val_batch_size, traj_size + NUM_BURN_IN_STEPS),
            strict=True,
        ):
            key, subkey = jax.random.split(key)
            loss_i = jitted_batched_trajectory_loss(params, static, subkey, 0.0, *batch)
            val_batch_losses.append(loss_i)
        losses["val"].append(jnp.mean(jnp.array(val_batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    model = eqx.combine(params, static)
    return model, losses


def single_act_trajectory_loss(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len state_dim"],
) -> jtp.Float[jtp.Array, ""]:
    """Computes NLL loss of the policy for a single trajectory episode.
    Recall that episodes are all the same length, padded by nans. So
    we need to take this into consideration.
    """
    action_flow = eqx.combine(action_params, action_static)

    # Initialize hidden state to zeros for the start of the sequence
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    state_mask = ~jnp.isnan(states)

    # Assumption is that the single mask works for all inputs (including actions).
    valid_steps = jnp.all(state_mask, axis=-1)

    # Replace padded values before passing them into the model.
    safe_states = jnp.where(valid_steps[:, None], states, 0.0)
    safe_actions = jnp.where(valid_steps[:, None], actions, 0.0)

    def scan_fn(
        hidden: jax.Array,
        data: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:

        prev_state, prev_action, is_valid = data

        next_hidden = jax.lax.cond(
            is_valid,
            # True branch: executes the world model memory update
            lambda: world_model.memory(
                jnp.concatenate([prev_state, prev_action]), hidden
            ),
            # False branch: returns the current hidden state unchanged
            lambda: hidden,
        )

        # Pass next_hidden forward, but also record it as the step output
        return next_hidden, next_hidden

    # Get the sequence of contexts (shape: [seq_len - 1, hidden_dim])
    last_hidden, _ = jax.lax.scan(
        scan_fn,
        init_hidden,
        (
            safe_states[:NUM_BURN_IN_STEPS],
            safe_actions[:NUM_BURN_IN_STEPS],
            valid_steps[:NUM_BURN_IN_STEPS],
        ),
    )

    _, hiddens = jax.lax.scan(
        scan_fn,
        last_hidden,
        (
            safe_states[NUM_BURN_IN_STEPS:],
            safe_actions[NUM_BURN_IN_STEPS:],
            valid_steps[NUM_BURN_IN_STEPS:],
        ),
    )

    nll_losses = -eqx.filter_vmap(action_flow.log_prob, in_axes=(0, 0, 0))(
        safe_actions[NUM_BURN_IN_STEPS:], safe_states[NUM_BURN_IN_STEPS:], hiddens
    )

    loss_mask = valid_steps[NUM_BURN_IN_STEPS:]
    return jnp.mean(nll_losses, where=loss_mask)


def batched_act_trajectory_loss(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: FlowDynamics,
    states: jtp.Float[jtp.Array, " seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " seq_len state_dim"],
) -> jtp.Float[jtp.Array, ""]:
    """Batched version of actor training loss."""
    loss = eqx.filter_vmap(
        single_act_trajectory_loss, in_axes=(None, None, None, 0, 0)
    )(
        action_params,
        action_static,
        world_model,
        states,
        actions,
    )
    return jnp.mean(loss)


def policy_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, " num_episodes seq_len state_dim"],
    world_model: FlowDynamics,
    action_flow: ActionFlow,
    optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 64,
    val_batch_size: int = 16,
    traj_size: int = 100,
    max_epochs: int = 1000,
    learning_rate: float = 3e-4,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[ActionFlow, dict[str, list]]:
    """Standard training loop for the policy (using a trained world model)."""

    key, subkey = jax.random.split(key)

    # Setup Optax optimizer
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
        )
    act_params, act_static = eqx.partition(
        action_flow,
        filter_spec=lambda x: eqx.is_inexact_array(x),
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = act_params
    opt_state = optimizer.init(act_params)

    data = (states, actions)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def act_train_step(
        params: jtp.PyTree,
        _opt_state: jtp.PyTree,
        wm: FlowDynamics,
        _states: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
        _actions: jtp.Float[jtp.Array, "batch_size seq_len state_dim"],
    ) -> tuple[ActionFlow, optax.OptState, jtp.Float[jtp.Array, ""]]:
        """Performs a single functional gradient step update."""
        loss_value, grads = eqx.filter_value_and_grad(batched_act_trajectory_loss)(
            params,
            act_static,
            wm,
            _states,
            _actions,
        )
        updates, opt_state = optimizer.update(grads, _opt_state, params=params)
        new_act_params = eqx.apply_updates(params, updates)
        return new_act_params, opt_state, loss_value

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_pct)
    losses: dict[str, list] = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)
    jitted_batched_act_trajectory_loss = eqx.filter_jit(batched_act_trajectory_loss)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jax.random.split(key, 3)
        train_data = [jax.random.permutation(subkeys[0], a) for a in train_data]
        val_data = [jax.random.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        train_batch_losses = []
        for batch in zip(
            *get_batches(train_data, batch_size, traj_size + NUM_BURN_IN_STEPS),
            strict=True,
        ):
            act_params, opt_state, loss_i = act_train_step(
                act_params, opt_state, world_model, *batch
            )
            train_batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(train_batch_losses)).item())

        # Val epoch
        val_batch_losses = []
        for batch in zip(
            *get_batches(val_data, val_batch_size, traj_size + NUM_BURN_IN_STEPS),
            strict=True,
        ):
            loss_i = jitted_batched_act_trajectory_loss(
                act_params, act_static, world_model, *batch
            )
            val_batch_losses.append(loss_i)
        losses["val"].append(jnp.mean(jnp.array(val_batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = act_params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else act_params
    model = eqx.combine(params, act_static)
    return model, losses


def imagination_trajectory(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    initial_states: jax.Array,
    initial_actions: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Roll out an imagined trajectory of length `horizon` after burn-in.

    Args:
        initial_states: Shape [NUM_BURN_IN_STEPS + 1, state_dim] (raw physical states).
        initial_actions: Shape [NUM_BURN_IN_STEPS + 1, action_dim] (observed actions).

    Returns:
        states: [horizon + 1, state_dim] starting at initial_states[-1].
        actions: [horizon, action_dim] sampled from action_flow.
        hiddens: [horizon + 1, hidden_dim] starting at post-burn-in hidden state.
        log_probs: [horizon] log densities of sampled actions.
    """
    action_flow = eqx.combine(action_params, action_static)
    hidden_dim = world_model.memory.hidden_size
    init_hidden = jnp.zeros((hidden_dim,))

    # 1. Burn-in on observed (state, action) pairs to warm up recurrent hidden state
    def burn_in_fn(
        hidden: jax.Array,
        step_inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        prev_state, prev_action = step_inputs
        next_hidden = world_model.step_hidden(prev_state, prev_action, hidden)
        return next_hidden, None

    burn_in_hidden, _ = jax.lax.scan(
        burn_in_fn,
        init_hidden,
        (initial_states[:NUM_BURN_IN_STEPS], initial_actions[:NUM_BURN_IN_STEPS]),
    )

    # 2. Sample the first imagined action at the end of burn-in (t = 0 of imagination)
    start_state = initial_states[-1]
    key, first_action_key = jax.random.split(key)
    first_action, first_log_prob = action_flow.sample_and_log_prob(
        first_action_key, start_state, burn_in_hidden
    )

    # 3. Imagine forward for `horizon` transitions
    def scan_fn(
        carry: tuple[jax.Array, jax.Array, jax.Array, jtp.Key[jtp.Array, ""]],
        _: None,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jtp.Key[jtp.Array, ""]],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        prev_state, prev_action, prev_hidden, step_key = carry
        next_key, state_key, action_key = jax.random.split(step_key, 3)

        # Advance world model (normalizes prev_state internally, returns raw next_state)
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
            state_key,
            prev_state,
            prev_action,
            prev_hidden,
        )

        # Sample next action & log_prob from policy (normalizes next_state internally)
        next_action, next_log_prob = action_flow.sample_and_log_prob(
            action_key, next_state, next_hidden
        )

        return (next_state, next_action, next_hidden, next_key), (
            next_state,
            next_action,
            next_hidden,
            next_log_prob,
        )

    _, (future_states, future_actions, future_hiddens, future_log_probs) = jax.lax.scan(
        scan_fn,
        (start_state, first_action, burn_in_hidden, key),
        xs=None,
        length=horizon,
    )

    # Prepend t=0 state & hidden so states/hiddens have length [horizon + 1]
    states = jnp.concatenate([start_state[None], future_states], axis=0)
    hiddens = jnp.concatenate([burn_in_hidden[None], future_hiddens], axis=0)

    # Keep the `horizon` actions [a_0, ..., a_{H-1}] that generated transitions 0..H-1
    actions = jnp.concatenate([first_action[None], future_actions[:-1]], axis=0)
    log_probs = jnp.concatenate([first_log_prob[None], future_log_probs[:-1]], axis=0)

    return states, actions, hiddens, log_probs


def batched_imagination_trajectory(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    states: jtp.Float[jtp.Array, "num_episodes burn_in_len state_dim"],
    actions: jtp.Float[jtp.Array, "num_episodes burn_in_len action_dim"],
) -> tuple[
    jtp.Float[jtp.Array, "num_episodes horizon_plus_1 state_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon action_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon_plus_1 hidden_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon"],
]:
    """Generate a batch of imagined trajectories from burn-in windows."""
    keys = jax.random.split(key, states.shape[0])
    return eqx.filter_vmap(
        imagination_trajectory, in_axes=(None, None, None, 0, None, 0, 0)
    )(
        action_params,
        action_static,
        world_model,
        keys,
        horizon,
        states,
        actions,
    )


def compute_step_rewards(
    pred_states: jtp.Float[jtp.Array, "B H state_dim"],
    pred_actions: jtp.Float[jtp.Array, "B H action_dim"],
    target_states: jtp.Float[jtp.Array, "B H state_dim"],
    target_actions: jtp.Float[jtp.Array, "B H action_dim"],
    world_model: FlowDynamics,
    huber_delta: float = 1.0,
) -> tuple[jtp.Float[jtp.Array, "B H"], jtp.Bool[jtp.Array, "B H"]]:
    """Compute per-step negative Huber loss as reward, plus valid step mask."""
    valid_state = jnp.isfinite(target_states).all(axis=-1)
    valid_action = jnp.isfinite(target_actions).all(axis=-1)
    valid_step = valid_state & valid_action  # [B, H]

    safe_target_states = jnp.where(valid_state[..., None], target_states, 0.0)
    safe_target_actions = jnp.where(valid_action[..., None], target_actions, 0.0)

    # Normalize states so all dimensions share unit scale
    norm_pred = jax.vmap(jax.vmap(world_model.normalize_state))(pred_states)
    norm_target = jax.vmap(jax.vmap(world_model.normalize_state))(safe_target_states)

    state_loss_t = jnp.mean(
        optax.losses.huber_loss(norm_pred, norm_target, delta=huber_delta),
        axis=-1,
    )  # [B, H]
    action_loss_t = jnp.mean(
        optax.losses.huber_loss(pred_actions, safe_target_actions, delta=huber_delta),
        axis=-1,
    )  # [B, H]

    # Negative loss = reward to maximize; zero out padded steps
    rewards = jnp.where(valid_step, -(state_loss_t + action_loss_t), 0.0)
    return rewards, valid_step


def batched_actor_critic_imagination_loss(
    actor_params: ActionFlow,
    actor_static: ActionFlow,
    critic_params: Critic,
    target_critic_params: Critic,
    critic_static: Critic,
    world_model: FlowDynamics,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    states: jtp.Float[jtp.Array, "num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coef: float = 1e-3,
    huber_delta: float = 1.0,
) -> tuple[jtp.Float[jtp.Array, ""], dict[str, jax.Array]]:
    """Compute joint actor and critic losses using an EMA target critic."""
    critic = eqx.combine(critic_params, critic_static)
    target_critic = eqx.combine(target_critic_params, critic_static)

    burn_in_states = states[:, : NUM_BURN_IN_STEPS + 1, :]
    burn_in_actions = actions[:, : NUM_BURN_IN_STEPS + 1, :]

    # 1. Roll out `horizon` imagined transitions from s_0
    pred_states, pred_actions, pred_hiddens, pred_log_probs = (
        batched_imagination_trajectory(
            actor_params,
            actor_static,
            world_model,
            key,
            horizon,
            burn_in_states,
            burn_in_actions,
        )
    )

    # 2. Align predicted transitions [1..H] with ground-truth targets
    pred_future_states = pred_states[:, 1:, :]
    target_states = states[
        :, NUM_BURN_IN_STEPS + 1 : NUM_BURN_IN_STEPS + 1 + horizon, :
    ]
    target_actions = actions[:, NUM_BURN_IN_STEPS : NUM_BURN_IN_STEPS + horizon, :]

    rewards, valid_step = compute_step_rewards(
        pred_states=pred_future_states,
        pred_actions=pred_actions,
        target_states=target_states,
        target_actions=target_actions,
        world_model=world_model,
        huber_delta=huber_delta,
    )

    # 3. Evaluate live Critic for V_0..V_{H-1} and EMA Target Critic for V_0..V_H
    norm_states = jax.vmap(jax.vmap(world_model.normalize_state))(pred_states)
    live_values = jax.vmap(jax.vmap(critic))(
        norm_states[:, :-1, :], pred_hiddens[:, :-1, :]
    )  # [B, H]
    target_values = jax.vmap(jax.vmap(target_critic))(
        norm_states, pred_hiddens
    )  # [B, H + 1]

    discounts = jnp.where(valid_step, gamma, 0.0)

    # 4. Compute TD(lambda) targets using the slow EMA target critic
    lambda_returns = compute_lambda_returns(
        rewards=rewards,
        values=target_values,
        discounts=discounts,
        lambda_=lambda_,
    )  # [B, H]

    valid_count = jnp.maximum(jnp.sum(valid_step), 1.0)

    # 5. Critic loss: fit live V(s_t, h_t) to stop_gradient(lambda_returns)
    critic_err = optax.losses.huber_loss(
        live_values,
        jax.lax.stop_gradient(lambda_returns),
        delta=huber_delta,
    )
    critic_loss = jnp.sum(jnp.where(valid_step, critic_err, 0.0)) / valid_count

    # 6. Actor loss: maximize lambda_returns + entropy bonus
    policy_value_loss = (
        -jnp.sum(jnp.where(valid_step, lambda_returns, 0.0)) / valid_count
    )
    entropy_loss = entropy_coef * (
        jnp.sum(jnp.where(valid_step, pred_log_probs, 0.0)) / valid_count
    )
    actor_loss = policy_value_loss + entropy_loss

    total_loss = actor_loss + critic_loss
    metrics = {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "mean_return": jnp.sum(jnp.where(valid_step, lambda_returns, 0.0))
        / valid_count,
        "mean_step_loss": -jnp.sum(jnp.where(valid_step, rewards, 0.0)) / valid_count,
        "entropy": -jnp.sum(jnp.where(valid_step, pred_log_probs, 0.0)) / valid_count,
    }
    return total_loss, metrics


def imagination_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jtp.Float[jtp.Array, "num_episodes seq_len state_dim"],
    actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
    world_model: FlowDynamics,
    action_flow: ActionFlow,
    critic: Critic,
    actor_optimizer: optax.GradientTransformation | None = None,
    critic_optimizer: optax.GradientTransformation | None = None,
    batch_size: int = 64,
    val_batch_size: int = 16,
    traj_size: int = 15,
    max_epochs: int = 1000,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    critic_warmup_epochs: int = 5,
    target_tau: float = 0.02,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coef: float = 1e-3,
    val_pct: float = 0.1,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[ActionFlow, Critic, dict[str, list]]:
    """Actor-critic imagination training loop with an EMA target critic."""

    if actor_optimizer is None:
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(actor_lr)
        )
    if critic_optimizer is None:
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(critic_lr)
        )

    partition_spec = dict(
        filter_spec=lambda x: eqx.is_inexact_array(x),
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    act_params, act_static = eqx.partition(action_flow, **partition_spec)
    crit_params, crit_static = eqx.partition(critic, **partition_spec)
    target_crit_params = crit_params

    best_act_params = act_params
    best_crit_params = crit_params

    act_opt_state = actor_optimizer.init(act_params)
    crit_opt_state = critic_optimizer.init(crit_params)

    def loss_fn(
        params: tuple[jtp.PyTree, jtp.PyTree],
        target_c_params: jtp.PyTree,
        wm: FlowDynamics,
        step_key: jtp.Key[jtp.Array, ""],
        horizon: int,
        _states: jtp.Float[jtp.Array, "B T state_dim"],
        _actions: jtp.Float[jtp.Array, "B T action_dim"],
    ) -> tuple[jtp.Float[jtp.Array, ""], dict[str, jax.Array]]:
        a_params, c_params = params
        return batched_actor_critic_imagination_loss(
            actor_params=a_params,
            actor_static=act_static,
            critic_params=c_params,
            target_critic_params=target_c_params,
            critic_static=crit_static,
            world_model=wm,
            key=step_key,
            horizon=horizon,
            states=_states,
            actions=_actions,
            gamma=gamma,
            lambda_=lambda_,
            entropy_coef=entropy_coef,
        )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def train_step(
        a_params: jtp.PyTree,
        c_params: jtp.PyTree,
        target_c_params: jtp.PyTree,
        a_opt_state: jtp.PyTree,
        c_opt_state: jtp.PyTree,
        wm: FlowDynamics,
        step_key: jtp.Key[jtp.Array, ""],
        horizon: int,
        update_actor: jax.Array,
        _states: jtp.Float[jtp.Array, "B T state_dim"],
        _actions: jtp.Float[jtp.Array, "B T action_dim"],
    ) -> tuple[
        jtp.PyTree,
        jtp.PyTree,
        jtp.PyTree,
        jtp.PyTree,
        jtp.PyTree,
        dict[str, jax.Array],
    ]:
        (_, metrics), (act_grads, crit_grads) = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(
            (a_params, c_params),
            target_c_params,
            wm,
            step_key,
            horizon,
            _states,
            _actions,
        )

        # 1. Update live critic
        crit_updates, new_crit_opt_state = critic_optimizer.update(
            crit_grads, c_opt_state, params=c_params
        )
        new_crit_params = eqx.apply_updates(c_params, crit_updates)

        # 2. Polyak / EMA update for slow target critic
        new_target_crit_params = optax.incremental_update(
            new_crit_params, target_c_params, step_size=target_tau
        )

        # 3. Update actor only after critic warmup epochs
        def apply_actor_step(
            operand: tuple[jtp.PyTree, jtp.PyTree, jtp.PyTree],
        ) -> tuple[jtp.PyTree, jtp.PyTree]:
            p, opt_s, g = operand
            updates, new_opt_s = actor_optimizer.update(g, opt_s, params=p)
            return eqx.apply_updates(p, updates), new_opt_s

        new_act_params, new_act_opt_state = jax.lax.cond(
            update_actor,
            apply_actor_step,
            lambda operand: (operand[0], operand[1]),
            (a_params, a_opt_state, act_grads),
        )

        return (
            new_act_params,
            new_crit_params,
            new_target_crit_params,
            new_act_opt_state,
            new_crit_opt_state,
            metrics,
        )

    jitted_eval_loss = eqx.filter_jit(loss_fn)

    key, subkey = jax.random.split(key)
    train_data, val_data = train_val_split(subkey, (states, actions), val_prop=val_pct)

    history: dict[str, list] = {
        "train_actor": [],
        "train_critic": [],
        "train_return": [],
        "train_step_loss": [],
        "val_actor": [],
        "val_critic": [],
        "val_return": [],
        "val_step_loss": [],
    }

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)
    chunk_len = traj_size + NUM_BURN_IN_STEPS + 1

    for epoch in loop:
        key, train_shuffle_key, val_shuffle_key = jax.random.split(key, 3)
        update_actor = jnp.asarray(epoch >= critic_warmup_epochs)

        train_metrics_list = []
        for batch in zip(
            *get_batches(train_data, batch_size, chunk_len, key=train_shuffle_key),
            strict=True,
        ):
            key, step_key = jax.random.split(key)
            (
                act_params,
                crit_params,
                target_crit_params,
                act_opt_state,
                crit_opt_state,
                step_metrics,
            ) = train_step(
                act_params,
                crit_params,
                target_crit_params,
                act_opt_state,
                crit_opt_state,
                world_model,
                step_key,
                traj_size,
                update_actor,
                *batch,
            )
            train_metrics_list.append(step_metrics)

        val_metrics_list = []
        for batch in zip(
            *get_batches(val_data, val_batch_size, chunk_len, key=val_shuffle_key),
            strict=True,
        ):
            key, step_key = jax.random.split(key)
            _, step_metrics = jitted_eval_loss(
                (act_params, crit_params),
                target_crit_params,
                world_model,
                step_key,
                traj_size,
                *batch,
            )
            val_metrics_list.append(step_metrics)

        for prefix, m_list in [
            ("train", train_metrics_list),
            ("val", val_metrics_list),
        ]:
            history[f"{prefix}_actor"].append(
                jnp.mean(jnp.array([m["actor_loss"] for m in m_list])).item()
            )
            history[f"{prefix}_critic"].append(
                jnp.mean(jnp.array([m["critic_loss"] for m in m_list])).item()
            )
            history[f"{prefix}_return"].append(
                jnp.mean(jnp.array([m["mean_return"] for m in m_list])).item()
            )
            history[f"{prefix}_step_loss"].append(
                jnp.mean(jnp.array([m["mean_step_loss"] for m in m_list])).item()
            )

        loop.set_postfix(
            {
                "tr_crit": f"{history['train_critic'][-1]:.3f}",
                "val_step": f"{history['val_step_loss'][-1]:.3f}",
                "val_ret": f"{history['val_return'][-1]:.3f}",
            }
        )

        if epoch >= critic_warmup_epochs:
            post_warmup_val = history["val_step_loss"][critic_warmup_epochs:]
            if history["val_step_loss"][-1] == min(post_warmup_val):
                best_act_params = act_params
                best_crit_params = crit_params
            elif count_fruitless(post_warmup_val) > max_patience:
                loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
                break

    final_act_params = best_act_params if return_best else act_params
    final_crit_params = best_crit_params if return_best else crit_params

    trained_actor = eqx.combine(final_act_params, act_static)
    trained_critic = eqx.combine(final_crit_params, crit_static)
    return trained_actor, trained_critic, history


def main() -> None:
    """Train the threat model from YAML configuration and CLI overrides.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If configuration values are invalid.
        FileNotFoundError: If a required input path cannot be resolved when used.
    """
    parser = script_tools.create_stage_parser(
        description="Train the threat model from a YAML config with optional overrides.",
        default_config=DEFAULT_CONFIG_PATH,
    )
    args = parser.parse_args()
    config = hypersonic.load_config(
        args.config,
        args.overrides,
    )

    train_config = config.train_threat_model

    if config.paths.dataset_path is None:
        raise ValueError("Configuration 'paths.dataset_path' must be specified.")
    if config.paths.plots_directory is None:
        raise ValueError("Configuration 'paths.plots_directory' must be specified.")
    if config.paths.model_path is None:
        raise ValueError("Configuration 'paths.model_path' must be specified.")
    if config.paths.network_path is None:
        raise ValueError("Configuration 'paths.network_path' must be specified.")

    dataset_path = config.paths.dataset_path
    plots_directory = config.paths.plots_directory
    model_path = config.paths.model_path

    resolved_log_path: pathlib.Path | None = None
    if args.log_file.lower() == "auto":
        resolved_log_path = plots_directory / "train_threat_model.log"
    elif args.log_file.lower() != "none":
        resolved_log_path = pathlib.Path(args.log_file)

    hyperlog.setup_logging(level=args.log_level, log_file=resolved_log_path)
    hyperlog.log_memory_usage(logger, level=logging.INFO, label="Training Start")

    retrain_model = train_config.training.retrain_model
    if not retrain_model and model_path.exists():
        logger.info(
            "Model already exists at '%s' and retrain_model is False. Skipping training.",
            model_path,
        )
        return

    os.makedirs(dataset_path.parent, exist_ok=True)
    os.makedirs(model_path.parent, exist_ok=True)
    os.makedirs(plots_directory, exist_ok=True)

    key = jax.random.key(seed=train_config.seed)
    key, subkey = jax.random.split(key)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{dataset_path}'. "
            "Please run scenario data generation first:\n"
            f"  .venv\\Scripts\\python scripts/generate_data.py --config {args.config}"
        )

    # Check config hash traceability
    curr_data_hash = hypersonic.compute_dataset_lineage_hash(config)
    dataset = data.loading.load_data(str(dataset_path))
    if "state_arr" not in dataset:
        raise ValueError(
            f"Dataset at '{dataset_path}' has an invalid or legacy format. "
            "Please regenerate the dataset:\n"
            f"  .venv\\Scripts\\python scripts/generate_data.py --config {args.config}"
        )
    ds_metadata = dataset.get("metadata", {})
    dataset_config_hash = ds_metadata.get("config_hash")

    if dataset_config_hash is None:
        warnings.warn(
            "Dataset metadata does not contain a config_hash. "
            "Traceability cannot be verified.",
            UserWarning,
        )
    elif dataset_config_hash[:8] != curr_data_hash[:8]:
        warnings.warn(
            f"Dataset config hash ({dataset_config_hash}) does not match "
            f"current simulation/data config hash ({curr_data_hash}). "
            "Simulation parameters may be inconsistent with generated data. "
            "Please rerun scripts/generate_data.py if you want exact consistency.",
            UserWarning,
        )
    state_arr = dataset["state_arr"]
    action_arr = dataset["control_arr"]

    states = jnp.stack(
        (
            state_arr.x,
            state_arr.y,
            state_arr.z,
            state_arr.vx,
            state_arr.vy,
            state_arr.vz,
        ),
        axis=-1,
    )
    actions = jnp.stack(
        (
            action_arr.angle_of_attack,
            action_arr.bank_angle,
        ),
        axis=-1,
    )
    breakpoint()
    prev_states = states[:, :-1, :]
    next_states = states[:, 1:, :]

    key, wm_key, act_key, critic_key = jax.random.split(key, 4)
    hidden_dim = 128
    world_model_baseline = FlowDynamics(
        key=wm_key,
        state_dim=states.shape[-1],
        action_dim=actions.shape[-1],
        state_low=states.min(axis=tuple(range(states.ndim - 1))),
        state_high=states.max(axis=tuple(range(states.ndim - 1))),
        hidden_dim=hidden_dim,
        state_mean=jnp.nanmean(states, axis=(0, 1)),
        state_std=jnp.nanstd(states, axis=(0, 1)),
    )
    action_flow_baseline = ActionFlow(
        key=act_key,
        state_dim=states.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=hidden_dim,
        state_mean=jnp.nanmean(states, axis=(0, 1)),
        state_std=jnp.nanstd(states, axis=(0, 1)),
    )
    critic = Critic(
        state_dim=states.shape[-1],
        rnn_hidden_dim=world_model_baseline.memory.hidden_size,
        mlp_hidden_dim=256,
        key=critic_key,
    )
    breakpoint()
    # critic = Critic(state_dim=states.shape[-1], hidden_dim=128, key=crit_key)
    train_wm_model = True
    train_act_model = True

    # Train the world model with one step updates
    if train_wm_model:
        key, subkey = jax.random.split(key)
        dynamics, d_losses = world_training_loop(
            subkey,
            prev_states,
            actions,
            next_states,
            world_model_baseline,
        )
        eqx.tree_serialise_leaves("world_model.eqx", dynamics)
    else:
        dynamics = eqx.tree_deserialise_leaves("world_model.eqx", world_model_baseline)
    breakpoint()
    if train_act_model:
        # Train the action policy with one step updates
        key, subkey = jax.random.split(key)
        policy, d_losses = policy_training_loop(
            subkey,
            prev_states,
            actions,
            dynamics,
            action_flow_baseline,
        )
        eqx.tree_serialise_leaves("action_model.eqx", policy)
    else:
        policy = eqx.tree_deserialise_leaves("action_model.eqx", action_flow_baseline)
    breakpoint()
    key, subkey = jax.random.split(key)
    trained_actor, trained_critic, history = imagination_training_loop(
        subkey,
        prev_states,
        actions,
        dynamics,
        policy,
        critic,
    )
    breakpoint()
    # Use both trained models for true imagination rollouts
    # (freeze world model and update action model)


if __name__ == "__main__":
    main()
