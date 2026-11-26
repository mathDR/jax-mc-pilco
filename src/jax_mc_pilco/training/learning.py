"""Methods to train Flows and generate data."""

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
import tqdm
from beartype import beartype

from jax_mc_pilco.model_learning.flow_model import LatentFlowWorldModel
from jax_mc_pilco.policy_learning.action_flows import ActionFlow, Critic
from jax_mc_pilco.training.loss import (
    batched_act_trajectory_loss,
    batched_actor_critic_imagination_loss,
    batched_wm_trajectory_loss,
)
from jax_mc_pilco.training.training_utils import (
    count_fruitless,
)

NUM_BURN_IN_STEPS = 10


def _build_world_step_fns(
    train_buffer: Any,
    train_buffer_state: Any,
    val_buffer: Any,
    val_buffer_state: Any,
    optimizer: optax.GradientTransformation,
    kl_scale: float,
    free_nats: float,
):
    """Build jitted train and eval step functions for LatentFlowWorldModel."""

    def loss_fn(
        _params: jtp.PyTree,
        _static: LatentFlowWorldModel,
        _key: jtp.Key[jtp.Array, ""],
        _obs: jtp.UInt8[jtp.Array, "num_episodes seq_len C H W"],
        _actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
        _rewards: jtp.Float[jtp.Array, "num_episodes seq_len"],
        _dones: jtp.Bool[jtp.Array, "num_episodes seq_len"],
    ) -> jtp.Float[jtp.Array, ""]:
        return batched_wm_trajectory_loss(
            _params,
            _static,
            _key,
            _obs,
            _actions,
            _rewards,
            _dones,
            kl_scale=kl_scale,
            free_nats=free_nats,
        )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def world_train_step(
        _params: jtp.PyTree,
        _static: LatentFlowWorldModel,
        _key: jtp.Key[jtp.Array, ""],
        _opt_state: jtp.PyTree,
    ) -> tuple[jtp.PyTree, optax.OptState, jtp.Float[jtp.Array, ""]]:
        sample_key, loss_key = jax.random.split(_key)
        exp = train_buffer.sample(train_buffer_state, sample_key).experience

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            _params,
            _static,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["reward"],
            exp["done"],
        )
        updates, new_opt_state = optimizer.update(grads, _opt_state, params=_params)
        return eqx.apply_updates(_params, updates), new_opt_state, loss_value

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def world_eval_step(
        _params: jtp.PyTree,
        _static: LatentFlowWorldModel,
        _key: jtp.Key[jtp.Array, ""],
    ) -> jtp.Float[jtp.Array, ""]:
        sample_key, loss_key = jax.random.split(_key)
        exp = val_buffer.sample(val_buffer_state, sample_key).experience

        return loss_fn(
            _params,
            _static,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["reward"],
            exp["done"],
        )

    return world_train_step, world_eval_step


@jtp.jaxtyped(typechecker=beartype)
def world_training_loop(
    key: jtp.Key[jtp.Array, ""],
    train_data: tuple[Any, Any],
    val_data: tuple[Any, Any],
    world_model: LatentFlowWorldModel,
    optimizer: optax.GradientTransformation | None = None,
    steps_per_epoch: int | None = None,
    val_steps_per_epoch: int | None = None,
    max_epochs: int = 1000,
    learning_rate: optax.ScalarOrSchedule = 3e-4,
    kl_scale: float = 0.1,
    free_nats: float = 1.0,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[LatentFlowWorldModel, dict[str, list]]:
    """Training loop for LatentFlowWorldModel using Flashbax trajectory buffers."""
    train_buffer, train_buffer_state = train_data
    val_buffer, val_buffer_state = val_data

    steps_per_epoch = steps_per_epoch or _compute_epoch_steps(train_buffer, train_buffer_state)
    val_steps_per_epoch = val_steps_per_epoch or _compute_epoch_steps(val_buffer, val_buffer_state)
    optimizer = optimizer or optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

    params, static = eqx.partition(
        world_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    world_train_step, world_eval_step = _build_world_step_fns(
        train_buffer=train_buffer,
        train_buffer_state=train_buffer_state,
        val_buffer=val_buffer,
        val_buffer_state=val_buffer_state,
        optimizer=optimizer,
        kl_scale=kl_scale,
        free_nats=free_nats,
    )

    losses: dict[str, list] = {"train": [], "val": []}
    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        train_batch_losses = []
        for _ in range(steps_per_epoch):
            key, subkey = jax.random.split(key)
            params, opt_state, loss_i = world_train_step(params, static, subkey, opt_state)
            train_batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(train_batch_losses)).item())

        val_batch_losses = []
        for _ in range(val_steps_per_epoch):
            key, subkey = jax.random.split(key)
            val_batch_losses.append(world_eval_step(params, static, subkey))
        losses["val"].append(jnp.mean(jnp.array(val_batch_losses)).item())

        loop.set_postfix({k: f"{v[-1]:.4f}" for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params
        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    final_params = best_params if return_best else params
    return eqx.combine(final_params, static), losses


def _build_policy_step_fns(
    train_buffer: Any,
    train_buffer_state: Any,
    val_buffer: Any,
    val_buffer_state: Any,
    act_static: ActionFlow,
    optimizer: optax.GradientTransformation,
):
    """Build jitted train and eval step functions for ActionFlow behavior cloning."""

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def act_train_step(
        _params: jtp.PyTree,
        _opt_state: jtp.PyTree,
        _wm: LatentFlowWorldModel,
        _key: jtp.Key[jtp.Array, ""],
    ) -> tuple[jtp.PyTree, optax.OptState, jtp.Float[jtp.Array, ""]]:
        sample_key, loss_key = jax.random.split(_key)
        exp = train_buffer.sample(train_buffer_state, sample_key).experience

        loss_value, grads = eqx.filter_value_and_grad(batched_act_trajectory_loss)(
            _params,
            act_static,
            _wm,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["done"],
        )
        updates, new_opt_state = optimizer.update(grads, _opt_state, params=_params)
        return eqx.apply_updates(_params, updates), new_opt_state, loss_value

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def act_eval_step(
        _params: jtp.PyTree,
        _wm: LatentFlowWorldModel,
        _key: jtp.Key[jtp.Array, ""],
    ) -> jtp.Float[jtp.Array, ""]:
        sample_key, loss_key = jax.random.split(_key)
        exp = val_buffer.sample(val_buffer_state, sample_key).experience

        return batched_act_trajectory_loss(
            _params,
            act_static,
            _wm,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["done"],
        )

    return act_train_step, act_eval_step


@jtp.jaxtyped(typechecker=beartype)
def policy_training_loop(
    key: jtp.Key[jtp.Array, ""],
    train_data: tuple[Any, Any],  # (train_buffer, train_buffer_state)
    val_data: tuple[Any, Any],  # (val_buffer, val_buffer_state)
    world_model: LatentFlowWorldModel,
    action_flow: ActionFlow,
    optimizer: optax.GradientTransformation | None = None,
    steps_per_epoch: int | None = None,
    val_steps_per_epoch: int | None = None,
    max_epochs: int = 1000,
    learning_rate: float = 3e-4,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[ActionFlow, dict[str, list]]:
    """Behavior-cloning training loop for ActionFlow using Flashbax trajectory buffers."""
    train_buffer, train_buffer_state = train_data
    val_buffer, val_buffer_state = val_data

    steps_per_epoch = steps_per_epoch or _compute_epoch_steps(train_buffer, train_buffer_state)
    val_steps_per_epoch = val_steps_per_epoch or _compute_epoch_steps(val_buffer, val_buffer_state)
    optimizer = optimizer or optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

    act_params, act_static = eqx.partition(
        action_flow,
        filter_spec=lambda x: eqx.is_inexact_array(x),
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = act_params
    opt_state = optimizer.init(act_params)

    act_train_step, act_eval_step = _build_policy_step_fns(
        train_buffer=train_buffer,
        train_buffer_state=train_buffer_state,
        val_buffer=val_buffer,
        val_buffer_state=val_buffer_state,
        act_static=act_static,
        optimizer=optimizer,
    )

    losses: dict[str, list] = {"train": [], "val": []}
    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        train_batch_losses = []
        for _ in range(steps_per_epoch):
            key, subkey = jax.random.split(key)
            act_params, opt_state, loss_i = act_train_step(act_params, opt_state, world_model, subkey)
            train_batch_losses.append(loss_i)
        losses["train"].append(jnp.mean(jnp.array(train_batch_losses)).item())

        val_batch_losses = []
        for _ in range(val_steps_per_epoch):
            key, subkey = jax.random.split(key)
            val_batch_losses.append(act_eval_step(act_params, world_model, subkey))
        losses["val"].append(jnp.mean(jnp.array(val_batch_losses)).item())

        loop.set_postfix({k: f"{v[-1]:.4f}" for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = act_params
        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    final_params = best_params if return_best else act_params
    return eqx.combine(final_params, act_static), losses


@jtp.jaxtyped(typechecker=beartype)
def latent_imagination_trajectory(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    burn_in_obs: jtp.Float[jtp.Array, "burn_in_len C H W"],
    burn_in_actions: jtp.Float[jtp.Array, "burn_in_len action_dim"],
    burn_in_dones: jtp.Bool[jtp.Array, " burn_in_len"],
) -> tuple[
    jtp.Float[jtp.Array, "horizon_plus_1 latent_dim"],
    jtp.Float[jtp.Array, "horizon action_dim"],
    jtp.Float[jtp.Array, "horizon_plus_1 hidden_dim"],
    jtp.Float[jtp.Array, " horizon"],
]:
    """Burn in on observed pixel frames to infer (z_0, h_0), then imagine `horizon` steps."""
    action_flow = eqx.combine(action_params, action_static)

    # 1. Encode burn-in frames and infer posterior latent trajectory (frozen world model)
    embeds = jax.lax.stop_gradient(eqx.filter_vmap(world_model.encoder)(burn_in_obs[: NUM_BURN_IN_STEPS + 1]))
    init_hidden = jnp.zeros((world_model.memory.hidden_size,), dtype=embeds.dtype)
    init_z = jnp.zeros((world_model.latent_dim,), dtype=embeds.dtype)

    def burn_in_fn(carry, inputs):
        prev_z, prev_hidden, prev_done, step_key = carry
        embed_t, action_t, done_t = inputs

        prev_hidden = jnp.where(prev_done, 0.0, prev_hidden)
        prev_z = jnp.where(prev_done, 0.0, prev_z)

        next_key, post_key = jax.random.split(step_key)
        z_t, _ = jax.lax.stop_gradient(world_model.posterior_sample_and_log_prob(post_key, embed_t, prev_hidden))
        next_hidden = jax.lax.stop_gradient(world_model.step_hidden(z_t, action_t, prev_hidden))
        return (z_t, next_hidden, done_t, next_key), None

    key, burn_key = jax.random.split(key)
    (last_z, burn_in_hidden, last_done, _), _ = jax.lax.scan(
        burn_in_fn,
        (init_z, init_hidden, jnp.asarray(True), burn_key),
        (
            embeds[:NUM_BURN_IN_STEPS],
            burn_in_actions[:NUM_BURN_IN_STEPS],
            burn_in_dones[:NUM_BURN_IN_STEPS],
        ),
    )

    # Infer z_0 at the end of burn-in (t = NUM_BURN_IN_STEPS)
    burn_in_hidden = jnp.where(last_done, 0.0, burn_in_hidden)
    last_z = jnp.where(last_done, 0.0, last_z)

    key, z0_key, first_action_key = jax.random.split(key, 3)
    start_z, _ = jax.lax.stop_gradient(
        world_model.posterior_sample_and_log_prob(z0_key, embeds[NUM_BURN_IN_STEPS], burn_in_hidden)
    )

    # 2. Sample first imagined action a_0 ~ pi(a_0 | z_0, h_0)
    first_action, first_log_prob = action_flow.sample_and_log_prob(first_action_key, start_z, burn_in_hidden)

    # 3. Roll out `horizon` steps in latent space using prior flow
    def scan_fn(carry, _):
        prev_z, prev_action, prev_hidden, step_key = carry
        next_key, state_key, action_key = jax.random.split(step_key, 3)

        next_z, next_hidden = world_model.predict_next_state_and_hidden(state_key, prev_z, prev_action, prev_hidden)
        next_action, next_log_prob = action_flow.sample_and_log_prob(action_key, next_z, next_hidden)
        return (next_z, next_action, next_hidden, next_key), (
            next_z,
            next_action,
            next_hidden,
            next_log_prob,
        )

    _, (future_zs, future_actions, future_hiddens, future_log_probs) = jax.lax.scan(
        scan_fn,
        (start_z, first_action, burn_in_hidden, key),
        xs=None,
        length=horizon,
    )

    zs = jnp.concatenate([start_z[None], future_zs], axis=0)
    hiddens = jnp.concatenate([burn_in_hidden[None], future_hiddens], axis=0)
    actions = jnp.concatenate([first_action[None], future_actions[:-1]], axis=0)
    log_probs = jnp.concatenate([first_log_prob[None], future_log_probs[:-1]], axis=0)

    return zs, actions, hiddens, log_probs


def _compute_epoch_steps(buffer: Any, buffer_state: Any) -> int:
    """Compute default gradient steps per epoch from Flashbax buffer shapes."""
    B, T = buffer_state.experience["obs"].shape[:2]

    # Probe sample_batch_size and sample_sequence_length without executing computation
    sample_shape = jax.eval_shape(buffer.sample, buffer_state, jax.random.key(0))
    batch_size, seq_len = sample_shape.experience["obs"].shape[:2]

    chunks_per_traj = max(1, T // seq_len)
    return max(1, (B * chunks_per_traj) // batch_size)


def _aggregate_imagination_metrics(
    metrics_list: list[dict[str, jax.Array]],
    prefix: str,
    history: dict[str, list],
) -> None:
    """Append epoch-averaged step metrics into the history dictionary."""
    metric_keys = [
        ("actor_loss", "actor"),
        ("critic_loss", "critic"),
        ("mean_return", "return"),
        ("mean_reward", "reward"),
    ]
    for src_key, dst_suffix in metric_keys:
        mean_val = jnp.mean(jnp.array([m[src_key] for m in metrics_list])).item()
        history[f"{prefix}_{dst_suffix}"].append(mean_val)


def _build_imagination_step_fns(
    train_buffer: Any,
    train_buffer_state: Any,
    val_buffer: Any,
    val_buffer_state: Any,
    act_static: ActionFlow,
    crit_static: Critic,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    horizon: int,
    target_tau: float,
    gamma: float,
    lambda_: float,
    entropy_coef: float,
    huber_delta: float,
):
    """Build jitted train and eval step functions for latent imagination."""

    def loss_fn(
        params: tuple[jtp.PyTree, jtp.PyTree],
        target_c_params: jtp.PyTree,
        wm: LatentFlowWorldModel,
        step_key: jtp.Key[jtp.Array, ""],
        _obs: jtp.UInt8[jtp.Array, "num_episodes seq_len C H W"],
        _actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
        _dones: jtp.Bool[jtp.Array, "num_episodes seq_len"],
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
            obs=_obs,
            actions=_actions,
            dones=_dones,
            gamma=gamma,
            lambda_=lambda_,
            entropy_coef=entropy_coef,
            huber_delta=huber_delta,
        )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def train_step(
        a_params: jtp.PyTree,
        c_params: jtp.PyTree,
        target_c_params: jtp.PyTree,
        a_opt_state: jtp.PyTree,
        c_opt_state: jtp.PyTree,
        wm: LatentFlowWorldModel,
        step_key: jtp.Key[jtp.Array, ""],
        update_actor: jax.Array,
    ):
        sample_key, loss_key = jax.random.split(step_key)
        exp = train_buffer.sample(train_buffer_state, sample_key).experience

        (_, metrics), (act_grads, crit_grads) = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            (a_params, c_params),
            target_c_params,
            wm,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["done"],
        )

        crit_updates, new_crit_opt_state = critic_optimizer.update(crit_grads, c_opt_state, params=c_params)
        new_crit_params = eqx.apply_updates(c_params, crit_updates)
        new_target_crit_params = optax.incremental_update(new_crit_params, target_c_params, step_size=target_tau)

        def apply_actor_step(
            operand: tuple[jtp.PyTree, jtp.PyTree, jtp.PyTree],
        ) -> tuple[jtp.PyTree, jtp.PyTree]:
            p, opt_s, g = operand
            updates, new_opt_s = actor_optimizer.update(g, opt_s, params=p)
            return eqx.apply_updates(p, updates), new_opt_s

        new_act_params, new_act_opt_state = cast(
            tuple[jtp.PyTree, jtp.PyTree],
            jax.lax.cond(
                update_actor,
                apply_actor_step,
                lambda operand: (operand, operand),
                (a_params, a_opt_state, act_grads),
            ),
        )

        return (
            new_act_params,
            new_crit_params,
            new_target_crit_params,
            new_act_opt_state,
            new_crit_opt_state,
            metrics,
        )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def eval_step(
        a_params: jtp.PyTree,
        c_params: jtp.PyTree,
        target_c_params: jtp.PyTree,
        wm: LatentFlowWorldModel,
        step_key: jtp.Key[jtp.Array, ""],
    ) -> dict[str, jax.Array]:
        sample_key, loss_key = jax.random.split(step_key)
        exp = val_buffer.sample(val_buffer_state, sample_key).experience

        _, metrics = loss_fn(
            (a_params, c_params),
            target_c_params,
            wm,
            loss_key,
            exp["obs"],
            exp["action"],
            exp["done"],
        )
        return metrics

    return train_step, eval_step


@jtp.jaxtyped(typechecker=beartype)
def imagination_training_loop(
    key: jtp.Key[jtp.Array, ""],
    train_data: tuple[Any, Any],
    val_data: tuple[Any, Any],
    world_model: LatentFlowWorldModel,
    action_flow: ActionFlow,
    critic: Critic,
    actor_optimizer: optax.GradientTransformation | None = None,
    critic_optimizer: optax.GradientTransformation | None = None,
    steps_per_epoch: int | None = None,
    val_steps_per_epoch: int | None = None,
    horizon: int = 15,
    max_epochs: int = 1000,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    critic_warmup_epochs: int = 5,
    target_tau: float = 0.02,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coef: float = 1e-3,
    huber_delta: float = 1.0,
    max_patience: int = 25,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[ActionFlow, Critic, dict[str, list]]:
    """Actor-critic latent imagination training loop using Flashbax trajectory buffers."""
    train_buffer, train_buffer_state = train_data
    val_buffer, val_buffer_state = val_data

    steps_per_epoch = steps_per_epoch or _compute_epoch_steps(train_buffer, train_buffer_state)
    val_steps_per_epoch = val_steps_per_epoch or _compute_epoch_steps(val_buffer, val_buffer_state)

    actor_optimizer = actor_optimizer or optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_lr))
    critic_optimizer = critic_optimizer or optax.chain(optax.clip_by_global_norm(1.0), optax.adam(critic_lr))

    partition_spec = {
        "filter_spec": lambda x: eqx.is_inexact_array(x),
        "is_leaf": lambda leaf: isinstance(leaf, paramax.NonTrainable),
    }
    act_params, act_static = eqx.partition(action_flow, **partition_spec)
    crit_params, crit_static = eqx.partition(critic, **partition_spec)
    target_crit_params = crit_params

    best_act_params, best_crit_params = act_params, crit_params
    act_opt_state = actor_optimizer.init(act_params)
    crit_opt_state = critic_optimizer.init(crit_params)

    train_step, eval_step = _build_imagination_step_fns(
        train_buffer=train_buffer,
        train_buffer_state=train_buffer_state,
        val_buffer=val_buffer,
        val_buffer_state=val_buffer_state,
        act_static=act_static,
        crit_static=crit_static,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        horizon=horizon,
        target_tau=target_tau,
        gamma=gamma,
        lambda_=lambda_,
        entropy_coef=entropy_coef,
        huber_delta=huber_delta,
    )

    history: dict[str, list] = {
        "train_actor": [],
        "train_critic": [],
        "train_return": [],
        "train_reward": [],
        "val_actor": [],
        "val_critic": [],
        "val_return": [],
        "val_reward": [],
    }

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for epoch in loop:
        update_actor = jnp.asarray(epoch >= critic_warmup_epochs)

        train_metrics_list = []
        for _ in range(steps_per_epoch):
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
                update_actor,
            )
            train_metrics_list.append(step_metrics)

        val_metrics_list = []
        for _ in range(val_steps_per_epoch):
            key, step_key = jax.random.split(key)
            val_metrics_list.append(eval_step(act_params, crit_params, target_crit_params, world_model, step_key))

        _aggregate_imagination_metrics(train_metrics_list, "train", history)
        _aggregate_imagination_metrics(val_metrics_list, "val", history)

        loop.set_postfix({
            "tr_crit": f"{history['train_critic'][-1]:.3f}",
            "val_rew": f"{history['val_reward'][-1]:.3f}",
            "val_ret": f"{history['val_return'][-1]:.3f}",
        })

        if epoch >= critic_warmup_epochs:
            post_warmup_val = history["val_return"][critic_warmup_epochs:]
            if history["val_return"][-1] == max(post_warmup_val):
                best_act_params, best_crit_params = act_params, crit_params
            elif count_fruitless([-r for r in post_warmup_val]) > max_patience:
                loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
                break

    final_act_params = best_act_params if return_best else act_params
    final_crit_params = best_crit_params if return_best else crit_params
    return eqx.combine(final_act_params, act_static), eqx.combine(final_crit_params, crit_static), history
