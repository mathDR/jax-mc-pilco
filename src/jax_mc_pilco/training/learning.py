"""Methods to train Flows and generate data."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
import tqdm

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.policy_learning.critic import DreamerCritic
from jax_mc_pilco.training.loss import actor_loss, single_trajectory_loss
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
            state_low=states.min(axis=0),
            state_high=states.max(axis=0),
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

        loss_value, grads = eqx.filter_value_and_grad(single_trajectory_loss)(
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
            loss_i = eqx.filter_jit(single_trajectory_loss)(params, static, *batch)
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


def actor_critic_training_loop(
    key: jtp.Key[jtp.Array, ""],
    states: jax.Array,
    actions: jax.Array,
    act_low: jax.Array,
    act_high: jax.Array,
    world_model: FlowDynamics,
    actor_model: FlowActor | None = None,
    act_optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 250,
    act_learning_rate: float = 3e-4,
    show_progress: bool = True,
) -> tuple[jtp.PyTree | FlowActor | None, dict[str, list]]:
    """Standard training loop for the actor models."""

    key, act_key = jax.random.split(key)
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    if actor_model is None:
        actor_model = FlowActor(
            act_key,
            state_dim + world_model.memory.hidden_size,
            action_dim,
            act_low,
            act_high,
        )

    # Setup optimizers
    if act_optimizer is None:
        act_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(act_learning_rate))

    actor, act_static = eqx.partition(
        actor_model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )

    act_opt_state = act_optimizer.init(actor)

    key, subkey = jax.random.split(key)
    losses: dict[str, list] = {"act": [], "crit": []}

    @eqx.filter_jit
    def actor_critic_train_step(
        key: jtp.Key[jtp.Array, ""],
        actor: jtp.PyTree,
        world_model: FlowDynamics,
        actor_opt_state: optax.OptState,
        actor_opt: optax.GradientTransformation,
        init_state: jax.Array,
        horizon: int = 15,
        ent_coef: float = 5e-4,
    ):
        loss_act, grads_act = eqx.filter_value_and_grad(actor_loss)(
            actor, act_static, world_model, init_state, horizon, ent_coef, key
        )

        actor_updates, new_actor_opt_state = actor_opt.update(grads_act, actor_opt_state, actor)
        new_actor = eqx.apply_updates(actor, actor_updates)

        return new_actor, new_actor_opt_state, loss_act

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        key, subkey = jax.random.split(key)
        init_state = jax.random.choice(subkey, states)
        actor, act_opt_state, loss_act = actor_critic_train_step(
            subkey,
            actor,
            world_model,
            act_opt_state,
            act_optimizer,
            init_state,
        )
        losses["act"].append(loss_act)

    actor_model = eqx.combine(actor, act_static)
    return actor_model, losses
