""" Optimize a controller on a given cost function."""

import copy
import equinox as eqx
import optax as ox
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr
import gymnasium


def fit_controller(  # noqa: PLR0913
    *,
    policy: eqx.Module,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    initial_state: ArrayLike,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    obj_func: Callable,
    optim: ox.GradientTransformation,
    key: ArrayLike = jr.PRNGKey(42),
    num_iters,
    max_steps: Int = 100,
    patience: Int = 7,
    unroll: Int = 5,
) -> Tuple[eqx.Module, Array]:
    """The optimization loop for fitting the policy parameters."""

    sample_train, _ = env.reset(
        options={"x_init": initial_state[0], "y_init": initial_state[1]}
    )
    key, subkey = jr.split(key)
    # Generate an initial action
    action_train = policy(sample_train, 0.0)
    # initialize some particles
    initial_train_particles = gp_model.get_samples(
        key, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )

    sample_val, _ = env.reset(
        options={"x_init": initial_state[0], "y_init": initial_state[1]}
    )
    key, subkey = jr.split(key)
    # Generate an initial action
    action_val = policy(sample_val, 0.0)
    # initialize some particles
    initial_val_particles = gp_model.get_samples(
        key, jnp.array([sample_val]), jnp.array([action_val]), num_particles
    )

    @eqx.debug.assert_max_traces(max_traces=1)
    def train_rollout(
        policy: eqx.Module,
        init_samples: ArrayLike,
        model: eqx.Module,
        timesteps: ArrayLike,
        key: ArrayLike = jr.key(123),
    ) -> Float:
        policy_params, policy_static = eqx.partition(policy, eqx.is_array)

        def one_rollout_step(
            carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float], timestep: Float
        ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float], Float]:
            policy_params, key, samples, total_cost = carry
            policy = eqx.combine(policy_params, policy_static)
            actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))

            key, subkey = jr.split(key)
            samples = model.get_samples(key, samples, actions, 1)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
            return (policy_params, key, samples, total_cost + cost), cost

        total_cost = 0
        (policy_params, key, samples, total_cost), result = jax.lax.scan(
            one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
        )
        return total_cost

    def val_rollout(
        policy: eqx.Module,
        init_samples: ArrayLike,
        model: eqx.Module,
        timesteps: ArrayLike,
        key: ArrayLike = jr.key(123),
    ) -> Float:
        policy_params, policy_static = eqx.partition(policy, eqx.is_array)

        def one_rollout_step(
            carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float], timestep: Float
        ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float], Float]:
            policy_params, key, samples, total_cost = carry
            policy = eqx.combine(policy_params, policy_static)
            actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))

            key, subkey = jr.split(key)
            samples = model.get_samples(key, samples, actions, 1)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
            return (policy_params, key, samples, total_cost + cost), cost

        total_cost = 0
        (policy_params, key, samples, total_cost), result = jax.lax.scan(
            one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
        )
        return total_cost

    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    @eqx.filter_jit
    def make_step(
        policy: eqx.Module,
        opt_state: PyTree,
    ) -> Tuple[eqx.Module, PyTree, Float]:
        loss_value, loss_gradient = eqx.filter_value_and_grad(train_rollout)(
            policy, initial_train_particles, gp_model, timesteps
        )
        updates, opt_state = optim.update(
            loss_gradient, opt_state, eqx.filter(policy, eqx.is_array)
        )
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, loss_value

    val_losses = []
    step = 0
    best_loss = np.inf
    iterations_since_improvement = 0

    while step < max_steps:
        policy, opt_state, train_loss = make_step(policy, opt_state)
        val_loss = val_rollout(policy, initial_val_particles, gp_model, timesteps)
        val_losses.append(val_loss)

        if len(val_losses) == 1 or val_loss < best_loss:
            print(f"\t   (New best performance {val_loss.item()})")
            best_loss = val_loss
            best_policy = jax.tree.map(
                lambda x: x.copy() if eqx.is_array(x) else copy.deepcopy(x), policy
            )
            iterations_since_improvement = 0

        elif iterations_since_improvement <= patience:
            print(
                f"Early stopping due to no improvement over the last {patience} steps"
            )
            break
        if (step % 50) == 0 or (step == max_steps - 1):
            print(f"{step=}, validation_loss={val_loss.item()}, ")
        step += 1
        iterations_since_improvement += 1

    return best_policy, jnp.array(val_losses)
