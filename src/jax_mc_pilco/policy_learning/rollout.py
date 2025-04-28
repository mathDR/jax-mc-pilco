""" Optimize a controller on a given cost function."""

import equinox as eqx
import optax as ox
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr
import gymnasium


def remake_state(x):
    return np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])


def fit_controller(  # noqa: PLR0913
    *,
    policy: eqx.Module,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    obj_func: Callable,
    optim: ox.GradientTransformation,
    key: ArrayLike = jr.PRNGKey(42),
    num_iters: Int = 100,
    unroll: Int = 5,
) -> Tuple[eqx.Module, Array]:
    """The optimization loop for fitting the policy parameters."""
    # Now do a rollout with this model
    # Generate an initial state
    x, _ = env.reset()
    sample = remake_state(x)
    key, subkey = jr.split(key)
    # Generate an initial (random) action
    u = env.action_space.sample()
    # initialize some particles
    initial_particles = gp_model.get_samples(
        key, jnp.array([sample]), jnp.array([u]), num_particles
    )

    @eqx.debug.assert_max_traces(max_traces=1)
    def rollout(
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
        loss_value, loss_gradient = eqx.filter_value_and_grad(rollout)(
            policy, initial_particles, gp_model, timesteps
        )
        updates, opt_state = optim.update(
            loss_gradient, opt_state, eqx.filter(policy, eqx.is_array)
        )
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, loss_value

    # Optimisation loop - hack an early stopping criteron
    best_loss: Float = float("inf")
    patience: int = 5  # Number of steps of no improvement before stopping
    patience_count: int = 0  # Number of steps since last improving update.
    min_delta: Float = (
        1e-3  # Minimum delta between updates to be considered an improvement
    )
    losses = []
    ## Need to apply the dropout schedule...
    for step in range(num_iters):
        policy, opt_state, train_loss = make_step(policy, opt_state)
        losses.append(train_loss)
        # patience_count = jax.lax.select(
        #     best_loss - train_loss > min_delta, 0, patience_count + 1
        # )
        # best_loss = jax.lax.select(train_loss < best_loss, train_loss, best_loss)
        # print(f"{step=}, train_loss={train_loss.item()}, best_loss={best_loss.item()}, ")
        # if patience_count > patience:
        #     print(f"Terminating due to early stopping at {step=}, train_loss={train_loss.item()}, ")
        #     break
        if (step % 100) == 0 or (step == num_iters - 1):
            print(f"{step=}, train_loss={train_loss.item()}, ")

    return policy, jnp.array(losses)


def rollout_check(
    policy: eqx.Module,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    model: eqx.Module,
    obj_func: Callable,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(123),
) -> Tuple[Float, Array]:
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    # Now do a rollout with this model
    # Generate an initial state
    x, _ = env.reset()
    key, subkey = jr.split(key)
    # Generate an initial (random) action
    u = env.action_space.sample()
    # initialize some particles
    init_samples = model.get_samples(key, jnp.array([x]), jnp.array([u]), num_particles)

    def one_rollout_step(
        carry: Tuple[ArrayLike, ArrayLike, List[ArrayLike], ArrayLike, Float],
        timestep: Float,
    ) -> Tuple[Tuple[ArrayLike, ArrayLike, List[ArrayLike], ArrayLike, Float], Float]:
        policy_params, key, all_samples, samples, total_cost = carry
        policy = eqx.combine(policy_params, policy_static)
        actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))

        key, subkey = jr.split(key)
        samples = model.get_samples(key, samples, actions, 1)
        all_samples.append(samples)
        cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
        return (
            policy_params,
            key,
            all_samples,
            samples,
            total_cost + cost,
        ), cost

    total_cost = 0
    all_samples = list(init_samples)
    (policy_params, key, all_samples, samples, total_cost), result = jax.lax.scan(
        one_rollout_step,
        (policy_params, key, all_samples, init_samples, total_cost),
        timesteps,
    )
    return total_cost, jnp.array(all_samples)
