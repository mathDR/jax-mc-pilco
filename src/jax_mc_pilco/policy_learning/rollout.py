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


def fit_controller(  # noqa: PLR0913
    *,
    policy: eqx.Module,
    starting_dropout_probability: Float,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    initial_state: ArrayLike,
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
    # Generate an initial state uniformly with damping around the final state
    sample, _ = env.reset(
        options={"x_init": initial_state[0], "y_init": initial_state[1]}
    )

    key, subkey = jr.split(key)
    # Generate an initial action
    u = policy(sample, 0.0)
    # initialize some particles
    initial_particles = gp_model.get_samples(
        key, jnp.array([sample]), jnp.array([u]), num_particles
    )

    # Reset the dropout probability for the policy
    where = lambda d: d.f_drop
    policy = eqx.tree_at(where, policy, eqx.nn.Dropout(p=starting_dropout_probability))

    # because we are changing the dropout, this might need to be recompiled
    # @eqx.debug.assert_max_traces(max_traces=1)
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

    losses = []
    policy, opt_state, train_loss = make_step(policy, opt_state)
    losses.append(train_loss)
    step = 0

    while step < num_iters:
        policy, opt_state, train_loss = make_step(policy, opt_state)
        losses.append(train_loss)
        if (step % 50) == 0 or (step == num_iters - 1):
            print(f"{step=}, train_loss={train_loss.item()}, ")
        step = step + 1

    return policy, jnp.array(losses)
