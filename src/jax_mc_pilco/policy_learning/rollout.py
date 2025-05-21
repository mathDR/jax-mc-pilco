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
    policies: ArrayLike,
    initial_samples: ArrayLike,
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

    @eqx.filter_vmap
    def evaluate_per_ensemble(model, x, t):
        return eqx.filter_vmap(model)(x, t)

    # Now do a rollout with this model
    # Generate initial actions
    breakpoint()
    initial_actions = evaluate_per_ensemble(
        policies, initial_samples[:, jnp.newaxis, :], 0.0
    )
    # initialize some particles
    key, subkey = jr.split(key)
    initial_particles = gp_model.get_samples(
        subkey, initial_samples[:, jnp.newaxis, :], initial_actions, num_particles
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
            actions = evaluate_per_ensemble(
                policy, samples, jnp.tile(timestep, num_particles)
            )

            key, subkey = jr.split(key)
            samples = model.get_samples(key, samples, actions, 1)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
            return (policy_params, key, samples, total_cost + cost), cost

        total_cost = 0
        (policy_params, key, samples, total_cost), result = jax.lax.scan(
            one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
        )
        return total_cost

    opt_states = eqx.filter_vmap(optim.init)(eqx.filter(policies, eqx.is_array))

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    @eqx.filter_jit
    @eqx.filter_vmap
    def make_step(
        policy: eqx.Module,
        opt_states: PyTree,
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
    policies, opt_states, train_losses = make_step(policies, opt_states)
    losses.append(train_losses)
    step = 0

    while step < num_iters:
        policies, opt_states, train_losses = make_step(policies, opt_states)
        losses.append(train_losses)
        if (step % 50) == 0 or (step == num_iters - 1):
            print(f"{step=}, train_losses={train_losses.item()}, ")
        step = step + 1

    return policies, jnp.array(losses)
