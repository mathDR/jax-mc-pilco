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
    starting_dropout_probability: Float,
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

    delta_dropout = 0.125
    min_learning_rate = 0.0001
    alpha_s = 0.99
    sigma_s = 0.08
    num_iterations_monitoring = 200
    lambda_s = 0.5

    losses = []
    variance_cost_delta = 0.0
    expected_cost_delta = 0.0
    observed_signal = [0.0]  # (s_0 from the paper)
    policy, opt_state, train_loss = make_step(policy, opt_state)
    losses.append(train_loss)
    step = 0
    print(f"{step=}, train_loss={train_loss.item()}, ")
    for step in range(1, num_iters):
        policy, opt_state, train_loss = make_step(policy, opt_state)
        cost_delta = train_loss - losses[-1]
        # We do the variance first because then we update the expected_cost_delta
        variance_cost_delta = alpha_s * (
            variance_cost_delta
            + (1 - alpha_s) * jnp.square(cost_delta - expected_cost_delta)
        )
        expected_cost_delta = alpha_s * expected_cost_delta + (1 - alpha_s) * cost_delta
        s_j = alpha_s * observed_signal[-1] + (
            1 - alpha_s
        ) * expected_cost_delta / jnp.sqrt(variance_cost_delta)
        observed_signal.append(jnp.abs(s_j))

        if len(observed_signal) >= num_iterations_monitoring and (
            policy.f_drop.p > 0
            or opt_state.hyperparams["learning_rate"] > min_learning_rate
        ):
            if jnp.all(
                jnp.array(observed_signal[-num_iterations_monitoring:]) < sigma_s
            ):
                print(f"Decreasing dropout probability at {step=},")
                # We will decrease the dropout probability
                dropout_probability = max(policy.f_drop.p - delta_dropout, 0.0)
                adam_learning_rate = max(
                    lambda_s * opt_state.hyperparams["learning_rate"], min_learning_rate
                )
                sigma_s = lambda_s * sigma_s
                # # Now set the dropout probability and the adam learning rate
                where = lambda d: d.f_drop
                policy = eqx.tree_at(
                    where, policy, eqx.nn.Dropout(p=dropout_probability)
                )
                opt_state.hyperparams["learning_rate"] = adam_learning_rate

        losses.append(train_loss)
        if (step % 50) == 0 or (step == num_iters - 1):
            print(f"{step=}, train_loss={train_loss.item()}, ")

    return policy, jnp.array(losses)
