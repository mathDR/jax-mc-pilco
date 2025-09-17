""" Optimize a controller on a given cost function."""

import copy
import equinox as eqx
import equinox.internal as eqxi
import optax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr
import gymnasium

from jax_mc_pilco.controllers import Controller


@eqx.filter_jit
def fit_controller(  # noqa: PLR0913
    *,
    policy: Controller,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    initial_state: ArrayLike,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    obj_func: Callable,
    optim: optax.GradientTransformation,
    key: ArrayLike = jr.key(42),
    max_steps: Int = 100,
    patience: Int = 7,
    gtol: float = 1e-5,
) -> Tuple[eqx.Module, eqx.Module, Float]:
    """The optimization loop for fitting the policy parameters."""

    sample_train, _ = env.reset(
        options={"x_init": initial_state[0], "y_init": initial_state[1]}
    )
    # Generate an initial action
    action_train = policy(sample_train, 0.0)
    # initialize some particles
    key, subkey = jr.split(key)
    initial_train_particles = gp_model.get_samples(
        subkey, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )
    key, subkey = jr.split(key)
    initial_train_particles = gp_model.get_samples(
        subkey, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )

    @eqx.debug.assert_max_traces(max_traces=2)
    def train_rollout(
        policy: Controller,
        init_samples: ArrayLike,
        model: eqx.Module,
        timesteps: ArrayLike,
        key: ArrayLike = jr.key(42),
    ) -> Float:
        p_params, p_static = eqx.partition(policy, eqx.is_array)

        def one_rollout_step(
            carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float], timestep: Float
        ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float], Float]:
            params, key, samples, total_cost = carry
            policy = eqx.combine(params, p_static)
            actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))

            key, subkey = jr.split(key)
            samples = model.get_samples(subkey, samples, actions, 1)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
            return (params, key, samples, total_cost + cost), cost

        total_cost = 0

        (params, key, samples, total_cost), result = jax.lax.scan(
            one_rollout_step, (p_params, key, init_samples, total_cost), timesteps
        )
        return total_cost

    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    # Optimisation step.

    def true_fun(arg):
        (
            params,
            best_params,
            loss_value,
            iterations_since_improvement,
            best_loss,
        ) = arg
        return (
            loss_value,
            params,
            iterations_since_improvement - iterations_since_improvement,
        )

    def false_fun(arg):
        (
            params,
            best_params,
            loss_value,
            iterations_since_improvement,
            best_loss,
        ) = arg
        return (best_loss, best_params, iterations_since_improvement + 1)

    def make_step(
        carry: Tuple[
            ArrayLike,
            ArrayLike,
            Int,
            PyTree,
            Float,
            ArrayLike,
        ]
    ):
        (
            policy_params,
            best_params,
            iterations_since_improvement,
            opt_state,
            best_loss,
            loss_gradient,
        ) = carry
        policy = eqx.combine(policy_params, policy_static)
        loss_value, loss_gradient = eqx.filter_value_and_grad(train_rollout)(
            policy, initial_train_particles, gp_model, timesteps
        )
        updates, opt_state = optim.update(loss_gradient, opt_state, policy_params)
        policy = eqx.apply_updates(policy, updates)
        policy_params = eqx.filter(policy, eqx.is_array)

        # best_loss is originally infinite, so we will overwrite it with loss_value if it is.
        # Otherwise, we return best_loss
        best_loss = jax.lax.cond(
            jnp.isfinite(best_loss),
            lambda z: z[0],
            lambda z: z[1],
            (best_loss, loss_value),
        )

        best_loss, best_params, iterations_since_improvement = jax.lax.cond(
            loss_value < best_loss,
            true_fun,
            false_fun,
            (
                policy_params,
                best_params,
                loss_value,
                iterations_since_improvement,
                best_loss,
            ),
        )
        return (
            policy_params,
            best_params,
            iterations_since_improvement,
            opt_state,
            best_loss,
            loss_gradient,
        )

    def continue_fn(
        carry: Tuple[ArrayLike, ArrayLike, Int, PyTree, Float, ArrayLike]
    ) -> Bool:
        _, _, iterations_since_improvement, opt_state, _, g = carry
        # There are two counts:  the adam optimizer and the lr decay scheduler, we
        # want the former.
        n = optax.tree_utils.tree_get_all_with_path(opt_state, "count")[0][1]
        g_l2_norm = optax.tree_utils.tree_norm(g)

        return (n == 0) | (
            ((n < max_steps) & (g_l2_norm >= gtol))
            & (iterations_since_improvement <= patience)
        )

    # Optimisation loop
    best_loss = jnp.inf
    loss_value, loss_gradient = eqx.filter_value_and_grad(train_rollout)(
        policy, initial_train_particles, gp_model, timesteps
    )
    iterations_since_improvement = 0
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    _, best_policy_params, _, _, best_loss, _ = jax.lax.while_loop(
        continue_fn,
        make_step,
        (
            policy_params,
            copy.deepcopy(policy_params),
            iterations_since_improvement,
            opt_state,
            best_loss,
            loss_gradient,
        ),
    )

    return eqx.combine(best_policy_params, policy_static), best_loss
