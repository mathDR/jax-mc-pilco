""" Optimize a controller on a given cost function."""

import copy
import equinox as eqx
import equinox.internal as eqxi
import optax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr
import gymnasium


def filter_cond(pred, true_fun, false_fun, *operands):
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = jax.lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)


@eqx.filter_jit
def fit_controller(  # noqa: PLR0913
    *,
    policy: eqx.Module,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    initial_state: ArrayLike,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    obj_func: Callable,
    optim: optax.GradientTransformation,
    key: ArrayLike = jr.PRNGKey(42),
    max_steps: Int = 100,
    patience: Int = 7,
    unroll: Int = 5,
    gtol: float = 1e-5,
) -> Tuple[eqx.Module, Float]:
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
            samples = model.get_samples(subkey, samples, actions, 1)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, actions))))
            return (policy_params, key, samples, total_cost + cost), cost

        total_cost = 0
        (policy_params, key, samples, total_cost), result = jax.lax.scan(
            one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
        )
        return total_cost

    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    # Optimisation step.

    def true_fun(arg):
        (
            policy_params,
            best_policy_params,
            loss_value,
            iterations_since_improvement,
            best_loss,
        ) = arg
        return (
            loss_value,
            policy_params,
            iterations_since_improvement - iterations_since_improvement,
        )

    def false_fun(arg):
        (
            policy_params,
            best_policy_params,
            loss_value,
            iterations_since_improvement,
            best_loss,
        ) = arg
        return best_loss, best_policy_params, iterations_since_improvement + 1

    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    @eqx.filter_jit
    def make_step(carry):
        (
            policy_params,
            best_policy_params,
            iterations_since_improvement,
            opt_state,
            best_loss,
        ) = carry
        policy = eqx.combine(policy_params, policy_static)

        loss_value, loss_gradient = eqx.filter_value_and_grad(train_rollout)(
            policy, initial_train_particles, gp_model, timesteps
        )
        updates, opt_state = optim.update(
            loss_gradient, opt_state, eqx.filter(policy, eqx.is_array)
        )

        policy = eqx.apply_updates(policy, updates)

        best_loss = jax.lax.cond(
            jnp.isfinite(best_loss),
            lambda z: z[0],
            lambda z: z[1],
            (best_loss, loss_value),
        )
        best_loss, best_policy_params, iterations_since_improvement = filter_cond(
            loss_value < best_loss,
            true_fun,
            false_fun,
            (
                policy_params,
                best_policy_params,
                loss_value,
                iterations_since_improvement,
                best_loss,
            ),
        )
        return (
            policy_params,
            best_policy_params,
            iterations_since_improvement,
            opt_state,
            best_loss,
        )

    def continue_fn(carry):
        _, _, iterations_since_improvement, opt_state, _ = carry
        n = optax.tree_utils.tree_get(opt_state[0], "count")
        g = optax.tree_utils.tree_get(opt_state, "grad")
        g_l2_norm = optax.tree_utils.tree_norm(g)
        return (
            (n == 0)
            | ((n < max_steps) & (g_l2_norm >= gtol))
            | (iterations_since_improvement <= patience)
        )

    # Optimisation loop
    best_loss = jnp.inf
    iterations_since_improvement = 0
    _, best_policy_params, _, _, best_loss = jax.lax.while_loop(
        continue_fn,
        make_step,
        (
            policy_params,
            policy_params,
            iterations_since_improvement,
            opt_state,
            best_loss,
        ),
    )

    return eqx.combine(best_policy_params, policy_static), best_loss
