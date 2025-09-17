""" Optimize a controller on a given cost function."""

import equinox as eqx
import gymnasium
import optax
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr


from jax_mc_pilco.controllers import Controller


def generate_starting_particles(
    policy: Controller,
    env: gymnasium.wrappers.common.TimeLimit,
    num_particles: Int,
    initial_state: ArrayLike,
    gp_model: eqx.Module,
    key: ArrayLike,
) -> jax.Array:
    sample, _ = env.reset(
        options={"x_init": initial_state[0], "y_init": initial_state[1]}
    )
    # Generate an initial action
    action = policy(sample, 0.0)
    # initialize the starting particles for the training
    key, subkey = jr.split(key)
    return gp_model.get_samples(
        subkey, jnp.array([sample]), jnp.array([action]), num_particles
    )


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
) -> eqx.Module:
    """The optimization loop for fitting the policy parameters.

    It returns the optimized policy and early stops based on an
    estimation of validation loss.  Unlike for neural networks, we
    do not split the data, but instead restart the rollout from
    different particles.
    """
    # Generate the starting samples for the training objective
    key, subkey = jr.split(key)
    initial_train_particles = generate_starting_particles(
        policy,
        env,
        num_particles,
        initial_state,
        gp_model,
        subkey,
    )
    key, subkey = jr.split(key)
    initial_val_particles = generate_starting_particles(
        policy,
        env,
        num_particles,
        initial_state,
        gp_model,
        subkey,
    )

    @eqx.debug.assert_max_traces(max_traces=2)
    def rollout(
        policy: Controller,
        init_samples: ArrayLike,
        model: eqx.Module,
        timesteps: ArrayLike,
        key: ArrayLike = jr.key(42),
    ) -> Float:
        """The function that produces the rollout.
        It starts from the init_samples and utilizes the policy to
        generate actions that allow for sampling from the model to
        get the next samples.
        """
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

    # Optimization step.
    def make_step(
        carry: Tuple[
            ArrayLike,
            Int,
            PyTree,
            Float,
        ]
    ) -> Tuple[ArrayLike, Int, PyTree, Float,]:
        (
            policy_params,
            iterations_since_improvement,
            opt_state,
            last_val_loss,
        ) = carry
        policy = eqx.combine(policy_params, policy_static)
        loss_gradient = eqx.filter_grad(rollout)(
            policy, initial_train_particles, gp_model, timesteps
        )
        updates, opt_state = optim.update(loss_gradient, opt_state, policy_params)
        policy = eqx.apply_updates(policy, updates)
        policy_params = eqx.filter(policy, eqx.is_array)

        val_loss = rollout(policy, initial_val_particles, gp_model, timesteps)

        iterations_since_improvement = jax.lax.cond(
            val_loss < last_val_loss,
            lambda x: x - x,
            lambda x: x + 1,
            iterations_since_improvement,
        )

        return (
            policy_params,
            iterations_since_improvement,
            opt_state,
            val_loss,
        )

    def continue_fn(carry: Tuple[ArrayLike, Int, PyTree, Float]) -> Bool:
        _, iterations_since_improvement, opt_state, _ = carry
        # There are two counts:  the adam optimizer and the lr decay scheduler,
        # we want the former.
        n = optax.tree_utils.tree_get_all_with_path(opt_state, "count")[0][1]

        return (n == 0) | ((n < max_steps) & (iterations_since_improvement <= patience))

    # Optimisation loop

    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    policy_params, _, _, _ = jax.lax.while_loop(
        continue_fn,
        make_step,
        (
            policy_params,
            0,
            opt_state,
            jnp.inf,
        ),
    )

    return eqx.combine(policy_params, policy_static)
