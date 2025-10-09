""" Optimize a controller on a given cost function."""

import equinox as eqx   # type: ignore
import optax
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, Float, Int, PyTree
from typing import Callable, Tuple
import jax.random as jr

from jax_mc_pilco.controllers import Controller


@eqx.filter_jit
def fit_controller(  # noqa: PLR0913
    *,
    policy: Controller,
    init_states: ArrayLike,
    init_actions: ArrayLike,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    obj_func: Callable,
    optim: optax.GradientTransformation,
    key: ArrayLike = jr.key(42),
    max_steps: Int = 100,
    patience: Int = 7,
) -> eqx.Module:
    """The optimization loop for fitting the policy parameters.

    It returns the optimized policy and early stops based on an
    estimation of validation loss.  Unlike for neural networks, we
    do not split the data, but instead restart the rollout from
    different particles.
    """

    @eqx.debug.assert_max_traces(max_traces=2)
    def rollout(
        policy: Controller,
        init_states: ArrayLike,
        init_actions: ArrayLike,
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

        def update_actions(action: ArrayLike, actions: ArrayLike):
            """Append action to the front of actions and pop off last actions
               value.
            """
            return jnp.concatenate(
                [
                    action[:, jnp.newaxis, :],
                    actions[:, :-1, :]
                ],
                axis=1
            )

        def update_states(state: ArrayLike, states: ArrayLike):
            """Append state to the front of states and pop off last states
               value.
            """
            theta = jnp.atan2(state[:, 1], state[:, 0])
            new_state = jnp.array(
                [jnp.cos(theta),
                 jnp.sin(theta),
                 jnp.clip(state[:, 2], min=-8, max=8)
                 ]
            ).T
            return jnp.concatenate(
                [new_state[:, jnp.newaxis, :], states[:, :-1, :]],
                axis=1
            )

        def one_rollout_step(
            carry: Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Float],
            timestep: Float,
        ) -> Tuple[
            Tuple[
                ArrayLike,
                ArrayLike,
                ArrayLike,
                ArrayLike,
                Float
                ],
            Float
        ]:
            params, key, states, actions, total_cost = carry
            policy = eqx.combine(params, p_static)
            # Compute the action from the most recent state
            policy_input = jax.vmap(
                model.data_to_policy_input
            )(states, actions)
            action = jax.vmap(
                policy, in_axes=(0, None)
            )(policy_input, timestep)
            actions = update_actions(action, actions)
            samples = jax.vmap(
                model.get_samples, in_axes=(None, 0, 0, None)
            )(subkey, states, actions, 1)
            states = update_states(samples, states)
            cost = jnp.mean(jax.vmap(obj_func)(jnp.hstack((samples, action))))

            return (params, key, states, actions, total_cost + cost), cost

        total_cost = 0
        key, subkey = jr.split(key)
        (_, _, _, _, total_cost), _ = jax.lax.scan(
            one_rollout_step,
            (p_params, subkey, init_states, init_actions, total_cost),
            timesteps,
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
            ArrayLike,
        ]
    ) -> Tuple[ArrayLike, Int, PyTree, Float, ArrayLike]:
        (
            policy_params,
            iterations_since_improvement,
            opt_state,
            last_val_loss,
            key,
        ) = carry
        policy = eqx.combine(policy_params, policy_static)
        key, subkey = jr.split(key)
        loss_gradient = eqx.filter_grad(rollout)(
            policy, init_states, init_actions, gp_model, timesteps, subkey
        )
        updates, opt_state = optim.update(
            loss_gradient,
            opt_state,
            policy_params
            )
        policy = eqx.apply_updates(policy, updates)
        policy_params = eqx.filter(policy, eqx.is_array)

        key, subkey = jr.split(key)
        val_loss = rollout(
            policy,
            init_states,
            init_actions,
            gp_model,
            timesteps,
            subkey
        )

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
            key,
        )

    def continue_fn(carry: Tuple[ArrayLike, Int, PyTree, Float]) -> Bool:
        _, iterations_since_improvement, opt_state, _, _ = carry
        # There are two counts:  the adam optimizer and the lr decay scheduler,
        # we want the former.
        n = optax.tree_utils.tree_get_all_with_path(opt_state, "count")[0][1]

        return (
            (n == 0) |
            ((n < max_steps) & (iterations_since_improvement <= patience))
        )

    # Optimisation loop

    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    key, subkey = jr.split(key)
    policy_params, _, _, _, _ = jax.lax.while_loop(
        continue_fn,
        make_step,
        (
            policy_params,
            0,
            opt_state,
            jnp.inf,
            subkey,
        ),
    )

    return eqx.combine(policy_params, policy_static)
