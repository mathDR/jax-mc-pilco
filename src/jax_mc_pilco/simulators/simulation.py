"""Functions for interacting with the gymnasium environments."""

import equinox as eqx  # type: ignore
import gymnasium as gym  # type: ignore
from jaxtyping import Array, ArrayLike, Int
from jax_mc_pilco.controllers import Controller
import jax.random as jr
from jax import config
import jax.numpy as jnp
import numpy as np
from typing import Tuple

config.update("jax_enable_x64", True)


def remake_state(x):
    theta = jnp.atan2(x[..., 1], x[..., 0])
    return jnp.array([
        jnp.cos(theta),
        jnp.sin(theta),
        jnp.clip(x[..., 2], min=-8, max=8)
    ]).T


def sample_from_environment(
    env: gym.wrappers.common.TimeLimit,
    timesteps: ArrayLike,
    num_trials: Int,
    policy: Controller,
    *,
    model: eqx.Module | None,
    key: ArrayLike | None,
) -> Tuple[Array, Array]:
    """Sample from the `env` environment using the passed in policy."""

    if key is None:
        key = jr.key(42)
    if model is None:
        x, _ = env.reset()
        state = remake_state(x)
        ret_states = [state]
        u = env.action_space.sample()
        ret_actions = [u]
        for timestep in timesteps:
            z = env.step(np.array(u))
            x = z[0]
            state = remake_state(x)
            ret_states.append(state)
            u = env.action_space.sample()
            ret_actions.append(u)

        for _ in range(num_trials - 1):
            x, _ = env.reset()
            state = remake_state(x)
            ret_states.append(state)
            u = env.action_space.sample()
            ret_actions.append(u)
            for timestep in timesteps:
                z = env.step(np.array(u))
                x = z[0]
                state = remake_state(x)
                ret_states.append(state)
                u = env.action_space.sample()
                ret_actions.append(u)
    else:
        # Initally seed the state that the policy requires
        x, _ = env.reset()
        state = remake_state(x)
        states = [state]
        u = env.action_space.sample()
        actions = [u]

        for i in range(max(model.position_memory, model.control_memory)):
            z = env.step(np.array(u))
            x = z[0]
            state = remake_state(x)
            states.append(state)
            u = env.action_space.sample()
            actions.append(u)

        key, subkey = jr.split(key)
        policy_input = model.data_to_policy_input(
            jnp.array(states),
            jnp.array(actions)
        )
        u = policy(policy_input, 0.0, subkey)
        ret_actions = []
        ret_states = []

        for timestep in timesteps:
            z = env.step(np.array(u))
            x = z[0]
            state = remake_state(x)
            ret_states.append(state)
            states.append(state)
            states.pop(-1)
            key, subkey = jr.split(key)
            policy_input = model.data_to_policy_input(
                jnp.array(states),
                jnp.array(actions)
            )
            u = policy(policy_input, 0.0, subkey)
            actions.append(u)
            actions.pop(-1)
            ret_actions.append(u)

        for _ in range(num_trials - 1):
            x, _ = env.reset()
            state = remake_state(x)
            states = [state]
            u = env.action_space.sample()
            actions = [u]
            for i in range(max(model.position_memory, model.control_memory)):
                z = env.step(np.array(u))
                x = z[0]
                state = remake_state(x)
                states.append(state)
                u = env.action_space.sample()
                actions.append(u)

            u = policy(policy_input, 0.0, subkey)

            for timestep in timesteps:
                z = env.step(np.array(u))
                x = z[0]
                state = remake_state(x)
                states.append(state)
                states.pop(-1)
                ret_states.append(state)
                key, subkey = jr.split(key)
                policy_input = model.data_to_policy_input(
                    jnp.array(states),
                    jnp.array(actions)
                )
                u = policy(policy_input, 0.0, subkey)
                actions.append(u)
                actions.pop(-1)
                ret_actions.append(u)

    return ret_states, ret_actions
