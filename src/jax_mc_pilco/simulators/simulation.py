"""Functions for interacting with the gymnasium environments."""

import numpy as np
import gymnasium as gym
from jaxtyping import Array, ArrayLike, Int
from ..controllers import Controller
import jax.random as jr
from jax import config

config.update("jax_enable_x64", True)


def remake_state(x):
    return np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])


def sample_from_environment(
    env: gym.wrappers.common.TimeLimit,
    timesteps: ArrayLike,
    num_trials: Int,
    policy: Controller,
    key: ArrayLike,
) -> Tuple[Array, Array]:
    # Randomly sample some points
    key = jr.key(42)
    x, _ = env.reset()
    state = remake_state(x)
    states = [state]
    key, subkey = jr.split(key)
    u = policy(state, 0.0)
    actions = [u]

    for timestep in timesteps:
        z = env.step(np.array(u))
        x = z[0]
        state = remake_state(x)
        states.append(state)
        key, subkey = jr.split(key)
        u = policy(state, timestep)
        actions.append(u)

    for _ in range(num_trials - 1):
        x, _ = env.reset()
        state = remake_state(x)
        states.append(state)
        key, subkey = jr.split(key)
        policy(state, 0.0)
        # u = exploration_policy(x,0,subkey)
        actions.append(u)

        for timestep in timesteps:
            z = env.step(np.array(u))
            x = z[0]
            state = remake_state(x)
            states.append(state)
            key, subkey = jr.split(key)
            policy(state, timestep)
            actions.append(u)

    return states, actions
