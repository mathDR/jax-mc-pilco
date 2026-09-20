"""Methods to generate data."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import numpy as np
from scipy.stats import qmc

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor

jax.config.update("jax_enable_x64", True)


def observation_to_qpos_qvel(
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts qpos and qvel from an 11-element InvertedDoublePendulum observation.
    """
    # 1. Reconstruct qpos (3 elements)
    cart_pos = obs[0]
    # np.arctan2 maps y (sin) and x (cos) to the range [-pi, pi]
    pole1_angle = np.arctan2(obs[1], obs[3])
    pole2_angle = np.arctan2(obs[2], obs[4])

    qpos = np.array([cart_pos, pole1_angle, pole2_angle])

    # 2. Reconstruct qvel (3 elements)
    qvel = np.array([obs[5], obs[6], obs[7]])

    return qpos, qvel


def generate_sobol_initial_states(
    num_states: int,
    dimensions: int,
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Generates low-discrepancy physical states for InvertedDoublePendulum-v4."""
    # 1. Dimensions: 1 cart pos, 2 link angles, 1 cart vel, 2 link angular vels = 6 dimensions
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    raw_samples = sampler.random(n=num_states)

    physical_states: np.ndarray = qmc.scale(raw_samples, l_bounds, u_bounds)
    return physical_states


def collect_mbrl_transitions(
    env: gym.Env,
    num_states: int,
    actions_per_state: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Forces the environment into a Sobol state, executes multiple action branches,
    and returns an MBRL transition dataset.
    """
    lb = np.array([-1.0, -0.6, -0.6, 0.8, 0.8, -5.0, -5.0, -5.0, -10.0])
    ub = np.array([1.0, 0.6, 0.6, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0])
    sobol_states = generate_sobol_initial_states(num_states, 9, lb, ub)

    # Storage arrays for MBRL training
    all_observations = []
    all_actions = []
    all_next_observations = []
    all_rewards = []

    # Unwrap to access MuJoCo mechanics directly
    raw_env = env.unwrapped

    for state in sobol_states:
        # Extract MuJoCo physics coordinates
        qpos, qvel = observation_to_qpos_qvel(state)
        observations = []
        actions = []
        next_observations = []
        rewards = []

        env.reset()
        raw_env.set_state(qpos, qvel)

        for _ in range(actions_per_state):
            obs = raw_env._get_obs()

            # C. Sample a random exploratory action
            action = env.action_space.sample()

            # D. Step the environment forward 1 timestep
            next_obs, reward, _, _, _ = env.step(action)

            # E. Store transition tuple
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
        all_observations.append(jnp.array(observations))
        all_actions.append(jnp.array(actions))
        all_next_observations.append(jnp.array(next_observations))
        all_rewards.append(jnp.array(rewards))

    return (
        jnp.array(all_observations),
        jnp.array(all_actions),
        jnp.array(all_next_observations),
        jnp.array(all_rewards),
    )


def collect_experience(
    env: gym.Env,
    num_steps: int,
    key: jtp.Key[jtp.Array, ""],
    actor: FlowActor | None = None,
    world_model: FlowDynamics | None = None,
    exploration: bool = True,
    use_sobol: bool = False,
) -> tuple[jtp.Float, jtp.Int, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples data from a Gymnasium environment using standard Python/NumPy,
    then converts the accumulated trajectory into a JAX array.
    """
    max_steps = num_steps if exploration else 1

    actions: list = []
    states: list = []
    next_states: list = []
    rewards: list = []
    max_episode_length = 0
    total_reward = 0.0

    if use_sobol and exploration:
        actions_per_state = 64
        states, actions, next_states, rewards = collect_mbrl_transitions(
            env,
            num_states=2 ** int(jnp.log2(num_steps)),
            actions_per_state=actions_per_state,
        )
        total_reward = 0.0
        max_episode_length = 0
    else:
        while len(actions) < max_steps:
            curr_state, _ = env.reset()

            s_curr_ep = []
            action_ep = []
            s_next_ep = []
            ep_rewards = []

            for _ in range(num_steps):
                key, ak = jax.random.split(key)
                action = env.action_space.sample() if actor is None else actor.sample_action(ak, jnp.array(curr_state))
                action_np = np.array(action)

                next_state, reward, terminated, truncated, _ = env.step(action_np)
                ep_rewards.append(float(reward))
                total_reward += float(reward)

                s_curr_ep.append(curr_state)
                action_ep.append(action_np)
                s_next_ep.append(next_state)

                curr_state = next_state
                if terminated or truncated:
                    break

            episode_length = len(action_ep)
            if episode_length > max_episode_length:
                max_episode_length = episode_length

            rewards.extend(ep_rewards)
            states.extend(s_curr_ep)
            next_states.extend(s_next_ep)
            actions.extend(action_ep)

    return (
        total_reward,
        max_episode_length,
        jnp.array(states),
        jnp.array(actions),
        jnp.array(next_states),
        jnp.array(rewards)[:, jnp.newaxis],
    )
