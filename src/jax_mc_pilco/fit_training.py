"""Full end to end training."""

import gymnasium as gym
import jax
import jax.numpy as jnp

from jax_mc_pilco.simulators.collect_data import collect_experience
from jax_mc_pilco.training.learning import actor_critic_training_loop, world_training_loop

jax.config.update("jax_enable_x64", True)

env = gym.make("InvertedPendulum-v5")

key = jax.random.key(seed=4)
key, subkey = jax.random.split(key)
_, max_episode_len, states, actions, next_states, rewards = collect_experience(env, 5_000, subkey, None)
print(f"Environment data collection complete. Sampled {states.shape[0]} points.")

key, subkey = jax.random.split(key)
dynamics, d_losses = world_training_loop(subkey, states, actions)

print("Dynamics model fit complete. Final validation loss:", d_losses["val"][-1])

key, subkey = jax.random.split(key)
actor, critic, ac_losses = actor_critic_training_loop(
    subkey,
    states,
    actions,
    jnp.min(actions, axis=0),
    jnp.max(actions, axis=0),
    dynamics,
)
breakpoint()
