"""Full end to end training."""

import gymnasium as gym
import jax

from jax_mc_pilco.simulators.collect_data import collect_experience
from jax_mc_pilco.training.learning import world_training_loop

jax.config.update("jax_enable_x64", True)

env = gym.make("InvertedPendulum-v5")
key = jax.random.key(seed=4)
key, subkey = jax.random.split(key)
_, max_episode_len, states, actions, next_states, rewards = collect_experience(env, 5_000, subkey, None)
print(f"Environment data collection complete. Sampled {states.shape[0]} points.")
# # ============================================================
# # Train FlowDynamics (world model) -- frozen afterwards
# # ============================================================
key, subkey = jax.random.split(key)
dynamics, final_hidden, losses = world_training_loop(subkey, states, actions)

print("Dynamics model fit complete. Final loss:", losses[-1])
breakpoint()
# # ============================================================
# # Train RewardGP (reward model) -- frozen afterwards
# # ============================================================
# key, subkey = jax.random.split(key)
# reward_fn = train_reward(states, actions, rewards, subkey)
# print("Reward model fit complete.")

# # ============================================================
# # 3. Instantiate the actor
# # ============================================================
# state_dim = states.shape[1]
# action_dim = actions.shape[1]

# act_low = jnp.array(env.action_space.low, dtype=float)
# act_high = jnp.array(env.action_space.high, dtype=float)
# key, actor_key = jax.random.split(key)

# actor = FlowActor(
#     key=actor_key,
#     state_dim=state_dim,
#     action_dim=action_dim,
#     action_low=act_low,
#     action_high=act_high,
#     flow_layers=4,
# )

# optim = optax.chain(
#     optax.clip_by_global_norm(1.0),
#     optax.adam(3e-4),
# )

# key, subkey = jax.random.split(key)
# actor = train_actor(actor, dynamics, states, subkey, optim, reward_fn)
