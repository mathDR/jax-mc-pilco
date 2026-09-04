"""
End-to-end: collect data -> train FlowDynamics world model -> train FlowActor
by backpropagating expected reward through the (frozen) learned dynamics.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.training.learning import collect_experience, train_actor, train_flow, train_reward

jax.config.update("jax_enable_x64", True)


# ============================================================
# Environment + data collection
# ============================================================
env = gym.make("InvertedDoublePendulum-v5")  # gym.make("InvertedPendulum-v5")

key = jax.random.key(seed=4)
key, subkey = jax.random.split(key)
states, actions, next_states, rewards = collect_experience(env, 5_000, subkey, None)
print(f"Environment data collection complete. Sampled {states.shape[0]} points.")

# ============================================================
# Train FlowDynamics (world model) -- frozen afterwards
# ============================================================
key, subkey = jax.random.split(key)
dynamics, losses = train_flow(states, actions, next_states, subkey)

print("Dynamics model fit complete. Final loss:", losses["val"][-1])
# ============================================================
# Train RewardGP (reward model) -- frozen afterwards
# ============================================================
key, subkey = jax.random.split(key)
reward_fn = train_reward(states, actions, rewards, subkey)
print("Reward model fit complete.")

# ============================================================
# 3. Instantiate the actor
# ============================================================
state_dim = states.shape[1]
action_dim = actions.shape[1]

act_low = jnp.array(env.action_space.low, dtype=float)
act_high = jnp.array(env.action_space.high, dtype=float)
key, actor_key = jax.random.split(key)

actor = FlowActor(
    key=actor_key,
    state_dim=state_dim,
    action_dim=action_dim,
    action_low=act_low,
    action_high=act_high,
    flow_layers=4,
)

optim = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(3e-4),
)

key, subkey = jax.random.split(key)
actor = train_actor(actor, dynamics, states, subkey, optim, reward_fn)
