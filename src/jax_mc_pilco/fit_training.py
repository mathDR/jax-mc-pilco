<<<<<<< HEAD
# """
# End-to-end: collect data -> train FlowDynamics world model -> train FlowActor
# by backpropagating expected reward through the (frozen) learned dynamics.
# """

# import gymnasium as gym
# import jax
# import jax.numpy as jnp
# import optax

# from jax_mc_pilco.policy_learning.action_flows import FlowActor
# from jax_mc_pilco.training.learning import collect_experience, train_actor, train_flow, train_reward

# jax.config.update("jax_enable_x64", True)


# # ============================================================
# # Environment + data collection
# # ============================================================
# env = gym.make("InvertedDoublePendulum-v5")  # gym.make("InvertedPendulum-v5")
=======
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

jax.config.update("jax_enable_x64", True)

env = gym.make("InvertedPendulum-v5")
>>>>>>> parent of 4769f5e (updated codez)

# key = jax.random.key(seed=4)
# key, subkey = jax.random.split(key)
# states, actions, next_states, rewards = collect_experience(env, 5_000, subkey, None)
# print(f"Environment data collection complete. Sampled {states.shape[0]} points.")

<<<<<<< HEAD
# # ============================================================
# # Train FlowDynamics (world model) -- frozen afterwards
# # ============================================================
# key, subkey = jax.random.split(key)
# dynamics, losses = train_flow(states, actions, next_states, subkey)

# print("Dynamics model fit complete. Final loss:", losses["val"][-1])
# # ============================================================
# # Train RewardGP (reward model) -- frozen afterwards
# # ============================================================
# key, subkey = jax.random.split(key)
# reward_fn = train_reward(states, actions, rewards, subkey)
# print("Reward model fit complete.")
=======
x, _ = env.reset()
replay_states = [x]
>>>>>>> parent of 4769f5e (updated codez)

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

<<<<<<< HEAD
# optim = optax.chain(
#     optax.clip_by_global_norm(1.0),
#     optax.adam(3e-4),
# )

# key, subkey = jax.random.split(key)
# actor = train_actor(actor, dynamics, states, subkey, optim, reward_fn)
=======
key = jax.random.key(seed=4)
key, subkey = jax.random.split(key)

cond_dim = state_dim + action_dim

base_dist = MultivariateNormal(
    loc=jnp.zeros(state_dim, dtype=float),
    covariance=jnp.eye(state_dim, dtype=float),
)

state_space_flow = coupling_flow(
    key=key,
    base_dist=base_dist,
    cond_dim=cond_dim,
    nn_width=256,
    nn_depth=2,
    flow_layers=4,
)

# 1. Prepare sequential transition data from your raw JAX arrays
# For a trajectory sequence: s_t -> a_t -> s_{t+1}
s_curr_data = states[:-1]
action_data = actions[:-1]
s_next_data = states[1:]
num_samples = s_curr_data.shape[0]

context = jnp.concatenate([s_curr_data, action_data], axis=-1)
delta_s = s_next_data - s_curr_data

fit_flow, losses = fit_to_data(
    key=subkey,
    dist=state_space_flow,
    data=(delta_s, context),
    learning_rate=5e-3,
    max_patience=10,
    max_epochs=70,
)

# Now construct a flowmodel and assign this fitted flow to the flow object

# Should rewrite fit_to_data where instead of data to pass in, we can pass in a MOGP and train against that.

# Now we should initialize an action flow then optimize that w.r.t. the env reward using the state space flow model.
>>>>>>> parent of 4769f5e (updated codez)
