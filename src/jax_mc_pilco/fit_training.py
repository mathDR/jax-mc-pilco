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

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

x, _ = env.reset()
replay_states = [x]

u = env.action_space.sample()
replay_actions = [u]

for _ in range(10_000):
    z = env.step(np.array(u))
    x = z[0]
    replay_states.append(x)
    u = env.action_space.sample()
    replay_actions.append(u)

states = jnp.array(replay_states)
actions = jnp.array(replay_actions)

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
