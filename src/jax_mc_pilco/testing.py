import jax
from jax import Array, config
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import jax.random as jr
from jaxtyping import ArrayLike, install_import_hook

config.update("jax_enable_x64", True)

key = jr.key(123)

import gymnasium as gym

from controllers import RandomController, Sum_of_Sinusoids, set_sos_params
from model_learning.gp_models import MGPR

import optax as ox

from typing import Tuple


def cart_pole_cost(
    states_sequence: ArrayLike,
    target_state: ArrayLike = jnp.array([jnp.pi, 0.0]),
    lengthscales: ArrayLike = jnp.array([3.0, 1.0]),
    angle_index: int = 2,
    pos_index: int = 0,
) -> Array:
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[pos_index]
    theta = states_sequence[angle_index]

    target_x = target_state[1]
    target_theta = target_state[0]

    return 1 - jnp.exp(
        -(jnp.square((jnp.abs(theta) - target_theta) / lengthscales[0]))
        - jnp.square((x - target_x) / lengthscales[1])
    )


num_particles = 400

num_trials = 5
T_sampling = 0.05
T_exploration = 3.0
T_control = 3.0
sim_timestep = 0.1

env = gym.make("InvertedPendulum-v5")
env_test = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0]
num_basis = 200
umax = 3.0
policy = Sum_of_Sinusoids(state_dim, action_dim, num_basis, True, umax)
policy_params = set_sos_params(action_dim, num_basis, -jnp.pi, jnp.pi, -1.0, 1.0)

# Initialize a random controller
exploration_policy = RandomController(state_dim, action_dim, True, 3.0)

# Randomly sample some points
key = jr.key(42)
x, _ = env.reset()
states = [x]
key, subkey = jr.split(key)
# u = env.action_space.sample()
u = exploration_policy(x, 0, subkey)
actions = [u]

for timestep in np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1):
    z = env.step(np.array(u))
    x = z[0]
    states.append(x)
    key, subkey = jr.split(key)
    # u = env.action_space.sample()
    u = exploration_policy(x, timestep, subkey)
    actions.append(u)

model = MGPR(states=jnp.array(states), actions=jnp.array(actions))
model.optimize()

# Now do a rollout with this model
# Generate an initial state
x, _ = env.reset()
key, subkey = jr.split(key)
# Generate an initial (action
u = exploration_policy(x, timestep, subkey)


test_inputs = model.data_to_gp_input(states, actions)
gp = model.build_gp(model.models[0], model.training_outputs[:, 0])
breakpoint()
cond_gp = gp.condition(model.training_outputs[:, 0], X_test=test_inputs).gp
