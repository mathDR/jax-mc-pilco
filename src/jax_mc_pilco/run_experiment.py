"""This script contains the full mc-pilco loop."""

import jax
import equinox as eqx
from jax import Array, config
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from jaxtyping import ArrayLike, install_import_hook, Array, Float, Int, PyTree

config.update("jax_enable_x64", True)

key = jr.key(123)

import gymnasium as gym

from controllers import Controller, RandomController, SumOfGaussians
from rewards import cart_pole_cost
from model_learning.gp_models import MGPR
from policy_learning.rollout import fit_controller
from simuators.simulation import sample_from_environment

import optax as ox

# Globals
num_particles = 400
num_trials = 8
T_sampling = 0.05
T_exploration = 0.35
T_control = 3.0
sim_timestep = 0.01
starting_dropout_probability = 0.25
control_horizon = int(T_control / T_sampling)
num_basis = 200
umax = 3.0

env = gym.make("InvertedPendulum-v5")
env_test = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0] + 1

timesteps = np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1)

random_policy = RandomController(state_dim, action_dim, to_squash=True, max_action=umax)

control_policy = SumOfGaussians(
    state_dim,
    action_dim,
    num_basis,
    initial_log_lengthscales=None,
    initial_centers=None,
    use_dropout=True,
    dropout_probability=starting_dropout_probability,
    to_squash=True,
    max_action=umax,
)

states, actions = sample_from_environment(
    env, timesteps, num_trials, random_policy, key
)
for outer_loop_iteration in range(5):
    optimizer = ox.inject_hyperparams(ox.adam)(learning_rate=1e-2)
    model = MGPR(states=jnp.array(states), actions=jnp.array(actions))
    model.optimize()
    control_policy, losses = fit_controller(
        policy=control_policy,
        starting_dropout_probability=starting_dropout_probability,
        env=env,
        num_particles=num_particles,
        timesteps=jnp.arange(control_horizon),
        gp_model=model,
        obj_func=cart_pole_cost,
        optim=optimizer,
        num_iters=1000,
    )
    these_states, these_actions = sample_from_environment(
        env, timesteps, num_trials, control_policy, key
    )
    states.extend(these_states)
    actions.extend(these_actions)
print(len(states))
breakpoint()
