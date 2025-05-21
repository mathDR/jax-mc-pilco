"""This script contains the full mc-pilco loop."""

import jax
import equinox as eqx
from jax import Array, config
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from jaxtyping import ArrayLike, install_import_hook, Array, Float, Int, PyTree
from typing import List

config.update("jax_enable_x64", True)

key = jr.key(123)

import gymnasium as gym

from jax_mc_pilco.controllers import Controller, RandomController, SumOfGaussians
from jax_mc_pilco.rewards import pendulum_cost  # , cart_pole_cost
from jax_mc_pilco.model_learning.gp_models import IMGPR
from jax_mc_pilco.policy_learning.rollout import fit_controller
from jax_mc_pilco.simulators.simulation import remake_state, sample_from_environment

import optax as ox

# Globals
num_particles = 400
num_trials = 8
T_sampling = 0.05
T_exploration = 0.35
T_control = 3.0
sim_timestep = 0.01
control_horizon = int(T_control / T_sampling)
num_basis = 200
umax = 2.0
num_init_conditions = 100


# Create an ensemble of controllers
@eqx.filter_vmap
def make_ensemble(key):
    return SumOfGaussians(
        state_dim,
        action_dim,
        num_basis,
        initial_log_lengthscales=None,
        initial_centers=None,
        to_squash=True,
        max_action=umax,
        key=key,
    )


def make_initial_samples(
    init_state: List,
    num_samples: Int,
    env: gym.wrappers.common.TimeLimit,
) -> Array:
    init_samples = []
    for i in range(num_samples):
        sample, _ = env.reset(
            options={"x_init": init_state[0], "y_init": init_state[1]}
        )
        init_samples.append(sample)
    return jnp.array(init_samples)


def best_control_policy(all_policies: Controller) -> Controller:
    """This function should determine the 'best' controller from the ensemble and return a controller with those parameters."""
    pass


env = gym.make("Pendulum-v1")
env_test = gym.make("Pendulum-v1", render_mode="rgb_array")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0] + 1

timesteps = jnp.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1)

random_policy = RandomController(state_dim, action_dim, to_squash=True, max_action=umax)


keys = jax.random.split(key, num_init_conditions)
control_policies = make_ensemble(keys)

states, actions = sample_from_environment(
    env, timesteps, num_trials, random_policy, key
)

states = []
actions = []
epsilon = 1e-4
for trial in range(10):
    optimizer = ox.inject_hyperparams(ox.adam)(learning_rate=1e-2)
    if trial == 0:
        exploration_policy = random_policy
        num_opt_steps = 2000
    else:
        exploration_policy = best_control_policy(control_policies)
        num_opt_steps = 4000

    key, subkey = jr.split(key)
    these_states, these_actions = sample_from_environment(
        env, timesteps, num_trials, exploration_policy, subkey
    )
    states.extend(these_states)
    actions.extend(these_actions)

    model = IMGPR(states=jnp.array(states), actions=jnp.array(actions))
    model.optimize()

    factor = min(1.0, max(0.0, (trial - 5) / 20.0))
    if factor == 0.0:
        init_state = [1e-2, 1e-2]  # Cannot use zero because of the reset
    else:
        key, subkey = jr.split(key)
        init_state = [float(factor * jnp.pi * jr.uniform(subkey))]
        key, subkey = jr.split(key)
        init_state.extend([float(factor * epsilon * jr.uniform(subkey))])
    breakpoint()
    key, subkey = jr.split(key)
    init_samples = make_initial_samples(init_state, num_init_conditions, env)
    control_policies, losses = fit_controller(
        policies=control_policies,
        initial_samples=init_samples,
        num_particles=num_particles,
        timesteps=jnp.arange(control_horizon),
        gp_model=model,
        obj_func=pendulum_cost,
        optim=optimizer,
        num_iters=num_opt_steps,
        key=subkey,
    )
    # plt.plot(losses)
    # plt.title(trial)
    # plt.show()


breakpoint()
