#!/usr/bin/env python
# coding: utf-8

# Testing the MC-PILCO framework
import time
import jax
import equinox as eqx
from jax import Array, config
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from jaxtyping import ArrayLike, install_import_hook, Array, Float, Int, PyTree
from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

import gymnasium as gym

from jax_mc_pilco.controllers import Controller, RandomController, SumOfGaussians
from jax_mc_pilco.rewards import pendulum_cost  # , cart_pole_cost
from jax_mc_pilco.model_learning.dynamical_models import (
    IMGPR,
    IMSGPR,
    optimize_imgpr,
    optimize_imsgpr,
)
from jax_mc_pilco.policy_learning.rollout import fit_controller
from jax_mc_pilco.simulators.simulation import remake_state, sample_from_environment
from jax_mc_pilco.model_learning.gp.kernels import ExpSquared

import optax

from IPython import display

## Globals

num_particles = 400
num_trials = 8
num_inducing_points = 200
T_sampling = 0.05
T_exploration = 0.35
T_control = 3.0
sim_timestep = 0.01
starting_dropout_probability = 0.25
control_horizon = int(T_control / T_sampling)
num_basis = 200
umax = 2.0


## Generate the environments

env = gym.make("Pendulum-v1")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0]
# state is cos_theta, sin_theta, theta_dot

timesteps = np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1)

## Initialize the Controllers

key = jr.key(42)

random_policy = RandomController(state_dim, action_dim, to_squash=True, max_action=umax)

control_policy = SumOfGaussians(
    state_dim,
    action_dim,
    num_basis,
    initial_log_lengthscales=None,
    initial_centers=None,
    to_squash=True,
    max_action=umax,
    key=key,
)

def train_rollout(
    policy: Controller,
    init_samples: ArrayLike,
    model: eqx.Module,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(123),
) -> Float:
    p_params, p_static = eqx.partition(policy, eqx.is_array)

    def one_rollout_step(
        carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float], timestep: Float
    ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float], Float]:
        params, key, samples, total_cost = carry
        policy = eqx.combine(params, p_static)
        actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))
        samples = model.get_samples(key, samples, actions, 1)
        cost = jnp.mean(jax.vmap(pendulum_cost)(jnp.hstack((samples, actions))))
        return (params, key, samples, total_cost + cost), cost


    total_cost = 0
    (params, key, samples, total_cost), result = jax.lax.scan(
       one_rollout_step, (p_params, key, init_samples, total_cost), timesteps
    )
    return total_cost

# Main Loop

model_params = [
    {
        "kernel": {
            "coefficient": jnp.array(0.9, dtype=jnp.float64),
            "log_scale": jnp.array(0.0, dtype=jnp.float64),
        },
        "mean": {},
        "likelihood": {
            "log_diag": jnp.array(-0.5, dtype=jnp.float64),
        },
    }
] * state_dim

states = []
actions = []
epsilon = 1e-4
timesteps=jnp.arange(control_horizon)
for trial in range(2):
    num_opt_steps = 5
    key, subkey = jr.split(key)
    these_states, these_actions = sample_from_environment(env, timesteps, num_trials,control_policy, key)
    states.extend(these_states)
    actions.extend(these_actions)

    states_array = jnp.array(states)
    actions_array = jnp.array(actions)
    model = IMGPR(
        states=states_array,
        actions=actions_array,
        kernel_funcs=ExpSquared,
        params=model_params,
    )
    model = optimize_imgpr(
        model,
        states=states_array,
        actions=actions_array,
    )

    init_state = [1e-6, 1e-6]  # Cannot use zero because of the reset
    cosine_decay_scheduler = optax.cosine_decay_schedule(
        0.0001, decay_steps=num_opt_steps, alpha=0.95
    )

    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    sample_train, _ = env.reset(
        options={"x_init": init_state[0], "y_init": init_state[1]}
    )
    action_train = control_policy(sample_train, 0.0)
    key, subkey = jr.split(key)

    initial_train_particles = model.get_samples(
        subkey, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )

    print(
        f"Initial cost = {train_rollout(control_policy,initial_train_particles,model,timesteps,key)}"
    )

    new_control_policy, best_loss = fit_controller(
        policy=control_policy,
        env=env,
        num_particles=num_particles,
        initial_state=init_state,
        timesteps=timesteps,
        gp_model=model,
        obj_func=pendulum_cost,
        optim=optimizer,
        max_steps=num_opt_steps,
        key=key,
    )
    print(f"Best Loss = {best_loss}")
