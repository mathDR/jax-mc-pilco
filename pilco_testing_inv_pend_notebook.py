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


def train_rollout(
    policy: eqx.Module,
    init_samples: ArrayLike,
    model: eqx.Module,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(123),
) -> Float:
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    def one_rollout_step(
        carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float], timestep: Float
    ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float], Float]:
        policy_params, key, samples, total_cost = carry
        policy = eqx.combine(policy_params, policy_static)
        actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))

        key, subkey = jr.split(key)
        samples = model.get_samples(subkey, samples, actions, 1)
        cost = jnp.mean(jax.vmap(pendulum_cost)(jnp.hstack((samples, actions))))
        return (policy_params, key, samples, total_cost + cost), cost

    total_cost = 0
    (policy_params, key, samples, total_cost), result = jax.lax.scan(
        one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
    )
    return total_cost


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

# Main Loop

params = [
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
for trial in range(4):
    if trial == 0:
        exploration_policy = random_policy
        num_opt_steps = 2
        params = params
    else:
        exploration_policy = control_policy
        num_opt_steps = 3
        params = model.params  # Start from previous model (?)

    key, subkey = jr.split(key)
    these_states, these_actions = sample_from_environment(
        env, timesteps, num_trials, exploration_policy, subkey
    )
    states.extend(these_states)
    actions.extend(these_actions)

    states_array = jnp.array(states)
    actions_array = jnp.array(actions)
    if len(states) < -1:
        # Update initial inducing point locations (if required)
        if "inducing_point_locations" not in params[0].keys():
            initial_inducing_points = jnp.linspace(
                jnp.array(states).min(),
                jnp.array(states).max(),
                num=num_inducing_points,
                endpoint=True,
            )
            for i in range(state_dim):
                params[i]["inducing_point_locations"] = np.array(
                    initial_inducing_points, dtype=jnp.float64
                )
        model = IMSGPR(
            states=states_array,
            actions=actions_array,
            kernel_funcs=ExpSquared,
            num_inducing_points=num_inducing_points,
            params=params,
        )
        start_time = time.perf_counter()
        model = optimize_imsgpr(
            model,
            states=states_array,
            actions=actions_array,
        )
        end_time = time.perf_counter()
        print(f"Model Optimization Time = {end_time-start_time}")
    else:
        model = IMGPR(
            states=states_array,
            actions=actions_array,
            kernel_funcs=ExpSquared,
            params=params,
        )
        start_time = time.perf_counter()
        model = optimize_imgpr(
            model,
            states=states_array,
            actions=actions_array,
        )
        end_time = time.perf_counter()
        print(f"Model Optimization Time = {end_time-start_time}")

    factor = min(1.0, max(0.0, (trial - 5) / 20.0))
    if factor == 0.0:
        init_state = [1e-6, 1e-6]  # Cannot use zero because of the reset
    else:
        key, subkey = jr.split(key)
        init_state = [float(factor * jnp.pi * jr.uniform(subkey))]
        key, subkey = jr.split(key)
        init_state.extend([float(factor * epsilon * jr.uniform(subkey))])
    key, subkey = jr.split(key)

    cosine_decay_scheduler = optax.cosine_decay_schedule(
        0.0001, decay_steps=num_opt_steps, alpha=0.95
    )
    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)

    sample_train, _ = env.reset(
        options={"x_init": init_state[0], "y_init": init_state[1]}
    )
    # Generate an initial action
    action_train = control_policy(sample_train, 0.0)
    # initialize some particles
    key, subkey = jr.split(key)
    initial_train_particles = model.get_samples(
        subkey, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )
    print(
        f"Initial cost = {train_rollout(control_policy,initial_train_particles,model,timesteps,subkey)}"
    )

    start_time = time.perf_counter()
    control_policy, best_loss = fit_controller(
        policy=control_policy,
        env=env,
        num_particles=num_particles,
        initial_state=init_state,
        timesteps=jnp.arange(control_horizon),
        gp_model=model,
        obj_func=pendulum_cost,
        optim=optimizer,
        max_steps=num_opt_steps,
        key=subkey,
    )
    end_time = time.perf_counter()
    print(f"Policy Optimization Time = {end_time-start_time}")
    print(f"Best Loss = {best_loss}")
    key, subkey = jr.split(key)
    print(
        f"Final cost = {train_rollout(control_policy,initial_train_particles,model,timesteps,subkey)}"
    )
