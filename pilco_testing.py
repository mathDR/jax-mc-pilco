# Testing the MC-PILCO framework
import time
import optax
import jax
import equinox as eqx
import gymnasium as gym
from jax import config
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from jaxtyping import ArrayLike, Float
from typing import Tuple

from jax_mc_pilco.controllers import Controller, RandomController, SumOfGaussians
from jax_mc_pilco.rewards import pendulum_cost  # , cart_pole_cost
from jax_mc_pilco.model_learning.dynamical_models import (
    IMGPR,
    optimize_imgpr,
)
from jax_mc_pilco.policy_learning.rollout import fit_controller
from jax_mc_pilco.simulators.simulation import sample_from_environment
from jax_mc_pilco.model_learning.gp.kernels import ExpSquared

config.update("jax_enable_x64", True)

# Globals

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

# Generate the environments
env = gym.make("Pendulum-v1")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0]
# state is cos_theta, sin_theta, theta_dot

timesteps = np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1)
# Initialize the Controllers

key = jr.key(42)

random_policy = RandomController(state_dim, action_dim, to_squash=True, max_action=umax)
key, subkey = jr.split(key)
control_policy = SumOfGaussians(
    state_dim,
    action_dim,
    num_basis,
    initial_log_lengthscales=None,
    initial_centers=None,
    to_squash=True,
    max_action=umax,
    key=subkey,
)


def policy_rollout_with_std(
    policy: Controller,
    init_samples: ArrayLike,
    model: eqx.Module,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(42),
) -> Tuple[Float, Float]:
    p_params, p_static = eqx.partition(policy, eqx.is_array)

    def one_rollout_step(
        carry: Tuple[ArrayLike, ArrayLike, ArrayLike, Float, Float],
        timestep: Float,
    ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Float, Float], Float]:
        params, key, samples, total_cost, total_var = carry
        policy = eqx.combine(params, p_static)
        actions = jax.vmap(policy)(samples, jnp.tile(timestep, num_particles))
        samples = model.get_samples(key, samples, actions, 1)
        cost = jnp.mean(jax.vmap(pendulum_cost)(jnp.hstack((samples, actions))))
        var = jnp.var(jax.vmap(pendulum_cost)(jnp.hstack((samples, actions))))
        return (params, key, samples, total_cost + cost, total_var + var), cost

    total_cost = 0
    total_var = 0
    (params, key, samples, total_cost, total_var), result = jax.lax.scan(
        one_rollout_step,
        (p_params, key, init_samples, total_cost, total_var),
        timesteps,
    )
    return (total_cost, jnp.sqrt(total_var))


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
exploration_policy = random_policy
for trial in range(num_trials):
    # Sample from Enviornment
    key, subkey = jr.split(key)
    these_states, these_actions = sample_from_environment(
        env, timesteps, num_trials, exploration_policy, subkey
    )
    states.extend(these_states)
    actions.extend(these_actions)

    states_array = jnp.array(states)
    actions_array = jnp.array(actions)

    # Initialize and Fit the GP Model
    model = IMGPR(
        states=states_array,
        actions=actions_array,
        kernel_funcs=ExpSquared,
        params=model_params,
    )
    if trial == 0:
        num_policy_opt_steps = 2000
        model_params = model_params
    else:
        num_policy_opt_steps = 4000
        model_params = model.params  # Start from previous model (?)

    start_time = time.perf_counter()
    model = optimize_imgpr(
        model,
        states=states_array,
        actions=actions_array,
    )
    end_time = time.perf_counter()
    print(f"Model Optimization Time = {end_time-start_time}")

    # Set up Optimizer for Policy Optimization
    factor = min(1.0, max(0.0, (trial - 5) / 20))
    if factor == 0.0:
        init_state = [1e-6, 1e-6]  # Cannot use zero because of the reset
    else:
        key, subkey = jr.split(key)
        init_state = [float(factor * jnp.pi * jr.uniform(subkey))]
        key, subkey = jr.split(key)
        init_state.extend([float(factor * epsilon * jr.uniform(subkey))])

    cosine_decay_scheduler = optax.cosine_decay_schedule(
        0.0001, decay_steps=num_policy_opt_steps, alpha=0.95
    )

    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    # Initialize some particles to run cost
    sample_train, _ = env.reset(
        options={"x_init": init_state[0], "y_init": init_state[1]}
    )
    action_train = control_policy(sample_train, 0.0)
    key, subkey = jr.split(key)

    initial_train_particles = model.get_samples(
        subkey, jnp.array([sample_train]), jnp.array([action_train]), num_particles
    )

    # Compute cost
    initial_mu = []
    initial_std = []
    for i in range(10):
        key, subkey = jr.split(key)
        mu, sig = policy_rollout_with_std(
            control_policy, initial_train_particles, model, timesteps, subkey
        )
        initial_mu.append(mu)
        initial_std.append(jnp.square(sig))
    initial_mu = jnp.mean(jnp.array(initial_mu))
    initial_std = jnp.sqrt(jnp.mean(jnp.array(initial_std)))
    print(f"Current cost and std = {initial_mu.item(),initial_std.item()}")

    # Optimize Policy using fitted GP model
    start_time = time.perf_counter()
    control_policy = fit_controller(
        policy=control_policy,
        env=env,
        num_particles=num_particles,
        initial_state=init_state,
        timesteps=timesteps,
        gp_model=model,
        obj_func=pendulum_cost,
        optim=optimizer,
        max_steps=num_policy_opt_steps,
        key=key,
    )
    end_time = time.perf_counter()
    print(f"Policy Optimization Time = {end_time-start_time}")

    # Explore with optimized control policy going forward
    exploration_policy = control_policy

# When done, print out an example
env_test = gym.make("Pendulum-v1", render_mode="rgb_array")
# Now try this policy on the real system
state, _ = env_test.reset()
key, subkey = jr.split(key)
u = control_policy(state, 0.0)
# Randomly sample some points
# img = plt.imshow(env_test.render()) # only call this once
for timestep in range(
    200
):  # np.linspace(0,5000*T_exploration,int(T_exploration/sim_timestep)+1):
    z = env_test.step(np.array(u))
    state = z[0]
    r = z[1]
    key, subkey = jr.split(key)
    u = control_policy(state, timestep)
    print(timestep, state, u, r)
    # img.set_data(env_test.render()) # just update the data
    # display.display(plt.gcf())
    # display.clear_output(wait=True)
