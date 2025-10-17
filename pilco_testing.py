"""Run end to end PILCO optimization."""

# Enable Float64 for more stable matrix inversions.
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

config.update("jax_enable_x64", True)

import gpjax
from jax_mc_pilco.controllers import Controller, RandomController, SumOfGaussians
from jax_mc_pilco.rewards import pendulum_cost  # , cart_pole_cost
from jax_mc_pilco.model_learning.dynamical_models import (
    IMGPR,
    IMSVGPR,
    optimize_imgpr,
    optimize_imsvgpr,
)
from jax_mc_pilco.policy_learning.rollout import fit_controller
from jax_mc_pilco.simulators.simulation import sample_from_environment, remake_state

num_particles = 400
num_trials = 8
num_inducing_points = 200
T_sampling = 0.05
T_exploration = 0.35
T_control = 3.0
sim_timestep = 0.01
num_basis = 200
umax = 2.0
num_inducing_points = 250

position_memory = 2
control_memory = 1

env = gym.make("Pendulum-v1")

action_dim = env.action_space.shape[0]
x, _ = env.reset()
state_dim = x.shape[0]

timesteps = np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1)
key = jr.key(42)

random_policy = RandomController(state_dim, action_dim, to_squash=True, max_action=umax)
key, subkey = jr.split(key)
control_policy = SumOfGaussians(
    state_dim * (1 + position_memory),
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
    init_states: ArrayLike,
    init_actions: ArrayLike,
    model: eqx.Module,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(42),
) -> Tuple[Float, Float]:
    p_params, p_static = eqx.partition(policy, eqx.is_array)

    def update_actions(action: ArrayLike, actions: ArrayLike):
        """Append action to the front of actions and pop off last actions value."""
        return jnp.concatenate([action[:, jnp.newaxis, :], actions[:, :-1, :]], axis=1)

    def update_states(state: ArrayLike, states: ArrayLike):
        """Append state to the front of states and pop off last states value."""
        theta = jnp.atan2(state[:, 1], state[:, 0])
        new_state = jnp.array(
            [jnp.cos(theta), jnp.sin(theta), jnp.clip(state[:, 2], min=-8, max=8)]
        ).T
        return jnp.concatenate(
            [new_state[:, jnp.newaxis, :], states[:, :-1, :]], axis=1
        )

    def one_rollout_step(
        carry: Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Float, Float],
        timestep: Float,
    ) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Float, Float], Float]:
        params, key, states, actions, total_cost, total_var = carry
        policy = eqx.combine(params, p_static)
        # Compute the action from the most recent state
        policy_input = jax.vmap(model.data_to_policy_input)(states, actions)
        action = jax.vmap(policy, in_axes=(0, None))(policy_input, timestep)
        actions = update_actions(action, actions)
        samples = model.get_samples(subkey, states, actions)
        states = update_states(samples, states)
        full_cost = jax.vmap(pendulum_cost)(jnp.hstack((samples, action)))
        cost = jnp.mean(full_cost)
        var = jnp.var(full_cost)
        return (params, key, states, actions, total_cost + cost, total_var + var), cost

    total_cost = 0
    total_var = 0
    (params, key, states, actions, total_cost, total_var), result = jax.lax.scan(
        one_rollout_step,
        (p_params, key, init_states, init_actions, total_cost, total_var),
        timesteps,
    )
    return (total_cost, jnp.sqrt(total_var))


# # Main Loop
states = []
actions = []
epsilon = 1e-4
exploration_policy = random_policy
for trial in range(num_trials):
    # Sample from Environment
    if trial == 0:
        these_states, these_actions = sample_from_environment(
            env, timesteps, num_trials, exploration_policy, model=None, key=subkey
        )
    else:
        key, subkey = jr.split(key)
        these_states, these_actions = sample_from_environment(
            env, timesteps, num_trials, exploration_policy, model=model, key=subkey
        )
    states.extend(these_states)
    actions.extend(these_actions)

    states_array = jnp.array(states, dtype=jnp.float64)
    actions_array = jnp.array(actions, dtype=jnp.float64)

    if trial == 0:
        num_policy_opt_steps = 2000
    else:
        num_policy_opt_steps = 4000

    # Initialize and Fit the GP Model
    # model = IMGPR(
    #     states=states_array,
    #     actions=actions_array,
    #     kernel_funcs=gpjax.kernels.RBF(),
    #     position_memory=position_memory,
    #     control_memory=control_memory,
    # )
    model = IMSVGPR(
        states=states_array,
        actions=actions_array,
        kernel_funcs=gpjax.kernels.RBF(),
        num_inducing_points=num_inducing_points,
        position_memory=position_memory,
        control_memory=control_memory,
    )

    start_time = time.perf_counter()
    # model = optimize_imgpr(
    #     model,
    #     states=states_array,
    #     actions=actions_array,
    # )
    optimize_model = optimize_imsvgpr(
        model,
        states=states_array,
        actions=actions_array,
    )
    end_time = time.perf_counter()
    print(f"Model Optimization Time = {end_time-start_time}")

    # Set up Optimizer for Policy Optimization
    factor = min(1.0, max(0.0, (trial - 5) / 20.0))
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
    key, subkey = jr.split(key)
    states_train, actions_train = sample_from_environment(
        env,
        timesteps[: max(model.position_memory, model.control_memory) + 1],
        1,
        control_policy,
        model=model,
        key=subkey,
    )
    initial_actions = jnp.tile(jnp.array(actions_train), (num_particles, 1, 1))
    initial_states = jnp.tile(
        jnp.array(states_train, dtype=jnp.float64), (num_particles, 1, 1)
    )
    breakpoint()

    # Compute cost
    initial_mu = []
    initial_std = []
    for i in range(10):
        key, subkey = jr.split(key)
        mu, sig = policy_rollout_with_std(
            control_policy, initial_states, initial_actions, model, timesteps, subkey
        )
        initial_mu.append(mu)
        initial_std.append(jnp.square(sig))
    initial_mu = jnp.mean(jnp.array(initial_mu))
    initial_std = jnp.sqrt(jnp.mean(jnp.array(initial_std)))
    print(f"Current cost and std = {initial_mu.item(),initial_std.item()}")
    breakpoint()

    # Optimize Policy using fitted GP model
    start_time = time.perf_counter()
    key, subkey = jr.split(key)
    control_policy = fit_controller(
        policy=control_policy,
        init_states=initial_states,
        init_actions=initial_actions,
        timesteps=timesteps,
        gp_model=model,
        obj_func=pendulum_cost,
        optim=optimizer,
        key=subkey,
        max_steps=num_policy_opt_steps,
    )
    end_time = time.perf_counter()
    print(f"Policy Optimization Time = {end_time-start_time}")

    # Explore with optimized control policy going forward
    exploration_policy = control_policy
env.close()

breakpoint()

# Now try this policy on the real system
env_test = gym.make("Pendulum-v1", render_mode="rgb_array_list")

x, _ = env_test.reset(seed=0)
state = remake_state(x)
states = [state]
u = env_test.action_space.sample()
actions = [u]

for i in range(max(model.position_memory, model.control_memory)):
    z = env_test.step(np.array(u))
    x = z[0]
    state = remake_state(x)
    states.append(state)
    u = env_test.action_space.sample()
    actions.append(u)

key, subkey = jr.split(key)
policy_input = model.data_to_policy_input(jnp.array(states), jnp.array(actions))
u = control_policy(policy_input, 0.0, subkey)

for i in range(200):
    z = env_test.step(np.array(u))
    x = z[0]
    state = remake_state(x)

    states.append(state)
    states.pop(-1)

    key, subkey = jr.split(key)
    policy_input = model.data_to_policy_input(jnp.array(states), jnp.array(actions))
    u = control_policy(policy_input, 0.0, subkey)
    actions.append(u)
    actions.pop(-1)
save_video(
    frames=env_test.render(),
    video_folder="videos",
    fps=env_test.metadata["render_fps"],
)
env_test.close()
