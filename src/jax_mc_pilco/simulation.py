import jax
import equinox as eqx
from jax import Array, config
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import jax.random as jr
from jaxtyping import ArrayLike, install_import_hook, Array, Float, Int, PyTree

config.update("jax_enable_x64", True)

key = jr.key(123)

import gymnasium as gym
from controllers import RandomController, LinearPolicy
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


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def rollout(
    policy: eqx.Module,
    init_samples: ArrayLike,
    model: eqx.Module,
    timesteps: ArrayLike,
    key: ArrayLike = jr.key(123),
) -> float:
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    def one_rollout_step(carry, t):
        policy_params, key, samples, total_cost = carry
        policy = eqx.combine(policy_params, policy_static)
        actions = jax.vmap(policy)(samples, jnp.tile(t, num_particles))

        key, subkey = jr.split(key)
        samples = model.get_samples(key, samples, actions, 1)
        cost = jnp.sum(jax.vmap(cart_pole_cost)(jnp.hstack((samples, actions))))
        return (policy_params, key, samples, total_cost + cost), cost

    total_cost = 0
    (policy_params, key, samples, total_cost), result = jax.lax.scan(
        one_rollout_step, (policy_params, key, init_samples, total_cost), timesteps
    )
    return total_cost / len(timesteps)


def fit_controller(  # noqa: PLR0913
    *,
    policy: eqx.Module,
    samples: ArrayLike,
    timesteps: ArrayLike,
    gp_model: eqx.Module,
    optim: ox.GradientTransformation,
    key: ArrayLike = jr.PRNGKey(42),
    num_iters: int = 100,
    unroll: int = 5,
) -> Tuple[eqx.Module, Array]:
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    @eqx.filter_jit
    def make_step(
        policy: eqx.Module,
        opt_state: PyTree,
    ) -> Tuple[eqx.Module, PyTree, float]:
        loss_value, loss_gradient = eqx.filter_value_and_grad(rollout)(
            policy, samples, gp_model, timesteps
        )
        updates, opt_state = optim.update(
            loss_gradient, opt_state, eqx.filter(policy, eqx.is_array)
        )
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, loss_value

    # Optimisation loop.
    for step in range(num_iters):
        policy, opt_state, train_loss = make_step(policy, opt_state)
        if (step % 100) == 0 or (step == num_iters - 1):
            print(f"{step=}, train_loss={train_loss.item()}, ")

    return policy


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
policy = LinearPolicy(state_dim, action_dim, True, umax)

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

# Do a rollout
x, _ = env.reset()
key, subkey = jr.split(key)
# Generate an initial action
u = exploration_policy(x, timestep, subkey)

# initialize some particles
initial_particles = model.get_samples(
    key, jnp.array([x]), jnp.array([u]), num_particles
)

control_horizon = int(T_control / T_sampling)
optimizer = ox.adam(
    learning_rate=ox.linear_schedule(
        init_value=1e-2, end_value=1e-6, transition_steps=100
    )
)
policy = fit_controller(
    policy=policy,
    samples=initial_particles,
    timesteps=jnp.arange(control_horizon),
    gp_model=model,
    optim=optimizer,
    num_iters=1000,
)

# Now try this policy on the real system
x, _ = env_test.reset()
key, subkey = jr.split(key)
u = policy(x, timestep, subkey)
# Randomly sample some points
states.append(x)
actions.append(u)
img = plt.imshow(env_test.render())  # only call this once
for timestep in np.linspace(0, T_exploration, int(T_exploration / sim_timestep) + 1):
    z = env_test.step(np.array(u))
    x = z[0]
    r = z[1]
    key, subkey = jr.split(key)
    u = policy(x, timestep, subkey)
    states.append(x)
    actions.append(u)
    img.set_data(env_test.render())  # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
