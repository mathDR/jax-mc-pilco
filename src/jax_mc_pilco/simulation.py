from __future__ import annotations

import typing as tp

import gpjax as gpx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
from controllers import RandomController
from controllers import Sum_of_Sinusoids
from flax import nnx
from gpjax.typing import Array
from gpjax.typing import KeyArray
from gpjax.typing import ScalarFloat
from jax import Array
from jax import config
from jax.tree_util import Partial
from jaxtyping import ArrayLike
from jaxtyping import install_import_hook
from model_learning.mgpr import DynamicalModel

Model = tp.TypeVar('Model', bound=nnx.Module)

config.update('jax_enable_x64', True)


# Function to sample from a single mean and covariance
def sample_mvnormal(key, mean, cov, num_samples):
    return jr.multivariate_normal(key, mean, cov, (num_samples,))


def one_rollout_step(carry, t):
    policy, predict_all_outputs, key, samples, total_cost = carry
    key, *subkeys = jr.split(key, num_particles + 1)
    u = jax.vmap(policy)(
        samples, jnp.tile(
            t, num_particles,
        ), jnp.array(subkeys),
    )
    this_state = jnp.hstack((samples, u))
    predictive_moments = predict_all_outputs(this_state)
    key, subkey = jr.split(key)
    samples = jnp.squeeze(
        vectorized_sample(
            key,
            predictive_moments[:, :, 0],
            jax.vmap(jnp.diag)(predictive_moments[:, :, 1]),
            1,
        ),
    )
    cost = jnp.sum(jax.vmap(cart_pole_cost)(samples))
    return (policy, predict_all_outputs, key, samples, total_cost + cost), cost


def rollout(
    policy,  #: Controller,
    init_samples: ArrayLike,
    model,  # Model
    timesteps: ArrayLike,
    key: KeyArray = jr.PRNGKey(42),
) -> ScalarFloat:
    action = Partial(policy)
    pao = Partial(model.predict_all_outputs)
    (action, pao, key, samples, total_cost), result = jax.lax.scan(
        one_rollout_step, (action, pao, key, init_samples, 0), timesteps,
    )
    return total_cost / len(timesteps)


# Vectorize the sampling function
vectorized_sample = jax.vmap(sample_mvnormal, in_axes=(None, 0, 0, None))


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
        - jnp.square((x - target_x) / lengthscales[1]),
    )


env = gym.make('InvertedPendulum-v5')

action_dim = env.action_space.shape[0]
initial_state_exploration, _ = env.reset()
state_dim = initial_state_exploration.shape[0]

# Initialize a random controller

exploration_policy = RandomController(state_dim, action_dim, True, 3.0)

initial_explore_timesteps = 10

X = []
Y = []
ep_return_full = 0
ep_return_sampled = 0
key = jr.key(42)
x = initial_state_exploration.copy()
for timestep in range(initial_explore_timesteps):
    key, subkey = jr.split(key)
    u = exploration_policy(x, timestep, subkey)
    # print(u)
    z = env.step(np.array(u))
    # print(z)
    # x_new, r, done, _, __ = env.step(np.array(u))
    x_new = z[0]
    r = z[1]
    X.append(jnp.hstack((x, u)))
    Y.append(x_new - x)
    ep_return_sampled += r
    x = x_new
X = jnp.array(X)
Y = jnp.array(Y)
D = gpx.Dataset(X=X, y=Y)

model = DynamicalModel(data=D)
model.optimize()

# Generate an initial state
x0, _ = env.reset()
key, subkey = jr.split(key)
# Generate an initial action
u0 = exploration_policy(x, timestep, subkey)
initial_state = jnp.hstack((x, u)).reshape(1, -1)
# Compute the moments from the trained GP transition function
predictive_moments = model.predict_all_outputs(initial_state)

num_particles = 100
init_samples = jnp.squeeze(
    vectorized_sample(
        key,
        predictive_moments[:, :, 0],
        jax.vmap(jnp.diag)(predictive_moments[:, :, 1]),
        num_particles,
    ),
)

time_horizon = 50
timesteps = jnp.arange(timestep + 1, timestep + time_horizon)

policy = Sum_of_Sinusoids(
    state_dim, action_dim, 6, 0.0, 2.0 * np.pi, -1.0, 1.0, True, 3.0,
)
objective_fun = Partial(rollout, model=model, timesteps=timesteps)

optimizer = ox.adam(learning_rate=1e-2)
print(policy.amplitudes, policy.omega, policy.phases)
breakpoint()
policy, history = gpx.fit(
    model=policy,
    objective=objective_fun,
    train_data=init_samples,
    optim=optimizer,
    params_bijection=None,
    safe=False,
)
print(policy.amplitudes, policy.omega, policy.phases)
