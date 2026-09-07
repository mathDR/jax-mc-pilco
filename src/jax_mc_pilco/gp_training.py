# from flowjax.flows import coupling_flow
import gpjax as gpx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow

from jax_mc_pilco.flow_fit_to_gp import fit_to_gp

# from flowjax.distributions import MultivariateNormal

jax.config.update("jax_enable_x64", True)

env = gym.make("InvertedPendulum-v5")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

x, _ = env.reset()
replay_states = [x]

u = env.action_space.sample()
replay_actions = [u]

for _ in range(100):
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

state_space_flow: Transformed = coupling_flow(
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

key, subkey1, subkey2 = jax.random.split(key, 3)

coreg1 = gpx.parameters.CoregionalizationMatrix(num_outputs=state_dim, rank=state_dim, key=subkey1)
coreg2 = gpx.parameters.CoregionalizationMatrix(num_outputs=state_dim, rank=state_dim, key=subkey2)

lcm_kernel = gpx.kernels.LCMKernel(
    kernels=[
        gpx.kernels.RBF(lengthscale=jnp.ones(cond_dim)),  # type: ignore
        gpx.kernels.Matern32(
            lengthscale=jnp.ones(cond_dim),
        ),  # type: ignore
    ],
    coregionalization_matrices=[coreg1, coreg2],
)

meanf_lcm = gpx.mean_functions.Zero()
prior_lcm = gpx.gps.Prior(mean_function=meanf_lcm, kernel=lcm_kernel)
likelihood_lcm = gpx.likelihoods.MultiOutputGaussian(
    num_datapoints=context.shape[0], num_outputs=state_dim, obs_stddev=1.0
)
gp_posterior = prior_lcm * likelihood_lcm

train_dataset = gpx.Dataset(X=context, y=delta_s)
print(f"Initial negative MLL: {-gpx.objectives.conjugate_mll(gp_posterior, train_dataset):.3f}")

opt_posterior, history = gpx.fit_scipy(
    model=gp_posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=train_dataset,
)
print(f"Optimized negative MLL: {-gpx.objectives.conjugate_mll(opt_posterior, train_dataset):.3f}")

breakpoint()

fit_flow, losses = fit_to_gp(
    key=subkey,
    dist=state_space_flow,
    gp_model=gp_posterior,
    train_data=train_dataset,
    learning_rate=5e-3,
    max_patience=10,
    max_epochs=70,
)

breakpoint()


# Now we should initialize an action flow then optimize that w.r.t. the env reward using the state space flow model.
