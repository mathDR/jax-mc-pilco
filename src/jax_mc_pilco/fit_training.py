"""Full end to end training."""

import dm_control.suite as suite
import jax
import jax.numpy as jnp

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.training.learning import world_training_loop

# 1. Define your sequence dimensions
NUM_TRAJECTORIES = 15
TRAJECTORY_LENGTH = 50  # 50 steps per episode

# Define the shapes based on dm_control acrobot image rendering
# Standard camera render is usually (Height, Width, Channels) -> e.g., (64, 64, 3)
IMAGE_SHAPE = (64, 64, 3)
ACTION_SHAPE = (1,)  # Acrobot has 1 continuous torque action
# from jax_mc_pilco.simulators.collect_data import collect_experience


key = jax.random.key(seed=4)

env = suite.load(domain_name="acrobot", task_name="swingup")
action_spec = env.action_spec()

with jnp.load("simulators/acrobot_replay_buffer.npz") as loaded_trajectories:  # type: ignore
    trajectories_dict = {key: jnp.array(loaded_trajectories[key]) for key in ["obs", "action"]}
# jax expects images to be C x H x W and we have H x W x C so we need to permute
trajectories_dict["obs"] = jnp.permute_dims(trajectories_dict["obs"], (0, 1, 4, 2, 3))
trajectories_dict = jax.device_put(trajectories_dict, jax.devices("cpu")[0])
# Now trajectories_dict is on cpu

key, subkey = jax.random.split(key)
baseline_dynamics = FlowDynamics(subkey, 1, action_spec.shape[0])

key, subkey = jax.random.split(key)
dynamics, d_losses = world_training_loop(subkey, trajectories_dict, baseline_dynamics, batch_size=32)

print("Dynamics model fit complete. Final validation loss:", d_losses["val"][-1])

breakpoint()
