import dm_control.suite as suite
import jax
import jax.numpy as jnp
import numpy as np

# 1. Define your sequence dimensions
NUM_TRAJECTORIES = 5000
TRAJECTORY_LENGTH = 50  # 50 steps per episode
ADD_BATCH_SIZE = 1
# Define the shapes based on dm_control acrobot image rendering
# Standard camera render is usually (Height, Width, Channels) -> e.g., (64, 64, 3)
IMAGE_SHAPE = (64, 64, 3)
ACTION_SHAPE = (1,)  # Acrobot has 1 continuous torque action

env = suite.load(domain_name="acrobot", task_name="swingup")
action_spec = env.action_spec()

print(f"Starting data collection for {NUM_TRAJECTORIES} trajectories...")

broadcast_fn = lambda x: jnp.broadcast_to(x, (ADD_BATCH_SIZE, *x.shape))

# 5. Core Data Collection Loop
trajectories = []
for traj_idx in range(NUM_TRAJECTORIES):
    # Initialize lists to hold the steps for the current trajectory
    ep_obs, ep_act, ep_rew, ep_done = [], [], [], []

    time_step = env.reset()
    # Capture the initial observation frame
    current_img = env.physics.render(*IMAGE_SHAPE[:2], camera_id=0)

    for _ in range(TRAJECTORY_LENGTH):
        # Sample a random continuous action within acrobot bounds
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=ACTION_SHAPE)

        # Step the environment
        time_step = env.step(action)
        next_img = env.physics.render(*IMAGE_SHAPE[:2], camera_id=0)

        # Aligning Step t data: [o_t, a_t, r_t, d_t]
        ep_obs.append(current_img)
        ep_act.append(action)
        ep_rew.append(time_step.reward if time_step.reward is not None else 0.0)
        ep_done.append(time_step.last())

        current_img = next_img

        if time_step.last():
            # If env ends early, pad or break depending on strict length requirements
            break

    # Handle environment early-termination padding if needed, ensuring strict length 50
    while len(ep_obs) < TRAJECTORY_LENGTH:
        ep_obs.append(current_img)
        ep_act.append(np.zeros(ACTION_SHAPE))
        ep_rew.append(0.0)
        ep_done.append(True)

    # Convert Python lists of arrays directly into a single trajectory JAX PyTree
    trajectory_data = {
        "obs": jnp.array(ep_obs, dtype=jnp.uint8),
        "action": jnp.array(ep_act, dtype=jnp.float32),
        "reward": jnp.array(ep_rew, dtype=jnp.float32),
        "done": jnp.array(ep_done, dtype=jnp.bool_),
    }
    batch_traj = jax.tree.map(broadcast_fn, trajectory_data)

    # Push into the JAX device buffer using our JIT function
    trajectories.append(batch_traj)

    if (traj_idx + 1) % 5 == 0:
        print(f"Collected {traj_idx + 1}/{NUM_TRAJECTORIES} trajectories.")

print("Data collection complete! Buffer is fully populated.")
breakpoint()
trajectories = jax.device_put(trajectories, jax.devices("cpu")[0])
breakpoint()
stacked_trajectories = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args, axis=0), *trajectories)
del trajectories
# Save the full buffer state to disk
# This writes all 5000 collected trajectories to the folder
# print(f"Buffer successfully written to disk at: {vault._base_path}")
jnp.savez("acrobot_replay_buffer.npz", **stacked_trajectories)
