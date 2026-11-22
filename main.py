import dm_env
import equinox as eqx
import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels


class Actor(eqx.Module):
    """Simple Equinox CNN policy to generate continuous actions from pixels."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, action_dim: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        # Expects (C, H, W) format for Equinox Conv2d
        self.conv1 = eqx.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, key=k1)
        self.conv2 = eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, key=k2)
        # For 64x64 input -> conv1 (31x31) -> conv2 (14x14) -> 32 * 14 * 14 = 6272
        self.fc = eqx.nn.Linear(in_features=6272, out_features=action_dim, key=k3)

    def __call__(self, obs: jax.Array) -> jax.Array:
        # Normalize uint8 pixels [0, 255] -> float32 [0, 1] and transpose (H, W, C) -> (C, H, W)
        x = jnp.transpose(obs.astype(jnp.float32) / 255.0, (2, 0, 1))
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jnp.ravel(x)
        return jnp.tanh(self.fc(x))  # Continuous actions in [-1, 1]


def make_env(img_height: int = 64, img_width: int = 64) -> dm_env.Environment:
    """Creates the dm_control acrobot environment wrapped to output pixel observations."""
    env = suite.load(domain_name="acrobot", task_name="swingup")
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={"height": img_height, "width": img_width, "camera_id": 0},
    )
    return env


def collect_trajectories(
    env: dm_env.Environment,
    actor: Actor,
    num_samples: int,
    num_steps: int,
    img_shape: tuple[int, int, int],
    action_dim: int,
) -> dict[str, jax.Array]:
    """Rolls out `num_samples` trajectories of length `num_steps` in dm_control."""
    obs_buf = np.zeros((num_samples, num_steps, *img_shape), dtype=np.uint8)
    act_buf = np.zeros((num_samples, num_steps, action_dim), dtype=np.float32)
    rew_buf = np.zeros((num_samples, num_steps), dtype=np.float32)
    done_buf = np.zeros((num_samples, num_steps), dtype=np.bool_)

    # JIT-compile the Equinox forward pass for fast step inference
    @eqx.filter_jit
    def select_action(model: Actor, obs: jax.Array) -> jax.Array:
        return model(obs)


    for i in range(num_samples):
        time_step = env.reset()
        for t in range(num_steps):
            pixels_obs = time_step.observation["pixels"]

            # Get action from Equinox actor
            action = np.asarray(select_action(actor, jnp.asarray(pixels_obs)))

            # Step environment
            next_time_step = env.step(action)

            obs_buf[i, t] = pixels_obs
            act_buf[i, t] = action
            rew_buf[i, t] = next_time_step.reward or 0.0
            done_buf[i, t] = next_time_step.last()

            time_step = next_time_step
            if time_step.last():
                time_step = env.reset()

    # Convert collected NumPy trajectories to JAX arrays on device
    return {
        "obs": jnp.asarray(obs_buf),
        "action": jnp.asarray(act_buf),
        "reward": jnp.asarray(rew_buf),
        "done": jnp.asarray(done_buf),
    }


def main() -> None:
    # Hyperparameters
    num_samples = 16       # Number of trajectories to collect
    num_steps = 100        # Length of each trajectory
    img_size = (64, 64)    # Rendered image resolution
    sample_seq_len = 20    # Sequence length when sampling from buffer
    sample_batch_size = 8  # Batch size when sampling from buffer

    key = jax.random.PRNGKey(42)
    key, actor_key, sample_key = jax.random.split(key, 3)

    # 1. Initialize Environment & Equinox Policy
    env = make_env(img_height=img_size[0], img_width=img_size[1])
    action_spec = env.action_spec()
    action_dim = action_spec.shape[0]
    img_shape = (*img_size, 3)

    actor = Actor(action_dim=action_dim, key=actor_key)

    # 2. Initialize Flashbax Trajectory Buffer
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=num_steps * 2,
        min_length_time_axis=sample_seq_len,
        sample_batch_size=sample_batch_size,
        add_batch_size=num_samples,
        sample_sequence_length=sample_seq_len,
        period=1,
    )

    # Create a dummy single-timestep structure to initialize buffer state
    dummy_timestep = {
        "obs": jnp.zeros(img_shape, dtype=jnp.uint8),
        "action": jnp.zeros((action_dim,), dtype=jnp.float32),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "done": jnp.array(False, dtype=jnp.bool_),
    }
    buffer_state = buffer.init(dummy_timestep)

    # 3. Collect pixel trajectories from dm_control acrobot
    print(f"Collecting {num_samples} trajectories of {num_steps} steps...")
    trajectories = collect_trajectories(
        env=env,
        actor=actor,
        num_samples=num_samples,
        num_steps=num_steps,
        img_shape=img_shape,
        action_dim=action_dim,
    )

    # 4. Add trajectories to Flashbax buffer (JIT-compiled)
    add_fn = jax.jit(buffer.add)
    buffer_state = add_fn(buffer_state, trajectories)
    print("Trajectories added to Flashbax buffer successfully.")

    # 5. Verify by sampling a batch of sub-trajectories
    sample_fn = jax.jit(buffer.sample)
    batch = sample_fn(buffer_state, sample_key)

    print("\nSampled Batch Shapes (batch_size, sequence_length, ...):")
    for k, v in batch.experience.items():
        print(f"  {k}: {v.shape} ({v.dtype})")
    
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Grab the first sampled trajectory: shape (sequence_length, 64, 64, 3)
    seq = np.asarray(batch.experience["obs"][0])
    num_frames = min(8, seq.shape[0])  # Plot up to 8 frames

    fig, axes = plt.subplots(1, num_frames, figsize=(2 * num_frames, 2.5))
    for t in range(num_frames):
        axes[t].imshow(seq[t])
        axes[t].axis("off")
        axes[t].set_title(f"t={t}")

    plt.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":
    main()


