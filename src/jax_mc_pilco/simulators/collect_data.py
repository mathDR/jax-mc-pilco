from collections.abc import Callable
from typing import Any

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferState
from flashbax.vault import Vault
from numpy.typing import NDArray

ReplayBuffer = tuple[TrajectoryBuffer, TrajectoryBufferState]


def create_trajectory_buffer(
    num_trajs: int,
    max_traj_len: int,
    batch_size: int,
    sample_seq_len: int,
) -> TrajectoryBuffer:
    return fbx.make_trajectory_buffer(
        max_length_time_axis=max_traj_len,
        min_length_time_axis=sample_seq_len,
        sample_batch_size=batch_size,
        add_batch_size=num_trajs,
        sample_sequence_length=sample_seq_len,
        period=1,
    )


def load_buffer_from_vault(
    vault_name: str,
    vault_uid: str,
    batch_size: int,
    sample_seq_len: int,
    rel_dir: str = "vaults",
    percentiles: tuple[int, int] = (0, 100),
) -> tuple[TrajectoryBuffer, TrajectoryBufferState]:
    # Load the saved buffer state (or a temporal slice via percentiles, e.g., (0, 90) for train, (90, 100) for val)
    vault = Vault(
        vault_name=vault_name,
        rel_dir=rel_dir,
        vault_uid=vault_uid,
    )
    buffer_state = vault.read(percentiles=percentiles)

    # Reconstruct the matching Flashbax buffer functions using the loaded dimensions
    B, T = buffer_state.experience["obs"].shape[:2]
    buffer = create_trajectory_buffer(
        num_trajs=B,
        max_traj_len=T,
        batch_size=batch_size,
        sample_seq_len=sample_seq_len,
    )
    return buffer, buffer_state


def _extract_obs(
    env: Any,
    timestep: Any,
    use_pixels: bool,
    img_size: int = 64,
    camera_id: int = 0,
) -> NDArray:
    """Extract either channel-first uint8 pixels (C, H, W) or a flat state vector."""
    if use_pixels:
        # dm_control renders (H, W, C) uint8; transpose to (C, H, W) for Equinox Conv2d
        rgb = env.physics.render(height=img_size, width=img_size, camera_id=camera_id)
        ret_array: np.ndarray = np.moveaxis(rgb, -1, 0).astype(np.uint8)
        return ret_array

    # Flatten OrderedDict observations (e.g., orientations, velocity) into 1D float32
    obs_parts = [np.ravel(np.asarray(v, dtype=np.float32)) for v in timestep.observation.values()]
    return np.concatenate(obs_parts, axis=0)


def collect_data(
    env: Any,
    num_trajectories: int,
    traj_length: int,
    key: jax.Array,
    replay_buffer: ReplayBuffer | None = None,
    *,
    policy_fn: Callable[[jax.Array, NDArray[Any]], NDArray[Any]] | None = None,
    use_pixels: bool = False,
    img_size: int = 64,
    camera_id: int = 0,
    action_repeat: int = 1,
    sample_batch_size: int = 64,
    sample_seq_len: int = 26,
    max_buffer_time_axis: int | None = None,
) -> ReplayBuffer:
    """Collect `num_trajectories` of length `traj_length` from a `dm_control` env
    and add them to a Flashbax trajectory buffer.

    Args:
        env: A `dm_control.suite` environment instance.
        num_trajectories: Number of parallel/batched trajectories (B) to collect.
        traj_length: Number of steps (T) per trajectory.
        key: JAX PRNG key for action sampling.
        replay_buffer: Optional `(buffer, buffer_state)` tuple. If None, creates a new one.
        policy_fn: Optional callable `(key, obs) -> action` returning an action within
            `[action_spec.minimum, action_spec.maximum]`. Defaults to uniform random actions.
        use_pixels: If True, renders `(3, img_size, img_size)` uint8 frames instead of state vectors.
        img_size: Height/width for pixel rendering when `use_pixels=True`.
        camera_id: Camera index passed to `env.physics.render`.
        action_repeat: Number of physics steps per selected action.
        sample_batch_size: Batch size configured on the Flashbax buffer (used if creating new).
        sample_seq_len: Sub-sequence length configured on the Flashbax buffer (used if creating new).
        max_buffer_time_axis: Capacity along the time axis. Defaults to `traj_length * 10`.

    Returns:
        `(buffer, buffer_state)` with the newly collected trajectories added.
    """
    action_spec = env.action_spec()
    act_low = np.asarray(action_spec.minimum, dtype=np.float32)
    act_high = np.asarray(action_spec.maximum, dtype=np.float32)

    # Probe shapes from an initial reset
    init_ts = env.reset()
    sample_obs = _extract_obs(env, init_ts, use_pixels, img_size, camera_id)

    obs_buf = np.empty((num_trajectories, traj_length, *sample_obs.shape), dtype=sample_obs.dtype)
    act_buf = np.empty((num_trajectories, traj_length, *action_spec.shape), dtype=np.float32)
    rew_buf = np.empty((num_trajectories, traj_length), dtype=np.float32)
    done_buf = np.empty((num_trajectories, traj_length), dtype=bool)

    for b in range(num_trajectories):
        ts = env.reset()
        obs = _extract_obs(env, ts, use_pixels, img_size, camera_id)

        for t in range(traj_length):
            key, act_key = jax.random.split(key)

            if policy_fn is None:
                action = np.asarray(
                    jax.random.uniform(act_key, shape=action_spec.shape, minval=act_low, maxval=act_high),
                    dtype=np.float32,
                )
            else:
                action = np.clip(np.asarray(policy_fn(act_key, obs), dtype=np.float32), act_low, act_high)

            # Step environment with optional action repeat
            total_reward = 0.0
            done = False
            for _ in range(action_repeat):
                ts = env.step(action)
                total_reward += float(ts.reward or 0.0)
                if ts.last():
                    done = True
                    break

            obs_buf[b, t] = obs
            act_buf[b, t] = action
            rew_buf[b, t] = total_reward
            done_buf[b, t] = done

            if done:
                ts = env.reset()
            obs = _extract_obs(env, ts, use_pixels, img_size, camera_id)

    # Create the Flashbax trajectory buffer if not provided
    if replay_buffer is None:
        max_time = max_buffer_time_axis if max_buffer_time_axis is not None else max(traj_length * 10, sample_seq_len)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=max_time,
            min_length_time_axis=sample_seq_len,
            sample_batch_size=sample_batch_size,
            add_batch_size=num_trajectories,
            sample_sequence_length=sample_seq_len,
            period=1,
        )
        dummy_timestep = {
            "obs": jnp.zeros(sample_obs.shape, dtype=jnp.dtype(sample_obs.dtype)),
            "action": jnp.zeros(action_spec.shape, dtype=jnp.float32),
            "reward": jnp.zeros((), dtype=jnp.float32),
            "done": jnp.zeros((), dtype=jnp.bool_),
        }
        buffer_state = buffer.init(dummy_timestep)
    else:
        buffer, buffer_state = replay_buffer

    # Add the (B, T, ...) batch of trajectories into the Flashbax buffer
    trajectory_batch = {
        "obs": jnp.asarray(obs_buf),
        "action": jnp.asarray(act_buf),
        "reward": jnp.asarray(rew_buf),
        "done": jnp.asarray(done_buf),
    }
    buffer_state = buffer.add(buffer_state, trajectory_batch)

    return buffer, buffer_state
