"""Closed-loop evaluation of trained LatentFlowWorldModel + ActionFlow in dm_control."""

from pathlib import Path
from typing import Any

import equinox as eqx
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from dm_control import suite
from numpy.typing import NDArray

from jax_mc_pilco.model_learning.flow_model import LatentFlowWorldModel
from jax_mc_pilco.policy_learning.action_flows import ActionFlow


def render_frame(env: Any, img_size: int = 64, camera_id: int = 0) -> NDArray[np.uint8]:
    """Render an RGB frame of shape (H, W, 3) uint8 from dm_control."""
    frame: NDArray[np.uint8] = env.physics.render(height=img_size, width=img_size, camera_id=camera_id)
    return frame


def preprocess_obs(frame_hwc: NDArray[np.uint8]) -> jax.Array:
    """Convert (H, W, 3) uint8 frame to normalized (3, H, W) float32 matching training."""
    obs_chw = np.transpose(frame_hwc, (2, 0, 1))
    return jnp.asarray(obs_chw, dtype=jnp.float32) / 255.0 - 0.5


@eqx.filter_jit
def select_action_and_update_state(
    world_model: LatentFlowWorldModel,
    action_flow: ActionFlow,
    obs_chw: jax.Array,
    prev_hidden: jax.Array,
    key: jax.Array,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Infer z_t from current frame, sample action a_t, and advance recurrent hidden state."""
    post_key, act_key = jax.random.split(key)

    # 1. Encode frame and infer posterior latent z_t given (h_t, e_t)
    embed_t = world_model.encoder(obs_chw)
    z_t, _ = world_model.posterior_sample_and_log_prob(post_key, embed_t, prev_hidden)

    # 2. Sample (or approximate mode via multi-sample average) action a_t ~ pi(a_t | z_t, h_t)
    if deterministic:
        sample_keys = jax.random.split(act_key, 32)
        actions, _ = jax.vmap(action_flow.sample_and_log_prob, in_axes=(0, None, None))(sample_keys, z_t, prev_hidden)
        action_t = jnp.mean(actions, axis=0)
    else:
        action_t, _ = action_flow.sample_and_log_prob(act_key, z_t, prev_hidden)

    # 3. Advance recurrent hidden state h_{t+1} = GRU([z_t, a_t], h_t)
    next_hidden = world_model.step_hidden(z_t, action_t, prev_hidden)
    return action_t, z_t, next_hidden


def evaluate_closed_loop(
    world_model: LatentFlowWorldModel,
    action_flow: ActionFlow,
    domain_name: str = "acrobot",
    task_name: str = "swingup",
    num_episodes: int = 5,
    max_steps: int = 1000,
    action_repeat: int = 1,
    policy_img_size: int = 64,
    video_img_size: int = 256,
    camera_id: int = 0,
    fps: int = 30,
    deterministic: bool = False,
    seed: int = 0,
    output_dir: str | Path = "videos/acrobot_eval",
) -> list[float]:
    """Run closed-loop episodes in dm_control and save each rollout as an MP4 video."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed})
    action_spec = env.action_spec()
    act_low = np.asarray(action_spec.minimum, dtype=np.float32)
    act_high = np.asarray(action_spec.maximum, dtype=np.float32)

    key = jax.random.key(seed)
    episode_returns: list[float] = []

    for ep in range(num_episodes):
        ts = env.reset()
        hidden = jnp.zeros((world_model.memory.hidden_size,), dtype=jnp.float32)
        ep_return = 0.0
        frames: list[NDArray[np.uint8]] = []

        for _ in range(max_steps):
            # Render policy input (64x64) and optional high-res video frame (256x256)
            policy_frame = render_frame(env, img_size=policy_img_size, camera_id=camera_id)
            video_frame = (
                policy_frame
                if video_img_size == policy_img_size
                else render_frame(env, img_size=video_img_size, camera_id=camera_id)
            )
            frames.append(video_frame)

            obs_chw = preprocess_obs(policy_frame)
            key, step_key = jax.random.split(key)

            action_jax, _, hidden = select_action_and_update_state(
                world_model=world_model,
                action_flow=action_flow,
                obs_chw=obs_chw,
                prev_hidden=hidden,
                key=step_key,
                deterministic=deterministic,
            )
            action = np.clip(np.asarray(action_jax, dtype=np.float32), act_low, act_high)

            done = False
            for _ in range(action_repeat):
                ts = env.step(action)
                ep_return += float(ts.reward or 0.0)
                if ts.last():
                    done = True
                    break

            if done:
                frames.append(render_frame(env, img_size=video_img_size, camera_id=camera_id))
                break

        episode_returns.append(ep_return)
        video_file = out_path / f"{domain_name}_{task_name}_ep{ep:02d}_ret{ep_return:.1f}.mp4"
        imageio.mimsave(video_file, frames, fps=fps, macro_block_size=1)
        print(f"Episode {ep + 1}/{num_episodes} | Return: {ep_return:.2f} | Saved: {video_file}")

    mean_ret = float(np.mean(episode_returns))
    std_ret = float(np.std(episode_returns))
    print(f"\nSummary over {num_episodes} episodes: {mean_ret:.2f} +/- {std_ret:.2f}")
    return episode_returns


if __name__ == "__main__":
    # Initialize template models matching your main.py hyperparameters, then load saved weights:
    env = suite.load(domain_name="acrobot", task_name="swingup")
    key = jax.random.key(seed=4)

    key, wm_key, act_key, critic_key = jax.random.split(key, 4)

    action_dim = env.action_spec().shape[0]
    latent_dim = 32
    hidden_dim = 128

    world_model_baseline = LatentFlowWorldModel(
        key=wm_key,
        action_dim=action_dim,
        in_channels=3,
        img_size=64,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    dynamics = eqx.tree_deserialise_leaves("world_model.eqx", world_model_baseline)
    #
    action_flow_baseline = ActionFlow(
        key=act_key,
        state_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    action_flow = eqx.tree_deserialise_leaves("imagination_policy.eqx", action_flow_baseline)
    #
    evaluate_closed_loop(
        world_model=dynamics,
        action_flow=action_flow,
        domain_name="acrobot",
        task_name="swingup",
        num_episodes=5,
        action_repeat=1,  # Match the action_repeat used when collecting training data!
        deterministic=False,
    )
