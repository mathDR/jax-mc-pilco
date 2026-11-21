from pathlib import Path

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.training.learning import world_training_loop


def load_acrobot_trajectories(
    data_path: str | Path,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Loads Acrobot image trajectories from disk (.npz).
    Expected arrays:
      - images: (num_episodes, seq_len + 1, H, W, C) or (num_episodes, seq_len + 1, C, H, W)
      - actions: (num_episodes, seq_len, action_dim) or (num_episodes, seq_len)
    Returns:
      obs_t:      (num_episodes, seq_len, C, H, W) float32 in [0, 1]
      actions:    (num_episodes, seq_len, action_dim) float32
      obs_next:   (num_episodes, seq_len, C, H, W) float32 in [0, 1]
    """
    data = np.load(data_path)
    images = data["images"].astype(np.float32)
    if images.max() > 1.0:
        images /= 255.0

    # Ensure channel-first (N, T, C, H, W) for Equinox Conv layers
    if images.shape[-1] in (1, 3):
        images = np.transpose(images, (0, 1, 4, 2, 3))

    actions = data["actions"].astype(np.float32)
    if actions.ndim == 2:
        # One-hot encode discrete Acrobot actions {0, 1, 2} -> (N, T, 3)
        actions = np.eye(3, dtype=np.float32)[actions.astype(np.int32)]

    obs_t = jnp.asarray(images[:, :-1])
    obs_next = jnp.asarray(images[:, 1:])
    actions = jnp.asarray(actions[:, : obs_t.shape[1]])
    return obs_t, actions, obs_next


def encode_episodes(
    world_model: FlowDynamics,
    images: jax.Array,
) -> jax.Array:
    """Encodes image episodes (N, T, C, H, W) into flattened latent states (N, T, D)."""
    encode_step = lambda img: world_model.encoder(img).reshape(-1)
    encode_ep = jax.vmap(encode_step)
    return jax.vmap(encode_ep)(images)


def compute_intrinsic_motivation(
    world_model: FlowDynamics,
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes per-step and per-episode intrinsic motivation (novelty/surprise)
    using negative log-likelihood under FlowDynamics: -log p(s_{t+1} | s_t, a_t, h_t).
    """
    hidden_dim = world_model.memory.hidden_size

    def rollout_episode(
        ep_states: jax.Array,
        ep_actions: jax.Array,
        ep_next_states: jax.Array,
    ) -> jax.Array:
        def step_fn(hidden: jax.Array, transition: tuple[jax.Array, jax.Array, jax.Array]):
            s_t, a_t, s_next = transition
            next_hidden = world_model.step_hidden(s_t, a_t, hidden)
            # Lower log_prob => higher surprise => higher intrinsic reward
            lp = world_model.log_prob(s_t, s_next, next_hidden)
            intrinsic_reward = -lp
            return next_hidden, intrinsic_reward

        init_hidden = jnp.zeros((hidden_dim,), dtype=ep_states.dtype)
        _, step_rewards = jax.lax.scan(
            step_fn, init_hidden, (ep_states, ep_actions, ep_next_states)
        )
        return step_rewards

    step_intrinsic_rewards = jax.vmap(rollout_episode)(states, actions, next_states)
    episode_scores = jnp.mean(step_intrinsic_rewards, axis=-1)
    return step_intrinsic_rewards, episode_scores


def collect_curiosity_guided_acrobot_data(
    key: jax.Array,
    world_model: FlowDynamics,
    num_episodes: int = 16,
    seq_len: int = 120,
    num_candidates: int = 5,
    img_size: tuple[int, int] = (64, 64),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Collects new Acrobot image episodes by evaluating candidate actions with the
    world model and picking actions that maximize expected surprise (-log_prob).
    """
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    hidden_dim = world_model.memory.hidden_size

    all_obs_t, all_actions, all_obs_next = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        frame = env.render()
        frame = jax.image.resize(frame, (*img_size, 3), method="bilinear") / 255.0
        curr_img = jnp.transpose(frame, (2, 0, 1))  # (C, H, W)
        hidden = jnp.zeros((hidden_dim,))

        ep_obs, ep_act, ep_next = [], [], []

        for _ in range(seq_len):
            curr_latent = world_model.encoder(curr_img).reshape(-1)

            # Score all 3 discrete Acrobot actions by predicted novelty (-log_prob)
            best_action = 0
            best_novelty = -jnp.inf
            for act_idx in range(3):
                act_vec = jax.nn.one_hot(act_idx, 3)
                key, subkey = jax.random.split(key)
                pred_next, cand_hidden = world_model.predict_next_state_and_hidden(
                    subkey, curr_latent, act_vec, hidden
                )
                novelty = -world_model.log_prob(curr_latent, pred_next, cand_hidden)
                if novelty > best_novelty:
                    best_novelty = novelty
                    best_action = act_idx

            # Step real environment with highest-curiosity action
            _, _, terminated, truncated, _ = env.step(best_action)
            next_frame = env.render()
            next_frame = jax.image.resize(next_frame, (*img_size, 3), method="bilinear") / 255.0
            next_img = jnp.transpose(next_frame, (2, 0, 1))

            act_vec = jax.nn.one_hot(best_action, 3)
            hidden = world_model.step_hidden(curr_latent, act_vec, hidden)

            ep_obs.append(curr_img)
            ep_act.append(act_vec)
            ep_next.append(next_img)

            curr_img = next_img
            if terminated or truncated:
                obs, _ = env.reset()

        all_obs_t.append(jnp.stack(ep_obs))
        all_actions.append(jnp.stack(ep_act))
        all_obs_next.append(jnp.stack(ep_next))

    env.close()
    return jnp.stack(all_obs_t), jnp.stack(all_actions), jnp.stack(all_obs_next)


def main() -> None:
    key = jax.random.key(0)

    # 1. Load initial Acrobot image trajectories
    obs_t, actions, obs_next = load_acrobot_trajectories("data/acrobot_trajectories.npz")
    action_dim = actions.shape[-1]  # 3 for one-hot Acrobot actions

    # Note: Encoder outputs (latent_channels, 4, 4) -> flattened dim = 16 * latent_channels
    latent_channels = 8
    flat_latent_dim = 16 * latent_channels

    key, model_key = jax.random.split(key)
    world_model = FlowDynamics(
        key=model_key,
        latent_state_dim=latent_channels,
        action_dim=action_dim,
        state_low=jnp.full((flat_latent_dim,), -10.0),
        state_high=jnp.full((flat_latent_dim,), 10.0),
        hidden_dim=128,
        flow_layers=4,
        in_channels=obs_t.shape[2],
        num_spatial_layers=4,
    )

    # 2. Iterative active world-model training loop
    num_active_rounds = 3
    for round_idx in range(num_active_rounds):
        print(f"\n--- Active Training Round {round_idx + 1}/{num_active_rounds} ---")

        # Encode images to latent states if batched_wm_trajectory_loss expects latent vectors.
        # (If rollout.py encodes images internally, pass obs_t and obs_next directly instead.)
        states = encode_episodes(world_model, obs_t)
        next_states = encode_episodes(world_model, obs_next)

        key, train_key = jax.random.split(key)
        world_model, losses = world_training_loop(
            key=train_key,
            states=states,
            actions=actions,
            next_states=next_states,
            world_model=world_model,
            batch_size=32,
            val_batch_size=8,
            traj_size=min(100, states.shape[1] - 10),
            max_epochs=200,
            learning_rate=3e-4,
        )

        # 3. Compute intrinsic motivation (-log_prob) on current dataset
        step_rewards, ep_scores = compute_intrinsic_motivation(
            world_model, states, actions, next_states
        )
        print(
            f"Mean intrinsic surprise (-log_prob): {float(jnp.mean(ep_scores)):.4f} | "
            f"Max episode surprise: {float(jnp.max(ep_scores)):.4f}"
        )

        # 4. Use the trained flow's log_prob to gather high-novelty trajectories for the next round
        if round_idx < num_active_rounds - 1:
            key, collect_key = jax.random.split(key)
            new_obs_t, new_actions, new_obs_next = collect_curiosity_guided_acrobot_data(
                key=collect_key,
                world_model=world_model,
                num_episodes=16,
                seq_len=obs_t.shape[1],
            )
            obs_t = jnp.concatenate([obs_t, new_obs_t], axis=0)
            actions = jnp.concatenate([actions, new_actions], axis=0)
            obs_next = jnp.concatenate([obs_next, new_obs_next], axis=0)

    # Save trained world model weights
    eqx.tree_serialise_leaves("acrobot_flow_dynamics.eqx", world_model)
    print("Saved trained FlowDynamics model to acrobot_flow_dynamics.eqx")


if __name__ == "__main__":
    main()
