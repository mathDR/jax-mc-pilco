import equinox as eqx
import jax
from dm_control import suite
from flashbax.vault import Vault

from jax_mc_pilco.model_learning.flow_model import LatentFlowWorldModel
from jax_mc_pilco.policy_learning.action_flows import ActionFlow, Critic
from jax_mc_pilco.simulators.collect_data import collect_data, load_buffer_from_vault
from jax_mc_pilco.training.learning import imagination_training_loop, policy_training_loop, world_training_loop


def main() -> None:
    env = suite.load(domain_name="acrobot", task_name="swingup")
    key = jax.random.key(seed=4)

    vault_name = "acrobot_pixels"
    vault_uid = "v1"
    rel_dir = "vaults"

    sample_batch_size = 16
    val_batch_size = 16
    sample_seq_len = 26  # NUM_BURN_IN_STEPS (10) + 1 + horizon (15)

    # 1. Create a new Flashbax buffer and collect 32 trajectories of length 200 (pixels)
    key, subkey = jax.random.split(key)
    buffer, buffer_state = collect_data(
        env,
        num_trajectories=32,
        traj_length=200,
        key=subkey,
        use_pixels=True,
        img_size=64,
        action_repeat=2,
        sample_batch_size=sample_batch_size,
        sample_seq_len=sample_seq_len,
    )

    # 2. Append 32 more trajectories to the existing buffer
    key, subkey = jax.random.split(key)
    buffer, buffer_state = collect_data(
        env,
        num_trajectories=32,
        traj_length=200,
        key=subkey,
        replay_buffer=(buffer, buffer_state),
        use_pixels=True,
        img_size=64,
        action_repeat=2,
    )

    # 3. Persist the buffer to disk using Vault
    vault = Vault(
        vault_name=vault_name,
        experience_structure=buffer_state.experience,
        rel_dir=rel_dir,
        vault_uid=vault_uid,
    )
    vault.write(buffer_state)

    # 4. Load non-overlapping train (0-90%) and validation (90-100%) buffers
    train_buffer, train_buffer_state = load_buffer_from_vault(
        vault_name=vault_name,
        vault_uid=vault_uid,
        batch_size=sample_batch_size,
        sample_seq_len=sample_seq_len,
        rel_dir=rel_dir,
        percentiles=(0, 90),
    )
    val_buffer, val_buffer_state = load_buffer_from_vault(
        vault_name=vault_name,
        vault_uid=vault_uid,
        batch_size=val_batch_size,
        sample_seq_len=sample_seq_len,
        rel_dir=rel_dir,
        percentiles=(90, 100),
    )

    # 5. Initialize models in latent space (latent_dim=32) rather than raw pixel space
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

    # Note: Ensure ActionFlow's action_low/high match dm_control's [-1.0, 1.0] bounds
    action_flow_baseline = ActionFlow(
        key=act_key,
        state_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    critic = Critic(
        state_dim=latent_dim,
        rnn_hidden_dim=hidden_dim,
        mlp_hidden_dim=256,
        key=critic_key,
    )

    train_wm_model = True
    train_act_model = True

    # 6. Train the Latent Pixel World Model (Encoder + Posterior + Flow Prior + Decoder + Reward)
    if train_wm_model:
        key, subkey = jax.random.split(key)
        dynamics, d_losses = world_training_loop(
            subkey,
            (train_buffer, train_buffer_state),
            (val_buffer, val_buffer_state),
            world_model_baseline,
        )
        eqx.tree_serialise_leaves("world_model.eqx", dynamics)
    else:
        dynamics = eqx.tree_deserialise_leaves("world_model.eqx", world_model_baseline)

    # 7. Train the Behavior Policy in latent space
    if train_act_model:
        key, subkey = jax.random.split(key)
        policy, p_losses = policy_training_loop(
            subkey,
            (train_buffer, train_buffer_state),
            (val_buffer, val_buffer_state),
            dynamics,
            action_flow_baseline,
        )
        eqx.tree_serialise_leaves("action_model.eqx", policy)
    else:
        policy = eqx.tree_deserialise_leaves("action_model.eqx", action_flow_baseline)
    # 8. Jointly refine Actor and Critic via latent imagination rollouts
    key, subkey = jax.random.split(key)
    trained_actor, trained_critic, history = imagination_training_loop(
        subkey,
        (train_buffer, train_buffer_state),
        (val_buffer, val_buffer_state),
        dynamics,
        policy,
        critic,
    )
    eqx.tree_serialise_leaves("imagination_policy.eqx", trained_actor)
    eqx.tree_serialise_leaves("imagination_critic.eqx", trained_critic)


if __name__ == "__main__":
    main()
