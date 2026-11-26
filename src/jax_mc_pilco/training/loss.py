import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
from beartype import beartype

from jax_mc_pilco.model_learning.flow_model import LatentFlowWorldModel
from jax_mc_pilco.policy_learning.action_flows import ActionFlow, Critic

NUM_BURN_IN_STEPS = 10


@jtp.jaxtyped(typechecker=beartype)
def compute_lambda_returns(
    rewards: jtp.Float[jtp.Array, "B H"],
    values: jtp.Float[jtp.Array, "B H_plus_1"],
    discounts: jtp.Float[jtp.Array, "B H"],
    lambda_: float = 0.95,
) -> jtp.Float[jtp.Array, "B H"]:
    """Compute TD(lambda) targets backward across imagined horizon H."""
    # Scan backward over time axis (transpose to [H, B] for jax.lax.scan)
    rewards_T = jnp.swapaxes(rewards, 0, 1)
    discounts_T = jnp.swapaxes(discounts, 0, 1)
    next_values_T = jnp.swapaxes(values[:, 1:], 0, 1)

    def backward_step(
        next_return: jtp.Float[jtp.Array, " B"],
        inputs: tuple[
            jtp.Float[jtp.Array, " B"],
            jtp.Float[jtp.Array, " B"],
            jtp.Float[jtp.Array, " B"],
        ],
    ) -> tuple[jtp.Float[jtp.Array, " B"], jtp.Float[jtp.Array, " B"]]:
        r_t, gamma_t, v_next = inputs
        target_t = r_t + gamma_t * ((1.0 - lambda_) * v_next + lambda_ * next_return)
        return target_t, target_t

    _, returns_T = jax.lax.scan(
        backward_step,
        init=values[:, -1],  # Bootstrap from V(s_H, h_H)
        xs=(rewards_T, discounts_T, next_values_T),
        reverse=True,
    )
    return jnp.swapaxes(returns_T, 0, 1)


@jtp.jaxtyped(typechecker=beartype)
def single_latent_wm_trajectory_loss(
    world_model_params: LatentFlowWorldModel,
    world_model_static: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    obs: jtp.Float[jtp.Array, "seq_len C H W"],
    actions: jtp.Float[jtp.Array, "seq_len action_dim"],
    rewards: jtp.Float[jtp.Array, " seq_len"],
    dones: jtp.Bool[jtp.Array, " seq_len"],
    *,
    kl_scale: float = 0.1,
    free_nats: float = 1.0,
) -> jtp.Float[jtp.Array, ""]:
    """Compute latent world-model ELBO loss with gradient checkpointing for low VRAM."""
    world_model = eqx.combine(world_model_params, world_model_static)

    # Checkpointed encoder pass over all frames in parallel
    embeds = eqx.filter_vmap(eqx.filter_checkpoint(world_model.encoder))(obs)

    init_hidden = jnp.zeros((world_model.memory.hidden_size,), dtype=embeds.dtype)
    init_z = jnp.zeros((world_model.latent_dim,), dtype=embeds.dtype)

    def scan_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        prev_z, prev_hidden, prev_done, step_key = carry
        embed_t, action_t, done_t = inputs

        prev_hidden = jnp.where(prev_done, 0.0, prev_hidden)
        prev_z = jnp.where(prev_done, 0.0, prev_z)

        next_key, post_key = jax.random.split(step_key)

        z_t, q_log_prob = world_model.posterior_sample_and_log_prob(post_key, embed_t, prev_hidden)
        p_log_prob = world_model.prior_log_prob(prev_z, z_t, prev_hidden)
        kl_loss = jnp.where(
            prev_done,
            0.0,
            kl_scale * jnp.maximum(q_log_prob - p_log_prob, free_nats),
        )

        next_hidden = world_model.step_hidden(z_t, action_t, prev_hidden)
        return (z_t, next_hidden, done_t, next_key), (z_t, prev_hidden, kl_loss)

    _, (latents, hiddens, kl_losses) = jax.lax.scan(
        scan_step,
        (init_z, init_hidden, jnp.asarray(True), key),
        (embeds, actions, dones),
    )

    # Decode frames and predict rewards outside the scan with gradient checkpointing
    post_latents = latents[NUM_BURN_IN_STEPS:]
    post_hiddens = hiddens[NUM_BURN_IN_STEPS:]
    post_obs = obs[NUM_BURN_IN_STEPS:]
    post_rewards = rewards[NUM_BURN_IN_STEPS:]
    post_kl = kl_losses[NUM_BURN_IN_STEPS:]

    recon_obs = eqx.filter_vmap(eqx.filter_checkpoint(world_model.decode_obs))(post_latents, post_hiddens)
    pred_rewards = eqx.filter_vmap(world_model.predict_reward)(post_latents, post_hiddens)

    recon_loss = jnp.mean((recon_obs - post_obs) ** 2, axis=(1, 2, 3))
    reward_loss = optax.losses.huber_loss(pred_rewards, post_rewards)

    return jnp.mean(recon_loss + reward_loss + post_kl)


@jtp.jaxtyped(typechecker=beartype)
def batched_wm_trajectory_loss(
    world_model_params: LatentFlowWorldModel,
    world_model_static: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    obs: jtp.UInt8[jtp.Array, "num_episodes seq_len C H W"],
    actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
    rewards: jtp.Float[jtp.Array, "num_episodes seq_len"],
    dones: jtp.Bool[jtp.Array, "num_episodes seq_len"],
    *,
    kl_scale: float = 0.1,
    free_nats: float = 1.0,
) -> jtp.Float[jtp.Array, ""]:
    """Batched latent world-model ELBO loss across sampled trajectory chunks."""
    # Convert uint8 pixels to centered float32 [-0.5, 0.5] on device
    norm_obs = obs.astype(jnp.float32) / 255.0 - 0.5
    episode_keys = jax.random.split(key, norm_obs.shape[0])

    losses = eqx.filter_vmap(
        lambda ep_key, ep_obs, ep_actions, ep_rewards, ep_dones: single_latent_wm_trajectory_loss(
            world_model_params,
            world_model_static,
            ep_key,
            ep_obs,
            ep_actions,
            ep_rewards,
            ep_dones,
            kl_scale=kl_scale,
            free_nats=free_nats,
        ),
        in_axes=(0, 0, 0, 0, 0),
    )(episode_keys, norm_obs, actions, rewards, dones)

    return jnp.mean(losses)


@jtp.jaxtyped(typechecker=beartype)
def single_act_trajectory_loss(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    obs: jtp.Float[jtp.Array, "seq_len C H W"],
    actions: jtp.Float[jtp.Array, "seq_len action_dim"],
    dones: jtp.Bool[jtp.Array, " seq_len"],
) -> jtp.Float[jtp.Array, ""]:
    """Compute behavior-cloning NLL loss in latent space for a single pixel trajectory."""
    action_flow = eqx.combine(action_params, action_static)

    # Encode all frames in parallel without building gradients into the frozen world model
    embeds = jax.lax.stop_gradient(eqx.filter_vmap(world_model.encoder)(obs))

    init_hidden = jnp.zeros((world_model.memory.hidden_size,), dtype=embeds.dtype)
    init_z = jnp.zeros((world_model.latent_dim,), dtype=embeds.dtype)

    def scan_fn(
        carry: tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        tuple[jax.Array, jax.Array],
    ]:
        prev_z, prev_hidden, prev_done, step_key = carry
        embed_t, action_t, done_t = inputs

        # Reset recurrent hidden state and previous latent at episode boundaries
        prev_hidden = jnp.where(prev_done, 0.0, prev_hidden)
        prev_z = jnp.where(prev_done, 0.0, prev_z)

        next_key, post_key = jax.random.split(step_key)

        # Infer current posterior latent z_t from (h_t, e_t)
        z_t, _ = jax.lax.stop_gradient(world_model.posterior_sample_and_log_prob(post_key, embed_t, prev_hidden))

        # Advance recurrent hidden state h_{t+1} = GRU([z_t, a_t], h_t)
        next_hidden = jax.lax.stop_gradient(world_model.step_hidden(z_t, action_t, prev_hidden))

        # Output (z_t, prev_hidden) so action_t is predicted strictly from pre-action context
        return (z_t, next_hidden, done_t, next_key), (z_t, prev_hidden)

    _, (latents, hiddens) = jax.lax.scan(
        scan_fn,
        (init_z, init_hidden, jnp.asarray(True), key),
        (embeds, actions, dones),
        unroll=5,
    )

    nll_losses = -eqx.filter_vmap(action_flow.log_prob, in_axes=(0, 0, 0))(
        actions[NUM_BURN_IN_STEPS:],
        latents[NUM_BURN_IN_STEPS:],
        hiddens[NUM_BURN_IN_STEPS:],
    )
    return jnp.mean(nll_losses)


@jtp.jaxtyped(typechecker=beartype)
def batched_act_trajectory_loss(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    obs: jtp.UInt8[jtp.Array, "batch seq_len C H W"],
    actions: jtp.Float[jtp.Array, "batch seq_len action_dim"],
    dones: jtp.Bool[jtp.Array, "batch seq_len"],
) -> jtp.Float[jtp.Array, ""]:
    """Batched latent behavior-cloning loss across Flashbax trajectory chunks."""
    norm_obs = obs.astype(jnp.float32) / 255.0 - 0.5
    episode_keys = jax.random.split(key, norm_obs.shape)

    losses = eqx.filter_vmap(
        single_act_trajectory_loss,
        in_axes=(None, None, None, 0, 0, 0, 0),
    )(
        action_params,
        action_static,
        world_model,
        episode_keys,
        norm_obs,
        actions,
        dones,
    )
    return jnp.mean(losses)


@jtp.jaxtyped(typechecker=beartype)
def imagination_trajectory(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    initial_obs: jtp.Float[jtp.Array, "burn_in_len C H W"],
    initial_actions: jtp.Float[jtp.Array, "burn_in_len action_dim"],
    initial_dones: jtp.Bool[jtp.Array, " burn_in_len"],
) -> tuple[
    jtp.Float[jtp.Array, "horizon_plus_1 latent_dim"],
    jtp.Float[jtp.Array, "horizon action_dim"],
    jtp.Float[jtp.Array, "horizon_plus_1 hidden_dim"],
    jtp.Float[jtp.Array, " horizon"],
]:
    """Burn in on observed pixel frames to infer (z_0, h_0), then imagine `horizon` steps."""
    action_flow = eqx.combine(action_params, action_static)

    # 1. Encode burn-in frames and infer posterior latent trajectory (frozen world model)
    embeds = jax.lax.stop_gradient(eqx.filter_vmap(world_model.encoder)(initial_obs[: NUM_BURN_IN_STEPS + 1]))
    init_hidden = jnp.zeros((world_model.memory.hidden_size,), dtype=embeds.dtype)
    init_z = jnp.zeros((world_model.latent_dim,), dtype=embeds.dtype)

    def burn_in_fn(
        carry: tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        step_inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray], None]:
        prev_z, prev_hidden, prev_done, step_key = carry
        embed_t, action_t, done_t = step_inputs

        prev_hidden = jnp.where(prev_done, 0.0, prev_hidden)
        prev_z = jnp.where(prev_done, 0.0, prev_z)

        next_key, post_key = jax.random.split(step_key)
        z_t, _ = jax.lax.stop_gradient(world_model.posterior_sample_and_log_prob(post_key, embed_t, prev_hidden))
        next_hidden = jax.lax.stop_gradient(world_model.step_hidden(z_t, action_t, prev_hidden))
        return (z_t, next_hidden, done_t, next_key), None

    key, burn_key, z0_key, first_action_key = jax.random.split(key, 4)
    (_, burn_in_hidden, last_done, _), _ = jax.lax.scan(
        burn_in_fn,
        (init_z, init_hidden, jnp.asarray(True), burn_key),
        (
            embeds[:NUM_BURN_IN_STEPS],
            initial_actions[:NUM_BURN_IN_STEPS],
            initial_dones[:NUM_BURN_IN_STEPS],
        ),
    )

    # Infer z_0 at the end of burn-in (t = NUM_BURN_IN_STEPS)
    burn_in_hidden = jnp.where(last_done, 0.0, burn_in_hidden)
    start_z, _ = jax.lax.stop_gradient(
        world_model.posterior_sample_and_log_prob(z0_key, embeds[NUM_BURN_IN_STEPS], burn_in_hidden)
    )

    # 2. Sample the first imagined action at t = 0 of imagination
    first_action, first_log_prob = action_flow.sample_and_log_prob(first_action_key, start_z, burn_in_hidden)

    # 3. Imagine forward for `horizon` transitions in latent space
    def scan_fn(
        carry: tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        _: None,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jtp.PRNGKeyArray],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        prev_z, prev_action, prev_hidden, step_key = carry
        next_key, state_key, action_key = jax.random.split(step_key, 3)

        next_z, next_hidden = world_model.predict_next_state_and_hidden(
            state_key,
            prev_z,
            prev_action,
            prev_hidden,
        )
        next_action, next_log_prob = action_flow.sample_and_log_prob(action_key, next_z, next_hidden)

        return (next_z, next_action, next_hidden, next_key), (
            next_z,
            next_action,
            next_hidden,
            next_log_prob,
        )

    _, (future_zs, future_actions, future_hiddens, future_log_probs) = jax.lax.scan(
        scan_fn,
        (start_z, first_action, burn_in_hidden, key),
        xs=None,
        length=horizon,
    )

    zs = jnp.concatenate([start_z[None], future_zs], axis=0)
    hiddens = jnp.concatenate([burn_in_hidden[None], future_hiddens], axis=0)
    actions = jnp.concatenate([first_action[None], future_actions[:-1]], axis=0)
    log_probs = jnp.concatenate([first_log_prob[None], future_log_probs[:-1]], axis=0)

    return zs, actions, hiddens, log_probs


@jtp.jaxtyped(typechecker=beartype)
def batched_imagination_trajectory(
    action_params: ActionFlow,
    action_static: ActionFlow,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    obs: jtp.Float[jtp.Array, "num_episodes burn_in_len C H W"],
    actions: jtp.Float[jtp.Array, "num_episodes burn_in_len action_dim"],
    dones: jtp.Bool[jtp.Array, "num_episodes burn_in_len"],
) -> tuple[
    jtp.Float[jtp.Array, "num_episodes horizon_plus_1 latent_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon action_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon_plus_1 hidden_dim"],
    jtp.Float[jtp.Array, "num_episodes horizon"],
]:
    """Generate a batch of imagined latent trajectories from pixel burn-in windows."""
    episode_keys = jax.random.split(key, obs.shape)
    return eqx.filter_vmap(
        imagination_trajectory,
        in_axes=(None, None, None, 0, None, 0, 0, 0),
    )(
        action_params,
        action_static,
        world_model,
        episode_keys,
        horizon,
        obs,
        actions,
        dones,
    )


@jtp.jaxtyped(typechecker=beartype)
def batched_actor_critic_imagination_loss(
    actor_params: ActionFlow,
    actor_static: ActionFlow,
    critic_params: Critic,
    target_critic_params: Critic,
    critic_static: Critic,
    world_model: LatentFlowWorldModel,
    key: jtp.Key[jtp.Array, ""],
    horizon: int,
    obs: jtp.UInt8[jtp.Array, "num_episodes seq_len C H W"],
    actions: jtp.Float[jtp.Array, "num_episodes seq_len action_dim"],
    dones: jtp.Bool[jtp.Array, "num_episodes seq_len"],
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coef: float = 1e-3,
    huber_delta: float = 1.0,
) -> tuple[jtp.Float[jtp.Array, ""], dict[str, jax.Array]]:
    """Compute joint actor and critic losses using imagined latent trajectories."""
    critic = eqx.combine(critic_params, critic_static)
    target_critic = eqx.combine(target_critic_params, critic_static)

    norm_obs = obs.astype(jnp.float32) / 255.0 - 0.5
    burn_in_obs = norm_obs[:, : NUM_BURN_IN_STEPS + 1]
    burn_in_actions = actions[:, : NUM_BURN_IN_STEPS + 1]
    burn_in_dones = dones[:, : NUM_BURN_IN_STEPS + 1]

    # 1. Roll out `horizon` imagined latent transitions from (z_0, h_0)
    pred_zs, _, pred_hiddens, pred_log_probs = batched_imagination_trajectory(
        actor_params,
        actor_static,
        world_model,
        key,
        horizon,
        burn_in_obs,
        burn_in_actions,
        burn_in_dones,
    )

    # 2. Predict rewards along imagined transitions [1..H]
    rewards = jax.vmap(jax.vmap(world_model.predict_reward))(pred_zs[:, 1:, :], pred_hiddens[:, 1:, :])  # [B, H]

    # 3. Evaluate live Critic for V_0..V_{H-1} and EMA Target Critic for V_0..V_H
    norm_zs = jax.vmap(jax.vmap(world_model.normalize_state))(pred_zs)
    live_values = jax.vmap(jax.vmap(critic))(norm_zs[:, :-1, :], pred_hiddens[:, :-1, :])  # [B, H]
    target_values = jax.vmap(jax.vmap(target_critic))(norm_zs, pred_hiddens)  # [B, H + 1]

    discounts = jnp.full_like(rewards, gamma)

    # 4. Compute TD(lambda) targets using the slow EMA target critic
    lambda_returns = compute_lambda_returns(
        rewards=rewards,
        values=target_values,
        discounts=discounts,
        lambda_=lambda_,
    )  # [B, H]

    # 5. Critic loss: fit live V(z_t, h_t) to stop_gradient(lambda_returns)
    critic_loss = jnp.mean(
        optax.losses.huber_loss(
            live_values,
            jax.lax.stop_gradient(lambda_returns),
            delta=huber_delta,
        )
    )

    # 6. Actor loss: maximize lambda_returns + entropy bonus
    policy_value_loss = -jnp.mean(lambda_returns)
    entropy_loss = entropy_coef * jnp.mean(pred_log_probs)
    actor_loss = policy_value_loss + entropy_loss

    total_loss = actor_loss + critic_loss
    metrics = {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "mean_return": jnp.mean(lambda_returns),
        "mean_reward": jnp.mean(rewards),
        "entropy": -jnp.mean(pred_log_probs),
    }
    return total_loss, metrics
