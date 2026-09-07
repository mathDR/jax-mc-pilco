"""
PPO training loop for the FlowActor (conditional normalizing-flow policy)
on the custom MJX inverted double-pendulum environment.

Reward follows the classic Gymnasium `InvertedDoublePendulum-v4` shaping:
    reward = alive_bonus - dist_penalty - vel_penalty
using the tip site position and the two pole angular velocities.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import mujoco
import optax
from mujoco import mjx

from jax_mc_pilco.policy_learning.action_flows import FlowActor

# ----------------------------------------------------------------------------
# 1. Environment: MJCF with a `tip` site added for reward computation
# ----------------------------------------------------------------------------
DOUBLE_PENDULUM_XML = """
<mujoco model="inverted_double_pendulum">
    <compiler inertiafromgeom="true"/>
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
        <camera name="side" pos="0 -4 1.2" xyaxes="1 0 0 0 0 1"/>
        <body name="cart" pos="0 0 0.05">
            <joint name="slider" type="slide" axis="1 0 0" range="-10 10"/>
            <geom name="cart_geom" type="box" size="0.2 0.1 0.05" rgba="0.8 0.1 0.1 1"/>
            <body name="pole1" pos="0 0 0">
                <joint name="hinge1" type="hinge" axis="0 1 0"/>
                <geom name="pole1_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.03" rgba="0 0.8 0 1"/>
                <body name="pole2" pos="0 0 0.6">
                    <joint name="hinge2" type="hinge" axis="0 1 0"/>
                    <geom name="pole2_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="0 0 0.8 1"/>
                    <site name="tip" pos="0 0 0.6" size="0.01"/>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="slide_motor" joint="slider" ctrlrange="-20 20"/>
    </actuator>
</mujoco>
"""


mj_model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
mjx_model = mjx.put_model(mj_model)
TIP_SITE_ID = mj_model.site("tip").id
MAX_TIP_HEIGHT = 1.2  # two 0.6m poles fully extended
FALL_HEIGHT = 1.0  # terminate/mask reward once tip drops below this

STATE_DIM = 6  # qpos(3) + qvel(3)
ACTION_DIM = 1
COND_DIM = STATE_DIM * 2  # policy conditions on [prev_state, curr_state]


def obs_from_data(data: mjx.Data) -> jtp.Float[jtp.Array, " 6"]:
    return jnp.concatenate([data.qpos, data.qvel])


def reward_and_done(data: mjx.Data) -> tuple[jtp.Float[jtp.Array, ""], jtp.Bool[jtp.Array, ""]]:
    tip = data.site_xpos[TIP_SITE_ID]
    x, _, z = tip[0], tip[1], tip[2]
    v1, v2 = data.qvel[1], data.qvel[2]
    dist_penalty = 0.01 * x**2 + (z - MAX_TIP_HEIGHT) ** 2
    vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    alive_bonus = 10.0
    reward = alive_bonus - dist_penalty - vel_penalty
    done = z <= FALL_HEIGHT
    return reward, done


# ----------------------------------------------------------------------------
# 3. Critic: simple MLP over the same [prev_state, curr_state] context
# ----------------------------------------------------------------------------
class Critic(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        cond_dim: int,
    ):
        self.mlp = eqx.nn.MLP(in_size=cond_dim, out_size="scalar", width_size=128, depth=2, key=key)

    def __call__(self, prev_state: jax.Array, curr_state: jax.Array) -> jax.Array:
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.mlp(context)


class Agent(eqx.Module):
    actor: FlowActor
    critic: Critic


# ----------------------------------------------------------------------------
# 4. Rollout collection (vmapped over N parallel envs, scanned over T steps)
# ----------------------------------------------------------------------------
def make_initial_carry(key: jtp.Key[jtp.Array, ""]) -> tuple[jax.Array, jax.Array]:
    data = mjx.forward(mjx_model, mjx.make_data(mjx_model))
    obs0 = obs_from_data(data)
    # prev_state == curr_state at t=0 (no history yet)
    return data, obs0


@eqx.filter_jit
def rollout(
    agent: Agent, data0: mjx.Data, prev0: jax.Array, key: jtp.Key[jtp.Array, ""], n_steps: int
) -> tuple[dict, jax.Array]:
    def step(
        carry: tuple,
        key: jtp.Key[jtp.Array, ""],
    ) -> tuple[
        tuple,
        dict,
    ]:
        data, prev_state = carry
        curr_state = obs_from_data(data)
        sample_key, reset_key = jax.random.split(key)
        action, logp = agent.actor.sample_action_and_log_prob(sample_key, prev_state, curr_state)
        value = agent.critic(prev_state, curr_state)
        next_data = mjx.step(mjx_model, data.replace(ctrl=action))
        reward, done = reward_and_done(next_data)

        # Auto-reset: once the pendulum falls, snap back to the (slightly
        # randomized) upright start rather than letting it lie fallen for the
        # rest of the fixed-length rollout. This keeps every step's learning
        # signal informative instead of diluting the batch with "already
        # fallen, nothing left to learn" transitions, and is the standard way
        # to handle episode boundaries inside a jax.lax.scan.
        reset_data = mjx.forward(mjx_model, mjx.make_data(mjx_model))
        noise = 0.05 * jax.random.normal(reset_key, reset_data.qpos.shape)
        reset_data = reset_data.replace(qpos=reset_data.qpos + noise)
        reset_data = mjx.forward(mjx_model, reset_data)
        next_data = jax.tree_util.tree_map(
            lambda reset, cur: jnp.where(done, reset, cur) if eqx.is_array(reset) and reset.shape == cur.shape else cur,
            reset_data,
            next_data,
        )
        next_curr_state = obs_from_data(next_data)
        # On a reset step, the "previous state" history is discarded too.
        new_prev_state = jnp.where(done, next_curr_state, curr_state)

        transition = {
            "prev_state": prev_state,
            "curr_state": curr_state,
            "action": action,
            "log_prob": logp,
            "value": value,
            "reward": reward,
            "done": done,
        }
        new_carry = (next_data, new_prev_state)
        return new_carry, transition

    keys = jax.random.split(key, n_steps)
    (final_data, final_prev), traj = jax.lax.scan(step, (data0, prev0), keys)
    last_curr = obs_from_data(final_data)
    last_value = agent.critic(final_prev, last_curr)
    return traj, last_value


@eqx.filter_jit
def batched_rollout(agent: Agent, keys: jtp.Key[jtp.Array, ""], n_steps: int) -> tuple:
    """keys: shape [B] of per-env PRNG keys."""

    def one_env(
        key: jtp.Key[jtp.Array, ""],
    ) -> tuple[dict, jax.Array]:
        init_key, roll_key = jax.random.split(key)
        data0, prev0 = make_initial_carry(init_key)
        return rollout(agent, data0, prev0, roll_key, n_steps)

    return jax.vmap(one_env)(keys)


# ----------------------------------------------------------------------------
# 5. GAE advantage estimation
# ----------------------------------------------------------------------------
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """rewards/values/dones: [T] for a single env. Returns advantages, returns: [T]."""
    not_done = 1.0 - dones.astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], last_value[None]])

    def scan_fn(carry: tuple, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        r, v, nv, nd = x
        delta = r + gamma * nv * nd - v
        adv = delta + gamma * lam * nd * carry
        return adv, adv

    _, advs_rev = jax.lax.scan(scan_fn, jnp.array(0.0), (rewards, values, next_values, not_done), reverse=True)
    returns = advs_rev + values
    return advs_rev, returns


# ----------------------------------------------------------------------------
# 6. PPO loss and update
# ----------------------------------------------------------------------------
def ppo_loss(
    agent: Agent,
    batch: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jax.Array, dict]:
    new_logp = jax.vmap(agent.actor.log_prob)(batch["action"], batch["prev_state"], batch["curr_state"])
    # The sigmoid-squash bijection's log-density diverges for samples that land
    # very close to the action bounds. Clamp to keep the PPO ratio finite -
    # this is the same trick used for tanh-squashed Gaussian policies in SAC.
    # Belt-and-suspenders: nan_to_num first (clip alone does not fix true NaNs,
    # only bounds finite values), then clip to keep the ratio's exponent sane.
    new_logp = jnp.clip(jnp.nan_to_num(new_logp, nan=-30.0), -30.0, 30.0)
    old_logp = jnp.clip(jnp.nan_to_num(batch["log_prob"], nan=-30.0), -30.0, 30.0)
    new_value = jax.vmap(agent.critic)(batch["prev_state"], batch["curr_state"])

    ratio = jnp.exp(jnp.clip(new_logp - old_logp, -20.0, 20.0))
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    value_loss = jnp.mean((new_value - batch["return"]) ** 2)
    entropy_bonus = jnp.mean(-new_logp)  # sample-based entropy estimate

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus
    return loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy_bonus}


@eqx.filter_jit
def ppo_minibatch_step(
    agent: Agent, opt_state: optax.OptState, optimizer: optax.GradientTransformation, mb, clip_eps, vf_coef, ent_coef
) -> tuple[Agent, optax.OptState, jax.Array, jax.Array]:
    (loss, info), grads = eqx.filter_value_and_grad(ppo_loss, has_aux=True)(agent, mb, clip_eps, vf_coef, ent_coef)
    updates, new_opt_state = optimizer.update(grads, opt_state, agent)
    new_agent = eqx.apply_updates(agent, updates)

    # A minibatch can occasionally contain an action that lands in a region
    # the flow's bijection struggles to invert (log-density -> -inf); even
    # after clipping the *value*, backprop through that point can still
    # yield a NaN *gradient* (a well-known jnp.where-gradient pitfall). Rather
    # than chase down every such singular point inside the flow, we simply
    # skip the update when this happens - one skipped minibatch out of many
    # has negligible effect, and this keeps training from derailing entirely.
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    grad_is_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(g)) for g in grad_leaves]))

    agent = jax.tree_util.tree_map(
        lambda new, old: jnp.where(grad_is_finite, new, old) if eqx.is_array(new) else new,
        new_agent,
        agent,
    )
    opt_state = jax.tree_util.tree_map(
        lambda new, old: jnp.where(grad_is_finite, new, old) if eqx.is_array(new) else new,
        new_opt_state,
        opt_state,
    )
    return agent, opt_state, loss, info


def ppo_update_epoch(agent, opt_state, optimizer, batch, perm, minibatch_size, clip_eps, vf_coef, ent_coef):
    # A plain Python loop over minibatches: `agent`'s pytree (the flow's bijection
    # stack) contains non-array closures as leaves, which `jax.lax.scan` cannot
    # carry. Each individual step below is still jit-compiled via
    # `ppo_minibatch_step`, so this stays fast - only the outer loop is Python.
    n = perm.shape[0]
    n_minibatches = n // minibatch_size
    losses, infos = [], []
    for i in range(n_minibatches):
        mb_idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        mb = jax.tree_util.tree_map(lambda x: x[mb_idx], batch)
        agent, opt_state, loss, info = ppo_minibatch_step(agent, opt_state, optimizer, mb, clip_eps, vf_coef, ent_coef)
        losses.append(loss)
        infos.append(info)
    infos = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *infos)
    return agent, opt_state, jnp.stack(losses), infos


# ----------------------------------------------------------------------------
# 7. Training driver
# ----------------------------------------------------------------------------
def train(
    n_envs: int = 16,
    n_steps: int = 128,
    n_iterations: int = 20,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.005,  # returns run to ~1000 (gamma=0.99, +10/step alive bonus),
    # so the squared value-loss term is naturally ~1e4-1e6x the policy loss scale;
    # this keeps the two gradient signals comparable without rescaling targets.
    ent_coef: float = 0.001,
    seed: int = 0,
):
    key = jax.random.key(seed)
    actor_key, critic_key, key = jax.random.split(key, 3)

    action_low = mjx_model.actuator_ctrlrange[:, 0]
    action_high = mjx_model.actuator_ctrlrange[:, 1]
    actor = FlowActor(actor_key, STATE_DIM, ACTION_DIM, action_low, action_high)
    critic = Critic(critic_key, COND_DIM)
    agent = Agent(actor=actor, critic=critic)

    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_array))

    for it in range(n_iterations):
        key, rollout_key, perm_key = jax.random.split(key, 3)
        env_keys = jax.random.split(rollout_key, n_envs)

        traj, last_value = batched_rollout(agent, env_keys, n_steps)
        # traj fields have shape [n_envs, n_steps, ...]; last_value: [n_envs]

        adv_fn = jax.vmap(lambda r, v, d, lv: compute_gae(r, v, d, lv, gamma, lam))
        advantages, returns = adv_fn(traj["reward"], traj["value"], traj["done"], last_value)

        flat = jax.tree_util.tree_map(lambda x: x.reshape((n_envs * n_steps,) + x.shape[2:]), traj)
        flat["advantage"] = advantages.reshape(-1)
        flat["return"] = returns.reshape(-1)

        n_total = n_envs * n_steps
        for _ in range(ppo_epochs):
            perm_key, sub = jax.random.split(perm_key)
            perm = jax.random.permutation(sub, n_total)
            agent, opt_state, losses, infos = ppo_update_epoch(
                agent, opt_state, optimizer, flat, perm, minibatch_size, clip_eps, vf_coef, ent_coef
            )

        mean_reward = jnp.mean(traj["reward"])
        alive_frac = jnp.mean(1.0 - traj["done"].astype(jnp.float32))
        print(
            f"iter {it:03d} | mean_reward={mean_reward:.3f} | alive_frac={alive_frac:.3f} "
            f"| policy_loss={jnp.mean(infos['policy_loss']):.4f} "
            f"| value_loss={jnp.mean(infos['value_loss']):.4f} "
            f"| entropy={jnp.mean(infos['entropy']):.4f}"
        )

    return agent


def eval_rollout_qpos(agent: Agent, key: jax.Array, n_steps: int):
    """Single-env rollout (stochastic actions from the trained flow) returning
    the qpos trajectory needed for rendering, plus reward/done for reporting."""

    def step(carry, key):
        data, prev_state = carry
        curr_state = obs_from_data(data)
        action = agent.actor.sample_action(key, prev_state, curr_state)
        next_data = mjx.step(mjx_model, data.replace(ctrl=action))
        reward, done = reward_and_done(next_data)
        return (next_data, curr_state), (next_data.qpos, reward, done)

    data0 = mjx.forward(mjx_model, mjx.make_data(mjx_model))
    keys = jax.random.split(key, n_steps)
    _, (qpos_traj, rewards, dones) = jax.lax.scan(step, (data0, obs_from_data(data0)), keys)
    return qpos_traj, rewards, dones


def render_video(
    agent: Agent,
    path: str,
    n_steps: int = 300,
    seed: int = 0,
    fps: int | None = None,
    camera: str = "side",
    width: int = 480,
    height: int = 352,
):
    """Rolls out `agent` for `n_steps` and writes an mp4 of the episode to
    `path`. Uses the CPU-based `mujoco.Renderer` (not mjx) since MJX has no
    rendering support of its own - the qpos trajectory computed on-device is
    replayed frame-by-frame through the regular MuJoCo model just to draw it.
    """
    import imageio.v2 as imageio

    key = jax.random.key(seed)
    qpos_traj, rewards, dones = eval_rollout_qpos(agent, key, n_steps)
    qpos_traj = jax.device_get(qpos_traj)
    fps = fps or int(round(1.0 / mj_model.opt.timestep))

    render_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []
    for qpos in qpos_traj:
        render_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, render_data)
        renderer.update_scene(render_data, camera=camera)
        frames.append(renderer.render())
    renderer.close()

    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)

    mean_reward = float(jnp.mean(rewards))
    fell = bool(jnp.any(dones))
    fall_step = int(jnp.argmax(dones)) if fell else None
    print(
        f"Saved {len(frames)}-frame video to {path} "
        f"(mean_reward={mean_reward:.3f}, "
        + (f"fell at step {fall_step}" if fell else "stayed upright throughout")
        + ")"
    )
    return path


if __name__ == "__main__":
    trained_agent = train(n_envs=32, n_steps=200, n_iterations=200, ppo_epochs=10, minibatch_size=512)
    render_video(trained_agent, "pendulum_policy.mp4", n_steps=300, seed=123)
