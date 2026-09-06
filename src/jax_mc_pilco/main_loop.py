# Now load your libraries cleanly
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import typing

import warnings
import logging

# Silence the specific warp missing warnings from mujoco
logging.getLogger("mujoco").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*warp.*")

# Now import your JAX / MuJoCo dependencies safely
import jax
from mujoco import mjx

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp

import optax
from brax import envs
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow

# ==========================================
# 1. CORE CONTAINERS & DATA STRUCTURES
# ==========================================

class TransitionBatch(typing.NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array

class ReplayBuffer(eqx.Module):
    buffers: TransitionBatch
    capacity: int = eqx.field(static=True)
    ptr: jax.Array

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.buffers = TransitionBatch(
            obs=jnp.zeros((capacity, obs_dim)),
            action=jnp.zeros((capacity, action_dim)),
            reward=jnp.zeros((capacity,)),
            next_obs=jnp.zeros((capacity, obs_dim)),
            done=jnp.zeros((capacity,))
        )
        self.ptr = jnp.array(0, dtype=jnp.int32)

    def add(self, transition: TransitionBatch) -> "ReplayBuffer":
        num_items: int = transition.obs.shape[0]
        indices: jax.Array = (self.ptr + jnp.arange(num_items)) % self.capacity

        new_buffers: TransitionBatch = jax.tree_util.tree_map(
            lambda buf, new_data: buf.at[indices].set(new_data),
            self.buffers, transition
        )
        new_ptr: jax.Array = (self.ptr + num_items) % self.capacity
        return eqx.tree_at(lambda r: [r.buffers, r.ptr], self, [new_buffers, new_ptr])

    def sample(self, key: jtp.Key[jtp.Array, ""], batch_size: int) -> TransitionBatch:
        max_idx: jax.Array = jnp.where(self.ptr == 0, self.capacity, self.ptr)
        idx: jax.Array = jax.random.randint(key, (batch_size,), 0, max_idx)
        return jax.tree_util.tree_map(lambda buf: buf[idx], self.buffers)

# ==========================================
# 2. EXPRESSIVE POLICY & CRITIC NETWORKS
# ==========================================

class FlowPolicy(eqx.Module):
    flow: masked_autoregressive_flow

    def __init__(self, action_dim: int, obs_dim: int, key: jtp.Key[jtp.Array, ""]) -> None:
        self.flow = masked_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros(action_dim)),
            cond_dim=obs_dim,
            nn_width=64,
            nn_depth=2,
            flow_layers=3
        )

    def get_action_and_log_prob(self, obs: jax.Array, key: jtp.Key[jtp.Array, ""]) -> tuple[jax.Array, jax.Array]:
        # 1. Sample raw unbounded actions from the flow
        raw_action: jax.Array = self.flow.sample(key, condition=obs)
        raw_log_prob: jax.Array = self.flow.log_prob(raw_action, condition=obs)

        # 2. Bound to MJX torque space via tanh
        action: jax.Array = jnp.tanh(raw_action)

        # 3. Change-of-variables adjustment
        tanh_correction: jax.Array = jnp.sum(jnp.log(1.0 - action**2 + 1e-6))
        log_prob: jax.Array = raw_log_prob - tanh_correction

        return action, log_prob

class CriticNetwork(eqx.Module):
    mlp1: eqx.nn.MLP
    mlp2: eqx.nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, key: jtp.Key[jtp.Array, ""]) -> None:
        k1, k2 = jax.random.split(key)
        in_dim: int = obs_dim + action_dim
        self.mlp1 = eqx.nn.MLP(in_dim, 1, width_size=256, depth=2, key=k1)
        self.mlp2 = eqx.nn.MLP(in_dim, 1, width_size=256, depth=2, key=k2)

    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        x: jax.Array = jnp.concatenate([obs, action], axis=-1)
        return self.mlp1(x).squeeze(-1), self.mlp2(x).squeeze(-1)

# ==========================================
# 3. GRADIENT UPDATES & OPTIMIZATION LOSSES
# ==========================================

@eqx.filter_value_and_grad
def critic_loss_fn(
    critic: CriticNetwork, 
    policy: FlowPolicy, 
    target_critic: CriticNetwork, 
    obs: jax.Array, 
    action: jax.Array, 
    reward: jax.Array, 
    next_obs: jax.Array, 
    done: jax.Array, 
    alpha: float, 
    gamma: float, 
    key: jtp.Key[jtp.Array, ""]
) -> jax.Array:
    q1, q2 = jax.vmap(critic)(obs, action)
    
    keys: jax.Array = jax.random.split(key, obs.shape[0])
    next_actions, next_log_probs = jax.vmap(policy.get_action_and_log_prob)(next_obs, keys)
    
    target_q1, target_q2 = jax.vmap(target_critic)(next_obs, next_actions)
    target_q: jax.Array = jnp.minimum(target_q1, target_q2) - alpha * next_log_probs
    
    y: jax.Array = reward + gamma * (1.0 - done) * target_q
    return jnp.mean((q1 - y) ** 2) + jnp.mean((q2 - y) ** 2)

@eqx.filter_value_and_grad
def policy_loss_fn(
    policy: FlowPolicy, 
    critic: CriticNetwork, 
    obs: jax.Array, 
    alpha: float, 
    key: jtp.Key[jtp.Array, ""]
) -> jax.Array:
    keys: jax.Array = jax.random.split(key, obs.shape[0])
    actions, log_probs = jax.vmap(policy.get_action_and_log_prob)(obs, keys)

    q1, q2 = jax.vmap(critic)(obs, actions)
    min_q: jax.Array = jnp.minimum(q1, q2)
    return jnp.mean(alpha * log_probs - min_q)

# ==========================================
# 4. UNIFIED ENGINE AND ROLLOUT STATE
# ==========================================

class TrainState(typing.NamedTuple):
    policy: FlowPolicy
    critic: CriticNetwork
    target_critic: CriticNetwork
    opt_state_p: optax.OptState
    opt_state_c: optax.OptState
    buffer: ReplayBuffer
    env_state: typing.Any  # Contains mutable brax/mjx environment state metadata
    current_obs: jax.Array

class MJXRolloutManager:
    env: envs.Env

    def __init__(self, env: envs.Env) -> None:
        self.env = env

    def collect_step(self, policy: FlowPolicy, env_state: typing.Any, key: jtp.Key[jtp.Array, ""]) -> tuple[typing.Any, TransitionBatch]:
        action_key, step_key = jax.random.split(key)
        action_keys: jax.Array = jax.random.split(action_key, env_state.obs.shape[0])

        actions, _ = jax.vmap(policy.get_action_and_log_prob)(env_state.obs, action_keys)
        next_env_state: typing.Any = self.env.step(env_state, actions)

        transition = TransitionBatch(
            obs=env_state.obs,
            action=actions,
            reward=next_env_state.reward,
            next_obs=next_env_state.obs,
            done=next_env_state.done.astype(jnp.float32)
        )
        return next_env_state, transition

# ==========================================
# 5. GLOBAL ENVIRONMENT STEP EXECUTOR
# ==========================================

def training_iteration(
    state: TrainState, 
    manager: MJXRolloutManager, 
    global_step: int, 
    key: jtp.Key[jtp.Array, ""], 
    config: dict[str, typing.Any], 
    opt_p: optax.GradientTransformation, 
    opt_c: optax.GradientTransformation
) -> tuple[TrainState, jax.Array]:
    step_key, sample_key, k_c, k_p = jax.random.split(key, 4)

    # Environment physics step execution
    next_env_state, transition = manager.collect_step(state.policy, state.env_state, step_key)
    new_buffer: ReplayBuffer = state.buffer.add(transition)

    def do_update(s: TrainState) -> tuple[FlowPolicy, CriticNetwork, CriticNetwork, optax.OptState, optax.OptState, jax.Array]:
        batch: TransitionBatch = s.buffer.sample(sample_key, config["batch_size"])

        # Optimize Critic
        c_loss, c_grads = critic_loss_fn(s.critic, s.policy, s.target_critic, batch.obs, batch.action, batch.reward, batch.next_obs, batch.done, config["alpha"], config["gamma"], k_c)
        c_updates, opt_state_c = opt_c.update(c_grads, s.opt_state_c, s.critic)
        critic: CriticNetwork = eqx.apply_updates(s.critic, c_updates)

        # Optimize Flow Policy
        p_loss, p_grads = policy_loss_fn(s.policy, critic, batch.obs, config["alpha"], k_p)
        p_updates, opt_state_p = opt_p.update(p_grads, s.opt_state_p, s.policy)
        policy: FlowPolicy = eqx.apply_updates(s.policy, p_updates)

        # Polyak Target Update
        target_critic: CriticNetwork = jax.tree_util.tree_map(
            lambda t, src: config["tau"] * src + (1.0 - config["tau"]) * t if eqx.is_array(t) else t,
            s.target_critic, critic
        )
        return policy, critic, target_critic, opt_state_p, opt_state_c, jnp.array([p_loss, c_loss])

    def no_update(s: TrainState) -> tuple[FlowPolicy, CriticNetwork, CriticNetwork, optax.OptState, optax.OptState, jax.Array]:
        return s.policy, s.critic, s.target_critic, s.opt_state_p, s.opt_state_c, jnp.zeros(2)

    policy, critic, target_critic, opt_state_p, opt_state_c, losses = jax.lax.cond(
        global_step > config["warmup_steps"], do_update, no_update, state
    )

    next_state = TrainState(
        policy=policy, critic=critic, target_critic=target_critic,
        opt_state_p=opt_state_p, opt_state_c=opt_state_c,
        buffer=new_buffer, env_state=next_env_state, current_obs=next_env_state.obs
    )
    return next_state, losses
