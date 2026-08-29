"""
End-to-end: collect data -> train FlowDynamics world model -> train FlowActor
by backpropagating expected reward through the (frozen) learned dynamics.
"""

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flowjax.train import fit_to_data

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor

jax.config.update("jax_enable_x64", True)


# ============================================================
# Environment + data collection
# ============================================================
env = gym.make("InvertedDoublePendulum-v5")  # gym.make("InvertedPendulum-v5")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

act_low = jnp.array(env.action_space.low, dtype=float)
act_high = jnp.array(env.action_space.high, dtype=float)

x, _ = env.reset()
replay_states = [x]

u = env.action_space.sample()
replay_actions = [u]

for _ in range(10_000):
    z = env.step(np.array(u))
    x = z[0]
    replay_states.append(x)
    u = env.action_space.sample()
    replay_actions.append(u)

states = jnp.array(replay_states)
actions = jnp.array(replay_actions)

key = jax.random.key(seed=4)
key, dyn_key, dyn_fit_key = jax.random.split(key, 3)


# ============================================================
# 2. Train FlowDynamics (world model) -- frozen afterwards
# ============================================================
dynamics = FlowDynamics(
    key=dyn_key,
    state_dim=state_dim,
    action_dim=action_dim,
    flow_layers=4,
)

s_curr_data = states[:-1]
action_data = actions[:-1]
s_next_data = states[1:]

context = jnp.concatenate([s_curr_data, action_data], axis=-1)
delta_s = s_next_data - s_curr_data

fit_inner_flow, losses = fit_to_data(
    key=dyn_fit_key,
    dist=dynamics.flow,
    data=(delta_s, context),
    learning_rate=5e-3,
    max_patience=25,
    max_epochs=1000,
)

dynamics = eqx.tree_at(lambda m: m.flow, dynamics, fit_inner_flow)
print("Dynamics model fit complete. Final loss:", losses["val"][-1])


# ============================================================
# 3. Instantiate the actor
# ============================================================
key, actor_key = jax.random.split(key)

actor = FlowActor(
    key=actor_key,
    state_dim=state_dim,
    action_dim=action_dim,
    action_low=act_low,
    action_high=act_high,
    flow_layers=4,
)


# ============================================================
# 4. Differentiable reward surrogate
# ============================================================
# InvertedPendulum-v5's real reward is a constant +1/step; termination is
# what actually matters (pole angle exceeding a threshold), so we penalize
# the states that drive termination to get a useful gradient signal.
# def reward_fn(state: jax.Array, action: jax.Array) -> jax.Array:
# pos, angle = state[0], state[1]
# alive_bonus = 1.0
# angle_penalty = 10.0 * angle**2
# pos_penalty = 0.01 * pos**2
# ctrl_penalty = 0.001 * jnp.sum(action**2)
# return alive_bonus - angle_penalty - pos_penalty - ctrl_penalty


def reward_fn(state: jax.Array, action: jax.Array) -> jax.Array:
    # Gym observation provides sin/cos elements and velocities
    # Explicit geometry is usually required for the exact tip position (x, y),
    # but if calculating directly from the physical/simulation state array:

    # Extract angular velocities
    omega1 = state[6]
    omega2 = state[7]

    # Calculate tip coordinates using standard double pendulum link lengths (l1=1, l2=1)
    # state[0] = cart_x
    # angles extracted via atan2 on sin/cos pairs if needed, or forward kinematics:
    # For Gymnasium MuJoCo xml lengths:
    s1, s2 = state[1], state[2]
    c1, c2 = state[3], state[4]

    # X and Y positions of the second tip relative to the world
    tip_x = state[0] + s1 + s2
    tip_y = c1 + c2  # Center height offset handled by (y - 2) target tracking

    # Reward component calculations matching native code
    alive_bonus = 10.0
    dist_penalty = 0.01 * (tip_x**2) + (tip_y - 2.0) ** 2
    vel_penalty = 1e-3 * (omega1**2) + 5e-3 * (omega2**2)

    return alive_bonus - dist_penalty - vel_penalty


# ============================================================
# 5. Differentiable rollout through the learned dynamics model
# ============================================================
# Carry (prev_state, curr_state) since the actor conditions on both.
horizon = 50
discount = 0.99


def rollout(
    actor: FlowActor,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0: jax.Array,
    curr_s0: jax.Array,
) -> jax.Array:
    Carry = tuple[jax.Array, jax.Array, jax.Array]

    def step(carry: Carry, _: None) -> tuple[Carry, jax.Array]:
        prev_s, curr_s, k = carry
        k, ak, sk = jax.random.split(k, 3)

        action = actor.sample_action(ak, prev_s, curr_s)  # already in [low, high]
        r = reward_fn(curr_s, action)
        next_s = dynamics.predict_next_state(sk, curr_s, action)

        return (curr_s, next_s, k), r

    (_, _, _), rewards = jax.lax.scan(step, (prev_s0, curr_s0, key), None, length=horizon)
    discounts = discount ** jnp.arange(horizon)
    return jnp.sum(discounts * rewards)


def batched_return(
    actor: FlowActor,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
) -> jax.Array:
    keys = jax.random.split(key, prev_s0_batch.shape[0])
    returns = jax.vmap(lambda k, ps0, cs0: rollout(actor, dynamics, k, ps0, cs0))(keys, prev_s0_batch, curr_s0_batch)
    return jnp.mean(returns)


# ============================================================
# 6. Optimize the actor against -expected return
# ============================================================
params, static = eqx.partition(actor, eqx.is_inexact_array)
optim = optax.adam(3e-4)
opt_state = optim.init(params)


@eqx.filter_jit
def loss_fn(
    params: FlowActor,
    static: FlowActor,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
) -> jax.Array:
    actor_ = eqx.combine(params, static)
    return -batched_return(actor_, dynamics, key, prev_s0_batch, curr_s0_batch)


@eqx.filter_jit
def train_step(
    params: FlowActor,
    opt_state: optax.OptState,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
) -> tuple[FlowActor, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(params, static, dynamics, key, prev_s0_batch, curr_s0_batch)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


# (prev_state, curr_state) pairs drawn from the real replay buffer as
# initial conditions for rollouts.
prev_states_pool = states[:-2]
curr_states_pool = states[1:-1]

num_init_states = 64
num_train_steps = 500

for step in range(num_train_steps):
    key, batch_key, step_key = jax.random.split(key, 3)
    idx = jax.random.randint(batch_key, (num_init_states,), 0, prev_states_pool.shape[0])
    prev_s0_batch = prev_states_pool[idx]
    curr_s0_batch = curr_states_pool[idx]

    params, opt_state, loss = train_step(params, opt_state, step_key, prev_s0_batch, curr_s0_batch)

    if step % 20 == 0:
        print(step, loss)

actor = eqx.combine(params, static)
breakpoint()
