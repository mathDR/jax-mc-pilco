"""
End-to-end: collect data -> train FlowDynamics world model -> train FlowActor
by backpropagating expected reward through the (frozen) learned dynamics.
"""

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import numpy as np
import optax
from flowjax.bijections import Affine, Chain, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

jax.config.update("jax_enable_x64", True)


# ============================================================
# Modules
# ============================================================
class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model.
    Maps noise z ~ N(0, I) -> Delta State given the context [s_t, a_t]
    """

    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        state_dim: int,
        action_dim: int,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        cond_dim = state_dim + action_dim

        base_dist = MultivariateNormal(
            loc=jnp.zeros(state_dim, dtype=float),
            covariance=jnp.eye(state_dim, dtype=float),
        )

        self.flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

    def predict_next_state(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        s_curr: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Samples a structural residual transition delta and adds it to s_t."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = self.flow.sample(key, condition=context)
        return s_curr + delta_s

    def log_prob(
        self,
        s_next: jax.Array,
        s_curr: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state delta given the context."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = s_next - s_curr
        return self.flow.log_prob(delta_s, condition=context)


class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the context [s_{t-1}, s_t].
    Actions are squashed (Sigmoid + Affine) into [action_low, action_high]
    as part of the flow's bijection, so sample() and log_prob() are both
    automatically consistent with the bounded action space.
    """

    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        state_dim: int,
        action_dim: int,
        action_low: jax.Array,
        action_high: jax.Array,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        cond_dim = state_dim * 2

        base_dist = MultivariateNormal(
            loc=jnp.zeros(action_dim, dtype=float),
            covariance=jnp.eye(action_dim, dtype=float),
        )

        base_flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

        # Sigmoid: R -> (0, 1); then affine: (0, 1) -> (low, high)
        loc = action_low
        scale = action_high - action_low
        squash = Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=scale)])

        full_bijection = Chain([base_flow.bijection, squash])
        self.flow = Transformed(base_dist, full_bijection)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Samples a bounded action given the history context."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.sample(key, condition=context)

    def log_prob(
        self,
        action: jax.Array,
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood log p(a | s_prev, s_curr)."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.log_prob(action, condition=context)


# ============================================================
# 1. Environment + data collection
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
#     pos, angle = state[0], state[1]
#     alive_bonus = 1.0
#     angle_penalty = 10.0 * angle**2
#     pos_penalty = 0.01 * pos**2
#     ctrl_penalty = 0.001 * jnp.sum(action**2)
#     return alive_bonus - angle_penalty - pos_penalty - ctrl_penalty


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
optim = optax.adam(3e-4)


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
    static: FlowActor,
    opt_state: optax.OptState,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
) -> tuple[FlowActor, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(params, static, dynamics, key, prev_s0_batch, curr_s0_batch)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


def train_actor(
    actor: FlowActor,
    dynamics: FlowDynamics,
    states: jax.Array,
    key: jax.Array,
    num_train_steps: int = 500,
    num_init_states: int = 64,
) -> FlowActor:
    """Optimizes `actor` against the (frozen) `dynamics` model via
    differentiable rollouts, using (prev_state, curr_state) pairs drawn
    from `states` as initial conditions."""
    params, static = eqx.partition(actor, eqx.is_inexact_array)
    opt_state = optim.init(params)

    prev_states_pool = states[:-2]
    curr_states_pool = states[1:-1]

    for step in range(num_train_steps):
        key, batch_key, step_key = jax.random.split(key, 3)
        idx = jax.random.randint(batch_key, (num_init_states,), 0, prev_states_pool.shape[0])
        prev_s0_batch = prev_states_pool[idx]
        curr_s0_batch = curr_states_pool[idx]

        params, opt_state, loss = train_step(
            params, static, opt_state, dynamics, step_key, prev_s0_batch, curr_s0_batch
        )

        if step % 20 == 0:
            print(f"  [actor] step {step}, loss {loss:.4f}")

    return eqx.combine(params, static)

"""
End-to-end: collect data -> train FlowDynamics world model -> train FlowActor
by backpropagating expected reward through the (frozen) learned dynamics.
"""
import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import numpy as np
import optax
from flowjax.bijections import Affine, Chain, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow
from flowjax.train import fit_to_data

jax.config.update("jax_enable_x64", True)


# ============================================================
# Modules
# ============================================================
class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model.
    Maps noise z ~ N(0, I) -> Delta State given the context [s_t, a_t]
    """

    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        state_dim: int,
        action_dim: int,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        cond_dim = state_dim + action_dim

        base_dist = MultivariateNormal(
            loc=jnp.zeros(state_dim, dtype=float),
            covariance=jnp.eye(state_dim, dtype=float),
        )

        self.flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

    def predict_next_state(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        s_curr: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Samples a structural residual transition delta and adds it to s_t."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = self.flow.sample(key, condition=context)
        return s_curr + delta_s

    def log_prob(
        self,
        s_next: jax.Array,
        s_curr: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state delta given the context."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = s_next - s_curr
        return self.flow.log_prob(delta_s, condition=context)


class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the context [s_{t-1}, s_t].
    Actions are squashed (Sigmoid + Affine) into [action_low, action_high]
    as part of the flow's bijection, so sample() and log_prob() are both
    automatically consistent with the bounded action space.
    """

    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        state_dim: int,
        action_dim: int,
        action_low: jax.Array,
        action_high: jax.Array,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        cond_dim = state_dim * 2

        base_dist = MultivariateNormal(
            loc=jnp.zeros(action_dim, dtype=float),
            covariance=jnp.eye(action_dim, dtype=float),
        )

        base_flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

        # Sigmoid: R -> (0, 1); then affine: (0, 1) -> (low, high)
        loc = action_low
        scale = action_high - action_low
        squash = Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=scale)])

        full_bijection = Chain([base_flow.bijection, squash])
        self.flow = Transformed(base_dist, full_bijection)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Samples a bounded action given the history context."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.sample(key, condition=context)

    def log_prob(
        self,
        action: jax.Array,
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood log p(a | s_prev, s_curr)."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.log_prob(action, condition=context)


# ============================================================
# 1. Environment + data collection
# ============================================================
env = gym.make("InvertedDoublePendulum-v5")

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
# 4. Reward = the real InvertedDoublePendulum-v5 reward, computed directly
# from the predicted observation (no hand-designed surrogate needed --
# unlike InvertedPendulum-v5's flat +1/step, this env's real reward is
# already densely shaped).
# ============================================================
# Observation layout (9-dim): [x_cart, sin(t1), sin(t2), cos(t1), cos(t2),
#                               v_x, v1, v2, qfrc_constraint]
POLE1_LEN = 0.6
POLE2_LEN = 0.6


def reward_fn(state: jax.Array, action: jax.Array) -> jax.Array:
    x_cart = state[0]
    sin1, sin2 = state[1], state[2]
    cos1, cos2 = state[3], state[4]
    v1, v2 = state[6], state[7]

    # sin/cos(theta1 + theta2) via angle-addition identities, using only
    # the sin/cos values already in the observation (avoids recovering a
    # raw angle and re-applying jnp.sin/jnp.cos).
    sin12 = sin1 * cos2 + cos1 * sin2
    cos12 = cos1 * cos2 - sin1 * sin2

    x_tip = x_cart + POLE1_LEN * sin1 + POLE2_LEN * sin12
    y_tip = POLE1_LEN * cos1 + POLE2_LEN * cos12

    distance_penalty = 0.01 * x_tip**2 + (y_tip - 2.0) ** 2
    velocity_penalty = 1e-3 * v1**2 + 5e-3 * v2**2

    # The real env's alive_bonus is a hard 10/0 based on y_tip > 1 (episode
    # terminates otherwise). A hard indicator gives zero gradient almost
    # everywhere, so use a smooth sigmoid gate instead -- steep enough to
    # closely track the real cutoff while keeping a useful gradient near
    # the failure boundary, which is exactly where the actor needs signal.
    alive_gate = jax.nn.sigmoid(50.0 * (y_tip - 1.0))
    alive_bonus = 10.0 * alive_gate

    return alive_bonus - distance_penalty - velocity_penalty


# ============================================================
# 5. Differentiable rollout through the learned dynamics model
# ============================================================
# Carry (prev_state, curr_state) since the actor conditions on both.
# horizon is intentionally short: with an autoregressively-sampled dynamics
# model, small per-step errors compound multiplicatively, so a long horizon
# gives the actor's optimizer more room to exploit model inaccuracy before
# a single gradient step is taken. Widen this once the model/actor are
# jointly stable.
horizon = 20
discount = 0.99

# Loose physical bounds on InvertedDoublePendulum-v5's 9-dim observation
# ([x_cart, sin1, sin2, cos1, cos2, v_x, v1, v2, qfrc_constraint]), used
# only to clip the *rolled-out* (model-predicted) state so a single bad
# sampled delta_s can't compound into an unbounded trajectory over the
# scan. sin/cos components are clipped to their true range [-1, 1] (the
# dynamics model has no reason to know this a priori); other components
# use generous bounds that only catch genuine divergence.
STATE_CLIP_LOW = jnp.array([-5.0, -1.0, -1.0, -1.0, -1.0, -20.0, -20.0, -20.0, -50.0])
STATE_CLIP_HIGH = jnp.array([5.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 50.0])


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
        next_s = jnp.clip(next_s, STATE_CLIP_LOW, STATE_CLIP_HIGH)

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
    returns = jax.vmap(
        lambda k, ps0, cs0: rollout(actor, dynamics, k, ps0, cs0)
    )(keys, prev_s0_batch, curr_s0_batch)
    return jnp.mean(returns)


# ============================================================
# 6. Optimize the actor against -expected return
# ============================================================
# Clip gradients by global norm before the Adam step: if one rollout in a
# batch produces an unusually large penalty (e.g. from a still-imperfect
# region of the dynamics model), its gradient shouldn't be allowed to
# dominate the update and drag the actor toward that instability.
optim = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(3e-4),
)


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
    static: FlowActor,
    opt_state: optax.OptState,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
) -> tuple[FlowActor, optax.OptState, jax.Array]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params, static, dynamics, key, prev_s0_batch, curr_s0_batch
    )
    updates, opt_state = optim.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


def train_actor(
    actor: FlowActor,
    dynamics: FlowDynamics,
    states: jax.Array,
    key: jax.Array,
    num_train_steps: int = 500,
    num_init_states: int = 64,
) -> FlowActor:
    """Optimizes `actor` against the (frozen) `dynamics` model via
    differentiable rollouts, using (prev_state, curr_state) pairs drawn
    from `states` as initial conditions."""
    params, static = eqx.partition(actor, eqx.is_inexact_array)
    opt_state = optim.init(params)

    prev_states_pool = states[:-2]
    curr_states_pool = states[1:-1]

    for step in range(num_train_steps):
        key, batch_key, step_key = jax.random.split(key, 3)
        idx = jax.random.randint(batch_key, (num_init_states,), 0, prev_states_pool.shape[0])
        prev_s0_batch = prev_states_pool[idx]
        curr_s0_batch = curr_states_pool[idx]

        params, opt_state, loss = train_step(
            params, static, opt_state, dynamics, step_key, prev_s0_batch, curr_s0_batch
        )

        if step % 20 == 0:
            print(f"  [actor] step {step}, loss {loss:.4f}")

    return eqx.combine(params, static)


def train_dynamics(
    dynamics: FlowDynamics,
    s_curr_buf: jax.Array,
    action_buf: jax.Array,
    s_next_buf: jax.Array,
    key: jax.Array,
) -> FlowDynamics:
    """Refits `dynamics.flow` on explicit (s_curr, action, s_next) transition
    triplets, warm-starting from the current dynamics parameters.

    Using explicit triplets (rather than flat concatenated state/action
    arrays sliced with [:-1]/[1:]) avoids fabricating a bogus transition at
    the boundary between two different episodes when the buffer is built
    up from multiple rollouts.
    """
    context = jnp.concatenate([s_curr_buf, action_buf], axis=-1)
    delta_s = s_next_buf - s_curr_buf

    fit_inner_flow, losses = fit_to_data(
        key=key,
        dist=dynamics.flow,
        data=(delta_s, context),
        learning_rate=5e-3,
        max_patience=25,
        max_epochs=1000,
    )
    print(f"  [dynamics] refit complete, final val loss {losses['val'][-1]:.4f}")
    return eqx.tree_at(lambda m: m.flow, dynamics, fit_inner_flow)


def diagnose_model_reality_gap(
    dynamics: FlowDynamics,
    s_curr_ep: jax.Array,
    action_ep: jax.Array,
    s_next_ep: jax.Array,
    key: jax.Array,
    num_samples: int = 20,
) -> None:
    """Prints, for each real transition in the episode, the model's
    log_prob of the real outcome and the discrepancy between a model
    *sample* and what actually happened. Large negative log_prob or large
    sample discrepancy on the very states the actor's rollout visited is
    direct evidence the actor is exploiting model error rather than
    learning real stabilizing behavior."""
    for i in range(s_curr_ep.shape[0]):
        s_c, a, s_n = s_curr_ep[i], action_ep[i], s_next_ep[i]
        lp = dynamics.log_prob(s_n, s_c, a)

        keys = jax.random.split(key, num_samples + 1)
        key = keys[0]
        sampled_next = jax.vmap(lambda k: dynamics.predict_next_state(k, s_c, a))(keys[1:])
        mean_sampled_next = jnp.mean(sampled_next, axis=0)
        real_delta = s_n - s_c
        model_mean_delta = mean_sampled_next - s_c

        print(
            f"  step {i}: log_prob(real s_next)={lp:.2f} | "
            f"real delta={np.array(real_delta)} | "
            f"model mean delta={np.array(model_mean_delta)}"
        )


# ============================================================
# 7. Real-environment evaluation of the trained actor
# ============================================================
def evaluate_actor_in_env(
    actor: FlowActor,
    env: gym.Env,
    key: jax.Array,
    max_steps: int = 1000,
) -> tuple[float, int, jax.Array, jax.Array, jax.Array]:
    """Rolls the actor out in the *real* environment (no learned dynamics).

    Returns (total_reward, episode_length, s_curr_ep, action_ep, s_next_ep)
    where the last three are explicit, self-contained transition triplets
    from this single episode -- safe to append directly to a multi-episode
    dynamics buffer without any boundary-alignment bugs.
    """
    curr_state, _ = env.reset()
    prev_state = curr_state  # no history yet; use s_0 as its own "previous" state

    s_curr_ep: list[np.ndarray] = []
    action_ep: list[np.ndarray] = []
    s_next_ep: list[np.ndarray] = []

    total_reward = 0.0
    for _ in range(max_steps):
        key, ak = jax.random.split(key)
        action = actor.sample_action(ak, jnp.array(prev_state), jnp.array(curr_state))
        action_np = np.array(action)

        next_state, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)

        s_curr_ep.append(curr_state)
        action_ep.append(action_np)
        s_next_ep.append(next_state)

        prev_state, curr_state = curr_state, next_state
        if terminated or truncated:
            break

    episode_length = len(action_ep)
    return (
        total_reward,
        episode_length,
        jnp.array(s_curr_ep),
        jnp.array(action_ep),
        jnp.array(s_next_ep),
    )


# ============================================================
# 8. Dyna-style outer loop: train -> evaluate -> augment -> repeat
# ============================================================
SUCCESS_LENGTH = 950  # out of max_episode_steps=1000 for InvertedPendulum-v5
MAX_OUTER_ITERS = 10

# Initial transition buffer, built as explicit (s_curr, action, s_next)
# triplets from the random-action collection in step 1. This collection
# was one continuous run (no resets), so states[:-1]/actions[:-1]/states[1:]
# is valid here -- it's only *cross-episode* concatenation that needs the
# triplet form to avoid boundary artifacts.
s_curr_buf = states[:-1]
action_buf = actions[:-1]
s_next_buf = states[1:]

# `buf_states` is used only to sample (prev_state, curr_state) starting
# pairs for the actor's rollout initial conditions -- an occasional pair
# spanning two different episodes here is a harmless approximation (it
# just gives the rollout a slightly less realistic starting point), unlike
# in the dynamics buffer where a fabricated transition is actually wrong
# training data.
buf_states_for_actor_init = states

for outer_iter in range(MAX_OUTER_ITERS):
    print(f"\n=== Outer iteration {outer_iter} ===")

    key, dyn_key, actor_key, eval_key = jax.random.split(key, 4)

    print("Refitting dynamics model on full transition buffer...")
    dynamics = train_dynamics(dynamics, s_curr_buf, action_buf, s_next_buf, dyn_key)

    print("Training actor against updated dynamics model...")
    actor = train_actor(
        actor, dynamics, buf_states_for_actor_init, actor_key, num_train_steps=500
    )

    print("Evaluating actor in the real environment...")
    total_reward, episode_length, s_curr_ep, action_ep, s_next_ep = evaluate_actor_in_env(
        actor, env, eval_key
    )
    print(f"Real-env return: {total_reward:.1f}, episode length: {episode_length}")

    print("Model-reality gap on this episode's real transitions:")
    diagnose_model_reality_gap(dynamics, s_curr_ep, action_ep, s_next_ep, key=jax.random.fold_in(key, outer_iter))

    # Fold the on-policy transitions back into the buffers regardless of
    # success/failure -- this is the data that corrects the dynamics model
    # in the state-action regions the actor actually visits. Each episode's
    # triplets are self-contained, so concatenation across episodes never
    # fabricates a transition.
    s_curr_buf = jnp.concatenate([s_curr_buf, s_curr_ep], axis=0)
    action_buf = jnp.concatenate([action_buf, action_ep], axis=0)
    s_next_buf = jnp.concatenate([s_next_buf, s_next_ep], axis=0)
    buf_states_for_actor_init = jnp.concatenate([buf_states_for_actor_init, s_curr_ep], axis=0)

    if episode_length >= SUCCESS_LENGTH:
        print(f"Success: balanced for {episode_length} steps. Stopping.")
        break
else:
    print("Reached MAX_OUTER_ITERS without success; consider more iterations, "
          "a longer rollout horizon, or a stronger reward surrogate.")

def train_dynamics(
    dynamics: FlowDynamics,
    s_curr_buf: jax.Array,
    action_buf: jax.Array,
    s_next_buf: jax.Array,
    key: jax.Array,
) -> FlowDynamics:
    """Refits `dynamics.flow` on explicit (s_curr, action, s_next) transition
    triplets, warm-starting from the current dynamics parameters.

    Using explicit triplets (rather than flat concatenated state/action
    arrays sliced with [:-1]/[1:]) avoids fabricating a bogus transition at
    the boundary between two different episodes when the buffer is built
    up from multiple rollouts.
    """
    context = jnp.concatenate([s_curr_buf, action_buf], axis=-1)
    delta_s = s_next_buf - s_curr_buf

    fit_inner_flow, losses = fit_to_data(
        key=key,
        dist=dynamics.flow,
        data=(delta_s, context),
        learning_rate=5e-3,
        max_patience=25,
        max_epochs=1000,
    )
    print(f"  [dynamics] refit complete, final val loss {losses['val'][-1]:.4f}")
    return eqx.tree_at(lambda m: m.flow, dynamics, fit_inner_flow)


# ============================================================
# 7. Real-environment evaluation of the trained actor
# ============================================================
def evaluate_actor_in_env(
    actor: FlowActor,
    env: gym.Env,
    key: jax.Array,
    max_steps: int = 1000,
) -> tuple[float, int, jax.Array, jax.Array, jax.Array]:
    """Rolls the actor out in the *real* environment (no learned dynamics).

    Returns (total_reward, episode_length, s_curr_ep, action_ep, s_next_ep)
    where the last three are explicit, self-contained transition triplets
    from this single episode -- safe to append directly to a multi-episode
    dynamics buffer without any boundary-alignment bugs.
    """
    curr_state, _ = env.reset()
    prev_state = curr_state  # no history yet; use s_0 as its own "previous" state

    s_curr_ep: list[np.ndarray] = []
    action_ep: list[np.ndarray] = []
    s_next_ep: list[np.ndarray] = []

    total_reward = 0.0
    for _ in range(max_steps):
        key, ak = jax.random.split(key)
        action = actor.sample_action(ak, jnp.array(prev_state), jnp.array(curr_state))
        action_np = np.array(action)

        next_state, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)

        s_curr_ep.append(curr_state)
        action_ep.append(action_np)
        s_next_ep.append(next_state)

        prev_state, curr_state = curr_state, next_state
        if terminated or truncated:
            break

    episode_length = len(action_ep)
    return (
        total_reward,
        episode_length,
        jnp.array(s_curr_ep),
        jnp.array(action_ep),
        jnp.array(s_next_ep),
    )


# ============================================================
# 8. Dyna-style outer loop: train -> evaluate -> augment -> repeat
# ============================================================
SUCCESS_LENGTH = 950  # out of max_episode_steps=1000 for InvertedPendulum-v5
MAX_OUTER_ITERS = 10

# Initial transition buffer, built as explicit (s_curr, action, s_next)
# triplets from the random-action collection in step 1. This collection
# was one continuous run (no resets), so states[:-1]/actions[:-1]/states[1:]
# is valid here -- it's only *cross-episode* concatenation that needs the
# triplet form to avoid boundary artifacts.
s_curr_buf = states[:-1]
action_buf = actions[:-1]
s_next_buf = states[1:]

# `buf_states` is used only to sample (prev_state, curr_state) starting
# pairs for the actor's rollout initial conditions -- an occasional pair
# spanning two different episodes here is a harmless approximation (it
# just gives the rollout a slightly less realistic starting point), unlike
# in the dynamics buffer where a fabricated transition is actually wrong
# training data.
buf_states_for_actor_init = states

for outer_iter in range(MAX_OUTER_ITERS):
    print(f"\n=== Outer iteration {outer_iter} ===")

    key, dyn_key, actor_key, eval_key = jax.random.split(key, 4)

    print("Refitting dynamics model on full transition buffer...")
    dynamics = train_dynamics(dynamics, s_curr_buf, action_buf, s_next_buf, dyn_key)

    print("Training actor against updated dynamics model...")
    actor = train_actor(actor, dynamics, buf_states_for_actor_init, actor_key, num_train_steps=500)

    print("Evaluating actor in the real environment...")
    total_reward, episode_length, s_curr_ep, action_ep, s_next_ep = evaluate_actor_in_env(actor, env, eval_key)
    print(f"Real-env return: {total_reward:.1f}, episode length: {episode_length}")

    # Fold the on-policy transitions back into the buffers regardless of
    # success/failure -- this is the data that corrects the dynamics model
    # in the state-action regions the actor actually visits. Each episode's
    # triplets are self-contained, so concatenation across episodes never
    # fabricates a transition.
    s_curr_buf = jnp.concatenate([s_curr_buf, s_curr_ep], axis=0)
    action_buf = jnp.concatenate([action_buf, action_ep], axis=0)
    s_next_buf = jnp.concatenate([s_next_buf, s_next_ep], axis=0)
    buf_states_for_actor_init = jnp.concatenate([buf_states_for_actor_init, s_curr_ep], axis=0)

    if episode_length >= SUCCESS_LENGTH:
        print(f"Success: balanced for {episode_length} steps. Stopping.")
        break
else:
    print(
        "Reached MAX_OUTER_ITERS without success; consider more iterations, "
        "a longer rollout horizon, or a stronger reward surrogate."
    )
