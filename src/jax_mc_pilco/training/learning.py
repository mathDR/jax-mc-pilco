"""Methods to train Flows and generate data."""

import typing

import equinox as eqx
import gpjax as gpx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import numpy as np
import optax
from flowjax.train import fit_to_data
from scipy.stats import qmc

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor
from jax_mc_pilco.training.gprewards import build_sparse_sm_model, make_sparse_predictive_function

jax.config.update("jax_enable_x64", True)


def observation_to_qpos_qvel(
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts qpos and qvel from an 11-element InvertedDoublePendulum observation.
    """
    # 1. Reconstruct qpos (3 elements)
    cart_pos = obs[0]
    # np.arctan2 maps y (sin) and x (cos) to the range [-pi, pi]
    pole1_angle = np.arctan2(obs[1], obs[3])
    pole2_angle = np.arctan2(obs[2], obs[4])

    qpos = np.array([cart_pos, pole1_angle, pole2_angle])

    # 2. Reconstruct qvel (3 elements)
    qvel = np.array([obs[5], obs[6], obs[7]])

    return qpos, qvel


def generate_sobol_initial_states(
    num_states: int,
    dimensions: int,
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Generates low-discrepancy physical states for InvertedDoublePendulum-v4."""
    # 1. Dimensions: 1 cart pos, 2 link angles, 1 cart vel, 2 link angular vels = 6 dimensions
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    raw_samples = sampler.random(n=num_states)

    physical_states: np.ndarray = qmc.scale(raw_samples, l_bounds, u_bounds)
    return physical_states


def collect_mbrl_transitions(
    env: gym.Env,
    num_states: int,
    actions_per_state: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Forces the environment into a Sobol state, executes multiple action branches,
    and returns an MBRL transition dataset.
    """
    lb = np.array([-1.0, -0.6, -0.6, 0.8, 0.8, -5.0, -5.0, -5.0, -10.0])
    ub = np.array([1.0, 0.6, 0.6, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0])
    sobol_states = generate_sobol_initial_states(num_states, 9, lb, ub)

    # Storage arrays for MBRL training
    observations = []
    actions = []
    next_observations = []
    rewards = []

    # Unwrap to access MuJoCo mechanics directly
    raw_env = env.unwrapped

    for state in sobol_states:
        # Extract MuJoCo physics coordinates
        # qpos = [cart_x, theta1, theta2]
        # qvel = [cart_x_dot, theta1_dot, theta2_dot]
        qpos, qvel = observation_to_qpos_qvel(state)

        for _ in range(actions_per_state):
            # A. HARD-RESET the simulator to the exact same Sobol state
            env.reset()
            raw_env.set_state(qpos, qvel)

            # B. Get the correct state observation representation
            # Gym's double pendulum observation is usually a vector of sines, cosines, and vels
            obs = raw_env._get_obs()

            # C. Sample a random exploratory action
            action = env.action_space.sample()

            # D. Step the environment forward 1 timestep
            next_obs, reward, _, _, _ = env.step(action)

            # E. Store transition tuple
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)

    return (
        jnp.array(observations),
        jnp.array(actions),
        jnp.array(next_observations),
        jnp.array(rewards),
    )


def collect_experience(
    env: gym.Env,
    num_steps: int,
    key: jtp.Key[jtp.Array, ""],
    actor: FlowActor | None = None,
    exploration: bool = True,
    use_sobol: bool = False,
) -> tuple[jtp.Float, jtp.Int, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples data from a Gymnasium environment using standard Python/NumPy,
    then converts the accumulated trajectory into a JAX array.
    """
    max_steps = num_steps if exploration else 1

    actions: list = []
    states: list = []
    next_states: list = []
    rewards: list = []
    max_episode_length = 0
    total_reward = 0.0

    if use_sobol and exploration:
        states, actions, next_states, rewards = collect_mbrl_transitions(
            env,
            # num_states=2 ** int(jnp.log2(num_steps // 4)),
            num_states=2**10,
            actions_per_state=4,
        )
        total_reward = 0.0
        max_episode_length = 0
    else:
        while len(actions) < max_steps:
            curr_state, _ = env.reset()
            prev_state = curr_state  # no history yet; use s_0 as its own "previous" state

            s_curr_ep: list[np.ndarray] = []
            action_ep: list[np.ndarray] = []
            s_next_ep: list[np.ndarray] = []
            ep_rewards: list[np.ndarray] = []

            for _ in range(num_steps):
                key, ak = jax.random.split(key)
                action = (
                    env.action_space.sample()
                    if actor is None
                    else actor.sample_action(ak, jnp.array(prev_state), jnp.array(curr_state))
                )
                action_np = np.array(action)

                next_state, reward, terminated, truncated, _ = env.step(action_np)
                ep_rewards.append(float(reward))
                total_reward += float(reward)

                s_curr_ep.append(curr_state)
                action_ep.append(action_np)
                s_next_ep.append(next_state)

                prev_state, curr_state = curr_state, next_state
                if terminated or truncated:
                    break

            episode_length = len(action_ep)
            if episode_length > max_episode_length:
                max_episode_length = episode_length

            rewards.extend(ep_rewards)
            states.extend(s_curr_ep)
            next_states.extend(s_next_ep)
            actions.extend(action_ep)

    return (
        total_reward,
        max_episode_length,
        jnp.array(states),
        jnp.array(actions),
        jnp.array(next_states),
        jnp.array(rewards)[:, jnp.newaxis],
    )


def train_flow(
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    key: jtp.Key[jtp.Array, ""],
    *,
    flow: FlowDynamics | None = None,
    flow_layers: int = 4,
    learning_rate: float = 5e-3,
    max_patience: int = 25,
    max_epochs: int = 1000,
    batch_size: int = 256,
) -> tuple[FlowDynamics, dict[str, list]]:
    """
    Trains a model of the state distribution of the collected environmental
    data.  If first time calling it, initalizes the model.  If subsequent
    calls, it overrides the constraints of the flow with the updated data.
    """
    key, flow_key = jax.random.split(key)
    if flow is None:
        dynamics = FlowDynamics(
            key=flow_key,
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            state_low=jnp.min(states, axis=0),
            state_high=jnp.max(states, axis=0),
            flow_layers=flow_layers,
        )
    else:
        dynamics = FlowDynamics(
            key=flow_key,
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            state_low=jnp.min(states, axis=0),
            state_high=jnp.max(states, axis=0),
            flow_layers=flow_layers,
            base_flow=flow.base_flow,
        )
    # Build data for training flow

    context = jnp.concatenate([states, actions], axis=-1)

    key, train_key = jax.random.split(key)
    train_flow, losses = fit_to_data(
        key=train_key,
        dist=dynamics,
        data=(next_states, context),
        learning_rate=learning_rate,
        max_patience=max_patience,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    return train_flow, losses


def train_reward(
    states: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    key: jtp.Key[jtp.Array, ""],
) -> typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]]:
    """Train a Spectral Mixture Gaussian Process on the Rewards and return a Callable."""

    context = jnp.concatenate([states, actions], axis=-1)

    # --- Configuration ---
    batch_size = 512
    num_epochs = 200
    num_iters = (states.shape[0] // batch_size) * num_epochs

    # --- Build Sparse Infrastructure ---
    # Using 150 sparse inducing coordinates instead of all 10k inputs
    key, subkey = jax.random.split(key)
    svgp_posterior, dataset = build_sparse_sm_model(
        subkey,
        context,
        rewards,
        num_mixtures=4,
        num_inducing=150,
    )

    print(f"Beginning SVGP Mini-batched Optimization for {num_iters} iterations...")

    # Native GPJax stochastic fitting engine handle
    key, subkey = jax.random.split(key)
    opt_svgp, _ = gpx.fit(
        model=svgp_posterior,
        # we want want to minimize the *negative* ELBO
        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
        train_data=dataset,
        optim=optax.adamw(learning_rate=1e-2),
        num_iters=num_iters,
        key=subkey,
    )
    gp_predictor: typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]] = (
        make_sparse_predictive_function(opt_svgp, train_data=dataset)
    )
    return gp_predictor


def rollout(
    actor: FlowActor,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0: jax.Array,
    curr_s0: jax.Array,
    reward_fn: typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]],
    *,
    horizon: int = 50,
    discount: float = 0.99,
) -> jax.Array:

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        prev_s, curr_s, k = carry
        k, ak, sk = jax.random.split(k, 3)

        action = actor.sample_action(ak, prev_s, curr_s)  # already in [low, high]
        r = reward_fn(jnp.atleast_2d(jnp.concatenate([curr_s, action], axis=-1)))
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
    reward_fn: typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]],
) -> jax.Array:
    keys = jax.random.split(key, prev_s0_batch.shape[0])
    returns = jax.vmap(lambda k, ps0, cs0: rollout(actor, dynamics, k, ps0, cs0, reward_fn))(
        keys, prev_s0_batch, curr_s0_batch
    )
    return jnp.mean(returns)


@eqx.filter_jit
def loss_fn(
    params: FlowActor,
    static: FlowActor,
    dynamics: FlowDynamics,
    key: jax.Array,
    prev_s0_batch: jax.Array,
    curr_s0_batch: jax.Array,
    reward_fn: typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]],
) -> jax.Array:
    actor_ = eqx.combine(params, static)
    return -batched_return(actor_, dynamics, key, prev_s0_batch, curr_s0_batch, reward_fn)


def train_actor(
    actor: FlowActor,
    dynamics: FlowDynamics,
    states: jax.Array,
    key: jax.Array,
    optimizer: optax.GradientTransformation,
    reward_fn: typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]],
    num_train_steps: int = 500,
    num_init_states: int = 64,
) -> FlowActor:
    """Optimizes `actor` against the (frozen) `dynamics` model via
    differentiable rollouts, using (prev_state, curr_state) pairs drawn
    from `states` as initial conditions."""
    params, static = eqx.partition(actor, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    prev_states_pool = states[:-2]
    curr_states_pool = states[1:-1]

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
            params, static, dynamics, key, prev_s0_batch, curr_s0_batch, reward_fn
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

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
