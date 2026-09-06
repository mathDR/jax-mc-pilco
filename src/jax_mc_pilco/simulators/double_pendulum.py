import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import mujoco
from mujoco import mjx

# 1. Define the Double Pendulum MJCF string
DOUBLE_PENDULUM_XML = """
<mujoco model="inverted_double_pendulum">
    <compiler inertiafromgeom="true"/>
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
        <body name="cart" pos="0 0 0.05">
            <joint name="slider" type="slide" axis="1 0 0" range="-10 10"/>
            <geom name="cart_geom" type="box" size="0.2 0.1 0.05" rgba="0.8 0.1 0.1 1"/>
            <body name="pole1" pos="0 0 0">
                <joint name="hinge1" type="hinge" axis="0 1 0"/>
                <geom name="pole1_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.03" rgba="0 0.8 0 1"/>
                <body name="pole2" pos="0 0 0.6">
                    <joint name="hinge2" type="hinge" axis="0 1 0"/>
                    <geom name="pole2_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="0 0 0.8 1"/>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="slide_motor" joint="slider" ctrlrange="-20 20"/>
    </actuator>
</mujoco>
"""

# Compile and place MuJoCo model on the device
mj_model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
mjx_model = mjx.put_model(mj_model)


# 2. Example Equinox Policy Structure with jaxtyping
class MLPPolicy(eqx.Module):
    layers: list

    def __init__(self, key: jtp.Key[jtp.Array, ""]):
        # Observation size = 6 (qpos=3, qvel=3), Action size = 1 (motor torque)
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(6, 32, key=keys[0]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 1, key=keys[1]),
        ]

    def __call__(self, x: jtp.Float[jtp.Array, "6"]) -> jtp.Float[jtp.Array, "1"]:
        for layer in self.layers:
            x = layer(x)
        return x


# 3. Flexible Step Function with Strict Dimension Verification
def step_env(
    model: mjx.Model,
    data: mjx.Data,
    rng_key: jtp.Key[jtp.Array, ""],
    policy: eqx.Module | None = None,
) -> tuple[mjx.Data, jtp.Float[jtp.Array, "3"]]:
    """Evaluates a single state transition validating exact jtp.Array dimensions.

    - Observations match: 3 positions + 3 velocities = 6 dimensions.
    - Resulting qpos jtp.Array returns exactly 3 dimensions.
    """
    # Construct observation vector: [positions, velocities] -> 6 dimensions
    obs: jtp.Float[jtp.Array, " 6"] = jnp.concatenate([data.qpos, data.qvel])

    if policy is not None:
        action: jtp.Float[jtp.Array, " 1"] = policy(obs)
    else:
        ctrl_min: jtp.Float[jtp.Array, " 1"] = model.actuator_ctrlrange[:, 0]
        ctrl_max: jtp.Float[jtp.Array, " 1"] = model.actuator_ctrlrange[:, 1]
        action = jax.random.uniform(rng_key, (1,), minval=ctrl_min, maxval=ctrl_max)

    # Apply forces and advance the simulation graph
    data = data.replace(ctrl=action)
    next_data: mjx.Data = mjx.step(model, data)

    return next_data, next_data.qpos


# 4. Batched Rollout Pipeline Mapping over 'B' Parallel Environments
@eqx.filter_jit
def rollout_batch(
    model: mjx.Model,
    states: mjx.Data,
    keys: jtp.Key[jtp.Array, ""],  # Batched keys have a shape prefix, e.g., [B, 2]
    policy: eqx.Module | None = None,
) -> tuple[mjx.Data, jtp.Float[jtp.Array, "B 3"]]:
    """Applies vmap vectorization across an explicit batch dimension 'B'."""
    step_fn = lambda d, k: step_env(model, d, k, policy=policy)
    return jax.vmap(step_fn)(states, keys)


# ==========================================
# Execution Verification
# ==========================================
num_envs = 4
master_key = jax.random.PRNGKey(42)
init_key, policy_key, run_key = jax.random.split(master_key, 3)

# Setup initial batch of states
batched_data: mjx.Data = jax.vmap(lambda _: mjx.make_data(mjx_model))(jnp.zeros(num_envs))
env_keys: jtp.Key[jtp.Array, ""] = jax.random.split(run_key, num_envs)

# Case A: Execute using Random Actions fallback
next_states_rand, positions_rand = rollout_batch(mjx_model, batched_data, env_keys, policy=None)
print("Random Actions - Step positions:\n", positions_rand)

# Case B: Execute using a valid Equinox Neural Net
trained_policy = MLPPolicy(policy_key)
next_states_nn, positions_nn = rollout_batch(mjx_model, batched_data, env_keys, policy=trained_policy)
print("\nTrained Equinox Policy - Step positions:\n", positions_nn)
