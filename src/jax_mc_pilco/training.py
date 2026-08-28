import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_mc_pilco.model_learning.flow_model import FlowDynamics

# from jax_mc_pilco.policy_learning.action_flows import FlowActor


def compute_loss(
    model: FlowDynamics, batch_s_curr: jax.Array, batch_action: jax.Array, batch_s_next: jax.Array
) -> jax.Array:
    # Vectorize the log_prob function over the batch dimension
    vmap_log_prob = jax.vmap(model.log_prob, in_axes=(0, 0, 0))

    # Calculate log probabilities for the entire batch
    log_probs = vmap_log_prob(batch_s_next, batch_s_curr, batch_action)

    # Minimize negative log-likelihood
    return -jnp.mean(log_probs)


# @eqx.filter_jit
def train_step(
    model: FlowDynamics,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict,
) -> tuple[FlowDynamics, optax.OptState, jax.Array]:
    """One step of training."""
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, batch["s_curr"], batch["action"], batch["s_next"])

    # Match the tree structure precisely using is_inexact_array
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


jax.config.update("jax_enable_x64", True)

env = gym.make("InvertedPendulum-v5")
env_test = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

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
key, subkey = jax.random.split(key)
state_space_model = FlowDynamics(key=subkey, state_dim=x.shape[0], action_dim=u.shape[0])

# 1. Prepare sequential transition data from your raw JAX arrays
# For a trajectory sequence: s_t -> a_t -> s_{t+1}
s_curr_data = states[:-1]
action_data = actions[:-1]
s_next_data = states[1:]
num_samples = s_curr_data.shape[0]

# 2. Initialize Model and Optimizer
model_key, train_key = jax.random.split(key)

model = FlowDynamics(key=model_key, state_dim=states.shape[-1], action_dim=actions.shape[-1])
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


# 3. Training Hyperparameters
epochs = 50
batch_size = 1024
num_batches = num_samples // batch_size

# 4. Main Training Loop
for epoch in range(epochs):
    # Shuffle data indices at the start of each epoch
    train_key, shuffle_key = jax.random.split(train_key)
    permutation = jax.random.permutation(shuffle_key, num_samples)

    epoch_loss = 0.0
    for i in range(num_batches):
        # Extract mini-batch indices
        batch_idx = permutation[i * batch_size : (i + 1) * batch_size]
        batch = {
            "s_curr": s_curr_data[batch_idx],
            "action": action_data[batch_idx],
            "s_next": s_next_data[batch_idx],
        }
        # Perform gradient update
        model, opt_state, loss_val = train_step(model, opt_state, optimizer, batch)
        epoch_loss += loss_val.item()

    print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {epoch_loss / num_batches:.4f}")
