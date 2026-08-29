import flowjax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flowjax.flows import MaskedAutoregressiveFlow
from flowjax.train import fit_to_data


def collect_experience(env_name: str, num_steps: int):
    """Samples data from a Gymnasium environment using standard Python/NumPy,
    then converts the accumulated trajectory into a JAX array.
    """
    env = gym.make(env_name)
    obs, info = env.reset()

    data_list = []

    for _ in range(num_steps):
        # Sample a random action (or use a policy network here)
        action = env.action_space.sample()

        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Construct the data point to model (e.g., state-action transitions)
        # Flattening ensures a 1D vector per timestep
        flat_obs = jnp.ravel(obs)
        flat_action = jnp.ravel(action)
        data_point = jnp.concatenate([flat_obs, flat_action])
        data_list.append(data_point)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    env.close()
    return jnp.stack(data_list)


def train_flow(data: jnp.ndarray, steps: int = 100):
    """Initializes a FlowJax flow and trains it to model the distribution
    of the collected environment data.
    """
    # 1. Define key and dimensions
    key = jax.random.PRNGKey(0)
    data_dim = data.shape[-1]

    # 2. Instantiate the Normalizing Flow
    key, flow_key = jax.random.split(key)
    flow = MaskedAutoregressiveFlow(
        key=flow_key,
        base_dist=flowjax.distributions.Normal(jnp.zeros(data_dim)),
        transformer=flowjax.bijections.RationalQuadraticSpline(knots=8, range_bound=3.0),
        flow_layers=4,
    )

    # 3. Configure the optimizer
    optimizer = optax.adam(learning_rate=1e-3)

    # 4. Train the flow using FlowJax's built-in fit_to_data function
    key, train_key = jax.random.split(key)
    flow, losses = fit_to_data(key=train_key, dist=flow, x=data, optimizer=optimizer, num_epochs=steps, batch_size=256)

    return flow, losses


if __name__ == "__main__":
    # Example execution using a standard classic control environment
    print("Collecting data from environment...")
    env_data = collect_experience("Pendulum-v1", num_steps=5000)
    print(f"Data collected with shape: {env_data.shape}")

    print("Training the flow model...")
    trained_flow, loss_history = train_flow(env_data, steps=10)
    print("Training complete!")
