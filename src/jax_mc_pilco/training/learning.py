# import gymnasium as gym
# import jax
# import jax.numpy as jnp


# def collect_experience(env_name: str, num_steps: int) -> jax.Array:
#     """Samples data from a Gymnasium environment using standard Python/NumPy,
#     then converts the accumulated trajectory into a JAX array.
#     """
#     env = gym.make(env_name)
#     obs, info = env.reset()

#     data_list = []

#     for _ in range(num_steps):
#         # Sample a random action (or use a policy network here)
#         action = env.action_space.sample()

#         # Step the environment
#         next_obs, reward, terminated, truncated, info = env.step(action)

#         # Construct the data point to model (e.g., state-action transitions)
#         # Flattening ensures a 1D vector per timestep
#         flat_obs = jnp.ravel(obs)
#         flat_action = jnp.ravel(action)
#         data_point = jnp.concatenate([flat_obs, flat_action])
#         data_list.append(data_point)

#         if terminated or truncated:
#             obs, info = env.reset()
#         else:
#             obs = next_obs

#     env.close()
#     return jnp.stack(data_list)


if __name__ == "__main__":
    # Example execution using a standard classic control environment
    print("Collecting data from environment...")
    # env_data = collect_experience("Pendulum-v1", num_steps=5000)
    # print(f"Data collected with shape: {env_data.shape}")

    print("Training the flow model...")
    # trained_flow, loss_history = train_flow(env_data, steps=10)
    print("Training complete!")
