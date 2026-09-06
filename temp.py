import jax
from brax import envs

print(f"JAX backend: {jax.devices()}")

# Try loading a standard Brax environment
env = envs.get_environment("ant")
state = env.reset(rng=jax.random.PRNGKey(0))
print("Brax environment successfully initialized!")
