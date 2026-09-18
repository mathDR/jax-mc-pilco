"""Generic Critic Model for learning a dreamer like model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp


class DreamerCritic(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, latent_dim: int, key: jtp.Key[jtp.Array, ""]):
        self.net = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=1,
            width_size=256,
            depth=3,
            activation=jax.nn.elu,
            key=key,
        )

    def __call__(self, latent: jax.Array) -> jax.Array:
        # Returns a scalar value for the given latent state representation
        return jnp.squeeze(self.net(latent), axis=-1)
