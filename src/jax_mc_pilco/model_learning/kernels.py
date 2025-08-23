"""Bespoke Spectral Mixture Kernel."""

import jax
import jax.numpy as jnp

import tinygp


class SpectralMixture(tinygp.kernels.Kernel):
    weight: jax.Array
    scale: jax.Array
    freq: jax.Array

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(-2 * jnp.pi**2 * tau**2 / self.scale**2)
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=0,
            )
        )
