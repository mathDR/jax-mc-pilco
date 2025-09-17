"""
Module for the t-SVGP model rewritten in JAX, Equinox, and CheX.
"""

import abc
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import chex
import equinox as eqx

# Assuming these are the JAX-rewritten versions from previous steps
from src.sites import DenseSites
from src.util import (
    conditional_from_precision_sites,
    gradient_transformation_mean_var_to_expectation,
    posterior_from_dense_site,
)

# Type aliases for clarity
InputData = chex.Array
RegressionData = Tuple[InputData, InputData]
MeanAndVariance = Tuple[chex.Array, chex.Array]

# Set JAX to use 64-bit floats for numerical stability
jax.config.update("jax_enable_x64", True)


class base_SVGP(eqx.Module, abc.ABC):
    """
    Modified gpflow.svgp.SVGP class to accommodate
    for different paramaterization of q(u)
    """
    kernel: eqx.Module
    likelihood: eqx.Module
    inducing_variable: eqx.Module
    mean_function: Optional[eqx.Module]
    num_latent_gps: int = eqx.field(static=True)
    num_data: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        kernel: eqx.Module,
        likelihood: eqx.Module,
        inducing_variable: eqx.Module,
        *,
        mean_function: Optional[eqx.Module] = None,
        num_latent_gps: int = 1,
        num_data: Optional[int] = None,
    ):
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.mean_function = mean_function
        self.num_latent_gps = num_latent_gps
        self.num_data = num_data

    @abc.abstractmethod
    def get_mean_chol_cov_inducing_posterior(self) -> Tuple[chex.Array, chex.Array]:
        """Returns the mean and cholesky factor of the covariance matrix of q(u)"""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def prior_kl(self) -> chex.Array:
        """Returns the KL divergence KL[q(u)|p(u)]"""
        # The GPflow Kullback-Leiblers are not directly available. We can rewrite the
        # core logic for the prior KL divergence.
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        Kuu = self.kernel.K_ff(self.inducing_variable.Z)
        
        # Calculate KL(q(u) || p(u)) = 0.5 * (log|Kuu| - log|S| + tr(Kuu⁻¹S) + μᵀKuu⁻¹μ - M)
        # We assume q_sqrt is the cholesky of S.
        
        S_sqrt = q_sqrt
        S = jnp.matmul(S_sqrt, S_sqrt.T)
        
        Kuu_chol = jnp.linalg.cholesky(Kuu + jnp.eye(Kuu.shape[0]) * 1e-6)
        
        logdet_Kuu = jnp.sum(jnp.