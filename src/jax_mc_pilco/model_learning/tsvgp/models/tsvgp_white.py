"""
Module for the t-SVGP model with whitened parameterization, rewritten for JAX.
"""
import abc
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import chex
import equinox as eqx
import optax

# Import the rewritten utility and module classes
from src.sites import DenseSites
from src.util import (
    conditional_from_precision_sites_white,
    gradient_transformation_mean_var_to_expectation,
    kl_from_precision_sites_white,
    posterior_from_dense_site_white,
)

# Type aliases for clarity
InputData = chex.Array
RegressionData = Tuple[InputData, InputData]
MeanAndVariance = Tuple[chex.Array, chex.Array]

# Set JAX to use 64-bit floats for numerical stability
jax.config.update("jax_enable_x64", True)


class base_SVGP(eqx.Module, abc.ABC):
    """
    Base SVGP class for JAX, accommodating different q(u) parameterizations.
    """
    kernel: eqx.Module
    likelihood: eqx.Module
    inducing_variable: eqx.Module
    mean_function: Optional[eqx.Module]
    num_latent_gps: int = eqx.field(static=True)
    num_data: Optional[int] = eqx.field(static=True)
    whiten: bool = eqx.field(static=True)

    def __init__(
        self,
        kernel: eqx.Module,
        likelihood: eqx.Module,
        inducing_variable: eqx.Module,
        *,
        mean_function: Optional[eqx.Module] = None,
        num_latent_gps: int = 1,
        num_data: Optional[int] = None,
        whiten: bool = True,
    ):
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.mean_function = mean_function
        self.num_latent_gps = num_latent_gps
        self.num_data = num_data
        self.whiten = whiten

    @abc.abstractmethod
    def get_mean_chol_cov_inducing_posterior(self) -> Tuple[chex.Array, chex.Array]:
        """Returns the mean and cholesky factor of the covariance matrix of q(u)"""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def prior_kl(self) -> chex.Array:
        """Returns the KL divergence KL[q(u)|p(u)]"""
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        Kuu = self.kernel.K_ff(self.inducing_variable.Z)
        
        # Calculate KL(q(u) || p(u))
        S_sqrt = q_sqrt
        S = jnp.matmul(S_sqrt, S_sqrt.T)
        
        Kuu_chol = jnp.linalg.cholesky(Kuu + jnp.eye(Kuu.shape[0]) * 1e-6)
        
        logdet_Kuu = jnp.sum(jnp.log(jnp.diag(Kuu_chol)))
        logdet_S = jnp.sum(jnp.log(jnp.diag(S_sqrt)))
        
        tr_term = jnp.trace(jnp.linalg.solve(Kuu, S))
        
        maha_term = jnp.sum(q_mu * jnp.linalg.solve(Kuu, q_mu))
        
        M = Kuu.shape[0]
        
        return 0.5 * (
            2 * (logdet_Kuu - logdet_S) + tr_term + maha_term - M
        )

    @partial(jax.jit, static_argnums=(0,))
    def elbo(self, data: RegressionData) -> chex.Array:
        """
        The variational lower bound
        :param data: input data (X, Y)
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)

        if self.num_data is not None:
            num_data = jnp.array(self.num_data, dtype=kl.dtype)
            minibatch_size = jnp.array(X.shape[0], dtype=kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = jnp.array(1.0, dtype=kl.dtype)
            
        return jnp.sum(var_exp) * scale - kl

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def predict_f(
        self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        """
        Posterior prediction at new input Xnew
        :param Xnew: N x D Tensor
        """
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        
        # JAX version of GPflow's conditional
        K_uu = self.kernel.K_ff(self.inducing_variable.Z)
        K_fu = self.kernel.K_ff(Xnew, self.inducing_variable.Z)
        K_ff_diag = self.kernel.K_diag(Xnew)
        
        K_uu_chol = jnp.linalg.cholesky(K_uu + jnp.eye(K_uu.shape[0]) * 1e-6)
        
        A = jnp.linalg.solve(K_uu_chol.T, jnp.linalg.solve(K_uu_chol, K_fu.T)).T
        
        f_mean = jnp.matmul(A, q_mu)
        f_var = K_ff_diag - jnp.sum(jnp.square(jnp.linalg.solve(K_uu_chol, K_fu.T)), axis=0) + jnp.sum(jnp.square(jnp.matmul(jnp.linalg.solve(K_uu_chol, K_fu.T).T, q_sqrt)), axis=-1)
        
        # Add mean function
        f_mean +=