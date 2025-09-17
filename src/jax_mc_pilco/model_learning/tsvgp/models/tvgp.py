"""
Module for the t-VGP model class using JAX, Equinox, and CheX.
"""
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import chex
import equinox as eqx

# Import your rewritten utility functions
from your_utils_module import DiagSites # Assuming you have saved the previous DiagSites in a file named your_utils_module.py

# Type aliases for clarity
InputData = chex.Array
RegressionData = Tuple[InputData, InputData]
MeanAndVariance = Tuple[chex.Array, chex.Array]

# Set JAX to use 64-bit floats for numerical stability
jax.config.update("jax_enable_x64", True)


class t_VGP(eqx.Module):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    The key reference is:
      Khan, M., & Lin, W. (2017). Conjugate-Computation Variational Inference:
      Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models.
      In Artificial Intelligence and Statistics (pp. 878-887).

    """
    kernel: eqx.Module
    likelihood: eqx.Module
    mean_function: Optional[eqx.Module]
    sites: DiagSites
    data: RegressionData = eqx.field(static=True)
    num_data: int = eqx.field(static=True)
    num_latent: int = eqx.field(static=True)
    q_alpha: Optional[chex.Array] = None

    def __init__(
        self,
        data: RegressionData,
        kernel: eqx.Module,
        likelihood: eqx.Module,
        mean_function: Optional[eqx.Module] = None,
        num_latent: Optional[int] = 1,
    ):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        kernel, likelihood, mean_function are appropriate JAX/Equinox objects
        """
        x_data, y_data = data
        self.data = data
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = mean_function
        
        self.num_data = x_data.shape[0]
        self.num_latent = num_latent or y_data.shape[1]
        
        lambda_1 = jnp.zeros((self.num_data, self.num_latent))
        lambda_2 = 1e-6 * jnp.ones((self.num_data, self.num_latent))
        self.sites = DiagSites(lambda_1, lambda_2)

    @property
    def lambda_1(self) -> chex.Array:
        """first natural parameter"""
        return self.sites.lambda_1

    @property
    def lambda_2(self) -> chex.Array:
        """second natural parameter"""
        return self.sites.lambda_2

    @partial(jax.jit, static_argnums=(0,))
    def elbo(self) -> chex.Array:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        x_data, y_data = self.data
        pseudo_y = self.lambda_1 / self.lambda_2
        sW = jnp.sqrt(jnp.abs(self.lambda_2))
        
        jitter = 1e-6
        
        # Computes conversion λ₁, λ₂ → m, V by using q(f) ≃ t(f)p(f)
        K = self.kernel.K_ff(x_data, full_cov=True) + jnp.eye(self.num_data, dtype=jnp.float64) * jitter

        # L = chol(I + √λ₂ᵀ K √λ₂ᵀ)
        L = jnp.linalg.cholesky(
            jnp.eye(self.num_data, dtype=jnp.float64) + (sW.T @ sW) * K
        )
        
        # T = L⁻¹ λ₂ K
        T = jnp.linalg.solve(L, jnp.tile(sW, (1, self.num_data)) * K)
        
        # Σ = (K⁻¹ + λ₂)⁻¹ = K - K√λ₂ (I + √λ₂ᵀK√λ₂)⁻¹√λ₂ᵀK
        post_v = jnp.expand_dims(
            jnp.diag(K) - jnp.sum(T * T, axis=0), 1
        )

        # μ = Σλ₁
        alpha = sW * jnp.linalg.solve(L.T, jnp.linalg.solve(L, sW * pseudo_y))
        post_m = K @ alpha
        
        # Store alpha for prediction
        self = eqx.tree_at(lambda model: model.q_alpha, self, alpha)
        
        # Get variational expectations.
        E_q_log_lik = jnp.sum(self.likelihood.variational_expectations(post_m, post_v, y_data))
        E_q_log_t = -0.5 * jnp.sum((self.lambda_2) * ((pseudo_y - post_m) ** 2 + post_v))
        log_Z = -0.5 * jnp.einsum("mi,mj->", pseudo_y, alpha) - jnp.sum(jnp.log(jnp.diag(L)))
        
        elbo = log_Z - E_q_log_t + E_q_log_lik
        return elbo

    @partial(jax.jit, static_argnums=(0,))
    def update_variational_parameters(self, beta: float = 0.05):
        """
        Takes a natural gradient step in local variational parameters.
        Returns a new, updated model instance.
        """
        x_data, y_data = self.data
        
        def loss_fn(lambda_1, lambda_2):
            pseudo_y = lambda_1 / lambda_2
            sW = jnp.sqrt(jnp.abs(lambda_2))
            jitter = 1e-6
            K = self.kernel.K_ff(x_data, full_cov=True) + jnp.eye(self.num_data, dtype=jnp.float64) * jitter
            
            L = jnp.linalg.cholesky(
                jnp.eye(self.num_data, dtype=jnp.float64) + (sW.T @ sW) * K
            )
            T = jnp.linalg.solve(L, jnp.tile(sW, (1, self.num_data)) * K)
            post_v = jnp.expand_dims(jnp.diag(K) - jnp.sum(T * T, axis=0), 1)
            alpha = sW * jnp.linalg.solve(L.T, jnp.linalg.solve(L, sW * pseudo_y))
            post_m = K @ alpha
            
            return jnp.sum(self.likelihood.variational_expectations(post_m, post_v, y_data)), (post_m, post_v)

        # Get gradients of variational expectations wrt post_m and post_v
        grad_fn = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)
        (d_exp_dm, d_exp_dv), (post_m, post_v) = grad_fn(self.lambda_1, self.lambda_2)

        # Take the tVGP step and transform to be ▽μ[Var_exp]
        lambda_1_new = (1.0 - beta) * self.lambda_1 + beta * (d_exp_dm - 2.0 * d_exp_dv * post_m)
        lambda_2_new = (1.0 - beta) * self.lambda_2 + beta * (-2.0 * d_exp_dv)
        
        # Return a new, updated model instance
        return eqx.tree_at(
            lambda model: (model.sites.lambda_1, model.sites.lambda_2),
            self,
            (lambda_1_new, lambda_2_new)
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def predict_f(
        self