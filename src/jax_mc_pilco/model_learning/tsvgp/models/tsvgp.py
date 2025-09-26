"""
Module for the t-SVGP model rewritten in JAX, Equinox, and CheX.
"""

from typing import Dict, Tuple
from jaxtyping import ArrayLike, Bool, Int

import jax
import jax.numpy as jnp

# import chex
import equinox as eqx
import jax.scipy as jsp

from jax_mc_pilco.model_learning.gp.kernels.base import Kernel
from jax_mc_pilco.model_learning.gp.means import ZeroMean, Mean

# Set JAX to use 64-bit floats for numerical stability
jax.config.update("jax_enable_x64", True)


def softplus(X: ArrayLike) -> jax.Array:
    return jnp.log(1 + jnp.exp(X))


class t_SVGP(eqx.Module):
    """
    Class for the t-SVGP model
    """

    """
    Modified SVGP class to accommodate
    a different paramaterization of q(u)

    Args:
        kernel (Kernel): The kernel function
        X (ArrayLike): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``
            the size of the data set.
        y (ArrayLike): The observed data. This should have the shape
            ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
            data provided when instantiating this object.
        params (Dict[ArrayLike]): the parameters of the model. Including
            kernel, mean, and likelihood hyperparameters.  Furthermore contains
            the location of the inducing inputs
        mean_function (Mean): The mean function.  If not specified, a zero
            mean will be used.
        sites (Dict[ArrayLike]): the natural parameters of the variational
            sites:
                lambda_1 (ArrayLike): First order natural parameter of the
                    variational site.
                Lambda_2_sqrt (ArrayLike): Cholesky Factor of the second order
                    natural parameter of the variational site.
    """

    num_data: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    kernel: Kernel
    X: ArrayLike
    y: ArrayLike
    num_inducing_points: Int
    mean_function: Mean
    params: Dict
    sites: Dict
    optimized: Bool

    def __init__(
        self,
        kernel: Kernel,
        X: ArrayLike,
        y: ArrayLike,
        params: Dict,
        sites: Dict,
        *,
        mean_function: Mean | None = None,
        optimized: Bool | None = None,
    ):
        self.kernel = kernel
        self.X = X
        self.y = y
        self.num_inducing_points = params["inducing_point_locations"].shape[0]

        if mean_function:
            self.mean_function = mean_function
        else:
            self.mean_function = ZeroMean

        self.num_data = X.shape[0]
        self.dtype = X.dtype

        self.params = params
        self.optimized = optimized

        self.sites = sites

    def jitter(self, d, value=1e-6):
        return jnp.eye(d) * value

    def variational_expectations(
        self, Fmu: ArrayLike, Fvar: ArrayLike, params: Dict
    ) -> jax.Array:
        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * log_noise
            - 0.5 * (jnp.square(self.y - Fmu) + Fvar) / sq_noise,
            axis=-1,
        )

    def prior_KL(self, params: Dict, sites: Dict) -> jax.Array:
        """Returns the KL divergence KL[q(u)|p(u)]"""

        kernel = self.kernel(**params["kernel"])
        z = params["inducing_point_locations"]

        K_uu = kernel(z, z) + self.jitter(self.num_inducing_points)
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior(params, sites)

        Lp = jnp.linalg.cholesky(K_uu)
        alpha = jsp.linalg.cho_solve((Lp, True), q_mu)

        # Mahalanobis term: μqᵀ Σp⁻¹ μq
        mahalanobis = jnp.sum(jnp.square(alpha))

        # Constant term: - L * M
        constant = -jnp.size(q_mu)

        # Log-determinant of the covariance of q(x):
        logdet_qcov = jnp.sum(jnp.log(jnp.square(jnp.diag(q_sqrt))))
        LpiLq = jsp.linalg.cho_solve((q_sqrt, True), q_sqrt)
        trace = jnp.sum(jnp.square(LpiLq))

        twoKL = mahalanobis + constant - logdet_qcov + trace
        # Log-determinant of the covariance of p(x):
        log_sqdiag_Lp = jnp.log(jnp.square(jnp.diag(Lp)))
        sum_log_sqdiag_Lp = jnp.sum(log_sqdiag_Lp)
        twoKL += sum_log_sqdiag_Lp

        return 0.5 * twoKL

    def elbo(self, params: Dict, sites: Dict) -> jax.Array:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        KL = self.prior_KL(params, sites)
        f_mean, f_var = self.predict(self.X, params, sites)
        var_exp = self.variational_expectations(f_mean, f_var, params)
        return jnp.sum(var_exp) - KL

    def get_mean_chol_cov_inducing_posterior(
        self,
        params: Dict,
        sites: Dict,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the mean and cholesky factor of the posterior
            on the inducing variables:
            q(u) = N(u; m, S)
            S = (K⁻¹ + Λ₂)⁻¹
              = (K⁻¹ + L₂L₂ᵀ)⁻¹
              = K - KL₂W⁻¹L₂ᵀK
            W = (I + L₂ᵀKL₂)⁻¹
            m = S λ₁
        """

        kernel = self.kernel(**params["kernel"])

        z = params["inducing_point_locations"]

        lambda_1 = sites["lambda_1"]
        Lambda_2_sqrt = sites["Lambda_2_sqrt"]

        K_uu = kernel(z, z) + self.jitter(self.num_inducing_points)
        L_uu = jnp.linalg.cholesky(K_uu)

        # chex.assert_rank([K_uu, lambda_1, Lambda_2_sqrt], [2, 2, 3])
        # chex.assert_axis_dimension(K_uu, 0, K_uu.shape[1])
        # chex.assert_axis_dimension(lambda_1, 0, K_uu.shape[0])
        # chex.assert_axis_dimension(Lambda_2_sqrt, 1, K_uu.shape[0])
        # chex.assert_axis_dimension(Lambda_2_sqrt, 2, K_uu.shape[1])

        Luu_Lambda2sqrt = jnp.matmul(L_uu, Lambda_2_sqrt.T)
        W = jnp.matmul(Luu_Lambda2sqrt, Luu_Lambda2sqrt.T) + jnp.eye(
            self.num_inducing_points, dtype=jnp.float64
        )
        chol_W = jnp.linalg.cholesky(W)

        LtK = jnp.matmul(Lambda_2_sqrt.T, K_uu)
        iwLtK = jnp.linalg.solve(chol_W.T, LtK)
        S_q = K_uu - jnp.matmul(iwLtK.T, iwLtK)
        chol_S_q = jnp.linalg.cholesky(S_q)
        m_q = jnp.matmul(S_q, lambda_1)

        return m_q, chol_S_q

    def conditional_from_precision_sites(
        self,
        Kuu: ArrayLike,
        Kff: ArrayLike,
        Kuf: ArrayLike,
        lambda_1: ArrayLike,
        Lambda_2_sqrt: ArrayLike | None = None,
    ):
        """
        Computes the mean and covariance...
        """
        Luu = jnp.linalg.cholesky(Kuu)

        # W = I + Lₜᵀ Lₚ Lₚᵀ Lₜ, chol(W)
        Luu_L = jnp.matmul(Luu.T, Lambda_2_sqrt)
        W = jnp.eye(self.num_inducing_points) + jnp.matmul(Luu_L.T, Luu_L)
        chol_W = jnp.linalg.cholesky(W)

        D = jsp.linalg.cho_solve((chol_W, True), Lambda_2_sqrt.T)
        D_Kuf = jnp.matmul(D, Kuf)

        mu = (
            jnp.matmul(Kuf.T, lambda_1)
            - jnp.sum(jnp.matmul(D, jnp.matmul(Kuu, lambda_1)) * D_Kuf, axis=-2).T
        )

        cov = Kff - jnp.sum(jnp.square(D_Kuf), axis=-2).T
        return mu, cov

    def predict(
        self, X_test: ArrayLike, params: Dict, sites: Dict
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Posterior prediction at inputs X_test
        """
        kernel = self.kernel(**params["kernel"])
        mean = self.mean_function(**params["mean"])

        z = params["inducing_point_locations"]

        lambda_1 = sites["lambda_1"]
        Lambda_2_sqrt = sites["Lambda_2_sqrt"]

        K_uu = kernel(z, z) + self.jitter(self.num_inducing_points)

        K_ut = kernel(z, X_test)
        K_tt = kernel(X_test, X_test)

        mu, var = self.conditional_from_precision_sites(
            K_uu,
            K_tt,
            K_ut,
            lambda_1,
            Lambda_2_sqrt,
        )
        # jnp.all([chex.assert_scalar_positive(v) for v in jnp.diag(var)])
        return mu + mean(X_test), var
