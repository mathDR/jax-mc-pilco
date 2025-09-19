from __future__ import annotations

__all__ = ["GaussianProcess", "SparseVariationalGaussianProcess"]

from jaxtyping import ArrayLike, Bool, Float
from typing import Dict, Tuple, Union
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import warnings
import jax.scipy as jsp

from jax_mc_pilco.model_learning.gp.kernels.base import Kernel
from jax_mc_pilco.model_learning.gp.means import ZeroMean, Mean


def softplus(X: ArrayLike) -> jax.Array:
    return jnp.log(1 + jnp.exp(X))


def svgp_fit(
    model: SparseVariationalGaussianProcess,
    *,
    max_iters: int = 500,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
) -> SparseVariationalGaussianProcess:
    r"""Maximize the collapsed expected lower bound (ELBO) for the given
    SparseVariationalGaussianProcess.

    Uses Optax's LBFGS implementation and a jax.lax.while loop.

     Args:
         params: the parameters of the kernel, mean, likelihood and inducing
         point locations.
         max_iters (int): The maximum number of optimisation steps to run.
            Defaults to 500.
         max_linesearch_steps (int): The maximum number of linesearch steps to
            use for finding the stepsize.Defaults to 32.
         gtol (float): Terminate the optimisation if the L2 norm of the
            gradient is below this threshold. Defaults to 1e-8.

     Returns:
         A new SparseVariationalGaussianProcess with the optimized parameters
            and properties.
    """
    vals, static = eqx.partition(model.params, eqx.is_array)

    @jax.jit
    def loss(vals: Dict) -> Float:
        params = eqx.combine(vals, static)
        return -model.collapsed_elbo(params)

    # Initialise optimiser
    optim = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        )
    )

    opt_state = optim.init(model.params)
    loss_value_and_grad = optax.value_and_grad_from_state(loss)

    # Optimisation step.
    @jax.jit
    def step(carry):
        vals, opt_state = carry
        # Using optax's value_and_grad_from_state is more efficient given
        # LBFGS uses a linesearch
        # See
        # https://optax.readthedocs.io/en/latest/api/utilities.html#optax.value_and_grad_from_state
        loss_val, loss_gradient = loss_value_and_grad(vals, state=opt_state)
        updates, opt_state = optim.update(
            loss_gradient,
            opt_state,
            vals,
            value=loss_val,
            grad=loss_gradient,
            value_fn=loss,
        )
        vals = optax.apply_updates(vals, updates)

        return vals, opt_state

    def continue_fn(carry):
        _, opt_state = carry
        n = optax.tree_utils.tree_get(opt_state, "count")
        g = optax.tree_utils.tree_get(opt_state, "grad")
        g_l2_norm = optax.tree_utils.tree_norm(g)
        return (n == 0) | ((n < max_iters) & (g_l2_norm >= gtol))

    # Optimisation loop
    opt_vals, opt_state = jax.lax.while_loop(
        continue_fn,
        step,
        (vals, opt_state),
    )
    final_params = eqx.combine(opt_vals, static)

    cached_choleskys = model.compute_cached_choleskys(final_params)

    return SparseVariationalGaussianProcess(
        model.kernel,
        model.X,
        model.y,
        model.num_inducing_points,
        final_params,
        mean=model.mean,
        optimized=True,
        cached_choleskys=cached_choleskys,
    )


class SparseVariationalGaussianProcess(eqx.Module):
    """An interface for designing a Sparse Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (ArrayLike): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the 0th dimension is ``N_data``,
            the size of the data set.
        y (ArrayLike): The observed data. This should have the shape
            ``(N_data,)``, where ``N_data`` was the 0th axis of the ``X``
            data provided when instantiating this object.
        num_inducing_points (int): the number of inducing points.
        mean (Mean): The mean function that will be evaluated with the ``X``
          as input: ``mean(X)`` if Left
    """

    num_data: int = eqx.field(static=True)
    num_inducing_points: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    kernel: Kernel
    X: ArrayLike
    y: ArrayLike
    mean: Mean
    params: Dict
    optimized: Bool
    cached_choleskys: Tuple[ArrayLike, ArrayLike, ArrayLike]

    def __init__(
        self,
        kernel: Kernel,
        X: ArrayLike,
        y: ArrayLike,
        num_inducing_points: int,
        params: Dict,
        *,
        mean: Mean | None = None,
        optimized: bool = False,
        cached_choleskys: Tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    ):
        self.kernel = kernel
        if mean:
            self.mean = mean
        else:
            self.mean = ZeroMean
        self.X = X
        self.y = y

        self.num_data = self.X.shape[0]
        self.dtype = self.X.dtype
        self.num_inducing_points = num_inducing_points

        self.params = params
        self.optimized = optimized

        if cached_choleskys:
            self.cached_choleskys = cached_choleskys
        else:
            self.cached_choleskys = self.compute_cached_choleskys(self.params)

    def jitter(self, d, value=1e-6):
        return jnp.eye(d) * value

    def collapsed_elbo(
        self,
        params: Dict[str, float],
    ) -> Union[float, Float[jax.Array]]:
        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        z = params["inducing_point_locations"]
        kernel = self.kernel(**params["kernel"])

        K_zz = kernel(z, z) + self.jitter(z.shape[0])
        K_zx = kernel(z, self.X)
        Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(self.X, self.X)
        mu_x = self.mean(**params["mean"])(self.X)

        Lz = jnp.linalg.cholesky(K_zz)  # m x m

        A = jsp.linalg.cho_solve((Lz, True), K_zx) / noise  # m x n
        AAT = jnp.matmul(A, A.T)  # m x m
        B = jnp.eye(z.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m

        log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(LB)))
        diff = jnp.squeeze(self.y) - mu_x

        L_inv_A_diff = jsp.linalg.cho_solve((LB, True), jnp.matmul(A, diff))
        quad = (
            jnp.sum(jnp.square(diff)) - jnp.sum(jnp.square(L_inv_A_diff))
        ) / sq_noise

        two_log_prob = (
            -self.num_data * jnp.log(2.0 * jnp.pi * sq_noise) -
            log_det_B - quad
        )
        two_trace = jnp.sum(Kxx_diag) / sq_noise - jnp.trace(AAT)

        return 0.5 * (two_log_prob - two_trace).squeeze()

    def compute_cached_choleskys(
        self,
        params: Dict[str, float],
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Computes some things for reuse"""

        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)

        z = params["inducing_point_locations"]
        kernel = self.kernel(**params["kernel"])

        mean = self.mean(**params["mean"])
        mu_x = mean(self.X)

        K_zz = kernel(z, z) + self.jitter(z.shape[0])
        L_z = jnp.linalg.cholesky(K_zz)

        K_zx = kernel(z, self.X)

        Lz_inv_Kzx = jsp.linalg.cho_solve((L_z, True), K_zx)

        A = Lz_inv_Kzx / noise
        AAT = jnp.matmul(A, A.T)
        L_AAT = jnp.linalg.cholesky(AAT + jnp.eye(self.num_inducing_points))

        diff = jnp.squeeze(self.y) - mu_x

        Lz_inv_Kzx_diff = jsp.linalg.cho_solve(
            (L_AAT, True), jnp.matmul(Lz_inv_Kzx, diff)
        )

        Kzz_inv_Kzx_diff = jsp.linalg.cho_solve(
            (L_z.T, False),
            Lz_inv_Kzx_diff
            )

        return (L_z, L_AAT, Kzz_inv_Kzx_diff)

    def predict(
        self,
        X_test: ArrayLike | None = None,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        """Predict the GP model at new test points conditioned on observed
            data.

        Args:
            X_test (ArrayLike, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.

        Returns:
            The mean and covariance of the predictive model evaluated at
            ``X_test``, with shape ``(N_test,)`` and ``(N_test, N_test)``
            where ``N_test`` is the zeroth dimension of ``X_test``.
        """

        # Compute mu and Covariance
        if not self.optimized:
            warnings.warn("You are calling predict on an unoptimized gp.")

        log_noise = self.params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        z = self.params["inducing_point_locations"]
        kernel = self.kernel(**self.params["kernel"])

        mean = self.mean(**self.params["mean"])

        L_z, L_AAT, Kzz_inv_Kzx_diff = self.cached_choleskys

        K_tt = kernel(X_test, X_test)
        K_zt = kernel(z, X_test)

        mu_t = mean(X_test)

        Lz_inv_Kzt = jsp.linalg.cho_solve((L_z, True), K_zt)
        L_inv_Lz_inv_Kzt = jsp.linalg.cho_solve((L_AAT, True), Lz_inv_Kzt)

        f_q = mu_t + jnp.matmul(K_zt.T / sq_noise, Kzz_inv_Kzx_diff)

        f_q_cov = (
            K_tt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
            + self.jitter(X_test.shape[0])
        )

        return jnp.atleast_1d(f_q.squeeze()), f_q_cov


def gp_fit(
    model: GaussianProcess,
    *,
    max_iters: int = 500,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
) -> GaussianProcess:
    r"""Maximize the log marginal likelihood for the given GaussianProcess

    Uses Optax's LBFGS implementation and a jax.lax.while loop.

     Args:
         params: the parameters of the kernel, mean, likelihood and inducing
            point locations.
         max_iters (int): The maximum number of optimisation steps to run.
            Defaults to 500.
         max_linesearch_steps (int): The maximum number of linesearch steps to
            use for finding the stepsize.Defaults to 32.
         gtol (float): Terminate the optimisation if the L2 norm of the
            gradient is below this threshold. Defaults to 1e-8.

     Returns:
         A new GaussianProcess with the optimized parameters and properties.
    """
    vals, static = eqx.partition(model.params, eqx.is_array)

    @jax.jit
    def loss(vals: Dict) -> Float:
        params = eqx.combine(vals, static)
        return -model.log_likelihood(params)

    # Initialise optimiser
    optim = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        )
    )

    opt_state = optim.init(model.params)
    loss_value_and_grad = optax.value_and_grad_from_state(loss)

    # Optimisation step.
    @jax.jit
    def step(carry):
        vals, opt_state = carry
        # Using optax's value_and_grad_from_state is more efficient given
        # LBFGS uses a linesearch
        # See
        # https://optax.readthedocs.io/en/latest/api/utilities.html#optax.value_and_grad_from_state
        loss_val, loss_gradient = loss_value_and_grad(vals, state=opt_state)
        updates, opt_state = optim.update(
            loss_gradient,
            opt_state,
            vals,
            value=loss_val,
            grad=loss_gradient,
            value_fn=loss,
        )
        vals = optax.apply_updates(vals, updates)

        return vals, opt_state

    def continue_fn(carry):
        _, opt_state = carry
        n = optax.tree_utils.tree_get(opt_state, "count")
        g = optax.tree_utils.tree_get(opt_state, "grad")
        g_l2_norm = optax.tree_utils.tree_norm(g)
        return (n == 0) | ((n < max_iters) & (g_l2_norm >= gtol))

    # Optimisation loop
    opt_vals, opt_state = jax.lax.while_loop(
        continue_fn,
        step,
        (vals, opt_state),
    )
    final_params = eqx.combine(opt_vals, static)

    cached_choleskys = model.compute_cached_choleskys(final_params)

    return GaussianProcess(
        model.kernel,
        model.X,
        model.y,
        final_params,
        mean=model.mean,
        optimized=True,
        cached_choleskys=cached_choleskys,
    )


class GaussianProcess(eqx.Module):
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (ArrayLike): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``
            the size of the data set.
        y (ArrayLike): The observed data. This should have the shape
            ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
            data provided when instantiating this object.
        mean (Mean): The mean function.  If not specified, a zero mean will be
            used.
    """

    num_data: int = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    kernel: Kernel
    X: ArrayLike
    y: ArrayLike
    mean: Mean
    params: Dict
    optimized: Bool
    cached_choleskys: Tuple[ArrayLike, ArrayLike, ArrayLike]

    def __init__(
        self,
        kernel: Kernel,
        X: ArrayLike,
        y: ArrayLike,
        params: Dict,
        *,
        mean: Mean | None = None,
        optimized: Bool | None = None,
        cached_choleskys: Tuple[ArrayLike, ArrayLike] | None = None,
    ):
        self.kernel = kernel
        self.X = X
        self.y = y

        if mean:
            self.mean = mean
        else:
            self.mean = ZeroMean

        self.num_data = X.shape[0]
        self.dtype = X.dtype

        self.params = params
        self.optimized = optimized

        if cached_choleskys is None:
            self.cached_choleskys = self.compute_cached_choleskys(self.params)
        else:
            self.cached_choleskys = cached_choleskys

    def jitter(self, d, value=1e-6):
        return jnp.eye(d) * value

    def log_likelihood(self, params: Dict) -> jax.Array:
        """Compute the log likelihood of this multivariate normal

        Args:
            params (Dict): The hyperparameters of the kernel, mean and
                likelihood

        Returns:
            The marginal log likelihood of this multivariate normal model,
            evaluated at ``self.y``.
        """

        kernel = self.kernel(**params["kernel"])
        mean = self.mean(**params["mean"])

        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        covariance = (
            kernel(self.X, self.X) +
            self.jitter(self.num_data, value=sq_noise)
        )
        L_xx = jsp.linalg.cholesky(covariance, lower=True)

        alpha = jsp.linalg.cho_solve((L_xx, True), self.y-mean(self.X))
        S2 = jsp.linalg.cho_solve((L_xx.T, False), alpha)

        # log_likelihood = -0.5 * jnp.einsum("ik,ik->k", self.y, alpha)
        log_likelihood = -0.5 * jnp.dot(self.y.T, S2)
        log_likelihood -= jnp.log(jnp.diag(L_xx)).sum()
        log_likelihood -= 0.5 * self.num_data * jnp.log(2.0 * jnp.pi)
        return jnp.squeeze(log_likelihood)

    def compute_cached_choleskys(self, params: Dict) -> jax.Array:
        """Compute the cholesky of the covariance as well as the alpha value.

        Args:
            params (Dict): The hyperparameters of the kernel, mean,
              and likelihood

        Returns:
            The cholesky of the covariance as well as K_xx^{-1} y
        """

        kernel = self.kernel(**params["kernel"])
        mean = self.mean(**params["mean"])

        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        covariance = (
            kernel(self.X, self.X) + self.jitter(self.num_data, value=sq_noise)
        )

        L_xx = jsp.linalg.cholesky(covariance, lower=True)
        alpha = jsp.linalg.cho_solve((L_xx, True), self.y-mean(self.X))

        return (L_xx, alpha)

    def predict(
        self,
        X_test: ArrayLike | None = None,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        """Predict the GP model at new test points conditioned on observed data

        Args:
            X_test (ArrayLike, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.
        Returns:
            The mean of the predictive model evaluated at ``X_test``, with
            shape ``(N_test,)`` where ``N_test`` is the zeroth dimension of
            ``X_test``. If either ``return_var`` or ``return_cov`` is ``True``,
            the covariance of the predicted process will also be
            returned with shape ``(N_test, N_test)``.
        """

        if not self.optimized:
            warnings.warn("You are calling predict on an unoptimized gp.")

        kernel = self.kernel(**self.params["kernel"])
        mean = self.mean(**self.params["mean"])

        log_noise = self.params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        L_xx, alpha = self.cached_choleskys

        K_tx = kernel(X_test, self.X)
        mu_t = mean(X_test)

        y_t = mu_t + jnp.matmul(K_tx, alpha)

        V = jsp.linalg.solve_triangular(L_xx, K_tx.T, lower=True)

        y_cov = (
            kernel(X_test)
            - jnp.matmul(V.T, V)
            + self.jitter(X_test.shape[0], value=sq_noise)
        )

        return y_t, y_cov
