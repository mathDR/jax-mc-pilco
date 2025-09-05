from __future__ import annotations

__all__ = ["GaussianProcess", "SparseVariationalGaussianProcess"]

from collections.abc import Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
)
from jaxtyping import ArrayLike

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import jax.scipy as jsp

from jax_mc_pilco.model_learning.gp.kernels.base import Kernel
from jax_mc_pilco.model_learning.gp.means import ZeroMean, Mean

from jax_mc_pilco.model_learning.gp.solver import DirectSolver

if TYPE_CHECKING:
    from jax_mc_pilco.model_learning.gp.numpyro_support import TinyDistribution


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def svgp_fit(
    model: SparseVariationalGaussianProcess,
    *,
    max_iters: int = 500,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
) -> SparseVariationalGaussianProcess:
    r"""Maximize the collapsed expected lower bound (ELBO) for the given SparseVariationalGaussianProcess

    Uses Optax's LBFGS implementation and a jax.lax.while loop.

     Args:
         params: the parameters of the kernel, mean, likelihood and inducing point locations.
         max_iters (int): The maximum number of optimisation steps to run. Defaults
             to 500.
         max_linesearch_steps (int): The maximum number of linesearch steps to use
            for finding the stepsize.Defaults to 32.
         gtol (float): Terminate the optimisation if the L2 norm of the gradient is
            below this threshold. Defaults to 1e-8.

     Returns:
         A new SparseVariationalGaussianProcess with the optimized parameters and properties.
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
        # Using optax's value_and_grad_from_state is more efficient given LBFGS uses a linesearch
        # See https://optax.readthedocs.io/en/latest/api/utilities.html#optax.value_and_grad_from_state
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
    final_loss = optax.tree_utils.tree_get(opt_state, "value")
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
            compatible with ``kernel`` where the zeroth dimension is ``N_data``,
            the size of the data set.
        y (ArrayLike): The observed data. This should have the shape
            ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
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

        self.cached_choleskys = cached_choleskys

    def jitter(self, d, value=1e-6):
        return jnp.eye(d) * value

    def collapsed_elbo(
        self,
        params: Dict[str, float],
    ) -> Union[float, Float[jax.Array, ""]]:
        log_noise = params["likelihood"]["log_diag"]
        noise = softplus(log_noise)
        sq_noise = jnp.square(noise)

        z = params["inducing_point_locations"]
        kernel = self.kernel(**params["kernel"])

        K_zz = kernel(z, z) + self.jitter(z.shape[0])
        K_zx = kernel(z, self.X)
        Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(self.X, self.X)
        mu = self.mean(**params["mean"])(self.X)

        Lz = jnp.linalg.cholesky(K_zz)  # m x m

        A = jsp.linalg.solve_triangular(Lz, K_zx, lower=True) / noise  # m x n
        AAT = jnp.matmul(A, A.T)  # m x m
        B = jnp.eye(z.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m

        log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(LB)))
        diff = self.y - mu

        L_inv_A_diff = jsp.linalg.solve_triangular(LB, jnp.matmul(A, diff), lower=True)
        quad = (
            jnp.sum(jnp.square(diff)) - jnp.sum(jnp.square(L_inv_A_diff))
        ) / sq_noise

        two_log_prob = (
            -self.num_data * jnp.log(2.0 * jnp.pi * sq_noise) - log_det_B - quad
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
        sq_noise = jnp.square(noise)

        z = params["inducing_point_locations"]
        kernel = self.kernel(**params["kernel"])

        mean = self.mean(**params["mean"])
        mu = mean(self.X)

        K_zz = kernel(z, z) + self.jitter(z.shape[0])
        L_z = jnp.linalg.cholesky(K_zz)

        K_zx = kernel(z, self.X)

        Lz_inv_Kzx = jsp.linalg.cho_solve((L_z, True), K_zx)

        A = Lz_inv_Kzx / noise
        AAT = jnp.matmul(A, A.T)
        L_AAT = jnp.linalg.cholesky(AAT + jnp.eye(self.num_inducing_points))

        diff = self.y - mu

        Lz_inv_Kzx_diff = jsp.linalg.cho_solve(
            (L_AAT, True), jnp.matmul(Lz_inv_Kzx, diff)
        )

        Kzz_inv_Kzx_diff = jsp.linalg.cho_solve((L_z, True), Lz_inv_Kzx_diff)

        return (L_z, L_AAT, Kzz_inv_Kzx_diff)

    def predict(
        self,
        X_test: ArrayLike | None = None,
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:
        """Predict the GP model at new test points conditioned on observed data.
           This method caches the intermediate

        Args:
            params (ArrayLike): The optimized parameters for the kernel, likelihood,
                mean and locations of the inducing points.
            X_test (ArrayLike, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.

        Returns:
            The mean and covariance of the predictive model evaluated at ``X_test``, with shape
            ``(N_test,)`` and ``(N_test, N_test)`` where ``N_test`` is the zeroth dimension of
            ``X_test``.
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
        mu = mean(self.X)

        L_z, L_AAT, Kzz_inv_Kzx_diff = self.compute_cached_choleskys(self.params)

        K_tt = kernel(X_test, X_test)
        K_zt = kernel(z, X_test)

        mu_t = jnp.atleast_2d(mean(X_test)).T

        Lz_inv_Kzt = jsp.linalg.cho_solve((L_z, True), K_zt)
        L_inv_Lz_inv_Kzt = jsp.linalg.solve_triangular(L_AAT, Lz_inv_Kzt, lower=True)

        f_q = mu_t + jnp.matmul(K_zt.T / sq_noise, Kzz_inv_Kzx_diff)

        f_q_cov = (
            K_tt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
            + self.jitter(X_test.shape[0])
        )

        return jnp.atleast_1d(f_q.squeeze()), f_q_cov


class GaussianProcess(eqx.Module):
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (ArrayLike): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``,
            the size of the data set.
        y (ArrayLike): The observed data. This should have the shape
            ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
            data provided when instantiating this object.
        mean (Mean): The mean function.  If not specified, a zero mean will be used.
    """

    num_data: int = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    kernel: Kernel
    X: ArrayLike
    y: ArrayLike
    mean: Mean
    solver: DirectSolver

    def __init__(self, kernel: Kernel, X: ArrayLike, y: ArrayLike, *, mean: means.Mean):
        self.kernel = kernel
        self.X = X
        self.y = y
        if mean:
            self.mean = mean
        else:
            self.mean = ZeroMean
        self.num_data = X.shape[0]
        self.dtype = X.dtype

        self.solver = DirectSolver(
            kernel,
            self.X,
            self.y,
        )

    def log_probability(self) -> jax.Array:
        """Compute the log probability of this multivariate normal

        Args:

        Returns:
            The marginal log probability of this multivariate normal model,
            evaluated at ``self.y``.
        """
        return self._compute_log_prob(self._get_alpha())

    def fit(
        self,
        X_test: ArrayLike | None = None,
    ) -> Tuple[Float, GaussianProcess]:
        """Condition the model on observed data and

        Args:
            X_test (ArrayLike, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            kernel (Kernel, optional): A kernel to optionally specify the
                covariance between the observed data and predicted data. See
                :ref:`mixture` for an example.

        Returns:
            A tuple where the first element ``log_probability`` is the log
            marginal probability of the model, and the second element ``gp`` is
            the :class:`GaussianProcess` object describing the conditional
            distribution evaluated at ``X_test``.
        """
        # If X_test is provided, we need to check that the tree structure
        # matches that of the input data, and that the shapes are all compatible
        # (i.e. the dimension of the inputs must match). This is slightly
        # convoluted since we need to support arbitrary pytrees.
        if X_test is not None:
            matches = jax.tree_util.tree_map(
                lambda a, b: jnp.ndim(a) == jnp.ndim(b)
                and jnp.shape(a)[1:] == jnp.shape(b)[1:],
                self.X,
                X_test,
            )
            if not jax.tree_util.tree_reduce(lambda a, b: a and b, matches):
                raise ValueError(
                    "`X_test` must have the same tree structure as the input `X`, "
                    "and all but the leading dimension must have matching sizes"
                )

        alpha, log_prob, mean_value = self._fit(X_test)

        covariance_value = self.solver.condition(kernel, X_test)
        if X_test is None:
            X_test = self.X

        # The conditional GP will also be a GP with the mean an covariance
        # specified by a :class:`means.Conditioned` and
        # :class:`kernels.Conditioned` respectively.
        gp = GaussianProcess(
            kernels.Conditioned(self.X, self.solver, kernel),
            X_test,
            mean=self.mean,
            covariance_value=covariance_value,
        )

        return ConditionResult(log_prob, gp)

    @jax.jit
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
            The mean of the predictive model evaluated at ``X_test``, with shape
            ``(N_test,)`` where ``N_test`` is the zeroth dimension of
            ``X_test``. If either ``return_var`` or ``return_cov`` is ``True``,
            the covariance of the predicted process will also be
            returned with shape ``(N_test, N_test)``.
        """
        _, cond = self.fit(X_test)
        return cond.loc, cond.covariance

    def sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None = None,
    ) -> jax.Array:
        """Generate samples from the prior process

        Args:
            key: A ``jax`` random number key array. shape (tuple, optional): The
            number and shape of samples to
                generate.

        Returns:
            The sampled realizations from the process with shape ``(N_data,) +
            shape`` where ``N_data`` is the zeroth dimension of the ``X``
            coordinates provided when instantiating this process.
        """
        return self._sample(key, shape)

    def numpyro_dist(self, **kwargs: Any) -> TinyDistribution:
        """Get the numpyro MultivariateNormal distribution for this process"""
        from jax_mc_pilco.model_learning.gp.numpyro_support import TinyDistribution

        return TinyDistribution(self, **kwargs)

    @partial(jax.jit, static_argnums=(2,))
    def _sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None,
    ) -> jax.Array:
        if shape is None:
            shape = (self.num_data,)
        else:
            shape = (self.num_data,) + tuple(shape)
        normal_samples = jax.random.normal(key, shape=shape, dtype=self.dtype)
        return self.mean + jnp.moveaxis(
            self.solver.dot_triangular(normal_samples), 0, -1
        )

    @jax.jit
    def _compute_log_prob(self, alpha: ArrayLike) -> jax.Array:
        loglike = -0.5 * jnp.sum(jnp.square(alpha)) - self.solver.normalization()
        return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)

    @jax.jit
    def _get_alpha(self) -> jax.Array:
        return self.solver.solve_triangular(self.y - self.loc)

    @jax.jit
    def _condition(
        self,
        X_test: ArrayLike | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        alpha = self._get_alpha(y)
        log_prob = self._compute_log_prob(alpha)

        # Below, we actually want alpha = K^-1 y instead of alpha = L^-1 y
        alpha = self.solver.solve_triangular(alpha, transpose=True)

        if X_test is None:
            X_test = self.X

            # In this common case (where we're predicting the GP at the data
            # points, using the original kernel), the mean is especially fast to
            # compute; so let's use that calculation here.
            if kernel is None:
                delta = self.noise @ alpha
                mean_value = y - delta
                if not include_mean:
                    mean_value -= self.loc

            else:
                mean_value = kernel.matmul(self.X, y=alpha)
                if include_mean:
                    mean_value += self.loc

        else:
            if kernel is None:
                kernel = self.kernel

            mean_value = kernel.matmul(X_test, self.X, alpha)
            if include_mean:
                mean_value += jax.vmap(self.mean_function)(X_test)

        return alpha, log_prob, mean_value


class ConditionResult(NamedTuple):
    """The result of conditioning a :class:`GaussianProcess` on data

    This has two entries, ``log_probability`` and ``gp``, that are described
    below.
    """

    log_probability: ArrayLike
    """The log probability of the conditioned model

    In other words, this is the marginal likelihood for the kernel parameters,
    given the observed data, or the multivariate normal log probability
    evaluated at the given data.
    """

    gp: GaussianProcess
    """A :class:`GaussianProcess` describing the conditional distribution

    This will have a mean and covariance conditioned on the observed data, but
    it is otherwise a fully functional GP that can sample from or condition
    further (although that's probably not going to be very efficient).
    """


def _default_diag(reference: ArrayLike) -> jax.Array:
    """Default to adding some amount of jitter to the diagonal, just in case,
    we use sqrt(eps) for the dtype of the mean function because that seems to
    give sensible results in general.
    """
    return jnp.sqrt(jnp.finfo(reference).eps)
