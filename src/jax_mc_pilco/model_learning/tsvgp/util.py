"""
Utilities for the model classes in JAX, Equinox, and CheX
"""

from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from equinox import Module
from equinox.nn import Conv2d, Linear

# Use float64 precision for numerical stability in scientific computing
# This must be set at the start of your program
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(4, 5, 6))
@chex.assert_trees_all_shapes_checked
def conditional_from_precision_sites_white(
    Kuu: jax.Array,
    Kff: jax.Array,
    Kuf: jax.Array,
    l: jax.Array,
    L: Optional[jax.Array] = None,
    L2: Optional[jax.Array] = None,
    jitter: float = 1e-9,
):
    """
    Computes the mean and covariance of q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂.
    """
    chex.assert_rank([Kuu, Kuf, Kff, l], [2, 2, 2, 2])
    chex.assert_axis_dimension(Kuu, 0, Kuu.shape[1])
    chex.assert_axis_dimension(Kuf, 0, Kuu.shape[0])
    chex.assert_axis_dimension(Kff, 1, 1)
    chex.assert_axis_dimension(l, 0, Kuu.shape[0])

    if L2 is None:
        L2 = jnp.matmul(L, L.T)

    m = Kuu.shape[-1]
    I = jnp.eye(m, dtype=jnp.float64)
    R = L2 + Kuu + I * jitter
    LR = jnp.linalg.cholesky(R)
    LA = jnp.linalg.cholesky(Kuu)[None]

    tmp1 = jnp.linalg.solve(LR, Kuf)
    tmp2 = jnp.linalg.solve(LA, Kuf)

    cov = Kff - jnp.expand_dims(
        jnp.sum(jnp.square(tmp2), axis=-2) - jnp.sum(jnp.square(tmp1), axis=-2),
        axis=-1,
    )
    mean = jnp.matmul(Kuf.T, jnp.linalg.solve(LR, l)).squeeze()
    return mean, cov


@partial(jax.jit, static_argnums=(4, 5))
@chex.assert_trees_all_shapes_checked
def conditional_from_precision_sites(
    Kuu: jax.Array,
    Kff: jax.Array,
    Kuf: jax.Array,
    l: jax.Array,
    L: Optional[jax.Array] = None,
    L2: Optional[jax.Array] = None,
):
    """
    Computes the mean and covariance of q(g₁) using a different form of natural parameters.
    """
    chex.assert_rank([Kuu, Kuf, Kff, l], [2, 2, 2, 2])
    chex.assert_axis_dimension(Kuu, 0, Kuu.shape[1])
    chex.assert_axis_dimension(Kuf, 0, Kuu.shape[0])
    chex.assert_axis_dimension(Kff, 1, 1)
    chex.assert_axis_dimension(l, 0, Kuu.shape[0])

    if L is None:
        L = jnp.linalg.cholesky(L2)

    m = Kuu.shape[-1]
    Id = jnp.eye(m, dtype=jnp.float64)
    C = jnp.linalg.cholesky(Kuu)

    CtL = jnp.matmul(C, L.T)
    W = Id + jnp.matmul(CtL.T, CtL)
    chol_W = jnp.linalg.cholesky(W)

    D = jnp.linalg.solve(chol_W, L.T)
    tmp = jnp.matmul(D.T, Kuf)

    mean = jnp.matmul(Kuf.T, l) - jnp.expand_dims(
        jnp.sum(jnp.matmul(D.T, jnp.matmul(Kuu, l)) * tmp, axis=-2), axis=-1
    )

    cov = Kff - jnp.expand_dims(jnp.sum(jnp.square(tmp), axis=-2), axis=-1)
    return mean, cov


@partial(jax.jit, static_argnums=(3, 4))
@chex.assert_trees_all_shapes_checked
def project_diag_sites(
    Kuf: jax.Array,
    lambda_1: jax.Array,
    lambda_2: jax.Array,
    Kuu: Optional[jax.Array] = None,
    cholesky: bool = True,
):
    """
    From Kuu, Kuf, λ₁, λ₂, computes the natural parameters L and l.
    """
    chex.assert_rank([Kuf, lambda_1, lambda_2], [2, 2, 2])
    if Kuu is not None:
        chex.assert_rank(Kuu, 2)
        chex.assert_axis_dimension(Kuu, 0, Kuu.shape[1])

    num_latent = lambda_1.shape[-1]
    P = jnp.tile(Kuf[None], (num_latent, 1, 1)) if Kuf.ndim == 2 else Kuf

    if Kuu is not None:
        Luu = jnp.linalg.cholesky(Kuu)
        P = jnp.linalg.solve(Luu, P)

    l = jnp.einsum("lmn,nl->ml", P, lambda_1)
    L_matrix = jnp.einsum("lmn,lon,nl->lmo", P, P, lambda_2)
    if cholesky:
        L_matrix = jnp.linalg.cholesky(L_matrix)
    return l, L_matrix


@partial(jax.jit, static_argnums=(2, 3))
@chex.assert_trees_all_shapes_checked
def posterior_from_dense_site_white(
    K: jax.Array,
    lambda_1: jax.Array,
    lambda_2: jax.Array,
    jitter: float = 1e-9,
):
    """
    Returns the mean and cholesky factor of the posterior density.
    """
    chex.assert_rank([K, lambda_1, lambda_2], [2, 2, 3])
    chex.assert_axis_dimension(K, 0, K.shape[1])
    chex.assert_axis_dimension(lambda_1, 0, K.shape[0])
    chex.assert_axis_dimension(lambda_2, 1, K.shape[0])
    chex.assert_axis_dimension(lambda_2, 2, K.shape[1])

    P = lambda_2
    m = K.shape[-1]
    Id = jnp.eye(m, dtype=jnp.float64)
    R = K + P
    LR = jnp.linalg.cholesky(R + Id * jitter)
    iLRK = jnp.linalg.solve(LR, K)
    S_q = jnp.matmul(iLRK.T, iLRK)
    chol_S_q = jnp.linalg.cholesky(S_q)
    m_q = jnp.matmul(K, jnp.linalg.solve(LR.T, jnp.linalg.solve(LR, lambda_1)))

    return m_q, chol_S_q


@partial(jax.jit, static_argnums=(2, 3))
@chex.assert_trees_all_shapes_checked
def kl_from_precision_sites_white(
    A: jax.Array,
    l: jax.Array,
    L: Optional[jax.Array] = None,
    L2: Optional[jax.Array] = None,
):
    """
    Computes the KL divergence KL[q(f)|p(f)]
    """
    chex.assert_rank([A, l], [2, 2])
    chex.assert_axis_dimension(A, 0, A.shape[1])
    chex.assert_axis_dimension(l, 0, A.shape[0])
    if L is not None:
        chex.assert_rank(L, 3)
    if L2 is not None:
        chex.assert_rank(L2, 3)

    if L2 is None:
        L2 = jnp.matmul(L, L.T)
    m = L2.shape[-2]

    R = L2 + A
    LR = jnp.linalg.cholesky(R)
    LA = jnp.linalg.cholesky(A)

    log_det = jnp.sum(
        jnp.log(jnp.square(jnp.diagonal(LR, axis1=-2, axis2=-1)))
    ) - jnp.sum(jnp.log(jnp.square(jnp.diagonal(LA, axis1=-2, axis2=-1))))

    tmp = jnp.linalg.solve(LR, LA)
    trace_plus_const = jnp.sum(jnp.square(tmp)) - m

    mahalanobis = jnp.sum(
        jnp.square(jnp.matmul(LA.T, jnp.linalg.solve(LR.T, jnp.linalg.solve(LR, l))))
    )
    return 0.5 * (log_det + trace_plus_const + mahalanobis)


@partial(jax.jit, static_argnums=(2, 3))
@chex.assert_trees_all_shapes_checked
def kl_from_precision_sites(
    A: jax.Array,
    l: jax.Array,
    L: Optional[jax.Array] = None,
    L2: Optional[jax.Array] = None,
):
    """
    Computes the KL divergence KL[q(f)|p(f)].
    """
    chex.assert_rank([A, l], [2, 2])
    chex.assert_axis_dimension(A, 0, A.shape[1])
    chex.assert_axis_dimension(l, 0, A.shape[0])
    if L is not None:
        chex.assert_rank(L, 3)
    if L2 is not None:
        chex.assert_rank(L2, 3)

    if L is None:
        L = jnp.linalg.cholesky(L2)
    m = L.shape[-2]

    C = jnp.linalg.cholesky(A)
    CtL = jnp.matmul(C.T, L)
    W = jnp.eye(m) + jnp.matmul(CtL, CtL.T)
    chol_W = jnp.linalg.cholesky(W)

    log_det = jnp.sum(jnp.log(jnp.square(jnp.diagonal(chol_W))))

    tmp = jnp.linalg.solve(chol_W, CtL.T)
    trace_term = jnp.sum(jnp.square(tmp))

    mahalanobis = jnp.sum(
        jnp.square(jnp.linalg.solve(chol_W, jnp.matmul(L.T, jnp.matmul(A, l))))
    )

    return 0.5 * (log_det - trace_term + mahalanobis)


@partial(jax.jit, static_argnums=(3,))
@chex.assert_trees_all_shapes_checked
def posterior_from_dense_site(
    K: jax.Array,
    lambda_1: jax.Array,
    lambda_2_sqrt: jax.Array,
):
    """
    Returns the mean and cholesky factor of the posterior density.
    """
    chex.assert_rank([K, lambda_1, lambda_2_sqrt], [2, 2, 3])
    chex.assert_axis_dimension(K, 0, K.shape[1])
    chex.assert_axis_dimension(lambda_1, 0, K.shape[0])
    chex.assert_axis_dimension(lambda_2_sqrt, 1, K.shape[0])
    chex.assert_axis_dimension(lambda_2_sqrt, 2, K.shape[1])

    L = lambda_2_sqrt
    m = K.shape[-1]
    Id = jnp.eye(m, dtype=jnp.float64)
    C = jnp.linalg.cholesky(K)

    CtL = jnp.matmul(C, L.T)
    W = Id + jnp.matmul(CtL, CtL.T)
    chol_W = jnp.linalg.cholesky(W)

    LtK = jnp.matmul(L.T, K)
    iwLtK = jnp.linalg.solve(chol_W.T, LtK)
    S_q = K - jnp.matmul(iwLtK.T, iwLtK)
    chol_S_q = jnp.linalg.cholesky(S_q)
    m_q = jnp.matmul(S_q, lambda_1)

    return m_q, chol_S_q


@partial(jax.jit, static_argnums=())
@chex.assert_trees_all_shapes_checked
def gradient_transformation_mean_var_to_expectation(inputs, grads):
    """
    Transforms gradient ∇g of a function wrt [μ, σ²] into its gradients wrt to [μ, σ² + μ²].
    """
    chex.assert_trees_all_equal_shapes(inputs, grads)

    mu, sigma_sq = inputs
    grad_mu, grad_sigma_sq = grads

    nabla_mu = grad_mu - 2.0 * jnp.matmul(grad_sigma_sq, mu)
    return (nabla_mu, grad_sigma_sq)


class BayesianModel(Module):
    linear: Linear

    def __init__(self, key: chex.PRNGKey, in_features: int, out_features: int):
        self.linear = Linear(in_features, out_features, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)
