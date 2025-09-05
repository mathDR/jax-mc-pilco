from __future__ import annotations

__all__ = ["DirectSolver"]

from typing import Any
from jaxtyping import ArrayLike
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

from abc import abstractmethod
from typing import Any
from jaxtyping import ArrayLike

import equinox as eqx

from .kernels.base import Kernel


class DirectSolver(eqx.Module):
    """A direct solver that uses ``jax``'s built in Cholesky factorization

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`DirectSolver.init` method instead of the
    usual constructor.
    """

    X: ArrayLike
    variance_value: ArrayLike
    covariance_value: ArrayLike
    scale_tril: ArrayLike

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: ArrayLike,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ):
        """Build a :class:`DirectSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed array with the covariance
                matrix. This should be equal to the result of calling ``kernel``
                and adding ``diag``, but that is not checked.
        """
        self.X = X
        self.variance_value = kernel(X) + noise.diagonal()
        if covariance is None:
            covariance = kernel(X, X) + noise
        self.covariance_value = covariance
        self.scale_tril = linalg.cholesky(covariance, lower=True)

    def variance(self) -> jax.Array:
        return self.variance_value

    def covariance(self) -> jax.Array:
        return self.covariance_value

    def normalization(self) -> jax.Array:
        return jnp.sum(
            jnp.log(jnp.diag(self.scale_tril))
        ) + 0.5 * self.scale_tril.shape[0] * np.log(2 * np.pi)

    def solve_triangular(self, y: ArrayLike, *, transpose: bool = False) -> jax.Array:
        if transpose:
            return linalg.solve_triangular(self.scale_tril, y, lower=True, trans=1)
        else:
            return linalg.solve_triangular(self.scale_tril, y, lower=True)

    def dot_triangular(self, y: ArrayLike) -> jax.Array:
        return jnp.einsum("ij,j...->i...", self.scale_tril, y)

    def condition(
        self, kernel: kernels.Kernel, X_test: ArrayLike | None, noise: Noise
    ) -> Any:
        """Compute the covariance matrix for a conditional GP

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
        """
        if X_test is None:
            Ks = kernel(self.X, self.X)
            Kss = Ks + noise
        else:
            Ks = kernel(self.X, X_test)
            Kss = kernel(X_test, X_test) + noise

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A
