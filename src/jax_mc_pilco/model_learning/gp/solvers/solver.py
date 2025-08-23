from __future__ import annotations

__all__ = ["Solver"]

from abc import abstractmethod
from typing import Any
from jaxtyping import ArrayLike

import equinox as eqx

from kernels.base import Kernel
from noise import Noise


class Solver(eqx.Module):
    def __init__(
        self,
        kernel: Kernel,
        X: ArrayLike,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ):
        del kernel, X, noise, covariance
        raise NotImplementedError

    # TODO(dfm): Add a deprecation warning. This exists for backwards
    # compatibility, but using __init__ directly is preferred.
    @classmethod
    def init(
        cls,
        kernel: Kernel,
        X: ArrayLike,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ) -> Solver:
        return cls(kernel, X, noise, covariance=covariance)

    @abstractmethod
    def variance(self) -> jax.Array:
        """The diagonal of the covariance matrix"""
        raise NotImplementedError

    @abstractmethod
    def covariance(self) -> jax.Array:
        """The evaluated covariance matrix"""
        raise NotImplementedError

    @abstractmethod
    def normalization(self) -> jax.Array:
        """The multivariate normal normalization constant

        This should be ``(log_det + n*log(2*pi))/2``, where ``n`` is the size of
        the covariance matrix, and ``log_det`` is the log determinant of the
        matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_triangular(self, y: ArrayLike, *, transpose: bool = False) -> jax.Array:
        """Solve the lower triangular linear system defined by this solver

        If the covariance matrix is ``K = L @ L.T`` for some lower triangular
        matrix ``L``, this method solves ``L @ x = y`` for some ``y``. If the
        ``transpose`` parameter is ``True``, this instead solves ``L.T @ x =
        y``.
        """
        raise NotImplementedError

    @abstractmethod
    def dot_triangular(self, y: ArrayLike) -> jax.Array:
        """Compute a matrix product with the lower triangular linear system

        If the covariance matrix is ``K = L @ L.T`` for some lower triangular
        matrix ``L``, this method returns ``L @ y`` for some ``y``.
        """
        raise NotImplementedError

    @abstractmethod
    def condition(self, kernel: Kernel, X_test: ArrayLike | None, noise: Noise) -> Any:
        raise NotImplementedError
