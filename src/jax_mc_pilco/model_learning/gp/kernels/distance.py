"""
This submodule defines a set of distance metrics that can be used when working
with multivariate data. By default, all
:class:`kernels.stationary.Stationary` kernels will use either an
:class:`L1Distance` or :class:`L2Distance`, when applied in multiple
dimensions, but it is possible to define custom metrics
"""

from __future__ import annotations

__all__ = ["Distance", "L1Distance", "L2Distance"]

from abc import abstractmethod

import equinox as eqx  # type: ignore
import jax.numpy as jnp
from jax import Array, vmap
from jaxtyping import ArrayLike


class Distance(eqx.Module):
    """An abstract base class defining a distance metric interface"""

    @abstractmethod
    def distance(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        """Compute the distance between two coordinates under this metric"""
        raise NotImplementedError()

    def squared_distance(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        """Compute the squared distance between two coordinates

        By default this returns the squared result of
        :func:`tinygp.kernels.stationary.Distance.distance`, but some metrics
        can take advantage of these separate implementations to avoid
        unnecessary square roots.
        """
        return jnp.square(self.distance(X1, X2))


class L1Distance(Distance):
    """The L1 or Manhattan distance between two coordinates"""

    def distance(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        def dist_func(x: ArrayLike, y: ArrayLike) -> Array:
            return jnp.sum(jnp.abs(x - y))
        return vmap(vmap(dist_func, (None, 0)), (0, None))(X1, X2)


class L2Distance(Distance):
    """The L2 or Euclidean distance between two coordinates"""

    def distance(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        r1 = L1Distance().distance(X1, X2)
        r2 = self.squared_distance(X1, X2)
        zeros = jnp.equal(r2, 0)
        r2 = jnp.where(zeros, jnp.ones_like(r2), r2)
        return jnp.where(zeros, r1, jnp.sqrt(r2))

    def squared_distance(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        def dist_func(x: ArrayLike, y: ArrayLike) -> Array:
            return jnp.sum(jnp.square(x - y))
        return vmap(vmap(dist_func, (None, 0)), (0, None))(X1, X2)
