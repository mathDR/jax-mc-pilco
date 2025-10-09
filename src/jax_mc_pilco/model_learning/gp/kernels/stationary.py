"""
Many of the most commonly used kernels are implemented as subclasses of the
:class:`Stationary` kernel. This means that each kernel in this section has (at
least) the two parameters:

- ``scale``: A scalar lengthscale for the kernel in the radial distance
  specified by ``distance``, and
- ``distance``: A :class:`kernels.distance.Distance` metric specifying
  how to compute the scalar distance between two input coordinates.

Most of these kernels use the :class:`kernels.distance.L1Distance` metric
by default, and ``scale`` defaults to ``1``.
"""

from __future__ import annotations

__all__ = [
    "Stationary",
    "Exp",
    "ExpSquared",
    "Matern32",
    "Matern52",
    "Cosine",
    "ExpSineSquared",
    "RationalQuadratic",
    "SpectralMixture",
]


import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_mc_pilco.model_learning.gp.kernels.base import Kernel, softplus
from jax_mc_pilco.model_learning.gp.kernels.distance import (
    Distance,
    L1Distance,
    L2Distance,
)


class Stationary(Kernel):
    """A stationary kernel is defined with respect to a distance metric

    Note that a stationary kernel is *always* isotropic. If you need more
    non-isotropic length scales, wrap your kernel in a transform using
    :class:`transforms.Linear` or :class:`transforms.Cholesky`.

    Args:
        coefficient: The coefficient of the kernel.  This must be a scalar
        scale: The length scale, in the same units as ``distance`` for the
            kernel. This must be a scalar.
        distance: An object that implements ``distance`` and
            ``squared_distance`` methods. Typically a subclass of
            :class:`kernels.stationary.Distance`. Each stationary kernel
            also has a ``default_distance`` property that is used when
            ``distance`` isn't provided.
    """

    coefficient: Array | float = eqx.field(
        default_factory=lambda: jnp.ones(())
    )
    log_scale: Array | float = eqx.field(default_factory=lambda: jnp.zeros(()))
    distance: Distance = eqx.field(default_factory=L1Distance)


class Exp(Stationary):
    r"""The exponential kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: Array, X2: Array) -> Array:
        if jnp.ndim(self.log_scale):
            raise ValueError(
                "Only scalar scales are permitted for stationary kernels; use"
                "transforms.Linear or transforms.Cholesky for more flexiblity"
            )
        sqrt_scale = jnp.sqrt(softplus(self.log_scale))
        return self.coefficient * jnp.exp(
            -self.distance.distance(X1 / sqrt_scale, X2 / sqrt_scale)
        )


class ExpSquared(Stationary):
    r"""The exponential squared or radial basis function kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r^2 / 2)

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
    """

    distance: Distance = eqx.field(default_factory=L2Distance)

    def evaluate(self, X1: Array, X2: Array) -> Array:
        scale = softplus(self.log_scale)
        r2 = self.distance.squared_distance(X1 / scale, X2 / scale)
        return self.coefficient * jnp.exp(-0.5 * r2)


class Matern32(Stationary):
    r"""The Matern-3/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{3}\,r)\,\exp(-\sqrt{3}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: Array, X2: Array) -> Array:
        sqrt_scale = jnp.sqrt(softplus(self.log_scale))
        r = self.distance.distance(X1 / sqrt_scale, X2 / sqrt_scale)
        arg = np.sqrt(3) * r
        return self.coefficient * (1. + arg) * jnp.exp(-arg)


class Matern52(Stationary):
    r"""The Matern-5/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{5}\,r +
            5\,r^2/3)\,\exp(-\sqrt{5}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: Array, X2: Array) -> Array:
        sqrt_scale = jnp.sqrt(softplus(self.log_scale))
        r = self.distance.distance(X1 / sqrt_scale, X2 / sqrt_scale)
        arg = np.sqrt(5) * r
        return (
            self.coefficient
            * (1. + arg + jnp.square(arg) / 3.)
            * jnp.exp(-arg)
        )


class Cosine(Stationary):
    r"""The cosine kernel/ softplus(self.log_scale)

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \cos(2\,\pi\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        scale: The parameter :math:`P`.
    """

    def evaluate(self, X1: Array, X2: Array) -> Array:
        sqrt_scale = jnp.sqrt(softplus(self.log_scale))
        r = self.distance.distance(X1 / sqrt_scale, X2 / sqrt_scale)
        return self.coefficient * jnp.cos(2 * jnp.pi * r)


class ExpSineSquared(Stationary):
    r"""The exponential sine squared or quasiperiodic kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-\Gamma\,\sin^2 \pi r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        scale: The parameter :math:`P`.
        gamma: The parameter :math:`\Gamma`.
    """

    gamma: Array | float | None = None

    def __check_init__(self):
        if self.gamma is None:
            raise ValueError("Missing required argument 'gamma'")

    def evaluate(self, X1: Array, X2: Array) -> Array:
        assert self.gamma is not None
        sqrt_scale = jnp.sqrt(softplus(self.log_scale))
        r = self.distance.distance(X1 / sqrt_scale, X2 / sqrt_scale)
        return self.coefficient * jnp.exp(
            -self.gamma * jnp.square(jnp.sin(jnp.pi * r))
        )


class RationalQuadratic(Stationary):
    r"""The rational quadratic

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + r^2 / 2\,\alpha)^{-\alpha}

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
        alpha: The parameter :math:`\alpha`.
    """

    alpha: Array | float | None = None

    def __check_init__(self):
        if self.alpha is None:
            raise ValueError("Missing required argument 'alpha'")

    def evaluate(self, X1: Array, X2: Array) -> Array:
        assert self.alpha is not None
        r2 = self.distance.squared_distance(
            X1 / softplus(self.log_scale),
            X2 / softplus(self.log_scale)
        )
        return (
            self.coefficient
            * (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
        )


class SpectralMixture(Stationary):
    r"""The [spectral mixture](https://arxiv.org/pdf/1302.4245) kernel.

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) =
        \sum_{q=1}{Q}w_q\prod_{n=1}{N}
          \exp\(-2\pi^2r_n^2v_q^{(n)})\cos(2\pi r_n\mu_p^{(n)})

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j)||

    Args:
        weight: The parameter
        log_scale: The parameter :math:`log(exp(v)-1)`.
        freq: The parameter :math:`\mu`.
    """
    weight: Array | float | None = None
    log_scale: Array | float | None = None
    freq: Array | float | None = None

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(
                    -2. * jnp.square(
                        jnp.pi * tau / softplus(self.log_scale)
                        )
                        )
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=0,
            )
        )
