"""The mean function classes."""

from __future__ import annotations
from jaxtyping import Float
__all__ = [
    "Mean",
    "CustomMean",
    "SumMean",
    "ProductMean",
    "ConstantMean",
    "LinearMean",
    "ZeroMean",
]

from abc import abstractmethod
from jaxtyping import ArrayLike

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class Mean(eqx.Module):
    """A base class for GP Means"""

    @abstractmethod
    def evaluate(self, X: ArrayLike) -> jax.Array:
        """Evaluate the mean at the inputs

        This should be overridden be subclasses to return the kernel-specific
        value. Two things to note:

        1. Users shouldn't generally call :func:`Mean.evaluate`. Instead,
           always "call" the kernel instance directly; for example, you can
           evaluate the ConstantMean (with the constant being 1.0) using
           ``ConstantMean(1.0)(x)``, for an array of inputs ``x``.
        2. When implementing a custom mean, this method should treat ``X``
           as a single datapoint. In other words, the input will
           typically either be a scalar of shape ``n_dim``, where ``n_dim``
           is the number of input dimensions, rather than ``n_data`` or
           ``(n_data, n_dim)``, and you should let the :class:`Mean` ``vmap``
           magic handle all the broadcasting for you.
        """
        del X
        raise NotImplementedError

    def __call__(self, X: ArrayLike | None = None) -> jax.Array:
        m = jax.vmap(self.evaluate, in_axes=0)(X)
        return m

    def __add__(self, other: Mean | ArrayLike) -> Mean:
        if isinstance(other, Mean):
            return SumMean(self, other)
        return SumMean(self, ConstantMean(other))

    def __radd__(self, other: Any) -> Mean:
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if isinstance(other, Mean):
            return SumMean(other, self)
        return SumMean(ConstantMean(other), self)

    def __mul__(self, other: Mean | ArrayLike) -> Mean:
        if isinstance(other, Mean):
            return ProductMean(self, other)
        return ProductMean(self, ConstantMean(other))

    def __rmul__(self, other: Any) -> Mean:
        if isinstance(other, Mean):
            return ProductMean(other, self)
        return ProductMean(ConstantMean(other), self)


class ZeroMean(Mean):
    r"""This mean returns zero

    .. math::

        m(\mathbf{x}) = 0

    """

    def evaluate(self, X: ArrayLike) -> Float:
        return 0.0


class ConstantMean(Mean):
    r"""This mean returns the constant

    .. math::

        m(\mathbf{x}) = c

    where :math:`c` is a parameter.

    Args:
        c: The parameter :math:`c` in the above equation.
    """

    value: jax.Array | Float

    def evaluate(self, X: ArrayLike) -> jax.Array:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant mean must be a scalar")
        return self.value


class LinearMean(Mean):
    r"""This mean returns the linear result

    .. math::

        m(\mathbf{x}) = value[0] + value[1] * X

    where :math:`value` is a parameter.

    Args:
        value: The parameter :math:`value` in the above equation.
    """

    value: jax.Array | float

    def evaluate(self, X: ArrayLike) -> jax.Array:
        if jnp.ndim(self.value) != 2:
            raise ValueError(
                "The value of a linear mean must have two elements"
                )
        return jnp.asarray(self.value[0] + self.value[1] * X)


class CustomMean(Mean):
    """A custom mean class implemented as a callable

    Args:
        function: A callable with a signature and behavior that matches
            :func:`Mean.evaluate`.
    """

    function: Callable[[Any], Any] = eqx.field(static=True)

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.function(X)


class SumMean(Mean):
    """A helper to represent the sum of two means"""

    mean1: Mean
    mean2: Mean

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.mean1.evaluate(X) + self.mean2.evaluate(X)


class ProductMean(Mean):
    """A helper to represent the product of two kernels"""

    mean1: Mean
    mean2: Mean

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.mean1.evaluate(X) * self.mean2.evaluate(X)
