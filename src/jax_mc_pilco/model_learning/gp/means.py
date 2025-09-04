"""The mean function classes."""

from __future__ import annotations

__all__ = [
    "Mean",
    "Custom",
    "Sum",
    "Product",
    "Constant",
    "Linear",
]

from jaxtyping import ArrayLike

from typing import TYPE_CHECKING, Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class Mean(eqx.Module):
    """A base class for GP Means"""

    value: ArrayLike | None = None

    def __init__(self, value: ArrayLike):
        self.value = value

    def __call__(self, X: ArrayLike) -> jax.Array:
        raise NotImplementedError

    def __add__(self, other: Mean | ArrayLike) -> Mean:
        if isinstance(other, Mean):
            return Sum(self, other)
        return Sum(self, Constant(other))

    def __radd__(self, other: Any) -> Mean:
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if isinstance(other, Mean):
            return Sum(other, self)
        return Sum(Constant(other), self)

    def __mul__(self, other: Mean | ArrayLike) -> Mean:
        if isinstance(other, Mean):
            return Product(self, other)
        return Product(self, Constant(other))

    def __rmul__(self, other: Any) -> Mean:
        if isinstance(other, Mean):
            return Product(other, self)
        return Product(Constant(other), self)


class Constant(Mean):
    r"""This mean returns the constant

    .. math::

        m(\mathbf{x}) = c

    where :math:`c` is a parameter.

    Args:
        c: The parameter :math:`c` in the above equation.
    """

    value: jax.Array | float

    def evaluate(self, X: ArrayLike) -> jax.Array:
        if jnp.ndim(self.value) != 0:
            raise ValueError("The value of a constant mean must be a scalar")
        return jnp.asarray(self.value)


class Linear(Mean):
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
            raise ValueError("The value of a linear mean must have two elements")
        return jnp.asarray(self.value[0] + self.value[1] * X)


class Custom(Mean):
    """A custom mean class implemented as a callable

    Args:
        function: A callable with a signature and behavior that matches
            :func:`Kernel.evaluate`.
    """

    function: Callable[[Any], Any] = eqx.field(static=True)

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.function(X)


class Sum(Mean):
    """A helper to represent the sum of two means"""

    mean1: Mean
    mean2: Mean

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.mean1.evaluate(X) + self.mean2.evaluate(X)


class Product(Kernel):
    """A helper to represent the product of two kernels"""

    mean1: Mean
    mean2: Mean

    def evaluate(self, X: ArrayLike) -> jax.Array:
        return self.mean1.evaluate(X) * self.mean2.evaluate(X)


class Polynomial(Kernel):
    r"""A polynomial kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = [(\mathbf{x}_i / \ell) \cdot
            (\mathbf{x}_j / \ell) + \sigma^2]^P

    Args:
        order: The power :math:`P`.
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """

    order: ArrayLike | float
    scale: ArrayLike | float = eqx.field(default_factory=lambda: jnp.ones(()))
    sigma: ArrayLike | float = eqx.field(default_factory=lambda: jnp.zeros(()))

    def evaluate(self, X1: ArrayLike, X2: ArrayLike) -> jax.Array:
        return (
            (X1 / self.scale) @ (X2 / self.scale) + jnp.square(self.sigma)
        ) ** self.order
