"""
Module to declare Gaussian Exponential Family sites objects using JAX, Equinox, and Chex.
"""

import abc
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import chex


class Sites(eqx.Module, metaclass=abc.ABCMeta):
    """
    The base sites class
    """

    pass


class DiagSites(Sites):
    """
    Sites with diagonal lambda_2
    """

    lambda_1: chex.Array
    lambda_2: chex.Array

    def __init__(self, lambda_1: chex.Array, lambda_2: chex.Array):
        """
        :param lambda_1: first order natural parameter [M, P]
        :param lambda_2: second order natural parameter [M, P]
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # Assertions using chex
        chex.assert_rank([self.lambda_1, self.lambda_2], [2, 2])
        chex.assert_trees_all_equal_shapes(self.lambda_1, self.lambda_2)
        chex.assert_scalar_in_tree(self.lambda_2, lambda x: jnp.all(x > 0))


class DenseSites(Sites):
    """
    Sites with dense lambda_2 saved as a Cholesky factor.
    """

    lambda_1: chex.Array
    _lambda_2_sqrt: Optional[chex.Array] = eqx.field(default=None)
    _lambda_2: Optional[chex.Array] = eqx.field(default=None)
    factor: bool = eqx.field(init=False)
    num_latent_gps: int = eqx.field(init=False)

    def __init__(
        self,
        lambda_1: chex.Array,
        lambda_2_sqrt: Optional[chex.Array] = None,
        lambda_2: Optional[chex.Array] = None,
    ):
        """
        :param lambda_1: first order natural parameter [P, M]
        :param lambda_2_sqrt: Cholesky factor of the second order natural parameter [P, M, M]
        :param lambda_2: second order natural parameter [P, M, M]
        """
        self.lambda_1 = lambda_1
        self.num_latent_gps = lambda_1.shape[0]

        chex.assert_rank(self.lambda_1, 2)

        if (lambda_2_sqrt is None) == (lambda_2 is None):
            raise ValueError(
                "Exactly one of lambda_2_sqrt or lambda_2 must be provided."
            )

        if lambda_2_sqrt is not None:
            self.factor = True
            self._lambda_2_sqrt = lambda_2_sqrt
            chex.assert_rank(self._lambda_2_sqrt, 3)
            chex.assert_axis_dimension(self._lambda_2_sqrt, 0, self.num_latent_gps)
            chex.assert_axis_dimension(
                self._lambda_2_sqrt, 1, self._lambda_2_sqrt.shape[2]
            )
        else:
            self.factor = False
            self._lambda_2 = lambda_2
            chex.assert_rank(self._lambda_2, 3)
            chex.assert_axis_dimension(self._lambda_2, 0, self.num_latent_gps)
            chex.assert_axis_dimension(self._lambda_2, 1, self._lambda_2.shape[2])

    @property
    def lambda_2(self) -> chex.Array:
        """Second natural parameter."""
        if self.factor:
            return jnp.einsum("pmn,pkn->pmk", self._lambda_2_sqrt, self._lambda_2_sqrt)
        return self._lambda_2

    @property
    def lambda_2_sqrt(self) -> chex.Array:
        """Cholesky factor of the second natural parameter."""
        if self.factor:
            return self._lambda_2_sqrt
        return jnp.linalg.cholesky(self._lambda_2)
