"""MultiOutput Kernels."""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Int, Num, Sequence

from jax_mc_pilco.model_learning.gp.kernels.base import Kernel


# TODO describe various output shapes
class MultioutputKernel(Kernel):
    """
    Multi Output Kernel class.

    This kernel can represent correlation between outputs of different datapoints.

    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """
    kernels: Sequence[Kernel]

    @property
    @abstractmethod
    def num_latent_gps(self) -> Int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_kernels(self) -> tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abstractmethod
    def K(
        self,
        X: ArrayLike,
        X2: ArrayLike | None = None,
        full_output_cov: Bool = True,
    ) -> Array:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param X2: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)]
        """
        raise NotImplementedError

    @abstractmethod
    def K_diag(self, X: ArrayLike, full_output_cov: Bool = True) -> Array:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)]
        """
        raise NotImplementedError

    def __call__(
        self,
        X: ArrayLike,
        X2: ArrayLike | None = None,
        *,
        full_cov: Bool = False,
        full_output_cov: Bool = True,
    ) -> Array:
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)


class SharedIndependent(MultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.

    .. warning::
       This class is created only for testing and comparison purposes.
       Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dim: Int) -> None:
        super().__init__()
        self.kernels = tuple(kernel)
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> Int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return self.kernels

    def K(
        self,
        X: ArrayLike,
        X2: ArrayLike | None = None,
        full_output_cov: Bool = True,
    ) -> Array:
        if X2 is None:
            K = self.kernel(X)
        else:
            K = self.kernel(X, X2)

        if full_output_cov:
            return jnp.tile(
                jnp.tile(K, (self.num_latent_gps, 1)), (1, self.num_latent_gps)
            )
        else:
            return jnp.kron(jnp.eye(self.num_latent_gps, dtype=int), K)

    def K_diag(self, X: ArrayLike, full_output_cov: Bool = True) -> Array:
        K = self.kernel.evaluate_diag(X)
        Ks = jnp.tile(K, (self.num_latent_gps, 1))
        return jnp.diag(Ks) if full_output_cov else Ks


class Independent(MultioutputKernel):
    """
    A stack of independent GP priors. 'kernels' is a list of GP kernels, and this class stacks
    the state space models such that each component is fed to the likelihood.
    This class differs from Sum only in the measurement model.
    """
    def __init__(self, kernels: Sequence[Kernel]):
        self.num_kernels = len(kernels)
        self.kernels = kernels

    def K(self, X: ArrayLike, X2: ArrayLike | None = None) -> Array:
        zeros = jnp.zeros(self.num_kernels)
        K0 = self.kernels[0].K(X, X2)
        index_vector = zeros.at[0].add(1.)
        Kstack = jnp.kron(K0, jnp.diag(index_vector))
        for i in range(1, self.num_kernels):
            index_vector = zeros.at[i].add(1.)
            Kstack += np.kron(self.kernels[i].K(X, X2), jnp.diag(index_vector))
        return Kstack

    def K_diag(self, X: ArrayLike, full_output_cov: Bool = True) -> Array:
        zeros = jnp.zeros(self.num_kernels)
        K0 = self.kernels[0].evaluate_diag(X)
        index_vector = zeros.at[0].add(1.)
        Kstack = jnp.kron(K0, jnp.diag(index_vector))
        for i in range(1, self.num_kernels):
            index_vector = zeros.at[i].add(1.)
            Kstack += np.kron(self.kernels[i].evaluate_diag(X), jnp.diag(index_vector))
        return jnp.diag(Kstack) if full_output_cov else Kstack
