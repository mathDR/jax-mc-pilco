from collections.abc import Sequence
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
#from typeguard import typechecked

from jax_mc_pilco.model_learning.flows._types import _BATCH_ARRAY, _LABEL_ARRAY, _OT_PLANS, _TIME
from jax_mc_pilco.model_learning.flows.optimal_transport import OTSampler


def _reshape_t(
    t: _TIME, bs: int, ndim: int
) -> Union[Float[Array, " bs"], Int[Array, "bs"]]:
    return (jnp.ones(bs) * t).reshape(bs, *[1 for _ in jnp.arange(ndim)])


def _random_t(key: PRNGKeyArray, shape: Sequence[int]) -> Float[Array, "..."]:
    return jax.random.uniform(
        key, shape=shape, dtype=jnp.float32, minval=0, maxval=1)


class CFM(eqx.Module):
    """Independent Coupling Conditional Flow Matching Class

    The class provides OT-free (independent) conditional flow matching from
    [Tong2023a]_. Additionally, it serves as the base class for all other
    implemented conditional flow matching schemes.

    This class is implemented as an :py:class:`eqx.Module` and thus compatible
    with :py:meth:`jax.jit`.

    References
    ----------
    .. [Tong2023a] A. Tong et al., "Improving and generalizing flow-based
       generative models with minibatch optimal transport", *Internation
       Conference on Machine Learning*, 2023
    """

    __slots__ = ("key", "sigma")

    sigma: Union[int, float]

    def __init__(self, sigma: Union[int, float] = .0):
        r"""Initialize a new CFM instance

        sigma: int | float, 0.0
            Hyperparameter reflecting the standard deviation of gaussian
            probability path
        """
        self.sigma = sigma

    @staticmethod
    def _mu_t(x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, t: _TIME) -> _BATCH_ARRAY:
        t = _reshape_t(t, x0.shape[0], x0.ndim)
        return t * x1 + (1 - t) * x0

    def _sigma_t(self, *args, **kwargs) -> Union[int, float]:
        return self.sigma

    def sample_xt(
        self, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, t: _TIME,
        epsilon: _BATCH_ARRAY
    ) -> _BATCH_ARRAY:
        r"""Sample from the probability path

        The probability path is defined as the following gaussian distribution
        :math:`\mathcal{N}(t \cdot x_1 + (1 - t) \cdot x_0, \sigma`
        (Eq. 14 in [Tong2023a]_)

        Parameters
        ----------
        x0: jnp.ndarray, shape (n, ...)
        x1: jnp.ndarray, shape (n, ...)
        t: jnp.ndarray | float | int, shape (n)
        epsilon: jnp.ndarray, shape (n)

        Returns
        -------
        jnp.ndarray, shape (n, ...)
            array of n random samples from the probability path
        """
        mu_t = self._mu_t(x0, x1, t)
        sigma_t = _reshape_t(self._sigma_t(t), x0.shape[0], x0.ndim)
        return mu_t + sigma_t * epsilon

    @staticmethod
    def conditional_flow(
        x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, *args, **kwargs
    ) -> _BATCH_ARRAY:
        return x1 - x0

    @staticmethod
    def sample_noise(
        key: PRNGKeyArray, shape: tuple[int, ...]
    ) -> Float[Array, "..."]:
        return jax.random.normal(key, shape, dtype=float)

    #@jaxtyped(typechecker=typechecked)
    def sample_location_and_conditional_flow(
        self, key: PRNGKeyArray,
        x0: Float[Array, "bs ..."], x1: Float[Array, "bs ..."],
        t: Float[Array, " bs"] | None = None
    ) -> tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]:
        subkey1, subkey2 = jax.random.split(key, 2)

        if t is None:
            t = _random_t(subkey1, [x0.shape[0]]).reshape(-1)

        noise = self.sample_noise(subkey2, x0.shape)
        xt = self.sample_xt(x0, x1, t, noise)
        ut = self.conditional_flow(x0, x1, t, xt)

        return t, xt, ut, noise

    @jax.named_scope("jaxcfm.CFM")
    def __call__(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        t: Float[Array, " bs"] | None = None
    ) -> tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]:
        return self.sample_location_and_conditional_flow(key, x0, x1, t)

    def compute_lambda(self, t: _TIME) -> _TIME:
        return 2 * self._sigma_t(t) / (self.sigma**2 + 1e-8)


class _OTSampleCFM(CFM):

    __slots__ = ("sampler")

    sampler: OTSampler

    def __init__(self, sigma: Union[int, float], **kwargs):
        super().__init__(sigma)
        self.sampler = OTSampler(**kwargs)

    def __call__(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        t: Float[Array, " bs"] | None = None
    ) -> Union[
        tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY],
        tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]
    ]:
        subkey1, subkey2 = jax.random.split(key, 2)
        x0, x1 = self.sampler.sample_plan(subkey1, x0, x1)
        return self.sample_location_and_conditional_flow(subkey2, x0, x1, t)

    def guided_sample_location_and_conditional_flow(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        y0: _LABEL_ARRAY | None = None,
        y1: _LABEL_ARRAY | None = None,
        t: Float[Array, " bs"] | None = None
    ) -> Union[
        tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY],
        tuple[
            Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY,
            _LABEL_ARRAY, _LABEL_ARRAY
        ]
    ]:
        if y0 is None and y1 is None:
            return self.__call__(x0, x1, t)

        subkey1, subkey2 = jax.random.split(key, 2)
        x0, x1, y0, y1 = self.sampler.sample_plan(
            subkey1, x0, x1, y0, y1)
        cf_sample = self.sample_location_and_conditional_flow(
            subkey2, x0, x1, t)
        return *cf_sample[:-1], y0, y1, cf_sample[-1]


class ExactOptimalTransportCFM(_OTSampleCFM):
    def __init__(
        self, sigma: Union[int, float] = 0.0, **kwargs
    ):
        super().__init__(sigma, solver="exact", **kwargs)

    @jax.named_scope("jaxcfm.ExactOptimalTransportCFM")
    def __call__(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        t: Float[Array, " bs"] | None = None
    ) -> tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]:
        return super().__call__(key, x0, x1, t)


class SchrodingerBridgeCFM(_OTSampleCFM):
    def __init__(
        self, sigma: Union[int, float],
        solver: _OT_PLANS = "sinkhorn", **kwargs
    ):
        super().__init__(sigma, solver=kwargs.pop("solver", solver), **kwargs)

    def _sigma_t(self, t: _TIME) -> _BATCH_ARRAY:
        return self.sigma * jnp.sqrt(t * (1 - t))

    def conditional_flow(
        self, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, t: _TIME, xt: _BATCH_ARRAY
    ) -> _BATCH_ARRAY:
        mu_t = self._mu_t(x0, x1, t)
        t = _reshape_t(t, x0.shape[0], x0.ndim)
        t_prime = (1 - 2 * t)
        return t_prime / (2 * t * (1 - t) + 1e-8) * (xt - mu_t) + x1 - x0

    @jax.named_scope("jaxcfm.SchrodingerBridgeCFM")
    def __call__(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        t: Float[Array, " bs"] | None = None,
    ) -> Union[
        tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY],
        tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]
    ]:
        return super().__call__(key, x0, x1, t)


class TargetCFM(CFM):

    @staticmethod
    def _mu_t(x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, t: _TIME) -> _BATCH_ARRAY:
        raise NotImplementedError

    def _sigma_t(self, t: _TIME) -> _BATCH_ARRAY:
        raise NotImplementedError

    @staticmethod
    def conditional_flow(
        x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, *args, **kwargs
    ) -> _BATCH_ARRAY:
        raise NotImplementedError

    @jax.named_scope("jaxcfm.TargetCFM")
    def __call__(
        self, key: PRNGKeyArray, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY,
        t: Float[Array, " bs"] | None = None,
    ) -> tuple[Float[Array, " bs"], _BATCH_ARRAY, _BATCH_ARRAY, _BATCH_ARRAY]:
        return super().__call__(key, x0, x1, t)


class VariancePreservingCFM(CFM):

    @staticmethod
    def _mu_t(x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, t: _TIME) -> _BATCH_ARRAY:
        raise NotImplementedError

    @staticmethod
    def conditional_flow(
        x0: _BATCH_ARRAY, x1: _BATCH_ARRAY, *args, **kwargs
    ) -> _BATCH_ARRAY:
        raise NotImplementedError
