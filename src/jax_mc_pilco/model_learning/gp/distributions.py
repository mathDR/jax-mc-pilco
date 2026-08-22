# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from beartype.typing import (
    Optional,
)
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import is_prng_key


class GaussianDistribution(Distribution):
    support = constraints.real_vector

    def __init__(
        self,
        loc: Optional[Float[Array, " N"]],
        scale: Optional[Float[Array, "N N"]],
        validate_args=None,
    ):
        self.loc = loc
        self.scale = scale
        batch_shape = ()
        event_shape = jnp.shape(self.loc)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        # Obtain covariance root.
        covariance_root = jsp.linalg.cho_factor(self.scale, lower=True)

        # Gather n samples from standard normal distribution Z = [z₁, ..., zₙ]ᵀ.
        white_noise = jr.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )

        # xᵢ ~ N(loc, cov) <=> xᵢ = loc + sqrt zᵢ, where zᵢ ~ N(0, I).
        def affine_transformation(_x):
            return self.loc + covariance_root @ _x

        return vmap(affine_transformation)(white_noise)

    @property
    def mean(self) -> Float[Array, " N"]:
        r"""Calculates the mean."""
        return self.loc

    @property
    def variance(self) -> Float[Array, " N"]:
        r"""Calculates the variance."""
        return diag(self.scale)

    def entropy(self) -> Float:
        r"""Calculates the entropy of the distribution."""
        s, ld = jsp.linalg.slogdet(self.scale)
        return 0.5 * (
            self.event_shape[0] * (1.0 + jnp.log(2.0 * jnp.pi)) + s*ld
        )

    def median(self) -> Float[Array, " N"]:
        r"""Calculates the median."""
        return self.loc

    def mode(self) -> Float[Array, " N"]:
        r"""Calculates the mode."""
        return self.loc

    def covariance(self) -> Float[Array, "N N"]:
        r"""Calculates the covariance matrix."""
        return self.scale

    @property
    def covariance_matrix(self) -> Float[Array, "N N"]:
        r"""Calculates the covariance matrix."""
        return self.covariance()

    def stddev(self) -> Float[Array, " N"]:
        r"""Calculates the standard deviation."""
        return jnp.sqrt(diag(self.scale))

    def log_prob(self, y: Float[Array, " N"]) -> Float:
        r"""Calculates the log pdf of the multivariate Gaussian.

        Args:
            y: the value of which to calculate the log probability.

        Returns:
            The log probability of the value as a scalar array.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]

        # diff, y - µ
        diff = y - mu

        # compute the pdf, -1/2[ n log(2π) + log|Σ| + (y - µ)ᵀΣ⁻¹(y - µ) ]
        s, ld = jsp.linalg.slogdet(sigma)
        return -0.5 * (
            n * jnp.log(2.0 * jnp.pi) + s*ld + diff.T @ solve(sigma, diff)
        )

    def kl_divergence(self, other: "GaussianDistribution") -> Float:
        return _kl_divergence(self, other)


def _check_and_return_dimension(
    q: GaussianDistribution, p: GaussianDistribution
) -> int:
    r"""Checks that the dimensions of the distributions are compatible."""
    if q.event_shape != p.event_shape:
        raise ValueError(
            "Distribution event shapes are not compatible: `q.event_shape ="
            f" {q.event_shape}` and `p.event_shape = {p.event_shape}`. Please check"
            " your mean and covariance shapes."
        )

    return q.event_shape[-1]


def _frobenius_norm_squared(matrix: Float[Array, "N N"]) -> Float:
    r"""Calculates the squared Frobenius norm of a matrix."""
    return jnp.sum(jnp.square(matrix))


def _kl_divergence(q: GaussianDistribution, p: GaussianDistribution) -> Float:
    r"""KL-divergence between two Gaussians.

    Computes the KL divergence, $\operatorname{KL}[q\mid\mid p]$, between two
    multivariate Gaussian distributions $q(x) = \mathcal{N}(x; \mu_q, \Sigma_q)$
    and $p(x) = \mathcal{N}(x; \mu_p, \Sigma_p)$.

    Args:
        q: a multivariate Gaussian distribution.
        p: another multivariate Gaussian distribution.

    Returns:
        Float: The KL divergence between q and p.
    """
    n_dim = _check_and_return_dimension(q, p)

    # Extract q mean and covariance.
    mu_q = q.loc
    sigma_q = q.scale

    # Extract p mean and covariance.
    mu_p = p.loc
    sigma_p = p.scale

    # Find covariance roots.
    sqrt_p = jsp.linalg.cho_factor(sigma_p, lower=True)
    sqrt_q = jsp.linalg.cho_factor(sigma_q, lower=True)

    # diff, μp - μq
    diff = mu_p - mu_q

    # trace term, tr[Σp⁻¹ Σq] = tr[(LpLpᵀ)⁻¹(LqLqᵀ)] = tr[(Lp⁻¹Lq)(Lp⁻¹Lq)ᵀ] = (fr[LqLp⁻¹])²
    trace = _frobenius_norm_squared(
        jsp.linalg.cho_solve(sqrt_p, sqrt_q)
    )

    # Mahalanobis term, (μp - μq)ᵀ Σp⁻¹ (μp - μq) = tr [(μp - μq)ᵀ [LpLpᵀ]⁻¹ (μp - μq)] = (fr[Lp⁻¹(μp - μq)])²
    mahalanobis = jnp.sum(jnp.square(jsp.linalg.cho_solve(sqrt_p, diff)))

    # KL[q(x)||p(x)] = [ [(μp - μq)ᵀ Σp⁻¹ (μp - μq)] - n - log|Σq| + log|Σp| + tr[Σp⁻¹ Σq] ] / 2
    sq, ldq = jsp.linalg.slogdet(sigma_q)
    sp, ldp = jsp.linalg.slogdet(sigma_p)
    return 0.5 * (mahalanobis - n_dim - sq*ldq + sp*ldp + trace)


__all__ = [
    "GaussianDistribution",
]