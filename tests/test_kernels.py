import pytest
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
from numpy.testing import assert_allclose
from typing import Dict
from jax_mc_pilco.model_learning.gp.kernels.base import Kernel, softplus

from jax_mc_pilco.model_learning.gp.kernels.stationary import (
    Stationary,
    Exp,
    ExpSquared,
    Matern32,
    Matern52,
    Cosine,
    ExpSineSquared,
    RationalQuadratic,
    SpectralMixture,
)


@pytest.fixture
def params() -> Dict:
    return {
        "kernel": {
            "coefficient": jnp.array(0.9),
            "log_scale": jnp.array(0.0),
        },
        "mean": {},
        "likelihood": {"log_diag": jnp.array(-0.5)},
    }


@pytest.fixture
def X() -> jax.Array:
    return jnp.array([1, 2, 3])[:, jnp.newaxis]


@pytest.fixture
def rbf_K(params: Dict) -> Kernel:
    return ExpSquared(**params["kernel"])


def test_squared_distance_function(K: Kernel, X: ArrayLike):
    expected_result = np.array([[0, 1, 4], [1, 0, 1], [0, 4, 1]])
    assert_all_close(K.distance.squared_distance(X, X).numpy(), expected_result)


def test_distance_function(K: Kernel, X: ArrayLike):
    expected_result = np.array([[0, 1, 2], [1, 0, 1], [0, 2, 1]])
    assert_all_close(K.distance.distance(X, X).numpy(), expected_result)
