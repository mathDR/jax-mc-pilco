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
            "log_scale": jnp.array(jnp.log(jnp.exp(1.0) - 1.0)),
        },
        "mean": {},
        "likelihood": {"log_diag": jnp.array(-0.5)},
    }


@pytest.fixture
def X() -> jax.Array:
    return jnp.array([1.0, 2.0, 3.0])[:, jnp.newaxis]


@pytest.fixture
def rbf_K(params: Dict) -> Kernel:
    return ExpSquared(**params["kernel"])


def test_squared_distance_function(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    assert_allclose(np.asarray(rbf_K.distance.squared_distance(X, X)), expected_result)


def test_distance_function(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    assert_allclose(np.asarray(rbf_K.distance.distance(X, X)), expected_result)


def test_rbf_kernel(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array(
        [
            [0.9, 0.9 * np.exp(-0.5), 0.9 * np.exp(-2.0)],
            [0.9 * np.exp(-0.5), 0.9, 0.9 * np.exp(-0.5)],
            [0.9 * np.exp(-2.0), 0.9 * np.exp(-0.5), 0.9],
        ]
    )
    assert_allclose(np.asarray(rbf_K(X, X)), expected_result)
