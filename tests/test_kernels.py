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
def ard_params() -> Dict:
    return {
        "kernel": {
            "coefficient": jnp.array(0.9),
            "log_scale": jnp.array(
                [
                    jnp.log(jnp.exp(3.0) - 1.0),
                    jnp.log(jnp.exp(2.0) - 1.0),
                ]
            ),
        },
        "mean": {},
        "likelihood": {"log_diag": jnp.array(-0.5)},
    }


@pytest.fixture
def X() -> jax.Array:
    return jnp.array([1.0, 2.0, 3.0])[:, jnp.newaxis]


@pytest.fixture
def X_D() -> jax.Array:
    return jnp.array([[1.0, -1.0], [2.0, 1.0], [3.0, 0.5]])[:, jnp.newaxis]


@pytest.fixture
def rbf_K(params: Dict) -> Kernel:
    return ExpSquared(**params["kernel"])


@pytest.fixture
def ard_rbf_K(ard_params: Dict) -> Kernel:
    return ExpSquared(**ard_params["kernel"])


@pytest.fixture
def ard_exp_K(ard_params: Dict) -> Kernel:
    return Exp(**ard_params["kernel"])


def test_squared_distance_function(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    assert_allclose(np.asarray(rbf_K.distance.squared_distance(X, X)), expected_result)


def test_squared_distance_function_D(ard_rbf_K: Kernel, X_D: ArrayLike):
    expected_result = np.array(
        [
            [
                0.0,
                jnp.sum(jnp.square(X_D[0, :] - X_D[1, :])),
                jnp.sum(jnp.square(X_D[0, :] - X_D[2, :])),
            ],
            [
                jnp.sum(jnp.square(X_D[1, :] - X_D[0, :])),
                0.0,
                jnp.sum(jnp.square(X_D[1, :] - X_D[2, :])),
            ],
            [
                jnp.sum(jnp.square(X_D[2, :] - X_D[0, :])),
                jnp.sum(jnp.square(X_D[2, :] - X_D[1, :])),
                0.0,
            ],
        ]
    )
    assert_allclose(
        np.asarray(ard_rbf_K.distance.squared_distance(X_D, X_D)), expected_result
    )


def test_distance_function(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    assert_allclose(np.asarray(rbf_K.distance.distance(X, X)), expected_result)


def test_distance_function_D(ard_exp_K: Kernel, X_D: ArrayLike):
    expected_result = np.array(
        [
            [
                0.0,
                jnp.sum(jnp.abs(X_D[0, :] - X_D[1, :])),
                jnp.sum(jnp.abs(X_D[0, :] - X_D[2, :])),
            ],
            [
                jnp.sum(jnp.abs(X_D[1, :] - X_D[0, :])),
                0.0,
                jnp.sum(jnp.abs(X_D[1, :] - X_D[2, :])),
            ],
            [
                jnp.sum(jnp.abs(X_D[2, :] - X_D[0, :])),
                jnp.sum(jnp.abs(X_D[2, :] - X_D[1, :])),
                0.0,
            ],
        ]
    )
    assert_allclose(np.asarray(ard_exp_K.distance.distance(X_D, X_D)), expected_result)


def test_rbf_kernel(rbf_K: Kernel, X: ArrayLike):
    expected_result = np.array(
        [
            [0.9, 0.9 * np.exp(-0.5), 0.9 * np.exp(-2.0)],
            [0.9 * np.exp(-0.5), 0.9, 0.9 * np.exp(-0.5)],
            [0.9 * np.exp(-2.0), 0.9 * np.exp(-0.5), 0.9],
        ]
    )
    assert_allclose(np.asarray(rbf_K(X, X)), expected_result)


def test_ard_rbf_kernel(ard_rbf_K: Kernel, X_D: ArrayLike):
    """The form of the params rescales X to be all ones."""
    scale = jnp.array([3.0, 2.0])
    expected_result = np.array(
        [
            [
                0.0,
                jnp.sum(jnp.square(X_D[0, :] / scale - X_D[1, :] / scale)),
                jnp.sum(jnp.square(X_D[0, :] / scale - X_D[2, :] / scale)),
            ],
            [
                jnp.sum(jnp.square(X_D[1, :] / scale - X_D[0, :] / scale)),
                0.0,
                jnp.sum(jnp.square(X_D[1, :] / scale - X_D[2, :] / scale)),
            ],
            [
                jnp.sum(jnp.square(X_D[2, :] / scale - X_D[0, :] / scale)),
                jnp.sum(jnp.square(X_D[2, :] / scale - X_D[1, :] / scale)),
                0.0,
            ],
        ]
    )
    assert_allclose(
        np.asarray(ard_rbf_K(X_D, X_D)), 0.9 * jnp.exp(-0.5 * expected_result)
    )
