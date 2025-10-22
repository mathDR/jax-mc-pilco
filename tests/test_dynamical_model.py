"""Tests for the Dynamical Model classes."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx  # type: ignore
from jaxtyping import ArrayLike, install_import_hook
from typing import List

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax

# Import the class to be tested
from jax_mc_pilco.model_learning.dynamical_models import DynamicalModel, IMGPR

# --- Mocking external dependencies ---
# We use mocks to avoid testing the complex logic of gpjax and its components,
# and instead focus on the logic of DynamicalModel itself.


class MockKernel(eqx.Module):
    def __init__(self):
        pass


class MockMeanFunction(eqx.Module):
    def __init__(self):
        pass


class MockLikelihood:
    def __init__(self):
        pass


class MockGaussianLikelihood(MockLikelihood):
    def __init__(self):
        super().__init__()


class MockDataset(eqx.Module):
    X: jax.Array
    y: jax.Array

    def __init__(self, X, y):
        self.X = X
        self.y = y


class MockAbstractPosterior(eqx.Module):
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, test_inputs, train_data):
        # A simple mock latent distribution
        num_test_points = test_inputs.shape[0]
        num_outputs = train_data.y.shape[1]
        loc = jnp.zeros((num_test_points, num_outputs))
        scale = jnp.eye(num_test_points)

        # A mock distribution with a loc and scale for sampling
        class MockLatentDist(eqx.Module):
            batch_shape = (num_test_points,)
            event_shape = (num_outputs,)
            loc = loc
            scale = scale

        return MockLatentDist()


# --- Pytest fixtures for test data ---


@pytest.fixture
def states() -> jax.Array:
    """Generates a mock states array."""
    return jnp.arange(120, dtype=jnp.float64).reshape(20, 6)


@pytest.fixture
def actions() -> jax.Array:
    """Generates a mock actions array."""
    return jnp.arange(60, dtype=jnp.float64).reshape(20, 3)


@pytest.fixture
def kernel_func() -> jax.Array:
    """Returns a mock kernel."""
    return MockKernel()


@pytest.fixture
def mean_func() -> MockMeanFunction:
    """Returns a mock mean function."""
    return MockMeanFunction()


@pytest.fixture
def likelihood_func() -> MockGaussianLikelihood:
    """Returns a mock likelihood."""
    return MockGaussianLikelihood()


@pytest.fixture
def model(states: ArrayLike, actions: ArrayLike) -> List[eqx.Module]:
    """Fixture for the models attribute, simulating the list of posteriors."""
    return DynamicalModel(states, actions, MockKernel())


# --- Unit test suite ---
def test_constructor_with_single_kernel_mean_likelihood(
    states: ArrayLike, actions: ArrayLike, kernel_func: MockKernel
) -> None:
    """Tests the constructor with a single kernel, mean, and likelihood."""
    model = DynamicalModel(
        states=states,
        actions=actions,
        kernel_funcs=kernel_func,
        mean_funcs=MockMeanFunction(),
        likelihoods=MockGaussianLikelihood(),
    )

    assert isinstance(model, DynamicalModel)
    assert len(model.kernels) == model.num_outputs
    assert isinstance(model.kernels[0], MockKernel)
    assert len(model.mean_functions) == model.num_outputs
    assert isinstance(model.mean_functions[0], MockMeanFunction)
    assert len(model.likelihoods) == model.num_outputs
    assert isinstance(model.likelihoods[0], MockGaussianLikelihood)


def test_constructor_with_lists(states: ArrayLike, actions: ArrayLike) -> None:
    """Tests the constructor with lists of kernels, means, and likelihoods."""
    num_outputs = states.shape[1]
    kernel_list = [MockKernel() for _ in range(num_outputs)]
    mean_list = [MockMeanFunction() for _ in range(num_outputs)]
    likelihood_list = [MockGaussianLikelihood for _ in range(num_outputs)]

    model = DynamicalModel(
        states=states,
        actions=actions,
        kernel_funcs=kernel_list,
        mean_funcs=mean_list,
        likelihoods=likelihood_list,
    )

    assert len(model.kernels) == num_outputs
    assert len(model.mean_functions) == num_outputs
    assert len(model.likelihoods) == num_outputs
    assert all(isinstance(k, MockKernel) for k in model.kernels)


def test_constructor_defaults(
    states: ArrayLike, actions: ArrayLike, kernel_func: MockKernel
) -> None:
    """Tests the constructor's default behavior for mean and likelihood."""
    model = DynamicalModel(states=states, actions=actions, kernel_funcs=kernel_func)

    assert len(model.mean_functions) == model.num_outputs
    assert isinstance(model.mean_functions[0], gpjax.mean_functions.Zero)
    assert len(model.likelihoods) == model.num_outputs
    assert model.likelihoods[0] is gpjax.likelihoods.Gaussian


def test_data_to_gp_input_output(states: ArrayLike, actions: ArrayLike) -> None:
    """Tests the data transformation method for GP input/output."""
    pm = 2
    cm = 1
    model = DynamicalModel(
        states, actions, MockKernel(), position_memory=pm, control_memory=cm
    )
    gp_input, gp_output = model.data_to_gp_input_output(states, actions)

    # Expected shapes based on logic
    num_data_points = states.shape[0] - max(pm, cm)
    input_dim = states.shape[1] * (pm + 1) + actions.shape[1] * cm
    output_dim = states.shape[1]

    assert gp_input.shape == (num_data_points, input_dim)
    assert gp_output.shape == (num_data_points, output_dim)

    # A simple spot check of the data content
    # The last state in the reversed array should be in the first gp_input element.
    # The first difference should be in the first gp_output.
    expected_output_0 = states[-1, :] - states[-2, :]
    assert jnp.all(jnp.isclose(gp_output[0], expected_output_0))


def test_data_to_policy_input(states: ArrayLike, actions: ArrayLike) -> None:
    """Tests the data transformation for policy inputs."""
    pm = 2
    model = DynamicalModel(states, actions, MockKernel(), position_memory=pm)
    policy_input = model.data_to_policy_input(states)

    # Expected shapes
    num_data_points = states.shape[0] - pm
    input_dim = states.shape[1] * (pm + 1)

    assert policy_input.shape == (num_data_points, input_dim)


def test_create_models_not_implemented(
    states: ArrayLike, actions: ArrayLike, kernel_func: MockKernel
):
    """Tests that create_models raises a NotImplementedError."""
    model = DynamicalModel(states, actions, kernel_func)
    with pytest.raises(NotImplementedError):
        model.create_models()
