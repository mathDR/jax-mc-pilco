import pytest
from unittest.mock import MagicMock, patch
import equinox as eqx
import optax as ox
import jax
import jax.numpy as jnp
import jax.random as jr
import gymnasium

from jax_mc_pilco.policy_learning.rollout import fit_controller


# Define some dummy Equinox Modules for testing
class DummyPolicy(eqx.Module):
    weight: jax.Array

    def __init__(self):
        self.weight = jnp.array(1.0)

    def __call__(self, state, timestep):
        return self.weight * state


class DummyGPModel(eqx.Module):
    def get_samples(self, key, states, actions, num_samples):
        # Return some dummy samples with appropriate shape
        # Assuming states are (N, state_dim) and actions are (N, action_dim)
        # We need to return (num_samples, state_dim)
        # For simplicity, let's assume state_dim = 1, action_dim = 1
        return jnp.ones((num_samples, states.shape[-1])) * 0.5


# Define a dummy objective function
def dummy_obj_func(state_action_pair):
    return jnp.sum(state_action_pair**2)


@pytest.fixture
def mock_env():
    env = MagicMock(spec=gymnasium.wrappers.common.TimeLimit)
    env.reset.return_value = (jnp.array([1.0, 2.0]), {})  # Mock initial observation
    return env


@pytest.fixture
def dummy_policy():
    return DummyPolicy()


@pytest.fixture
def dummy_gp_model():
    return DummyGPModel()


@pytest.fixture
def dummy_optim():
    # A simple SGD optimizer for testing
    return ox.sgd(learning_rate=0.01)


@pytest.fixture
def default_fit_controller_args(mock_env, dummy_policy, dummy_gp_model, dummy_optim):
    return {
        "policy": dummy_policy,
        "starting_dropout_probability": 0.0,
        "env": mock_env,
        "num_particles": 10,
        "initial_state": jnp.array([0.0, 0.0]),
        "timesteps": jnp.arange(5),
        "gp_model": dummy_gp_model,
        "obj_func": dummy_obj_func,
        "optim": dummy_optim,
        "key": jr.PRNGKey(0),
        "num_iters": 2,  # Keep num_iters low for faster tests
        "unroll": 1,
    }


class TestFitController:
    def test_fit_controller_returns_policy_and_losses(
        self, default_fit_controller_args
    ):
        policy, losses = fit_controller(**default_fit_controller_args)
        assert isinstance(policy, eqx.Module)
        assert isinstance(losses, jax.Array)
        assert losses.shape == (default_fit_controller_args["num_iters"],)
        assert losses.ndim == 1

    def test_fit_controller_optimizes_policy_parameters(
        self, default_fit_controller_args
    ):
        initial_policy = default_fit_controller_args["policy"]
        initial_weight = initial_policy.weight

        # Run optimization
        optimized_policy, _ = fit_controller(**default_fit_controller_args)

        # Assert that the policy parameters have changed
        # This is a weak check, but it confirms optimization happened
        assert not jnp.array_equal(initial_weight, optimized_policy.weight)

    @patch(
        "your_module.rollout"
    )  # Patch the inner rollout function if it were defined globally
    def test_rollout_called_with_correct_arguments(
        self, mock_rollout, default_fit_controller_args
    ):
        mock_rollout.return_value = jnp.array(10.0)  # Mock a dummy loss

        fit_controller(**default_fit_controller_args)

        # Assert that rollout was called at least once
        mock_rollout.assert_called()
        # You can add more specific assertions about the arguments passed to rollout
        # e.g., mock_rollout.assert_called_with(policy, initial_particles, gp_model, timesteps)
        # This requires more careful mocking of initial_particles if you want to assert their values.

    def test_initial_particles_shape(self, default_fit_controller_args):
        # We need to test the logic that happens before the optimization loop
        # This is harder to isolate without refactoring the function or deeper mocking.
        # For now, we'll rely on the full function execution.
        policy = default_fit_controller_args["policy"]
        env = default_fit_controller_args["env"]
        gp_model = default_fit_controller_args["gp_model"]
        key = default_fit_controller_args["key"]
        num_particles = default_fit_controller_args["num_particles"]
        initial_state = default_fit_controller_args["initial_state"]

        sample, _ = env.reset(
            options={"x_init": initial_state[0], "y_init": initial_state[1]}
        )
        u = policy(sample, 0.0)

        # This part is internal, so we need to either refactor it out
        # or test it by letting the function run and checking side effects
        # For this test, we can directly call get_samples with expected inputs
        # and check its output shape given our DummyGPModel.
        initial_particles = gp_model.get_samples(
            key, jnp.array([sample]), jnp.array([u]), num_particles
        )
        assert initial_particles.shape == (
            num_particles,
            sample.shape[-1],
        )  # Assuming state_dim from sample

    def test_loss_values_decrease_over_iterations(self, default_fit_controller_args):
        # This test is good for verifying the optimization process itself.
        # It might be flaky depending on the complexity of your actual models and data.
        # For simple dummy models, it should generally pass.
        policy, losses = fit_controller(**default_fit_controller_args)

        # Assert that the loss generally decreases (or at least doesn't strictly increase)
        # for a simple case. For more complex scenarios, you might need a tolerance.
        assert losses[-1] < losses[0] or jnp.isclose(
            losses[-1], losses[0], atol=1e-5
        )  # Allow for very small changes

    def test_zero_iterations(self, default_fit_controller_args):
        args = default_fit_controller_args.copy()
        args["num_iters"] = 0
        policy, losses = fit_controller(**args)
        assert isinstance(policy, eqx.Module)
        assert losses.shape == (0,)  # Expect an empty array of losses

    def test_single_iteration(self, default_fit_controller_args):
        args = default_fit_controller_args.copy()
        args["num_iters"] = 1
        policy, losses = fit_controller(**args)
        assert isinstance(policy, eqx.Module)
        assert losses.shape == (1,)
        assert losses.ndim == 1

    def test_timesteps_impact(self, default_fit_controller_args):
        # Test with different number of timesteps
        args_short = default_fit_controller_args.copy()
        args_short["timesteps"] = jnp.arange(2)  # Shorter rollout
        _, losses_short = fit_controller(**args_short)

        args_long = default_fit_controller_args.copy()
        args_long["timesteps"] = jnp.arange(10)  # Longer rollout
        _, losses_long = fit_controller(**args_long)

        # Verify that losses are computed for the respective number of iterations
        assert losses_short.shape == (args_short["num_iters"],)
        assert losses_long.shape == (args_long["num_iters"],)
        # You might also assert that the initial loss for longer timesteps is generally higher
        # (assuming a positive cost function), but this can be fragile.

    def test_obj_func_is_used(self, default_fit_controller_args):
        mock_obj_func = MagicMock(return_value=jnp.array(1.0))
        args = default_fit_controller_args.copy()
        args["obj_func"] = mock_obj_func

        fit_controller(**args)
        mock_obj_func.assert_called()  # Assert that the objective function was called

    def test_optim_update_is_called(self, default_fit_controller_args):
        # Mock the optimizer to ensure its update method is called
        mock_optim = MagicMock(spec=ox.GradientTransformation)
        mock_optim.init.return_value = "mock_opt_state"
        mock_optim.update.return_value = (
            {"weight": jnp.array(-0.1)},
            "new_mock_opt_state",
        )  # Dummy update

        args = default_fit_controller_args.copy()
        args["optim"] = mock_optim

        fit_controller(**args)
        mock_optim.init.assert_called_once()
        # The update method should be called num_iters times
        assert mock_optim.update.call_count == args["num_iters"]
