""" The main model class. """

__all__ = ["DynamicalModel", "IMGPR"]

from typing import Callable, Dict, List, Tuple, Union
from jax import Array, config, jit, value_and_grad, vmap
from jax.tree_util import Partial, tree_map
import equinox as eqx
import jax.numpy as jnp

from jax_mc_pilco.model_learning.gp.gaussian_process import gp_fit, GaussianProcess
from jax_mc_pilco.model_learning.gp import Kernel, Mean, ZeroMean
import jax.random as jr
from jaxtyping import ArrayLike, Bool, Float, Int, PyTree

config.update("jax_enable_x64", True)


class DynamicalModel(eqx.Module):
    """The base class for forward model of the system dynamics.

    Args:
        mean (Mean): The mean function, if left blank will default to zero mean
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
    """

    # pylint: disable=too-many-instance-attributes
    mean: Mean | List[Mean] | None
    kernel: Kernel | List[Kernel]
    training_data: ArrayLike
    training_outputs: ArrayLike
    num_outputs: Int
    input_dimension: Int
    num_datapoints: Int
    models: List[ArrayLike]
    name: str | None

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_func: Kernel,
        params: List[Dict[str, Union[Dict[str, Float], Float]]],
        *,
        mean_func: Mean | None = None,
        name: str | None = None,
    ) -> None:
        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self.num_outputs: Int = self.training_outputs.shape[1]
        self.input_dimension: Int = self.training_data.shape[1]
        self.num_datapoints: Int = self.training_data.shape[0]

        if mean_func is None:
            self.mean = ZeroMean
        else:
            self.mean = mean_func

        self.kernel = kernel_func

        self.create_models(params)

        self.name = name

    def data_to_gp_output(self, states: ArrayLike) -> Array:
        """Transforms data into PILCO data format."""
        val = jnp.diff(states, n=1, axis=0)
        if val.ndim == 1:
            val = jnp.atleast_2d(val).T
        return val

    def data_to_gp_input(self, states: ArrayLike, actions: ArrayLike) -> Array:
        """Transforms data into PILCO data format."""
        val = jnp.hstack((states, actions))
        if val.ndim == 1:
            val = jnp.atleast_2d(val).T
        return val

    def data_to_gp_input_output(
        self, states: ArrayLike, actions: ArrayLike
    ) -> Tuple[Array, Array]:
        """Transforms data into PILCO data format."""
        return self.data_to_gp_input(states, actions)[:-1, :], self.data_to_gp_output(
            states
        )

    def create_models(
        self,
        params: List[Dict[str, float]],
    ) -> None:
        """Create the models for each output dimension."""
        raise NotImplementedError()

    def optimize(self, maxiter: int = 1000, key: ArrayLike | None = None):
        """Minimize negative marginal likelihood for the model over the hyperparameters."""
        raise NotImplementedError()

    def predict_all_outputs(self, test_inputs: ArrayLike) -> Tuple[Array, Array]:
        """TODO."""
        raise NotImplementedError()

    def get_samples(
        self, key: ArrayLike, states: ArrayLike, actions: ArrayLike, num_samples: int
    ) -> Array:
        """TODO."""
        raise NotImplementedError()


class IMGPR(DynamicalModel):
    """The forward model of the system dynamics.

    Independent Multiple Gaussian Process regression - has an independent GP for every output dimension

    Args:
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
    """

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_func: Kernel,
        params: List[Dict[str, Union[Dict[str, float], float]]],
        *,
        mean_func: Callable | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(states, actions, kernel_func, params, mean_func=mean_func, name=name)

    def build_gp(self, y: ArrayLike, params: ArrayLike, optimized: Bool) -> GaussianProcess:
        """Constructs a GP from the parameter list.  Should figure out how to parameterize the kernel."""
        return GaussianProcess(
            self.kernel,
            self.training_data,
            y,
            params,
            mean=self.mean,
            optimized=optimized,
        )

    def create_models(
        self,
        params: List[Dict[str, Float]],
    ) -> None:
        """Create GP models using params list"""

        self.models = []

        for i in range(self.num_outputs):
            self.models.append(params[i])

    def optimize(self, maxiter: int = 1000, key: ArrayLike | None = None):
        """Optimize the hyperparameters of the models using MAP nlml."""

        if key is None:
            key = jr.key(123)

        for i in range(self.num_outputs):  # Iterate with index
            self.models[i] = gp_fit(self.build_gp(self.training_outputs[:,i], self.models[i], optimized=False)).params  # Update the model with the optimized posterior.

    @jax.jit
    def predict_all_outputs(self, test_inputs: ArrayLike) -> Tuple[Array, Array]:
        """
        Return the gp ouputs (mean and variance) for each output dimension for each test input

        Args:
        test_inputs (List[JAXArray]): A list containing the test inputs.

        returns a tuple containing the means and covariances of the test inputs. This assumes that the gps have been fit!

        Because the GP models the differences in the states, we must add back the state to get
        the state mean (the variance is the same).

        """
        predictive_means = []
        predictive_vars = []
        for i in range(self.num_outputs):
            gp = self.build_gp(self.training_outputs[:,i],self.models[i],optimized=True)
            mu, cov = gp.predict(test_inputs)
            predictive_means.append(mu+test_inputs[:,i])
            predictive_vars.append(cov)
        predictive_moments = jnp.dstack(
            (
                jnp.array(predictive_means),
                jnp.array(predictive_vars),
            ),
        )
        return predictive_moments

    @jax.jit
    def get_samples(
        self, key: ArrayLike, states: ArrayLike, actions: ArrayLike, num_samples: int
    ) -> Array:
        # Function to sample from a single mean and covariance
        def sample_mvnormal(key, mean, cov, num_samples):
            return jr.multivariate_normal(key, mean, cov, (num_samples,))

        # Vectorize the sampling function
        vectorized_sample = vmap(sample_mvnormal, in_axes=(None, 0, 0, None))
        test_inputs = self.data_to_gp_input(states, actions)
        predictive_moments = self.predict_all_outputs(test_inputs)
        return jnp.squeeze(
            vectorized_sample(
                key,
                predictive_moments[:, :, 0],
                vmap(jnp.diag)(predictive_moments[:, :, 1]),
                num_samples,
            ).T
        )
