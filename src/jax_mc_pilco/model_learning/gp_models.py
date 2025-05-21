""" The main model class. """

__all__ = ["DynamicalModel"]

from typing import Callable, Dict, List, Optional, Tuple
from jax import Array, config, jit, value_and_grad, vmap
from jax.tree_util import Partial, tree_map
from tinygp import kernels, GaussianProcess, transforms
import tinygp
import equinox as eqx
import jax.numpy as jnp

import jax.random as jr
from jaxtyping import ArrayLike, PyTree

from jax_mc_pilco.model_learning.kernels import SpectralMixture

import optax as ox

config.update("jax_enable_x64", True)


class DynamicalModel(eqx.Module):
    """The forward model of the system dynamics.

    Currently is a Multiple Gaussian Process regression with an independent GP for
    every output dimension but
    TODO: use a multioutput kernel.

    Args:
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
    """

    # pylint: disable=too-many-instance-attributes
    mean_func: Optional[Callable] = None
    training_data: ArrayLike
    training_outputs: ArrayLike
    num_outputs: int
    input_dimension: int
    num_datapoints: int
    optimizers: List[ox._src.base.GradientTransformationExtraArgs]
    models: List[ArrayLike]
    name: Optional[str]

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        params: Optional[List[Dict[str, float]]] = None,
        mean_func: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> None:
        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self.num_outputs: int = self.training_outputs.shape[1]
        self.input_dimension: int = self.training_data.shape[1]
        self.num_datapoints: int = self.training_data.shape[0]

        if mean_func is None:
            self.mean_func = lambda param, x: 0.0
        else:
            self.mean_func = mean_func

        self.create_models(params)
        self.optimizers: List[ox._src.base.GradientTransformationExtraArgs] = []

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
        params: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Create the models for each output dimension."""
        raise NotImplementedError()

    def optimize(self, maxiter: int = 1000, key: Optional[ArrayLike] = None):
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
        params: Optional[List[Dict[str, float]]] = None,
        mean_func: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(states, actions, params, mean_func, name)

    def build_gp(self, param: ArrayLike) -> tinygp.gp.GaussianProcess:
        """Constructs a GP from the parameter list.  Should figure out how to parameterize the kernel."""
        # kernel = jnp.exp(param["log_amp"]) * transforms.Linear(
        #     jnp.exp(param["log_scale"]), kernels.ExpSquared()
        # )
        kernel = SpectralMixture(
            jnp.exp(param["log_weight"]),
            jnp.exp(param["log_scale"]),
            jnp.exp(param["log_freq"]),
        )
        return GaussianProcess(
            kernel,
            self.training_data,
            diag=jnp.square(jnp.exp(param["log_diag"])),
            mean=Partial(self.mean_func, param),
        )

    def create_models(
        self,
        params: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Create GP models using params list"""

        self.models = []

        if params is None:
            params = [
                {
                    "log_weight": jnp.log(jnp.array([1.0, 1.0])),
                    "log_scale": jnp.log(jnp.array([10.0, 20.0])),
                    "log_freq": jnp.log(jnp.array([1.0, 0.5])),
                    "log_diag": jnp.log(0.1),
                }
            ] * self.num_outputs
            # params = [
            #     {
            #         "log_amp": -0.1,
            #         "log_scale": 0.0,
            #         "log_diag": -2.5,
            #     }
            # ] * self.num_outputs

        for i in range(self.num_outputs):
            self.models.append(params[i])

    def optimize(self, maxiter: int = 1000, key: Optional[ArrayLike] = None):
        """Optimize the hyperparameters of the models using MAP nlml."""

        if key is None:
            key = jr.key(123)

        if not self.optimizers:  # More Pythonic way to check if list is empty
            for i in range(self.num_outputs):
                self.optimizers.append(
                    ox.adam(
                        learning_rate=ox.linear_schedule(
                            init_value=1e-1, end_value=1e-6, transition_steps=100
                        )
                    )
                )

        patience = 3
        for i in range(self.num_outputs):  # Iterate with index

            @jit
            def loss(params):
                gp = self.build_gp(params)
                return -gp.log_probability(self.training_outputs[:, i])

            @jit
            def make_step(
                params: ArrayLike,
                opt_state: PyTree,
            ):
                loss_value, grads = value_and_grad(loss)(params)
                updates, opt_state = self.optimizers[i].update(grads, opt_state, params)
                params = ox.apply_updates(params, updates)
                return params, opt_state, loss_value

            patience_count = 0
            params = tree_map(jnp.asarray, self.models[i])
            opt_state = self.optimizers[i].init(params)
            best_val = float("inf")
            for _ in range(maxiter):
                params, opt_state, train_loss = make_step(params, opt_state)
                if train_loss < best_val:
                    best_val = train_loss
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count > patience:
                    break
            self.models[i] = params  # Update the model with the optimized posterior.

    def predict_all_outputs(self, test_inputs: ArrayLike) -> Tuple[Array, Array]:
        """
        Return the gp ouputs (mean and variance) for each output dimension for each test input

        Args:
        test_inputs (List[JAXArray]): A list containing the test inputs.

        returns a tuple containing the means and covariances of the test inputs.

        Because the GP models the differences in the states, we must add back the state to get
        the state mean (the variance is the same).

        """
        predictive_means = []
        predictive_vars = []
        for i in range(self.num_outputs):
            gp = self.build_gp(self.models[i])
            cond_gp = gp.condition(self.training_outputs[:, i], test_inputs).gp
            predictive_means.append(cond_gp.loc)
            predictive_vars.append(cond_gp.variance)
        predictive_moments = jnp.stack(
            (
                jnp.array(predictive_means).T + test_inputs[:, : self.num_outputs],
                jnp.array(predictive_vars).T,
            ),
            axis=2,
        )
        return predictive_moments

    @eqx.filter_vmap
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
            )
        )
