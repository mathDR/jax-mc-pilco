__all__ = ["DynamicalModel"]

from jax import Array, config, grad, jit, vmap

config.update("jax_enable_x64", True)

import gpjax as gpx
from flax import nnx

import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.typing import ArrayLike
from typing import List, Optional, Tuple
import optax as ox


class DynamicalModel(nnx.Module):
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

    states: ArrayLike
    actions: ArrayLike
    mean_func: Optional[gpx.mean_functions] = None
    name: Optional[str]
    num_outputs: int
    input_dimension: int
    num_datapoints: int
    models: List[gpx.gps.ConjugatePosterior]
    optimizers: List[ox._src.base.GradientTransformationExtraArgs]

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        mean_func: Optional[gpx.mean_functions] = None,
        name: Optional[str] = None,
    ) -> None:
        X, y = self.data_to_gp_input_output(states, actions)

        self.data = gpx.Dataset(X=X, y=y)
        self.mean_func = mean_func

        self.num_outputs: int = self.data.y.shape[1]
        self.input_dimension: int = self.data.X.shape[1]
        self.num_datapoints: int = self.data.X.shape[0]

        if mean_func:
            self.mean_func = mean_func
        else:
            self.mean_func = [gpx.mean_functions.Zero()] * self.num_outputs

        self.create_models()
        self.optimizers: List = []

        self.name = name

    def data_to_gp_output(self, states: ArrayLike) -> Array:
        return jnp.diff(states, n=1, axis=0)

    def data_to_gp_input(self, states: ArrayLike, actions: ArrayLike) -> Array:
        return jnp.hstack((states, actions))

    def data_to_gp_input_output(
        self, states: ArrayLike, actions: ArrayLike
    ) -> Tuple[Array, Array]:
        return self.data_to_gp_input(states, actions)[:-1, :], self.data_to_gp_output(
            states
        )

    def create_models(self) -> None:
        raise NotImplementedError()

    def optimize(self, maxiter: int = 1000, key: Optional[ArrayLike] = None):
        raise NotImplementedError()

    def predict_all_outputs(self, test_inputs: ArrayLike) -> Tuple[Array, Array]:
        raise NotImplementedError()

    def get_samples(
        self, key: ArrayLike, states: ArrayLike, actions: ArrayLike, num_samples: int
    ) -> Array:
        raise NotImplementedError()


class MGPR(DynamicalModel):
    """The forward model of the system dynamics.

    Multiple Gaussian Process regression with an independent GP for every output dimension

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
        mean_func: Optional[gpx.mean_functions] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(states, actions, mean_func, name)

    def create_models(self) -> None:
        self.models = []
        # self.lower_cholesky_K = []
        # self.K_inverse_y = []

        for i in range(self.num_outputs):
            kern = gpx.kernels.RBF(
                variance=1.0,
                lengthscale=0.1
                * jnp.ones((self.input_dimension,)),  # makes an ARD kernel by default
            )
            meanf = self.mean_func[i]
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kern)

            lik = gpx.likelihoods.Gaussian(num_datapoints=self.num_datapoints)

            self.models.append(prior * lik)

    def optimize(self, maxiter: int = 1000, key: Optional[ArrayLike] = None):
        if key is None:
            key = jr.key(123)

        if not self.optimizers:  # More Pythonic way to check if list is empty
            for model in self.models:
                self.optimizers.append(ox.adam(1e-1))

        for i, model in enumerate(self.models):  # Iterate with index
            opt_posterior, history = gpx.fit(
                model=model,  # Use the current model
                objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
                train_data=gpx.Dataset(
                    self.data.X, self.data.y[:, i].reshape(-1, 1)
                ),  # Use self.data
                optim=self.optimizers[i],  # Use the correct optimizer
                num_iters=maxiter,
                safe=True,
                key=key,
            )
            self.models[
                i
            ] = opt_posterior  # Update the model with the optimized posterior.
            # TODO:
            # Cache K^-1 y and the lower cholesky of K for use in predict
            # self.lower_cholesky_K[i] =
            # self.K_inverse_y[i] =

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
        predictive_stds = []
        for i in range(self.num_outputs):
            latent_dist = self.models[i].predict(
                test_inputs,
                train_data=gpx.Dataset(
                    X=self.data.X, y=self.data.y[:, i].reshape(-1, 1)
                ),
            )
            predictive_dist = self.models[i].likelihood(latent_dist)

            predictive_means.append(predictive_dist.mean())
            predictive_stds.append(predictive_dist.stddev())
        predictive_moments = jnp.stack(
            (
                jnp.array(predictive_means).T + test_inputs[:, : self.num_outputs],
                jnp.array(predictive_stds).T,
            ),
            axis=2,
        )
        return predictive_moments

    def get_samples(
        self, key: ArrayLike, states: ArrayLike, actions: ArrayLike, num_samples: int
    ) -> Array:
        # Function to sample from a single mean and covariance
        def sample_mvnormal(key, mean, cov, num_samples):
            return jr.multivariate_normal(key, mean, cov, (num_samples,))

        # Vectorize the sampling function
        vectorized_sample = vmap(sample_mvnormal, in_axes=(None, 0, 0, None))
        X = self.data_to_gp_input(states, actions)
        predictive_moments = self.predict_all_outputs(X)
        return jnp.squeeze(
            vectorized_sample(
                key,
                predictive_moments[:, :, 0],
                vmap(jnp.diag)(predictive_moments[:, :, 1]),
                num_samples,
            )
        )
