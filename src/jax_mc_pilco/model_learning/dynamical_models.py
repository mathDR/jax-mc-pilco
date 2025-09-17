""" The main model class. """

__all__ = ["DynamicalModel", "IMGPR", "IMSGPR", "optimize_imgpr","optimize_imsgpr"]

import time
from typing import Callable, Dict, List, Tuple, Union
from jax import Array, config, jit, value_and_grad, vmap
import equinox as eqx
import jax.numpy as jnp
from functools import partial

from jax_mc_pilco.model_learning.gp.gaussian_process import (
    GaussianProcess,
    gp_fit,
    SparseVariationalGaussianProcess,
    svgp_fit,
)
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
    params: List[Dict[str, Union[Dict[str, ArrayLike], ArrayLike]]]
    models: List[Union[GaussianProcess,SparseVariationalGaussianProcess]]
    name: str | None

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_funcs: Kernel | List[Kernel],
        params: Dict[str, Union[Dict[str, ArrayLike], ArrayLike]],
        *,
        mean_funcs: List[Mean] | Mean | None = None,
        models: List[Union[GaussianProcess, SparseVariationalGaussianProcess]] | None = None,
        name: str | None = None,
    ) -> None:
        self.training_data, self.training_outputs = self.data_to_gp_input_output(
            states, actions
        )

        self.num_outputs: Int = self.training_outputs.shape[1]
        self.input_dimension: Int = self.training_data.shape[1]
        self.num_datapoints: Int = self.training_data.shape[0]

        if mean_funcs is None:
            # Give each output a zero mean
            self.mean = [ZeroMean] * self.num_outputs
        elif isinstance(mean_funcs,list):
            self.mean = mean_funcs
        else:
            # Give each output the same mean
            self.mean = [mean_funcs] * self.num_outputs

        if isinstance(kernel_funcs,list):
            self.kernel = kernel_funcs
        else:
            self.kernel = [kernel_funcs] * self.num_outputs

        self.params = params
        if models:
            self.models = models
        else:
            self.models = self.create_models(self.params)

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
    ) -> List[Union[GaussianProcess,SparseVariationalGaussianProcess]]:
        """Create the models for each output dimension.
        Following https://docs.kidger.site/equinox/tricks/#ensembling
        """
        raise NotImplementedError()

    @eqx.filter_jit
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
            mu, cov = self.models[i].predict(test_inputs)
            predictive_means.append(mu+test_inputs[:,i])
            predictive_vars.append(cov)
        predictive_moments = jnp.dstack(
            (
                jnp.array(predictive_means),
                jnp.array(predictive_vars),
            ),
        )
        return predictive_moments

    @eqx.filter_jit
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
        kernel_funcs: List[Kernel] | Kernel,
        params: Dict[str, Union[Dict[str, ArrayLike], ArrayLike]],
        *,
        mean_funcs: List[Mean] | Mean | None = None,
        models: List[Union[GaussianProcess, SparseVariationalGaussianProcess]] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(states, actions, kernel_funcs, params, mean_funcs=mean_funcs, models=models, name=name)

    def build_gp(self, output_gp_index: Int, param: Dict[str, Union[Dict[str,Float],Float]], optimized: Bool) -> GaussianProcess:
        """Constructs a GP from the parameter Dict."""
        return GaussianProcess(
            self.kernel[output_gp_index],
            self.training_data,
            self.training_outputs[:,output_gp_index],
            param,
            mean=self.mean[output_gp_index],
            optimized=optimized,
        )

    def create_models(
        self,
        params: List[Dict[str, Union[Dict[str, ArrayLike], ArrayLike]]],
    ) -> List[GaussianProcess]:
        """Create GP models.  We should be able to use equinox ensembling and vmap to build all of these, but
           that is left for a TODO.
        """

        #@partial(vmap,in_axes=(1,0))
        #def make_model(y: ArrayLike, param: Dict[str, Union[Dict[str,Float],Float]])->GaussianProcess:
        #    return self.build_gp(y,param,optimized=False)

        #return make_model(self.training_outputs, params)

        return [self.build_gp(i, params[i], optimized=False) for i in range(self.num_outputs)]


class IMSGPR(DynamicalModel):
    """The forward model of the system dynamics.

    Independent Multiple Sparse Gaussian Process regression - has an independent Sparse GP for every output dimension

    Args:
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
    """

    num_inducing_points: Int

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_funcs: Kernel,
        num_inducing_points: Int,
        params: Dict[str, Union[Dict[str, ArrayLike], ArrayLike]],
        *,
        mean_funcs: List[Mean] | Mean | None = None,
        models: List[Union[GaussianProcess, SparseVariationalGaussianProcess]] | None = None,
        name: str | None = None,
    ) -> None:
        self.num_inducing_points = num_inducing_points
        super().__init__(states, actions, kernel_funcs, params, mean_funcs=mean_funcs, models=models, name=name)

    def build_gp(self, output_gp_index: Int, param: Dict[str, Union[Dict[str,Float],Float]], optimized: Bool) -> SparseVariationalGaussianProcess:
        """Constructs a GP from the parameter Dict."""
        return SparseVariationalGaussianProcess(
            self.kernel[output_gp_index],
            self.training_data,
            self.training_outputs[:,output_gp_index],
            self.num_inducing_points,
            param,
            mean=self.mean[output_gp_index],
            optimized=optimized,
        )

    def create_models(
        self,
        params: List[Dict[str, Union[Dict[str, ArrayLike], ArrayLike]]],
    ) -> List[SparseVariationalGaussianProcess]:
        """Create Sparse GP models using params list"""

        #@partial(vmap,in_axes=(1,0))
        #def make_model(y: ArrayLike, param: Dict[str, Union[Dict[str,Float],Float]],)->SparseVariationalGaussianProcess:
        #    return self.build_gp(y, param, optimized=False)
        #return make_model(self.training_outputs, params)

        return [self.build_gp(i, params[i], optimized=False) for i in range(self.num_outputs)]

def optimize_imgpr(
    dynamical_model: IMGPR,
    states: ArrayLike,
    actions: ArrayLike,
    *,
    max_iters: int = 500,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
)->IMGPR:
    """Optimize the imgpr dynamical model and return a new instance."""
    models = []
    params = []
    for i in range(dynamical_model.num_outputs):
        model = gp_fit(dynamical_model.models[i],max_iters=max_iters,max_linesearch_steps=max_linesearch_steps,gtol=gtol)
        models.append(model)
        params.append(model.params)

    return IMGPR(
        states=states,
        actions=actions,
        kernel_funcs=dynamical_model.kernel,
        params=params,
        mean_funcs=dynamical_model.mean,
        models=models,
        name=dynamical_model.name,
    )

def optimize_imsgpr(
    dynamical_model: IMSGPR,
    states: ArrayLike,
    actions: ArrayLike,
    *,
    max_iters: int = 500,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
)->IMSGPR:
    """Optimize the imgpr dynamical model and return a new instance."""
    models = []
    params = []

    for i in range(dynamical_model.num_outputs):
        #print(f"Output {i}:")
        #breakpoint()
        #start_time = time.perf_counter()
        model = svgp_fit(dynamical_model.models[i],max_iters=max_iters,max_linesearch_steps=max_linesearch_steps,gtol=gtol)
        #end_time = time.perf_counter()
        #print(end_time-start_time)
        #breakpoint()
        models.append(model)
        params.append(model.params)
    return IMSGPR(
        states=states,
        actions=actions,
        kernel_funcs=dynamical_model.kernel,
        num_inducing_points=dynamical_model.num_inducing_points,
        params=params,
        mean_funcs=dynamical_model.mean,
        models=models,
        name=dynamical_model.name,
    )
