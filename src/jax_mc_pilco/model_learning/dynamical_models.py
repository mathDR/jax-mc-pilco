""" The main model class. """

__all__ = ["DynamicalModel", "IMGPR", "IMSVGPR"]

import jax.random as jr
from typing import List, Optional, Tuple, Union
from jaxtyping import ArrayLike, install_import_hook, Int
from jax import Array, config, vmap

import equinox as eqx  # type: ignore
import jax.numpy as jnp

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax
from gpjax import lower_cholesky
from gpjax.kernels import AbstractKernel
from gpjax.likelihoods import AbstractLikelihood, Gaussian
from gpjax.mean_functions import AbstractMeanFunction, Zero
from sklearn.cluster import KMeans

config.update("jax_enable_x64", True)


class DynamicalModel(eqx.Module):
    """The base class for forward model of the system dynamics.

    Args:
        states (ArrayLike): The input states $x_t$.
        actions (ArrayLike): The input controls $u_t.$
        kernel_funcs (AbstractKernel): The kernel function(s) for each GP. If
          there are multiple outputs but a single kernel is passed, each GP
          will get this kernel.
        mean_funcs (AbstractMeanFunction): The mean function(s) for each GP.
          If there are multiple outputs, but a single mean is passed, each
          GP will get this mean. If left blank, will default to zero mean
        likelihood (AbstractLikelihood): The likelihood of the posterior. If
          there are multiple outputs and a single likelihood is passed, each
          GP will get that likelihood. If left blank, will default to a
          Gaussian Likelihood.
        models: The GP model(s) for each output dimension.
        position_memory (Int): the number of previous states that are appended
          together (with the action) to form the GP inputs.
        control_memory (Int): the number of previous actions that are appended
          together (along with the states) to form the GP inputs.
    """

    # pylint: disable=too-many-instance-attributes
    mean_functions: AbstractMeanFunction | List[AbstractMeanFunction] | None
    kernels: AbstractKernel | List[AbstractKernel]
    likelihoods: AbstractLikelihood | List[AbstractLikelihood] | None
    training_data: ArrayLike
    training_outputs: ArrayLike
    data: List[gpjax.Dataset]
    num_outputs: Int
    input_dimension: Int

    position_memory: Int
    control_memory: Int
    name: str | None

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_funcs: AbstractKernel | List[AbstractKernel],
        *,
        mean_funcs: Optional[
            Union[List[AbstractMeanFunction], AbstractMeanFunction]
        ] = None,
        likelihoods: Optional[
            Union[AbstractLikelihood, List[AbstractLikelihood]]
        ] = None,
        position_memory: Int = 2,
        control_memory: Int = 1,
        name: str | None = None,
    ) -> None:
        self.position_memory = position_memory
        self.control_memory = control_memory

        io_data = self.data_to_gp_input_output(states, actions)
        self.training_data, self.training_outputs = io_data
        self.num_outputs = self.training_outputs.shape[1]
        self.input_dimension = self.training_data.shape[1]

        self.data = [
            gpjax.Dataset(
                X=self.training_data, y=self.training_outputs[:, i].reshape(-1, 1)
            )
            for i in range(self.num_outputs)
        ]

        if mean_funcs is None:
            # Give each output a zero mean
            self.mean_functions = [Zero()] * self.num_outputs
        elif isinstance(mean_funcs, list):
            self.mean_functions = mean_funcs
        else:
            # Give each output the same mean
            self.mean_functions = [mean_funcs] * self.num_outputs

        if likelihoods is None:
            # Give each output a Gaussian likelihood
            self.likelihoods = [Gaussian] * self.num_outputs
        elif isinstance(likelihoods, list):
            self.likelihoods = likelihoods
        else:
            # Give each output the same mean
            self.likelihoods = [likelihoods] * self.num_outputs

        if isinstance(kernel_funcs, list):
            self.kernels = kernel_funcs
        else:
            # Give each output the same kernel
            self.kernels = [kernel_funcs] * self.num_outputs

        self.name = name

    def data_to_gp_output(self, states: ArrayLike) -> Array:
        """Transforms data into PILCO data format."""
        return jnp.flip(jnp.diff(states, n=1, axis=0), axis=0)[
            0 : states.shape[0] - max(self.control_memory, self.position_memory), :
        ]

    def data_to_gp_input(self, states: ArrayLike, actions: ArrayLike) -> Array:
        """Transforms data into PILCO data format."""
        reversed_states_array = jnp.flip(states, axis=0)
        states_diff = jnp.diff(reversed_states_array, n=1, axis=0)
        reversed_actions_array = jnp.flip(actions, axis=0)
        return jnp.array(
            [
                jnp.hstack(
                    [
                        jnp.ravel(reversed_states_array[i, :], order="C"),
                        jnp.ravel(
                            states_diff[i : i + self.position_memory, :], order="C"
                        ),
                        jnp.ravel(
                            reversed_actions_array[i : i + self.control_memory, :],
                            order="C",
                        ),
                    ]
                )
                for i in range(
                    0, states.shape[0] - max(self.control_memory, self.position_memory)
                )
            ]
        )

    def data_to_policy_input(
        self, states: ArrayLike, actions: ArrayLike | None = None
    ) -> Array:
        """Transforms data into policy data format."""
        reversed_states_array = jnp.flip(states, axis=0)
        states_diff = jnp.diff(reversed_states_array, n=1, axis=0)
        return jnp.array(
            [
                jnp.hstack(
                    [
                        jnp.ravel(reversed_states_array[i, :], order="C"),
                        jnp.ravel(
                            states_diff[i : i + self.position_memory, :], order="C"
                        ),
                    ]
                )
                for i in range(0, states.shape[0] - self.position_memory)
            ]
        )

    def data_to_gp_input_output(
        self, states: ArrayLike, actions: ArrayLike
    ) -> Tuple[Array, Array]:
        """Transforms data into PILCO data format."""
        return (self.data_to_gp_input(states, actions), self.data_to_gp_output(states))

    def create_models(
        self,
    ) -> List[gpjax.gps.AbstractPosterior]:
        """Create the models for each output dimension."""
        raise NotImplementedError()

    @eqx.filter_jit
    def get_samples(
        self,
        key: ArrayLike,
        states: ArrayLike,
        actions: ArrayLike,
    ) -> Array:
        """Samples `num_samples` draws from the dynamical model.

        Because the GP models the differences in the states, we must add back
          the state to get the state mean (the variance is the same).

        """
        test_inputs = vmap(self.data_to_gp_input)(states, actions)
        ret_samples = []
        for i in range(self.num_outputs):
            latent_dist = self.models[i](test_inputs, train_data=self.data[i])
            key, subkey = jr.split(key)
            white_noise = jr.normal(
                subkey, shape=() + latent_dist.batch_shape + latent_dist.event_shape
            )
            covariance_root = lower_cholesky(latent_dist.scale)
            samples = (
                test_inputs[:, 0, i] + latent_dist.loc + (covariance_root @ white_noise)
            )
            ret_samples.append(samples)
        return jnp.array(ret_samples).T


class IMGPR(DynamicalModel):
    """The forward model of the system dynamics.

    Independent Multiple Gaussian Process regression - has an independent GP
      for every output dimension

    Args:
        states (ArrayLike): The input states $x_t$.
        actions (ArrayLike): The input controls $u_t.$
        kernel_funcs (AbstractKernel): The kernel function(s) for each GP. If
          there are multiple outputs but a single kernel is passed, each GP
          will get this kernel.
        mean_funcs (AbstractMeanFunction): The mean function(s) for each GP.
          If there are multiple outputs, but a single mean is passed, each
          GP will get this mean. If left blank, will default to zero mean
        likelihood (AbstractLikelihood): The likelihood of the posterior. If
          there are multiple outputs and a single likelihood is passed, each
          GP will get that likelihood. If left blank, will default to a
          Gaussian Likelihood.
        models: The GP model(s) for each output dimension.
        position_memory (Int): the number of previous states that are appended
          together (with the action) to form the GP inputs.
        control_memory (Int): the number of previous actions that are appended
          together (along with the states) to form the GP inputs.
    """

    models: List[gpjax.gps.AbstractPosterior]

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_funcs: List[AbstractKernel] | AbstractKernel,
        *,
        mean_funcs: Optional[
            Union[List[AbstractMeanFunction], AbstractMeanFunction]
        ] = None,
        likelihoods: Optional[
            Union[AbstractLikelihood, List[AbstractLikelihood]]
        ] = None,
        models: List[gpjax.gps.AbstractPosterior] | None = None,
        position_memory: Int = 2,
        control_memory: Int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(
            states,
            actions,
            kernel_funcs,
            mean_funcs=mean_funcs,
            likelihoods=likelihoods,
            position_memory=position_memory,
            control_memory=control_memory,
            name=name,
        )
        if models is None:
            self.models = self.create_models()
        else:
            self.models = models

    def create_models(
        self,
    ) -> List[gpjax.gps.AbstractPosterior]:
        """Create GP model posteriors."""
        return [
            gpjax.gps.Prior(
                mean_function=self.mean_functions[i], kernel=self.kernels[i]
            )
            * self.likelihoods[i](num_datapoints=self.data[i].n)
            for i in range(self.num_outputs)
        ]


class IMSVGPR(DynamicalModel):
    """The forward model of the system dynamics.

    Independent Multiple Sparse Variational Gaussian Process regression. Each
      GP has an independent SVGP for every output dimension

    Args:
        kernel (Kernel): The kernel function
        data (JAXArray): The input data. This is either state-action pairs
            $(x_t, u_t)$, or (extension) will be observable-action pairs
            $(y_t, u_t).$
    """

    models: List[gpjax.gps.AbstractPosterior]
    num_inducing_points: Int
    inducing_points: ArrayLike

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        kernel_funcs: List[AbstractKernel] | AbstractKernel,
        num_inducing_points: Int,
        *,
        mean_funcs: Optional[
            Union[List[AbstractMeanFunction], AbstractMeanFunction]
        ] = None,
        likelihoods: Optional[
            Union[AbstractLikelihood, List[AbstractLikelihood]]
        ] = None,
        models: List[gpjax.gps.AbstractPosterior] | None = None,
        inducing_points: ArrayLike | None = None,
        position_memory: Int = 2,
        control_memory: Int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(
            states,
            actions,
            kernel_funcs,
            mean_funcs=mean_funcs,
            likelihoods=likelihoods,
            position_memory=position_memory,
            control_memory=control_memory,
            name=name,
        )
        self.num_inducing_points = num_inducing_points
        if inducing_points is None:
            km = KMeans(n_clusters=self.num_inducing_points, n_init=10)
            km.fit(self.data_to_gp_input(states, actions))
            self.inducing_points = jnp.array(km.cluster_centers_, dtype=jnp.float64)
        else:
            self.inducing_points = inducing_points

        if models is None:
            self.models = self.create_models()
        else:
            self.models = models

    def create_models(
        self,
    ) -> List[gpjax.gps.AbstractPosterior]:
        """Create GP model posteriors."""
        return [
            gpjax.variational_families.CollapsedVariationalGaussian(
                posterior=gpjax.gps.Prior(
                    mean_function=self.mean_functions[i], kernel=self.kernels[i]
                )
                * self.likelihoods[i](num_datapoints=self.data[i].n),
                inducing_inputs=self.inducing_points,
            )
            for i in range(self.num_outputs)
        ]


def optimize_imgpr(
    dynamical_model: IMGPR,
    states: ArrayLike,
    actions: ArrayLike,
) -> IMGPR:
    """Optimize the imgpr dynamical model and return a new instance."""
    opt_models = []
    for i in range(dynamical_model.num_outputs):
        opt_model, _ = gpjax.fit_lbfgs(
            model=dynamical_model.models[i],
            objective=lambda p, d: -gpjax.objectives.conjugate_mll(p, d),
            train_data=dynamical_model.data[i],
            trainable=gpjax.parameters.Parameter,
        )
        opt_models.append(opt_model)

    return IMGPR(
        states=states,
        actions=actions,
        kernel_funcs=dynamical_model.kernels,
        mean_funcs=dynamical_model.mean_functions,
        likelihoods=dynamical_model.likelihoods,
        models=opt_models,
        position_memory=dynamical_model.position_memory,
        control_memory=dynamical_model.control_memory,
        name=dynamical_model.name,
    )


def optimize_imsvgpr(
    dynamical_model: IMGPR,
    states: ArrayLike,
    actions: ArrayLike,
) -> IMSVGPR:
    """Optimize the imgpr dynamical model and return a new instance."""
    opt_models = []
    inducing_points = []
    for i in range(dynamical_model.num_outputs):
        opt_model, _ = gpjax.fit_lbfgs(
            model=dynamical_model.models[i],
            objective=lambda p, d: -gpjax.objectives.collapsed_elbo(p, d),
            train_data=dynamical_model.data[i],
            trainable=gpjax.parameters.Parameter,
        )
        opt_models.append(opt_model)
        inducing_points.append(opt_model.inducing_inputs.value)

    return IMSVGPR(
        states=states,
        actions=actions,
        kernel_funcs=dynamical_model.kernels,
        num_inducing_points=dynamical_model.num_inducing_points,
        mean_funcs=dynamical_model.mean_functions,
        likelihoods=dynamical_model.likelihoods,
        models=opt_models,
        inducing_points=inducing_points,
        position_memory=dynamical_model.position_memory,
        control_memory=dynamical_model.control_memory,
        name=dynamical_model.name,
    )
