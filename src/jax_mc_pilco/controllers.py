"""The available Policies for the repo."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Int


class Controller(eqx.Module):
    """
    Superclass of controller objects
    """

    state_dim: Int
    action_dim: Int
    max_action: Float
    f_squash: Callable

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: Float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # set squashing function
        if to_squash:
            self.f_squash = eqx.nn.Lambda(lambda x: self.squashing(x))
        else:
            # assign the identity function
            self.f_squash = eqx.nn.Lambda(lambda x: x)

    def __call__(
        self,
        state: ArrayLike,
        params: ArrayLike,
        time_for_action: Float,
    ) -> Array:
        """Generate an action from the controller in state `state` at time `time_for_action`."""
        raise NotImplementedError()

    def squashing(self, u: ArrayLike) -> Array:
        """
        Squash the inputs inside (-max_action, +max_action)
        """
        return self.max_action * jnp.tanh(u / self.max_action)


class RandomController(Controller):
    """Returns a random control output"""

    def __init__(
        self,
        state_dim: Int,
        action_dim: Int,
        to_squash: bool = False,
        max_action: Float = 1.0,
    ):
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )

    def __call__(
        self,
        state: ArrayLike,
        time_for_action: Float,
        key: Optional[ArrayLike] = None,
    ) -> Array:
        """
        Simple random action
        IN: current state, time_for_action and key to use for random action
        OUT: the action value (uniform in (-max_action,+max_action))
        """
        if key is not None:
            key, subkey = jr.split(key)
        else:
            key = jr.key(123)
            key, subkey = jr.split(key)

        return jr.uniform(
            subkey,
            shape=(self.action_dim,),
            minval=-self.max_action,
            maxval=self.max_action,
        )


class LinearPolicy(Controller):
    """
    Linear Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning
    in Robotics and Control
    Section 3.5.2 (pg 43)
    """

    phi: ArrayLike
    offset: ArrayLike

    def __init__(
        self,
        state_dim: Int,
        action_dim: Int,
        to_squash: bool = False,
        max_action: Float = 1.0,
        key: Optional[ArrayLike] = None,
    ):
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )
        if key is not None:
            key, subkey = jr.split(key)
        else:
            key = jr.key(123)
            key, subkey = jr.split(key)

        self.phi = jr.uniform(
            subkey, shape=(action_dim, state_dim)
        )  # parameter matrix of weights (n, D)
        key, subkey = jr.split(key)
        self.offset = jr.uniform(
            subkey, shape=(1, action_dim)
        )  # offset/bias vector (1, D )

    def __call__(
        self,
        state_mean: ArrayLike,
        time_for_action: Float,
        key: Optional[ArrayLike] = None,
    ) -> Tuple[Array, Array]:
        """
        Predict Gaussian distribution for action given a state distribution input
        :param params: concatenated weight matrix and offset
        :param m: mean of the state
        :return: mean (M) of action
        """
        action_mean = jnp.dot(self.phi, state_mean) + self.offset

        return action_mean.reshape(
            self.action_dim,
        )


class SumOfSinusoids(Controller):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    num_sin: Int
    amplitudes: ArrayLike
    omega: ArrayLike
    phases: ArrayLike

    def __init__(
        self,
        state_dim: Int,
        action_dim: Int,
        num_sin: Int,
        omega_min: ArrayLike,
        omega_max: ArrayLike,
        amplitude_min: ArrayLike,
        amplitude_max: ArrayLike,
        to_squash: bool = False,
        max_action: Float = 1.0,
        key: Optional[ArrayLike] = None,
    ):
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )

        self.num_sin: Int = num_sin

        if key is None:
            key = jr.key(123)

        # generate random parameters
        key, subkey = jr.split(key)
        self.amplitudes: ArrayLike = jr.uniform(
            subkey,
            shape=(num_sin, action_dim),
            minval=amplitude_min,
            maxval=amplitude_max,
        )

        key, subkey = jr.split(key)
        self.omega: ArrayLike = jr.uniform(
            subkey,
            shape=(
                num_sin,
                action_dim,
            ),
            minval=omega_min,
            maxval=omega_max,
        )

        key, subkey = jr.split(key)

        self.phases: ArrayLike = jr.uniform(
            subkey,
            shape=(num_sin, action_dim),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    def __call__(
        self,
        state: ArrayLike,
        timestep: ArrayLike,
    ) -> Array:
        return self.f_squash(
            jnp.sum(
                self.amplitudes * (jnp.sin(self.omega * timestep + self.phases)),
                axis=0,
            ).reshape(
                self.action_dim,
            )
        )


class SumOfGaussians(Controller):
    """
    Control policy: sum of 'num_basis' gaussians
    """

    num_basis: Int
    log_lengthscales: ArrayLike
    centers: ArrayLike
    f_linear: eqx.nn.Linear
    scale_factor: Optional[ArrayLike]
    f_drop: Union[eqx.nn.Dropout, eqx.nn.Linear]

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        initial_log_lengthscales: Optional[ArrayLike] = None,
        initial_centers: Optional[ArrayLike] = None,
        centers_init_min: Float = -1.0,
        centers_init_max: Float = 1.0,
        use_bias: bool = True,
        scale_factor: Optional[ArrayLike] = None,
        use_dropout: bool = True,
        dropout_probability: Float = 0.5,
        to_squash: bool = False,
        max_action: Float = 1.0,
        key: Optional[ArrayLike] = None,
    ):
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )
        if key is None:
            key = jr.key(123)

        # set number of gaussian basis functions
        self.num_basis = num_basis
        # get initial log lengthscales
        if initial_log_lengthscales is None:
            initial_log_lengthscales = jnp.ones(state_dim)
        self.log_lengthscales = jnp.log(initial_log_lengthscales).reshape([1, -1])

        # get initial centers
        if initial_centers is None:
            key, subkey = jr.split(key)
            initial_centers = centers_init_min * jnp.ones([num_basis, state_dim]) + (
                centers_init_max - centers_init_min
            ) * jr.uniform(subkey, shape=(num_basis, state_dim))
        self.centers = initial_centers
        # initilize the linear ouput layer
        key, subkey = jr.split(key)
        self.f_linear = eqx.nn.Linear(
            in_features=num_basis,
            out_features=action_dim,
            use_bias=use_bias,
            key=subkey,
        )

        if scale_factor is None:
            scale_factor = jnp.ones(state_dim)
        self.scale_factor = scale_factor.reshape([1, -1])

        # set dropout
        # Could do this with the inference flag in the Dropout class, but this is more readable
        if use_dropout:
            self.f_drop = eqx.nn.Dropout(p=dropout_probability)
        else:
            self.f_drop = eqx.nn.Lambda(lambda x: x)

    def __call__(
        self,
        states: ArrayLike,
        timestep: Optional[Float] = None,
        key: Optional[ArrayLike] = None,
    ):
        """
        Returns a linear combination of gaussian functions
        with input given by the the distances between that state
        and the vector of centers of the gaussian functions
        """
        if key is None:
            key = jr.key(123)
        # get the lengthscales from log
        lengthscales = jnp.exp(self.log_lengthscales)

        states = states.reshape([-1, self.state_dim])
        # normalize states and centers
        norm_states = states / lengthscales
        norm_centers = self.centers / lengthscales
        # get the square distances
        distances = jnp.squeeze(
            jnp.linalg.norm(norm_states[:, None, :] - norm_centers[None, :, :], axis=2)
        )
        rbf_activations = jnp.exp(-0.5 * jnp.square(distances))
        # apply exp and get output
        key, subkey = jr.split(key)
        exp_dist_dropped = self.f_drop(rbf_activations, key=subkey)
        inputs = self.f_linear(exp_dist_dropped).reshape(
            [
                self.action_dim,
            ]
        )

        # returns the constrained control action
        return self.f_squash(inputs)


# class SumOfGaussiansWithAngles(SumOfGaussians):
#     """
#     Extends sum of gaussians policy. Angle indices are mapped in cos and sin before computing
#     the policy
#     """

#     angle_indices: ArrayLike
#     non_angle_indices: ArrayLike
#     num_angle_indices: Int
#     num_non_angle_indices: Int

#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         num_basis: int,
#         angle_indices: ArrayLike,
#         non_angle_indices: ArrayLike,
#         initial_log_lengthscales: Optional[ArrayLike] = None,
#         initial_centers: Optional[ArrayLike] = None,
#         centers_init_min: Float = -1.0,
#         centers_init_max: Float = 1.0,
#         use_bias: bool = True,
#         scale_factor: Optional[ArrayLike] = None,
#         use_dropout: bool = True,
#         dropout_probability: Float = 0.5,
#         to_squash: bool = False,
#         max_action: Float = 1.0,
#         key: Optional[ArrayLike] = None,
#     ):
#         self.angle_indices = angle_indices
#         self.non_angle_indices = non_angle_indices
#         self.num_angle_indices = angle_indices.size
#         self.num_non_angle_indices = non_angle_indices.size

#         super(SumOfGaussiansWithAngles, self).__init__(
#             state_dim + self.num_angle_indices,
#             action_dim,
#             num_basis,
#             initial_log_lengthscales,
#             initial_centers,
#             centers_init_min,
#             centers_init_max,
#             use_bias,
#             scale_factor,
#             use_dropout,
#             dropout_probability,
#             to_squash,
#             max_action,
#             key,
#         )

#     def __call__(
#         self,
#         states: ArrayLike,
#         timestep: Optional[Float] = None,
#         key: Optional[ArrayLike] = None,
#     ):
#         # build a state with non angle features and cos,sin of angle features
#         states = jnp.squeeze(states)
#         #     states.reshape([-1, self.state_dim - self.num_angle_indices])
#         # )
#         new_states = jnp.concatenate(
#             [
#                 jnp.take(states, self.non_angle_indices),
#                 jnp.cos(jnp.take(states, self.angle_indices)),
#                 jnp.sin(jnp.take(states, self.angle_indices)),
#             ]
#         ).reshape([-1, self.state_dim])
#         return super(SumOfGaussiansWithAngles, self).__call__(new_states, timestep, key)
