from __future__ import annotations

from collections.abc import Generator
from typing import List
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax import Array
from jax.typing import ArrayLike


class Controller(eqx.Module):
    """
    Superclass of controller objects
    """

    state_dim: int
    action_dim: int
    max_action: float
    f_squash: Callable

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
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
        time_for_action: float,
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
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
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
        time_for_action: float,
        key: ArrayLike | None = None,
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
        state_dim: int,
        action_dim: int,
        to_squash: bool = False,
        max_action: float = 1.0,
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

    def reduce_params(self, params: ArrayLike) -> Tuple[Array, Array]:
        """Break params into its respective components"""
        return (params[:, :-1], params[:, -1])

    def __call__(
        self,
        state_mean: ArrayLike,
        time_for_action: float,
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


def set_sos_params(
    action_dim: int,
    num_sin: int,
    omega_min: ArrayLike,
    omega_max: ArrayLike,
    amplitude_min: ArrayLike,
    amplitude_max: ArrayLike,
    key: Optional[ArrayLike] = None,
) -> Array:
    if key is None:
        key = jr.key(123)

    # generate random parameters
    key, subkey = jr.split(key)
    amplitudes: ArrayLike = jr.uniform(
        subkey,
        shape=(num_sin, action_dim),
        minval=amplitude_min,
        maxval=amplitude_max,
    )

    key, subkey = jr.split(key)
    omega: ArrayLike = jr.uniform(
        subkey,
        shape=(
            num_sin,
            action_dim,
        ),
        minval=omega_min,
        maxval=omega_max,
    )

    key, subkey = jr.split(key)
    phases: ArrayLike = jr.uniform(
        subkey,
        shape=(num_sin, action_dim),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )

    return jnp.vstack((amplitudes, omega, phases))


class Sum_of_Sinusoids(Controller):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    num_sin: int

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_sin: int,
        to_squash: bool = False,
        max_action: float = 1.0,
    ) -> Tuple[Array, Array, Array]:
        super().__init__(
            state_dim,
            action_dim,
            to_squash,
            max_action,
        )

        self.num_sin = num_sin

    def reduce_params(self, params: ArrayLike) -> Tuple[Array, Array, Array]:
        """Break params into its respective components"""
        return (
            params[: self.num_sin],
            params[self.num_sin : 2 * self.num_sin],
            params[2 * self.num_sin :],
        )

    def __call__(
        self,
        params: ArrayLike,
        state: ArrayLike,
        t: ArrayLike,
    ) -> Array:
        amplitudes, omega, phases = self.reduce_params(params)
        return self.f_squash(
            jnp.sum(
                amplitudes * (jnp.sin(omega * t + phases)),
                axis=0,
            ).reshape(
                self.action_dim,
            )
        )
