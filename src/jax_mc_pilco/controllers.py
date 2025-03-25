from __future__ import annotations

from collections.abc import Generator
from typing import List
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax import Array
from jax.typing import ArrayLike


class Controller(nnx.Module):
    """
    Superclass of controller objects
    """

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
            self.f_squash = lambda x: self.squashing(x)
        else:
            # assign the identity function
            self.f_squash = lambda x: x

    def __call__(
        self,
        state: ArrayLike,
        params: ArrayLike,
        time_for_action: float,
    ) -> Array:
        """Generate an action from the controller in state `state` at time `time_for_action`."""
        raise NotImplementedError()

    def squashing(self, u: Array) -> ArrayLike:
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


class Sum_of_Sinusoids(Controller):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_sin: int,
        omega_min: ArrayLike,
        omega_max: ArrayLike,
        amplitude_min: ArrayLike,
        amplitude_max: ArrayLike,
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
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.max_action = max_action

        # # set squashing function
        # self.f_squash = lambda x: self.squashing(x)

        if key is None:
            key = jr.key(123)
        self.num_sin = num_sin
        # generate random parameters
        key, subkey = jr.split(key)
        self.amplitudes: nnx.Variable = nnx.Variable(
            jr.uniform(
                subkey,
                shape=(num_sin, action_dim),
                minval=amplitude_min,
                maxval=amplitude_max,
            ),
        )
        key, subkey = jr.split(key)
        self.omega: nnx.Variable = nnx.Variable(
            jr.uniform(
                subkey,
                shape=(
                    num_sin,
                    action_dim,
                ),
                minval=omega_min,
                maxval=omega_max,
            ),
        )
        key, subkey = jr.split(key)
        self.phases: nnx.Variable = nnx.Variable(
            jr.uniform(
                subkey,
                shape=(num_sin, action_dim),
                minval=-jnp.pi,
                maxval=jnp.pi,
            ),
        )

    # def reduce_params(self, params: ArrayLike) -> Tuple[Array, Array, Array]:
    #     """Break params into its respective components"""
    #     return (
    #         params[: self.num_sin],
    #         params[self.num_sin : 2 * self.num_sin],
    #         params[2 * self.num_sin :],
    #     )

    def __call__(
        self,
        state: ArrayLike,
        t: ArrayLike,
    ) -> Array:
        # returns the controller values at times t
        # return self.f_squash(
        #     jnp.sum(
        #         self.amplitudes * (jnp.sin(self.omega * t + self.phases)), axis=0
        #     ).reshape(
        #         self.action_dim,
        #     )
        # )
        # amplitudes, omega, phases = self.reduce_params(params)
        return jnp.sum(
            self.amplitudes * (jnp.sin(self.omega * t + self.phases)),
            axis=0,
        ).reshape(
            self.action_dim,
        )
