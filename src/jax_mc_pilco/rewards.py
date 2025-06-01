"""The respective reward functions."""

import math
import jax.numpy as jnp
from jaxtyping import ArrayLike, Array, Float


from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import transforms


def cart_pole_cost(
    states_sequence: ArrayLike,
    target_state: ArrayLike = jnp.array(
        [
            0.0,
            0.0,
        ]
    ),
    lengthscales: ArrayLike = jnp.array([3.0, 1.0]),
) -> Array:
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    # cart_pos = states_sequence[0]
    # sin_pole_angle = states_sequence[1]
    # cos_pole_angle = states_sequence[2]
    # cart_velocity = states_sequence[3]
    # angle_velocity = states_sequence[4]
    cart_pos = states_sequence[0]
    pole_angle = states_sequence[1]
    cart_velocity = states_sequence[2]
    angle_velocity = states_sequence[3]
    target_theta = 0.0
    target_cart_velocity = 0.0
    target_theta_dot = 0.0

    # return 1 - jnp.exp(
    #     -(jnp.square((sin_pole_angle - target_theta) / lengthscales[0]))
    #     - (jnp.square((cos_pole_angle - 1.0) / lengthscales[0]))
    #     - (jnp.square((angle_velocity - target_theta_dot) / lengthscales[1]))
    #     - (jnp.square((cart_velocity - target_cart_velocity) / lengthscales[1]))
    # )
    return 1 - jnp.exp(
        -(jnp.square((pole_angle - target_theta) / lengthscales[0]))
        - (jnp.square((angle_velocity - target_theta_dot) / lengthscales[1]))
        - (jnp.square((cart_velocity - target_cart_velocity) / lengthscales[1]))
    )


def pendulum_cost(
    states_sequence: ArrayLike,
    target_state: ArrayLike = jnp.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    ),
    lengthscales: ArrayLike = jnp.array([3.0, 3.0, 1.0]),
) -> Array:
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle'.
    """

    cos_pole_angle = states_sequence[0]
    sin_pole_angle = states_sequence[1]
    angle_velocity = states_sequence[2]
    target_theta = 0.0
    target_theta_dot = target_state[2]

    return 1 - jnp.exp(
        -(jnp.square((sin_pole_angle - jnp.sin(target_theta)) / lengthscales[0]))
        - (jnp.square((angle_velocity - target_theta_dot) / lengthscales[1]))
        - (jnp.square((sin_pole_angle - jnp.cos(target_theta)) / lengthscales[0]))
    )


def balloon_cost(
    simulator_state: simulator_data.SimulatorState,
    *,
    station_keeping_radius_km: Float = 50.0,
    reward_dropoff: Float = 0.4,
    reward_halflife: Float = 100.0
) -> Float:
    balloon_state = simulator_state.balloon_state
    x, y = balloon_state.x, balloon_state.y
    radius = units.Distance(km=station_keeping_radius_km)

    # x, y are in meters.
    distance = units.relative_distance(x, y)

    # Base reward - distance to station keeping radius.
    if distance <= radius:
        # Reward is 1.0 within the radius.
        reward = 1.0
    else:
        # Exponential decay outside boundary with drop
        # ln(0.5) is approximately -0.69314718056.
        reward = reward_dropoff * math.exp(
            -0.69314718056 / reward_halflife * (distance - radius).kilometers
        )

    # Power regularization. Only applied when using more power (going down)
    # and there isn't excess energy available.
    if (
        balloon_state.last_command == control.AltitudeControlCommand.DOWN
        and not balloon_state.excess_energy
    ):
        max_multiplier = 0.95
        penalty_skew = 0.3
        scale = transforms.linear_rescale_with_saturation(
            balloon_state.acs_power.watts, 100.0, 300.0
        )
        multiplier = max_multiplier - penalty_skew * scale
        reward *= multiplier

    return reward
