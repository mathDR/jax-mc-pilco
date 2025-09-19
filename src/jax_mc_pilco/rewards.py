import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def cart_pole_cost(
    states_sequence: ArrayLike,
    lengthscales: ArrayLike = jnp.array([3.0, 1.0]),
) -> Array:
    """
    Cost function given by the combination of the saturated distance between
      |theta| and 'target angle', and between x and 'target position'.
    """

    pole_angle = states_sequence[1]
    cart_velocity = states_sequence[2]
    angle_velocity = states_sequence[3]
    target_theta = 0.0
    target_cart_velocity = 0.0
    target_theta_dot = 0.0
    return 1 - jnp.exp(
        -(jnp.square((pole_angle - target_theta) / lengthscales[0]))
        - (jnp.square((angle_velocity - target_theta_dot) / lengthscales[1]))
        - (
            jnp.square(
                (cart_velocity - target_cart_velocity) / lengthscales[1]
            )
        )
    )


def pendulum_cost(
    states_sequence: ArrayLike,
) -> Array:
    """
    Replicated Cost function from gymnasium:
        -(theta**2 + 0.1*theta_dt**2 + 0.001*torque**2)
    but we minimize it, so we return the negation.
    """

    x = states_sequence[0]
    y = states_sequence[1]
    angle_velocity = states_sequence[2]
    torque = states_sequence[3]
    theta = jnp.atan2(x, y)

    return (
        jnp.square(theta) +
        0.1*jnp.square(angle_velocity) +
        0.001*jnp.square(torque)
        )
