import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


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
    cart_pos = states_sequence[0]
    sin_pole_angle = states_sequence[1]
    cos_pole_angle = states_sequence[2]
    cart_velocity = states_sequence[3]
    angle_velocity = states_sequence[4]
    target_theta = 0.0
    target_cart_velocity = 0.0
    target_theta_dot = 0.0

    return 1 - jnp.exp(
        -(jnp.square((sin_pole_angle - target_theta) / lengthscales[0]))
        - (jnp.square((cos_pole_angle - 1.0) / lengthscales[0]))
        - (jnp.square((angle_velocity - target_theta_dot) / lengthscales[1]))
        - (jnp.square((cart_velocity - target_cart_velocity) / lengthscales[1]))
    )
