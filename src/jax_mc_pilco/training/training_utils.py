"""Utility functions for training."""

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping as jtp
from jax import jit


def train_val_split(
    key: jtp.Key[jtp.Array, ""],
    arrays: Sequence[jtp.Shaped[jtp.Array, "batch ..."]],
    val_prop: float = 0.1,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Random train validation split for a sequence of arrays.

    Args:
        key: Jax random key.
        arrays: Sequence of arrays, with matching size on axis 0.
        val_prop: Proportion of data to use for validation. Defaults to 0.1.

    Returns:
        A tuple containing the train arrays and the validation arrays.
    """
    # assert 0 <= val_prop <= 1, f"val_prop should be between 0 and 1. You provided {val_prop}"

    num_samples = arrays[0].shape[0]
    # if not all(isinstance(a, jtp.Shaped[jtp.Array, " dim ..."]) for a in arrays):
    #     raise ValueError("Array dimensions must match along axis 0.")

    n_train = num_samples - round(val_prop * num_samples)
    arrays = [jax.random.permutation(key, a) for a in arrays]
    train_arrays = [arr[:n_train] for arr in arrays]
    val_arrays = [arr[n_train:] for arr in arrays]
    return train_arrays, val_arrays


@partial(jit, static_argnums=1)
def get_batches(arrays: Sequence[jtp.Array], batch_size: int) -> tuple[jtp.Array, ...]:
    """Reshape a sequence of arrays to have an additional dimension for the batches.

    Specifically, this will transform an array with shape ``(data_len, *rest)`` to
    ``(num_batches, batch_size, *rest)``.

    Considerations:
        - The values in the last batch are dropped if truncated, i.e. if
            ``data_len % batch_size != 0``.
        - If the batch size is greater than the length of the data, then we set the
            batch size to be equal to the data length.

    Args:
        arrays: Sequence of arrays, with shape matching on axis 0.
        batch_size: The batch size.
    """
    # data_len = arrays[0].shape[0]
    # if not all(arr.shape[0] == data_len for arr in arrays):
    #     raise ValueError("Array dimensions do not match along the batch axis.")

    return tuple(_add_batch(arr, batch_size) for arr in arrays)


def _add_batch(arr: jtp.Array, batch_size: int) -> jtp.Array:
    """Adds a leading dimension for batches, dropping the last batch if truncated."""
    batch_size = min(batch_size, arr.shape[0])
    n_batches = arr.shape[0] // batch_size
    return jax.device_put(
        arr[: n_batches * batch_size].reshape(n_batches, batch_size, *arr.shape[1:]), jax.devices("gpu")[0]
    )


def count_fruitless(losses: list[float]) -> int:
    """Count the number of epochs since the minimum loss in a list of losses.

    Args:
        losses: List of losses.

    """
    min_idx = jnp.argmin(jnp.array(losses)).item()
    return len(losses) - min_idx - 1
