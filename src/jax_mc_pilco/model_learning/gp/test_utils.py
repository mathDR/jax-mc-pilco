from typing import Any

import jax
from jax._src.public_test_util import check_close


def assert_allclose(
    calculated: jax.Array, expected: jax.Array, *args: Any, **kwargs: Any
):
    kwargs["atol"] = kwargs.get(
        "atol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    kwargs["rtol"] = kwargs.get(
        "rtol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    check_close(calculated, expected, *args, **kwargs)


def assert_pytrees_allclose(calculated: Any, expected: Any, *args: Any, **kwargs: Any):
    jax.tree_util.tree_map(
        lambda a, b: assert_allclose(a, b, *args, **kwargs), calculated, expected
    )
