from typing import Literal, Union

from jaxtyping import Array, Float, Int

_BATCH_ARRAY = Float[Array, "bs ..."]
_LABEL_ARRAY = Union[Float[Array, " bs"], Int[Array, " bs"]]
_OT_PLANS = Literal["exact", "sinkhorn", "unbalanced", "partial"]
_TIME = Union[Float[Array, " bs"], Int[Array, " bs"], float, int]
