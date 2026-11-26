import warnings
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from ott.geometry.costs import CostFn, SqEuclidean
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear import sinkhorn

# from typeguard import typechecked
from jax_mc_pilco.model_learning.flows._types import _BATCH_ARRAY, _LABEL_ARRAY, _OT_PLANS


def _check_finite(mat: jnp.ndarray) -> jnp.ndarray:
    warnings.warn("Transport matrix contains non-finite values. Defaulting to uniform transport plan")
    return jnp.ones_like(mat) / mat.size


def _check_stability(mat: jnp.ndarray) -> jnp.ndarray:
    warnings.warn("Numerically unstable transport plan. Defaulting to uniform plan.")
    return jnp.ones_like(mat) / mat.size


class OTSampler(eqx.Module):
    """Optimal Transport Sampler

    The class provides coordinate sampling with different OT plans.

    This class is implemented as an :py:class:`eqx.Module` and thus compatible
    with :py:meth:`jax.jit`.
    """

    solver: callable
    reg: float
    reg_m: float
    cost_fun: CostFn
    normalize_cost: bool

    def __init__(
        self,
        solver: _OT_PLANS,
        reg: float = 0.05,
        reg_m: float = 1.0,
        cost_fun: CostFn = SqEuclidean,
        normalize_cost: bool = False,
        **kwargs,
    ):
        match solver:
            case "exact":
                raise NotImplementedError("Exact solver is not implemented")
            case "sinkhorn":
                self.solver = sinkhorn.Sinkhorn(**kwargs)
            case "unbalanced":
                raise NotImplementedError("Unbalanced solver is not implemented")
            case "partial":
                raise NotImplementedError("Partial solver is not implemented")
            case _:
                raise ValueError(f"Unknown OT solver: {solver}")

        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.cost_fun = cost_fun

    def __str__(self):
        return "OTSampler"

    # @jaxtyped(typechecker=typechecked)
    def get_map(self, x0: _BATCH_ARRAY, x1: _BATCH_ARRAY):
        if x0.ndim > 2:
            raise ValueError(f"Currently only 2D matrices are supported. You provided a {x0.ndim}D matrix.")
        geometry = PointCloud(
            x0, x1, cost_fn=self.cost_fun(), scale_cost="max_cost" if self.normalize_cost else 1.0, epsilon=self.reg
        )
        # uniform probabilities
        a = jnp.ones(x0.shape[0]) / x0.shape[0]
        b = jnp.ones(x1.shape[0]) / x1.shape[0]
        # setup OT problem
        problem = LinearProblem(geometry, a, b)
        p = self.solver(problem).matrix

        p = jax.lax.cond(jnp.all(jnp.isfinite(p)), lambda: _check_finite(p), lambda: p)
        p = jax.lax.cond(jnp.abs(p.sum()) < 1e-10, lambda: _check_stability(p), lambda: p)
        return p

    # @jaxtyped(typechecker=typechecked)
    def sample_map(
        self,
        key: PRNGKeyArray,
        pi: Float[Array, "bs bs"],
        batch_size: int,
        replace: bool = True,
    ) -> tuple[Int[Array, " bs"], Int[Array, " bs"]]:
        choices = jax.random.choice(key, np.prod(pi.shape), shape=[batch_size], replace=replace)
        return jnp.divmod(choices, pi.shape[1])

    # er=typechecked)
    def sample_plan(
        self,
        key: PRNGKeyArray,
        x0: _BATCH_ARRAY,
        x1: _BATCH_ARRAY,
        y0: _LABEL_ARRAY | None = None,
        y1: _LABEL_ARRAY | None = None,
        replace: bool = True,
    ) -> Union[
        tuple[_BATCH_ARRAY, _BATCH_ARRAY],
        tuple[_BATCH_ARRAY, _BATCH_ARRAY, Union[_LABEL_ARRAY, None], Union[_LABEL_ARRAY, None]],
    ]:
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(key, pi, x0.shape[0], replace=replace)
        if y0 is None:
            if y1 is None:
                return x0[i], x1[j]
            return x0[i], x1[j], None, y1[j]
        if y1 is None:
            return x0[i], x1[j], y0[i], None
        return x0[i], x1[j], y0[i], y1[j]

    # @jaxtyped(typechecker=typechecked)
    def sample_trajectory(self, X: Float[Array, "bs times dim"]) -> Float[Array, "bs times dim"]:
        pops = [self.get_map(X[:, t], X[:, t + 1]) for t in jnp.arange(X.shape[1] - 1)]
        return pops
