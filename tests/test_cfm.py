from typing import Callable

import jax
import pytest

from jax_mc_pilco.model_learning.flows import CFM, ExactOptimalTransportCFM, SchrodingerBridgeCFM, _random_t

# chosen to test different "cases" including data types
SIGMAS = [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, 0, 1]
N_SAMPLES = 100
# chosen to test different numbers of dimensions
SHAPES = [[N_SAMPLES, *dims] for dims in [[1], [10], [10, 40], [10, 20, 30]]]
X_ARRS = [
    (
        jax.random.normal(jax.random.PRNGKey(123), shape=shape, dtype=float),
        jax.random.normal(jax.random.PRNGKey(423), shape=shape, dtype=float),
    )
    for shape in SHAPES
]


class TestConditionFlowMatchers:
    key = jax.random.PRNGKey(42)

    def run_conditional_flow_matching(self, cfm: CFM, x0, x1) -> Callable:
        cfm_jit = jax.jit(cfm)
        self.key, key1, key2 = jax.random.split(self.key, 3)
        # without specifying t
        _ = cfm_jit(key1, x0, x1)
        # with specified t
        key = jax.random.split(self.key)
        return cfm_jit(key2, x0, x1, _random_t(key[1], [x0.shape[0]]).reshape(-1))

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_cfm(self, sigma, xs):
        t, xt, ut, noise = self.run_conditional_flow_matching(CFM(sigma), *xs)

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_exact_cfm(self, sigma, xs):
        # TODO: actual tests once exact plan is implemented
        with pytest.raises(NotImplementedError):
            t, xt, ut, noise = self.run_conditional_flow_matching(ExactOptimalTransportCFM(sigma), *xs)

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_schrodinger_cfm(self, sigma, xs):
        if xs[0].ndim < 3:
            t, xt, ut, noise = self.run_conditional_flow_matching(SchrodingerBridgeCFM(sigma), *xs)
        else:
            with pytest.raises(ValueError):
                t, xt, ut, noise = self.run_conditional_flow_matching(SchrodingerBridgeCFM(sigma), *xs)
