import jax
import pytest
from jaxcfm import OTSampler

N_SAMPLES = 100
# these are chosen to test different numbers of dimensions
SHAPES = [[N_SAMPLES, *dims] for dims in [[1], [10], [10, 40], [10, 20, 30]]]
X_ARRS = [
    (
        jax.random.normal(jax.random.PRNGKey(123), shape=shape, dtype=float),
        jax.random.normal(jax.random.PRNGKey(423), shape=shape, dtype=float),
    )
    for shape in SHAPES
]


class TestOTSampler:
    key = jax.random.PRNGKey(123)

    y0 = jax.random.randint(jax.random.PRNGKey(12), [N_SAMPLES], 0, 3)
    y1 = jax.random.randint(jax.random.PRNGKey(42), [N_SAMPLES], 0, 3)

    def run_sampler_test(self, ot_sampler: OTSampler, x0, x1):
        # plan_sampler = jax.jit(ot_sampler.sample_plan)
        self.key, key1, key2 = jax.random.split(self.key, 3)
        ot_sampler.sample_plan(key1, x0, x1, None, None)
        ot_sampler.sample_plan(key2, x0, x1, self.y0, self.y1)

        # TODO
        # ot_sampler.sample_trajectory()

    @pytest.mark.parametrize("norm", [True, False])
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_exact(self, xs, norm):
        # TODO
        pass

    @pytest.mark.parametrize("norm", [True, False])
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_sinkhorn(self, xs, norm):
        sampler = OTSampler(solver="sinkhorn", normalize_cost=norm)
        if xs[0].ndim < 3:
            self.run_sampler_test(sampler, *xs)
        else:
            # TODO: remove once implemented
            with pytest.raises(ValueError):
                self.run_sampler_test(sampler, *xs)

    @pytest.mark.parametrize("norm", [True, False])
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_partial(self, xs, norm):
        # TODO
        pass

    @pytest.mark.parametrize("norm", [True, False])
    @pytest.mark.parametrize("xs", X_ARRS)
    def test_unbalanced(self, xs, norm):
        # TODO
        pass

    def test_incorrect(self):
        with pytest.raises(ValueError):
            OTSampler(self.key, "UnknownSolver")
