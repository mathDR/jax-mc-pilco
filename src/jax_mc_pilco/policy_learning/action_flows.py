### Equinox Module that acts as a world model given actions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import paramax
from flowjax.bijections import Affine, Chain, RationalQuadraticSpline, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow, masked_autoregressive_flow
from paramax import non_trainable

EPSILON = 1e-4


class ActionFlow(eqx.Module):
    """State-conditioned behavior model used to sample imagined actions."""

    flow: Transformed
    action_low: jtp.Array
    action_high: jtp.Array
    state_mean: jtp.Array
    state_std: jtp.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        flow_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ):
        cond_dim = state_dim + hidden_dim

        action_low = jnp.full((action_dim,), -jnp.pi / 2.0)
        action_high = jnp.full((action_dim,), jnp.pi / 2.0)
        self.action_low = paramax.non_trainable(action_low)
        self.action_high = paramax.non_trainable(action_high)

        if state_mean is None:
            state_mean = jnp.zeros((state_dim,))
        if state_std is None:
            state_std = jnp.ones((state_dim,))

        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(
            jnp.where(state_std < 1e-6, 1.0, state_std)
        )

        if base_flow is None:
            unbounded_flow = masked_autoregressive_flow(
                key=key,
                base_dist=MultivariateNormal(
                    loc=jnp.zeros(action_dim), covariance=jnp.eye(action_dim)
                ),
                cond_dim=cond_dim,
                flow_layers=flow_layers,
                invert=True,
                nn_depth=2,
                transformer=RationalQuadraticSpline(knots=12, interval=8),
            )
            eps = 1e-3

            # Pad bounds by eps so true [-pi/2, pi/2] values stay strictly inside (0, 1)
            padded_low = action_low - eps
            padded_scale = (action_high - action_low) + 2.0 * eps
            bound_bijection = Chain(
                [
                    Sigmoid(shape=(action_dim,)),
                    Affine(loc=padded_low, scale=padded_scale),
                ]
            )
            base_flow = Transformed(
                unbounded_flow,
                paramax.non_trainable(bound_bijection),
            )

        self.flow = base_flow

    def normalize_state(self, state: jax.Array) -> jax.Array:
        ret_state: jax.Array =  (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(
            self.state_std
        )
        return ret_state

    def _build_context(self, state: jax.Array, hidden: jax.Array) -> jax.Array:
        norm_state = self.normalize_state(state)
        return jnp.concatenate([norm_state, hidden], axis=-1)

    def _clip_action(self, raw_sample: jax.Array) -> jax.Array:
        clipped = jnp.clip(
            raw_sample,
            paramax.unwrap(self.action_low),
            paramax.unwrap(self.action_high),
        )
        return raw_sample + jax.lax.stop_gradient(clipped - raw_sample)

    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Sample a bounded action conditioned on the current context."""
        context = self._build_context(state, hidden)
        raw_sample = self.flow.sample(key, (), context)
        return self._clip_action(raw_sample)

    def sample_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        state: jax.Array,
        hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample both a bounded action and its log density."""
        context = self._build_context(state, hidden)
        raw_sample, log_prob = self.flow.sample_and_log_prob(key, (), context)
        action = self._clip_action(raw_sample)
        return action, log_prob

    def log_prob(
        self,
        action: jax.Array,
        state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Return the behavior-model log density of an observed action."""
        context = self._build_context(state, hidden)
        return self.flow.log_prob(action, context)

class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the state s_t
    """

    flow: Transformed
    action_low: jax.Array
    action_high: jax.Array
    margin: jax.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        cond_dim: int,
        action_dim: int,
        action_low: jax.Array,
        action_high: jax.Array,
        flow_layers: int = 4,
    ):
        self.action_low = jnp.broadcast_to(action_low, (action_dim,))
        self.action_high = jnp.broadcast_to(action_high, (action_dim,))

        base_dist = MultivariateNormal(
            loc=jnp.zeros(action_dim),
            covariance=jnp.eye(action_dim),
        )
        base_flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

        loc = action_low - EPSILON
        self.margin = 2 * EPSILON * (self.action_high - self.action_low)
        squash = non_trainable(Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=self.margin)]))
        full_bijection = Chain([base_flow.bijection, squash])
        self.flow = Transformed(base_dist, full_bijection)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],
        context: jax.Array,
    ) -> jax.Array:
        return self.flow.sample(key, condition=context)

    def sample_action_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        context: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action, logp = self.flow.sample_and_log_prob(key, condition=context)
        return action, logp

    def log_prob(
        self,
        action: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        return self.flow.log_prob(action, condition=context)
