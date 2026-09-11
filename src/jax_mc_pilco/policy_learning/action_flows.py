### Equinox Module that acts as a world model given actions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.bijections import Affine, Chain, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow
from paramax import non_trainable

EPSILON = 1e-4


class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the context [s_{t-1}, s_t]
    """

    flow: Transformed
    action_low: jax.Array
    action_high: jax.Array
    margin: jax.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        context_dim: int,
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
            cond_dim=context_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

        loc = action_low - EPSILON
        self.margin = 2 * EPSILON * (self.action_high - self.action_low)
        squash = non_trainable(Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=self.margin)]))
        full_bijection = Chain([base_flow.bijection, squash])
        self.flow = Transformed(base_dist, full_bijection)

    def _debounce(self, action: jax.Array) -> jax.Array:
        return jnp.clip(action, self.action_low + self.margin, self.action_high - self.margin)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],
        context: jax.Array,
    ) -> jax.Array:
        action = self.flow.sample(key, condition=context)
        return self._debounce(action)

    def sample_action_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        context: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action, _ = self.flow.sample_and_log_prob(key, condition=context)
        action = self._debounce(action)
        logp = self.flow.log_prob(action, condition=context)
        return action, logp

    def log_prob(
        self,
        action: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        action = self._debounce(action)
        return self.flow.log_prob(action, condition=context)
