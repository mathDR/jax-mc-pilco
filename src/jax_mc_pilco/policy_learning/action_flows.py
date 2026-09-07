### Equinox Module that acts as a world model given actions."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.bijections import Affine, Chain, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow
from paramax import non_trainable


class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the context [s_{t-1}, s_t]
    """

    flow: Transformed
    state_dim: int
    action_dim: int
    action_low: jax.Array
    action_high: jax.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        action_low: jax.Array,
        action_high: jax.Array,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        cond_dim = state_dim * 2
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
        # Sigmoid: R -> (0, 1); then affine: (0, 1) -> (low, high).
        # These bounds are fixed action-space limits, not learned parameters:
        # wrap the squash bijection in paramax.non_trainable so flowjax's
        # internal `unwrap()` (called at the top of every log_prob/sample/
        # sample_and_log_prob) applies stop_gradient to its loc/scale before
        # use. Without this, PPO's gradient updates silently drift the action
        # bounds themselves - and once action_high shrinks below an
        # already-sampled action, log_prob for that action diverges.
        loc = action_low
        scale = action_high - action_low
        squash = non_trainable(Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=scale)]))
        full_bijection = Chain([base_flow.bijection, squash])
        self.flow = Transformed(base_dist, full_bijection)

    # A sample that lands on (or numerically rounds to) the exact action
    # boundary makes the squash bijection's inverse compute log(0), a true
    # NaN that survives naive clipping. Nudging samples a hair inside the
    # bounds keeps log_prob's inverse-Sigmoid step well defined.
    _EPS = 1e-4

    def _debounce(self, action: jax.Array) -> jax.Array:
        margin = self._EPS * (self.action_high - self.action_low)
        return jnp.clip(action, self.action_low + margin, self.action_high - margin)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        action = self.flow.sample(key, condition=context)
        return self._debounce(action)

    def sample_action_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        action, _ = self.flow.sample_and_log_prob(key, condition=context)
        action = self._debounce(action)
        # log_prob changes negligibly for an eps-sized nudge, but recompute
        # exactly so the stored (action, log_prob) pair stays consistent.
        logp = self.flow.log_prob(action, condition=context)
        return action, logp

    def log_prob(
        self,
        action: jax.Array,
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        action = self._debounce(action)
        return self.flow.log_prob(action, condition=context)
