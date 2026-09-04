### Equinox Module that produces actions given a state using a CNF."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.bijections import Affine, Chain, Sigmoid
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow


class FlowActor(eqx.Module):
    """
    Conditional Normalizing Flow policy.
    Maps noise z ~ N(0, I) -> Action given the context [s_{t-1}, s_t]
    """

    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        state_dim: int,
        action_dim: int,
        action_low: jax.Array,
        action_high: jax.Array,
        flow_layers: int = 4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Context consists of the last two states concatenated
        cond_dim = state_dim * 2

        # Define a standard Gaussian base distribution matching the action dimension
        base_dist = MultivariateNormal(
            loc=jnp.zeros(action_dim),
            covariance=jnp.eye(action_dim),
        )
        # Build the conditional flow using rational quadratic splines
        base_flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

        # Sigmoid: R -> (0, 1); then affine: (0, 1) -> (low, high)
        loc = action_low
        scale = action_high - action_low
        squash = Chain([Sigmoid(shape=(action_dim,)), Affine(loc=loc, scale=scale)])

        self.flow = Transformed(base_flow, squash)

    def sample_action(
        self,
        key: jtp.Key[jtp.Array, ""],  # noqa: F722
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Samples a multi-modal action strategy given the history context."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.sample(key, condition=context)

    def log_prob(
        self,
        action: jax.Array,
        prev_state: jax.Array,
        curr_state: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood log p(a | s_prev, s_curr) for optimization."""
        context = jnp.concatenate([prev_state, curr_state], axis=-1)
        return self.flow.log_prob(action, condition=context)
