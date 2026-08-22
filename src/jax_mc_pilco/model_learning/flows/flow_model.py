import equinox as eqx
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.flows import coupling_flow

class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model.
    Maps noise z ~ N(0, I) -> Delta State given the context [s_t, a_t]
    """
    flow: Transformed
    state_dim: int
    action_dim: int

    def __init__(self, key: jtp.Key[jtp.Array,""], state_dim: int, action_dim: int, flow_layers: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Context consists of current state and taken action
        cond_dim = state_dim + action_dim
        
        # Define a base distribution matching the state delta dimension
        base_dist = MultivariateNormal(loc=jnp.zeros(state_dim,, dtype=float), covariance=jnp.eye(state_dim, dtype=float),)
        
        self.flow = coupling_flow(
            key=key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )
    base_dist,
    transformer=None,
    cond_dim=None,
    flow_layers=8,
    nn_width=50,
    nn_depth=1,
    nn_activation=<jax._src.custom_derivatives.custom_jvp object>,
    invert=True,
    def predict_next_state(self, key: jtp.Key[jtp.Array,""], s_curr: jax.Array, action: jax.Array) -> jax.Array:
        """Samples a structural residual transition delta and adds it to s_t."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = self.flow.sample(key, condition=context)
        return s_curr + delta_s

    def log_prob(self, s_next: jnp.ndarray, s_curr: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Calculates exact log-likelihood of the state delta given the context."""
        context = jnp.concatenate([s_curr, action], axis=-1)
        delta_s = s_next - s_curr
        return self.flow.log_prob(delta_s, condition=context)