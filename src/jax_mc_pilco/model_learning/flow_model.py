### Equinox Module that acts as a world model given actions."""
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import paramax
from beartype import beartype
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, MultivariateNormal, Transformed
from flowjax.flows import masked_autoregressive_flow

from jax_mc_pilco.model_learning.codecs import Decoder, Encoder

EPSILON = 1e-5


class ConditionalMVN(AbstractDistribution):
    """Multivariate normal base distribution whose loc and covariance
    (via its Cholesky factor) are produced from the conditioning variable.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    max_loc: float = eqx.field(static=True)
    max_scale: float = eqx.field(static=True)
    tril_rows: jtp.Int[jtp.Array, " n_offdiag"]
    tril_cols: jtp.Int[jtp.Array, " n_offdiag"]

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        dim: int,
        cond_dim: int,
        width_size: int = 32,
        depth: int = 2,
        max_loc: float = 4.0,
        max_scale: float = 2.0,
    ) -> None:
        self.shape = (dim,)
        self.cond_shape: tuple[int, ...] = (cond_dim,)
        self.dim = dim
        self.max_loc = max_loc
        self.max_scale = max_scale
        n_offdiag = dim * (dim - 1) // 2
        out_size = 2 * dim + n_offdiag  # loc, chol-diag, chol-off-diag
        self.mlp = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )
        rows, cols = jnp.tril_indices(dim, k=-1)
        self.tril_rows = paramax.non_trainable(rows)
        self.tril_cols = paramax.non_trainable(cols)

    def _loc_and_chol(
        self,
        condition: jtp.Float[jtp.Array, " cond_dim"] | None,
    ) -> tuple[jtp.Float[jtp.Array, " dim"], jtp.Float[jtp.Array, "dim dim"]]:
        out: jtp.Float[jtp.Array, " out_size"] = self.mlp(typing.cast(jtp.Array, condition))
        loc_raw, diag_raw, off_diag_raw = jnp.split(out, (self.dim, 2 * self.dim))
        # Bound loc and Cholesky factors so base samples stay within the spline's active [-8, 8] interval
        loc = self.max_loc * jnp.tanh(loc_raw / self.max_loc)
        diag = 1e-4 + (self.max_scale - 1e-4) * jax.nn.sigmoid(diag_raw)
        off_diag = self.max_scale * jnp.tanh(off_diag_raw / self.max_scale)
        rows = paramax.unwrap(self.tril_rows)
        cols = paramax.unwrap(self.tril_cols)
        L = jnp.zeros((self.dim, self.dim), dtype=out.dtype)
        L = L.at[rows, cols].set(off_diag)
        L = L + jnp.diag(diag)
        return loc, L

    def _log_prob(
        self,
        x: jtp.Float[jtp.Array, " dim"],
        condition: jtp.Float[jtp.Array, " cond_dim"] | None = None,
    ) -> jtp.Float[jtp.Array, ""]:
        loc, L = self._loc_and_chol(condition)
        y = jax.scipy.linalg.solve_triangular(L, x - loc, lower=True)
        log_det = jnp.sum(jnp.log(jnp.diag(L)))
        quad = jnp.sum(y**2)
        return -0.5 * quad - log_det - 0.5 * self.dim * jnp.log(2.0 * jnp.pi)

    def _sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        condition: jtp.Float[jtp.Array, " cond_dim"] | None = None,
    ) -> jtp.Float[jtp.Array, " dim"]:
        loc, L = self._loc_and_chol(condition)
        z = jax.random.normal(key, (self.dim,), dtype=L.dtype)
        return loc + L @ z


class FlowDynamics(eqx.Module):
    """
    Conditional Normalizing Flow dynamics model with GRUCell for recurrent memory.
    """

    flow: Transformed
    memory: eqx.nn.GRUCell
    state_high: jtp.Array
    state_low: jtp.Array
    state_mean: jtp.Array
    state_std: jtp.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        state_low: jax.Array,
        state_high: jax.Array,
        hidden_dim: int = 128,
        flow_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ):
        # Context consists of current normalized state and taken action
        cond_dim = state_dim + action_dim
        self.state_low = paramax.non_trainable(jnp.broadcast_to(state_low, (state_dim,)))
        self.state_high = paramax.non_trainable(jnp.broadcast_to(state_high, (state_dim,)))
        if state_mean is None:
            state_mean = jnp.zeros((state_dim,))
        if state_std is None:
            state_std = jnp.ones((state_dim,))
        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(jnp.where(state_std < 1e-6, 1.0, state_std))
        memory_key, norm_flow_key = jax.random.split(key, 2)
        self.memory = eqx.nn.GRUCell(input_size=cond_dim, hidden_size=hidden_dim, key=memory_key)
        if base_flow is None:
            base_flow = masked_autoregressive_flow(
                key=norm_flow_key,
                base_dist=MultivariateNormal(loc=jnp.zeros(state_dim), covariance=jnp.eye(state_dim)),
                cond_dim=hidden_dim,
                transformer=RationalQuadraticSpline(knots=12, interval=8, min_derivative=1e-2),
                flow_layers=flow_layers,
                invert=True,
                nn_depth=2,
            )
        self.flow = base_flow

    @jtp.jaxtyped(typechecker=beartype)
    def normalize_state(self, state: jax.Array) -> jax.Array:
        ret_array: jax.Array = (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(self.state_std)
        return ret_array

    @jtp.jaxtyped(typechecker=beartype)
    def unnormalize_state(self, norm_state: jax.Array) -> jax.Array:
        ret_array: jax.Array = norm_state * paramax.unwrap(self.state_std) + paramax.unwrap(self.state_mean)
        return ret_array

    @jtp.jaxtyped(typechecker=beartype)
    def step_hidden(
        self,
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        """Advance recurrent hidden state using normalized state and action."""
        norm_state = self.normalize_state(prev_state)
        context = jnp.concatenate([norm_state, prev_action])
        new_hidden: jax.Array = self.memory(context, prev_hidden)
        return new_hidden

    @jtp.jaxtyped(typechecker=beartype)
    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
        bound_margin: float = 0.1,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition delta and adds it to s_t."""
        next_hidden = self.step_hidden(prev_state, prev_action, prev_hidden)
        norm_state_delta = self.flow.sample(key, (), next_hidden)
        state_delta = norm_state_delta * paramax.unwrap(self.state_std)
        low = paramax.unwrap(self.state_low)
        high = paramax.unwrap(self.state_high)
        margin = bound_margin * (high - low)
        next_state = jnp.clip(
            prev_state + state_delta,
            low - margin,
            high + margin,
        )
        return next_state, next_hidden

    @jtp.jaxtyped(typechecker=beartype)
    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        next_state, _ = self.predict_next_state_and_hidden(key, prev_state, prev_action, prev_hidden)
        return next_state

    @jtp.jaxtyped(typechecker=beartype)
    def log_prob(
        self,
        prev_state: jax.Array,
        next_state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the normalized state delta given hidden."""
        norm_state_delta = self.normalize_state(next_state) - self.normalize_state(prev_state)
        return self.flow.log_prob(norm_state_delta, hidden)


class LatentFlowWorldModel(eqx.Module):
    """Pixel world model composing Encoder, Decoder, Posterior, Reward head,
    and your existing FlowDynamics module as the latent prior.
    """

    encoder: Encoder
    decoder: Decoder
    dynamics: FlowDynamics
    posterior_net: eqx.nn.Sequential
    reward_net: eqx.nn.Sequential
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        action_dim: int,
        in_channels: int = 3,
        img_size: int = 64,
        conv_channels: int = 64,
        num_spatial_layers: int = 4,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        flow_layers: int = 4,
        latent_bound: float = 10.0,
    ):
        self.latent_dim = latent_dim
        k_enc, k_dec, k_dyn, k_post1, k_post2, k_rew1, k_rew2 = jax.random.split(key, 7)
        spatial_out = img_size // (2**num_spatial_layers)
        embed_dim = conv_channels * spatial_out * spatial_out
        self.encoder = Encoder(
            in_channels=in_channels,
            hid_channels=conv_channels,
            num_spatial_layers=num_spatial_layers,
            key=k_enc,
        )
        self.decoder = Decoder(
            in_dim=latent_dim + hidden_dim,
            out_channels=in_channels,
            hid_channels=conv_channels,
            num_spatial_layers=num_spatial_layers,
            img_size=img_size,
            key=k_dec,
        )
        # Reuse FlowDynamics directly in latent space (mean=0, std=1 by default)
        self.dynamics = FlowDynamics(
            key=k_dyn,
            state_dim=latent_dim,
            action_dim=action_dim,
            state_low=jnp.full((latent_dim,), -latent_bound),
            state_high=jnp.full((latent_dim,), latent_bound),
            hidden_dim=hidden_dim,
            flow_layers=flow_layers,
        )
        # Posterior q(z_t | h_t, e_t) -> (mean, raw_std)
        self.posterior_net = eqx.nn.Sequential((
            eqx.nn.Linear(hidden_dim + embed_dim, 256, key=k_post1),
            eqx.nn.LayerNorm(256),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(256, 2 * latent_dim, key=k_post2),
        ))
        # Reward predictor r(z_t, h_t) -> scalar reward
        self.reward_net = eqx.nn.Sequential((
            eqx.nn.Linear(latent_dim + hidden_dim, 256, key=k_rew1),
            eqx.nn.LayerNorm(256),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(256, 1, key=k_rew2),
        ))

    @property
    def memory(self) -> eqx.nn.GRUCell:
        """Expose dynamics.memory so existing code reading world_model.memory works."""
        return self.dynamics.memory

    def normalize_state(self, z: jax.Array) -> jax.Array:
        return self.dynamics.normalize_state(z)

    def step_hidden(
        self,
        prev_z: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        return self.dynamics.step_hidden(prev_z, prev_action, prev_hidden)

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_z: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
        bound_margin: float = 0.1,
    ) -> tuple[jax.Array, jax.Array]:
        return self.dynamics.predict_next_state_and_hidden(
            key, prev_z, prev_action, prev_hidden, bound_margin=bound_margin
        )

    def prior_log_prob(
        self,
        prev_z: jax.Array,
        next_z: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        return self.dynamics.log_prob(prev_z, next_z, hidden)

    def posterior_sample_and_log_prob(
        self,
        key: jtp.Key[jtp.Array, ""],
        obs_embed: jax.Array,
        hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        stats = self.posterior_net(jnp.concatenate([hidden, obs_embed], axis=-1))
        mean, raw_std = jnp.split(stats, 2, axis=-1)
        std = jax.nn.softplus(raw_std) + 1e-3
        eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        z = mean + std * eps
        log_prob = -0.5 * jnp.sum(((z - mean) / std) ** 2 + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi))
        return z, log_prob

    def decode_obs(self, z: jax.Array, hidden: jax.Array) -> jax.Array:
        return self.decoder(jnp.concatenate([z, hidden], axis=-1))

    def predict_reward(self, z: jax.Array, hidden: jax.Array) -> jax.Array:
        return jnp.squeeze(self.reward_net(jnp.concatenate([z, hidden], axis=-1)), axis=-1)
