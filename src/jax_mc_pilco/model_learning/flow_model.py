### Equinox Module that acts as a world model given actions."""
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import paramax
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.flows import masked_autoregressive_flow

from jax_mc_pilco.model_learning.modules import Decoder, Encoder

EPSILON = 1e-5


class ConditionalMVN(AbstractDistribution):
    """Multivariate normal base distribution whose loc and covariance
    (via its Cholesky factor) are produced from the conditioning variable.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    tril_rows: jtp.Int[jtp.Array, " n_offdiag"]
    tril_cols: jtp.Int[jtp.Array, " n_offdiag"]

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        dim: int,
        cond_dim: int,
        width_size: int = 32,
        depth: int = 2,
    ) -> None:
        self.shape = (dim,)
        self.cond_shape: tuple[int, ...] = (cond_dim,)  # type: ignore
        self.dim = dim

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
        self.tril_rows = rows
        self.tril_cols = cols

    def _loc_and_chol(
        self,
        condition: jtp.Float[jtp.Array, " cond_dim"] | None,
    ) -> tuple[jtp.Float[jtp.Array, " dim"], jtp.Float[jtp.Array, "dim dim"]]:

        out: jtp.Float[jtp.Array, " out_size"] = self.mlp(typing.cast(jtp.Array, condition))

        # Static split points (self.dim is a static Python int), so this
        # is ordinary shape-level splitting, not a dynamic gather.
        loc, diag_raw, off_diag = jnp.split(out, (self.dim, 2 * self.dim))

        diag: jtp.Float[jtp.Array, " dim"] = jax.nn.softplus(diag_raw) + 1e-4

        L: jtp.Float[jtp.Array, "dim dim"] = jnp.zeros((self.dim, self.dim))
        L = L.at[self.tril_rows, self.tril_cols].set(off_diag)
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
        return -0.5 * quad - log_det - 0.5 * self.dim * jnp.log(2 * jnp.pi)

    def _sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        condition: jtp.Float[jtp.Array, " cond_dim"] | None = None,
    ) -> jtp.Float[jtp.Array, " dim"]:
        loc, L = self._loc_and_chol(condition)
        z = jax.random.normal(key, (self.dim,))
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
    latent_state_dim: int
    decoder: Decoder
    encoder: Encoder

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        latent_state_dim: int,
        action_dim: int,
        state_low: jax.Array,
        state_high: jax.Array,
        hidden_dim: int = 128,
        flow_layers: int = 4,
        in_channels: int = 3,
        num_spatial_layers: int = 4,
        *,
        state_mean: jax.Array | None = None,
        state_std: jax.Array | None = None,
        base_flow: Transformed | None = None,
    ) -> None:
        # Context consists of current normalized state and taken action
        # The encoder returns arrays of size (latent_state_dim, 4, 4)
        self.latent_state_dim = 16 * latent_state_dim
        cond_dim = self.latent_state_dim + action_dim

        self.state_low = paramax.non_trainable(
            jnp.broadcast_to(state_low, (latent_state_dim,))
        )
        self.state_high = paramax.non_trainable(
            jnp.broadcast_to(state_high, (latent_state_dim,))
        )
        self.latent_state_dim = latent_state_dim

        if state_mean is None:
            state_mean = jnp.zeros((latent_state_dim,))
        if state_std is None:
            state_std = jnp.ones((latent_state_dim,))

        self.state_mean = paramax.non_trainable(state_mean)
        self.state_std = paramax.non_trainable(
            jnp.where(state_std < 1e-6, 1.0, state_std)
        )

        memory_key, base_key, norm_flow_key, encoder_key, decoder_key = jax.random.split(key, 5)


        self.memory = eqx.nn.GRUCell(
            input_size=cond_dim, hidden_size=hidden_dim, key=memory_key
        )

        if base_flow is None:
            base_flow = masked_autoregressive_flow(
                key=norm_flow_key,
                base_dist=ConditionalMVN(base_key, dim=self.latent_state_dim, cond_dim=hidden_dim),
                cond_dim=hidden_dim,
                transformer=RationalQuadraticSpline(
                    knots=8,
                    interval=10.0,
                ),
                nn_width=256,
                nn_depth=2,
                flow_layers=flow_layers,
            )
        self.flow = base_flow

        self.decoder = Decoder(
            out_channels=in_channels,
            hid_channels=latent_state_dim,
            num_spatial_layers=num_spatial_layers,
            key=decoder_key,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            hid_channels=latent_state_dim,
            num_spatial_layers=num_spatial_layers,
            key=encoder_key,
        )

    def normalize_state(self, state: jax.Array) -> jax.Array:
        ret_state: jax.Array =  (state - paramax.unwrap(self.state_mean)) / paramax.unwrap(
            self.state_std
        )
        return ret_state

    def unnormalize_state(self, norm_state: jax.Array) -> jax.Array:
        ret_state: jax.Array = norm_state * paramax.unwrap(self.state_std) + paramax.unwrap(
            self.state_mean
        )
        return ret_state

    def step_hidden(
        self,
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        """Advance recurrent hidden state using normalized state and action."""
        norm_state = self.normalize_state(prev_state)
        context = jnp.concatenate([norm_state, prev_action])
        ret_hidden: jax.Array = self.memory(context, prev_hidden)
        return ret_hidden

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition delta and adds it to s_t."""
        norm_state = self.normalize_state(prev_state)
        next_hidden = self.memory(
            jnp.concatenate([norm_state, prev_action]), prev_hidden
        )

        # No stop_gradient here: gradients flow through the world model's inputs
        # back to `prev_action` (and thus `ActionFlow`), while world_model
        # weights remain frozen because they are not passed to the optimizer.
        norm_state_delta = self.flow.sample(key, (), next_hidden)

        next_state = jnp.clip(
            self.unnormalize_state(norm_state + norm_state_delta),
            paramax.unwrap(self.state_low),
            paramax.unwrap(self.state_high),
        )
        return next_state, next_hidden

    def sample(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> jax.Array:
        next_state, _ = self.predict_next_state_and_hidden(
            key, prev_state, prev_action, prev_hidden
        )
        return next_state

    def log_prob(
        self,
        prev_state: jax.Array,
        next_state: jax.Array,
        hidden: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the normalized state delta given hidden."""
        norm_state_delta = self.normalize_state(next_state) - self.normalize_state(
            prev_state
        )
        return self.flow.log_prob(norm_state_delta, hidden)

    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array,
        hidden: jax.Array,
        key: jtp.Key[jtp.Array, ""],
    ) -> tuple[jax.Array, jax.Array]:
        latent_state = self.encoder(obs)
        next_state, next_hidden = self.predict_next_state_and_hidden(key, latent_state, action, hidden)
        out = self.decoder(next_state)

        return out, next_hidden
