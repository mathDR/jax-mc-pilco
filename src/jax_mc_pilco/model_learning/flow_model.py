### Equinox Module that acts as a world model given actions."""
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
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

    latent_state_dim: int
    decoder: Decoder
    flow: Transformed
    memory: eqx.nn.GRUCell
    encoder: Encoder

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        latent_state_dim: int,
        action_dim: int,
        deter_dim: int = 64,
        flow_layers: int = 2,
        in_channels=3,
        num_spatial_layers: int = 4,
    ):
        # Context consists of current state and taken action
        # The encoder returns arrays of size (latent_state_dim, 4, 4)
        self.latent_state_dim = 16 * latent_state_dim
        cond_dim = self.latent_state_dim + action_dim

        key, base_key, flow_key, encoder_key, decoder_key = jax.random.split(key, 5)

        self.memory = eqx.nn.GRUCell(input_size=cond_dim, hidden_size=deter_dim, key=key)

        self.flow = masked_autoregressive_flow(
            key=flow_key,
            base_dist=ConditionalMVN(base_key, dim=self.latent_state_dim, cond_dim=deter_dim),
            cond_dim=deter_dim,
            transformer=RationalQuadraticSpline(
                knots=8,
                interval=10.0,
            ),
            nn_width=256,
            nn_depth=2,
            flow_layers=flow_layers,
        )

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

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition state."""
        context = jnp.concatenate([prev_state, prev_action], axis=-1)
        # Update hidden state
        next_hidden = self.memory(context, prev_hidden)
        next_state_delta = self.flow.sample(key, condition=next_hidden)
        return next_state_delta + prev_state, next_hidden

    def log_prob(
        self,
        state_delta: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state given the context."""
        return self.flow.log_prob(state_delta, condition=context)

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
