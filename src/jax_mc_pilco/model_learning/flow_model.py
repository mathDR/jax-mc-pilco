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
        self.cond_shape: tuple[int, ...] = (cond_dim,)
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


class mLSTMCell(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    i_gate: eqx.nn.Linear
    f_gate: eqx.nn.Linear
    o_gate: eqx.nn.Linear
    out_norm: eqx.nn.GroupNorm
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, num_heads: int, head_dim: int, *, key: jtp.Key[jtp.Array, ""]):
        d_model = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        keys = jax.random.split(key, 6)
        self.q_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[0])
        self.k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[1])
        self.v_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[2])

        # Input and forget gates produce one scalar gate per head
        self.i_gate = eqx.nn.Linear(d_model, num_heads, key=keys[3])
        self.f_gate = eqx.nn.Linear(d_model, num_heads, key=keys[4])
        self.o_gate = eqx.nn.Linear(d_model, d_model, key=keys[5])

        # Per-head normalization as specified in the xLSTM paper
        self.out_norm = eqx.nn.GroupNorm(groups=num_heads, channels=d_model)

    def __call__(self, x: jtp.Float[jtp.Array, "seq d_model"]) -> jtp.Float[jtp.Array, "seq d_model"]:
        """Parallel forward pass over a single sequence (use jax.vmap for batches)."""
        T, D = x.shape
        H, dh = self.num_heads, self.head_dim

        # 1. Project queries, keys, values -> shape: (H, T, dh)
        q = jax.vmap(self.q_proj)(x).reshape(T, H, dh).transpose(1, 0, 2)
        k = jax.vmap(self.k_proj)(x).reshape(T, H, dh).transpose(1, 0, 2) / jnp.sqrt(dh)
        v = jax.vmap(self.v_proj)(x).reshape(T, H, dh).transpose(1, 0, 2)

        # 2. Pre-activation gates -> shape: (H, T)
        i_tilde = jax.vmap(self.i_gate)(x).T
        f_tilde = jax.vmap(self.f_gate)(x).T
        o = jax.nn.sigmoid(jax.vmap(self.o_gate)(x))  # (T, D)

        # 3. Log-space stabilization for exponential gates (prevents exp overflow)
        log_f = jax.nn.log_sigmoid(f_tilde)  # (H, T)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # Cumulative log forget gates: sum_{k=j+1}^t log(f_k)
        cum_log_f = jnp.cumsum(log_f, axis=-1)  # (H, T)
        log_f_matrix = cum_log_f[:, :, None] - cum_log_f[:, None, :]  # (H, T, T)
        log_f_matrix = jnp.where(causal_mask[None, :, :], log_f_matrix, -jnp.inf)

        # Combine with input gate: log(D_{t,j}) = log(i_j) + sum_{k=j+1}^t log(f_k)
        log_D = log_f_matrix + i_tilde[:, None, :]  # (H, T, T)

        # Stabilize by subtracting max log-weight per timestep t
        m = jnp.max(log_D, axis=-1, keepdims=True)  # (H, T, 1)
        D_tilde = jnp.exp(log_D - m)  # (H, T, T)

        # 4. Parallel mLSTM attention-style readout
        qk = jnp.einsum("htd,hjd->htj", q, k)  # (H, T, T)
        C = qk * D_tilde  # (H, T, T)

        # Normalizer state n_t with max-stabilized threshold
        n = jnp.abs(jnp.sum(C, axis=-1, keepdims=True))  # (H, T, 1)
        denom = jnp.maximum(n, jnp.exp(-m)) + 1e-6

        h = jnp.einsum("htj,hjd->htd", C / denom, v)  # (H, T, dh)

        # 5. Reshape, apply per-head GroupNorm, and multiply by output gate
        h = h.transpose(1, 0, 2).reshape(T, D)
        h_norm = jax.vmap(self.out_norm)(h)
        return o * h_norm

    def step(
        self,
        x_t: jtp.Float[jtp.Array, " d_model"],
        state: tuple[jtp.Float[jtp.Array, " H dh dh"], jtp.Float[jtp.Array, " H dh"], jtp.Float[jtp.Array, " H"]],
    ) -> tuple[jtp.Array, tuple[jtp.Array,jtp.Array,jtp.Array]]:
        """O(1) recurrent step for autoregressive inference."""
        C_prev, n_prev, m_prev = state
        H, dh = self.num_heads, self.head_dim

        q_t = self.q_proj(x_t).reshape(H, dh)
        k_t = self.k_proj(x_t).reshape(H, dh) / jnp.sqrt(dh)
        v_t = self.v_proj(x_t).reshape(H, dh)

        i_tilde = self.i_gate(x_t)  # (H,)
        f_tilde = self.f_gate(x_t)  # (H,)
        o_t = jax.nn.sigmoid(self.o_gate(x_t))  # (D,)

        # Log-domain stabilizer update
        log_f = jax.nn.log_sigmoid(f_tilde)
        m_t = jnp.maximum(log_f + m_prev, i_tilde)
        f_t = jnp.exp(log_f + m_prev - m_t)
        i_t = jnp.exp(i_tilde - m_t)

        # Update matrix memory C_t and normalizer vector n_t
        C_t = f_t[:, None, None] * C_prev + i_t[:, None, None] * jnp.einsum("hd,he->hde", v_t, k_t)
        n_t = f_t[:, None] * n_prev + i_t[:, None] * k_t

        # Readout
        num = jnp.einsum("hde,he->hd", C_t, q_t)
        denom = jnp.maximum(jnp.abs(jnp.sum(n_t * q_t, axis=-1, keepdims=True)), jnp.exp(-m_t)[:, None]) + 1e-6
        h_t = (num / denom).reshape(-1)

        out = o_t * self.out_norm(h_t)
        return out, (C_t, n_t, m_t)


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
