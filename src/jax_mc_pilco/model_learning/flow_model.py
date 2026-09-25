### Equinox Module that acts as a world model given actions."""
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.flows import coupling_flow


class DreamerTitansWrapper(eqx.Module):
    """
    A drop-in replacement for the GRU cell block in DreamerV3.
    Supports dual-mode execution:
      - Parallel (associative scan) for sequence training chunks.
      - Sequential (step-by-step) for active environment collection.
    """

    w_key: eqx.nn.Linear
    w_val: eqx.nn.Linear
    w_gating: eqx.nn.Linear
    w_out: eqx.nn.Linear
    hidden_dim: int
    mem_dim: int

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        mem_dim: int = 32,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        keys = jax.random.split(key, 4)
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim

        # Inputs are strictly the concatenated (s_t, a_t)
        input_dim = state_dim + action_dim

        w_key = eqx.nn.Linear(input_dim, mem_dim, use_bias=False, key=keys[0])
        w_val = eqx.nn.Linear(input_dim, mem_dim, use_bias=False, key=keys[1])

        # Scale down values so we do not get explosions
        self.w_key = eqx.tree_at(lambda layer: layer.weight, w_key, 0.01 * w_key.weight)
        self.w_val = eqx.tree_at(lambda layer: layer.weight, w_val, 0.01 * w_val.weight)

        self.w_gating = eqx.nn.Linear(input_dim, 1, use_bias=True, key=keys[2])
        self.w_out = eqx.nn.Linear(mem_dim, hidden_dim, use_bias=True, key=keys[3])

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        fast_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        SEQUENTIAL MODE: Run live step-by-step during environment execution.
        state: (state_dim,)
        action: (action_dim,)

        hidden_and_fast_weights: tuple of (h_prev, W_prev)
        """

        W_prev = fast_weights

        context = jnp.concatenate([state, action], axis=-1)

        # Project to memory spaces
        k = self.w_key(context)
        v = self.w_val(context)
        eta = jax.nn.sigmoid(self.w_gating(context))

        # Update fast-weights with continuous decay
        k_norm = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-5)
        alpha = 1.0 - 0.1 * eta
        mem_retrieved = jnp.dot(W_prev, k_norm)
        delta = v - mem_retrieved
        W_next = (alpha * W_prev) + eta * jnp.outer(delta, k_norm)

        # Generate next hidden feature
        h_next = jax.nn.silu(self.w_out(mem_retrieved))

        return h_next, W_next

    def scan_sequence(
        self,
        seq_states: jax.Array,
        seq_actions: jax.Array,
        initial_state: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        PARALLEL MODE: Process an entire sequence window instantly using associative scans.
        seq_states: (Time, state_dim)

        seq_actions: (Time, action_dim)

        initial_state: tuple of (h_0, W_0) or None
        """

        if initial_state is None:
            W_0 = jnp.zeros((self.mem_dim, self.mem_dim))
        else:
            _, W_0 = initial_state

        # Concatenate sequential pairs [state, actions] across the whole time axis

        seq_inputs = jnp.concatenate([seq_states, seq_actions], axis=-1)  # (Time, input_dim)

        # Broad parallel batch projection across time
        keys = jax.vmap(self.w_key)(seq_inputs)
        vals = jax.vmap(self.w_val)(seq_inputs)
        etas = jax.nn.sigmoid(jax.vmap(self.w_gating)(seq_inputs))

        # Formulate parallel scan operators (Linear Matrix Recurrence mapping)

        def make_operators(
            k: jax.Array,
            v: jax.Array,
            e: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            """Normalize the key to prevent large outer products."""
            k_norm = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-5)
            alpha = 1.0 - 0.1 * e
            A = alpha * jnp.eye(self.mem_dim) - e * jnp.outer(k_norm, k_norm)
            B = e * jnp.outer(v, k_norm)
            return A, B

        As, Bs = jax.vmap(make_operators)(keys, vals, etas)
        Bs = Bs.at[0].add(jnp.matmul(W_0, As[0]))  # correctly fold in initial state: W_1 = W_0 @ A_1 + B_1

        # Associative state contraction tree

        def associative_op(
            state_left: tuple[jax.Array, jax.Array],
            state_right: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            A_l, B_l = state_left
            A_r, B_r = state_right
            return jnp.matmul(A_l, A_r), jnp.matmul(B_l, A_r) + B_r

        _, W_all = jax.lax.associative_scan(associative_op, (As, Bs), axis=0)
        # Propagate weights to step sequentially

        W_prev_all = jnp.concatenate([jnp.expand_dims(W_0, axis=0), W_all[:-1]], axis=0)

        # Extract mapped memories and build vector hidden representations
        mem_retrieved = jax.vmap(lambda W, k: jnp.dot(W, k))(W_prev_all, keys)
        h_all = jax.nn.silu(jax.vmap(self.w_out)(mem_retrieved))

        # Returns full trajectory predictions for loss calculation, and final terminal carry-over state

        return h_all, (h_all[-1], W_all[-1])

    def init_state(
        self,
        batch_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Initial state zero allocation utility."""
        if batch_size is None:
            return (jnp.zeros((self.hidden_dim,)), jnp.zeros((self.mem_dim, self.mem_dim)))
        else:
            return (jnp.zeros((batch_size, self.hidden_dim)), jnp.zeros((batch_size, self.mem_dim, self.mem_dim)))


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
        self.cond_shape = (cond_dim,)
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
    memory: DreamerTitansWrapper
    state_high: jax.Array
    state_low: jax.Array

    def __init__(
        self,
        key: jtp.Key[jtp.Array, ""],
        state_dim: int,
        action_dim: int,
        state_low: jax.Array,
        state_high: jax.Array,
        deter_dim: int = 64,
        flow_layers: int = 4,
        *,
        flow: Transformed | None = None,
    ):

        self.state_low = jnp.broadcast_to(state_low, (state_dim,))
        self.state_high = jnp.broadcast_to(state_high, (state_dim,))

        self.memory = DreamerTitansWrapper(state_dim=state_dim, action_dim=action_dim, hidden_dim=deter_dim, key=key)
        base_key, flow_key = jax.random.split(key)

        if flow is None:
            # Define a base distribution matching the state delta dimension
            base_dist = ConditionalMVN(base_key, dim=state_dim, cond_dim=deter_dim)
            flow = coupling_flow(
                key=flow_key,
                base_dist=base_dist,
                cond_dim=deter_dim,
                nn_width=256,
                nn_depth=2,
                flow_layers=flow_layers,
            )
        self.flow = flow

        # TODO: utilize a chained sigmoid and affine to constrain the flow
        # (somehow) so deltas are within bounds.

    def predict_next_state_and_hidden(
        self,
        key: jtp.Key[jtp.Array, ""],
        prev_state: jax.Array,
        prev_action: jax.Array,
        prev_fast_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Samples a structural residual transition delta and adds it to s_t."""
        # Update hidden state and fast weights
        next_hidden, next_fast_weights = self.memory.step(prev_state, prev_action, prev_fast_weights)
        delta_s = self.flow.sample(key, condition=next_hidden)
        return jnp.clip(prev_state + delta_s, self.state_low, self.state_high), next_fast_weights

    def log_prob(
        self,
        delta_s: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        """Calculates exact log-likelihood of the state delta given the context."""
        return self.flow.log_prob(delta_s, condition=context)

    def predict_reward(self, state: jax.Array) -> jax.Array:
        # Returns a scalar value for the given latent state representation
        x = state[0]
        y = jnp.arctan2(state[2], state[4])
        v1 = state[5]
        v2 = state[6]

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        # Condition: y <= 1 -> invert it so True means "alive" (y > 1)
        is_alive = y > 1.0

        # jax.lax.cond(pred, true_fn, false_fn, *operands)
        alive_bonus = jax.lax.cond(
            is_alive,
            lambda _: 10.0,  # If True (y > 1), return 10.0
            lambda _: 0.0,  # If False (y <= 1), return 0.0
            operand=None,
        )
        return jnp.array(alive_bonus - dist_penalty - vel_penalty)
