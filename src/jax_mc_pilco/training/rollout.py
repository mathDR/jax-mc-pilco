"""Multi-step imagination rollout."""
import jax
import jaxtyping as jtp
from typeguard import typechecked as typechecker

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.policy_learning.action_flows import FlowActor


@typechecker
def functional_rollout(
    world_model: FlowDynamics,
    action_model: FlowActor,
    init_state: jtp.Float[jtp.Array, " state_dim"],
    init_hidden: jtp.Float[jtp.Array, " hidden_dim"],
    horizon: int,
    key: jtp.Key[jtp.Array, ""],
) -> tuple[jtp.Float[jtp.Array, "horizon state_dim"], jtp.Float[jtp.Array, "horizon action_dim"]]:
    """Imagines a trajectory using independent world and action model PyTrees.

    Args:
        world_model: FlowRSSM instance (Equinox module PyTree).
        action_model: eqx.nn.MLP instance (Equinox module PyTree).
        init_state: Starting raw state array.
        init_hidden: Starting GRU hidden state array.
        horizon: Number of steps to imagine forward.
        key: PRNGKey for the trajectory.

    Returns:
        imagined_states: Logged states over the horizon steps.
        imagined_actions: Logged actions over the horizon steps.
    """
    # Generate an array of unique random keys for every horizon step
    keys: jtp.Key[jtp.Array, ""] = jax.random.split(key, horizon)

    # Define the single-step pure transition function for jax.lax.scan
    def scan_fn(
        carry: tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " hidden_dim"]],
        step_key: jtp.Key[jtp.Array, ""], action_key: jtp.Key[jtp.Array, ""],
    ) -> tuple[
        tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " hidden_dim"]], 
        tuple[jtp.Float[jtp.Array, " state_dim"], jtp.Float[jtp.Array, " action_dim"]]
    ]:
        current_state, current_hidden = carry

        # Action model decides the action based on state and hidden context
        action = action_model.sample_action(action_key, current_state)

        # World model predicts the next state using its internal logic
        next_state, next_hidden = world_model.predict_next_state_and_hidden(
            current_state, action, current_hidden, step_key
        )

        next_carry = (next_state, next_hidden)
        step_outputs = (next_state, action)

        return next_carry, step_outputs

    # 3. Initial loop carry state
    init_carry = (init_state, init_hidden)

    # 4. Run the fast scanned loop across the horizon steps
    _, (imagined_states, imagined_actions) = jax.lax.scan(
        scan_fn, init_carry, keys, length=horizon
    )

    return imagined_states, imagined_actions
