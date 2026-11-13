import equinox as eqx
import gymnasium as gym
import jax
import paramax

from jax_mc_pilco.model_learning.flow_model import FlowDynamics
from jax_mc_pilco.simulators.collect_data import collect_experience
from jax_mc_pilco.training.learning import rollout_training_loop, world_training_loop
from jax_mc_pilco.training.loss import batched_rollout_loss, imagine_trajectory_with_true_actions

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

problem: str = "InvertedDoublePendulum-v5"
env = gym.make(problem)

key = jax.random.key(seed=4)
key, subkey = jax.random.split(key)
_, max_episode_len, full_states, actions, full_next_states, rewards = collect_experience(env, 5_000, subkey, None, use_sobol = True)
print(f"Environment data collection complete. Sampled {full_states.shape[0]} points.")

states = full_states[:,:,:-1]
next_states = full_next_states[:,:,:-1]

init_states = states[:,0,:]

key_subkey = jax.random.split(key)
world_model_baseline = FlowDynamics(
    key=subkey,
    state_dim=states.shape[-1],
    action_dim=actions.shape[-1],
    state_low=states.min(axis=tuple(range(states.ndim - 1))),
    state_high=states.max(axis=tuple(range(states.ndim - 1))),
)
params, static = eqx.partition(
    world_model_baseline,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
key, subkey = jax.random.split(key)
print(batched_rollout_loss(params, static, subkey, init_states, next_states, actions,))

key, subkey = jax.random.split(key)
dynamics, d_losses = world_training_loop(subkey, states, next_states, actions, world_model_baseline, max_patience=50, max_epochs=2500)

print("Dynamics model fit complete. Final validation loss:", d_losses["val"][-1])


# plt.plot(d_losses["train"], color='blue')
# plt.plot(d_losses["val"],color="goldenrod")
# plt.show()

#eqx.tree_serialise_leaves("fitted_world_model.eqx", dynamics)

#fit_dynamics = eqx.tree_deserialise_leaves("fitted_world_model.eqx", world_model_baseline)


subkeys = jax.random.split(subkey, 1000)
idx = 0
pred_next_states = eqx.filter_vmap(imagine_trajectory_with_true_actions, in_axes=(None, 0, None, None, None))(
    dynamics, subkeys, init_states[idx], next_states[idx], actions[idx]
)

# for d in range(pred_next_states.shape[-1]):
#     for i in range(1000):
#         plt.plot(pred_next_states[i,:,d], color='orange', alpha = 0.05);
#     plt.plot(pred_next_states[:,:,d].mean(axis=0), color='firebrick', label='Pred');
#     plt.plot(next_states[idx,:,d], color='dodgerblue', label='True')
#     plt.legend()
#     plt.show()


# What is the current rollout loss?
params, static = eqx.partition(
    dynamics,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
key_subkey = jax.random.split(key)
print(batched_rollout_loss(params, static, subkey, init_states, next_states, actions,))


# In[21]:


# Better but what if we train this loss specifically?
key, subkey = jax.random.split(key)
new_dynamics, losses = rollout_training_loop(subkey, states, next_states, actions, dynamics)


# In[22]:


# What is the current rollout loss?
params, static = eqx.partition(
    new_dynamics,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
key_subkey = jax.random.split(key)
print(batched_rollout_loss(params, static, subkey, init_states, next_states, actions,))


subkeys = jax.random.split(subkey, 1000)
idx = 0
pred_next_states = eqx.filter_vmap(imagine_trajectory_with_true_actions, in_axes=(None, 0, None, None, None))(
    dynamics, subkeys, init_states[idx], next_states[idx], actions[idx]
)

for d in range(pred_next_states.shape[-1]):
    for i in range(1000):
        plt.plot(pred_next_states[i,:,d], color='orange', alpha = 0.05);
    plt.plot(pred_next_states[:,:,d].mean(axis=0), color='firebrick', label='Pred');
    plt.plot(next_states[idx,:,d], color='dodgerblue', label='True')
    plt.legend()
    plt.show()

