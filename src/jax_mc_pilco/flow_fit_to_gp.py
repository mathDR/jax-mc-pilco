"""Fitting a flow to a Gaussian Process."""

from collections.abc import Callable

import equinox as eqx
import gpjax
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    step,
)
from tqdm import tqdm


def sample_between_bounds(key: jtp.Key[jtp.Array, ""], bounds: jax.Array, num_samples: int) -> jax.Array:
    # bounds shape is (2, D)
    lb = bounds[0]  # Shape: (D,)
    ub = bounds[1]  # Shape: (D,)

    # Get dimension D
    dim = bounds.shape[1]

    # Sample uniformly from 0 to 1 with shape (num_samples, D)
    u = jax.random.uniform(key, shape=(num_samples, dim))

    # Broadcast and scale to the bounds
    return lb + (ub - lb) * u


def get_batch(
    key: jtp.Key[jtp.Array, ""],
    gp_model: gpjax.gps.ConjugatePosterior,
    train_data: gpjax.Dataset,
    bounds: jax.Array,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    key, subkey1, subkey2 = jax.random.split(key, 3)
    batch_X = sample_between_bounds(subkey1, bounds, batch_size)
    ## Should check to see if we should use sample_approx over the posterior
    # if true sampling is speed prohibitive
    post_dist = gp_model.predict(batch_X, train_data=train_data)
    batch_Y = jax.random.multivariate_normal(
        key=subkey2,
        mean=post_dist.mean,
        cov=post_dist.covariance_matrix,
    ).reshape(batch_size, train_data.y.shape[1])
    batch = (batch_Y, batch_X)  # Note the ORDER!
    return batch


def fit_to_gp(
    key: jtp.Key[jtp.Array, ""],
    dist: jtp.PyTree,
    gp_model: gpjax.gps.ConjugatePosterior,
    train_data: gpjax.Dataset,
    *,
    loss_fn: Callable | None = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    return_best: bool = True,
    show_progress: bool = True,
):
    r"""Train a PyTree (e.g. a distribution) to samples from a Gaussian Process.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        gp_model: The Gaussian Process Posterior
        train_data: The training data used to fit the Gaussian Process.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        loss_fn: Loss function. The signature should be of the form
            ``(params, static, *arrays, key)``. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained distribution and the losses.
    """

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    data_bounds = jnp.stack((train_data.X.min(axis=0), train_data.X.max(axis=0)))  # type: ignore  # noqa: PGH003

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jax.random.split(key)
    losses = {"train": [], "val": []}
    num_train_batches = 100
    num_val_batches = 10

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Train epoch
        batch_losses = []
        for _ in range(num_train_batches):
            key, subkey = jax.random.split(key)
            batch = get_batch(subkey, gp_model, train_data, data_bounds, batch_size)

            params, opt_state, loss_idx = step(
                params,
                static,
                *batch,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=subkey,
            )
            batch_losses.append(loss_idx)
        losses["train"].append(sum(batch_losses) / len(batch_losses))

        # Val epoch
        batch_losses = []
        for _ in range(num_val_batches):
            key, subkey = jax.random.split(key)
            batch = get_batch(subkey, gp_model, train_data, data_bounds, batch_size)
            loss_idx = eqx.filter_jit(loss_fn)(params, static, *batch, key=subkey)
            batch_losses.append(loss_idx)
        losses["val"].append(sum(batch_losses) / len(batch_losses))

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses
