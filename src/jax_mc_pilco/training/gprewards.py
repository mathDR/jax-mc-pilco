"""Bespoke GPJax Spectral Mixture Kernel with GPyTorch intialization."""

import typing

import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import jaxtyping as jtp
import optax
import paramax
from gpjax.kernels import AbstractKernel
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import PositiveReal
from gpjax.variational_families import CollapsedVariationalGaussian

jax.config.update("jax_enable_x64", True)


def initialize_ard_sm_from_data(
    key: jtp.Key[jtp.Array, ""],
    train_x: jtp.Float[jtp.Array, "N D"],
    train_y: jtp.Float[jtp.Array, " N"],
    num_mixtures: int,
) -> tuple[jtp.Float[jtp.Array, " Q"], jtp.Float[jtp.Array, "Q D"], jtp.Float[jtp.Array, "Q D"]]:
    """
    Multidimensional adaptation of data-driven SM initialization strategy.
    """

    key_freq, key_scale = jax.random.split(key)
    n_samples: int = train_x.shape[0]
    num_dimensions: int = train_x.shape[1]

    train_y_flat = jnp.ravel(train_y)

    # --- 1. Weights [Shape: (Q,)] ---
    y_std: jtp.Float[jtp.Array, ""] = jnp.std(train_y_flat)
    weights_init: jtp.Float[jtp.Array, " Q"] = jnp.ones((num_mixtures,)) * (y_std / num_mixtures)

    # --- 2. Frequencies (Spectral Means) [Shape: (Q, D)] ---
    emp_spect: jtp.Float[jtp.Array, " N"] = jnp.abs(jnp.fft.fft(train_y_flat)) ** 2 / n_samples
    half_n: int = n_samples // 2

    freqs: jtp.Float[jtp.Array, " half_n"] = jnp.fft.fftfreq(n_samples)[: half_n + 1]
    emp_spect = emp_spect[: half_n + 1]
    spect_probs: jtp.Float[jtp.Array, " half_n"] = emp_spect / jnp.sum(emp_spect)

    # Draw frequencies dimension-by-dimension
    freq_keys: jtp.Float[jtp.Array, "D 2"] = jax.random.split(key_freq, num_dimensions)

    def sample_dim_freq(
        sub_key: jtp.Key[jtp.Array, ""],
    ) -> jtp.Float[jtp.Array, " Q"]:
        indices = jax.random.choice(sub_key, a=half_n + 1, shape=(num_mixtures,), p=spect_probs)
        return freqs[indices]

    frequencies_init: jtp.Float[jtp.Array, "Q D"] = jax.vmap(sample_dim_freq)(freq_keys).T

    # --- 3. Variances (Spectral Scales) [Shape: (Q, D)] ---
    # Calculate dimension-wise min/max distances
    distances: jtp.Float[jtp.Array, "N N D"] = jnp.abs(train_x[:, None, :] - train_x[None, :, :])

    # Reduce across spatial coordinates to find dimension-wise extents
    max_dists: jtp.Float[jtp.Array, " D"] = jnp.max(distances, axis=(0, 1))

    # Filter zeros to avoid self-distance calculation bounds
    filtered_dists: jtp.Float[jtp.Array, "N N D"] = jnp.where(distances > 0, distances, jnp.inf)
    min_dists: jtp.Float[jtp.Array, " D"] = jnp.min(filtered_dists, axis=(0, 1))
    min_dists = jnp.where(jnp.isinf(min_dists), 1e-4, min_dists)

    raw_scales: jtp.Float[jtp.Array, "Q D"] = jax.random.normal(key_scale, shape=(num_mixtures, num_dimensions))
    scaled_raw: jtp.Float[jtp.Array, "Q D"] = jnp.abs(raw_scales) * max_dists[None, :]

    variances_init: jtp.Float[jtp.Array, "Q D"] = jnp.clip(
        1.0 / (scaled_raw**2), min=1.0 / (max_dists[None, :] ** 2), max=1.0 / (min_dists[None, :] ** 2)
    )

    return weights_init, variances_init, frequencies_init


def build_sparse_sm_model(
    key: jtp.Key[jtp.Array, ""],
    train_x: jtp.Float[jtp.Array, "N D"],
    train_y: jtp.Float[jtp.Array, " N"],
    num_mixtures: int,
    num_inducing: int = 150,
) -> tuple[CollapsedVariationalGaussian, gpx.Dataset]:

    num_dimensions: int = train_x.shape[1]
    n_datapoints: int = train_x.shape[0]

    # 1. Instantiate Kernel & enforce parameter constraints
    sm_kernel = ARDSpectralMixture(num_mixtures=num_mixtures, num_dimensions=num_dimensions)

    w_init = jnp.ones((num_mixtures,)) * (jnp.std(train_y) / num_mixtures)
    v_init = jnp.ones((num_mixtures, num_dimensions)) * 1.0
    f_init = jax.random.uniform(key, shape=(num_mixtures, num_dimensions), minval=0.1, maxval=2.0)

    sm_kernel = eqx.tree_at(lambda k: k.weight, sm_kernel, PositiveReal(w_init))
    sm_kernel = eqx.tree_at(lambda k: k.variance, sm_kernel, PositiveReal(v_init))
    sm_kernel = eqx.tree_at(lambda k: k.frequency, sm_kernel, PositiveReal(f_init))

    # 2. Construct underlying GP structures
    dataset = gpx.Dataset(X=train_x, y=train_y)
    mean_function = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=sm_kernel)
    likelihood = gpx.gps.Gaussian(num_datapoints=n_datapoints)

    # 3. Initialize Inducing Inputs (Z)
    # Standard choice: Sub-sample points randomly from the input space
    inducing_key, _ = jax.random.split(key)
    inducing_idx = jax.random.choice(inducing_key, a=n_datapoints, shape=(num_inducing,), replace=False)
    inducing_inputs = train_x[inducing_idx]

    # 4. Formulate the Variational Posterior (SVGP)

    cvg = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=prior * likelihood,
        inducing_inputs=inducing_inputs,
    )

    return cvg, dataset


def build_constrained_ard_model(
    key: jtp.Key[jtp.Array, ""],
    train_x: jtp.Float[jtp.Array, "N D"],
    train_y: jtp.Float[jtp.Array, " N"],
    num_mixtures: int,
) -> tuple[gpx.gps.ConjugatePosterior, gpx.Dataset]:

    num_dimensions: int = train_x.shape[1]

    # 1. Evaluate data-driven initial values
    w, v, f = initialize_ard_sm_from_data(key, train_x, train_y, num_mixtures)

    # 2. Instantiate and bound parameter tracking constraints
    sm_ard_kernel = ARDSpectralMixture(num_mixtures=num_mixtures, num_dimensions=num_dimensions)

    sm_ard_kernel = eqx.tree_at(lambda k: k.weight, sm_ard_kernel, PositiveReal(w))
    sm_ard_kernel = eqx.tree_at(lambda k: k.variance, sm_ard_kernel, PositiveReal(v))
    sm_ard_kernel = eqx.tree_at(lambda k: k.frequency, sm_ard_kernel, PositiveReal(f))

    # 3. Formulate GP structures cleanly
    dataset = gpx.Dataset(X=train_x, y=train_y[:, None])
    mean_function = gpx.mean_functions.Constant()

    prior = gpx.gps.Prior(mean_function=mean_function, kernel=sm_ard_kernel)
    likelihood = gpx.gps.Gaussian(num_datapoints=dataset.n)

    posterior: gpx.gps.ConjugatePosterior = prior * likelihood

    return posterior, dataset


class ARDSpectralMixture(AbstractKernel):
    # Structural static parameters (marked for Equinox compilation)
    num_mixtures: int = eqx.field(static=True)
    num_dimensions: int = eqx.field(static=True)

    # Differentiable parameter arrays
    weight: jtp.Float[jtp.Array, "Q"]
    variance: jtp.Float[jtp.Array, "Q D"]  # Q mixtures by D dimensions
    frequency: jtp.Float[jtp.Array, "Q D"]  # Q mixtures by D dimensions

    def __init__(
        self,
        num_mixtures: int,
        num_dimensions: int,
        active_dims: typing.Union[list[int], slice] | None = None,
        n_dims: int | None = None,
    ) -> None:
        # Initialize base kernel structure
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        self.num_mixtures = num_mixtures
        self.num_dimensions = num_dimensions

        # Initial matrix configurations
        self.weight = jnp.ones((num_mixtures,)) / num_mixtures
        self.variance = jnp.ones((num_mixtures, num_dimensions)) * 1.0
        self.frequency = jnp.ones((num_mixtures, num_dimensions)) * 0.5

    def __call__(
        self,
        x: jtp.Float[gpx.typing.Array, " D"],
        y: jtp.Float[gpx.typing.Array, " D"],
    ) -> jtp.Float[jtp.Array, ""]:
        # Unpack wrappers to access the physical hyperparameter matrices
        constrained_self: ARDSpectralMixture = paramax.unwrap(self)

        # Element-wise dimension distance array of shape (D,)
        delta: jtp.Float[gpx.typing.Array, " D"] = x - y

        # Broad-casted computation matrices of shape (Q, D)
        exponent_term: jtp.Float[gpx.typing.Array, "Q D"] = -2 * jnp.pi**2 * (delta**2) * constrained_self.variance
        cosine_term: jtp.Float[gpx.typing.Array, "Q D"] = 2 * jnp.pi * delta * constrained_self.frequency

        # Individual covariance component matrix of shape (Q, D)
        dim_covariance: jtp.Float[jtp.Array, "Q D"] = jnp.exp(exponent_term) * jnp.cos(cosine_term)

        # Compute product across feature dimensions (axis=-1) to yield (Q,) vector
        mixture_covariance: jtp.Float[jtp.Array, " Q"] = constrained_self.weight * jnp.prod(dim_covariance, axis=-1)

        # Final scalar evaluation sum over mixtures
        return jnp.sum(mixture_covariance)


def make_predictive_function(
    optimized_posterior: gpx.gps.ConjugatePosterior,
    train_data: gpx.Dataset,
) -> typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]]:
    """
    Creates a JIT-compiled closure over the optimized GP model.
    Returns a function that directly maps new inputs to the latent mean vector.
    """

    # Define the raw closure
    def predict_mean_fn(
        xtest: jtp.Float[jtp.Array, "N_test D"],
    ) -> jtp.Float[jtp.Array, " N_test"]:
        latent_dist = optimized_posterior.predict(xtest, train_data=train_data, return_covariance_type="diagonal")
        # Squeeze removes any trailing axes to give a clean 1D or flat vector
        mean_vector: jtp.Float[jtp.Array, " N_test"] = jnp.asarray(latent_dist.mean).squeeze()
        return mean_vector

    # Return the highly optimized, JIT-compiled version
    return jax.jit(predict_mean_fn)


def make_sparse_predictive_function(
    optimized_svgp: CollapsedVariationalGaussian,
    train_data: gpx.Dataset,
) -> typing.Callable[[jtp.Float[jtp.Array, "N_test D"]], jtp.Float[jtp.Array, " N_test"]]:
    """
    Creates a JIT-compiled closure over an optimized Sparse GP (SVGP).
    Returns a function that maps test coordinates directly to a flat latent mean vector.
    """

    def predict_mean_fn(
        xtest: jtp.Float[jtp.Array, "N_test D"],
    ) -> jtp.Float[jtp.Array, " N_test"]:
        # Execute the variational posterior directly as a callable.
        # This scales at O(M^2) per test point against M inducing points, ignoring N=10000.
        latent_dist = optimized_svgp(xtest, train_data=train_data)

        # Squeeze and cast safely to a JAX array to satisfy Pylance
        mean_vector: jtp.Float[jtp.Array, " N_test"] = jnp.asarray(latent_dist.mean).squeeze()
        return mean_vector

    # Return the JIT-compiled version for accelerated, cached execution
    return jax.jit(predict_mean_fn)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Setup Dummy Time-Series Data ---
    key = jax.random.key(123)
    x_data = jnp.linspace(0, 15, 200)[:, jnp.newaxis]

    # Signal with complex frequency behavior
    y_data = jnp.sin(1.2 * x_data) + jnp.cos(0.4 * x_data) + jax.random.normal(key, (200, 1)) * 0.15

    # --- Build the model structure ---
    num_mixtures = 3
    posterior, dataset = build_constrained_ard_model(key, x_data, y_data.squeeze(), num_mixtures)

    # --- Define Objective Function ---
    def negative_mll(posterior_model, data):
        return -gpx.objectives.conjugate_mll(posterior_model, data)

    # --- Run the Optimization Loop ---
    print("Optimizing Spectral Mixture Hyperparameters...")
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=dataset,
        optim=optax.adam(learning_rate=0.01),
        num_iters=800,
        key=key,
    )
    print("Optimization complete!")

    # --- Out-of-sample Prediction Grid ---
    gp_predictor = make_predictive_function(opt_posterior, dataset)

    xtest = jnp.linspace(-5, 20, 500)[:, jnp.newaxis]
    latent_dist = opt_posterior.predict(xtest, train_data=dataset, return_covariance_type="diagonal")
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean
    predictive_std = jnp.sqrt(predictive_dist.variance)

    # --- Plotting Results ---
    fig, ax = plt.subplots(figsize=(7.5, 2.5))
    ax.plot(x_data, y_data, "x", label="Observations", color="k", alpha=0.5)
    ax.fill_between(
        xtest.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.2,
        label="Two sigma",
        color="pink",
    )
    ax.plot(xtest, predictive_mean, label="Predictive mean", color="red")
    ax.plot(xtest, gp_predictor(xtest), label="Generative mean", color="blue")
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.tight_layout()
    plt.show()

    # --- Generate 10,000 synthetic data points ---
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)

    N_total = 10000
    x_large = jnp.linspace(0, 50, N_total)[:, jnp.newaxis]
    y_large = jnp.sin(1.5 * x_large) * jnp.cos(0.3 * x_large) + jax.random.normal(subkey, (N_total, 1)) * 0.2

    # --- Configuration ---
    batch_size = 256
    num_epochs = 20
    num_iters = (N_total // batch_size) * num_epochs

    # --- Build Sparse Infrastructure ---
    # Using 150 sparse inducing coordinates instead of all 10k inputs
    svgp_posterior, dataset = build_sparse_sm_model(key, x_large, y_large, num_mixtures=4, num_inducing=150)

    print(f"Beginning SVGP Mini-batched Optimization for {num_iters} iterations...")

    # Native GPJax stochastic fitting engine handle
    opt_svgp, history = gpx.fit(
        model=svgp_posterior,
        # we want want to minimize the *negative* ELBO
        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
        train_data=dataset,
        optim=optax.adamw(learning_rate=1e-2),
        num_iters=500,
        key=key,
    )
    print("Sparse optimization complete!")

    gp_fast_predictor = make_sparse_predictive_function(opt_svgp, train_data=dataset)

    # Generate a massive out-of-sample prediction matrix to evaluate speed
    xtest_large = jnp.linspace(-5, 55, 2000)[:, jnp.newaxis]

    print("Compiling and evaluating sparse GP closure...")
    # First call: JAX compiles the graph
    latent_mean = gp_fast_predictor(xtest_large)
    print("Inference successful! Predicted array shape:", latent_mean.shape)

    # Secondary call: runs at native device/C++ speed (sub-millisecond execution)
    more_points = jnp.array([[12.4], [25.0], [42.1]])
    isolated_predictions = gp_fast_predictor(more_points)
    print("Fast prediction for new coordinates:", isolated_predictions)
