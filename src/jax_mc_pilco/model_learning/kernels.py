"""Bespoke Spectral Mixture Kernel."""

import jax
import jax.numpy as jnp

import tinygp


class SpectralMixture(tinygp.kernels.Kernel):
    weight: jax.Array
    scale: jax.Array
    freq: jax.Array

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(-2 * jnp.pi**2 * tau**2 / self.scale**2)
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=0,
            )
        )


class MultiOutputKernel(tinygp.kernels.Kernel):
    """
    The `MultiOutputKernel` is a base class for multi-output kernels. It assumes that the first dimension of `X` contains channel IDs (integers) and calculates the final kernel matrix accordingly. Concretely, it will call the `Ksub` method for derived kernels from this class, which should return the kernel matrix between channel `i` and `j`, given inputs `X1` and `X2`. This class will automatically split and recombine the input vectors and kernel matrices respectively, in order to create the final kernel matrix of the multi-output kernel.

    Be aware that for implementation of `Ksub`, `i==j` is true for the diagonal matrices. `X2==None` is true when calculating the Gram matrix (i.e. `X1==X2`) and when `i==j`. It is thus a subset of the case `i==j`, and if `X2==None` than `i` is always equal to `j`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
    """

    # TODO: seems to accumulate a lot of memory in the loops to call Ksub, perhaps it's keeping the computational graph while indexing?

    def __init__(self, output_dims, input_dims=None, active_dims=None):
        super().__init__(input_dims, active_dims)
        self.output_dims = output_dims

    def _check_input(self, X1, X2=None):
        X1, X2 = super()._check_input(X1, X2)
        if not torch.all(X1[:, 0] == X1[:, 0].long()) or not torch.all(
            X1[:, 0] < self.output_dims
        ):
            raise ValueError(
                "X must have integers for the channel IDs in the first input dimension"
            )
        if (
            X2 is not None
            and not torch.all(X2[:, 0] == X2[:, 0].long())
            or not torch.all(X1[:, 0] < self.output_dims)
        ):
            raise ValueError(
                "X must have integers for the channel IDs in the first input dimension"
            )
        return X1, X2

    def K(self, X1, X2=None):
        # X has shape (data_points,1+input_dims) where the first column is the channel ID
        # extract channel mask, get data, and find indices that belong to the channels
        c1 = X1[:, 0].long()
        m1 = [c1 == i for i in range(self.output_dims)]
        x1 = [X1[m1[i], 1:] for i in range(self.output_dims)]
        r1 = [torch.nonzero(m1[i], as_tuple=False) for i in range(self.output_dims)]

        if X2 is None:
            r2 = [r1[i].reshape(1, -1) for i in range(self.output_dims)]

            res = torch.empty(
                X1.shape[0], X1.shape[0], device=config.device, dtype=config.dtype
            )  # NxM
            for i in range(self.output_dims):
                for j in range(i + 1):
                    # calculate sub kernel matrix and add to main kernel matrix
                    if i == j:
                        k = self.Ksub(i, i, x1[i])
                        res[r1[i], r2[i]] = k
                    else:
                        k = self.Ksub(i, j, x1[i], x1[j])
                        res[r1[i], r2[j]] = k
                        res[r1[j], r2[i]] = k.T
        else:
            # extract channel mask, get data, and find indices that belong to the channels
            c2 = X2[:, 0].long()
            m2 = [c2 == j for j in range(self.output_dims)]
            x2 = [X2[m2[j], 1:] for j in range(self.output_dims)]
            r2 = [
                torch.nonzero(m2[j], as_tuple=False).reshape(1, -1)
                for j in range(self.output_dims)
            ]

            res = torch.empty(
                X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype
            )  # NxM
            for i in range(self.output_dims):
                for j in range(self.output_dims):
                    # calculate sub kernel matrix and add to main kernel matrix
                    res[r1[i], r2[j]] = self.Ksub(i, j, x1[i], x2[j])

        return res

    def K_diag(self, X1):
        # extract channel mask, get data, and find indices that belong to the channels
        c1 = X1[:, 0].long()
        m1 = [c1 == i for i in range(self.output_dims)]
        x1 = [X1[m1[i], 1:] for i in range(self.output_dims)]
        r1 = [
            torch.nonzero(m1[i], as_tuple=False)[:, 0] for i in range(self.output_dims)
        ]

        res = torch.empty(X1.shape[0], device=config.device, dtype=config.dtype)  # NxM

        for i in range(self.output_dims):
            # calculate sub kernel matrix and add to main kernel matrix
            res[r1[i]] = self.Ksub_diag(i, x1[i])
        return res

    def Ksub(self, i, j, X1, X2=None):
        """
        Calculate kernel matrix between two channels. If `X2` is not given, it is assumed to be the same as `X1`. Not passing `X2` may be faster for some kernels.

        Args:
            X1 (torch.tensor): Input of shape (data_points0,input_dims).
            X2 (torch.tensor): Input of shape (data_points1,input_dims).

        Returns:
            torch.tensor: Kernel matrix of shape (data_points0,data_points1).
        """
        raise NotImplementedError()

    def Ksub_diag(self, i, X1):
        """
        Calculate the diagonal of the kernel matrix between two channels. This is usually faster than `Ksub(X1).diagonal()`.

        Args:
            X1 (torch.tensor): Input of shape (data_points,input_dims).

        Returns:
            torch.tensor: Kernel matrix diagonal of shape (data_points,).
        """
        return self.Ksub(i, i, X1).diagonal()


class MultiOutputSpectralMixtureKernel(MultiOutputKernel):
    """
    Multi-output spectral mixture kernel (MOSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1].

    $$ K_{ij}(x,x') = \\sum_{q=0}^Q\\alpha_{ijq} \\exp\\left(-\\frac{1}{2}(\\tau+\\theta_{ijq})^T\\Sigma_{ijq}(\\tau+\\theta_{ijq})\\right) \\cos((\\tau+\\theta_{ijq})^T\\mu_{ijq} + \\phi_{ijq}) $$

    $$ \\alpha_{ijq} = w_{ijq}\\sqrt{\\left((2\\pi)^n|\\Sigma_{ijq}|\\right)} $$

    $$ w_{ijq} = w_{iq}w_{jq}\\exp\\left(-\\frac{1}{4}(\\mu_{iq}-\\mu_{jq})^T(\\Sigma_{iq}+\\Sigma_{jq})^{-1}(\\mu_{iq}-\\mu_{jq})\\right) $$

    $$ \\mu_{ijq} = (\\Sigma_{iq}+\\Sigma_{jq})^{-1}(\\Sigma_{iq}\\mu_{jq} + \\Sigma_{jq}\\mu_{iq}) $$

    $$ \\Sigma_{ijq} = 2\\Sigma_{iq}(\\Sigma_{iq}+\\Sigma_{jq})^{-1}\\Sigma_{jq}$$

    with \\(\\theta_{ijq} = \\theta_{iq}-\\theta_{jq}\\), \\(\\phi_{ijq} = \\phi_{iq}-\\phi_{jq}\\), \\(\\tau = |x-x'|\\), \\(w\\) the weight, \\(\\mu\\) the mean, \\(\\Sigma\\) the variance, \\(\\theta\\) the delay, and \\(\\phi\\) the phase.

    Args:
        Q (int): Number mixture components.
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,Q).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (output_dims,Q,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,Q,input_dims).
        delay (mogptk.gpr.parameter.Parameter): Delay \\(\\theta\\) of shape (output_dims,Q,input_dims).
        phase (mogptk.gpr.parameter.Parameter): Phase \\(\\phi\\) in hertz of shape (output_dims,Q).

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """

    def __init__(
        self, num_mixture_components: Int, num_outputs: Int, input_dimension: Int
    ):
        super().__init__(num_outputs, input_dims)

        weight = jnp.ones((output_dims, num_mixture_components))
        mean = jnp.zeros((output_dims, num_mixture_components, input_dims))
        variance = jnp.ones((output_dims, num_mixture_components, input_dims))
        delay = jnp.zeros((output_dims, num_mixture_components, input_dims))
        phase = jnp.zeros((output_dims, num_mixture_components))

        self.input_dims = input_dims
        self.weight = weight
        self.mean = mean
        self.variance = variance
        self.delay = delay
        self.phase = phase

        self.twopi = jnp.power(2.0 * jnp.pi, 0.5 * float(self.input_dims))

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1, X2)  # NxMxD
        if i == j:
            variance = self.variance()[i]  # QxD
            alpha = (
                self.weight()[i] ** 2 * self.twopi * variance.prod(dim=1).sqrt()
            )  # Q
            _exp = jnp.exp(
                -0.5 * jnp.einsum("nmd,qd->qnm", tau**2, variance)
            )  # QxNxM
            _cos = jnp.cos(
                2.0 * jnp.pi * jnp.einsum("nmd,qd->qnm", tau, self.mean()[i])
            )  # QxNxM
            Kq = alpha[:, None, None] * _exp * _cos  # QxNxM
        else:
            inv_variances = 1.0 / (self.variance()[i] + self.variance()[j])  # QxD

            diff_mean = self.mean()[i] - self.mean()[j]  # QxD
            magnitude = (
                self.weight()[i]
                * self.weight()[j]
                * jnp.exp(
                    -jnp.square(jnp.pi)
                    * jnp.sum(diff_mean * inv_variances * diff_mean, dim=1)
                )
            )  # Q

            mean = inv_variances * (
                self.variance()[i] * self.mean()[j]
                + self.variance()[j] * self.mean()[i]
            )  # QxD
            variance = (
                2.0 * self.variance()[i] * inv_variances * self.variance()[j]
            )  # QxD
            delay = self.delay()[i] - self.delay()[j]  # QxD
            phase = self.phase()[i] - self.phase()[j]  # Q

            alpha = magnitude * self.twopi * variance.prod(dim=1).sqrt()  # Q
            tau_delay = tau[None, :, :, :] + delay[:, None, None, :]  # QxNxMxD
            _exp = jnp.exp(
                -0.5 * jnp.einsum("qnmd,qd->qnm", jnp.square(tau_delay), variance)
            )  # QxNxM
            _cos = jnp.cos(
                2.0
                * jnp.pi
                * (jnp.einsum("qnmd,qd->qnm", tau_delay, mean) + phase[:, None, None])
            )  # QxNxM
            Kq = alpha[:, None, None] * _exp * _cos  # QxNxM
        return jnp.sum(Kq, dim=0)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variance = self.variance()[i]
        alpha = self.weight()[i] ** 2 * self.twopi * variance.prod(dim=1).sqrt()  # Q
        return jnp.sum(alpha).repeat(X1.shape[0])
