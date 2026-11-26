from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    activation: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, key=key)
        self.norm = eqx.nn.GroupNorm(groups=min(8, out_channels), channels=out_channels)
        self.activation = jax.nn.silu

    def __call__(self, x: jtp.Float[jtp.Array, "C H W"]) -> jtp.Float[jtp.Array, "OutC NewH NewW"]:
        return self.activation(self.norm(self.conv(x)))


class DeconvBlock(eqx.Module):
    deconv: eqx.nn.ConvTranspose2d
    norm: eqx.nn.GroupNorm
    activation: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        self.deconv = eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            key=key,
        )
        self.norm = eqx.nn.GroupNorm(groups=min(8, out_channels), channels=out_channels)
        self.activation = jax.nn.silu

    def __call__(self, x: jtp.Float[jtp.Array, "C H W"]) -> jtp.Float[jtp.Array, "OutC NewH NewW"]:
        return self.activation(self.norm(self.deconv(x)))


class Encoder(eqx.Module):
    blocks: list[ConvBlock]

    def __init__(
        self,
        in_channels: int = 3,
        hid_channels: int = 64,
        num_spatial_layers: int = 4,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        keys = jax.random.split(key, num_spatial_layers)
        layers = [ConvBlock(in_channels, hid_channels, kernel_size=3, stride=2, padding=1, key=keys[0])]
        for i in range(1, num_spatial_layers):
            layers.append(ConvBlock(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, key=keys[i]))
        self.blocks = layers

    def __call__(self, x: jtp.Float[jtp.Array, "C H W"]) -> jtp.Float[jtp.Array, " FlattenedDim"]:
        for block in self.blocks:
            x = block(x)
        return jnp.ravel(x)


class Decoder(eqx.Module):
    hid_channels: int
    spatial_init: int
    proj: eqx.nn.Linear
    blocks: list[DeconvBlock]
    readout: eqx.nn.Conv2d

    def __init__(
        self,
        in_dim: int,
        out_channels: int = 3,
        hid_channels: int = 64,
        num_spatial_layers: int = 4,
        img_size: int = 64,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        self.hid_channels = hid_channels
        self.spatial_init = img_size // (2**num_spatial_layers)  # 4 for 64x64 with 4 layers
        keys = jax.random.split(key, num_spatial_layers + 2)

        self.proj = eqx.nn.Linear(in_dim, hid_channels * self.spatial_init * self.spatial_init, key=keys[0])
        self.blocks = [DeconvBlock(hid_channels, hid_channels, key=keys[i + 1]) for i in range(num_spatial_layers)]
        self.readout = eqx.nn.Conv2d(hid_channels, out_channels, kernel_size=1, stride=1, padding=0, key=keys[-1])

    def __call__(self, latent_and_hidden: jtp.Float[jtp.Array, " InDim"]) -> jtp.Float[jtp.Array, "OutC H W"]:
        x = jax.nn.silu(self.proj(latent_and_hidden))
        x = jnp.reshape(x, (self.hid_channels, self.spatial_init, self.spatial_init))
        for block in self.blocks:
            x = block(x)
        return self.readout(x)
