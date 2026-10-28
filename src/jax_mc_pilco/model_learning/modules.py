from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jtp


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    activation: Callable[[jtp.Float[jtp.Array, "..."]], jtp.Float[jtp.Array, "..."]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, key=key)
        # Using GroupNorm (groups=8) as a stable alternative to BatchNorm in JAX
        self.norm = eqx.nn.GroupNorm(groups=out_channels, channels=out_channels)
        self.activation = jax.nn.silu

    def __call__(self, x: jtp.Float[jtp.Array, "C H W"], **kwargs) -> jtp.Float[jtp.Array, "OutC NewH NewW"]:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


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
        layers = []

        # First layer handles input channels (e.g., RGB frames concatenated or batched)
        layers.append(ConvBlock(in_channels, hid_channels, kernel_size=3, stride=2, padding=1, key=keys[0]))

        # Subsequent downsampling layers
        for i in range(1, num_spatial_layers):
            layers.append(ConvBlock(hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, key=keys[i]))

        self.blocks = layers

    def __call__(self, x: jtp.Float[jtp.Array, "C H W"], **kwargs) -> jtp.Float[jtp.Array, " FlattenedDim"]:
        # Input shape expected: (Channels, Height, Width)
        for block in self.blocks:
            x = block(x)
        return jnp.ravel(x)


class Decoder(eqx.Module):
    hid_channels: int
    blocks: list[eqx.nn.ConvTranspose2d]
    readout: eqx.nn.Conv2d

    def __init__(
        self,
        out_channels: int = 3,
        hid_channels: int = 64,
        num_spatial_layers: int = 4,
        *,
        key: jtp.Key[jtp.Array, ""],
    ):
        self.hid_channels = hid_channels
        keys = jax.random.split(key, num_spatial_layers + 1)
        layers = []

        # Upsampling layers using ConvTranspose2d
        for i in range(num_spatial_layers - 1):
            layers.append(
                eqx.nn.ConvTranspose2d(
                    hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[i]
                )
            )

        # Final upsampling layer
        layers.append(
            eqx.nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[-2]
            )
        )
        self.blocks = layers

        # Readout mapping back to target video channels
        self.readout = eqx.nn.Conv2d(hid_channels, out_channels, kernel_size=1, stride=1, padding=0, key=keys[-1])

    def __call__(self, x: jtp.Float[jtp.Array, " FlattenedDim"], **kwargs) -> jtp.Float[jtp.Array, "OutC H W"]:
        # Input shape: hidden features from Translator/Encoder
        x = jnp.reshape(x, (self.hid_channels, 4, 4))
        for block in self.blocks:
            x = block(x)
            x = jax.nn.silu(x)  # Activation between upsamplings
        return self.readout(x)
