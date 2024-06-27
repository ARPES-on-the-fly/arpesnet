""" Description: Module containing the ARPESNet model for the neural ARPES project.

This is an evolution of the "large128" model, with all parameters exposed for easy tuning.

The model is composed of an encoder and a decoder, each with a number of blocks. 
Each block is composed of a number of convolutional layers, with the number of layers and the kernel 
size decreasing as the depth of the network increases.

"""
from torch import nn
import torch

def conv_block(
    in_channels: int,
    out_channels: int,
    n_layers: int,
    kernel_size: int,
    stride: int,
    relu=nn.LeakyReLU,
    relu_kwargs={"negative_slope": 0.01},
) -> list:
    layers = []
    for _ in range(n_layers - 1):
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        layers.append(relu(**relu_kwargs))
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
    )
    layers.append(relu(**relu_kwargs))
    return layers


def deconv_block(
    in_channels: int,
    out_channels: int,
    n_layers: int,
    kernel_size: int,
    stride: int,
    relu=nn.LeakyReLU,
    relu_kwargs={"negative_slope": 0.01},
) -> list:
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=0 if stride == 1 else stride // 2,
        )
    )
    layers.append(relu(**relu_kwargs))
    for _ in range(n_layers - 1):
        layers.append(
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                output_padding=0,
            )
        )
        layers.append(relu(**relu_kwargs))
    return layers


def base_model(
    kernel_size: int = 11,
    kernel_decay: float = 2,
    n_layers: int = 1,
    start_channels: int = 4,
    max_channels: int = 32,
    n_blocks: int = 6,
    relu=nn.PReLU,
    relu_kwargs=dict(num_parameters=1, init=0.25),
) -> tuple[nn.Sequential, nn.Sequential]:
    """Create a base model for the encoder and decoder of the ARPESNet

    Args:
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 1.
        negative_slope (float, optional): Negative slope for the LeakyReLU activation. Defaults to 0.001.
        n_layers (int, optional): Number of layers for each block. Defaults to 1.
        start_channels (int, optional): Number of channels for the first block. Defaults to 4.
        max_channels (int, optional): Maximum number of channels for the encoder. Defaults to 32.
        n_blocks (int, optional): Number of blocks for the encoder. Defaults to 6.

    Returns:
        tuple[nn.Sequential, nn.Sequential]: Tuple with the encoder and decoder models"""

    enc_layers = []
    io_chans = []
    if isinstance(relu, str):
        relu = getattr(nn, relu)
    for i in range(n_blocks):
        k = max(3, kernel_size - kernel_decay * i)
        if i == 0:
            ic, oc = 1, start_channels
        else:
            ic, oc = oc, min(oc * 2, max_channels)
        io_chans.append((oc, ic))
        enc_layers.extend(
            conv_block(
                in_channels=ic,
                out_channels=oc,
                n_layers=n_layers,
                kernel_size=k,
                stride=2,
                relu=relu,
                relu_kwargs=relu_kwargs,
            )
        )
    encoder = nn.Sequential(*enc_layers)

    io_chans = io_chans[::-1]
    dec_layers = []
    for i in range(n_blocks):
        k = max(3, kernel_size - kernel_decay * (n_blocks - i - 1))
        ic, oc = io_chans[i]
        dec_layers.extend(
            deconv_block(
                in_channels=ic,
                out_channels=oc,
                n_layers=n_layers,
                kernel_size=k,
                stride=2,

                relu=relu,
                relu_kwargs=relu_kwargs,
            )
        )
    decoder = nn.Sequential(*dec_layers)
    return encoder, decoder


class Encoder(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        n_layers: int = 1,
        start_channels: int = 4,
        max_channels: int = 32,
        n_blocks: int = 6,
        input_shape: tuple[int, int] = (256, 256),
        relu=nn.PReLU,
        relu_kwargs=dict(num_parameters=1, init=0.25),
        **kwargs,
    ) -> None:
        super().__init__()

        self.input_shape = torch.Size(input_shape)
        self.encoder_cnn = base_model(
            kernel_size=kernel_size,
            n_layers=n_layers,
            start_channels=start_channels,
            max_channels=max_channels,
            n_blocks=n_blocks,
            relu=relu,
            relu_kwargs=relu_kwargs,
        )[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Add unsqueeze if necessary.
        if x.shape[-2:] != self.input_shape:
            raise ValueError(
                f"Input shape {x.shape[-2:]} does not match expected shape {self.input_shape}"
            )
        x = self.encoder_cnn(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        n_layers: int = 1,
        start_channels: int = 4,
        max_channels: int = 32,
        n_blocks: int = 6,
        input_shape: tuple[int, int] = (256, 256),
        relu=nn.PReLU,
        relu_kwargs=dict(num_parameters=1, init=0.25),
        **kwargs,
    ) -> None:
        super().__init__()

        self.input_shape = torch.tensor(input_shape)
        self.decoder_cnn = base_model(
            kernel_size=kernel_size,
            n_layers=n_layers,
            start_channels=start_channels,
            max_channels=max_channels,
            n_blocks=n_blocks,
            relu=relu,
            relu_kwargs=relu_kwargs,
        )[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_cnn(x)
        return x
