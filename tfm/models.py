from __future__ import annotations

from typing import List, Sequence

from torch import nn


class MLP(nn.Module):
    def __init__(self, layers_dims: List[int], batch_normalization: bool = False):
        super().__init__()
        if len(layers_dims) < 2:
            raise ValueError("layers_dims must contain at least input and output dimensions")

        layers = []
        for i in range(len(layers_dims) - 2):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1], bias=True))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layers_dims[-2], layers_dims[-1], bias=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.layers(x)


DEFAULT_VGG_CHANNELS = [32, 64, 128]


class VGG(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        batch_normalization: bool = False,
        block_channels: Sequence[int] = DEFAULT_VGG_CHANNELS,
    ):
        super().__init__()
        config = self._block_config(block_channels)
        final_channels = int(block_channels[-1])
        self.features = self._make_features(
            input_channels,
            config,
            batch_normalization,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(final_channels, final_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(final_channels, output_dim),
        )

    @staticmethod
    def _block_config(block_channels: Sequence[int]) -> list[int | str]:
        if len(block_channels) != 3:
            raise ValueError("VGG block_channels must contain exactly three channel widths")
        if any(int(channels) <= 0 for channels in block_channels):
            raise ValueError("VGG block_channels must be positive")
        config: list[int | str] = []
        for channels in block_channels:
            config.extend([int(channels), int(channels), "M"])
        return config

    @staticmethod
    def _make_features(input_channels: int, config: Sequence[int | str], batch_normalization: bool) -> nn.Sequential:
        layers = []
        channels = input_channels
        for item in config:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_channels = int(item)
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=not batch_normalization))
            if batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(
    input_dim: int,
    hidden_layers: List[int],
    output_dim: int,
    batch_norm: bool,
    model_arch: str = "mlp",
    input_shape: tuple[int, ...] | None = None,
    vgg_channels: Sequence[int] = DEFAULT_VGG_CHANNELS,
) -> nn.Module:
    if model_arch == "mlp":
        return MLP([input_dim] + hidden_layers + [output_dim], batch_normalization=batch_norm)

    if model_arch == "vgg":
        if input_shape is None or len(input_shape) != 3:
            raise ValueError("VGG requires image input with shape (channels, height, width)")
        return VGG(
            input_channels=input_shape[0],
            output_dim=output_dim,
            batch_normalization=batch_norm,
            block_channels=vgg_channels,
        )

    raise ValueError(f"Unsupported model architecture: {model_arch}")
