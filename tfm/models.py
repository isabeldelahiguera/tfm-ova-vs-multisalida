from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
from torchvision.models import VGG16_Weights, ViT_B_16_Weights, vgg16, vit_b_16


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
        if len(block_channels) < 1:
            raise ValueError("VGG block_channels must contain at least one channel width")
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


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class PretrainedVGG16(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, finetune: str = "frozen"):
        super().__init__()
        if input_channels != 3:
            raise ValueError("vgg16-pretrained requires RGB images with 3 input channels")
        if finetune not in {"frozen", "block5", "full"}:
            raise ValueError("finetune must be one of: frozen, block5, full")

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.normalize = ImageNetNormalize()
        self.features = backbone.features
        if finetune != "full":
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        if finetune == "block5":
            for layer in self.features[24:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.classifier(self.features(x))


class PretrainedViTB16(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, finetune: str = "frozen"):
        super().__init__()
        if input_channels != 3:
            raise ValueError("vit-b-16-pretrained requires RGB images with 3 input channels")
        if finetune == "block5":
            finetune = "last-block"
        if finetune not in {"frozen", "last-block", "full"}:
            raise ValueError("finetune must be one of: frozen, last-block, full")

        self.normalize = ImageNetNormalize()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if finetune != "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        if finetune == "last-block":
            for parameter in self.backbone.encoder.layers[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.backbone.encoder.ln.parameters():
                parameter.requires_grad = True

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.backbone(self.normalize(x))


def build_model(
    input_dim: int,
    hidden_layers: List[int],
    output_dim: int,
    batch_norm: bool,
    model_arch: str = "mlp",
    input_shape: tuple[int, ...] | None = None,
    vgg_channels: Sequence[int] = DEFAULT_VGG_CHANNELS,
    pretrained_finetune: str = "frozen",
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

    if model_arch == "vgg16-pretrained":
        if input_shape is None or len(input_shape) != 3:
            raise ValueError("vgg16-pretrained requires image input with shape (channels, height, width)")
        return PretrainedVGG16(
            input_channels=input_shape[0],
            output_dim=output_dim,
            finetune=pretrained_finetune,
        )

    if model_arch == "vit-b-16-pretrained":
        if input_shape is None or len(input_shape) != 3:
            raise ValueError("vit-b-16-pretrained requires image input with shape (channels, height, width)")
        return PretrainedViTB16(
            input_channels=input_shape[0],
            output_dim=output_dim,
            finetune=pretrained_finetune,
        )

    raise ValueError(f"Unsupported model architecture: {model_arch}")
