from typing import List

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
        return self.layers(x)


def build_model(input_dim: int, hidden_layers: List[int], output_dim: int, batch_norm: bool) -> MLP:
    return MLP([input_dim] + hidden_layers + [output_dim], batch_normalization=batch_norm)

