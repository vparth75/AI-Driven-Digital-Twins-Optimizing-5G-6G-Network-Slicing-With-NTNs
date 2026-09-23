"""Actor and critic networks used by the paper-style DDPG agent."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, output_activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.extend((nn.Linear(previous, output_dim), output_activation))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Continuous policy μ(s); tanh bounds each raw action to [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.model = _mlp(state_dim, hidden_dims, action_dim, nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)


class Critic(nn.Module):
    """Action-value Q(s, a), with no output activation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.model = _mlp(state_dim + action_dim, hidden_dims, 1, nn.Identity())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((state, action), dim=-1))
