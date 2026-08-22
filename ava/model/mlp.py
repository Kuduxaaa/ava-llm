"""Gated feed-forward network."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AvaConfig

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "relu2": lambda x: F.relu(x).square(),
}


class AvaMLP(nn.Module):
    """SwiGLU by default (Shazeer, 2020), any gated activation via ``hidden_act``."""

    def __init__(self, config: AvaConfig) -> None:
        super().__init__()
        if config.hidden_act not in ACTIVATIONS:
            available = ", ".join(sorted(ACTIVATIONS))
            raise ValueError(
                f"Unknown hidden_act {config.hidden_act!r}. Available: {available}"
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACTIVATIONS[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
