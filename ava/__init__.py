"""Ava -- a from-scratch language-model framework in pure PyTorch.

Transformer, Mamba (selective state-space) and hybrid stacks share one config,
one cache, one training loop and one generation path.

    from ava import AvaConfig, AvaForCausalLM

    model = AvaForCausalLM.from_preset("hybrid-130m")
    print(model.num_parameters())
"""

from __future__ import annotations

from .config import PRESETS, AvaConfig
from .model import (
    AvaCache,
    AvaDecoderLayer,
    AvaForCausalLM,
    AvaModel,
    GenerationConfig,
    MambaBlock,
    SelectiveSSM,
)

__version__ = "0.2.0"

__all__ = [
    "PRESETS",
    "AvaCache",
    "AvaConfig",
    "AvaDecoderLayer",
    "AvaForCausalLM",
    "AvaModel",
    "GenerationConfig",
    "MambaBlock",
    "SelectiveSSM",
    "__version__",
]
