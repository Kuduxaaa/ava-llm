"""Incremental decoding state.

A single :class:`AvaCache` object carries the state for *every* layer, whatever
its type. That matters for hybrid models: attention layers need a growing K/V
buffer while Mamba layers need a fixed-size recurrent state, and the old
"tuple of tuples" convention could not express both. It also lets the model ask
one authoritative question -- ``cache.seen_tokens`` -- instead of reverse
engineering the position offset from a tensor shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttentionLayerCache:
    """Growing key/value buffers for one attention layer.

    Keys are stored **after** RoPE has been applied, so a cached key keeps the
    rotation for the absolute position it was computed at and is never rotated
    twice.
    """

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def update(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys, self.values = key, value
        else:
            self.keys = torch.cat([self.keys, key], dim=2)
            self.values = torch.cat([self.values, value], dim=2)
        return self.keys, self.values

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def reorder(self, index: torch.Tensor) -> None:
        if self.keys is not None:
            self.keys = self.keys.index_select(0, index)
            self.values = self.values.index_select(0, index)

    def crop(self, max_length: int) -> None:
        """Keep only the most recent ``max_length`` positions."""
        if self.keys is not None and self.keys.shape[2] > max_length:
            self.keys = self.keys[:, :, -max_length:, :]
            self.values = self.values[:, :, -max_length:, :]


@dataclass
class MambaLayerCache:
    """Recurrent SSM state plus the depthwise-conv lookback window.

    ``ssm_state`` is ``(batch, inner_dim, d_state)`` and ``conv_state`` is
    ``(batch, inner_dim, d_conv - 1)``. Both are fixed size, which is the whole
    point of a state-space model: decoding cost does not grow with context.
    """

    ssm_state: torch.Tensor | None = None
    conv_state: torch.Tensor | None = None

    def reorder(self, index: torch.Tensor) -> None:
        if self.ssm_state is not None:
            self.ssm_state = self.ssm_state.index_select(0, index)
        if self.conv_state is not None:
            self.conv_state = self.conv_state.index_select(0, index)


LayerCache = AttentionLayerCache | MambaLayerCache


@dataclass
class AvaCache:
    """Per-layer decoding state for a whole model."""

    layers: list[LayerCache] = field(default_factory=list)
    seen_tokens: int = 0

    @classmethod
    def from_config(cls, config) -> AvaCache:
        layers: list[LayerCache] = [
            AttentionLayerCache() if kind == "attention" else MambaLayerCache()
            for kind in config.layer_types()
        ]
        return cls(layers=layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> LayerCache:
        return self.layers[index]

    def advance(self, num_tokens: int) -> None:
        self.seen_tokens += num_tokens

    def reorder(self, index: torch.Tensor) -> None:
        """Reorder the batch dimension -- needed for beam search."""
        for layer in self.layers:
            layer.reorder(index)

    def reset(self) -> None:
        for i, layer in enumerate(self.layers):
            self.layers[i] = type(layer)()
        self.seen_tokens = 0
