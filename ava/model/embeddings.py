"""Rotary position embeddings, with optional context-extension scaling."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _yarn_ramp(low: float, high: float, dim: int, device) -> torch.Tensor:
    """Linear ramp from 0 to 1 over the frequency band ``[low, high]``."""
    if low == high:
        high += 0.001  # avoid a zero-width band
    index = torch.arange(dim, dtype=torch.float32, device=device)
    return ((index - low) / (high - low)).clamp(0.0, 1.0)


def _yarn_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """The rotary dimension whose wavelength completes ``num_rotations``."""
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


class AvaRotaryEmbedding(nn.Module):
    """Computes ``cos``/``sin`` for arbitrary absolute positions.

    Unlike a cache-and-slice implementation, this takes explicit ``position_ids``
    and evaluates the frequencies for exactly those positions. That is what makes
    incremental decoding correct: at step *t* the query really is rotated by
    angle *t*, not by whatever offset a sliced cache happened to yield.

    ``rope_scaling`` supports the three standard context-extension schemes:

    ``linear``
        Divide every position by ``factor`` (position interpolation).
    ``ntk``
        Raise the base so high frequencies stay intact and low ones stretch.
    ``yarn``
        Per-band interpolation plus attention temperature -- the strongest of
        the three, and the one to reach for past a 4x extension.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 500000.0,
        scaling: dict[str, Any] | None = None,
        device=None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling = scaling or {}
        self.scaling_type = self.scaling.get("type")
        self.factor = float(self.scaling.get("factor", 1.0))
        self.attention_scale = 1.0

        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, device) -> torch.Tensor:
        half = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        exponent = half / self.dim

        if self.scaling_type == "ntk":
            # Raise the base so the longest wavelength stretches by `factor`.
            base = self.base * self.factor ** (self.dim / (self.dim - 2))
            return 1.0 / (base**exponent)

        inv_freq = 1.0 / (self.base**exponent)

        if self.scaling_type == "linear":
            return inv_freq / self.factor

        if self.scaling_type == "yarn":
            original = int(
                self.scaling.get(
                    "original_max_position_embeddings", self.max_position_embeddings
                )
            )
            beta_fast = float(self.scaling.get("beta_fast", 32.0))
            beta_slow = float(self.scaling.get("beta_slow", 1.0))

            low = math.floor(_yarn_correction_dim(beta_fast, self.dim, self.base, original))
            high = math.ceil(_yarn_correction_dim(beta_slow, self.dim, self.base, original))
            low = max(low, 0)
            high = min(high, self.dim - 1)

            # 0 -> keep the original (extrapolate), 1 -> fully interpolate.
            ramp = _yarn_ramp(low / 2, high / 2, self.dim // 2, device)
            inv_freq = inv_freq / self.factor * ramp + inv_freq * (1 - ramp)

            # YaRN also raises attention temperature to counter the entropy
            # increase from a longer context.
            self.attention_scale = 0.1 * math.log(self.factor) + 1.0

        return inv_freq

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(batch, 1, seq_len, head_dim)``.

        ``position_ids`` is ``(batch, seq_len)`` of absolute positions.
        """
        inv_freq = self.inv_freq.to(position_ids.device)
        # (batch, dim/2, 1) x (batch, 1, seq) -> (batch, dim/2, seq)
        freqs = inv_freq[None, :, None].float() @ position_ids[:, None, :].float()
        freqs = freqs.transpose(1, 2)  # (batch, seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scale
        sin = emb.sin() * self.attention_scale
        return cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key.

    ``cos``/``sin`` broadcast over the head dimension, so this is correct for
    grouped-query attention where ``q`` and ``k`` have different head counts.
    """
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
