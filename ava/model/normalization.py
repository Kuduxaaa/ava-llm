"""Normalisation layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_FUSED_RMS_NORM = hasattr(F, "rms_norm")


class AvaRMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    Uses ``F.rms_norm`` where the PyTorch build provides it. That is not just
    tidier: written out by hand this is seven separate kernels (upcast, square,
    mean, add, rsqrt, scale, downcast), and a model runs one of these per
    attention input, per MLP input, and -- with QK-norm on -- twice more per
    layer. On a decode step those launches dominate a small model's runtime.

    Either path reduces in fp32 regardless of the surrounding autocast dtype.
    Under bf16 the mean-square of a wide hidden state is exactly where silent
    precision loss creeps into a long run, so that upcast is not optional; the
    fused kernel does it internally, and the fallback does it explicitly.

    The gain is cast to the activation dtype before the call. Parameters stay
    fp32 under autocast while activations are bf16, and a dtype mismatch makes
    PyTorch refuse to dispatch to the fused kernel -- so without the cast the
    fast path is silently off in exactly the configuration it exists for.
    """

    def __init__(self, hidden_size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _HAS_FUSED_RMS_NORM:
            weight = self.weight
            if weight.dtype != hidden_states.dtype:
                weight = weight.to(hidden_states.dtype)
            return F.rms_norm(hidden_states, (self.hidden_size,), weight, self.epsilon)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.epsilon}"
