"""Weight-only post-training quantisation for inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """int8 / int4 weight-only linear with per-output-channel scales.

    Per-channel scaling rather than one scale for the whole tensor: a single
    outlier row otherwise sets the step size for every other row and quantised
    quality collapses. The weight is stored small and dequantised per call, so
    this buys memory, not arithmetic speed -- for speed you want a fused kernel
    (bitsandbytes, torchao), which this class deliberately does not pretend to be.

    Note that 4-bit values are still held in an ``int8`` container -- the
    quantisation error is 4-bit, the storage is not packed.
    """

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None, bits: int = 8
    ) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"Only 4- and 8-bit quantisation are supported, got {bits}.")

        self.bits = bits
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

        q_max = 2 ** (bits - 1) - 1
        q_min = -(2 ** (bits - 1))

        weight = weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True) / q_max
        scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (weight / scale).round().clamp(q_min, q_max).to(torch.int8)

        self.register_buffer("quantized_weight", quantized)
        self.register_buffer("scale", scale.squeeze(1))
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def dequantize(self) -> torch.Tensor:
        return self.quantized_weight.to(self.scale.dtype) * self.scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}"
        )


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    skip_modules: Sequence[str] = ("lm_head",),
) -> nn.Module:
    """Replace every ``nn.Linear`` with a quantised equivalent, in place.

    ``lm_head`` is skipped by default: it is the layer whose error feeds
    straight into the sampled distribution, and it is usually tied to the
    embeddings anyway.
    """
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear) and name not in skip_modules
    ]

    for parent, name, layer in targets:
        setattr(
            parent,
            name,
            QuantizedLinear(layer.weight, layer.bias, bits=bits),
        )
    return model


def quantized_size_bytes(model: nn.Module) -> int:
    """Actual on-device footprint, counting int8 buffers as one byte each."""
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total
