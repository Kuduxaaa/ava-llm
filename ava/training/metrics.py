"""Evaluation and throughput measurement."""

from __future__ import annotations

import contextlib
import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dense bf16/fp16 tensor-core peaks, in FLOP/s. Sparsity numbers are excluded
# on purpose -- quoting them makes MFU look about twice as good as it is.
_PEAK_FLOPS: dict[str, float] = {
    "H200": 989e12,
    "H100": 989e12,
    "A100": 312e12,
    "L40S": 362e12,
    "L4": 121e12,
    "A10G": 125e12,
    "V100": 125e12,
    "T4": 65e12,
    "RTX 4090": 165e12,
    "RTX 3090": 71e12,
}


def device_peak_flops(device: torch.device) -> float | None:
    """Best-effort peak throughput for the current accelerator."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for key, value in _PEAK_FLOPS.items():
        if key.lower() in name.lower():
            return value
    return None


def flops_per_token(config, seq_len: int) -> int:
    """Forward+backward FLOPs per token, Chinchilla-style.

    ``6 * parameters`` covers every matmul whose cost is independent of context;
    the second term is the attention score/value product, which is the part that
    grows with sequence length and the reason MFU drops as context grows.
    """
    dense = 6 * config.estimate_parameters()
    num_attention_layers = config.layer_types().count("attention")
    attention = 12 * num_attention_layers * config.attention_dim * seq_len
    return dense + attention


class ThroughputMeter:
    """Tokens/second and model FLOPs utilisation over a sliding window."""

    def __init__(
        self,
        config,
        device: torch.device,
        window: float = 30.0,
        seq_len: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.window = window
        self.peak_flops = device_peak_flops(device)
        # The sequence length actually being trained on, which is often well
        # below max_position_embeddings. Using the latter instead inflates the
        # attention term and reports an MFU that is too good.
        self.seq_len = seq_len or config.max_position_embeddings

        self.tokens = 0
        self._window_tokens = 0
        self._window_start = time.perf_counter()
        self._last: dict[str, Any] = {"tokens_per_second": 0.0, "mfu": None}

    def update(self, num_tokens: int, seq_len: int | None = None) -> None:
        if seq_len:
            self.seq_len = seq_len
        self.tokens += num_tokens
        self._window_tokens += num_tokens

        elapsed = time.perf_counter() - self._window_start
        if elapsed < self.window / 6:
            return

        tokens_per_second = self._window_tokens / elapsed
        mfu = None
        if self.peak_flops:
            per_token = flops_per_token(self.config, self.seq_len)
            mfu = tokens_per_second * per_token / self.peak_flops

        self._last = {"tokens_per_second": tokens_per_second, "mfu": mfu}
        self._window_tokens = 0
        self._window_start = time.perf_counter()

    def report(self) -> dict[str, Any]:
        return dict(self._last)


def compute_perplexity(loss: float) -> float:
    """Perplexity from a loss value, clamped so a diverged run still prints."""
    return math.exp(min(loss, 20))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    max_batches: int | None = None,
) -> float:
    """Mean cross-entropy over ``dataloader``.

    Averaging is token-weighted, not batch-weighted: with variable-length
    batches a plain mean over batches quietly over-weights the short ones.
    """
    was_training = model.training
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    context = (
        torch.amp.autocast(device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )

    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in ("input_ids", "attention_mask", "labels")
        }
        num_tokens = (inputs["labels"][:, 1:] != -100).sum()
        if num_tokens == 0:
            continue
        with context:
            output = model(**inputs)
        total_loss += output["loss"].float() * num_tokens
        total_tokens += num_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        return float("nan")
    return (total_loss / total_tokens).item()
