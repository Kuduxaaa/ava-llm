"""Human-readable model summaries and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch.

    ``deterministic=True`` also pins cuDNN algorithm selection, which makes runs
    bit-reproducible at a real throughput cost -- use it to chase a bug, not for
    production training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def format_parameters(count: int) -> str:
    """``1_234_567`` -> ``1.23M``."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= threshold:
            return f"{count / threshold:.2f}{suffix}"
    return str(count)


def model_summary(model: nn.Module, dtype_bytes: int = 2) -> str:
    """A short report: parameter counts, layer mix and estimated memory."""
    from ..model.ava_model import AvaForCausalLM

    lines = []
    total = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    trainable = sum(
        p.numel()
        for p in {id(p): p for p in model.parameters()}.values()
        if p.requires_grad
    )

    lines.append(f"parameters      {format_parameters(total)} ({total:,})")
    if trainable != total:
        lines.append(f"trainable       {format_parameters(trainable)} ({trainable:,})")

    if isinstance(model, AvaForCausalLM):
        config = model.config
        layer_types = config.layer_types()
        lines.append(f"architecture    {config.architecture_type}")
        lines.append(
            f"layers          {len(layer_types)} "
            f"({layer_types.count('attention')} attention, "
            f"{layer_types.count('mamba')} mamba)"
        )
        lines.append(f"hidden / ffn    {config.hidden_size} / {config.intermediate_size}")
        lines.append(
            f"heads           {config.num_attention_heads} q, {config.kv_heads} kv, "
            f"dim {config.head_dim}"
        )
        lines.append(f"context         {config.max_position_embeddings}")
        lines.append(f"vocab           {config.vocab_size:,}")
        lines.append(f"tied embeddings {config.tie_word_embeddings}")

    weights_gb = total * dtype_bytes / 1e9
    # AdamW keeps two fp32 moments plus an fp32 master copy of each parameter.
    training_gb = weights_gb + total * 12 / 1e9
    lines.append(f"weights         {weights_gb:.2f} GB @ {dtype_bytes * 8}-bit")
    lines.append(f"training state  ~{training_gb:.2f} GB (weights + AdamW moments)")

    return "\n".join(lines)
