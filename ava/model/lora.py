"""Low-rank adaptation (Hu et al., 2021)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable rank-``r`` update.

    ``B`` starts at zero, so the adapter is an exact no-op at step 0 and the
    wrapped model's outputs are bit-identical until training moves it.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        device, dtype = base_layer.weight.device, base_layer.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self._merged:
            return result
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + update * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight and return the plain layer.

        Merged inference costs exactly what the original model cost -- no extra
        matmul, no extra memory.
        """
        if not self._merged:
            self.base_layer.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
            self._merged = True
        self.base_layer.weight.requires_grad_(True)
        return self.base_layer

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def _iter_target_linears(
    model: nn.Module, target_modules: Sequence[str]
) -> list[tuple[nn.Module, str, nn.Linear]]:
    """Collect ``(parent, attribute, layer)`` triples before mutating anything.

    Materialising the list first matters: ``named_modules()`` is a live
    generator over the module tree, and swapping layers while it walks is how
    you get silently skipped -- or doubly wrapped -- modules.
    """
    found = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and name in target_modules:
                found.append((parent, name, child))
    return found


def apply_lora(
    model: nn.Module,
    target_modules: Sequence[str] = DEFAULT_TARGETS,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """Freeze ``model`` and attach LoRA adapters to the named linear layers.

    Matching is on the attribute name (``q_proj``), not a substring of the full
    dotted path, so ``o_proj`` can never accidentally capture ``out_proj``.
    """
    targets = _iter_target_linears(model, tuple(target_modules))
    if not targets:
        raise ValueError(
            f"No nn.Linear modules named {tuple(target_modules)} were found. "
            f"Available leaf names: "
            f"{sorted({n for _, n, _ in _all_linears(model)})[:20]}"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    for parent, name, layer in targets:
        setattr(parent, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)

    return model


def _all_linears(model: nn.Module) -> list[tuple[nn.Module, str, nn.Linear]]:
    return [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear)
    ]


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter back into its base layer, in place."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter weights -- a few MB instead of a full checkpoint."""
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


# Backwards-compatible alias for the pre-0.2 name.
apply_lora_to_model = apply_lora
LoRALayer = LoRALinear
