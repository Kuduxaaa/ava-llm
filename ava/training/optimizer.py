"""Optimizers and learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _split_decay_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight-decay every matrix, none of the vectors.

    Decaying a norm gain or a bias pulls it toward zero for no benefit; the
    convention of splitting on ``ndim >= 2`` is both simpler and more reliable
    than name matching, which breaks the moment a layer is renamed.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.ndim >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = None,
) -> Optimizer:
    """AdamW with sane LLM defaults and the fused kernel when it is available."""
    groups = _split_decay_groups(model, weight_decay)

    if fused is None:
        fused = torch.cuda.is_available() and all(
            p.is_cuda for group in groups for p in group["params"]
        )

    try:
        return AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
    except (RuntimeError, ValueError):
        # Older builds, or a CPU/meta parameter slipping into a fused group.
        return AdamW(groups, lr=lr, betas=betas, eps=eps)


@torch.no_grad()
def _newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a matrix via a quintic Newton-Schulz iteration.

    The coefficients are the ones tuned by Jordan et al. for Muon: they do not
    converge to the exact polar factor, but they push every singular value into
    roughly ``[0.7, 1.3]`` in five steps, which is all the update direction
    needs and costs a handful of matmuls instead of an SVD.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    x = x / (x.norm() + eps)

    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * gram @ gram
        x = a * x + correction @ x

    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(Optimizer):
    """Momentum with orthogonalised updates, for 2D hidden weights only.

    Muon (Jordan et al., 2024) replaces the raw momentum buffer with its nearest
    semi-orthogonal matrix. In practice it reaches a given loss in noticeably
    fewer tokens than AdamW on the hidden layers of a transformer -- but it is
    defined for matrices, so embeddings, the LM head, biases and norm gains must
    stay on AdamW. :func:`create_hybrid_optimizer` wires that split for you.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "Muon only handles 2D parameters; route embeddings, "
                        "heads, biases and norms to AdamW instead."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - group["momentum"])

                update = (
                    grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                )
                update = _newton_schulz(update, steps=group["ns_steps"])

                # Keep the update's scale comparable across differently shaped
                # matrices, so one learning rate works for the whole stack.
                scale = max(1.0, param.size(0) / param.size(1)) ** 0.5

                if group["weight_decay"]:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"] * scale)

        return loss


def create_hybrid_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> tuple[Optimizer, Optimizer]:
    """Muon for the hidden matrices, AdamW for everything else.

    Returns both optimizers; :func:`ava.training.train_model` steps them
    together and drives them from a single schedule.
    """
    embedding_params = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embedding_params.add(id(module.weight))

    output_head = getattr(model, "lm_head", None)
    if output_head is not None:
        embedding_params.add(id(output_head.weight))

    muon_params, adamw_params = [], []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim == 2 and id(param) not in embedding_params:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
    adamw = AdamW(
        [
            {
                "params": [p for p in adamw_params if p.ndim >= 2],
                "weight_decay": weight_decay,
            },
            {"params": [p for p in adamw_params if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )
    return muon, adamw


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.02,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
    stable_ratio: float = 0.8,
) -> LambdaLR:
    """Learning-rate schedule with linear warmup.

    ``cosine``
        The familiar decay to ``min_lr_ratio``.
    ``wsd``
        Warmup-Stable-Decay: hold the peak rate for ``stable_ratio`` of the run,
        then decay. The flat middle means you can stop, add data, and continue
        without the schedule having already decayed away -- which is why it has
        displaced cosine for pretraining runs that grow over time.
    ``linear`` / ``constant``
        Baselines.
    """
    warmup_steps = int(num_training_steps * warmup_ratio)

    def cosine(progress: float) -> float:
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def wsd(progress: float) -> float:
        if progress < stable_ratio:
            return 1.0
        decay_progress = (progress - stable_ratio) / max(1e-8, 1 - stable_ratio)
        return min_lr_ratio + (1 - min_lr_ratio) * (1 - decay_progress)

    shapes = {
        "cosine": cosine,
        "wsd": wsd,
        "linear": lambda p: min_lr_ratio + (1 - min_lr_ratio) * (1 - p),
        "constant": lambda p: 1.0,
    }
    if schedule not in shapes:
        raise ValueError(f"Unknown schedule {schedule!r}. Available: {sorted(shapes)}")
    shape = shapes[schedule]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # (step + 1), not step: LambdaLR evaluates at 0 before the first
            # update, and a first step at lr=0 is a step thrown away.
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return shape(min(1.0, progress))

    return LambdaLR(optimizer, lr_lambda)
