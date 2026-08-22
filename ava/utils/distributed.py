"""Data-parallel setup helpers.

Nothing here is required for single-GPU work -- import it only when you launch
with ``torchrun``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed(backend: str | None = None) -> tuple[int, int, torch.device]:
    """Initialise the process group from ``torchrun``'s environment variables.

    Returns ``(rank, world_size, device)``. Safe to call in a single-process run:
    it detects the absence of ``torchrun`` and returns sensible defaults instead
    of failing.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def wrap_ddp(
    model: nn.Module,
    device: torch.device,
    static_graph: bool = True,
    gradient_as_bucket_view: bool = True,
) -> nn.Module:
    """Wrap in ``DistributedDataParallel`` if a process group is active.

    ``static_graph=True`` is the default, and it is not merely an optimisation.
    DDP normally marks each parameter ready exactly once per backward pass; with
    non-reentrant activation checkpointing a parameter is touched again during
    recomputation, and DDP raises *"Expected to mark a variable ready only
    once"*. Declaring the graph static tells DDP to record the reduction order
    on the first iteration and reuse it, which makes checkpointing work and
    removes a per-iteration bucket rebuild as a side effect.

    Ava's stacks have no data-dependent branching and no unused parameters, so
    the "static" claim holds. Pass ``static_graph=False`` if you wrap a model
    that does not share those properties.

    ``gradient_as_bucket_view=True`` lets gradients alias DDP's communication
    buckets instead of being copied into them -- one fewer full copy of the
    gradients in memory.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return model
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
