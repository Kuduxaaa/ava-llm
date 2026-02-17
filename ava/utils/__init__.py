import torch

from .custumize import print_ava_ascii
from .data_utils import prepare_data_from_json


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0) -> torch.Tensor:

    frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    time_steps = torch.arange(end, device=frequencies.device)

    return torch.polar(
        torch.ones_like(time_steps), torch.outer(time_steps, frequencies).float()
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


__all__ = ["prepare_data_from_json", "print_ava_ascii", "collate_fn"]
