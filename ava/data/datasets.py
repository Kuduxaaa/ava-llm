"""Datasets for pretraining and instruction tuning."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .packing import load_packed, pack_corpus


class PackedDataset(Dataset):
    """Fixed-size blocks cut from a memory-mapped token stream.

    Every returned block is ``block_size`` real tokens -- no padding, no
    attention mask, nothing wasted. ``labels`` is just a copy of ``input_ids``;
    the model does the causal shift internally.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        block_size: int = 1024,
        stride: int | None = None,
    ) -> None:
        self.tokens, self.corpus = load_packed(path)
        self.block_size = block_size
        self.stride = stride or block_size

        usable = self.tokens.size - 1  # need one extra token for the shift
        if usable < block_size:
            raise ValueError(
                f"Corpus has {self.tokens.size} tokens but block_size is "
                f"{block_size}; pack more data or lower block_size."
            )
        self.num_blocks = 1 + (usable - block_size) // self.stride

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.stride
        window = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
        input_ids = torch.from_numpy(window)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        tokenizer,
        output_path: str | os.PathLike,
        block_size: int = 1024,
        **pack_kwargs,
    ) -> PackedDataset:
        pack_corpus(texts, tokenizer, output_path, **pack_kwargs)
        return cls(output_path, block_size=block_size)

    def split(self, val_fraction: float = 0.01) -> tuple[PackedDataset, PackedDataset]:
        """Contiguous train/val split by block index, no shuffling of the file."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        num_val = max(1, int(self.num_blocks * val_fraction))
        return (
            _BlockSubset(self, range(0, self.num_blocks - num_val)),
            _BlockSubset(self, range(self.num_blocks - num_val, self.num_blocks)),
        )


class _BlockSubset(Dataset):
    """A contiguous slice of a :class:`PackedDataset`."""

    def __init__(self, base: PackedDataset, indices: range) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base[self.indices[index]]


class ChatDataset(Dataset):
    """Multi-turn conversations with loss masked to assistant turns only.

    Label masking is computed by encoding each turn **separately** and tracking
    token counts as the sequence is built. The alternative -- encoding a prefix
    of the joined string and using its length as a token offset -- is quadratic
    *and* wrong, because a subword tokenizer does not guarantee that the
    encoding of a prefix is a prefix of the encoding.
    """

    def __init__(
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        tokenizer,
        max_length: int = 1024,
        train_on_last_turn_only: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict[str, torch.Tensor]] = []

        for conversation in conversations:
            example = self._build(conversation, train_on_last_turn_only)
            if example is not None:
                self.examples.append(example)

    def _build(
        self, conversation: Sequence[dict[str, str]], last_turn_only: bool
    ) -> dict[str, torch.Tensor] | None:
        tokenizer = self.tokenizer
        input_ids: list[int] = []
        labels: list[int] = []

        if tokenizer.bos_token_id >= 0:
            input_ids.append(tokenizer.bos_token_id)
            labels.append(-100)

        assistant_turns = [
            i for i, m in enumerate(conversation) if m.get("role") == "assistant"
        ]
        if not assistant_turns:
            return None
        supervised = {assistant_turns[-1]} if last_turn_only else set(assistant_turns)

        for index, message in enumerate(conversation):
            role, content = message.get("role"), message.get("content")
            if role is None or content is None:
                continue

            header = tokenizer.encode(f"<|{role}|>\n", add_special_tokens=False)
            body = tokenizer.encode(f"{content}\n", add_special_tokens=False)

            input_ids.extend(header)
            labels.extend([-100] * len(header))

            input_ids.extend(body)
            # The header is a prompt, the body is the target. Supervising the
            # header would teach the model to emit role markers on its own.
            labels.extend(body if index in supervised else [-100] * len(body))

        if tokenizer.eos_token_id >= 0:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        if all(label == -100 for label in labels):
            return None  # nothing to learn from; drop rather than emit NaN loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class PaddingCollator:
    """Pad a batch to its own longest sequence, not to a global maximum.

    Batch-local padding is why this is a collator rather than something the
    dataset does: the dataset cannot know what it will be batched with.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        longest = max(item["input_ids"].numel() for item in batch)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            longest = -(-longest // multiple) * multiple

        input_ids, labels, masks = [], [], []
        for item in batch:
            ids = item["input_ids"]
            pad_len = longest - ids.numel()
            input_ids.append(torch.cat([ids, ids.new_full((pad_len,), self.pad_token_id)]))
            item_labels = item.get("labels", ids)
            labels.append(torch.cat([item_labels, item_labels.new_full((pad_len,), -100)]))
            masks.append(torch.cat([torch.ones_like(ids), ids.new_zeros(pad_len)]))

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(masks),
        }


def collate_packed(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collator for :class:`PackedDataset` -- every block is already full length."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
