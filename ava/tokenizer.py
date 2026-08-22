"""A small, dependency-light tokenizer wrapper around SentencePiece.

Language-agnostic by construction: nothing here assumes a script, a direction,
or a segmentation strategy. Train it on whatever corpus you have.

    from ava.tokenizer import AvaTokenizer

    AvaTokenizer.train("corpus.txt", "data/tokenizer", vocab_size=32000)
    tokenizer = AvaTokenizer.from_pretrained("data/tokenizer")

    batch = tokenizer(
        ["first document", "second one"],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from typing import Any, Literal, overload

import torch

try:  # pragma: no cover - exercised only when the extra is missing
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "AvaTokenizer needs sentencepiece. Install it with: pip install 'ava-llm[tokenizer]'"
    ) from exc

MODEL_FILENAME = "tokenizer.model"
# 256 byte-fallback pieces + <pad>/<s>/</s>/<unk>.
MIN_BYTE_FALLBACK_VOCAB = 260
CONFIG_FILENAME = "tokenizer_config.json"

PaddingSide = Literal["left", "right"]


class AvaTokenizer:
    """SentencePiece with the parts a training loop actually needs.

    Two behaviours are worth stating explicitly because getting them wrong is
    quiet and expensive:

    * ``__call__`` and :meth:`encode` add the same special tokens by default.
      When they disagree, a model trains on sequences that never contain EOS
      and then never learns to stop generating.
    * Truncation keeps the closing EOS. A truncated sequence that lost its EOS
      teaches the model that long documents simply do not end.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        add_bos: bool = True,
        add_eos: bool = True,
        padding_side: PaddingSide = "right",
    ) -> None:
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.padding_side = padding_side

        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()
        pad_id = self.sp.pad_id()
        # SentencePiece returns -1 when no pad piece was reserved at training
        # time; padding with -1 would index out of the embedding table.
        self.pad_token_id = pad_id if pad_id >= 0 else self.eos_token_id

    # --- basics ---

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
            if token_id >= 0
        }

    def tokenize(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def id_to_token(self, token_id: int) -> str:
        return self.sp.id_to_piece(token_id)

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    # --- encoding ---

    def _wrap(self, ids: list[int], add_special_tokens: bool) -> list[int]:
        if not add_special_tokens:
            return ids
        if self.add_bos and self.bos_token_id >= 0:
            ids = [self.bos_token_id, *ids]
        if self.add_eos and self.eos_token_id >= 0:
            ids = [*ids, self.eos_token_id]
        return ids

    def _truncate(
        self, ids: list[int], max_length: int, add_special_tokens: bool
    ) -> list[int]:
        if len(ids) <= max_length:
            return ids
        ids = ids[:max_length]
        if add_special_tokens and self.add_eos and self.eos_token_id >= 0:
            ids[-1] = self.eos_token_id
        return ids

    @overload
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    @overload
    def encode(self, text: Sequence[str], **kwargs: Any) -> list[list[int]]: ...

    def encode(
        self,
        text: str | Sequence[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> Any:
        """Text to ids. Accepts a single string or a batch of them."""
        if isinstance(text, str):
            ids = self._wrap(self.sp.encode(text), add_special_tokens)
            if truncation and max_length:
                ids = self._truncate(ids, max_length, add_special_tokens)
            return torch.tensor([ids]) if return_tensors == "pt" else ids

        return [
            self.encode(
                item,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for item in text
        ]

    def decode(
        self, ids: torch.Tensor | Sequence[int], skip_special_tokens: bool = True
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            raise ValueError("Use batch_decode() for a batch of sequences.")
        if skip_special_tokens:
            special = self.special_token_ids
            ids = [i for i in ids if i not in special]
        return self.sp.decode(list(ids))

    def batch_decode(
        self,
        sequences: torch.Tensor | Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def __call__(
        self,
        text: str | Sequence[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        padding_side: PaddingSide | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Encode with padding and an attention mask, HF-style.

        ``padding=True`` pads to the longest sequence in the batch;
        ``padding="max_length"`` pads to ``max_length``. Use
        ``padding_side="left"`` before generation so every sequence's last
        token really is its last token.
        """
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        side = padding_side or self.padding_side

        batch = [
            self.encode(
                item,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for item in texts
        ]
        masks = [[1] * len(ids) for ids in batch]

        if padding:
            if padding == "max_length":
                if not max_length:
                    raise ValueError('padding="max_length" requires max_length.')
                target = max_length
            else:
                target = max(len(ids) for ids in batch)

            for i, ids in enumerate(batch):
                pad_len = target - len(ids)
                if pad_len <= 0:
                    continue
                pad = [self.pad_token_id] * pad_len
                if side == "left":
                    batch[i] = pad + ids
                    masks[i] = [0] * pad_len + masks[i]
                else:
                    batch[i] = ids + pad
                    masks[i] = masks[i] + [0] * pad_len

        if return_tensors == "pt":
            lengths = {len(ids) for ids in batch}
            if len(lengths) > 1:
                raise ValueError(
                    "Cannot build a tensor from ragged sequences; pass padding=True."
                )
            return {
                "input_ids": torch.tensor(batch, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        if single:
            return {"input_ids": batch[0], "attention_mask": masks[0]}
        return {"input_ids": batch, "attention_mask": masks}

    # --- chat ---

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Render a message list into the plain-text format Ava trains on.

        Deliberately minimal and role-agnostic -- swap it for your own scheme,
        but keep training and inference on the *same* one.
        """
        parts = []
        for message in messages:
            role = message["role"]
            parts.append(f"<|{role}|>\n{message['content']}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    # --- persistence ---

    def save_pretrained(self, path: str | os.PathLike) -> None:
        os.makedirs(path, exist_ok=True)
        destination = os.path.join(path, MODEL_FILENAME)
        if os.path.abspath(self.model_path) != os.path.abspath(destination):
            shutil.copy(self.model_path, destination)

        config = {
            "model_file": MODEL_FILENAME,
            "vocab_size": len(self),
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "padding_side": self.padding_side,
        }
        with open(os.path.join(path, CONFIG_FILENAME), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike, **overrides: Any) -> AvaTokenizer:
        path = str(path)
        if os.path.isfile(path):
            return cls(path, **overrides)

        config_path = os.path.join(path, CONFIG_FILENAME)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No {CONFIG_FILENAME} in {path}.")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        kwargs = {
            key: config[key]
            for key in ("add_bos", "add_eos", "padding_side")
            if key in config
        }
        kwargs.update(overrides)
        return cls(os.path.join(path, config["model_file"]), **kwargs)

    # --- training ---

    @classmethod
    def train(
        cls,
        corpus: str | os.PathLike | Sequence[str],
        output_dir: str | os.PathLike,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        max_sentence_length: int = 16384,
        num_threads: int = os.cpu_count() or 4,
        byte_fallback: bool = True,
        input_sentence_size: int = 2_000_000,
        **trainer_kwargs: Any,
    ) -> AvaTokenizer:
        """Train a SentencePiece model and return the loaded tokenizer.

        ``character_coverage`` is the one knob that is genuinely script-
        dependent: 1.0 for alphabets with a small character set, ~0.9995 when
        the corpus has a long tail of rare characters (CJK, mixed-script web
        text). Ids 0-3 are pinned to pad/bos/eos/unk so a checkpoint's special
        tokens stay stable across retrains.

        ``byte_fallback`` reserves 256 pieces for raw UTF-8 bytes so no input is
        ever unrecoverable. Those pieces come out of ``vocab_size``, which is
        why a byte-fallback tokenizer needs a few hundred slots before it can
        hold a single merge.

        ``input_sentence_size`` caps how many sentences the trainer holds in
        memory, sampling uniformly across the file. SentencePiece otherwise
        loads the *entire* corpus, which is fine for a few hundred megabytes and
        fatal for the tens of gigabytes a real pretraining corpus runs to. Two
        million sentences is far more than a 32k vocabulary can use; raising it
        buys nothing but RAM.
        """
        if byte_fallback and vocab_size < MIN_BYTE_FALLBACK_VOCAB:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for byte fallback: 256 byte "
                "pieces plus <pad>/<s>/</s>/<unk> already need "
                f"{MIN_BYTE_FALLBACK_VOCAB} slots, before a single character of "
                "the corpus is covered. Raise vocab_size (>= 512 is a sane floor) "
                "or pass byte_fallback=False."
            )

        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(str(output_dir), "tokenizer")

        inputs = corpus if isinstance(corpus, (str, os.PathLike)) else ",".join(corpus)

        try:
            spm.SentencePieceTrainer.train(
                input=str(inputs),
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                max_sentence_length=max_sentence_length,
                num_threads=num_threads,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                pad_piece="<pad>",
                bos_piece="<s>",
                eos_piece="</s>",
                unk_piece="<unk>",
                byte_fallback=byte_fallback,
                input_sentence_size=input_sentence_size,
                shuffle_input_sentence=True,
                train_extremely_large_corpus=vocab_size > 64000,
                **trainer_kwargs,
            )
        except RuntimeError as exc:
            # SentencePiece reports this as an INTERNAL assertion failure with a
            # C++ file and line number, which tells you nothing about what to
            # change. Translate the one failure people actually hit.
            message = str(exc)
            if "required_chars" in message:
                raise ValueError(
                    f"vocab_size={vocab_size} cannot cover this corpus: the "
                    "alphabet alone (plus reserved and byte-fallback pieces) "
                    "needs more slots. Raise vocab_size, or lower "
                    f"character_coverage below {character_coverage} so the rare "
                    "character tail falls through to byte fallback."
                ) from exc
            if "Vocabulary size too high" in message:
                ceiling = message.rsplit("<=", 1)[-1].strip().rstrip(".")
                raise ValueError(
                    f"vocab_size={vocab_size} is larger than this corpus can "
                    f"support; SentencePiece can only form {ceiling} pieces from "
                    "it. Use a bigger corpus, or lower vocab_size."
                ) from exc
            raise

        tokenizer = cls(f"{prefix}.model")
        tokenizer.save_pretrained(output_dir)
        return tokenizer
