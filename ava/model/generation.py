"""Sampling configuration and logits processing for autoregressive decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Decoding parameters.

    ``max_new_tokens`` is the primary length control and means exactly what it
    says. ``max_length`` (prompt + completion) is honoured too, and whichever
    binds first wins -- there is no hidden cap on top of either.
    """

    max_new_tokens: int | None = 256
    max_length: int | None = None
    do_sample: bool = True

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    """Keep tokens whose probability is at least ``min_p`` times the most likely
    token's. Cheaper and better behaved than top-p at high temperature."""

    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None

    use_cache: bool = True
    seed: int | None = None

    _stop: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                "temperature must be > 0; use do_sample=False for greedy decoding."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError(f"min_p must be in [0, 1), got {self.min_p}.")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0.")

        if self.eos_token_id is None:
            self._stop = []
        elif isinstance(self.eos_token_id, int):
            self._stop = [self.eos_token_id]
        else:
            self._stop = list(self.eos_token_id)

    @property
    def stop_token_ids(self) -> list[int]:
        return self._stop

    def budget(self, prompt_length: int) -> int:
        """How many new tokens to generate for a prompt of this length."""
        limits = []
        if self.max_new_tokens is not None:
            limits.append(self.max_new_tokens)
        if self.max_length is not None:
            limits.append(max(0, self.max_length - prompt_length))
        if not limits:
            raise ValueError("Set max_new_tokens or max_length.")
        return min(limits)


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalise already-generated tokens (Keskar et al., 2019).

    Negative logits are *multiplied* and positive ones divided; dividing a
    negative logit would make the token more likely, which is the opposite of
    the intent and a common bug in hand-rolled samplers.
    """
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, generated)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits.scatter(1, generated, scores)


def ban_repeated_ngrams(
    logits: torch.Tensor, generated: torch.Tensor, ngram_size: int
) -> torch.Tensor:
    """Forbid any continuation that would repeat an existing n-gram."""
    if ngram_size <= 0 or generated.shape[1] < ngram_size:
        return logits

    batch_size = generated.shape[0]
    for i in range(batch_size):
        sequence = generated[i].tolist()
        prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned = {
            sequence[start + ngram_size - 1]
            for start in range(len(sequence) - ngram_size + 1)
            if tuple(sequence[start : start + ngram_size - 1]) == prefix
        }
        if banned:
            logits[i, list(banned)] = -float("inf")
    return logits


def filter_logits(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0
) -> torch.Tensor:
    """Apply top-k, min-p and nucleus (top-p) truncation, in that order."""
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -float("inf"))

    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        threshold = min_p * probs.amax(dim=-1, keepdim=True)
        logits = logits.masked_fill(probs < threshold, -float("inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Shift so the token that crosses the threshold is itself kept.
        remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
        remove[..., 0] = False
        logits = logits.masked_fill(
            remove.scatter(-1, sorted_indices, remove), -float("inf")
        )

    return logits


def select_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generated: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Turn a row of logits into one token id per sequence."""
    logits = logits.float()
    logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)
    logits = ban_repeated_ngrams(logits, generated, config.no_repeat_ngram_size)

    if not config.do_sample:
        return logits.argmax(dim=-1)

    logits = logits / config.temperature
    logits = filter_logits(logits, config.top_k, config.top_p, config.min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
