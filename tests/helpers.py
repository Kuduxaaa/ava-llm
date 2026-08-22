"""Shared builders for the test suite.

Kept out of ``conftest.py`` on purpose: conftest is pytest's fixture-discovery
hook, not an importable module, and importing across conftest files is exactly
the kind of thing that breaks when the rootdir changes.
"""

import torch
import torch.nn as nn

from ava import AvaConfig


def tiny(**overrides) -> AvaConfig:
    """A model small enough to run a full forward/backward in milliseconds."""
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        kv_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        d_state=4,
        d_conv=4,
        expand=2,
        ssm_chunk_size=8,
        num_attention_layers=2,
        z_loss_coef=0.0,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return AvaConfig(**defaults)


class ConstantHead(nn.Module):
    """An ``lm_head`` replacement that makes one token the argmax, always.

    Zeroing a real head and spiking one row does *not* do this: the logit for
    that row is ``1e3 * hidden.sum()``, so whenever the hidden state sums
    negative the spiked token becomes the *least* likely one and every other
    logit sits at a tied zero.
    """

    def __init__(self, vocab_size: int, token: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token = token
        self.weight = nn.Parameter(torch.zeros(vocab_size, 1), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = hidden_states.new_full((*hidden_states.shape[:-1], self.vocab_size), -1e4)
        logits[..., self.token] = 0.0
        return logits
