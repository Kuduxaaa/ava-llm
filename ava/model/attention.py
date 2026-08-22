"""Grouped-query attention with RoPE and optional QK-norm."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AvaConfig
from .cache import AttentionLayerCache
from .embeddings import apply_rotary_pos_emb
from .normalization import AvaRMSNorm


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(b, kv_heads, seq, dim)`` to ``(b, kv_heads * n_rep, seq, dim)``.

    ``expand`` allocates nothing; the ``reshape`` that follows does, but only
    because SDPA wants contiguous heads. On PyTorch builds where SDPA accepts
    ``enable_gqa`` this whole function is bypassed.
    """
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(
        batch, kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


_GQA_FUSED_SUPPORT: dict[str, bool] = {}


def sdpa_fused_supports_gqa(device: torch.device) -> bool:
    """Can a *fused* SDPA kernel take mismatched query and key head counts?

    Checking that ``enable_gqa=True`` does not raise is not enough. On builds
    where no fused kernel supports it the call still succeeds and silently
    drops to the math backend, which materialises the full attention matrix --
    measured at roughly 13x slower for a decode step than repeating the KV
    heads and letting the memory-efficient kernel run. So the probe excludes
    the math backend and asks whether anything fused is left.

    The answer is cached per device type and computed on first use, not at
    import, so importing ``ava`` never initialises CUDA.
    """
    key = device.type
    if key in _GQA_FUSED_SUPPORT:
        return _GQA_FUSED_SUPPORT[key]

    supported = False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        fused = [
            getattr(SDPBackend, name)
            for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION")
            if hasattr(SDPBackend, name)
        ]
        dtype = torch.float16 if key == "cuda" else torch.float32
        query = torch.zeros(1, 2, 1, 16, device=device, dtype=dtype)
        key_value = torch.zeros(1, 1, 1, 16, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sdpa_kernel(fused):
                F.scaled_dot_product_attention(query, key_value, key_value, enable_gqa=True)
        supported = True
    except Exception:
        supported = False

    _GQA_FUSED_SUPPORT[key] = supported
    return supported


class AvaAttention(nn.Module):
    """Multi-head attention with grouped key/value heads.

    Two properties are worth calling out because they are the usual source of
    "training works, generation is gibberish" bugs:

    1. RoPE is applied to the *new* keys only, **before** they enter the cache.
       Cached keys therefore keep the rotation of the absolute position they
       were produced at, and are never rotated a second time.
    2. Positions come from explicit ``position_ids``, not from a tensor shape,
       so left-padded batches and cached decoding both get the right angles.
    """

    def __init__(self, config: AvaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_heads = config.kv_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        if config.qk_norm:
            self.q_norm = AvaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
            self.k_norm = AvaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        layer_cache: AttentionLayerCache | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(
            batch, seq_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(batch, seq_len, self.kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(
            batch, seq_len, self.kv_heads, self.head_dim
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Cache stores post-RoPE keys, so nothing is ever rotated twice.
        if layer_cache is not None:
            key, value = layer_cache.update(key, value)

        dropout_p = self.attention_dropout if self.training else 0.0

        if output_attentions:
            return self._eager_attention(query, key, value, attention_mask, batch, seq_len)

        if self.num_key_value_groups > 1 and sdpa_fused_supports_gqa(query.device):
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=attention_mask is None and seq_len > 1,
                enable_gqa=True,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(value, self.num_key_value_groups),
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=attention_mask is None and seq_len > 1,
            )

        attn_output = attn_output.transpose(1, 2).reshape(
            batch, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output), None

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        batch: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialised attention -- only used when the weights are requested."""
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key.shape[2]]
        else:
            causal = torch.triu(
                torch.full(
                    (seq_len, key.shape[2]),
                    torch.finfo(scores.dtype).min,
                    device=scores.device,
                    dtype=scores.dtype,
                ),
                diagonal=key.shape[2] - seq_len + 1,
            )
            scores = scores + causal

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = self.dropout(probs)
        attn_output = torch.matmul(probs, value)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output), probs

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, kv_heads={self.kv_heads}, "
            f"head_dim={self.head_dim}, qk_norm={self.q_norm is not None}"
        )


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    seq_len: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Additive attention bias, or ``None`` when the fast causal path applies.

    Returning ``None`` matters for throughput: it lets SDPA take the fused
    ``is_causal`` kernel instead of reading a materialised
    ``(batch, 1, seq, kv_len)`` bias. That is why an all-ones padding mask is
    treated as "no mask at all".

    Rows are also guaranteed never to be fully masked. A fully masked row makes
    softmax produce NaN, which is the classic way left-padded batches poison a
    whole training run.
    """
    total_len = past_length + seq_len

    if attention_mask is not None and attention_mask.all():
        attention_mask = None

    if attention_mask is None and seq_len == 1:
        return None  # single decode step attends to everything cached
    if attention_mask is None:
        return None  # SDPA handles the causal triangle itself

    min_value = torch.finfo(dtype).min
    causal = (
        torch.arange(total_len, device=device)[None, :]
        > (torch.arange(past_length, total_len, device=device)[:, None])
    )
    mask = torch.zeros(seq_len, total_len, dtype=dtype, device=device)
    mask.masked_fill_(causal, min_value)
    mask = mask[None, None].expand(batch, 1, seq_len, total_len).clone()

    padding = attention_mask[:, None, None, :].to(device)
    mask.masked_fill_(padding == 0, min_value)

    # Un-mask the diagonal so no query row is left with an all -inf row.
    diagonal = torch.arange(seq_len, device=device)
    mask[:, :, diagonal, past_length + diagonal] = 0.0
    return mask
