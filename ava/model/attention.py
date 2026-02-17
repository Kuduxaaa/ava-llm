import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ava.config import AvaConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, :, : q.shape[2], :]
    sin = sin[:, :, : q.shape[2], :]

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed


class AvaAttention(nn.Module):
    def __init__(self, config: AvaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.kv_heads = getattr(config, "kv_heads", self.num_heads)
        self.kv_dim = self.head_dim * self.kv_heads
        self._dropout_p = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        rotary_emb=None,
        position_ids=None,
    ):
        B, T, _ = hidden_states.shape

        query = (
            self.q_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(B, T, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(B, T, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        past_key_value = (key, value) if use_cache else None

        if rotary_emb is not None:
            cos, sin = rotary_emb(query, seq_len=query.shape[2])
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        if output_attentions:
            # Manual path — needed for returning attention weights
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_probs = self.dropout(attn_probs)
            context = torch.matmul(attn_probs, value)
            context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
            return self.o_proj(context), past_key_value, attn_probs

        # Fast SDPA path — uses FlashAttention / MemEfficientAttention on GPU
        dp = self._dropout_p if self.training else 0.0
        if attention_mask is not None:
            context = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=dp,
            )
        else:
            context = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, dropout_p=dp,
            )

        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(context), past_key_value
