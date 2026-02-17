# Architecture

## Overview

Ava supports three architecture types, selectable via `AvaConfig.architecture_type`:

| Type | Layers | Attention | RoPE | KV Cache |
|------|--------|-----------|------|----------|
| `transformer` | All AvaDecoderLayer | Yes | Yes | Yes |
| `mamba` | All MambaBlock | No | No | No |
| `hybrid` | MambaBlock + AvaDecoderLayer | Last N layers | Last N layers | Partial |

## Transformer Architecture

Standard decoder-only Transformer:

```
Input → Embedding → [DecoderLayer × N] → RMSNorm → LM Head → Logits
```

Each `AvaDecoderLayer`:
- Pre-RMSNorm → Multi-Head Attention (GQA) → Residual
- Pre-RMSNorm → SwiGLU MLP → Residual

Features:
- **Rotary Position Embeddings (RoPE)** — Relative positional encoding
- **Grouped Query Attention (GQA)** — Configurable KV heads for memory efficiency
- **SwiGLU Activation** — gate_proj * SiLU(up_proj) → down_proj

## Mamba-2 SSM Architecture

State Space Model with selective scan mechanism:

```
Input → Embedding → [MambaBlock × N] → RMSNorm → LM Head → Logits
```

Each `MambaBlock`:
- Pre-RMSNorm → SelectiveSSM → Residual

### Selective SSM Details

The SelectiveSSM processes sequences through a recurrent state-space formulation:

```
h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
y[t] = C[t] @ h[t]
```

Where:
- **A_bar** = exp(dt * A) — Discretized state transition
- **B_bar** = dt * B — Discretized input matrix
- **dt** — Input-dependent delta time (selective mechanism)
- **A** — Learnable in log-space for stability

Data flow in SelectiveSSM:
1. `in_proj`: d_model → inner_dim * 2 (x branch + gate z)
2. `conv1d`: Causal depthwise convolution (kernel=d_conv)
3. `x_proj`: inner_dim → dt + B + C (SSM parameters)
4. Selective scan (sequential loop)
5. Skip connection with learnable D
6. Gating: y * SiLU(z)
7. `out_proj`: inner_dim → d_model

### Why Mamba?

- **O(n) complexity** vs O(n^2) for attention — handles long sequences efficiently
- **No KV cache** needed — constant memory during generation
- **Hardware friendly** — Sequential scan works well on GPUs without custom CUDA kernels

## Hybrid Architecture (Recommended)

Combines Mamba SSM for bulk processing with Attention for final refinement:

```
Input → Embedding → [MambaBlock × M] → [DecoderLayer × A] → RMSNorm → LM Head
```

Default `hybrid-130m`: 10 Mamba blocks + 2 Attention layers

This approach:
- Uses Mamba for efficient sequence modeling in early layers
- Uses Attention in final layers for precise token interactions
- RoPE and causal mask only apply to attention layers
- Achieves near-Transformer quality with significantly lower compute

## Parameter Counts

For `hybrid-130m` (hidden=768, 12 layers):
- Embedding: vocab_size × 768
- 10 × MambaBlock: ~8M each
- 2 × DecoderLayer: ~9M each
- Total: ~115-130M (depending on vocab size)
