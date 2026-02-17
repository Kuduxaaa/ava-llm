# Ava-LLM Documentation

## Table of Contents

- [Architecture](architecture.md) — Transformer, Mamba SSM, and Hybrid architectures
- [Configuration](configuration.md) — AvaConfig reference, presets, and Mamba parameters
- [Training](training.md) — TrainingConfig, AMP, gradient accumulation, T4 memory guide
- [Georgian Language](georgian.md) — Georgian language specifics, tokenizer recommendations

## Project Overview

Ava is a custom-built language model framework supporting three architecture modes:

1. **Transformer** — Standard decoder-only Transformer with RoPE, GQA, and SwiGLU
2. **Mamba** — Pure Selective State Space Model (SSM) with no attention
3. **Hybrid** — Mamba SSM layers followed by Attention layers (recommended)

The framework is designed to run on low-end hardware (Colab T4, 15GB VRAM) and is particularly suited for Georgian language modeling.

## Quick Start

```python
from ava import AvaConfig, AvaForCausalLM

# Hybrid Mamba-2 model (~130M params)
config = AvaConfig().apply_for("hybrid-130m")
model = AvaForCausalLM(config)

# Or pure transformer (backward compatible)
config = AvaConfig().apply_for("100m")
model = AvaForCausalLM(config)
```
