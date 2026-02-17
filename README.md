# Ava Language Model

**Ava** is a custom-designed language model framework supporting **Transformer**, **Mamba SSM**, and **Hybrid** architectures. Built from scratch in PyTorch for experimentation, education, and low-resource language modeling — particularly **Georgian** (ქართული).

---

## Architecture

Ava supports three architecture modes via `AvaConfig.architecture_type`:

| Mode | Description | Complexity |
|------|------------|------------|
| **Transformer** | Standard decoder with RoPE, GQA, SwiGLU | O(n^2) |
| **Mamba** | Pure Selective State Space Model | O(n) |
| **Hybrid** | Mamba layers + Attention layers (recommended) | O(n) + O(n^2) for last layers |

### Hybrid Architecture (Recommended)

```
Input → Embedding → [MambaBlock × 10] → [DecoderLayer × 2] → RMSNorm → LM Head
```

- Mamba blocks handle bulk sequence processing efficiently
- Attention layers at the end provide precise token interactions
- ~130M parameters, fits on Colab T4 (15GB VRAM)

## Quick Start

```python
from ava import AvaConfig, AvaForCausalLM

# Create a hybrid Mamba-2 model (~130M params)
config = AvaConfig().apply_for("hybrid-130m")
model = AvaForCausalLM(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Or use pure transformer (backward compatible)
config = AvaConfig().apply_for("100m")
model = AvaForCausalLM(config)
```

## Training

```python
from ava.training import TrainingConfig, train_model
from ava.data import AvaDataset
from torch.utils.data import DataLoader

# Prepare dataset
dataset = AvaDataset(data, tokenizer, max_length=512)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train with AMP and gradient accumulation
training_config = TrainingConfig(
    num_epochs=3,
    learning_rate=5e-4,
    use_amp=True,
    gradient_accumulation_steps=8,
)

model, history = train_model(model, loader, training_config=training_config)
```

## Features

- **Hybrid Mamba-2 Architecture** — Attention-free bulk processing + attention for refinement
- **Selective State Space Model** — Pure PyTorch, no custom CUDA kernels
- **Mixed Precision Training** — AMP for ~2x speedup and memory savings
- **Gradient Accumulation** — Large effective batch sizes on limited hardware
- **Cosine LR Schedule** — With linear warmup
- **Label Masking** — Loss computed only on assistant responses
- **Multi-turn Conversations** — Full conversation history utilization
- **Rotary Position Embeddings** — RoPE for attention layers
- **Grouped Query Attention** — Configurable KV heads
- **LoRA Fine-tuning** — Parameter-efficient adaptation
- **8-bit Quantization** — For low-memory inference

## Model Presets

| Preset | Architecture | Params | Use Case |
|--------|-------------|--------|----------|
| `hybrid-130m` | Hybrid | ~130M | Georgian LM, T4 training |
| `mamba-130m` | Mamba | ~130M | Long sequences, efficient inference |
| `100m` | Transformer | ~100M | Small experiments |
| `500m`–`100b` | Transformer | 500M–100B | Various scales |

## Documentation

See the [`docs/`](docs/) folder:
- [Architecture](docs/architecture.md) — Transformer vs Mamba vs Hybrid
- [Configuration](docs/configuration.md) — Full AvaConfig reference
- [Training](docs/training.md) — Training guide with T4 memory tips
- [Georgian](docs/georgian.md) — Georgian language specifics

## Citations

- [Vaswani et al. (2017). *Attention is All You Need.*](https://arxiv.org/abs/1706.03762)
- [Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*](https://arxiv.org/abs/2312.00752)
- [Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*](https://arxiv.org/abs/2106.09685)

## Author

Created and maintained by Nika Kudukhashvili. [GitHub](https://github.com/Kuduxaaa)
