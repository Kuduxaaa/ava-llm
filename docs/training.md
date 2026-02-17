# Training Guide

## TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 3 | Number of training epochs |
| `learning_rate` | 5e-4 | Peak learning rate |
| `weight_decay` | 0.1 | AdamW weight decay |
| `betas` | (0.9, 0.95) | AdamW beta parameters |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `use_amp` | True | Enable mixed precision training |
| `gradient_accumulation_steps` | 1 | Accumulation steps for effective batch size |
| `warmup_ratio` | 0.05 | Fraction of steps for LR warmup |
| `checkpoint_dir` | "checkpoints" | Directory for saving checkpoints |
| `log_interval` | 10 | Steps between log outputs |

## Mixed Precision Training (AMP)

AMP is enabled by default when training on CUDA. Benefits:
- ~2x speedup on Tensor Core GPUs (T4, V100, A100)
- ~2x memory reduction (float16/bfloat16 for forward/backward)
- Automatic loss scaling prevents underflow

```python
from ava.training import TrainingConfig, train_model

config = TrainingConfig(use_amp=True)
model, history = train_model(model, train_loader, training_config=config)
```

## Gradient Accumulation

Simulates larger batch sizes without increasing memory:

```python
# Effective batch size = batch_size * gradient_accumulation_steps
# Example: batch_size=4, accumulation=8 → effective batch = 32
config = TrainingConfig(gradient_accumulation_steps=8)
```

## LR Schedule

Uses cosine annealing with linear warmup:

1. **Warmup phase** (default 5% of steps): Linear increase from 0 to `learning_rate`
2. **Decay phase**: Cosine decay from `learning_rate` to 0

## Optimizer

AdamW with automatic weight decay exclusion:
- **Decayed**: Weight matrices (Linear layers)
- **Not decayed**: Biases, LayerNorm/RMSNorm parameters

## T4 (15GB VRAM) Memory Guide

Recommended settings for Colab T4:

| Model | Batch Size | Accumulation | Seq Length | Est. Memory |
|-------|-----------|--------------|------------|-------------|
| `hybrid-130m` | 4 | 8 | 512 | ~8GB |
| `hybrid-130m` | 2 | 16 | 1024 | ~12GB |
| `mamba-130m` | 8 | 4 | 512 | ~6GB |
| `100m` (transformer) | 4 | 8 | 512 | ~10GB |

Tips:
- Always use `use_amp=True` on T4
- Reduce `max_length` in dataset if OOM
- Use `gradient_accumulation_steps` to maintain effective batch size
- Mamba models use less memory than Transformers at same size

## Full Training Example

```python
from ava import AvaConfig, AvaForCausalLM
from ava.training import TrainingConfig, train_model
from ava.data import AvaDataset
from torch.utils.data import DataLoader

# Setup
config = AvaConfig().apply_for("hybrid-130m")
model = AvaForCausalLM(config)

# Dataset
dataset = AvaDataset(data, tokenizer, max_length=512)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train
training_config = TrainingConfig(
    num_epochs=3,
    learning_rate=5e-4,
    use_amp=True,
    gradient_accumulation_steps=8,
)

model, history = train_model(
    model, loader, device="cuda", training_config=training_config
)
```
