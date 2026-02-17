# Configuration Reference

## AvaConfig

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 32000 | Vocabulary size |
| `hidden_size` | 2048 | Model dimension |
| `intermediate_size` | 8192 | MLP intermediate dimension |
| `num_hidden_layers` | 16 | Total number of layers |
| `num_attention_heads` | 16 | Number of attention heads |
| `hidden_act` | "silu" | Activation function |
| `max_position_embeddings` | 2048 | Maximum sequence length |
| `initializer_range` | 0.02 | Weight initialization std |
| `rms_norm_eps` | 1e-5 | RMSNorm epsilon |
| `tie_word_embeddings` | False | Tie input/output embeddings |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `attention_dropout` | 0.0 | Attention dropout rate |

### Mamba SSM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture_type` | "transformer" | "transformer", "mamba", or "hybrid" |
| `d_state` | 16 | SSM state dimension |
| `d_conv` | 4 | Convolution kernel size |
| `expand` | 2 | Inner dimension expansion factor (inner_dim = hidden_size * expand) |
| `num_attention_layers` | 2 | Number of attention layers in hybrid mode |

### GQA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `head_dim` | hidden_size // num_attention_heads | Per-head dimension |
| `kv_heads` | num_attention_heads | Number of KV heads (set lower for GQA) |

## Presets

Use `config.apply_for(preset)` to load a preset:

### Transformer Presets

| Preset | Hidden | Layers | Heads | Params |
|--------|--------|--------|-------|--------|
| `100m` | 768 | 6 | 12 | ~100M |
| `500m` | 1024 | 8 | 16 | ~500M |
| `1b` | 1280 | 12 | 16 | ~1B |
| `3b` | 1600 | 24 | 16 | ~3B |
| `7b` | 4096 | 32 | 32 | ~7B |
| `13b` | 5120 | 40 | 40 | ~13B |
| `30b` | 6656 | 60 | 52 | ~30B |
| `65b` | 8192 | 80 | 64 | ~65B |
| `100b` | 12288 | 96 | 96 | ~100B |

### Mamba/Hybrid Presets

| Preset | Architecture | Hidden | Layers | Mamba/Attn | Params |
|--------|-------------|--------|--------|------------|--------|
| `mamba-130m` | mamba | 768 | 12 | 12/0 | ~130M |
| `hybrid-130m` | hybrid | 768 | 12 | 10/2 | ~130M |

## Usage Examples

```python
from ava import AvaConfig

# Default transformer config
config = AvaConfig()

# Load a preset
config = AvaConfig().apply_for("hybrid-130m")

# Custom hybrid config
config = AvaConfig(
    architecture_type="hybrid",
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    d_state=16,
    d_conv=4,
    expand=2,
    num_attention_layers=2,
)

# Inspect config
print(config.to_dict())
```
