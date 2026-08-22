# Configuration

`AvaConfig` is a dataclass describing *what a model is*. How it trains lives in
`TrainingConfig` — see [training.md](training.md).

```python
from ava import AvaConfig

config = AvaConfig.from_preset("hybrid-130m", vocab_size=32000)
config = AvaConfig(hidden_size=1024, num_hidden_layers=24, kv_heads=4)
```

Every field is validated at construction, so an inconsistent config fails
immediately rather than several layers into a forward pass.

---

## Core

| Field | Default | Notes |
|---|---|---|
| `vocab_size` | 32000 | must match the tokenizer |
| `hidden_size` | 2048 | residual stream width |
| `intermediate_size` | 8192 | SwiGLU inner width; ~8/3 × hidden is conventional |
| `num_hidden_layers` | 16 | |
| `num_attention_heads` | 16 | |
| `kv_heads` | `= num_attention_heads` | GQA; must divide `num_attention_heads` |
| `head_dim` | `hidden_size / heads` | set explicitly to decouple from `hidden_size` |
| `hidden_act` | `"silu"` | `silu`, `gelu`, `gelu_tanh`, `relu`, `relu2` |
| `max_position_embeddings` | 2048 | training context length |
| `tie_word_embeddings` | `True` | share input and output embeddings |

`kv_heads` is the single highest-leverage inference knob: the KV cache shrinks
by `num_attention_heads / kv_heads`, and decode speed is bound by exactly that
memory traffic. 4–8 KV heads is the usual choice regardless of model size.

## Stability

| Field | Default | Notes |
|---|---|---|
| `rms_norm_eps` | 1e-5 | |
| `qk_norm` | `True` | per-head RMSNorm on Q and K before RoPE |
| `z_loss_coef` | 1e-4 | auxiliary `logsumexp²` penalty; `0.0` disables |
| `scaled_residual_init` | `True` | `1/sqrt(2N)` on output projections |
| `initializer_range` | 0.02 | |

Leave all four alone unless you are reproducing a specific paper. `z_loss_coef`
is the one worth raising (to ~1e-3) if you see loss spikes late in a long bf16
run.

## Position encoding

| Field | Default | Notes |
|---|---|---|
| `rope_theta` | 500000.0 | large base; the 10000 default only suits ≤2k context |
| `rope_scaling` | `None` | context extension, see below |

```python
# Extend a model trained at 8k to 32k
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 8192,
}
config.max_position_embeddings = 32768
```

| Type | Cost | When |
|---|---|---|
| `linear` | needs fine-tuning | simple position interpolation |
| `ntk` | often works zero-shot | up to ~2× extension |
| `yarn` | best quality | 4× and beyond; also raises attention temperature |

## Architecture

| Field | Default | Notes |
|---|---|---|
| `architecture_type` | `"transformer"` | `transformer`, `mamba`, `hybrid` |
| `num_attention_layers` | 2 | hybrid only: trailing attention layers |
| `d_state` | 16 | SSM state size; 64 is the modern default for larger models |
| `d_conv` | 4 | depthwise conv kernel |
| `expand` | 2 | SSM inner width multiplier |
| `dt_rank` | `ceil(hidden/16)` | rank of the `dt` projection |
| `ssm_chunk_size` | `None` (64, derived) | scan window; memory knob only, never changes results |

`ssm_chunk_size` sets the scan window. Retained memory does not depend on it —
that is handled by where the checkpoint boundary sits, see
[architecture.md](architecture.md#why-the-scan-is-chunked) — but the transient
slab is `O(batch × chunk × hidden × expand × d_state)`, and the window size
trades that against how many kernel launches the scan costs.

Left at `None` it resolves to 64, which measured fastest *and* leanest on both
SSM presets; it drops below 64 only when `d_state` is wide enough to make that
slab expensive. The numbers are identical at any setting — only speed and
memory change.

| preset | chunk | train tok/s | peak |
|---|---|---|---|
| `mamba-130m` | 32 | 96 | 2.07 GB |
| `mamba-130m` | **64** | **206** | **2.10 GB** |
| `mamba-130m` | 170 | 125 | 2.58 GB |

(batch 2 × 512 tokens, bf16, RTX 4050 Laptop.)

## Runtime

| Field | Default | Notes |
|---|---|---|
| `use_cache` | `True` | |
| `gradient_checkpointing` | `False` | also settable via `model.gradient_checkpointing_enable()` |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | 0 / 1 / 2 | must match the tokenizer |

---

## Sizing a model before building it

```python
config = AvaConfig.from_preset("hybrid-1b")
print(f"{config.estimate_parameters():,}")     # analytic, allocates nothing
print(config.layer_types())
```

`estimate_parameters()` is exact to well under 1% and costs nothing, which makes
it the right way to explore a design space.

For a quick memory sanity check:

| Quantity | Rule of thumb |
|---|---|
| weights (bf16) | `2 × params` bytes |
| AdamW state | `12 × params` bytes (fp32 master + two moments) |
| total training | `~14 × params` bytes before activations |

A 1B model therefore needs roughly 14 GB before a single activation is stored —
which is why `gradient_checkpointing` and `gradient_accumulation_steps` exist.

`ava.utils.model_summary(model)` prints all of this for a built model.

---

## Presets

```python
AvaConfig.available_presets()
# ['130m', '1b', '30b', '350m', '3b', '70b', '7b', '13b',
#  'hybrid-130m', 'hybrid-1b', 'mamba-130m']
```

Overrides are keyword arguments:

```python
AvaConfig.from_preset("1b", vocab_size=50257, rope_theta=1e6, qk_norm=False)
```

---

## Persistence

```python
config.save_pretrained("checkpoints/final")   # writes config.json
config = AvaConfig.from_pretrained("checkpoints/final")
```

Unknown keys in a loaded `config.json` are preserved in `_extra` and written
back out unchanged, so a config from a newer version round-trips through an
older one without losing information.
