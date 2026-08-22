# Ava

A from-scratch language-model framework in pure PyTorch. Transformer, Mamba
(selective state-space) and hybrid stacks share one config, one cache, one
training loop and one generation path — no custom CUDA kernels, no framework
lock-in, nothing you cannot read in an afternoon.

Built for people who want to *understand* a modern LLM stack end to end, and to
train real models on hardware they actually have.

```python
from ava import AvaConfig, AvaForCausalLM

model = AvaForCausalLM.from_preset("hybrid-130m")
print(f"{model.num_parameters():,} parameters")
```

---

## Install

```bash
pip install -e ".[all]"      # everything
pip install -e ".[dev]"      # + pytest and ruff
```

Requires Python 3.10+ and PyTorch 2.4+.

---

## Architectures

| Mode | Layer stack | Decode cost per token | Memory during decode |
|---|---|---|---|
| `transformer` | attention + SwiGLU | O(context) | grows with context |
| `mamba` | selective SSM | O(1) | constant |
| `hybrid` | Mamba blocks, then attention blocks | O(1) for most layers | mostly constant |

Hybrid is the default recommendation: the Mamba blocks carry the bulk of the
sequence cheaply, and a few trailing attention layers restore the precise
token-to-token lookups that pure SSMs are weakest at.

```
input → embed → [MambaBlock × 10] → [DecoderLayer × 2] → RMSNorm → lm_head
```

## What is in the box

**Modelling**
- Grouped-query attention with SDPA / FlashAttention fast paths
- RoPE with `linear`, `ntk` and `yarn` context extension
- QK-norm and a z-loss term for bf16 stability at depth
- SwiGLU (any gated activation via `hidden_act`)
- Chunked selective scan — SSM memory bounded by chunk size, not by sequence length
- Tied embeddings, scaled residual init, gradient checkpointing

**Training**
- bf16 by default, fp16 + GradScaler only where bf16 is unavailable
- AdamW (fused) or Muon on hidden matrices with AdamW on the rest
- Cosine or WSD (warmup-stable-decay) schedules
- Gradient accumulation, DDP, `torch.compile`
- Step-based checkpointing with full resume (optimizer, schedule, RNG)
- Live tok/s and MFU reporting

**Data**
- Corpus packing into a memory-mapped token stream — no padding, no RAM ceiling
- Chat dataset with loss masked to assistant turns
- Batch-local padding collator

**Inference**
- Unified KV + SSM cache; incremental decoding matches full-sequence logits
- top-k, top-p, min-p, repetition penalty, n-gram blocking
- LoRA with merge-back, and int8 / int4 weight-only quantisation

**Internal world** — a persistent simulated organism, in tensors
- 136 coupled channels: biochemistry, physiology, drives, emotion, cognition
- Time constants from two minutes to a month, asymmetric rise and fall
- Circadian rhythm, sleep pressure, hunger, rumination, isolation — running
  whether or not anyone is talking
- Appraisal conditioned on the world, so identical words are not identical events
- Personality as a standing bias; per-person bonds with their own clocks
- Reaches the model by modulating the residual stream, not by narration

---

## Quick start

```bash
# 1. Get a corpus (any Hub dataset, any language)
python scripts/download_corpus.py --dataset HuggingFaceFW/fineweb \
    --config sample-10BT --max-docs 500000

# 2. Train a tokenizer and pack the corpus
python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000

# 3. Pretrain
python scripts/pretrain.py --tokens data/tokens/train.bin --preset hybrid-130m

# 4. Sample
python scripts/generate.py --model checkpoints/final --prompt "Once upon a time" --stream
```

Multi-GPU is the same command under `torchrun`:

```bash
torchrun --nproc_per_node=4 scripts/pretrain.py --tokens data/tokens/train.bin
```

### From Python

```python
import torch
from torch.utils.data import DataLoader

from ava import AvaConfig, AvaForCausalLM
from ava.data import PackedDataset, collate_packed
from ava.training import TrainingConfig, train_model

dataset = PackedDataset("data/tokens/train.bin", block_size=1024)
train_set, val_set = dataset.split(val_fraction=0.01)

model = AvaForCausalLM(AvaConfig.from_preset("hybrid-130m", vocab_size=32000))

model, history = train_model(
    model,
    DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_packed),
    DataLoader(val_set, batch_size=8, collate_fn=collate_packed),
    training_config=TrainingConfig(
        max_steps=20_000,
        learning_rate=3e-4,
        lr_schedule="wsd",
        gradient_accumulation_steps=8,
        precision="bf16",
    ),
)
model.save_pretrained("checkpoints/final")
```

### The internal world

```python
from ava.world import WorldEngine, Personality

engine = WorldEngine(personality=Personality(neuroticism=0.7))

engine.observe({"loss": 0.9, "psychological_threat": 0.7}, dt=45, person="nika")
engine.idle(1800)          # half an hour later, still not fine
print(engine.summary())
print(engine.why("internal_state.stress"))

output = model(input_ids=ids, world_state=engine.state)
```

Emotion is an output here, not an input: nothing writes to `emotion.sadness`.
See [docs/world.md](docs/world.md), or watch it run:

```bash
python scripts/world_demo.py --scenario job --personality anxious --explain
```

### Generating

```python
from ava import AvaForCausalLM, GenerationConfig
from ava.tokenizer import AvaTokenizer

tokenizer = AvaTokenizer.from_pretrained("data/tokenizer")
model = AvaForCausalLM.from_pretrained("checkpoints/final").eval()

ids = tokenizer.encode("The key insight is", return_tensors="pt")
output = model.generate(
    ids,
    generation_config=GenerationConfig(
        max_new_tokens=200,
        temperature=0.8,
        min_p=0.05,
        eos_token_id=tokenizer.eos_token_id,
    ),
)
print(tokenizer.decode(output[0]))
```

---

## Presets

| Preset | Architecture | Parameters | Context | Notes |
|---|---|---|---|---|
| `130m` | transformer | ~130M | 2048 | ablations, unit tests |
| `350m` | transformer | ~350M | 4096 | deep-and-narrow |
| `1b` | transformer | ~1.1B | 8192 | on-device assistants |
| `3b` | transformer | ~3B | 8192 | |
| `7b` | transformer | ~7B | 8192 | untied embeddings |
| `13b` / `30b` / `70b` | transformer | — | 8192 | needs FSDP; DDP alone will not fit |
| `mamba-130m` | mamba | ~130M | 2048 | long sequences, constant-memory decode |
| `hybrid-130m` | hybrid | ~130M | 2048 | trains on a single 16 GB GPU |
| `hybrid-1b` | hybrid | ~1.1B | 8192 | |

`AvaConfig.from_preset(name, **overrides)` — every field is overridable, and
`config.estimate_parameters()` tells you the size before anything is allocated.

---

## Testing

```bash
pytest                      # full suite
pytest tests/test_cache.py  # the decoding-correctness invariants
ruff check ava scripts tests
```

The suite's centre of gravity is `tests/test_cache.py`, which asserts that
running a sequence in one shot and running it token-by-token through the cache
produce the same logits. Nearly every decoding bug — RoPE at the wrong offset,
keys rotated twice, an SSM that forgets its state — breaks that equality and
nothing else notices.

---

## Documentation

- [Architecture](docs/architecture.md) — transformer, Mamba, hybrid, and the cache
- [Configuration](docs/configuration.md) — every `AvaConfig` field
- [Training](docs/training.md) — precision, optimizers, schedules, scaling, memory
- [Data](docs/data.md) — packing, chat data, collators
- [Tokenizer](docs/tokenizer.md) — training one for any language
- [Generation](docs/generation.md) — sampling, caching, LoRA, quantisation
- [World](docs/world.md) — the internal world engine

## References

- [Vaswani et al. (2017), *Attention is All You Need*](https://arxiv.org/abs/1706.03762)
- [Su et al. (2021), *RoFormer: Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)
- [Shazeer (2020), *GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202)
- [Ainslie et al. (2023), *GQA: Grouped-Query Attention*](https://arxiv.org/abs/2305.13245)
- [Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling*](https://arxiv.org/abs/2312.00752)
- [Peng et al. (2023), *YaRN: Efficient Context Window Extension*](https://arxiv.org/abs/2309.00071)
- [Hu et al. (2021), *LoRA: Low-Rank Adaptation*](https://arxiv.org/abs/2106.09685)
- [Jordan et al. (2024), *Muon: an optimizer for hidden layers*](https://kellerjordan.github.io/posts/muon/)
- [Hu et al. (2024), *MiniCPM* — the WSD schedule](https://arxiv.org/abs/2404.06395)

## Author

Created and maintained by Nika Kudukhashvili — [GitHub](https://github.com/Kuduxaaa).

Licensed under the MIT License.
