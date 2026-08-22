# Ava documentation

| Page | Covers |
|---|---|
| [architecture.md](architecture.md) | Transformer, Mamba and hybrid stacks; the cache; stability |
| [configuration.md](configuration.md) | Every `AvaConfig` field, presets, sizing a model |
| [training.md](training.md) | Precision, optimizers, schedules, memory, DDP, resume |
| [pretraining.md](pretraining.md) | Runbook for the first real job: measured costs, sequence, gotchas |
| [kaggle.md](kaggle.md) | Chained sessions on Kaggle: no bf16, 12-hour cap, resume |
| [data.md](data.md) | Corpus packing, chat data, collators |
| [tokenizer.md](tokenizer.md) | Training a tokenizer for any language |
| [generation.md](generation.md) | Sampling, caching, LoRA, quantisation |
| [world.md](world.md) | The internal world: 136 coupled channels on their own clocks |

## Where to start

**Training a model from nothing** — [data.md](data.md) →
[tokenizer.md](tokenizer.md) → [training.md](training.md).

**Understanding the code** — [architecture.md](architecture.md), then
`ava/model/cache.py` and `tests/test_cache.py`. The cache is where the three
architectures meet, and the equivalence test in that file is the tightest
statement of what the model is supposed to do.

**Deploying something you trained** — [generation.md](generation.md).

## Layout

```
ava/
  config/       AvaConfig and the presets
  model/
    attention.py      grouped-query attention, RoPE, masking
    mamba.py          selective SSM, chunked scan, recurrent step
    cache.py          unified KV + SSM decoding state
    embeddings.py     rotary embeddings, linear/NTK/YaRN scaling
    generation.py     sampling parameters and logits processing
    ava_model.py      the stack, the LM head, generate(), persistence
    lora.py           adapters
    quantization.py   int8 / int4 weight-only
  world/
    schema.py         the 136 channels, baselines, time constants
    coupling.py       ~220 edges: what drives what
    dynamics.py       the integrator
    clock.py          absolute time, circadian and homeostatic drives
    personality.py    traits as baseline and tau shifts
    appraisal.py      event -> meaning, expectation, prediction error
    perception.py     prompt -> hidden states -> context, before generation
    relationships.py  per-person bonds
    conditioning.py   FiLM and soft prefix into the model
    engine.py         the loop that ties it together
  data/
    packing.py        corpus -> memory-mapped token stream
    datasets.py       PackedDataset, ChatDataset, collators
  training/
    trainer.py        the loop: AMP, accumulation, DDP, checkpointing
    optimizer.py      AdamW, Muon, schedules
    metrics.py        evaluation, throughput, MFU
  tokenizer.py        SentencePiece wrapper
  utils/              distributed setup, summaries, seeding

scripts/          preflight, download_corpus, prepare_data, pretrain, generate, world_demo
tests/            pytest suite
```

## Contributing

```bash
pip install -e ".[all,dev]"
pytest
ruff check ava scripts tests
ruff format ava scripts tests
```

New behaviour needs a test. In particular, anything touching attention, the
cache or the SSM scan should keep
`tests/test_cache.py::test_incremental_matches_full_forward` passing for all
three architectures — that single assertion catches most of what can go wrong in
this codebase.
