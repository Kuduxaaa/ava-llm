# Ava documentation

| Page | Covers |
|---|---|
| [architecture.md](architecture.md) | Transformer, Mamba and hybrid stacks; the cache; stability |
| [configuration.md](configuration.md) | Every `AvaConfig` field, presets, sizing a model |
| [training.md](training.md) | Precision, optimizers, schedules, memory, DDP, resume |
| [data.md](data.md) | Corpus packing, chat data, collators |
| [tokenizer.md](tokenizer.md) | Training a tokenizer for any language |
| [generation.md](generation.md) | Sampling, caching, LoRA, quantisation |

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
  data/
    packing.py        corpus -> memory-mapped token stream
    datasets.py       PackedDataset, ChatDataset, collators
  training/
    trainer.py        the loop: AMP, accumulation, DDP, checkpointing
    optimizer.py      AdamW, Muon, schedules
    metrics.py        evaluation, throughput, MFU
  tokenizer.py        SentencePiece wrapper
  utils/              distributed setup, summaries, seeding

scripts/          download_corpus, prepare_data, pretrain, generate
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
