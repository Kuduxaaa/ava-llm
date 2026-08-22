# Generation

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

Or from the CLI:

```bash
python scripts/generate.py --model checkpoints/final --prompt "..." --stream
```

---

## Length control

`max_new_tokens` is the primary control and means exactly what it says.
`max_length` bounds prompt + completion. Whichever binds first wins, and there is
no hidden cap on top of either.

```python
GenerationConfig(max_new_tokens=200)                  # 200 new tokens
GenerationConfig(max_new_tokens=200, max_length=512)  # or fewer, if 512 is hit
```

## Sampling

| Field | Default | What it does |
|---|---|---|
| `do_sample` | `True` | `False` gives greedy argmax |
| `temperature` | 1.0 | <1 sharpens, >1 flattens |
| `top_k` | 0 | keep the k highest logits (0 = off) |
| `top_p` | 1.0 | keep the smallest set whose mass exceeds p |
| `min_p` | 0.0 | keep tokens with probability ≥ `min_p × max_prob` |
| `repetition_penalty` | 1.0 | discourage already-generated tokens |
| `no_repeat_ngram_size` | 0 | hard-block repeating an n-gram |
| `seed` | `None` | reproducible sampling |

Filters are applied in order: top-k, then min-p, then top-p.

**Prefer min-p over top-p.** Top-p keeps a fixed probability mass regardless of
how confident the model is, so at high temperature it admits a long tail of
junk. Min-p sets its threshold relative to the top token, so it stays tight when
the model is confident and opens up when it genuinely is not. A good starting
point is `temperature=0.8, min_p=0.05` with top-p and top-k off.

**Repetition penalty** divides positive logits and multiplies negative ones.
Dividing a negative logit would make the token *more* likely — the opposite of
the intent, and a common bug in hand-rolled samplers. Keep it near 1.05; higher
values visibly distort grammar.

## Batched prompts

```python
batch = tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
output = model.generate(
    batch["input_ids"],
    attention_mask=batch["attention_mask"],
    generation_config=config,
)
```

Left padding, so every sequence's last real token is in the last column.
Positions are derived from the attention mask, so each sequence gets the
positions it would have had unpadded. Finished sequences emit `pad_token_id`
rather than continuing to sample.

## Streaming

Any object with `put(token_ids)` and `end()` works:

```python
class Printer:
    def put(self, token_ids): print(tokenizer.decode(token_ids[0]), end="", flush=True)
    def end(self): print()

model.generate(ids, generation_config=config, streamer=Printer())
```

`scripts/generate.py --stream` uses a slightly more careful version that
re-decodes the buffer each step, so multi-token characters render correctly.

---

## The cache

`generate` allocates an `AvaCache` and reuses it across steps. For a transformer
that is the KV cache; for Mamba layers it is a fixed-size SSM state plus the
depthwise-conv lookback window. Hybrid models get both, in one object.

Decoding is asserted to be equivalent to a full forward pass — see
[architecture.md](architecture.md#the-invariant). `use_cache=False` still works
and produces identical output, at quadratic cost; it is useful when you suspect
a caching bug.

Only the last position's logits are computed during decoding
(`num_logits_to_keep=1`). For a small model with a large vocabulary, the LM head
is a meaningful share of per-token cost, and the other positions are discarded
anyway.

### Decode throughput

Single-token decoding on a small model is **kernel-launch bound**, not compute
bound: a `130m` step is ~7 ms of GPU work spread over ~370 launches. Whether
that is fast depends almost entirely on your platform's launch latency — around
4 µs on Linux, but ~40 µs measured on Windows, which is the difference between
~100 tok/s and ~30 tok/s for the same model and the same GPU.

Two things help, in order:

- **`torch.compile`** fuses those launches away. It needs Triton, which is not
  available on every platform — check before assuming it will help.
- **Larger batches.** Decoding is bandwidth- and launch-bound, so batching
  requests is nearly free until the KV cache fills memory.

`kv_heads` is the model-side lever; see [configuration.md](configuration.md).

**Context length.** Nothing truncates automatically. Past
`max_position_embeddings` the model extrapolates and quality degrades; for long
contexts set `rope_scaling` (see [configuration.md](configuration.md)) and, if
you need a hard bound on cache growth, call `layer_cache.crop(n)` yourself.

---

## LoRA

```python
from ava.model.lora import apply_lora, merge_lora, lora_state_dict

apply_lora(model, target_modules=["q_proj", "v_proj"], rank=16, alpha=32)
# ... fine-tune; only lora_A / lora_B carry gradients ...

torch.save(lora_state_dict(model), "adapter.pt")   # a few MB
merge_lora(model)                                  # fold in; zero inference cost
```

`B` is initialised to zero, so the adapter is an exact no-op at step 0 and the
model's outputs are bit-identical until training moves it.

Targets match on the **leaf attribute name**, not a substring of the dotted path.
That is why `o_proj` cannot accidentally capture Mamba's `out_proj`. Layers are
collected before any swapping happens — mutating the module tree while
`named_modules()` walks it is how layers get skipped or wrapped twice.

`rank` 8–16 is enough for style and format adaptation; 32–64 for new knowledge.
`alpha` is conventionally `2 × rank`.

## Quantisation

```python
from ava.model.quantization import quantize_model

quantize_model(model, bits=8)     # or bits=4
```

Weight-only, with **per-output-channel** scales. A single scale for the whole
tensor lets one outlier row set the step size for every other row, and quality
collapses.

This buys memory, not arithmetic speed: weights are stored small and
dequantised per call. For actual speedups you want a fused kernel
(bitsandbytes, torchao) — this implementation deliberately does not pretend to
be one. Note also that 4-bit values are held in an `int8` container: the
quantisation error is 4-bit, the storage is not packed.

`lm_head` is skipped by default. Its error feeds straight into the sampled
distribution, and it is usually tied to the embeddings anyway.

---

## Saving and loading

```python
model.save_pretrained("checkpoints/final")               # safetensors if available
model = AvaForCausalLM.from_pretrained("checkpoints/final", dtype=torch.bfloat16)
```

Writes `config.json` plus `model.safetensors` (or `pytorch_model.bin` when
safetensors is not installed). Tied `lm_head.weight` is not duplicated on disk
and is re-tied on load.
