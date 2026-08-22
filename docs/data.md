# Data

Two shapes of data, two pipelines:

| Stage | Dataset | Loss on | Padding |
|---|---|---|---|
| pretraining | `PackedDataset` | every token | none |
| instruction tuning | `ChatDataset` | assistant turns | batch-local |

---

## Pretraining: pack, do not pad

Padding every document to `max_length` is the default in tutorial code, and it
is why small pretraining runs are so much slower than they should be. At a 512
block size with a median document of ~90 tokens, roughly four out of five
positions the GPU processes are padding.

Packing concatenates documents into one contiguous token stream separated by
EOS, then cuts fixed-size blocks out of it. Every position carries signal.

```bash
python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000
```

or from Python:

```python
from ava.data import iter_text_file, pack_corpus, PackedDataset

pack_corpus(iter_text_file("data/corpus.txt"), tokenizer, "data/tokens/train.bin")

dataset = PackedDataset("data/tokens/train.bin", block_size=1024)
train_set, val_set = dataset.split(val_fraction=0.005)
```

The output is a memory-mapped `.bin`, so the dataset no longer has to fit in
RAM, plus a `meta.json` recording the token count and dtype. `pack_corpus`
streams — memory use is bounded by `batch_size`, not by corpus size.

**Dtype.** `uint16` for vocabularies up to 65536, `uint32` above. `load_packed`
reads the dtype from `meta.json` and raises if it is missing rather than
guessing — a wrong guess silently halves or doubles the token count, and nothing
downstream would notice.

**Blocks.** `PackedDataset` yields `input_ids` and an identical `labels`; the
model applies the causal shift internally. Use `collate_packed`, which stacks
and nothing else — there is no attention mask because there is no padding.

`stride` defaults to `block_size` (disjoint blocks). Setting it lower produces
overlapping windows, which is occasionally useful for a small corpus but means
the model sees each token more than once per epoch.

---

## Instruction tuning

```python
from ava.data import ChatDataset, PaddingCollator
from torch.utils.data import DataLoader

conversations = [
    [
        {"role": "user", "content": "What is a state-space model?"},
        {"role": "assistant", "content": "A sequence model that ..."},
    ],
]

dataset = ChatDataset(conversations, tokenizer, max_length=2048)
loader = DataLoader(
    dataset, batch_size=8, shuffle=True,
    collate_fn=PaddingCollator(pad_token_id=tokenizer.pad_token_id),
)
```

### How label masking works

Loss is computed on assistant content only. Role headers and user turns get
`-100`, which `cross_entropy(ignore_index=-100)` drops.

The masking is built by encoding each turn **separately** and tracking token
counts as the sequence is assembled. The tempting alternative — encode a prefix
of the joined string and use its length as a token offset — is both quadratic
and wrong: a subword tokenizer does not guarantee that the encoding of a prefix
is a prefix of the encoding. A single merge across the boundary shifts every
subsequent offset, so the mask silently drifts out of alignment with the tokens.

Two smaller details that matter:

- The final EOS is supervised. If it is not, the model never learns to stop and
  generation runs to `max_new_tokens` every time.
- Conversations with no assistant turn, or where everything ends up masked, are
  dropped. An all-`-100` example produces a NaN loss, not a zero one.

`train_on_last_turn_only=True` supervises only the final assistant turn — useful
when earlier turns come from a different model and you do not want to distil
them.

### Chat template

`tokenizer.apply_chat_template(messages, add_generation_prompt=True)` renders the
same `<|role|>\ncontent\n` format `ChatDataset` trains on. Swap it for your own
scheme if you like — the only rule is that training and inference must use the
*same* one. A mismatch here degrades a model in a way that looks like
undertraining.

---

## Collators

**`collate_packed`** — for `PackedDataset`. Stacks blocks. No mask.

**`PaddingCollator(pad_token_id, pad_to_multiple_of=8)`** — for variable-length
data. Pads to the batch's own longest sequence rather than a global maximum,
which is why this is a collator and not something the dataset does: the dataset
cannot know what it will be batched with. `pad_to_multiple_of=8` keeps sequence
lengths friendly to tensor-core kernels. Padded positions get `-100` labels and
a zero attention mask.

---

## Getting a corpus

```bash
python scripts/download_corpus.py --dataset HuggingFaceFW/fineweb --config sample-10BT
python scripts/download_corpus.py --dataset wikimedia/wikipedia --config 20231101.fr
python scripts/download_corpus.py --dataset uonlp/CulturaX --config ja --max-docs 1000000
```

Any Hub dataset with a text column works; `--text-column` names it. The quality
filters are deliberately structural — length, run-length, symbol density,
replacement characters — because those flag boilerplate and mojibake in any
script, whereas a stopword list only works for the language it was written for.

## How much data?

Chinchilla-optimal is ~20 tokens per parameter. For a small model you will
usually want considerably more, since inference cost is what you are actually
optimising:

| Model | Chinchilla | A practical target |
|---|---|---|
| 130M | 2.6B tokens | 10–20B |
| 350M | 7B | 30–50B |
| 1B | 20B | 100B+ |

`load_packed()` returns the exact token count, so you can check before starting:

```python
_, corpus = load_packed("data/tokens/train.bin")
print(f"{corpus.num_tokens:,} tokens from {corpus.num_documents:,} documents")
```
