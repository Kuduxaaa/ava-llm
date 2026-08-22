# Tokenizer

`AvaTokenizer` wraps SentencePiece with the parts a training loop actually needs.
It is language-agnostic by construction — nothing in it assumes a script, a
writing direction, or a segmentation strategy.

```bash
pip install "ava-llm[tokenizer]"
```

---

## Training one

```python
from ava.tokenizer import AvaTokenizer

tokenizer = AvaTokenizer.train(
    "data/corpus.txt",
    "data/tokenizer",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,
)
```

or via the script, which also packs the corpus afterwards:

```bash
python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000
```

### The knobs that matter

**`vocab_size`.** Larger vocabularies mean fewer tokens per document (cheaper
training, longer effective context) but a larger embedding table and a slower
softmax. For a single language, 32k is a reasonable default; multilingual
corpora want 64k–128k. Below ~16k, morphologically rich languages fragment badly.

**`character_coverage`.** The one genuinely script-dependent setting. Use `1.0`
for a small alphabet where you want every character in the vocabulary; use
`0.9995` when the corpus has a long tail of rare characters (CJK, or mixed-script
web text), so that tail falls through to byte fallback instead of consuming
thousands of vocabulary slots.

**`model_type`.** `bpe` is the default and the safe choice. `unigram` sometimes
segments agglutinative morphology more cleanly, at the cost of a slower training
step.

**Byte fallback** is always on. Any character the tokenizer never saw is encoded
as raw UTF-8 bytes rather than `<unk>`, so no input is ever unrecoverable —
which matters for code, emoji, and any script the corpus under-represents.

### Special tokens

Ids 0–3 are pinned to `<pad>`, `<s>`, `</s>`, `<unk>`. Pinning them means a
checkpoint's special-token ids stay stable across retrains, so a config written
against one tokenizer still lines up with the next.

If a SentencePiece model reserves no pad id, `AvaTokenizer` falls back to the EOS
id rather than leaving it at `-1` — padding with `-1` indexes out of the
embedding table.

---

## Using it

```python
tokenizer = AvaTokenizer.from_pretrained("data/tokenizer")

tokenizer.encode("hello world")                    # [1, ..., 2]
tokenizer.encode("hello world", add_special_tokens=False)

batch = tokenizer(
    ["short", "a considerably longer document"],
    padding=True, truncation=True, max_length=512, return_tensors="pt",
)
batch["input_ids"], batch["attention_mask"]
```

`__call__` and `encode` add the same special tokens by default. When those two
disagree — which is easy to do by accident — a model trains on sequences that
never contain EOS and then never learns to stop generating.

**Truncation keeps the closing EOS.** A truncated sequence that lost its EOS
teaches the model that long documents simply do not end.

### Padding side

```python
tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
```

Use `left` before generation so every sequence's last real token is in the last
column. With right padding the model is asked to continue from a pad token.
`right` is correct for training.

Ava's attention handles either correctly — positions come from the attention
mask, and `build_causal_mask` guarantees no query row is fully masked — but left
padding is still what you want at inference time.

---

## Chat template

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    add_generation_prompt=True,
)
# '<|user|>\nhello\n<|assistant|>\n'
```

Deliberately minimal and role-agnostic. Replace it with your own scheme if you
prefer — the only requirement is that training and inference use the same one.

---

## Saving

```python
tokenizer.save_pretrained("data/tokenizer")
```

Writes `tokenizer.model` and `tokenizer_config.json` (vocab size, special token
ids, padding side, BOS/EOS policy). `from_pretrained` restores all of it.

---

## Sanity checks

Before committing to a tokenizer, look at what it actually does:

```python
text = open("data/corpus.txt", encoding="utf-8").readline()
pieces = tokenizer.tokenize(text)

print(pieces[:20])
print(f"{len(pieces) / len(text.split()):.2f} tokens per word")
assert tokenizer.decode(tokenizer.encode(text)) == text
```

Fertility — tokens per word — is the number to watch. Around 1.3–1.8 is healthy.
Much above 2.5 means the vocabulary does not fit the corpus, and every downstream
cost (training time, context length, inference latency) is inflated by that
factor for the life of the model.

The round-trip assertion is worth keeping in your pipeline. Byte fallback makes
it hold for any input; if it ever fails, something is wrong with the model file.
