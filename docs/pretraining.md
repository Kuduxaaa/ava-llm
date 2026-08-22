# Running the first real pretraining job

A runbook for going from nothing to a trained base model on a rented GPU, with
measured numbers where they exist and clearly labelled estimates where they do
not.

The base model is the prerequisite for everything else. The appraisal head reads
Ava's hidden states; until those hidden states mean something, the head can
memorise a training set but cannot generalise. So this comes first.

---

## What the hardware costs

Measured on an RTX 4050 Laptop (6.4 GB, 20 SMs), bf16, AdamW fused, no
`torch.compile` (Triton is unavailable on Windows):

| Config | Params | Batch | tok/s | Peak VRAM | 1B tokens |
|---|---|---|---|---|---|
| 30M transformer | 24.9M | 8 × 512 | 24,378 | 3.7 GB | 11 h |
| `130m` transformer | 100.1M | 8 × 512 | 6,487 | 6.1 GB | 43 h |
| `130m` transformer | 100.1M | 16 × 512 | **1,235** | **10.7 GB** | 225 h |
| `hybrid-130m` | 74.9M | 8 × 512 | 323 | 6.1 GB | 861 h |

Three things fall out of that table, and all three are worth knowing before
renting anything.

**Exceeding VRAM does not fail, it crawls.** Batch 16 asked for 10.7 GB on a
6.4 GB card. Windows spilled into shared memory and throughput dropped **5×**
rather than raising an error. An OOM would have been kinder. Size the batch to
fit and reach the effective batch you want with `--grad-accum`.

**Do not pretrain a hybrid on a small machine.** 20× slower than the
transformer at the same size. The selective scan is launch-bound in pure
PyTorch and `torch.compile` — the usual remedy — needs Triton. On Linux with
Triton available this gap narrows substantially; it has not been measured here,
so treat the hybrid as unproven at scale rather than as a known cost.

**Chinchilla is out of reach on a laptop.** A `130m` model wants 2.6B tokens
minimum, which is 111 hours of continuous compute. For a model worth using you
want 10–20B. That is a cloud job.

### Cloud estimates

**These are extrapolations, not measurements.** They assume ~40% MFU, which is
what the laptop actually achieved, applied to an A100 80GB's 312 TFLOP/s dense
bf16 peak. Verify with a short run before committing to a long one.

| Model | Tokens | Est. tok/s | Est. time | Est. cost @ $1.30/h |
|---|---|---|---|---|
| `130m` | 10B | ~150k | ~19 h | ~$25 |
| `350m` | 10B | ~60k | ~46 h | ~$60 |

`130m` on 10B tokens is the recommended first target: 77 tokens per parameter,
comfortably past Chinchilla, which is what you want for a model whose point is
to be cheap at inference.

---

## The sequence

### 0. Preflight, before anything expensive

```bash
pip install -e ".[all]"
python scripts/preflight.py
```

About a minute. It runs tokenizer training, packing, dataset construction, a
real optimizer loop, checkpoint save and resume, generation, and the world — on
a tiny synthetic corpus — and fails loudly at whichever stage is broken. Run it
on the rented machine, not just locally: it catches the environment problems
too.

```
ok    tokenizer trains                  512 pieces
ok    corpus packs                      100,000 tokens, uint16
ok    model builds                      970,624 parameters
ok    training runs                     40 steps
ok    loss goes down                    4.338 -> 0.457
ok    checkpoint resumes
ok    generation runs                   the quick brown fox jumps over a lazy dog
ok    perceive then respond             (1, 12)
```

### 1. Corpus

```bash
python scripts/download_corpus.py \
    --dataset HuggingFaceFW/fineweb --config sample-10BT \
    --output data/corpus.txt
```

Streams, so it never holds the dataset in memory. Roughly 40 GB of text for a
10B-token target — check the disk on the rented box before starting.

### 2. Tokenizer and packing

```bash
python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000
```

Tokenizer training samples 2M sentences rather than loading the corpus, which
is the difference between a few minutes and an out-of-memory kill. Packing runs
at about 3.4M tokens/s per core — under an hour for 10B tokens.

Check the fertility line it prints. Around 1.3–1.8 tokens per word is healthy
for English. Much above 2.5 and every downstream cost is inflated for the life
of the model.

**Do this on a cheap CPU box if you can.** It is an hour of GPU sitting idle
otherwise. Copy `data/tokenizer/` and `data/tokens/` to the GPU machine.

### 3. Pretrain

```bash
python scripts/pretrain.py \
    --tokens data/tokens/train.bin \
    --preset 130m \
    --block-size 1024 \
    --batch-size 24 \
    --grad-accum 8 \
    --max-steps 40000 \
    --lr 6e-4 \
    --schedule wsd \
    --compile \
    --save-every 500 \
    --eval-every 1000
```

Batch and accumulation give an effective batch of ~200k tokens, which is a
reasonable target at this scale. Raise `--batch-size` until VRAM is ~85% full
and lower `--grad-accum` to keep the product constant.

`--compile` is worth 1.3–2× on Linux and does nothing without Triton.

`--schedule wsd` holds the peak learning rate through the middle of the run, so
stopping early and continuing later does not mean training at an already-decayed
rate. That flexibility matters when the machine is rented by the hour.

### 4. Do not lose the checkpoints

Rented machines are preemptible, and `--save-every` writes locally. Sync them
off the box:

```bash
while true; do
  aws s3 sync checkpoints/ s3://your-bucket/ava/checkpoints/ --exclude '*' --include 'ava_step_*.pt'
  sleep 900
done &
```

Resuming restores the optimizer moments, the schedule position and the RNG
state, not just the weights:

```bash
python scripts/pretrain.py ... --resume checkpoints/ava_step_12000.pt
```

Restoring weights alone restarts the schedule from zero and discards the Adam
moments, which shows up as a loss spike right after every resume and is
routinely misdiagnosed as a data problem.

---

## Reading the log

```
step   1200 | loss 3.4127 | ppl    30.35 | lr 2.87e-04 | grad 0.41 | 48,213 tok/s | mfu 34.2% | 612s
```

- **loss** — a window mean, not a running average, so it responds when
  something changes. For English at 32k vocab, expect ~10.4 at init, under 4
  within the first billion tokens, and 3.0–3.4 by 10B for a 130M model.
- **grad** — pre-clip norm. Steadily rising means trouble; occasional spikes
  are normal.
- **mfu** — against the device's dense bf16 peak, using the real block size.
  30–50% is healthy. If it is under 15%, something is wrong: check the batch
  actually fits in VRAM, and that `--compile` took.

Stop and investigate if loss plateaus above 5 after a billion tokens, or if
`grad` climbs monotonically for thousands of steps.

---

## After the base model

The order is forced by what depends on what.

1. **Base model** — this document.
2. **Appraisal head.** It reads the base model's hidden states, so it needs
   them to be meaningful first. Requires labelled `(text → 23 context channels)`
   data; the practical source is a strong model labelling a dialogue corpus.
   The head predicts *context*, never emotions — see
   [world.md](world.md#automatic-appraisal).
3. **Conditioner.** Hardest, because the signal is "this prompt should read
   differently under a different internal world", and no dataset of that exists.
   Likely synthetic, and likely after the appraisal head is known to work.

Each stage is independently trainable with the earlier ones frozen, which is
also the order in which failures are diagnosable.
