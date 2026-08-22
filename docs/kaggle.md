# Training on Kaggle

Kaggle gives away real GPU hours, but with two constraints that shape the whole
plan: **sessions are capped at about 12 hours**, and **neither GPU it offers
supports bf16**. Everything below follows from those.

Quotas and limits change. Check your own quota page rather than trusting these
numbers, and verify the GPU you were actually given in the first cell.

---

## What you get, and what it means

| | P100 | T4 ×2 |
|---|---|---|
| Compute capability | sm_60 | sm_75 |
| Runs at all on Kaggle's PyTorch | **no** | yes |
| bf16 hardware | no | no |
| fp16 tensor cores | no | yes |
| Memory | 16 GB | 15 GB each |

**You must select T4 ×2. The P100 does not work.** Kaggle's PyTorch build ships
kernels for `sm_70` and up, and the P100 is `sm_60`. Cubins are binary
compatible upward across *minor* versions only, not across majors, so there is
nothing for it to run:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation. The current PyTorch install supports CUDA
capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```

That warning is easy to scroll past, and the failure it produces arrives later
as `CUDA error: no kernel image is available for execution on the device` —
which names neither the GPU nor the fix. `preflight.py` checks for it before
anything else and says so plainly.

Neither card has bf16 hardware, so `precision="auto"` resolves to **fp16 with a
gradient scaler**. Note that `torch.cuda.is_bf16_supported()` cannot be used to
decide this: it defaults to `including_emulation=True` and answers yes on a
P100. Ava checks the compute capability directly, because emulated bf16 on a T4
would cost it its tensor cores.

fp16 is the path where loss scaling matters, so watch for `grad` reaching `nan`.

### Rough budget

Estimates, scaled from measured throughput on other hardware at ~40% MFU. Verify
against the `tok/s` your first session actually reports.

| Setup | Est. tok/s | Per 11 h session | 10B tokens |
|---|---|---|---|
| 1 × T4 | ~43k | ~1.7B | ~6 sessions |
| 2 × T4 (DDP) | ~80k | ~3.2B | ~3 sessions |

A `130m` model on 5–10B tokens is a realistic Kaggle target. `350m` is not.

---

## The shape of it

Kaggle sessions are ephemeral, so a real run is a *chain* of them:

```
notebook A (CPU, internet)   corpus -> tokenizer -> packed .bin   -> output
notebook B run 1 (GPU)       A's output + nothing    -> checkpoint -> output
notebook B run 2 (GPU)       A's output + run 1      -> checkpoint -> output
notebook B run 3 (GPU)       A's output + run 2      -> ...
```

Data preparation happens **once**, on a CPU session, so you are not paying GPU
quota to tokenize. Training resumes from the previous run's output each time.

---

## Notebook A — prepare the data

CPU accelerator. **Settings → Internet → On.**

```python
!git clone https://github.com/Kuduxaaa/ava-llm.git /kaggle/working/ava
%cd /kaggle/working/ava
!pip install -q -e ".[all]"
```

```python
!python scripts/preflight.py
```

One minute, and it fails loudly if anything in the environment is wrong. Do not
skip it — this is the cheapest place to find out.

```python
# ~10B tokens is roughly 40 GB of text; start smaller and see how the disk holds up.
!python scripts/download_corpus.py \
    --dataset HuggingFaceFW/fineweb --config sample-10BT \
    --max-docs 8000000 \
    --output /kaggle/working/data/corpus.txt
```

```python
!python scripts/prepare_data.py \
    --corpus /kaggle/working/data/corpus.txt \
    --tokenizer-dir /kaggle/working/data/tokenizer \
    --output-dir /kaggle/working/data/tokens \
    --vocab-size 32000
```

Check the **fertility** line it prints: 1.3–1.8 tokens per word is healthy for
English. Much above 2.5 means the vocabulary does not fit the corpus and every
downstream cost is inflated for the life of the model.

Then delete the raw text so it does not eat the output quota, and save:

```python
!rm /kaggle/working/data/corpus.txt
!du -sh /kaggle/working/data/*
```

**Save Version → Save & Run All (Commit).** The output becomes reusable input.

---

## Notebook B — train

GPU T4 ×2. Internet on (for `pip install`).

**Add data:** notebook A's output. From the second run onward, also add the
previous run of notebook B.

```python
import torch
print(torch.cuda.device_count(), "GPU(s)  build supports:", torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
```

If you see `sm_60`, stop and switch the accelerator to **T4 ×2** before doing
anything else.

```python
!git clone https://github.com/Kuduxaaa/ava-llm.git /kaggle/working/ava
%cd /kaggle/working/ava
!pip install -q -e ".[all]"
```

Copy the packed data and any previous checkpoint into a writable place.
`/kaggle/input` is read-only, and the trainer needs to write next to what it
reads:

```python
import glob, os, shutil

DATA = glob.glob("/kaggle/input/*/data/tokens/train.bin")[0]
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

# The newest checkpoint from any attached input, if there is one.
found = sorted(
    glob.glob("/kaggle/input/*/checkpoints/ava_step_*.pt"),
    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
)
if found:
    shutil.copy(found[-1], "/kaggle/working/checkpoints/")
    print("resuming from", found[-1])
else:
    print("starting from scratch")
```

Then train. `--resume auto` picks up whatever was just copied, and starts fresh
if there was nothing:

```python
!cd /kaggle/working/ava && torchrun --nproc_per_node=2 scripts/pretrain.py \
    --tokens {DATA} \
    --tokenizer-dir /kaggle/input/*/data/tokenizer \
    --preset 130m \
    --block-size 512 \
    --batch-size 16 \
    --grad-accum 6 \
    --max-steps 60000 \
    --lr 6e-4 \
    --schedule wsd \
    --compile \
    --checkpoint-dir /kaggle/working/checkpoints \
    --save-every 500 \
    --eval-every 2000 \
    --resume auto \
    --max-hours 11
```

`--max-hours 11` stops cleanly and saves before Kaggle pulls the session. Being
killed loses everything since the last periodic save and leaves no final
checkpoint; stopping an hour early loses an hour.

Single GPU is the same command without `torchrun`:

```python
!python scripts/pretrain.py --tokens {DATA} ... --max-hours 11
```

Keep only what you need in the output — Kaggle caps it:

```python
!ls -la /kaggle/working/checkpoints
!find /kaggle/working/checkpoints -name 'ava_step_*.pt' | sort -V | head -n -1 | xargs -r rm
!du -sh /kaggle/working/checkpoints
```

**Save Version → Save & Run All (Commit).** Next run, attach this run's output
and repeat.

---

## Sizing the batch

`--batch-size 16 --grad-accum 6` at block 512 across 2 GPUs is ~98k tokens per
optimizer step, which is reasonable at this scale.

Raise `--batch-size` until memory is around 85% full, then lower `--grad-accum`
to keep the product roughly constant. Check with:

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Do not let it exceed the card.** Measured on a 6.4 GB card, a batch needing
10.7 GB did not raise an error — it spilled into host memory and ran **5×
slower**. An out-of-memory error would have been kinder. If `tok/s` is far below
what the first session reported, this is the first thing to check.

---

## What to watch

```
step   1200 | loss 3.4127 | ppl 30.35 | lr 2.87e-04 | grad 0.41 | 48,213 tok/s | mfu 34.2% | 612s
```

- **loss** — around 10.4 at init for a 32k vocabulary, under 4 within the first
  billion tokens, 3.0–3.4 by 10B for a 130M model.
- **grad** — pre-clip norm. Under fp16 specifically, watch for `nan`: that is the
  gradient scaler losing the fight, and it means lowering the learning rate.
- **tok/s** — compare against your first session. A large drop means the batch
  no longer fits.
- **mfu** — 30–50% is healthy. Under 15% means the batch is spilling or
  `--compile` did not take.

Do not panic at the first thousand steps. Loss falls very fast, then appears to
stall; that is normal.

---

## Things that will bite

**Internet is off by default.** `pip install` and the corpus download both fail
without it, and the error does not say so clearly.

**`/kaggle/input` is read-only.** Copy checkpoints to `/kaggle/working` before
resuming.

**Interactive sessions end when you close the tab.** Use *Save & Run All
(Commit)* for anything long — it runs headless to the session limit.

**Output size is capped.** Delete the raw corpus after packing and keep only the
latest checkpoint.

**Resuming restores more than weights.** Optimizer moments, schedule position
and RNG state all come back. Loading weights alone restarts the schedule from
zero and throws away the Adam moments, which shows up as a loss spike after
every resume and is routinely misdiagnosed as a data problem.

---

See [pretraining.md](pretraining.md) for measured throughput on other hardware
and for what comes after the base model.
