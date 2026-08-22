# Training

```python
from ava.training import TrainingConfig, train_model

model, history = train_model(
    model, train_loader, val_loader,
    training_config=TrainingConfig(
        max_steps=20_000,
        learning_rate=3e-4,
        lr_schedule="wsd",
        gradient_accumulation_steps=8,
        precision="bf16",
        save_every=1000,
    ),
)
```

Or drive `Trainer` directly when you need the object (to resume, to checkpoint
manually, to inspect state).

---

## Precision

`precision="auto"` picks bf16 on any GPU that supports it and falls back to fp16
with a `GradScaler` otherwise. **Prefer bf16.** It has the same exponent range
as fp32: no loss scaling, no skipped steps from inf/NaN, no scaler state to
checkpoint. fp16's 5-bit exponent is the reason gradient scaling exists at all,
and it is a source of silent divergence in long runs.

| Setting | Use |
|---|---|
| `auto` | default; bf16 where available |
| `bf16` | force bf16, error if unsupported |
| `fp16` | pre-Ampere hardware only |
| `fp32` | debugging, and CPU runs |

Norm reductions and the loss are computed in fp32 regardless.

## Optimizers

**AdamW** (default) with fused kernels on CUDA. Weight decay applies to matrices
only — decaying a norm gain or bias pulls it toward zero for no benefit.

**Muon** (`optimizer="muon"`) replaces the momentum buffer with its nearest
semi-orthogonal matrix via a Newton–Schulz iteration, for 2D hidden weights. It
typically reaches a given loss in noticeably fewer tokens than AdamW. It is
defined for matrices, so embeddings, the LM head, biases and norm gains stay on
AdamW; `create_hybrid_optimizer` wires that split and `Trainer` steps both from
one schedule.

```python
TrainingConfig(optimizer="muon", muon_lr=0.02, learning_rate=3e-4)
```

`muon_lr` is roughly 50–100× the AdamW rate — the orthogonalised update has unit
scale by construction, so the two are not comparable numbers.

## Schedules

| `lr_schedule` | Shape |
|---|---|
| `cosine` | warmup, then cosine decay to `min_lr_ratio` |
| `wsd` | warmup, hold the peak for `stable_ratio`, then decay |
| `linear` | warmup, then linear decay |
| `constant` | warmup, then flat |

WSD is the better default for pretraining. Cosine bakes the total token budget
into the curve, so extending a run means either restarting the schedule or
training at an already-decayed rate. WSD's flat middle lets you stop, add data
and continue; you only decay when you actually intend to finish.

Warmup is `(step + 1) / warmup_steps`, not `step / warmup_steps` — otherwise the
first optimizer step runs at `lr = 0` and is simply thrown away.

## Batch size

Effective batch = `batch_size × gradient_accumulation_steps × world_size`.
The trainer prints tokens per optimizer step at startup; that is the number to
hold constant when you change hardware.

Accumulation is numerically exact — `tests/test_training.py` asserts that
`(batch 2, accum 4)` and `(batch 8, accum 1)` produce identical weights. Under
DDP, non-boundary micro-batches run inside `no_sync()`, so gradients are
all-reduced once per optimizer step rather than once per micro-batch.

## Memory

In rough order of value per unit of slowdown:

1. **`gradient_checkpointing=True`** — recompute activations in the backward
   pass. Roughly √N activation memory for ~30% more compute. This is the one
   that turns "does not fit" into "fits".
2. **Lower `batch_size`, raise `gradient_accumulation_steps`** — same effective
   batch, lower peak.
3. **Lower `ssm_chunk_size`** — for Mamba/hybrid, bounds the scan's transient
   slab. Identical numerics. The default of 64 was the fastest *and* leanest
   setting measured, so this is a last resort, not a first knob.
4. **Lower `kv_heads`** — smaller KV cache. This is a model change, not a
   training flag.

## Scaling out

```bash
torchrun --nproc_per_node=8 scripts/pretrain.py --tokens data/tokens/train.bin
```

`ava.utils.setup_distributed()` reads torchrun's environment and returns
`(rank, world_size, device)`; it degrades to `(0, 1, device)` in a single
process, so the same script works either way. `wrap_ddp` is a no-op when no
process group is active.

`wrap_ddp` sets `static_graph=True`. That is required, not cosmetic: DDP marks
each parameter ready once per backward pass, and activation checkpointing
touches parameters again during recomputation, which otherwise raises
*"Expected to mark a variable ready only once"*. It also enables
`gradient_as_bucket_view=True`, saving one full copy of the gradients.

DDP replicates the full model on every device, so it covers models up to roughly
the `3b` preset on 80 GB cards. Anything larger needs FSDP or tensor
parallelism, which this framework does not implement — the `13b`+ presets are
there to describe shapes, not to promise they will fit.

## Checkpointing and resume

```python
TrainingConfig(save_every=1000, keep_last=3)
```

Checkpoints carry model, optimizer(s), scheduler(s), scaler, step counter and
RNG state. Resume with:

```python
train_model(model, loader, training_config=config, resume_from="checkpoints/ava_step_5000.pt")
```

Restoring weights alone restarts the LR schedule from zero and discards the Adam
moments. That shows up as a loss spike right after every resume, and it is
routinely misdiagnosed as a data problem.

Only rank 0 writes. `ava_best.pt` is kept separately and never pruned.

## Reading the log

```
step   1200 | loss 3.4127 | ppl    30.35 | lr 2.87e-04 | grad 0.41 | 48,213 tok/s | mfu 34.2% | 612s
```

- **loss** — mean over the window since the last log line, not a running average
  over the whole epoch. A windowed mean actually responds when something changes.
- **grad** — pre-clip gradient norm. Steadily rising means trouble; occasional
  spikes are normal.
- **tok/s** — measured over a sliding window across all ranks.
- **mfu** — model FLOPs utilisation against the device's dense bf16 peak. 30–50%
  is healthy for a small model on one GPU. Shown only for recognised hardware.

There is exactly one GPU synchronisation per log interval. Per-batch `.item()`
calls are the most common accidental throughput killer in a training loop.

## `torch.compile`

```python
TrainingConfig(compile_model=True)
```

Worth 1.3–2× on a modern GPU, at the cost of a slow first step. Checkpointing
and the returned model both unwrap `_orig_mod` automatically, so a compiled run
produces a normal state dict.

## Evaluation

```python
from ava.training import evaluate_model
loss = evaluate_model(model, val_loader, device, autocast_dtype=torch.bfloat16)
```

Token-weighted, not batch-weighted: with variable-length batches a plain mean
over batches over-weights the short ones. `model.training` is restored on exit.
