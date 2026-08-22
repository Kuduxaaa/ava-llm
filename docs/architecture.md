# Architecture

Ava builds three stacks out of the same parts. `AvaConfig.architecture_type`
picks one, and `config.layer_types()` returns the resulting per-layer plan —
index-aligned with `model.model.layers`, so nothing downstream has to guess what
kind of layer it is looking at.

```python
AvaConfig.from_preset("hybrid-130m").layer_types()
# ['mamba'] * 10 + ['attention'] * 2
```

---

## The decoder layer

Pre-norm residual, as in every modern decoder:

```
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

Pre-norm rather than post-norm because the residual stream stays an unmodified
identity path — gradients reach layer 0 without passing through a normalisation
at every step, which is what makes deep stacks trainable without warmup tricks.

### Attention

Grouped-query attention (`ava/model/attention.py`). `num_attention_heads`
queries share `kv_heads` key/value heads, so the KV cache shrinks by
`num_attention_heads / kv_heads` — an 8:1 ratio is the usual setting and cuts
decode-time memory bandwidth by the same factor, which is the actual bottleneck
during generation.

Three details are load-bearing:

**RoPE is applied before caching.** Keys enter the cache already rotated for the
absolute position they were computed at, so they are never rotated a second
time. Positions come from explicit `position_ids`, never from a tensor shape.

**QK-norm** (`config.qk_norm`, on by default) puts an RMSNorm on each head of Q
and K before the rotation. Attention logits are a dot product of two learned
vectors with nothing bounding their norm; in long bf16 runs they drift upward
until softmax saturates and gradients vanish. Normalising the inputs removes the
failure mode outright.

**The mask is `None` whenever it can be.** `build_causal_mask` returns `None`
for an unpadded batch so SDPA takes its fused `is_causal` kernel instead of
reading a materialised bias tensor. When a real mask is needed, the diagonal is
explicitly unmasked so no query row can end up fully masked — a fully masked row
makes softmax emit NaN and poisons the entire batch.

### MLP

SwiGLU: `down(silu(gate(x)) * up(x))`. Three matrices instead of two, so
`intermediate_size` is conventionally ~8/3 × `hidden_size` rather than 4× to
keep the parameter count comparable. Other gated activations are available
through `hidden_act`.

---

## The Mamba block

```
h = h + SelectiveSSM(RMSNorm(h))
```

A selective state-space model (`ava/model/mamba.py`) with input-dependent `dt`,
`B` and `C`:

```
x, z   = in_proj(h).chunk(2)          # content and gate
x      = silu(depthwise_conv1d(x))    # short-range mixing
dt,B,C = split(x_proj(x))             # per-token, per-channel parameters
A      = -exp(A_log)                  # stable by construction
h[t]   = exp(dt[t] * A) * h[t-1] + dt[t] * B[t] * x[t]
y[t]   = sum(h[t] * C[t]) + D * x[t]
out    = out_proj(y * silu(z))
```

### Why the scan is chunked

The naive way to evaluate that recurrence in parallel materialises an
`(batch, seq_len, inner_dim, d_state)` tensor. For `hybrid-1b` at 8k context and
batch 8 that is well over 100 GB — which is how a "linear-time" model manages to
OOM where a quadratic transformer does not.

`_chunked_scan` instead runs a parallel prefix scan inside a window of
`ssm_chunk_size` steps and carries one `(batch, inner_dim, d_state)` state across
windows, gradient-checkpointing each window.

**Where the checkpoint boundary sits is the whole mechanism.** A checkpoint
saves its *inputs* so it can recompute in the backward pass. Build `a_bar` and
`b_x` in the caller and they become those inputs — one
`(batch, chunk, inner_dim, d_state)` slab retained per window, so total memory
scales with `seq_len` again and chunking buys nothing. `_ssm_window` therefore
discretises, scans and projects out inside the checkpointed function; only
`(batch, chunk, inner_dim)` and `(batch, chunk, d_state)` cross the boundary.

Getting that placement wrong is invisible from the outside: results are
identical, the loop looks chunked, and memory simply refuses to fall. Measured
on `mamba-130m` at batch 2 × 512 tokens, moving the slab inside the checkpoint
took peak memory from 6.6 GB to 2.1 GB and training from 46 to 206 tok/s.

The window size defaults to 64 and shrinks only for unusually wide `d_state`.

Chunk size is a memory knob only — the numbers are identical at any setting
(`tests/test_mamba.py::test_chunking_does_not_change_the_result`).

The carried state is `clone()`d rather than returned as a view of the chunk. A
view would keep the entire slab alive inside the decoding cache, long after the
chunk it came from was finished with — shape-checking the cache does not catch
that, so `test_cached_state_does_not_alias_the_scan_slab` checks storage size.

### Why `dt` is initialised the way it is

`dt_proj.bias` is set so `softplus(bias)` is log-uniform over `[1e-3, 1e-1]`.
Left at zero, every channel starts with the same timescale and the model spends
its first thousands of steps just learning to differentiate them.

---

## The cache

`AvaCache` (`ava/model/cache.py`) holds one entry per layer, and the entry type
matches the layer type:

| Layer | Cache | Shape | Growth |
|---|---|---|---|
| attention | `AttentionLayerCache` | `(batch, kv_heads, seen, head_dim)` × 2 | linear in context |
| mamba | `MambaLayerCache` | `(batch, inner, d_state)` + `(batch, inner, d_conv-1)` | constant |

A single `cache.seen_tokens` counter is the authoritative position offset, so
nothing has to reverse-engineer it from a tensor shape.

This is what makes hybrid decoding work at all. With the older
"tuple of key/value tuples" convention there is nowhere to put an SSM state, so
a Mamba layer contributes `None`, the generation loop still believes a cache
exists, and every step after the first feeds the SSM a single token with no
history — the model keeps producing plausible tokens while having silently
forgotten the prompt.

### The invariant

Running a sequence in one shot and running it token-by-token must produce the
same logits:

```python
reference = model(input_ids=ids).logits

cache, pieces = AvaCache.from_config(model.config), []
for t in range(ids.shape[1]):
    out = model(input_ids=ids[:, t:t+1], cache=cache, use_cache=True)
    pieces.append(out.logits)

torch.testing.assert_close(torch.cat(pieces, 1), reference)
```

`tests/test_cache.py` asserts this for all three architectures. It is the single
highest-value test in the repository.

---

## Initialisation

- Linear and embedding weights: `N(0, initializer_range)`, default 0.02.
- Output projections (`o_proj`, `down_proj`, `out_proj`) are scaled by
  `1/sqrt(2 · num_hidden_layers)`. Every layer writes into the same residual
  stream, so without this the stream's variance grows linearly in depth and a
  deep model starts far outside the range its norms expect.
- Layers with a purpose-built init (Mamba's `dt` schedule) mark their tensors
  `_no_reinit` and are skipped by the global pass.
- `nn.Conv1d` is left at PyTorch's fan-in default. Mamba's depthwise kernel has
  a fan-in of `d_conv` (4), so that default is ~0.29; applying
  `initializer_range` (0.02) to it instead attenuates the entire SSM branch by
  more than 10×, leaving the residual stream an identity path. With tied
  embeddings that degenerates into "predict the current token" and starts
  training well *above* `ln(vocab_size)` — measured at 15.2 against a uniform
  9.0 for `mamba-130m`, before the branch was allowed to contribute.

---

## Numerical stability

| Mechanism | Where | What it prevents |
|---|---|---|
| fp32 reduction in RMSNorm | `normalization.py` | variance underflow in bf16 |
| QK-norm | `attention.py` | attention-logit growth over long runs |
| z-loss (`z_loss_coef`) | `ava_model.py` | logit drift, late-run loss spikes |
| `A = -exp(A_log)` | `mamba.py` | the SSM cannot become unstable |
| unmasked diagonal | `attention.py` | NaN from a fully masked query row |
| scaled residual init | `ava_model.py` | activation blow-up with depth |
