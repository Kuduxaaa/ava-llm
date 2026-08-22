"""Selective state-space (Mamba) blocks in pure PyTorch.

Two things separate this from a textbook implementation:

**Bounded memory.** The naive selective scan materialises an
``(batch, seq_len, inner_dim, d_state)`` tensor -- for a 1B hybrid at 8k context
that is tens of gigabytes, which is why a "linear-time" model can OOM where a
quadratic transformer does not. The scan here is *chunked*: it runs a parallel
prefix scan inside a window of ``ssm_chunk_size`` steps and carries a single
``(batch, inner_dim, d_state)`` state across windows, gradient-checkpointing
each window. The ``d_state`` dimension exists only *inside* that checkpoint --
see :func:`_ssm_window` for why that placement, rather than the loop itself, is
what actually bounds the memory.

**Real recurrent decoding.** The block returns and accepts an explicit SSM state
and a depthwise-conv lookback window, so generating token *t+1* costs the same
as generating token 1. Without that, an SSM has no incremental mode at all and
generation silently degrades to "predict from a one-token context".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import AvaConfig
from .cache import MambaLayerCache
from .normalization import AvaRMSNorm


def _prefix_scan(
    gates: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele inclusive scan of ``h[t] = gates[t] * h[t-1] + values[t]``.

    ``O(log L)`` sequential steps over dimension 1 instead of ``O(L)``, which is
    the difference between a scan the GPU can saturate and a Python loop.
    Returns the scanned values *and* the cumulative gate products -- the latter
    is what folds in a state carried from a previous chunk.
    """
    length = gates.shape[1]
    stride = 1
    while stride < length:
        shifted_gates = gates[:, : length - stride]
        shifted_values = values[:, : length - stride]
        values = torch.cat(
            [
                values[:, :stride],
                gates[:, stride:] * shifted_values + values[:, stride:],
            ],
            dim=1,
        )
        gates = torch.cat([gates[:, :stride], gates[:, stride:] * shifted_gates], dim=1)
        stride *= 2
    return values, gates


def _scan_chunk(
    a_bar: torch.Tensor, b_x: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan one chunk given an incoming ``state``; return all states and the last.

    ``a_bar`` and ``b_x`` are ``(batch, chunk, inner_dim, d_state)`` and
    ``state`` is ``(batch, inner_dim, d_state)``.

    The carried state is ``clone()``d rather than returned as a view. A view of
    the last timestep keeps the whole ``(batch, chunk, inner_dim, d_state)``
    slab alive -- which would pin gigabytes inside a decoding cache long after
    the chunk itself is finished with.
    """
    local, cumulative_gates = _prefix_scan(a_bar, b_x)
    states = local + cumulative_gates * state.unsqueeze(1)
    return states, states[:, -1].clone()


def _ssm_window(
    dt: torch.Tensor,
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    c_matrix: torch.Tensor,
    x: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretise, scan and project one window, in a single checkpointable unit.

    Everything of size ``(batch, chunk, inner_dim, d_state)`` -- the discretised
    ``A``, the input term, and the per-step states -- is born and dies inside
    this function. That placement is the whole point: tensors created *outside*
    a checkpoint are saved as its inputs, so computing ``a_bar`` in the caller
    would retain one such slab per window and make total memory scale with
    ``seq_len`` again, exactly what chunking is meant to avoid.

    What crosses the boundary is only ``O(chunk * inner_dim)`` and
    ``O(chunk * d_state)``, smaller by a factor of ``d_state``.
    """
    dt = dt.unsqueeze(-1)  # (b, chunk, inner, 1)
    a_bar = torch.exp(dt * a_matrix)  # (b, chunk, inner, state)
    b_x = dt * b_matrix.unsqueeze(2) * x.unsqueeze(-1)

    states, last_state = _scan_chunk(a_bar, b_x, state)
    y = (states * c_matrix.unsqueeze(2)).sum(-1)  # (b, chunk, inner)
    return y, last_state


class SelectiveSSM(nn.Module):
    """The S6 layer: input-dependent ``dt``, ``B`` and ``C``."""

    def __init__(self, config: AvaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.inner_dim = config.ssm_inner_dim
        self.dt_rank = config.ssm_dt_rank
        self.chunk_size = config.ssm_effective_chunk_size

        self.in_proj = nn.Linear(self.d_model, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.inner_dim,
            bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # S4D-real initialisation: A = -diag(1..N), stored in log space so the
        # sign can never flip during training and the system stays stable.
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).expand(self.inner_dim, -1).contiguous())
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self._init_dt_bias()

    def _init_dt_bias(self, dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        """Bias ``dt`` so ``softplus(bias)`` is log-uniform over ``[dt_min, dt_max]``.

        Without this the timescales all start identical and the model spends the
        first few thousand steps just learning to differentiate them.
        """
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(1e-4)
        # Inverse of softplus.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        self.dt_proj.weight._no_reinit = True

    # --- parallel (training / prefill) path ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: MambaLayerCache | None = None,
    ) -> torch.Tensor:
        if (
            cache is not None
            and cache.ssm_state is not None
            and hidden_states.shape[1] == 1
        ):
            return self.step(hidden_states, cache)

        batch, seq_len, _ = hidden_states.shape

        x_branch, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)  # (b, inner, seq)

        if cache is not None:
            # Keep the last d_conv-1 inputs so the next single-token step sees
            # the same receptive field it would have seen in the parallel pass.
            lookback = self.d_conv - 1
            window = x_branch[:, :, -lookback:].detach()
            cache.conv_state = F.pad(window, (lookback - window.shape[-1], 0))

        x_branch = self.conv1d(x_branch)[:, :, :seq_len].transpose(1, 2)
        x_branch = F.silu(x_branch)

        dt, b_matrix, c_matrix = self._project_ssm_params(x_branch)
        a_matrix = -torch.exp(self.A_log.float())

        state = (
            cache.ssm_state
            if cache is not None and cache.ssm_state is not None
            else x_branch.new_zeros(batch, self.inner_dim, self.d_state)
        )
        y, state = self._chunked_scan(x_branch, dt, a_matrix, b_matrix, c_matrix, state)

        if cache is not None:
            cache.ssm_state = state.detach()

        y = y + self.D.float() * x_branch
        y = y.to(gate.dtype) * F.silu(gate)
        return self.out_proj(y)

    def _project_ssm_params(
        self, x_branch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.x_proj(x_branch)
        dt_low_rank, b_matrix, c_matrix = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low_rank))
        return dt.float(), b_matrix.float(), c_matrix.float()

    def _chunked_scan(
        self,
        x_branch: torch.Tensor,
        dt: torch.Tensor,
        a_matrix: torch.Tensor,
        b_matrix: torch.Tensor,
        c_matrix: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the selective scan in fixed-size windows.

        Retained activation memory is ``O(batch * seq_len * inner_dim)`` rather
        than ``O(batch * seq_len * inner_dim * d_state)``: the ``d_state``
        dimension only ever exists inside :func:`_ssm_window`, which is
        gradient-checkpointed and so recomputes it in the backward pass.

        ``config.ssm_chunk_size`` then trades the transient peak against the
        number of kernel launches; it does not change the result.
        """
        seq_len = x_branch.shape[1]
        x_float = x_branch.float()
        checkpointing = self.training and torch.is_grad_enabled()
        outputs = []

        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            window = (
                dt[:, start:stop],
                a_matrix,
                b_matrix[:, start:stop],
                c_matrix[:, start:stop],
                x_float[:, start:stop],
                state,
            )
            if checkpointing:
                y, state = checkpoint(_ssm_window, *window, use_reentrant=False)
            else:
                y, state = _ssm_window(*window)
            outputs.append(y)

        return torch.cat(outputs, dim=1), state

    # --- recurrent (decode) path ---

    def step(self, hidden_states: torch.Tensor, cache: MambaLayerCache) -> torch.Tensor:
        """Advance one token using the cached state. Constant cost per token."""
        x_branch, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)  # (b, inner, 1)

        conv_window = torch.cat([cache.conv_state, x_branch], dim=-1)
        cache.conv_state = conv_window[:, :, 1:]

        x_conv = (conv_window * self.conv1d.weight.squeeze(1)).sum(-1)
        x_conv = F.silu(x_conv + self.conv1d.bias).unsqueeze(1)  # (b, 1, inner)

        dt, b_matrix, c_matrix = self._project_ssm_params(x_conv)
        a_matrix = -torch.exp(self.A_log.float())

        dt = dt.squeeze(1).unsqueeze(-1)  # (b, inner, 1)
        a_bar = torch.exp(dt * a_matrix)
        b_x = (
            dt * b_matrix.squeeze(1).unsqueeze(1) * x_conv.float().squeeze(1).unsqueeze(-1)
        )

        cache.ssm_state = a_bar * cache.ssm_state + b_x
        y = (cache.ssm_state * c_matrix.squeeze(1).unsqueeze(1)).sum(-1).unsqueeze(1)

        y = y + self.D.float() * x_conv.float()
        y = y.to(gate.dtype) * F.silu(gate)
        return self.out_proj(y)

    def allocate_cache(self, batch: int, device, dtype) -> MambaLayerCache:
        return MambaLayerCache(
            ssm_state=torch.zeros(
                batch, self.inner_dim, self.d_state, device=device, dtype=torch.float32
            ),
            conv_state=torch.zeros(
                batch, self.inner_dim, self.d_conv - 1, device=device, dtype=dtype
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, "
            f"dt_rank={self.dt_rank}, chunk={self.chunk_size}"
        )


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around :class:`SelectiveSSM`."""

    is_mamba = True

    def __init__(self, config: AvaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.ssm = SelectiveSSM(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_cache: MambaLayerCache | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Signature mirrors :class:`AvaDecoderLayer` positionally so that a
        hybrid stack can call either layer type the same way -- including the
        positional-only call that ``torch.utils.checkpoint`` makes."""
        residual = hidden_states
        hidden_states = self.ssm(self.norm(hidden_states), cache=layer_cache)
        return residual + hidden_states, None
