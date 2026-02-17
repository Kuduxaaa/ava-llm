import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .normalization import AvaRMSNorm


def _associative_scan(gates, values):
    """Parallel prefix scan (Hillis-Steele) for linear recurrence.

    Computes h[t] = gates[t] * h[t-1] + values[t], h[-1] = 0.
    O(log L) sequential steps — optimal for GPU parallelism.
    """
    L = gates.shape[1]
    num_steps = int(math.ceil(math.log2(max(L, 2))))

    for i in range(num_steps):
        stride = 2 ** i
        if stride >= L:
            break
        new_g = gates[:, stride:] * gates[:, :L - stride]
        new_v = gates[:, stride:] * values[:, :L - stride] + values[:, stride:]
        gates = torch.cat([gates[:, :stride], new_g], dim=1)
        values = torch.cat([values[:, :stride], new_v], dim=1)

    return values


def _cpu_sequential_scan(x, dt, A, B, C, d_state):
    """Per-step sequential scan optimized for CPU cache locality.

    Computes exp/discretize inline per timestep with small tensor ops.
    """
    batch, seq_len, inner_dim = x.shape
    h = torch.zeros(batch, inner_dim, d_state, device=x.device, dtype=x.dtype)
    outputs = torch.empty(batch, seq_len, inner_dim, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        dt_t = dt[:, t, :].unsqueeze(-1)         # (B, D, 1)
        A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, D, N)
        B_bar = dt_t * B[:, t, :].unsqueeze(1)    # (B, D, N)
        x_t = x[:, t, :].unsqueeze(-1)            # (B, D, 1)

        h = A_bar * h + B_bar * x_t               # (B, D, N)
        outputs[:, t] = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)

    return outputs


class SelectiveSSM(nn.Module):
    """Selective State Space Model with adaptive scan strategy.

    GPU: parallel prefix scan with gradient checkpointing — O(log L) depth
    CPU: per-step sequential scan — cache-friendly O(L) with small tensors
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = d_model * expand

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.inner_dim, bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner_dim, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.inner_dim, -1)))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Input projection: x branch + gate z
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv1d + SiLU
        x_branch = F.silu(
            self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        )

        # SSM parameters (fully vectorized)
        ssm_params = self.x_proj(x_branch)
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:self.d_state * 2]
        dt = F.softplus(self.dt_proj(ssm_params[..., -1:]))
        A = -torch.exp(self.A_log)

        if x.is_cuda:
            # GPU path: vectorized pre-compute + parallel scan + grad checkpoint
            dt_4d = dt.unsqueeze(-1)
            A_bar = torch.exp(dt_4d * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
            B_x = dt_4d * B.unsqueeze(2) * x_branch.unsqueeze(-1)   # (B, L, D, N)
            h = grad_checkpoint(_associative_scan, A_bar, B_x, use_reentrant=False)
            y = (h * C.unsqueeze(2)).sum(dim=-1)
        else:
            # CPU path: per-step sequential (cache-friendly)
            y = _cpu_sequential_scan(x_branch, dt, A, B, C, self.d_state)

        # Skip connection + gating
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Mamba block: Pre-LN + SelectiveSSM + residual."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_mamba = True

        self.norm = AvaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.ssm = SelectiveSSM(
            d_model=config.hidden_size,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        rotary_emb=None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.ssm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (None,)
        if output_attentions:
            outputs += (None,)
        return outputs
