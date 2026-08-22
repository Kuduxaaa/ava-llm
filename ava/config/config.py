"""Model configuration for the Ava family."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

ArchitectureType = Literal["transformer", "mamba", "hybrid"]


@dataclass
class AvaConfig:
    """Architecture hyper-parameters.

    Only structural facts live here. Anything about *how* a model is trained
    belongs in :class:`ava.training.TrainingConfig`.
    """

    # --- core transformer ---
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    kv_heads: int | None = None
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048

    # --- normalisation / stability ---
    rms_norm_eps: float = 1e-5
    qk_norm: bool = True
    """Per-head RMSNorm on Q and K before RoPE. Standard since OLMo 2 / Gemma 2 --
    removes the attention-logit blow-up that kills long bf16 runs."""
    z_loss_coef: float = 1e-4
    """Coefficient for the log-partition (logsumexp squared) auxiliary loss that
    keeps output logits from drifting. Set to 0.0 to disable."""
    scaled_residual_init: bool = True
    """Scale output projections by 1/sqrt(2 * num_hidden_layers) at init."""

    # --- position encoding ---
    rope_theta: float = 500000.0
    rope_scaling: dict[str, Any] | None = None
    """None, or e.g. {"type": "yarn", "factor": 4.0,
    "original_max_position_embeddings": 8192}."""

    # --- regularisation ---
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    # --- embeddings / head ---
    tie_word_embeddings: bool = True

    # --- special tokens ---
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # --- mamba / SSM ---
    architecture_type: ArchitectureType = "transformer"
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    """Rank of the low-rank dt projection. Defaults to ceil(hidden_size / 16)."""
    ssm_chunk_size: int | None = None
    """Chunk length for the chunked associative scan, or ``None`` to derive one.

    The scan's transient activation is proportional to
    ``chunk * ssm_inner_dim * d_state``. Leaving this at ``None`` picks a window
    that is fast for ordinary shapes and shrinks it only when ``d_state`` is
    wide enough to make that expensive -- see :attr:`ssm_effective_chunk_size`."""
    num_attention_layers: int = 2
    """Hybrid only: how many trailing layers are attention rather than Mamba."""

    # --- runtime ---
    use_cache: bool = True
    gradient_checkpointing: bool = False
    model_type: str = "ava"

    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} is not divisible by "
                    f"num_attention_heads={self.num_attention_heads}; "
                    "set head_dim explicitly."
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.kv_heads is None:
            self.kv_heads = self.num_attention_heads

        self._validate()

    def _validate(self) -> None:
        if self.num_attention_heads % self.kv_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be a multiple "
                f"of kv_heads={self.kv_heads} for grouped-query attention."
            )
        if self.architecture_type not in ("transformer", "mamba", "hybrid"):
            raise ValueError(f"Unknown architecture_type: {self.architecture_type!r}")
        if self.architecture_type == "hybrid" and not (
            0 < self.num_attention_layers < self.num_hidden_layers
        ):
            raise ValueError(
                "hybrid models need 0 < num_attention_layers < "
                f"num_hidden_layers, got {self.num_attention_layers} and "
                f"{self.num_hidden_layers}."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}.")
        if self.ssm_chunk_size is not None and self.ssm_chunk_size < 1:
            raise ValueError(
                f"ssm_chunk_size must be >= 1 or None, got {self.ssm_chunk_size}."
            )
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("type")
            if rope_type not in ("linear", "ntk", "yarn"):
                raise ValueError(f"Unknown rope_scaling type: {rope_type!r}")

    # --- derived properties ---

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.kv_heads

    @property
    def attention_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def has_attention(self) -> bool:
        return self.architecture_type in ("transformer", "hybrid")

    @property
    def has_mamba(self) -> bool:
        return self.architecture_type in ("mamba", "hybrid")

    @property
    def ssm_inner_dim(self) -> int:
        return self.hidden_size * self.expand

    @property
    def ssm_dt_rank(self) -> int:
        return self.dt_rank or max(1, -(-self.hidden_size // 16))

    @property
    def ssm_effective_chunk_size(self) -> int:
        """Scan window: the explicit ``ssm_chunk_size``, or one derived from shape.

        The chunked scan holds a handful of ``(batch, chunk, inner_dim,
        d_state)`` fp32 slabs at once, so the memory that matters scales with
        ``chunk * inner_dim * d_state``. Solving for a fixed per-sequence
        element budget keeps that product roughly constant across model sizes
        instead of letting it grow by 16x between the 130M and 1B hybrids.

        This is the honest cost of a pure-PyTorch selective scan: a fused kernel
        would not materialise these slabs at all. Set ``ssm_chunk_size``
        explicitly to trade memory back for fewer, larger kernel launches.
        """
        if self.ssm_chunk_size is not None:
            return self.ssm_chunk_size

        # 64 measured fastest across both SSM presets, and it is not a close
        # call: smaller windows spend their time in the Python loop, larger
        # ones in memory traffic on the slabs. The shape-derived budget only
        # takes over for state widths large enough to make 64 expensive.
        budget = 2**24  # transient slab ceiling, ~64 MB fp32 per sequence
        derived = budget // max(1, self.ssm_inner_dim * self.d_state)
        return max(16, min(64, derived))

    def layer_types(self) -> list[str]:
        """The per-layer plan, index-aligned with ``AvaModel.layers``."""
        n = self.num_hidden_layers
        if self.architecture_type == "transformer":
            return ["attention"] * n
        if self.architecture_type == "mamba":
            return ["mamba"] * n
        num_mamba = n - self.num_attention_layers
        return ["mamba"] * num_mamba + ["attention"] * self.num_attention_layers

    def estimate_parameters(self) -> int:
        """Analytic parameter count -- useful before allocating anything."""
        h, v = self.hidden_size, self.vocab_size
        total = h * v  # input embeddings
        if not self.tie_word_embeddings:
            total += h * v

        mlp = 3 * h * self.intermediate_size
        for layer_type in self.layer_types():
            if layer_type == "attention":
                total += h * self.attention_dim  # q_proj
                total += 2 * h * self.kv_heads * self.head_dim  # k_proj, v_proj
                total += self.attention_dim * h  # o_proj
                total += mlp + 2 * h  # + input/post-attention norms
                if self.qk_norm:
                    total += 2 * self.head_dim
            else:
                inner = self.ssm_inner_dim
                total += h * inner * 2  # in_proj
                total += inner * self.d_conv + inner  # depthwise conv + bias
                total += inner * (self.ssm_dt_rank + 2 * self.d_state)  # x_proj
                total += self.ssm_dt_rank * inner + inner  # dt_proj + bias
                total += inner * self.d_state + inner  # A_log, D
                total += inner * h  # out_proj
                total += h  # norm
        return total + h  # final norm

    # --- serialisation ---

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("_extra", {})
        return {**data, **extra}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvaConfig:
        known = {f.name for f in fields(cls)} - {"_extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        config = cls(**kwargs)
        config._extra = extra
        return config

    def save_pretrained(self, path: str | os.PathLike) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike) -> AvaConfig:
        config_file = os.path.join(path, "config.json")
        if not os.path.isfile(config_file):
            config_file = str(path)
        with open(config_file, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # --- presets ---

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> AvaConfig:
        """Build a config from a named preset, e.g. ``AvaConfig.from_preset("1b")``."""
        if name not in PRESETS:
            available = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset {name!r}. Available: {available}")
        return cls(**{**PRESETS[name], **overrides})

    @classmethod
    def available_presets(cls) -> list[str]:
        return sorted(PRESETS)

    def apply_for(self, model: str = "1b") -> AvaConfig:
        """Deprecated in-place variant of :meth:`from_preset`."""
        warnings.warn(
            "AvaConfig().apply_for(name) is deprecated; "
            "use AvaConfig.from_preset(name) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model not in PRESETS:
            raise ValueError(f"Configuration for {model!r} is not defined.")
        for key, value in PRESETS[model].items():
            setattr(self, key, value)
        self.__post_init__()
        return self


# fmt: off
PRESETS: dict[str, dict[str, Any]] = {
    # --- tiny: edge devices, unit tests, ablations ---
    "130m": dict(
        hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
        num_attention_heads=12, kv_heads=4, head_dim=64,
        max_position_embeddings=2048, tie_word_embeddings=True,
    ),
    "350m": dict(
        hidden_size=1024, intermediate_size=2816, num_hidden_layers=24,
        num_attention_heads=16, kv_heads=4, head_dim=64,
        max_position_embeddings=4096, tie_word_embeddings=True,
    ),
    # --- small: on-device assistants, distillation targets ---
    "1b": dict(
        hidden_size=2048, intermediate_size=5632, num_hidden_layers=16,
        num_attention_heads=16, kv_heads=8, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=True,
    ),
    "3b": dict(
        hidden_size=2560, intermediate_size=6912, num_hidden_layers=32,
        num_attention_heads=20, kv_heads=4, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=True,
    ),
    # --- medium: general chat, code, reasoning ---
    "7b": dict(
        hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
        num_attention_heads=32, kv_heads=8, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=False,
    ),
    "13b": dict(
        hidden_size=5120, intermediate_size=13824, num_hidden_layers=40,
        num_attention_heads=40, kv_heads=8, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=False,
    ),
    # --- large: research scale ---
    "30b": dict(
        hidden_size=6656, intermediate_size=17920, num_hidden_layers=60,
        num_attention_heads=52, kv_heads=4, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=False,
    ),
    "70b": dict(
        hidden_size=8192, intermediate_size=28672, num_hidden_layers=80,
        num_attention_heads=64, kv_heads=8, head_dim=128,
        max_position_embeddings=8192, tie_word_embeddings=False,
    ),
    # --- state-space and hybrid ---
    "mamba-130m": dict(
        architecture_type="mamba",
        hidden_size=768, intermediate_size=2048, num_hidden_layers=24,
        num_attention_heads=12, kv_heads=4, head_dim=64,
        max_position_embeddings=2048, d_state=16, d_conv=4, expand=2,
        tie_word_embeddings=True,
    ),
    "hybrid-130m": dict(
        architecture_type="hybrid",
        hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
        num_attention_heads=12, kv_heads=4, head_dim=64,
        max_position_embeddings=2048, d_state=16, d_conv=4, expand=2,
        num_attention_layers=2, tie_word_embeddings=True,
    ),
    "hybrid-1b": dict(
        architecture_type="hybrid",
        hidden_size=2048, intermediate_size=5632, num_hidden_layers=24,
        num_attention_heads=16, kv_heads=8, head_dim=128,
        max_position_embeddings=8192, d_state=64, d_conv=4, expand=2,
        num_attention_layers=4, tie_word_embeddings=True,
    ),
}
# fmt: on
