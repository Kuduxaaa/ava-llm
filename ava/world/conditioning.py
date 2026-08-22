"""How the internal world reaches the language model.

The interesting requirement is that the world influences what Ava says without
being *told* to her. Writing "you are feeling anxious (0.71)" into the prompt
makes a model describe anxiety; it does not make it anxious, and it puts the
state where the user can read it. A state that only matters when it is narrated
is not an internal world.

So conditioning happens inside the network instead:

**FiLM on the residual stream.** Each conditioned layer gets a per-channel scale
and shift derived from the world, applied as ``h * (1 + gamma) + beta``. This is
a bias on how the model computes, not a fact it reasons about -- closer to being
in a mood than to knowing you are in one.

**A soft prefix**, optionally. A handful of continuous tokens the attention can
look at. More expressive, and more likely to be reflected explicitly in the
output, which is sometimes what you want.

Both are zero-initialised, so an untrained conditioner is exactly a no-op and
attaching one never damages a model that was trained without it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .schema import NUM_CHANNELS
from .state import WorldState


class WorldConditioner(nn.Module):
    """Projects a world state into per-layer FiLM parameters and a soft prefix.

    Args:
        hidden_size: the language model's residual width.
        num_layers: how many layers exist in the stack.
        conditioning_layers: which of them to modulate. ``None`` conditions all.
            Conditioning a few is usually enough and much cheaper; early layers
            bias representation, late layers bias expression.
        width: bottleneck width of the shared trunk.
        num_prefix_tokens: soft prompt length, 0 to disable.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        conditioning_layers: tuple[int, ...] | None = None,
        width: int = 128,
        num_prefix_tokens: int = 0,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.width = width
        self.num_prefix_tokens = num_prefix_tokens
        self.scale = scale

        if conditioning_layers is None:
            conditioning_layers = tuple(range(num_layers))
        for index in conditioning_layers:
            if not 0 <= index < num_layers:
                raise ValueError(
                    f"conditioning layer {index} is outside a {num_layers}-layer stack."
                )
        self.conditioning_layers = tuple(sorted(set(conditioning_layers)))

        self.trunk = nn.Sequential(
            nn.Linear(NUM_CHANNELS, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )

        self.film = nn.ModuleDict(
            {
                str(index): nn.Linear(width, 2 * hidden_size)
                for index in self.conditioning_layers
            }
        )
        for layer in self.film.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        if num_prefix_tokens > 0:
            self.prefix = nn.Linear(width, num_prefix_tokens * hidden_size)
            nn.init.zeros_(self.prefix.weight)
            nn.init.zeros_(self.prefix.bias)
            self.prefix_embedding = nn.Parameter(
                torch.zeros(num_prefix_tokens, hidden_size)
            )
            nn.init.normal_(self.prefix_embedding, std=0.02)
        else:
            self.prefix = None

    # --- forward ---

    def encode(self, state: WorldState | torch.Tensor) -> torch.Tensor:
        values = state.values if isinstance(state, WorldState) else state
        if values.shape[-1] != NUM_CHANNELS:
            raise ValueError(
                f"Expected a {NUM_CHANNELS}-channel world state, got {values.shape[-1]}."
            )
        return self.trunk(values.to(next(self.parameters()).dtype))

    def forward(
        self, state: WorldState | torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer ``(gamma, beta)``, each ``(batch, 1, hidden_size)``."""
        code = self.encode(state)
        modulation = {}
        for index in self.conditioning_layers:
            gamma, beta = self.film[str(index)](code).chunk(2, dim=-1)
            modulation[index] = (
                self.scale * torch.tanh(gamma).unsqueeze(1),
                self.scale * beta.unsqueeze(1),
            )
        return modulation

    def prefix_tokens(self, state: WorldState | torch.Tensor) -> torch.Tensor | None:
        """``(batch, num_prefix_tokens, hidden_size)`` soft prompt, or ``None``."""
        if self.prefix is None:
            return None
        code = self.encode(state)
        batch = code.shape[0]
        delta = self.prefix(code).view(batch, self.num_prefix_tokens, self.hidden_size)
        return self.prefix_embedding.unsqueeze(0) + self.scale * delta

    def extra_repr(self) -> str:
        return (
            f"layers={list(self.conditioning_layers)}, width={self.width}, "
            f"prefix={self.num_prefix_tokens}, scale={self.scale}"
        )


def apply_film(
    hidden_states: torch.Tensor, modulation: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """``h * (1 + gamma) + beta``, broadcast over the sequence."""
    if modulation is None:
        return hidden_states
    gamma, beta = modulation
    return hidden_states * (1 + gamma.to(hidden_states.dtype)) + beta.to(
        hidden_states.dtype
    )


class WorldSummary:
    """A short natural-language rendering of the world, for when you *do* want it.

    Useful for logs, for debugging, and for the case where a system prompt is the
    only lever available. Deliberately not the primary path: everything it says
    is something the user can read, and a state that has to be narrated to have
    an effect is not an internal state.
    """

    def __init__(self, threshold: float = 0.12, top_k: int = 4) -> None:
        self.threshold = threshold
        self.top_k = top_k

    def __call__(self, state: WorldState, index: int = 0) -> str:
        parts = []
        vad = (
            state.get("output.valence", index),
            state.get("output.arousal", index),
            state.get("output.dominance", index),
        )
        parts.append(f"valence {vad[0]:.2f}, arousal {vad[1]:.2f}, dominance {vad[2]:.2f}")

        salient = [
            f"{key.split('.')[-1]} {'+' if delta > 0 else ''}{delta:.2f}"
            for key, delta in state.deviations(self.top_k * 3, index)
            if abs(delta) >= self.threshold
        ][: self.top_k]
        if salient:
            parts.append("; ".join(salient))
        return " | ".join(parts)
