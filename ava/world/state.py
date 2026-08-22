"""The world state tensor and its named views."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from . import schema
from .schema import CHANNELS, GROUP_ORDER, GROUP_SLICES, NUM_CHANNELS


class WorldState:
    """A batch of internal worlds, stored as one ``(batch, 136)`` tensor.

    The tensor is the state. Names are a lookup table over its columns, not a
    parallel structure -- so every dynamics step is a handful of elementwise
    tensor operations, the whole thing lives on the GPU beside the model, and
    gradients flow through it if you ever want to train the dynamics.

    Indexing is by qualified name, because three names are reused across groups
    (``fatigue``, ``uncertainty``, ``arousal``)::

        state["emotion.trust"]
        state["internal_state.stress"] = 0.7
        state.group("emotion")            # -> {"happiness": tensor, ...}
    """

    __slots__ = ("values",)

    def __init__(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[-1] != NUM_CHANNELS:
            raise ValueError(
                f"World state must be (batch, {NUM_CHANNELS}), got {tuple(values.shape)}."
            )
        self.values = values

    # --- construction ---

    @classmethod
    def baseline(
        cls, batch_size: int = 1, device=None, dtype: torch.dtype = torch.float32
    ) -> WorldState:
        """A rested, neutral world: every channel at its setpoint."""
        row = torch.tensor(schema.baselines(), device=device, dtype=dtype)
        return cls(row.expand(batch_size, -1).clone())

    @classmethod
    def zeros(
        cls, batch_size: int = 1, device=None, dtype: torch.dtype = torch.float32
    ) -> WorldState:
        return cls(torch.zeros(batch_size, NUM_CHANNELS, device=device, dtype=dtype))

    @classmethod
    def from_dict(
        cls, data: dict[str, dict[str, float]], device=None, dtype=torch.float32
    ) -> WorldState:
        """Load the nested JSON form. Missing channels fall back to baseline."""
        state = cls.baseline(1, device=device, dtype=dtype)
        for group, channels in data.items():
            if group not in GROUP_SLICES:
                continue
            if group == "biochemistry" and isinstance(channels, list):
                channels = {item["name"]: item["value"] for item in channels}
            for name, value in channels.items():
                key = f"{group}.{name}"
                if key in schema.INDEX:
                    state.values[:, schema.INDEX[key]] = float(value)
        return state

    # --- shape and device ---

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def to(self, *args, **kwargs) -> WorldState:
        return WorldState(self.values.to(*args, **kwargs))

    def clone(self) -> WorldState:
        return WorldState(self.values.clone())

    def detach(self) -> WorldState:
        return WorldState(self.values.detach())

    def expand(self, batch_size: int) -> WorldState:
        """Broadcast a single world to a batch, e.g. for parallel rollouts."""
        if self.batch_size == batch_size:
            return self
        if self.batch_size != 1:
            raise ValueError(f"Cannot expand batch {self.batch_size} to {batch_size}.")
        return WorldState(self.values.expand(batch_size, -1).clone())

    def __len__(self) -> int:
        return self.batch_size

    # --- named access ---

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.values[:, schema.index_of(key)]

    def __setitem__(self, key: str, value) -> None:
        self.values[:, schema.index_of(key)] = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.device, dtype=self.dtype)
        )

    def get(self, key: str, index: int = 0) -> float:
        """A single scalar, for logging and assertions."""
        return float(self.values[index, schema.index_of(key)])

    def slice(self, group: str) -> torch.Tensor:
        """The contiguous block for one group, as a view."""
        if group not in GROUP_SLICES:
            raise KeyError(f"Unknown group {group!r}. Groups: {', '.join(GROUP_ORDER)}")
        return self.values[:, GROUP_SLICES[group]]

    def group(self, name: str, index: int = 0) -> dict[str, float]:
        block = self.slice(name)[index]
        names = [c.name for c in CHANNELS if c.group == name]
        return dict(zip(names, block.tolist(), strict=True))

    def add(self, key: str, amount) -> None:
        self.values[:, schema.index_of(key)] += amount

    def clamp_(self, low: float = 0.0, high: float = 1.0) -> WorldState:
        self.values.clamp_(low, high)
        return self

    # --- reporting ---

    def top(
        self, group: str = "emotion", k: int = 5, index: int = 0
    ) -> list[tuple[str, float]]:
        """The ``k`` strongest channels in a group -- what is actually salient."""
        items = sorted(self.group(group, index).items(), key=lambda kv: -kv[1])
        return items[:k]

    def deviations(
        self, k: int = 8, index: int = 0, integrated_only: bool = True
    ) -> list[tuple[str, float]]:
        """Channels furthest from their baseline, signed.

        More informative than raw magnitudes: a serotonin of 0.5 is unremarkable
        because that is where it rests, while a ``surprise`` of 0.5 is enormous.

        Clock and readout channels are excluded by default. ``time_of_day`` has a
        baseline of 0 and spends most of the day far from it, so including them
        means the salience list is permanently topped by the fact that time is
        passing.
        """
        base = torch.tensor(schema.baselines(), device=self.device, dtype=self.dtype)
        delta = (self.values[index] - base).tolist()
        ranked = sorted(
            (
                (i, d)
                for i, d in enumerate(delta)
                if not integrated_only or CHANNELS[i].kind == "integrated"
            ),
            key=lambda pair: -abs(pair[1]),
        )
        return [(CHANNELS[i].key, round(d, 4)) for i, d in ranked[:k]]

    def to_dict(self, index: int = 0, biochemistry_as_list: bool = True) -> dict[str, Any]:
        """The nested JSON form, matching the schema this was specified with."""
        data: dict[str, Any] = {}
        for group in GROUP_ORDER:
            values = self.group(group, index)
            if group == "biochemistry" and biochemistry_as_list:
                data[group] = [
                    {"name": name, "value": round(value, 6)}
                    for name, value in values.items()
                ]
            else:
                data[group] = {name: round(value, 6) for name, value in values.items()}
        return data

    def save(self, path: str | os.PathLike, index: int = 0) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(index), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | os.PathLike, device=None) -> WorldState:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), device=device)

    def __repr__(self) -> str:
        if self.batch_size != 1:
            return f"WorldState(batch={self.batch_size})"
        parts = [f"{key.split('.')[-1]}={value:+.2f}" for key, value in self.deviations(4)]
        return (
            f"WorldState(valence={self.get('output.valence'):.2f} "
            f"arousal={self.get('output.arousal'):.2f} "
            f"dominance={self.get('output.dominance'):.2f} | " + " ".join(parts) + ")"
        )
