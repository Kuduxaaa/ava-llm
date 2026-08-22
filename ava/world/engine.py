"""The world engine: everything above, driven by a clock."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .appraisal import (
    Appraisal,
    DirectAppraisal,
    ExpectationTracker,
    context_impulse,
    prediction_error_impulse,
)
from .clock import WorldClock, circadian_offset
from .conditioning import WorldSummary
from .dynamics import WorldDynamics
from .personality import Personality
from .relationships import RelationshipBook
from .schema import NUM_CHANNELS
from .state import WorldState


@dataclass
class EngineConfig:
    """Knobs that shape how strongly the world reacts, not what it contains."""

    appraisal_gain: float = 0.90
    """How much of a full-intensity appraisal lands as an immediate impulse."""

    prediction_error_gain: float = 0.5
    expectation_tau: float = 2 * 3600.0
    expectation_rate: float = 0.35

    habituation: float = 0.65
    """How much a fully expected event is discounted.

    Without this the eighth identical compliment lands as hard as the first and
    happiness ratchets to its ceiling. Routine should stop registering; that is
    what makes anything else feel like an event."""

    coupling_gain: float = 1.0
    learnable_dynamics: bool = False

    min_step_seconds: float = 15.0
    max_step_seconds: float = 1800.0
    step_growth: float = 1.25
    """Long gaps are integrated in growing slices, from ``min_step_seconds`` up
    to ``max_step_seconds``.

    The exponential update is stable at any ``dt``, but stability is not
    accuracy: it evaluates the coupling *once* per slice. Advance half an hour
    in one jump and cortisol will climb -- driven by the event -- while stress
    never sees it climb, because stress read cortisol once, at the start, when
    it had not moved yet. Every indirect pathway in the graph silently
    disappears.

    Slices start short, where the fast channels are still moving, and grow
    geometrically once the trajectory settles. A six-hour idle costs about sixty
    steps rather than fourteen hundred, and each step is a 136x136 matmul."""

    history_size: int = 512
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class WorldEngine:
    """A persistent simulated organism.

    The loop is::

        engine.observe(event=..., person="nika", dt=12.0)   # something happened
        state = engine.state                                # read it
        engine.idle(3600)                                   # an hour passes

    Nothing about that requires a language model. The engine is a dynamical
    system in its own right, and stays meaningful with the LLM detached -- which
    is the point of separating them. The model is the cognitive part; this is the
    environment it lives in.
    """

    def __init__(
        self,
        personality: Personality | None = None,
        appraisal: Appraisal | None = None,
        config: EngineConfig | None = None,
        clock: WorldClock | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or EngineConfig()
        self.device = torch.device(device or self.config.device or "cpu")
        self.dtype = dtype

        self.personality = personality or Personality()
        self.clock = clock or WorldClock()
        self.appraisal = appraisal or DirectAppraisal()
        self.relationships = RelationshipBook()
        self.expectations = ExpectationTracker(
            tau=self.config.expectation_tau, rate=self.config.expectation_rate
        )

        self.dynamics = WorldDynamics(
            coupling_gain=self.config.coupling_gain,
            learnable=self.config.learnable_dynamics,
            device=self.device,
            dtype=dtype,
        )

        self.state = WorldState.baseline(1, device=self.device, dtype=dtype)
        self.history: list[dict[str, Any]] = []
        self.summarise = WorldSummary()

        # Settle the clock-driven channels and the VAD readout immediately, so a
        # freshly built engine reports a coherent world rather than zeros.
        self.step(0.0)

    # --- the main loop ---

    def observe(
        self,
        event: dict[str, float] | None = None,
        dt: float = 0.0,
        person: str | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        note: str | None = None,
    ) -> WorldState:
        """Something happened. Appraise it, let it land, advance ``dt`` seconds."""
        if person is not None and person != self.relationships.current:
            self.relationships.enter(person, self.clock.now)

        relationship = self.relationships.active
        relationship_vector = (
            None
            if relationship is None
            else relationship.vector(device=self.device, dtype=self.dtype)
        )

        intensities = self.appraisal(
            self.state,
            hidden_states=hidden_states,
            event=event,
            relationship=relationship_vector,
            attention_mask=attention_mask,
        ).to(self.device, self.dtype)

        expected = self.expectations.expected
        error = self.expectations.update(intensities, max(dt, 1.0))

        # Discount the routine. The event is still true; it just stops being
        # news, which is why the surprise term below is computed separately.
        if expected is not None:
            intensities = intensities * (1.0 - self.config.habituation * expected)

        impulse = context_impulse(
            intensities,
            gain=self.config.appraisal_gain * self.personality.reactivity,
            device=self.device,
            dtype=self.dtype,
        )
        impulse = impulse + prediction_error_impulse(
            error,
            dopamine_gain=self.config.prediction_error_gain,
            device=self.device,
            dtype=self.dtype,
        )

        if person is not None:
            self.clock.mark_social()

        state = self.step(dt, impulse=impulse, note=note)

        if relationship is not None:
            relationship.observe(state, max(dt, 1.0))
        return state

    def idle(self, seconds: float, note: str | None = None) -> WorldState:
        """Let time pass with nothing happening.

        Not a no-op. Cortisol clears, adenosine builds, the circadian baseline
        moves, and rumination keeps stress alive on its own. What Ava is like
        when you come back depends on how long you were gone.
        """
        return self.step(seconds, note=note or "idle")

    def sleep(self, hours: float = 8.0) -> WorldState:
        """Sleep clears the homeostatic load that idling only accumulates."""
        self.step(hours * 3600.0, note=f"sleep {hours:g}h")
        self.clock.sleep(0.0)  # times already advanced; just reset the wake clock
        self.clock.slept_hours = hours
        self.state["biochemistry.adenosine"] = torch.full_like(
            self.state["biochemistry.adenosine"], 0.12
        )
        self.state["physiology.sleep_pressure"] = torch.full_like(
            self.state["physiology.sleep_pressure"], 0.15
        )
        self.state["internal_state.fatigue"] = torch.full_like(
            self.state["internal_state.fatigue"], 0.15
        )
        self.state["physiology.energy_level"] = torch.full_like(
            self.state["physiology.energy_level"], 0.8
        )
        # Sleepiness is left slightly elevated rather than cleared: waking is not
        # instantaneous, and the residue decays over the next hour or so.
        self.state["internal_state.sleepiness"] = torch.full_like(
            self.state["internal_state.sleepiness"], 0.30
        )
        self.state["internal_state.mental_energy"] = torch.full_like(
            self.state["internal_state.mental_energy"], 0.75
        )
        return self.step(0.0, note="woke")

    def step(
        self,
        dt: float,
        impulse: torch.Tensor | None = None,
        note: str | None = None,
    ) -> WorldState:
        """Advance the world, slicing long gaps so the feedback loops still run."""
        if dt < 0:
            raise ValueError("dt must be non-negative.")

        remaining = float(dt)
        slice_dt = self.config.min_step_seconds
        first = True
        while True:
            slice_dt = min(remaining, slice_dt)
            self.clock.advance(slice_dt)

            self.state = self.dynamics(
                self.state,
                slice_dt,
                impulse=impulse if first else None,
                personality_offset=self._static_offset(),
                tau_scale=self.personality.tau_scale(self.device, self.dtype),
                circadian_offset=circadian_offset(self.clock, self.device, self.dtype),
                clock=self.clock.temporal_vector(self.device, self.dtype),
            )
            first = False
            remaining -= slice_dt
            if remaining <= 1e-6:
                break
            slice_dt = min(slice_dt * self.config.step_growth, self.config.max_step_seconds)

        self._track_stress()
        self._record(note)
        return self.state

    # --- pieces ---

    def _static_offset(self) -> torch.Tensor:
        """Personality plus the current relationship, as one baseline shift."""
        offset = self.personality.baseline_offset(self.device, self.dtype)
        relationship = self.relationships.active
        if relationship is not None:
            offset = offset + relationship.baseline_offset(self.device, self.dtype)
        return offset

    def _track_stress(self) -> None:
        """Remember when stress started, so ``recent_stress_duration`` is real."""
        stressed = self.state.get("internal_state.stress") > 0.45
        if stressed and self.clock.stress_since is None:
            self.clock.stress_since = self.clock.now
        elif not stressed:
            self.clock.stress_since = None

    def _record(self, note: str | None) -> None:
        if self.config.history_size <= 0:
            return
        self.history.append(
            {
                "t": round(self.clock.now, 3),
                "note": note,
                "valence": round(self.state.get("output.valence"), 4),
                "arousal": round(self.state.get("output.arousal"), 4),
                "dominance": round(self.state.get("output.dominance"), 4),
            }
        )
        if len(self.history) > self.config.history_size:
            del self.history[: len(self.history) - self.config.history_size]

    # --- reading the world ---

    def feeling(self, k: int = 5) -> list[tuple[str, float]]:
        return self.state.top("emotion", k)

    def salient(self, k: int = 8) -> list[tuple[str, float]]:
        """What is furthest from normal right now, signed."""
        return self.state.deviations(k)

    def why(self, channel: str, k: int = 5) -> list[tuple[str, float]]:
        """What is currently driving one channel."""
        return self.dynamics.explain(self.state, channel, k)

    def summary(self) -> str:
        return self.summarise(self.state)

    def trace(self, keys: tuple[str, ...] = ("output.valence",)) -> list[dict[str, float]]:
        """Recent trajectory, for plotting."""
        return self.history

    # --- persistence ---

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "state": self.state.to_dict(),
            "clock": self.clock.to_dict(),
            "personality": self.personality.to_dict(),
            "relationships": self.relationships.to_dict(),
            "expectations": self.expectations.state_dict(),
            "history": self.history[-64:],
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.state = WorldState.from_dict(
            data["state"], device=self.device, dtype=self.dtype
        )
        self.clock = WorldClock.from_dict(data["clock"])
        self.personality = Personality.from_dict(data["personality"])
        self.relationships = RelationshipBook.from_dict(data["relationships"])
        self.expectations.load_state_dict(data.get("expectations", {}), self.device)
        self.history = list(data.get("history", []))

    def save(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | os.PathLike, device=None, **kwargs) -> WorldEngine:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        engine = cls(device=device, **kwargs)
        engine.restore(data)
        return engine

    def __repr__(self) -> str:
        hour = self.clock.time_of_day * 24
        person = self.relationships.current or "-"
        return (
            f"WorldEngine(t={hour:05.2f}h with={person} "
            f"channels={NUM_CHANNELS} | {self.summary()})"
        )
