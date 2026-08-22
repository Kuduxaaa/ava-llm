"""Wall-clock bookkeeping and endogenous rhythms.

The world is driven by two clocks. The *circadian* one is a function of the time
of day and does not care what happened: melatonin rises at night whether or not
anyone is talking. The *homeostatic* one accumulates with time awake and only
resets with sleep.

Both act by shifting **baselines** rather than by injecting impulses, which is
the honest description: at 3 a.m. a body is not being pushed toward sleep by an
event, it simply rests somewhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .schema import INDEX, NUM_CHANNELS

DAY_SECONDS = 86400.0


def _bump(phase: torch.Tensor, center: float, width: float) -> torch.Tensor:
    """A smooth periodic bump on ``[0, 1)`` peaking at ``center``.

    Periodic by construction, so nothing discontinuous happens at midnight.
    """
    concentration = 1.0 / max(width, 1e-3) ** 2
    return torch.exp(concentration * (torch.cos(2 * math.pi * (phase - center)) - 1.0))


#: (channel, peak hour, width in days, amplitude)
CIRCADIAN_RHYTHMS: tuple[tuple[str, float, float, float], ...] = (
    ("biochemistry.melatonin", 3.0, 0.16, 0.60),
    ("biochemistry.cortisol", 8.0, 0.22, 0.28),
    ("biochemistry.orexin", 14.0, 0.28, 0.20),
    ("biochemistry.histamine", 14.0, 0.28, 0.12),
    ("physiology.body_temperature", 17.0, 0.30, 0.14),
    ("cognitive_state.alertness", 11.0, 0.32, 0.15),
    ("biochemistry.testosterone", 7.0, 0.24, 0.12),
)

#: Channels that build with time awake rather than with the hour.
HOMEOSTATIC_DRIVES: tuple[tuple[str, str, float], ...] = (
    ("biochemistry.adenosine", "time_since_waking", 0.55),
    ("physiology.fatigue", "time_since_waking", 0.30),
    ("biochemistry.ghrelin", "time_since_last_meal", 0.45),
    ("physiology.energy_level", "time_since_last_meal", -0.20),
    ("context.social_isolation", "time_since_social_interaction", 0.45),
)


@dataclass
class WorldClock:
    """Absolute simulated time, and the event times the world measures against.

    All fields are seconds. ``now`` is an absolute timestamp, so a session can be
    saved and resumed a day later and the world will have aged correctly.
    """

    now: float = 9 * 3600.0
    woke_at: float = 7 * 3600.0
    slept_hours: float = 7.5
    last_meal_at: float = 7.5 * 3600.0
    last_exercise_at: float = -12 * 3600.0
    last_social_at: float = 8 * 3600.0
    stress_since: float | None = None

    _horizons: dict[str, float] = field(
        default_factory=lambda: {
            "sleep_duration": 12 * 3600.0,
            "time_since_waking": 16 * 3600.0,
            "time_since_last_meal": 8 * 3600.0,
            "time_since_exercise": 48 * 3600.0,
            "time_since_social_interaction": 72 * 3600.0,
            "recent_stress_duration": 6 * 3600.0,
        },
        repr=False,
    )

    # --- advancing ---

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Time does not run backwards.")
        self.now += seconds

    def sleep(self, hours: float) -> None:
        """Sleep for ``hours``, then wake. Resets the homeostatic clock."""
        self.advance(hours * 3600.0)
        self.woke_at = self.now
        self.slept_hours = hours
        self.stress_since = None

    def mark_meal(self) -> None:
        self.last_meal_at = self.now

    def mark_exercise(self) -> None:
        self.last_exercise_at = self.now

    def mark_social(self) -> None:
        self.last_social_at = self.now

    # --- readouts ---

    @property
    def time_of_day(self) -> float:
        """Fraction of the day elapsed; 0.0 is midnight."""
        return (self.now % DAY_SECONDS) / DAY_SECONDS

    @property
    def hours_awake(self) -> float:
        return max(0.0, (self.now - self.woke_at) / 3600.0)

    def temporal_vector(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """The seven ``temporal`` channels, normalised to ``[0, 1]``."""
        horizons = self._horizons
        stress_duration = (
            0.0 if self.stress_since is None else max(0.0, self.now - self.stress_since)
        )
        values = [
            self.time_of_day,
            min(1.0, self.slept_hours * 3600.0 / horizons["sleep_duration"]),
            min(1.0, (self.now - self.woke_at) / horizons["time_since_waking"]),
            min(1.0, (self.now - self.last_meal_at) / horizons["time_since_last_meal"]),
            min(1.0, (self.now - self.last_exercise_at) / horizons["time_since_exercise"]),
            min(
                1.0,
                (self.now - self.last_social_at)
                / horizons["time_since_social_interaction"],
            ),
            min(1.0, stress_duration / horizons["recent_stress_duration"]),
        ]
        return torch.tensor(
            [max(0.0, v) for v in values], device=device, dtype=dtype
        ).unsqueeze(0)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "now": self.now,
            "woke_at": self.woke_at,
            "slept_hours": self.slept_hours,
            "last_meal_at": self.last_meal_at,
            "last_exercise_at": self.last_exercise_at,
            "last_social_at": self.last_social_at,
            "stress_since": self.stress_since,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorldClock:
        known = {
            k: v
            for k, v in data.items()
            if k in {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        }
        return cls(**known)


def circadian_offset(
    clock: WorldClock, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Baseline shifts for the current moment, as a ``(1, D)`` tensor.

    Combines the time-of-day rhythms with the drives that build across a waking
    day. The result is added to the resting baseline, so a tired, late-evening
    Ava settles somewhere genuinely different from a rested morning one, with no
    event required to put her there.
    """
    offsets = torch.zeros(1, NUM_CHANNELS, device=device, dtype=dtype)
    phase = torch.tensor([clock.time_of_day], device=device, dtype=dtype)

    for key, peak_hour, width, amplitude in CIRCADIAN_RHYTHMS:
        offsets[0, INDEX[key]] += amplitude * _bump(phase, peak_hour / 24.0, width)[0]

    temporal = clock.temporal_vector(device=device, dtype=dtype)[0]
    temporal_index = {
        "time_since_waking": 2,
        "time_since_last_meal": 3,
        "time_since_social_interaction": 5,
    }
    for key, source, amplitude in HOMEOSTATIC_DRIVES:
        offsets[0, INDEX[key]] += amplitude * temporal[temporal_index[source]]

    return offsets
