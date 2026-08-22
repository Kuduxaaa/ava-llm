"""Stable traits, expressed as shifts in where the world rests and how fast it moves.

Personality is not a separate module that occasionally intervenes. It is a
standing bias on the same dynamics everyone else runs: a neurotic Ava has a
higher cortisol setpoint *and* a longer stress time constant, so the same event
puts her in a worse place and keeps her there longer. Nothing in the step
function knows about traits; it only sees a shifted baseline and a scaled tau.

Traits are on ``[0, 1]`` with ``0.5`` meaning average, so a default personality
is exactly the unmodified system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .schema import INDEX, NUM_CHANNELS

#: trait -> {channel: baseline shift at trait = 1.0}
TRAIT_BASELINES: dict[str, dict[str, float]] = {
    "neuroticism": {
        "biochemistry.cortisol": 0.15,
        "biochemistry.serotonin": -0.10,
        "biochemistry.GABA": -0.08,
        "internal_state.stress": 0.10,
        "emotion.anxiety": 0.12,
        "emotion.sadness": 0.06,
        "cognitive_state.rumination": 0.12,
        "cognitive_state.perceived_control": -0.10,
    },
    "extraversion": {
        "biochemistry.dopamine": 0.12,
        "biochemistry.oxytocin": 0.08,
        "emotion.excitement": 0.10,
        "emotion.happiness": 0.06,
        "internal_state.arousal": 0.06,
        "emotion.loneliness": 0.08,  # more affected by being alone, not less
    },
    "openness": {
        "emotion.curiosity": 0.18,
        "biochemistry.acetylcholine": 0.06,
        "emotion.boredom": -0.06,
        "cognitive_state.uncertainty": -0.05,
    },
    "agreeableness": {
        "biochemistry.oxytocin": 0.12,
        "emotion.empathy": 0.14,
        "emotion.compassion": 0.14,
        "emotion.trust": 0.12,
        "emotion.anger": -0.06,
    },
    "conscientiousness": {
        "cognitive_state.perceived_control": 0.12,
        "cognitive_state.impulsivity": -0.14,
        "cognitive_state.focus": 0.10,
        "emotion.motivation": 0.10,
    },
}

#: trait -> {channel: multiplier on tau at trait = 1.0}
#:
#: A value above 1 means the channel is *slower to let go*. This is where
#: rumination and grudges come from -- not from a bigger response, but from a
#: longer one.
TRAIT_TAU_SCALES: dict[str, dict[str, float]] = {
    "neuroticism": {
        "internal_state.stress": 1.8,
        "emotion.anxiety": 1.7,
        "emotion.sadness": 1.6,
        "cognitive_state.rumination": 1.8,
        "biochemistry.cortisol": 1.4,
    },
    "extraversion": {
        "emotion.excitement": 1.3,
        "emotion.sadness": 0.75,
        "emotion.happiness": 1.2,
    },
    "agreeableness": {
        "emotion.anger": 0.7,
        "emotion.trust": 1.3,
        "emotion.attachment": 1.2,
    },
    "conscientiousness": {
        "cognitive_state.focus": 1.3,
        "emotion.motivation": 1.3,
    },
    "openness": {
        "emotion.curiosity": 1.3,
        "emotion.boredom": 0.8,
    },
}


@dataclass
class Personality:
    """Five traits on ``[0, 1]``, plus how strongly the world reacts at all."""

    neuroticism: float = 0.5
    extraversion: float = 0.5
    openness: float = 0.5
    agreeableness: float = 0.5
    conscientiousness: float = 0.5

    reactivity: float = 1.0
    """Global gain on incoming appraisals. Below 1 is phlegmatic, above 1 is raw."""

    resilience: float = 1.0
    """Global multiplier on how fast negative states return to baseline. Above 1
    recovers quickly; below 1 stays hurt."""

    def __post_init__(self) -> None:
        for name in (
            "neuroticism",
            "extraversion",
            "openness",
            "agreeableness",
            "conscientiousness",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.reactivity <= 0 or self.resilience <= 0:
            raise ValueError("reactivity and resilience must be positive.")

    @property
    def traits(self) -> dict[str, float]:
        return {
            "neuroticism": self.neuroticism,
            "extraversion": self.extraversion,
            "openness": self.openness,
            "agreeableness": self.agreeableness,
            "conscientiousness": self.conscientiousness,
        }

    # --- compiled into tensors the dynamics can use ---

    def baseline_offset(self, device=None, dtype=torch.float32) -> torch.Tensor:
        offsets = torch.zeros(1, NUM_CHANNELS, device=device, dtype=dtype)
        for trait, value in self.traits.items():
            weight = (value - 0.5) * 2.0  # -1 .. +1
            for key, shift in TRAIT_BASELINES[trait].items():
                offsets[0, INDEX[key]] += weight * shift
        return offsets

    def tau_scale(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """Multiplicative, so opposing traits compose instead of overwriting."""
        scale = torch.ones(1, NUM_CHANNELS, device=device, dtype=dtype)
        for trait, value in self.traits.items():
            weight = (value - 0.5) * 2.0
            for key, multiplier in TRAIT_TAU_SCALES.get(trait, {}).items():
                # weight=+1 gives the full multiplier, -1 gives its reciprocal.
                scale[0, INDEX[key]] *= multiplier**weight

        if self.resilience != 1.0:
            for key in NEGATIVE_CHANNELS:
                scale[0, INDEX[key]] /= self.resilience
        return scale.clamp_min(1e-3)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Personality:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


#: What ``resilience`` accelerates the recovery of.
NEGATIVE_CHANNELS: tuple[str, ...] = (
    "internal_state.stress",
    "internal_state.pain",
    "emotion.sadness",
    "emotion.anger",
    "emotion.fear",
    "emotion.anxiety",
    "emotion.frustration",
    "emotion.despair",
    "emotion.shame",
    "emotion.guilt",
    "emotion.loneliness",
    "cognitive_state.rumination",
    "cognitive_state.intrusive_thoughts",
    "biochemistry.cortisol",
)


#: A few ready-made dispositions, mostly for tests and demos.
ARCHETYPES: dict[str, Personality] = {
    "balanced": Personality(),
    "anxious": Personality(
        neuroticism=0.85, extraversion=0.35, agreeableness=0.6, resilience=0.6
    ),
    "warm": Personality(
        agreeableness=0.9, extraversion=0.7, neuroticism=0.35, resilience=1.3
    ),
    "stoic": Personality(
        neuroticism=0.15, conscientiousness=0.8, reactivity=0.6, resilience=1.6
    ),
    "curious": Personality(openness=0.95, extraversion=0.65, conscientiousness=0.6),
}
