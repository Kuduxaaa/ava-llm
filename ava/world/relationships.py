"""Who Ava is talking to, and what that has become over time.

The emotion channels hold how Ava feels *now*. A relationship holds what she has
come to feel *about someone*, which is a different object with a much longer
time constant. Keeping them separate is what lets her be irritable with a friend
without that irritation eroding the friendship, and warm to a stranger without
that warmth being mistaken for closeness.

Coupling runs both ways:

* On entry, a relationship shifts the baselines of the social emotion channels.
  Talking to someone trusted literally *rests* at a higher trust, so the same
  remark from a friend and from a stranger starts from different places.
* On exit, the world writes back. Oxytocin and warmth during a conversation feed
  bond; rejection and anger feed conflict. Slowly, and asymmetrically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .schema import INDEX, NUM_CHANNELS
from .state import WorldState

DAY = 86400.0

#: The per-person state vector, in order.
RELATIONSHIP_FIELDS: tuple[str, ...] = (
    "familiarity",
    "bond",
    "trust",
    "conflict",
    "continuity_expectation",
    "valence_history",
)
RELATIONSHIP_SIZE = len(RELATIONSHIP_FIELDS)

#: How each field decays when the person is absent, in seconds.
_TAU = {
    "familiarity": 60 * DAY,
    "bond": 21 * DAY,
    "trust": 30 * DAY,
    "conflict": 5 * DAY,
    "continuity_expectation": 14 * DAY,
    "valence_history": 7 * DAY,
}

#: What a relationship does to the resting world while that person is present.
_BASELINE_EFFECT: dict[str, dict[str, float]] = {
    "bond": {
        "emotion.attachment": 0.45,
        "emotion.affection": 0.35,
        "emotion.love": 0.20,
        "biochemistry.oxytocin": 0.25,
    },
    "trust": {
        "emotion.trust": 0.50,
        "cognitive_state.uncertainty": -0.15,
        "internal_state.stress": -0.10,
    },
    "conflict": {
        "internal_state.stress": 0.25,
        "emotion.anger": 0.15,
        "cognitive_state.uncertainty": 0.15,
        "emotion.trust": -0.25,
    },
    "familiarity": {
        "context.novelty": -0.20,
        "emotion.calmness": 0.10,
    },
    "continuity_expectation": {
        # Expecting to be remembered is what makes being forgotten hurt.
        "context.uncertainty": -0.10,
    },
}


@dataclass
class Relationship:
    """One durable bond, keyed by person."""

    person_id: str
    familiarity: float = 0.0
    bond: float = 0.0
    trust: float = 0.25
    conflict: float = 0.0
    continuity_expectation: float = 0.2
    valence_history: float = 0.5

    interactions: int = 0
    last_seen_at: float | None = None
    notes: dict = field(default_factory=dict)

    # --- tensor views ---

    def vector(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(
            [getattr(self, name) for name in RELATIONSHIP_FIELDS],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)

    def baseline_offset(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """How this relationship shifts the resting world, as ``(1, D)``."""
        offsets = torch.zeros(1, NUM_CHANNELS, device=device, dtype=dtype)
        for name, effects in _BASELINE_EFFECT.items():
            level = getattr(self, name)
            for key, weight in effects.items():
                offsets[0, INDEX[key]] += level * weight
        return offsets

    # --- time and updates ---

    def decay(self, seconds: float) -> None:
        """Absence fades a relationship, at a different rate for each part.

        Conflict fades fastest and trust slowest, which is roughly how it goes:
        an argument stops mattering long before you stop relying on someone.
        """
        if seconds <= 0:
            return
        for name in RELATIONSHIP_FIELDS:
            rest = 0.25 if name == "trust" else 0.5 if name == "valence_history" else 0.0
            value = getattr(self, name)
            alpha = 1.0 - math.exp(-seconds / _TAU[name])
            setattr(self, name, value + alpha * (rest - value))

    def observe(self, state: WorldState, dt: float, index: int = 0) -> None:
        """Fold the world just experienced back into the bond.

        Rates are deliberately asymmetric. ``bond`` and ``trust`` climb over many
        conversations; ``conflict`` arrives in one. That asymmetry is most of
        what makes a relationship feel earned rather than assigned.
        """
        warmth = 0.5 * state.get("biochemistry.oxytocin", index) + 0.5 * state.get(
            "emotion.affection", index
        )
        friction = 0.5 * state.get("context.social_rejection", index) + 0.5 * state.get(
            "emotion.anger", index
        )
        valence = state.get("output.valence", index)

        minutes = max(dt, 1.0) / 60.0
        self.familiarity = _approach(self.familiarity, 1.0, 0.006 * minutes)
        self.bond = _approach(self.bond, warmth, 0.010 * minutes)
        self.trust = _approach(self.trust, 0.5 + 0.5 * (valence - 0.5) * 2, 0.006 * minutes)
        self.conflict = _approach(self.conflict, friction, 0.05 * minutes)
        self.valence_history = _approach(self.valence_history, valence, 0.02 * minutes)
        self.continuity_expectation = _approach(
            self.continuity_expectation, min(1.0, self.familiarity + 0.2), 0.01 * minutes
        )

        for name in RELATIONSHIP_FIELDS:
            setattr(self, name, min(1.0, max(0.0, getattr(self, name))))
        self.interactions += 1

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            **{name: round(getattr(self, name), 6) for name in RELATIONSHIP_FIELDS},
            "interactions": self.interactions,
            "last_seen_at": self.last_seen_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def __repr__(self) -> str:
        return (
            f"Relationship({self.person_id!r} bond={self.bond:.2f} "
            f"trust={self.trust:.2f} conflict={self.conflict:.2f} "
            f"seen={self.interactions})"
        )


def _approach(current: float, target: float, rate: float) -> float:
    return current + min(rate, 1.0) * (target - current)


class RelationshipBook:
    """Every relationship Ava has, and who she is speaking to now."""

    def __init__(self) -> None:
        self._people: dict[str, Relationship] = {}
        self.current: str | None = None

    def __len__(self) -> int:
        return len(self._people)

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._people

    def __iter__(self):
        return iter(self._people.values())

    def get(self, person_id: str) -> Relationship:
        if person_id not in self._people:
            self._people[person_id] = Relationship(person_id=person_id)
        return self._people[person_id]

    def enter(self, person_id: str, now: float) -> Relationship:
        """Someone arrives. Decay their bond by however long they were away."""
        relationship = self.get(person_id)
        if relationship.last_seen_at is not None:
            relationship.decay(max(0.0, now - relationship.last_seen_at))
        relationship.last_seen_at = now
        self.current = person_id
        return relationship

    def leave(self, now: float) -> None:
        if self.current is not None:
            self.get(self.current).last_seen_at = now
        self.current = None

    @property
    def active(self) -> Relationship | None:
        return None if self.current is None else self._people[self.current]

    def to_dict(self) -> dict:
        return {
            "current": self.current,
            "people": [r.to_dict() for r in self._people.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RelationshipBook:
        book = cls()
        for record in data.get("people", []):
            relationship = Relationship.from_dict(record)
            book._people[relationship.person_id] = relationship
        book.current = data.get("current")
        return book
