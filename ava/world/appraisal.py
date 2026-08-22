"""Turning what happened into what it meant.

Appraisal is the stage where the same words become different events. It reads
the incoming message *and* the world already in progress, and emits intensities
over the ``context`` channels -- how much rejection, how much novelty, how much
threat. Those become impulses, and the dynamics take it from there.

Two implementations, because honesty matters more than uniformity here:

:class:`DirectAppraisal`
    You supply the intensities. Use it when an external classifier already
    exists, when scripting a scenario, or in tests.

:class:`NeuralAppraisal`
    A learned head over the language model's hidden state. It needs training
    data to be worth anything -- an untrained head on an untrained model
    produces noise, and no amount of architecture fixes that.

Both are conditioned on the current world, which is the whole point: "fine" from
someone whose ``emotion.trust`` is 0.8 and ``internal_state.stress`` is 0.1 is
not the same event as "fine" after twenty minutes of ``context.social_rejection``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .schema import CHANNELS, GROUP_SLICES, INDEX, NUM_CHANNELS
from .state import WorldState

CONTEXT_NAMES: tuple[str, ...] = tuple(c.name for c in CHANNELS if c.group == "context")
NUM_CONTEXT = len(CONTEXT_NAMES)
CONTEXT_INDEX: dict[str, int] = {name: i for i, name in enumerate(CONTEXT_NAMES)}

#: Context channels whose surprise value should reach dopamine rather than
#: merely startle. Getting less reward than expected is a different signal from
#: an unexpected noise.
REWARD_CONTEXTS: tuple[str, ...] = (
    "reward_received",
    "achievement",
    "social_acceptance",
)


class Appraisal(ABC):
    """Maps an event plus the current world onto context intensities."""

    @abstractmethod
    def __call__(
        self,
        state: WorldState,
        hidden_states: torch.Tensor | None = None,
        event: dict[str, float] | None = None,
        relationship: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``(batch, NUM_CONTEXT)`` intensities in ``[0, 1]``."""


class DirectAppraisal(Appraisal):
    """Pass the intensities straight through.

    Unknown names raise rather than being ignored: a typo in a scenario that
    silently produces a flat world is a bad way to spend an afternoon.
    """

    def __call__(
        self,
        state: WorldState,
        hidden_states: torch.Tensor | None = None,
        event: dict[str, float] | None = None,
        relationship: torch.Tensor | None = None,
    ) -> torch.Tensor:
        intensities = torch.zeros(
            state.batch_size, NUM_CONTEXT, device=state.device, dtype=state.dtype
        )
        if not event:
            return intensities

        for name, value in event.items():
            key = name.split(".")[-1]
            if key not in CONTEXT_INDEX:
                raise KeyError(
                    f"{name!r} is not a context channel. Available: "
                    f"{', '.join(CONTEXT_NAMES)}"
                )
            intensities[:, CONTEXT_INDEX[key]] = float(value)
        return intensities.clamp(0.0, 1.0)


class NeuralAppraisal(nn.Module, Appraisal):
    """A learned appraisal head over hidden states, the world, and a relationship.

    Deliberately small. Its job is not to understand language -- the language
    model does that -- but to decide what the understood thing *means* for this
    particular Ava, right now.
    """

    def __init__(
        self,
        hidden_size: int,
        relationship_size: int = 0,
        width: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.relationship_size = relationship_size

        self.from_text = nn.Linear(hidden_size, width)
        self.from_world = nn.Linear(NUM_CHANNELS, width)
        self.from_relationship = (
            nn.Linear(relationship_size, width) if relationship_size else None
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Linear(width, NUM_CONTEXT)

        # Start quiet: an untrained head should not invent events. The world
        # then behaves exactly as it would with no appraisal at all, which makes
        # "is the dynamics right?" separable from "is the appraisal right?".
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(
        self,
        state: WorldState,
        hidden_states: torch.Tensor | None = None,
        event: dict[str, float] | None = None,
        relationship: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("NeuralAppraisal needs hidden_states.")
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.mean(dim=1)

        features = self.from_text(hidden_states) + self.from_world(state.values)
        if self.from_relationship is not None and relationship is not None:
            features = features + self.from_relationship(relationship)

        intensities = torch.sigmoid(self.head(self.trunk(features)))

        if event:
            # An explicit event overrides the learned guess for those channels.
            override = DirectAppraisal()(state, event=event)
            mask = override > 0
            intensities = torch.where(mask, override, intensities)
        return intensities

    __call__ = nn.Module.__call__


class ExpectationTracker:
    """A decaying memory of what usually happens, and what did not.

    Prediction error is the difference between the appraisal that just arrived
    and this running average. It is what makes the *second* compliment land more
    softly than the first, and an ordinary evening after a terrible week feel
    like relief rather than merely neutral.

    Expectations move on two clocks, and both are needed:

    * they **decay with time** (``tau``), so returning after a week away is
      genuinely surprising again;
    * they **learn per event** (``rate``), so the eighth compliment in ten
      minutes is not.

    A purely time-based average cannot habituate at conversational speed. With a
    two-hour constant and sixty seconds between turns it moves 0.8% per message,
    so the eighth compliment lands exactly as hard as the first.
    """

    def __init__(self, tau: float = 3600.0, rate: float = 0.35) -> None:
        if tau <= 0:
            raise ValueError("Expectation tau must be positive.")
        if not 0.0 < rate <= 1.0:
            raise ValueError("Expectation rate must be in (0, 1].")
        self.tau = tau
        self.rate = rate
        self.values: torch.Tensor | None = None
        self.seen = False

    def reset(self) -> None:
        self.values = None
        self.seen = False

    def update(self, intensities: torch.Tensor, dt: float) -> torch.Tensor:
        """Fold ``intensities`` in and return the prediction error, signed."""
        if self.values is None or self.values.shape != intensities.shape:
            self.values = torch.zeros_like(intensities)

        decay = torch.exp(torch.tensor(-max(dt, 0.0) / self.tau, device=intensities.device))
        self.values = self.values * decay

        error = intensities - self.values
        if not self.seen:
            # Nothing is surprising before there is anything to be surprised
            # against. You do not startle at the world existing.
            error = torch.zeros_like(error)
            self.seen = True

        self.values = (self.values + self.rate * (intensities - self.values)).clamp(0, 1)
        return error

    @property
    def expected(self) -> torch.Tensor | None:
        """How routine each context channel currently is, in ``[0, 1]``."""
        return self.values

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "rate": self.rate,
            "seen": self.seen,
            "values": None if self.values is None else self.values.tolist(),
        }

    def load_state_dict(self, data: dict, device=None) -> None:
        self.tau = data.get("tau", self.tau)
        self.rate = data.get("rate", self.rate)
        self.seen = data.get("seen", False)
        values = data.get("values")
        self.values = None if values is None else torch.tensor(values, device=device)


def context_impulse(
    intensities: torch.Tensor,
    gain: float = 1.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scatter context intensities into a full ``(batch, D)`` impulse vector."""
    batch = intensities.shape[0]
    impulse = torch.zeros(
        batch, NUM_CHANNELS, device=device or intensities.device, dtype=dtype
    )
    impulse[:, GROUP_SLICES["context"]] = intensities * gain
    return impulse


def prediction_error_impulse(
    error: torch.Tensor,
    dopamine_gain: float = 0.5,
    surprise_gain: float = 0.6,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Route prediction error to dopamine and surprise.

    Dopamine follows the *signed* error over reward-like channels -- better than
    expected is a burst, worse than expected is a dip. Surprise follows the
    *magnitude* over everything, because being wrong is startling regardless of
    which direction you were wrong in.
    """
    batch = error.shape[0]
    impulse = torch.zeros(batch, NUM_CHANNELS, device=device or error.device, dtype=dtype)

    reward_columns = [CONTEXT_INDEX[name] for name in REWARD_CONTEXTS]
    signed = error[:, reward_columns].sum(dim=-1)
    impulse[:, INDEX["biochemistry.dopamine"]] = dopamine_gain * signed

    magnitude = error.abs().amax(dim=-1)
    impulse[:, INDEX["emotion.surprise"]] = surprise_gain * magnitude
    impulse[:, INDEX["context.novelty"]] = 0.5 * surprise_gain * magnitude
    return impulse
