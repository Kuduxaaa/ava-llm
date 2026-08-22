"""The channel registry: what the internal world is made of.

Every quantity Ava carries lives in one flat tensor. This module is the single
place that decides what index each quantity occupies, where it rests when
nothing is happening, and how fast it forgets.

Three properties per channel do most of the work:

``baseline``
    The setpoint homeostasis pulls toward. Not always zero -- resting heart
    rate, serotonin tone and blood oxygen all sit well above their floor.

``tau``
    The time constant, in **seconds**. This is what makes the simulation feel
    alive rather than reactive: epinephrine clears in a couple of minutes,
    cortisol takes an hour, adenosine builds across a whole waking day, and
    attachment moves on the order of a week. An event that spikes several
    channels at once leaves them decaying at completely different rates, so the
    state an hour later is not a scaled copy of the state at the time -- it has
    a different shape.

``rise``
    Time constants are **asymmetric**. ``tau`` governs the fall; the rise uses
    ``tau * rise``, and ``rise`` is well below 1 for nearly everything. This is
    not a tuning convenience -- a single time constant cannot describe an
    emotion. Sadness arrives in seconds and leaves in hours; adrenaline is
    instant and clears in minutes; cortisol takes twenty minutes to climb and an
    hour to fall. With one constant per channel, a twelve-second conversational
    turn against a ninety-minute decay moves sadness by 0.2% and the world is
    inert during exactly the moments that matter.

``kind``
    ``integrated`` channels obey the dynamics. ``clock`` channels are written
    from wall-clock time each step. ``readout`` channels are computed from the
    rest and never integrated.

All channels are normalised to ``[0, 1]``. That is a deliberate simplification:
it keeps the state a single homogeneous tensor, at the cost of every value being
a fraction of a plausible range rather than a physical unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChannelKind = Literal["integrated", "clock", "readout"]

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
WEEK = 7 * DAY


@dataclass(frozen=True)
class Channel:
    """One scalar quantity in the world state."""

    group: str
    name: str
    baseline: float
    tau: float
    kind: ChannelKind = "integrated"
    rise: float = 1.0
    """Rise time as a fraction of ``tau``. Below 1 means faster to arrive than
    to leave."""

    @property
    def tau_rise(self) -> float:
        return self.tau * self.rise

    @property
    def key(self) -> str:
        """Qualified name. Required: ``fatigue`` and ``uncertainty`` each appear
        in two groups, and ``arousal`` in three."""
        return f"{self.group}.{self.name}"


# (name, baseline, tau_seconds)
#
# Half-lives are the honest part of this table; the couplings elsewhere are a
# caricature, but the timescales are chosen to match what the real substance or
# state actually does.
_BIOCHEMISTRY = [
    # --- monoamines: minutes ---
    ("dopamine", 0.35, 3 * MINUTE),
    ("serotonin", 0.50, 8 * MINUTE),
    ("norepinephrine", 0.25, 4 * MINUTE),
    ("epinephrine", 0.10, 2 * MINUTE),
    ("histamine", 0.30, 5 * MINUTE),
    ("acetylcholine", 0.40, 2 * MINUTE),
    # --- fast transmitters: seconds ---
    ("GABA", 0.50, 45.0),
    ("glutamate", 0.45, 45.0),
    # --- peptides and hormones: tens of minutes to hours ---
    ("cortisol", 0.25, 1.2 * HOUR),
    ("oxytocin", 0.20, 6 * MINUTE),
    ("vasopressin", 0.20, 15 * MINUTE),
    ("endorphins", 0.15, 12 * MINUTE),
    ("enkephalins", 0.15, 6 * MINUTE),
    ("dynorphins", 0.10, 20 * MINUTE),
    ("substance_P", 0.10, 10 * MINUTE),
    ("neuropeptide_Y", 0.30, 30 * MINUTE),
    ("orexin", 0.45, 25 * MINUTE),
    ("melatonin", 0.10, 1.5 * HOUR),
    # --- endocannabinoids ---
    ("anandamide", 0.25, 10 * MINUTE),
    ("2_AG", 0.25, 8 * MINUTE),
    # --- sleep pressure: accumulates across a waking day ---
    ("adenosine", 0.20, 5 * HOUR),
    # --- neurotrophins: days ---
    ("BDNF", 0.45, 2 * DAY),
    ("NGF", 0.45, 3 * DAY),
    ("GDNF", 0.45, 3 * DAY),
    # --- steroids: a day or more ---
    ("testosterone", 0.45, 1 * DAY),
    ("estradiol", 0.45, 1 * DAY),
    ("progesterone", 0.40, 1 * DAY),
    ("DHEA", 0.45, 1 * DAY),
    ("allopregnanolone", 0.35, 8 * HOUR),
    # --- metabolic ---
    ("insulin", 0.30, 25 * MINUTE),
    ("glucagon", 0.30, 25 * MINUTE),
    ("GLP_1", 0.25, 20 * MINUTE),
    ("ghrelin", 0.35, 40 * MINUTE),
    ("leptin", 0.45, 3 * HOUR),
    # --- inflammatory: slow to rise, slow to clear ---
    ("interleukin_6", 0.10, 3 * HOUR),
    ("TNF_alpha", 0.10, 3 * HOUR),
    ("prostaglandin_E2", 0.10, 1 * HOUR),
    ("nitric_oxide", 0.30, 2 * MINUTE),
]

_PHYSIOLOGY = [
    ("heart_rate", 0.40, 90.0),
    ("heart_rate_variability", 0.55, 4 * MINUTE),
    ("respiration_rate", 0.35, 90.0),
    ("skin_conductance", 0.20, 45.0),
    ("skin_temperature", 0.50, 8 * MINUTE),
    ("body_temperature", 0.50, 25 * MINUTE),
    ("blood_pressure_systolic", 0.45, 3 * MINUTE),
    ("blood_pressure_diastolic", 0.45, 3 * MINUTE),
    ("blood_oxygen", 0.95, 90.0),
    ("pupil_dilation", 0.35, 30.0),
    ("muscle_tension", 0.25, 5 * MINUTE),
    ("movement_activity", 0.20, 3 * MINUTE),
    ("sleep_pressure", 0.25, 4 * HOUR),
    ("energy_level", 0.65, 2 * HOUR),
    ("fatigue", 0.20, 2.5 * HOUR),
]

# Context channels are event *traces*, not switches. An appraisal writes an
# impulse and the trace fades, so "an argument ten minutes ago" is still partly
# present while "an argument yesterday" is not -- without anything having to
# remember to clear a flag.
_CONTEXT = [
    ("social_interaction", 0.0, 6 * MINUTE),
    ("social_acceptance", 0.0, 12 * MINUTE),
    ("social_rejection", 0.0, 20 * MINUTE),
    ("social_isolation", 0.0, 2 * HOUR),
    ("physical_threat", 0.0, 4 * MINUTE),
    ("psychological_threat", 0.0, 15 * MINUTE),
    ("uncertainty", 0.0, 20 * MINUTE),
    ("novelty", 0.0, 5 * MINUTE),
    ("reward_expectation", 0.0, 10 * MINUTE),
    ("reward_received", 0.0, 5 * MINUTE),
    ("loss", 0.0, 1 * HOUR),
    ("competition", 0.0, 15 * MINUTE),
    ("achievement", 0.0, 40 * MINUTE),
    ("failure", 0.0, 40 * MINUTE),
    ("physical_pain", 0.0, 10 * MINUTE),
    ("mental_effort", 0.0, 8 * MINUTE),
    ("workload", 0.0, 30 * MINUTE),
    ("time_pressure", 0.0, 20 * MINUTE),
    ("environmental_stress", 0.0, 20 * MINUTE),
    ("physical_activity", 0.0, 15 * MINUTE),
    ("food_intake", 0.0, 10 * MINUTE),
    ("caffeine", 0.0, 4 * HOUR),
    ("alcohol", 0.0, 3 * HOUR),
]

_INTERNAL_STATE = [
    ("hunger", 0.25, 1.5 * HOUR),
    ("thirst", 0.20, 1 * HOUR),
    ("sleepiness", 0.20, 2 * HOUR),
    ("pain", 0.05, 20 * MINUTE),
    ("fatigue", 0.20, 2 * HOUR),
    ("arousal", 0.35, 6 * MINUTE),
    ("stress", 0.15, 35 * MINUTE),
    ("relaxation", 0.50, 20 * MINUTE),
    ("physical_energy", 0.65, 2 * HOUR),
    ("mental_energy", 0.65, 1.5 * HOUR),
]

# Emotions span three orders of magnitude in persistence, which is most of what
# separates "a feeling" from "a mood" from "how she feels about you".
_EMOTION = [
    ("happiness", 0.45, 25 * MINUTE),
    ("sadness", 0.10, 1.5 * HOUR),
    ("anger", 0.05, 12 * MINUTE),
    ("fear", 0.05, 8 * MINUTE),
    ("anxiety", 0.10, 45 * MINUTE),
    ("disgust", 0.05, 10 * MINUTE),
    ("surprise", 0.02, 40.0),
    ("excitement", 0.15, 10 * MINUTE),
    ("calmness", 0.50, 25 * MINUTE),
    ("contentment", 0.45, 1 * HOUR),
    ("frustration", 0.08, 20 * MINUTE),
    ("loneliness", 0.15, 6 * HOUR),
    ("love", 0.20, 30 * DAY),
    ("affection", 0.25, 1 * DAY),
    ("attachment", 0.20, 1 * WEEK),
    ("trust", 0.30, 3 * DAY),
    ("jealousy", 0.03, 2 * HOUR),
    ("guilt", 0.05, 3 * HOUR),
    ("shame", 0.05, 6 * HOUR),
    ("pride", 0.20, 2 * HOUR),
    ("envy", 0.05, 2 * HOUR),
    ("hope", 0.40, 4 * HOUR),
    ("despair", 0.05, 6 * HOUR),
    ("gratitude", 0.25, 3 * HOUR),
    ("empathy", 0.45, 30 * MINUTE),
    ("compassion", 0.45, 1 * HOUR),
    ("motivation", 0.45, 1 * HOUR),
    ("curiosity", 0.50, 30 * MINUTE),
    ("boredom", 0.15, 25 * MINUTE),
]

_COGNITIVE_STATE = [
    ("attention", 0.55, 2 * MINUTE),
    ("focus", 0.50, 5 * MINUTE),
    ("alertness", 0.60, 10 * MINUTE),
    ("mental_clarity", 0.60, 20 * MINUTE),
    ("working_memory", 0.60, 4 * MINUTE),
    ("decision_confidence", 0.50, 15 * MINUTE),
    ("impulsivity", 0.25, 20 * MINUTE),
    ("rumination", 0.10, 45 * MINUTE),
    ("intrusive_thoughts", 0.05, 40 * MINUTE),
    ("perceived_control", 0.55, 1 * HOUR),
    ("uncertainty", 0.25, 30 * MINUTE),
]

# Written from the clock, never integrated.
_TEMPORAL = [
    ("time_of_day", 0.0, 1.0),
    ("sleep_duration", 0.3, 1.0),
    ("time_since_waking", 0.0, 1.0),
    ("time_since_last_meal", 0.0, 1.0),
    ("time_since_exercise", 0.0, 1.0),
    ("time_since_social_interaction", 0.0, 1.0),
    ("recent_stress_duration", 0.0, 1.0),
]

# Derived each step from everything above.
_OUTPUT = [
    ("valence", 0.5, 1.0),
    ("arousal", 0.35, 1.0),
    ("dominance", 0.5, 1.0),
]

#: Default rise fraction per group.
_GROUP_RISE: dict[str, float] = {
    "biochemistry": 0.25,  # release outpaces clearance
    "physiology": 0.40,  # the body follows, with lag
    "context": 0.03,  # an event registers immediately
    "internal_state": 0.12,
    "emotion": 0.06,  # fast onset, slow offset
    "cognitive_state": 0.30,
    "temporal": 1.0,
    "output": 1.0,
}

#: Channels whose asymmetry is not the group default.
_RISE_OVERRIDES: dict[str, float] = {
    # Adrenaline is essentially instantaneous; cortisol is famously not.
    "biochemistry.epinephrine": 0.08,
    "biochemistry.norepinephrine": 0.10,
    "biochemistry.cortisol": 0.30,
    "biochemistry.oxytocin": 0.15,
    "biochemistry.adenosine": 1.00,  # an accumulator, symmetric by nature
    "biochemistry.melatonin": 0.60,
    "biochemistry.testosterone": 0.50,
    "biochemistry.estradiol": 0.50,
    "biochemistry.progesterone": 0.50,
    "biochemistry.BDNF": 0.80,
    "biochemistry.NGF": 0.80,
    "biochemistry.GDNF": 0.80,
    # A startle is over almost as fast as it began.
    "emotion.surprise": 0.50,
    # The long bonds still take hours to move, not a week.
    "emotion.love": 0.015,
    "emotion.attachment": 0.02,
    "emotion.trust": 0.03,
    "emotion.affection": 0.05,
    "emotion.loneliness": 0.08,
    "physiology.sleep_pressure": 0.80,
    "physiology.energy_level": 0.60,
    "internal_state.hunger": 0.50,
    "internal_state.fatigue": 0.50,
}

_GROUP_KINDS: dict[str, ChannelKind] = {
    "biochemistry": "integrated",
    "physiology": "integrated",
    "context": "integrated",
    "internal_state": "integrated",
    "emotion": "integrated",
    "cognitive_state": "integrated",
    "temporal": "clock",
    "output": "readout",
}

_GROUP_TABLES = {
    "biochemistry": _BIOCHEMISTRY,
    "physiology": _PHYSIOLOGY,
    "context": _CONTEXT,
    "internal_state": _INTERNAL_STATE,
    "emotion": _EMOTION,
    "cognitive_state": _COGNITIVE_STATE,
    "temporal": _TEMPORAL,
    "output": _OUTPUT,
}

#: Group order fixes the tensor layout; do not reorder without a migration.
GROUP_ORDER = (
    "biochemistry",
    "physiology",
    "context",
    "internal_state",
    "emotion",
    "cognitive_state",
    "temporal",
    "output",
)

CHANNELS: tuple[Channel, ...] = tuple(
    Channel(
        group,
        name,
        baseline,
        tau,
        _GROUP_KINDS[group],
        _RISE_OVERRIDES.get(f"{group}.{name}", _GROUP_RISE[group]),
    )
    for group in GROUP_ORDER
    for name, baseline, tau in _GROUP_TABLES[group]
)

NUM_CHANNELS = len(CHANNELS)
INDEX: dict[str, int] = {channel.key: i for i, channel in enumerate(CHANNELS)}
GROUP_SLICES: dict[str, slice] = {}

_offset = 0
for _group in GROUP_ORDER:
    _size = len(_GROUP_TABLES[_group])
    GROUP_SLICES[_group] = slice(_offset, _offset + _size)
    _offset += _size
del _offset, _group, _size


def index_of(key: str) -> int:
    """Tensor index for a qualified channel name such as ``emotion.trust``."""
    try:
        return INDEX[key]
    except KeyError:
        group = key.split(".", 1)[0]
        if group in GROUP_SLICES:
            available = [c.name for c in CHANNELS if c.group == group]
            raise KeyError(
                f"{key!r} is not a channel. {group!r} has: {', '.join(available)}"
            ) from None
        raise KeyError(
            f"{key!r} is not a channel. Groups: {', '.join(GROUP_ORDER)}"
        ) from None


def indices_of(keys: list[str]) -> list[int]:
    return [index_of(key) for key in keys]


def group_of(index: int) -> str:
    return CHANNELS[index].group


def keys() -> list[str]:
    return [channel.key for channel in CHANNELS]


def kind_mask(kind: ChannelKind) -> list[bool]:
    return [channel.kind == kind for channel in CHANNELS]


def baselines() -> list[float]:
    return [channel.baseline for channel in CHANNELS]


def taus() -> list[float]:
    return [channel.tau for channel in CHANNELS]


def rise_taus() -> list[float]:
    """Time constants for movement *toward* a higher value."""
    return [channel.tau_rise for channel in CHANNELS]
