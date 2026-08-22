"""How channels drive each other.

This is the content of the simulation. Everything else is plumbing.

Each edge is ``(source, target, weight)``: a positive weight means that pushing
the source above its baseline pushes the target up too, negative means it pulls
it down. Compiled into a dense ``(D, D)`` matrix so the whole graph evaluates as
one matrix-vector product.

Two things to be honest about:

* **Coupling acts on deviations, not levels.** The drive is
  ``W @ (s - baseline)``, so a world sitting at rest injects nothing and stays
  at rest. Using raw levels instead would make every channel permanently shout
  at every other one.
* **This is a caricature.** The timescales in :mod:`ava.world.schema` are chosen
  to match what the real substance does; these weights are chosen so the
  *behaviour* is recognisable. Cortisol does not literally multiply stress by
  0.7. What matters is that the loops exist, have the right sign, and settle at
  the right speed -- and the matrix is an ``nn.Parameter``, so if you ever have
  data, this is a starting point rather than a commitment.
"""

from __future__ import annotations

import torch

from .schema import INDEX, NUM_CHANNELS

Edge = tuple[str, str, float]

# --- threat, and the HPA axis it drives -------------------------------------
_THREAT: list[Edge] = [
    ("context.physical_threat", "biochemistry.epinephrine", 0.85),
    ("context.physical_threat", "biochemistry.norepinephrine", 0.70),
    ("context.physical_threat", "biochemistry.cortisol", 0.55),
    ("context.physical_threat", "emotion.fear", 0.75),
    ("context.psychological_threat", "biochemistry.norepinephrine", 0.55),
    ("context.psychological_threat", "biochemistry.cortisol", 0.60),
    ("context.psychological_threat", "biochemistry.epinephrine", 0.30),
    ("context.psychological_threat", "emotion.anxiety", 0.55),
    ("context.psychological_threat", "emotion.calmness", -0.45),
    ("context.psychological_threat", "emotion.contentment", -0.30),
    ("context.uncertainty", "biochemistry.cortisol", 0.35),
    ("context.uncertainty", "cognitive_state.uncertainty", 0.70),
    ("context.time_pressure", "biochemistry.cortisol", 0.30),
    ("context.time_pressure", "biochemistry.epinephrine", 0.25),
    ("context.environmental_stress", "biochemistry.cortisol", 0.25),
    ("context.environmental_stress", "internal_state.stress", 0.25),
    # Cortisol is slow to rise and slow to clear, which is what makes stress
    # outlast the thing that caused it.
    ("biochemistry.cortisol", "internal_state.stress", 0.70),
    ("biochemistry.cortisol", "emotion.anxiety", 0.35),
    ("biochemistry.cortisol", "physiology.energy_level", -0.20),
    ("biochemistry.cortisol", "cognitive_state.mental_clarity", -0.25),
    ("biochemistry.cortisol", "biochemistry.BDNF", -0.20),
    # Negative feedback: the axis shuts itself off, eventually.
    ("biochemistry.cortisol", "biochemistry.cortisol", -0.20),
    ("internal_state.stress", "biochemistry.cortisol", 0.25),
    ("internal_state.stress", "internal_state.relaxation", -0.60),
    ("internal_state.stress", "physiology.muscle_tension", 0.50),
    ("internal_state.stress", "physiology.heart_rate_variability", -0.50),
    ("internal_state.stress", "cognitive_state.rumination", 0.40),
    ("internal_state.stress", "emotion.calmness", -0.50),
    ("internal_state.stress", "biochemistry.interleukin_6", 0.20),
]

# --- sympathetic arousal reaching the body ----------------------------------
_AROUSAL: list[Edge] = [
    ("biochemistry.epinephrine", "physiology.heart_rate", 0.80),
    ("biochemistry.epinephrine", "physiology.skin_conductance", 0.75),
    ("biochemistry.epinephrine", "physiology.blood_pressure_systolic", 0.60),
    ("biochemistry.epinephrine", "physiology.blood_pressure_diastolic", 0.45),
    ("biochemistry.epinephrine", "physiology.respiration_rate", 0.55),
    ("biochemistry.epinephrine", "physiology.pupil_dilation", 0.60),
    ("biochemistry.norepinephrine", "internal_state.arousal", 0.65),
    ("biochemistry.norepinephrine", "cognitive_state.alertness", 0.45),
    ("biochemistry.norepinephrine", "cognitive_state.attention", 0.35),
    ("biochemistry.norepinephrine", "physiology.heart_rate", 0.40),
    ("biochemistry.norepinephrine", "emotion.anxiety", 0.25),
    ("internal_state.arousal", "physiology.heart_rate", 0.30),
    ("internal_state.arousal", "physiology.respiration_rate", 0.25),
    ("biochemistry.nitric_oxide", "physiology.blood_pressure_systolic", -0.25),
]

# --- the brakes --------------------------------------------------------------
_CALMING: list[Edge] = [
    ("biochemistry.GABA", "internal_state.stress", -0.55),
    ("biochemistry.GABA", "emotion.anxiety", -0.55),
    ("biochemistry.GABA", "internal_state.relaxation", 0.60),
    ("biochemistry.GABA", "internal_state.arousal", -0.35),
    ("biochemistry.glutamate", "internal_state.arousal", 0.35),
    ("biochemistry.glutamate", "cognitive_state.working_memory", 0.25),
    ("biochemistry.allopregnanolone", "biochemistry.GABA", 0.45),
    ("biochemistry.allopregnanolone", "emotion.anxiety", -0.30),
    ("biochemistry.progesterone", "biochemistry.allopregnanolone", 0.40),
    ("biochemistry.anandamide", "internal_state.relaxation", 0.35),
    ("biochemistry.anandamide", "emotion.anxiety", -0.25),
    ("biochemistry.anandamide", "emotion.contentment", 0.25),
    ("biochemistry.2_AG", "internal_state.relaxation", 0.25),
    ("biochemistry.2_AG", "emotion.happiness", 0.15),
    ("internal_state.relaxation", "physiology.heart_rate_variability", 0.45),
    ("internal_state.relaxation", "physiology.muscle_tension", -0.45),
]

# --- being with someone ------------------------------------------------------
_SOCIAL: list[Edge] = [
    ("context.social_interaction", "biochemistry.oxytocin", 0.35),
    ("context.social_interaction", "emotion.loneliness", -0.45),
    ("context.social_acceptance", "biochemistry.oxytocin", 0.75),
    ("context.social_acceptance", "biochemistry.serotonin", 0.35),
    ("context.social_acceptance", "biochemistry.dopamine", 0.30),
    ("context.social_acceptance", "emotion.happiness", 0.30),
    # Social rejection recruits the same machinery as physical pain, which is
    # why being ignored genuinely hurts rather than merely disappointing.
    ("context.social_rejection", "biochemistry.cortisol", 0.55),
    ("context.social_rejection", "biochemistry.substance_P", 0.50),
    ("context.social_rejection", "biochemistry.dynorphins", 0.45),
    ("context.social_rejection", "biochemistry.serotonin", -0.35),
    ("context.social_rejection", "emotion.sadness", 0.40),
    ("context.social_rejection", "emotion.shame", 0.30),
    ("context.social_rejection", "emotion.happiness", -0.35),
    ("context.social_rejection", "emotion.trust", -0.30),
    ("context.social_isolation", "emotion.loneliness", 0.60),
    ("context.social_isolation", "biochemistry.oxytocin", -0.30),
    ("biochemistry.oxytocin", "emotion.trust", 0.55),
    ("biochemistry.oxytocin", "emotion.attachment", 0.45),
    ("biochemistry.oxytocin", "emotion.affection", 0.55),
    ("biochemistry.oxytocin", "emotion.empathy", 0.40),
    ("biochemistry.oxytocin", "internal_state.stress", -0.18),
    ("biochemistry.oxytocin", "biochemistry.cortisol", -0.25),
    ("biochemistry.vasopressin", "emotion.attachment", 0.30),
    ("biochemistry.vasopressin", "emotion.jealousy", 0.25),
    ("emotion.attachment", "emotion.love", 0.35),
    ("emotion.trust", "emotion.attachment", 0.30),
    ("emotion.trust", "cognitive_state.uncertainty", -0.25),
    ("emotion.love", "emotion.affection", 0.40),
    ("emotion.loneliness", "emotion.sadness", 0.35),
    ("emotion.loneliness", "biochemistry.serotonin", -0.20),
]

# --- pain, and what turns it off --------------------------------------------
_PAIN: list[Edge] = [
    ("context.physical_pain", "biochemistry.substance_P", 0.70),
    ("context.physical_pain", "internal_state.pain", 0.70),
    ("context.physical_pain", "biochemistry.prostaglandin_E2", 0.40),
    ("biochemistry.substance_P", "internal_state.pain", 0.55),
    ("biochemistry.substance_P", "emotion.sadness", 0.25),
    ("biochemistry.dynorphins", "emotion.despair", 0.35),
    ("biochemistry.dynorphins", "emotion.happiness", -0.25),
    ("biochemistry.endorphins", "internal_state.pain", -0.55),
    ("biochemistry.endorphins", "emotion.happiness", 0.40),
    ("biochemistry.endorphins", "emotion.contentment", 0.35),
    ("biochemistry.enkephalins", "internal_state.pain", -0.35),
    ("biochemistry.enkephalins", "internal_state.relaxation", 0.25),
    ("biochemistry.prostaglandin_E2", "internal_state.pain", 0.35),
    ("biochemistry.prostaglandin_E2", "physiology.body_temperature", 0.30),
    ("internal_state.pain", "internal_state.stress", 0.40),
    ("internal_state.pain", "cognitive_state.attention", -0.30),
    ("internal_state.pain", "emotion.sadness", 0.20),
]

# --- wanting things, getting them, not getting them --------------------------
_REWARD: list[Edge] = [
    ("context.reward_received", "biochemistry.dopamine", 0.70),
    ("context.reward_received", "biochemistry.endorphins", 0.40),
    ("context.reward_expectation", "biochemistry.dopamine", 0.30),
    ("context.achievement", "biochemistry.dopamine", 0.55),
    ("context.achievement", "emotion.pride", 0.60),
    ("context.achievement", "biochemistry.testosterone", 0.25),
    ("context.achievement", "cognitive_state.decision_confidence", 0.30),
    ("context.failure", "biochemistry.dopamine", -0.40),
    ("context.failure", "emotion.frustration", 0.50),
    ("context.failure", "emotion.shame", 0.30),
    ("context.failure", "cognitive_state.decision_confidence", -0.35),
    ("context.failure", "emotion.happiness", -0.35),
    ("context.failure", "emotion.pride", -0.45),
    ("context.failure", "emotion.hope", -0.30),
    ("context.loss", "emotion.sadness", 0.60),
    ("context.loss", "emotion.despair", 0.25),
    ("context.loss", "biochemistry.dopamine", -0.30),
    ("context.loss", "emotion.happiness", -0.50),
    ("context.loss", "emotion.contentment", -0.45),
    ("context.loss", "emotion.hope", -0.35),
    ("context.loss", "biochemistry.serotonin", -0.25),
    ("context.competition", "biochemistry.testosterone", 0.35),
    ("context.competition", "internal_state.arousal", 0.30),
    ("biochemistry.dopamine", "emotion.motivation", 0.60),
    ("biochemistry.dopamine", "emotion.excitement", 0.45),
    ("biochemistry.dopamine", "emotion.happiness", 0.35),
    ("biochemistry.dopamine", "emotion.curiosity", 0.30),
    ("biochemistry.dopamine", "cognitive_state.impulsivity", 0.25),
    ("biochemistry.dopamine", "emotion.boredom", -0.40),
    ("biochemistry.testosterone", "cognitive_state.decision_confidence", 0.25),
    ("biochemistry.testosterone", "emotion.pride", 0.20),
]

# --- resting mood tone -------------------------------------------------------
_MOOD: list[Edge] = [
    ("biochemistry.serotonin", "emotion.contentment", 0.50),
    ("biochemistry.serotonin", "emotion.calmness", 0.40),
    ("biochemistry.serotonin", "emotion.happiness", 0.30),
    ("biochemistry.serotonin", "emotion.sadness", -0.40),
    ("biochemistry.serotonin", "cognitive_state.impulsivity", -0.35),
    ("biochemistry.serotonin", "cognitive_state.rumination", -0.30),
    ("biochemistry.DHEA", "internal_state.stress", -0.20),
    ("biochemistry.estradiol", "emotion.empathy", 0.15),
    ("biochemistry.BDNF", "cognitive_state.mental_clarity", 0.25),
    ("biochemistry.BDNF", "emotion.hope", 0.15),
]

# --- something new -----------------------------------------------------------
_NOVELTY: list[Edge] = [
    ("context.novelty", "biochemistry.dopamine", 0.35),
    ("context.novelty", "biochemistry.acetylcholine", 0.30),
    ("context.novelty", "emotion.curiosity", 0.55),
    ("context.novelty", "emotion.surprise", 0.45),
    ("context.novelty", "emotion.boredom", -0.50),
    ("emotion.curiosity", "emotion.motivation", 0.30),
    ("emotion.curiosity", "cognitive_state.attention", 0.30),
    ("emotion.boredom", "emotion.motivation", -0.35),
    ("emotion.boredom", "cognitive_state.attention", -0.30),
]

# --- thinking, and thinking too much ----------------------------------------
_COGNITION: list[Edge] = [
    ("biochemistry.acetylcholine", "cognitive_state.attention", 0.55),
    ("biochemistry.acetylcholine", "cognitive_state.focus", 0.50),
    ("biochemistry.acetylcholine", "cognitive_state.working_memory", 0.40),
    ("biochemistry.histamine", "cognitive_state.alertness", 0.40),
    ("context.mental_effort", "internal_state.mental_energy", -0.45),
    ("context.mental_effort", "internal_state.fatigue", 0.25),
    ("context.workload", "internal_state.stress", 0.35),
    ("context.workload", "internal_state.mental_energy", -0.30),
    ("internal_state.mental_energy", "cognitive_state.focus", 0.45),
    ("internal_state.mental_energy", "cognitive_state.mental_clarity", 0.40),
    # Rumination is the loop that keeps stress alive with no input at all: it
    # feeds stress, stress feeds it back, and only the time constants stop it.
    ("cognitive_state.rumination", "internal_state.stress", 0.30),
    ("cognitive_state.rumination", "emotion.sadness", 0.25),
    ("cognitive_state.rumination", "cognitive_state.mental_clarity", -0.35),
    ("cognitive_state.rumination", "cognitive_state.intrusive_thoughts", 0.40),
    ("cognitive_state.intrusive_thoughts", "cognitive_state.attention", -0.35),
    ("cognitive_state.uncertainty", "emotion.anxiety", 0.35),
    ("cognitive_state.uncertainty", "cognitive_state.decision_confidence", -0.45),
    ("cognitive_state.perceived_control", "internal_state.stress", -0.40),
    ("cognitive_state.perceived_control", "emotion.hope", 0.35),
    ("cognitive_state.perceived_control", "emotion.despair", -0.35),
]

# --- sleep pressure and the clock -------------------------------------------
_SLEEP: list[Edge] = [
    ("biochemistry.adenosine", "physiology.sleep_pressure", 0.75),
    ("biochemistry.adenosine", "internal_state.sleepiness", 0.65),
    ("biochemistry.adenosine", "cognitive_state.alertness", -0.50),
    ("biochemistry.adenosine", "biochemistry.orexin", -0.30),
    ("biochemistry.melatonin", "internal_state.sleepiness", 0.45),
    ("biochemistry.melatonin", "cognitive_state.alertness", -0.35),
    ("biochemistry.melatonin", "physiology.body_temperature", -0.20),
    ("biochemistry.orexin", "cognitive_state.alertness", 0.50),
    ("biochemistry.orexin", "internal_state.sleepiness", -0.45),
    ("biochemistry.orexin", "internal_state.arousal", 0.25),
    ("context.caffeine", "biochemistry.adenosine", -0.60),
    ("context.caffeine", "cognitive_state.alertness", 0.40),
    ("context.caffeine", "physiology.heart_rate", 0.25),
    ("context.caffeine", "emotion.anxiety", 0.15),
    ("internal_state.sleepiness", "cognitive_state.attention", -0.40),
    ("internal_state.sleepiness", "cognitive_state.mental_clarity", -0.35),
    ("internal_state.fatigue", "internal_state.physical_energy", -0.50),
    ("internal_state.fatigue", "emotion.motivation", -0.30),
    ("internal_state.fatigue", "cognitive_state.focus", -0.30),
    ("physiology.fatigue", "internal_state.fatigue", 0.50),
    ("physiology.energy_level", "internal_state.physical_energy", 0.55),
    ("physiology.energy_level", "emotion.motivation", 0.25),
]

# --- eating ------------------------------------------------------------------
_METABOLIC: list[Edge] = [
    ("biochemistry.ghrelin", "internal_state.hunger", 0.75),
    ("biochemistry.ghrelin", "biochemistry.dopamine", 0.15),
    ("biochemistry.leptin", "internal_state.hunger", -0.50),
    ("biochemistry.GLP_1", "internal_state.hunger", -0.30),
    ("biochemistry.insulin", "physiology.energy_level", 0.25),
    ("biochemistry.glucagon", "physiology.energy_level", 0.15),
    ("context.food_intake", "biochemistry.insulin", 0.65),
    ("context.food_intake", "biochemistry.ghrelin", -0.75),
    ("context.food_intake", "biochemistry.leptin", 0.35),
    ("context.food_intake", "biochemistry.GLP_1", 0.50),
    ("internal_state.hunger", "emotion.frustration", 0.25),
    ("internal_state.hunger", "cognitive_state.impulsivity", 0.20),
    ("internal_state.hunger", "biochemistry.neuropeptide_Y", 0.30),
    ("biochemistry.neuropeptide_Y", "internal_state.stress", -0.30),
    ("biochemistry.neuropeptide_Y", "emotion.anxiety", -0.25),
]

# --- moving, and being ill ---------------------------------------------------
_BODY: list[Edge] = [
    ("context.physical_activity", "physiology.movement_activity", 0.70),
    ("context.physical_activity", "physiology.heart_rate", 0.50),
    ("context.physical_activity", "biochemistry.endorphins", 0.45),
    ("context.physical_activity", "biochemistry.BDNF", 0.35),
    ("context.physical_activity", "physiology.body_temperature", 0.30),
    ("context.physical_activity", "internal_state.physical_energy", -0.30),
    ("biochemistry.interleukin_6", "internal_state.fatigue", 0.45),
    ("biochemistry.interleukin_6", "emotion.sadness", 0.25),
    ("biochemistry.TNF_alpha", "internal_state.fatigue", 0.35),
    ("biochemistry.TNF_alpha", "physiology.body_temperature", 0.25),
    ("context.alcohol", "biochemistry.GABA", 0.50),
    ("context.alcohol", "biochemistry.glutamate", -0.35),
    ("context.alcohol", "biochemistry.dopamine", 0.25),
    ("context.alcohol", "cognitive_state.impulsivity", 0.40),
    ("context.alcohol", "cognitive_state.mental_clarity", -0.45),
]

# --- feelings pushing on other feelings --------------------------------------
_EMOTION_INTERACTION: list[Edge] = [
    ("emotion.happiness", "emotion.sadness", -0.35),
    ("emotion.sadness", "emotion.happiness", -0.35),
    ("emotion.happiness", "emotion.contentment", 0.30),
    ("emotion.anger", "emotion.frustration", 0.40),
    ("emotion.frustration", "emotion.anger", 0.30),
    ("emotion.anger", "emotion.calmness", -0.45),
    ("emotion.fear", "emotion.anxiety", 0.50),
    ("emotion.anxiety", "emotion.calmness", -0.45),
    ("emotion.anxiety", "cognitive_state.rumination", 0.35),
    ("emotion.calmness", "emotion.anxiety", -0.45),
    ("emotion.guilt", "emotion.shame", 0.35),
    ("emotion.guilt", "emotion.sadness", 0.25),
    ("emotion.shame", "cognitive_state.perceived_control", -0.30),
    ("emotion.pride", "cognitive_state.decision_confidence", 0.30),
    ("emotion.pride", "emotion.shame", -0.35),
    ("emotion.envy", "emotion.jealousy", 0.30),
    ("emotion.jealousy", "emotion.anger", 0.25),
    ("emotion.jealousy", "emotion.trust", -0.30),
    ("emotion.hope", "emotion.despair", -0.50),
    ("emotion.despair", "emotion.hope", -0.50),
    ("emotion.despair", "emotion.motivation", -0.45),
    ("emotion.gratitude", "emotion.happiness", 0.25),
    ("emotion.gratitude", "emotion.trust", 0.25),
    ("emotion.empathy", "emotion.compassion", 0.50),
    ("emotion.compassion", "emotion.affection", 0.25),
    ("emotion.contentment", "internal_state.relaxation", 0.30),
    ("emotion.excitement", "internal_state.arousal", 0.40),
    ("emotion.surprise", "internal_state.arousal", 0.35),
    ("emotion.surprise", "cognitive_state.attention", 0.35),
    ("emotion.motivation", "cognitive_state.focus", 0.30),
    ("emotion.disgust", "emotion.happiness", -0.20),
]

EDGES: tuple[Edge, ...] = tuple(
    edge
    for block in (
        _THREAT,
        _AROUSAL,
        _CALMING,
        _SOCIAL,
        _PAIN,
        _REWARD,
        _MOOD,
        _NOVELTY,
        _COGNITION,
        _SLEEP,
        _METABOLIC,
        _BODY,
        _EMOTION_INTERACTION,
    )
    for edge in block
)

# --- the valence/arousal/dominance readout ----------------------------------
#
# Not integrated: recomputed from the rest of the world every step, because it
# is a summary rather than a cause.
VALENCE_WEIGHTS: dict[str, float] = {
    "emotion.happiness": 0.9,
    "emotion.contentment": 0.8,
    "emotion.calmness": 0.5,
    "emotion.love": 0.6,
    "emotion.gratitude": 0.5,
    "emotion.hope": 0.5,
    "emotion.pride": 0.4,
    "emotion.excitement": 0.3,
    "emotion.affection": 0.4,
    "emotion.sadness": -0.9,
    "emotion.fear": -0.7,
    "emotion.anger": -0.7,
    "emotion.anxiety": -0.7,
    "emotion.despair": -0.9,
    "emotion.shame": -0.6,
    "emotion.guilt": -0.5,
    "emotion.loneliness": -0.6,
    "emotion.frustration": -0.5,
    "emotion.disgust": -0.5,
    "internal_state.pain": -0.6,
    "internal_state.stress": -0.5,
}

AROUSAL_WEIGHTS: dict[str, float] = {
    "internal_state.arousal": 1.0,
    "physiology.heart_rate": 0.6,
    "biochemistry.norepinephrine": 0.6,
    "biochemistry.epinephrine": 0.6,
    "emotion.excitement": 0.5,
    "emotion.anger": 0.5,
    "emotion.fear": 0.5,
    "emotion.anxiety": 0.4,
    "emotion.surprise": 0.4,
    "cognitive_state.alertness": 0.3,
    "internal_state.relaxation": -0.6,
    "emotion.calmness": -0.5,
    "internal_state.sleepiness": -0.6,
    "emotion.boredom": -0.3,
}

DOMINANCE_WEIGHTS: dict[str, float] = {
    "cognitive_state.perceived_control": 0.9,
    "cognitive_state.decision_confidence": 0.7,
    "emotion.pride": 0.6,
    "emotion.anger": 0.4,
    "biochemistry.testosterone": 0.3,
    "emotion.motivation": 0.3,
    "emotion.fear": -0.7,
    "emotion.shame": -0.6,
    "emotion.anxiety": -0.5,
    "emotion.despair": -0.6,
    "cognitive_state.uncertainty": -0.5,
    "emotion.guilt": -0.3,
}


def build_coupling_matrix(
    edges: tuple[Edge, ...] = EDGES, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Compile the edge list into ``W`` with ``W[target, source] = weight``.

    Duplicate edges accumulate rather than overwrite, so two blocks can each
    contribute to the same pathway without one silently winning.
    """
    matrix = torch.zeros(NUM_CHANNELS, NUM_CHANNELS, device=device, dtype=dtype)
    for source, target, weight in edges:
        if source not in INDEX:
            raise KeyError(f"Coupling source {source!r} is not a channel.")
        if target not in INDEX:
            raise KeyError(f"Coupling target {target!r} is not a channel.")
        matrix[INDEX[target], INDEX[source]] += weight
    return matrix


def build_readout_matrix(
    device=None, dtype: torch.dtype = torch.float32, active_channels: float = 3.0
) -> torch.Tensor:
    """``(3, D)`` mapping the world onto valence, arousal and dominance.

    Normalising by the total positive mass would be correct only if every
    contributing channel moved at once. In practice three or four do, so a row
    scaled that way barely leaves its baseline no matter what happens.
    ``active_channels`` sets how many simultaneous full-scale movements should
    saturate the readout.
    """
    rows = [VALENCE_WEIGHTS, AROUSAL_WEIGHTS, DOMINANCE_WEIGHTS]
    matrix = torch.zeros(3, NUM_CHANNELS, device=device, dtype=dtype)
    for row, weights in enumerate(rows):
        for key, weight in weights.items():
            if key not in INDEX:
                raise KeyError(f"Readout source {key!r} is not a channel.")
            matrix[row, INDEX[key]] = weight
        positive = matrix[row].clamp_min(0).sum()
        if positive > 0:
            matrix[row] *= active_channels / positive
    return matrix


def edges_into(target: str) -> list[Edge]:
    """Every edge that writes to ``target`` -- for explaining a change."""
    return [edge for edge in EDGES if edge[1] == target]


def edges_out_of(source: str) -> list[Edge]:
    return [edge for edge in EDGES if edge[0] == source]
