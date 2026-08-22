"""Ava's internal world: a persistent simulated organism, in tensors.

The language model is the cognitive part. This is the environment it lives in --
136 coupled channels running on their own time constants, from epinephrine
clearing in two minutes to attachment moving over a week.

    from ava.world import WorldEngine, Personality

    engine = WorldEngine(personality=Personality(neuroticism=0.7))
    engine.observe({"social_rejection": 0.7}, dt=8.0, person="nika")
    engine.idle(1800)                       # half an hour later, still not fine
    print(engine.summary())

Emotion is an output of this, not an input to it. Nothing writes to
``emotion.sadness``: sadness is what happens when serotonin drops, substance P
rises and rumination gets going, and it fades on its own schedule rather than
when the conversation moves on.
"""

from .appraisal import (
    CONTEXT_NAMES,
    NUM_CONTEXT,
    Appraisal,
    DirectAppraisal,
    ExpectationTracker,
    NeuralAppraisal,
)
from .clock import WorldClock, circadian_offset
from .conditioning import WorldConditioner, WorldSummary, apply_film
from .coupling import EDGES, build_coupling_matrix, edges_into, edges_out_of
from .dynamics import WorldDynamics
from .engine import EngineConfig, WorldEngine
from .personality import ARCHETYPES, Personality
from .relationships import RELATIONSHIP_SIZE, Relationship, RelationshipBook
from .schema import CHANNELS, GROUP_ORDER, NUM_CHANNELS, Channel, index_of, keys
from .state import WorldState

__all__ = [
    "ARCHETYPES",
    "CHANNELS",
    "CONTEXT_NAMES",
    "EDGES",
    "GROUP_ORDER",
    "NUM_CHANNELS",
    "NUM_CONTEXT",
    "RELATIONSHIP_SIZE",
    "Appraisal",
    "Channel",
    "DirectAppraisal",
    "EngineConfig",
    "ExpectationTracker",
    "NeuralAppraisal",
    "Personality",
    "Relationship",
    "RelationshipBook",
    "WorldClock",
    "WorldConditioner",
    "WorldDynamics",
    "WorldEngine",
    "WorldState",
    "WorldSummary",
    "apply_film",
    "build_coupling_matrix",
    "circadian_offset",
    "edges_into",
    "edges_out_of",
    "index_of",
    "keys",
]
