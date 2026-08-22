"""The internal world: schema, dynamics, time, personality, bonds, conditioning."""

import math

import pytest
import torch

from ava import AvaForCausalLM
from ava.world import (
    ARCHETYPES,
    EDGES,
    NUM_CHANNELS,
    DirectAppraisal,
    EngineConfig,
    ExpectationTracker,
    NeuralAppraisal,
    Personality,
    Relationship,
    WorldClock,
    WorldConditioner,
    WorldDynamics,
    WorldEngine,
    WorldState,
    index_of,
    schema,
)
from ava.world.appraisal import NUM_CONTEXT
from ava.world.clock import circadian_offset
from ava.world.coupling import build_coupling_matrix, build_readout_matrix

from .helpers import tiny

# --- schema ------------------------------------------------------------------


def test_schema_matches_the_specified_layout():
    assert NUM_CHANNELS == 136
    sizes = {
        group: len([c for c in schema.CHANNELS if c.group == group])
        for group in schema.GROUP_ORDER
    }
    assert sizes == {
        "biochemistry": 38,
        "physiology": 15,
        "context": 23,
        "internal_state": 10,
        "emotion": 29,
        "cognitive_state": 11,
        "temporal": 7,
        "output": 3,
    }


def test_reused_names_are_disambiguated_by_group():
    """``fatigue``, ``uncertainty`` and ``arousal`` each appear in two groups."""
    assert index_of("physiology.fatigue") != index_of("internal_state.fatigue")
    assert index_of("context.uncertainty") != index_of("cognitive_state.uncertainty")
    assert index_of("internal_state.arousal") != index_of("output.arousal")


def test_unknown_channel_names_suggest_the_group():
    with pytest.raises(KeyError, match="happiness"):
        index_of("emotion.happines")


def test_rise_is_never_slower_than_fall():
    for channel in schema.CHANNELS:
        assert channel.tau_rise <= channel.tau + 1e-9, channel.key


def test_timescales_span_the_intended_range():
    """Two minutes to a month. Collapsing that range is what makes a simulation
    feel like a lookup table."""
    integrated = [c for c in schema.CHANNELS if c.kind == "integrated"]
    assert min(c.tau for c in integrated) < 60
    assert max(c.tau for c in integrated) > 20 * 86400


# --- state -------------------------------------------------------------------


def test_state_is_one_contiguous_tensor():
    state = WorldState.baseline(4)
    assert state.values.shape == (4, NUM_CHANNELS)
    assert state.values.is_contiguous()


def test_named_access_reads_and_writes_the_tensor():
    state = WorldState.baseline()
    state["emotion.trust"] = 0.8
    assert state.get("emotion.trust") == pytest.approx(0.8)
    assert state.values[0, index_of("emotion.trust")] == pytest.approx(0.8)


def test_json_round_trip_preserves_every_channel():
    state = WorldState.baseline()
    state["internal_state.stress"] = 0.63
    state["biochemistry.cortisol"] = 0.71

    restored = WorldState.from_dict(state.to_dict())
    torch.testing.assert_close(restored.values, state.values, rtol=1e-5, atol=1e-6)


def test_salience_ignores_the_clock():
    """time_of_day has a baseline of 0 and sits far from it most of the day; if
    it counts as a deviation it tops every salience list forever."""
    state = WorldState.baseline()
    state["temporal.time_of_day"] = 0.9
    state["emotion.anger"] = 0.4
    assert state.deviations(3)[0][0] == "emotion.anger"


# --- coupling ----------------------------------------------------------------


def test_every_coupling_edge_names_real_channels():
    matrix = build_coupling_matrix()
    assert matrix.shape == (NUM_CHANNELS, NUM_CHANNELS)
    assert int((matrix != 0).sum()) > 150


def test_duplicate_edges_accumulate():
    edges = (("emotion.anger", "emotion.frustration", 0.2),) * 3
    matrix = build_coupling_matrix(edges)
    assert matrix[
        index_of("emotion.frustration"), index_of("emotion.anger")
    ] == pytest.approx(0.6)


def test_unknown_edge_endpoints_are_rejected():
    with pytest.raises(KeyError, match="not a channel"):
        build_coupling_matrix((("emotion.anger", "emotion.nonsense", 0.2),))


def test_readout_rows_have_the_expected_signs():
    matrix = build_readout_matrix()
    assert matrix[0, index_of("emotion.happiness")] > 0
    assert matrix[0, index_of("emotion.sadness")] < 0
    assert matrix[2, index_of("emotion.fear")] < 0


# --- dynamics ----------------------------------------------------------------


def test_a_world_at_rest_stays_at_rest():
    """The fixed point has to be the baseline, or nothing else means anything."""
    dynamics = WorldDynamics()
    state = WorldState.baseline()
    for _ in range(20):
        state = dynamics(state, 300.0)

    base = torch.tensor(schema.baselines())
    integrated = torch.tensor(schema.kind_mask("integrated"))
    drift = ((state.values[0] - base).abs() * integrated).max()
    assert drift < 1e-3, f"a resting world drifted by {float(drift):.4f}"


def test_perturbations_decay_back_to_baseline():
    dynamics = WorldDynamics()
    state = WorldState.baseline()
    state["emotion.anger"] = 0.9

    for _ in range(200):
        state = dynamics(state, 120.0)
    assert state.get("emotion.anger") == pytest.approx(0.05, abs=0.02)


def test_decay_follows_the_stated_half_life():
    """An hour of a one-hour time constant should leave 1/e of the excursion."""
    dynamics = WorldDynamics(coupling_gain=0.0)
    state = WorldState.baseline()
    start = 0.9
    state["emotion.sadness"] = start

    tau = 1.5 * 3600
    state = dynamics(state, tau)
    baseline = 0.10
    expected = baseline + (start - baseline) * math.exp(-1.0)
    assert state.get("emotion.sadness") == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("dt", [1.0, 60.0, 3600.0, 86400.0, 30 * 86400.0])
def test_any_step_size_is_stable(dt):
    """A Euler step with dt=86400 against tau=45 does not decay, it detonates."""
    dynamics = WorldDynamics()
    state = WorldState.baseline()
    state["internal_state.stress"] = 0.95

    for _ in range(5):
        state = dynamics(state, dt)
    assert torch.isfinite(state.values).all()
    assert (state.values >= 0).all() and (state.values <= 1).all()


def test_rise_is_faster_than_fall():
    """Sadness arrives in seconds and leaves in hours; one constant cannot do both."""
    dynamics = WorldDynamics()
    key = "emotion.sadness"
    assert dynamics.half_life(key, rising=True) < dynamics.half_life(key) / 5


def test_negative_dt_is_rejected():
    with pytest.raises(ValueError, match="backwards"):
        WorldDynamics()(WorldState.baseline(), -1.0)


def test_explain_attributes_a_change_to_its_causes():
    dynamics = WorldDynamics()
    state = WorldState.baseline()
    state["biochemistry.cortisol"] = 0.9

    causes = dict(dynamics.explain(state, "internal_state.stress"))
    assert causes["biochemistry.cortisol"] > 0


def test_state_is_batched():
    dynamics = WorldDynamics()
    state = WorldState.baseline(8)
    state.values[3, index_of("emotion.anger")] = 0.9
    state = dynamics(state, 60.0)

    assert state.values.shape == (8, NUM_CHANNELS)
    assert (
        state.values[3, index_of("emotion.anger")]
        > state.values[0, index_of("emotion.anger")]
    )


# --- clock -------------------------------------------------------------------


def test_circadian_rhythm_is_continuous_across_midnight():
    """Nothing may jump at midnight; the rhythm is periodic by construction."""
    woke = 7 * 3600.0
    before = circadian_offset(WorldClock(now=23.99 * 3600, woke_at=woke))
    after = circadian_offset(WorldClock(now=24.01 * 3600, woke_at=woke))
    assert (before - after).abs().max() < 0.02


def test_melatonin_is_higher_at_night():
    night = circadian_offset(WorldClock(now=3 * 3600))[
        0, index_of("biochemistry.melatonin")
    ]
    noon = circadian_offset(WorldClock(now=12 * 3600))[
        0, index_of("biochemistry.melatonin")
    ]
    assert night > noon + 0.3


def test_sleep_pressure_builds_with_time_awake():
    rested = WorldClock(now=8 * 3600, woke_at=7 * 3600)
    tired = WorldClock(now=22 * 3600, woke_at=7 * 3600)
    index = index_of("biochemistry.adenosine")
    assert circadian_offset(tired)[0, index] > circadian_offset(rested)[0, index] + 0.2


def test_temporal_channels_are_normalised():
    clock = WorldClock(now=20 * 3600, woke_at=6 * 3600, last_meal_at=0.0)
    vector = clock.temporal_vector()
    assert vector.shape == (1, 7)
    assert (vector >= 0).all() and (vector <= 1).all()


# --- personality -------------------------------------------------------------


def test_average_personality_changes_nothing():
    personality = Personality()
    assert personality.baseline_offset().abs().max() == pytest.approx(0.0)
    assert personality.tau_scale().sub(1.0).abs().max() == pytest.approx(0.0)


def test_neuroticism_raises_the_setpoint_and_slows_recovery():
    anxious = Personality(neuroticism=0.95)
    assert anxious.baseline_offset()[0, index_of("biochemistry.cortisol")] > 0.1
    assert anxious.tau_scale()[0, index_of("internal_state.stress")] > 1.4


def test_opposing_traits_compose_rather_than_overwrite():
    both = Personality(neuroticism=0.9, extraversion=0.9)
    index = index_of("emotion.sadness")
    assert both.tau_scale()[0, index] != Personality(neuroticism=0.9).tau_scale()[0, index]


def test_traits_are_bounded():
    with pytest.raises(ValueError, match="neuroticism"):
        Personality(neuroticism=1.5)


def test_the_same_event_lands_differently_on_different_people():
    """The point of personality: identical input, divergent trajectory."""
    outcomes = {}
    for name in ("anxious", "stoic"):
        engine = WorldEngine(personality=ARCHETYPES[name])
        engine.observe({"psychological_threat": 0.8, "failure": 0.7}, dt=60)
        engine.idle(1800)
        peak = engine.state.get("internal_state.stress")
        engine.idle(3 * 3600)
        outcomes[name] = (peak, engine.state.get("internal_state.stress"))

    # Both get stressed. Only one is still stressed three hours later, which is
    # what personality is: how long a state is held, not how loud it starts.
    assert outcomes["anxious"][1] > outcomes["stoic"][1] + 0.1
    assert outcomes["stoic"][1] < outcomes["stoic"][0] * 0.6


# --- appraisal and expectation ----------------------------------------------


def test_direct_appraisal_rejects_unknown_channels():
    with pytest.raises(KeyError, match="not a context channel"):
        DirectAppraisal()(WorldState.baseline(), event={"vibes": 0.5})


def test_untrained_neural_appraisal_invents_nothing():
    appraisal = NeuralAppraisal(hidden_size=32)
    intensities = appraisal(WorldState.baseline(), hidden_states=torch.randn(1, 4, 32))
    assert intensities.max() < 0.05


def test_neural_appraisal_reads_the_current_world():
    """Same text, different world, different meaning -- the whole premise."""
    torch.manual_seed(0)
    appraisal = NeuralAppraisal(hidden_size=32)
    with torch.no_grad():
        appraisal.head.weight.normal_(std=0.5)

    hidden = torch.randn(1, 4, 32)
    calm = WorldState.baseline()
    tense = WorldState.baseline()
    tense["internal_state.stress"] = 0.9
    tense["emotion.trust"] = 0.05

    assert not torch.allclose(
        appraisal(calm, hidden_states=hidden), appraisal(tense, hidden_states=hidden)
    )


def test_the_first_event_is_not_a_surprise():
    tracker = ExpectationTracker(tau=600.0)
    first = torch.full((1, NUM_CONTEXT), 0.8)
    assert tracker.update(first, 10.0).abs().max() == pytest.approx(0.0)


def test_expectations_learn_per_event_not_only_with_time():
    """A two-hour average moves 0.8% per message; that cannot habituate."""
    tracker = ExpectationTracker(tau=2 * 3600.0, rate=0.35)
    event = torch.full((1, NUM_CONTEXT), 0.8)
    for _ in range(4):
        tracker.update(event, 60.0)
    assert tracker.expected.max() > 0.5


def test_a_long_gap_makes_things_surprising_again():
    tracker = ExpectationTracker(tau=600.0, rate=0.5)
    event = torch.full((1, NUM_CONTEXT), 0.8)
    for _ in range(6):
        tracker.update(event, 30.0)
    routine = tracker.update(event, 30.0).abs().max()

    after_a_week = tracker.update(event, 7 * 86400.0).abs().max()
    assert after_a_week > routine + 0.3


def test_the_eighth_compliment_lands_more_softly_than_the_first():
    """Routine has to stop registering, or happiness ratchets to its ceiling."""
    engine = WorldEngine()
    deltas = []
    for _ in range(8):
        before = engine.state.get("emotion.happiness")
        engine.observe({"social_acceptance": 0.8}, dt=60, person="nika")
        engine.idle(900)
        deltas.append(engine.state.get("emotion.happiness") - before)

    assert deltas[0] > 0.1
    assert deltas[-1] < deltas[0] * 0.2


def test_habituation_can_be_disabled():
    engine = WorldEngine(config=EngineConfig(habituation=0.0))
    peaks = []
    for _ in range(6):
        engine.observe({"social_acceptance": 0.8}, dt=60, person="nika")
        engine.idle(900)
        peaks.append(engine.state.get("emotion.happiness"))
    assert peaks[-1] >= peaks[2]  # without it, nothing wears off


def test_repetition_stops_being_surprising():
    tracker = ExpectationTracker(tau=60.0)
    event = torch.full((1, NUM_CONTEXT), 0.8)
    tracker.update(event, 10.0)
    first = tracker.update(torch.zeros_like(event), 300.0).abs().max()
    for _ in range(10):
        tracker.update(torch.zeros_like(event), 300.0)
    later = tracker.update(torch.zeros_like(event), 300.0).abs().max()
    assert later < first


# --- relationships -----------------------------------------------------------


def test_a_bond_shifts_where_the_world_rests():
    close = Relationship("nika", bond=0.9, trust=0.9)
    offset = close.baseline_offset()
    assert offset[0, index_of("emotion.attachment")] > 0.3
    assert offset[0, index_of("emotion.trust")] > 0.3


def test_conflict_and_trust_pull_in_opposite_directions():
    index = index_of("emotion.trust")
    trusting = Relationship("a", trust=0.9).baseline_offset()[0, index]
    fraught = Relationship("b", conflict=0.9).baseline_offset()[0, index]
    assert trusting > 0 > fraught


def test_absence_fades_conflict_faster_than_trust():
    relationship = Relationship("nika", trust=0.9, conflict=0.9)
    before = (relationship.trust, relationship.conflict)
    relationship.decay(7 * 86400)
    assert relationship.conflict < before[1] * 0.5
    assert relationship.trust > before[0] * 0.7


def test_different_people_are_tracked_separately():
    engine = WorldEngine()
    engine.observe({"social_acceptance": 0.9}, dt=600, person="friend")
    engine.observe({"social_rejection": 0.9}, dt=600, person="stranger")

    friend = engine.relationships.get("friend")
    stranger = engine.relationships.get("stranger")
    assert friend.valence_history > stranger.valence_history


# --- engine ------------------------------------------------------------------


def test_a_fresh_engine_reports_a_coherent_world():
    engine = WorldEngine()
    assert 0.4 < engine.state.get("output.valence") < 0.7
    assert engine.state.get("temporal.time_of_day") > 0


def test_good_news_raises_valence_and_bad_news_lowers_it():
    engine = WorldEngine()
    start = engine.state.get("output.valence")

    engine.observe({"social_acceptance": 0.8, "achievement": 0.7}, dt=60)
    engine.idle(300)
    good = engine.state.get("output.valence")

    engine = WorldEngine()
    engine.observe({"loss": 0.9, "failure": 0.8}, dt=60)
    engine.idle(300)
    bad = engine.state.get("output.valence")

    assert good > start > bad


def test_emotion_outlasts_the_event_that_caused_it():
    """Inertia is the point: the trace of the event fades before the feeling does."""
    engine = WorldEngine()
    engine.observe({"loss": 0.9, "social_rejection": 0.8}, dt=60)
    peak_trace = engine.state.get("context.loss")

    engine.idle(3 * 3600)
    trace_left = engine.state.get("context.loss") / peak_trace
    sadness_left = (engine.state.get("emotion.sadness") - 0.10) / 0.10

    assert trace_left < 0.1, "the event trace should be nearly gone"
    assert sadness_left > 0.5, "the feeling should not be"


def test_indirect_pathways_actually_run():
    """Cortisol drives stress, but only if stress gets to *see* cortisol rise.

    A single large step evaluates the coupling once, so every second-order
    pathway silently vanishes. This is the test that catches that.
    """
    engine = WorldEngine()
    engine.observe({"psychological_threat": 0.9}, dt=60)
    engine.idle(1800)

    assert engine.state.get("biochemistry.cortisol") > 0.4
    assert engine.state.get("internal_state.stress") > 0.2


def test_idling_is_not_a_no_op():
    engine = WorldEngine()
    before = engine.state.values.clone()
    engine.idle(4 * 3600)
    assert not torch.allclose(before, engine.state.values, atol=1e-3)


def test_sleep_clears_what_the_day_accumulated():
    engine = WorldEngine()
    engine.idle(14 * 3600)
    tired = engine.state.get("physiology.sleep_pressure")
    engine.sleep(8)
    assert engine.state.get("physiology.sleep_pressure") < tired


def test_an_acute_response_resolves_once_the_event_is_over():
    engine = WorldEngine()
    engine.observe({"loss": 0.95, "psychological_threat": 0.9}, dt=60)

    # Keep her in contact, so isolation does not accumulate while it resolves.
    # Eight hours, not twenty-four: past that she is simply sleep-deprived, and
    # cortisol climbs again for an entirely different and correct reason.
    for _ in range(8):
        engine.observe({"social_interaction": 0.4}, dt=60, person="nika")
        engine.idle(3600)

    assert engine.state.get("context.loss") < 0.02
    assert engine.state.get("biochemistry.cortisol") == pytest.approx(0.25, abs=0.08)
    assert engine.state.get("emotion.sadness") == pytest.approx(0.10, abs=0.08)


def test_prolonged_isolation_produces_loneliness_with_no_event_at_all():
    """Nothing happens for three days. That is itself something that happened.

    Loneliness here is emergent, not scripted: the clock raises
    ``context.social_isolation``, which feeds loneliness, which feeds sadness
    and suppresses serotonin. No code anywhere decides she should be lonely.
    """
    engine = WorldEngine()
    engine.observe({"social_interaction": 0.6}, dt=120, person="nika")
    before = engine.state.get("emotion.loneliness")

    engine.idle(3 * 86400)

    assert engine.state.get("emotion.loneliness") > before + 0.2
    assert engine.state.get("emotion.sadness") > 0.18
    assert engine.state.get("context.social_isolation") > 0.3


def test_context_shapes_what_an_event_means():
    """Identical input, different history, different outcome."""
    calm = WorldEngine()
    calm.observe({"social_rejection": 0.5}, dt=60)
    calm.idle(600)

    fraught = WorldEngine()
    fraught.observe({"social_rejection": 0.8, "psychological_threat": 0.7}, dt=60)
    fraught.idle(1200)
    fraught.observe({"social_rejection": 0.5}, dt=60)
    fraught.idle(600)

    assert fraught.state.get("emotion.sadness") > calm.state.get("emotion.sadness")


def test_snapshot_round_trip_preserves_everything():
    engine = WorldEngine(personality=ARCHETYPES["warm"])
    engine.observe({"social_acceptance": 0.8}, dt=120, person="nika")
    engine.idle(600)

    snapshot = engine.snapshot()
    restored = WorldEngine()
    restored.restore(snapshot)

    torch.testing.assert_close(
        restored.state.values, engine.state.values, atol=1e-5, rtol=1e-4
    )
    assert restored.clock.now == engine.clock.now
    assert restored.personality.agreeableness == engine.personality.agreeableness
    assert "nika" in restored.relationships


def test_long_gaps_stay_affordable():
    engine = WorldEngine(config=EngineConfig(history_size=0))
    engine.idle(30 * 86400)
    assert torch.isfinite(engine.state.values).all()


# --- conditioning ------------------------------------------------------------


def test_an_untrained_conditioner_is_exactly_a_no_op():
    """Attaching a world to a model trained without one must change nothing."""
    config = tiny(world_conditioning=True)
    model = AvaForCausalLM(config).eval()
    ids = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        plain = model(input_ids=ids).logits
        conditioned = model(input_ids=ids, world_state=WorldState.baseline()).logits
    torch.testing.assert_close(plain, conditioned)


def test_a_trained_conditioner_lets_the_world_change_the_output():
    torch.manual_seed(0)
    model = AvaForCausalLM(tiny(world_conditioning=True)).eval()
    with torch.no_grad():
        for layer in model.world_conditioner.film.values():
            layer.weight.normal_(std=0.1)

    ids = torch.randint(0, 64, (1, 8))
    calm = WorldState.baseline()
    distressed = WorldState.baseline()
    distressed["internal_state.stress"] = 0.9
    distressed["emotion.sadness"] = 0.8

    with torch.no_grad():
        assert not torch.allclose(
            model(input_ids=ids, world_state=calm).logits,
            model(input_ids=ids, world_state=distressed).logits,
            atol=1e-5,
        )


def test_conditioning_without_a_conditioner_is_an_explicit_error():
    model = AvaForCausalLM(tiny()).eval()
    with pytest.raises(ValueError, match="without world conditioning"):
        model(input_ids=torch.randint(0, 64, (1, 4)), world_state=WorldState.baseline())


def test_conditioning_layers_must_exist():
    with pytest.raises(ValueError, match="outside a"):
        WorldConditioner(hidden_size=32, num_layers=4, conditioning_layers=(9,))


def test_soft_prefix_has_the_right_shape():
    conditioner = WorldConditioner(hidden_size=32, num_layers=4, num_prefix_tokens=3)
    prefix = conditioner.prefix_tokens(WorldState.baseline(2))
    assert prefix.shape == (2, 3, 32)


def test_gradients_reach_the_world_through_the_model():
    """The world is a tensor in the graph, not a side-channel of numbers."""
    model = AvaForCausalLM(tiny(world_conditioning=True))
    with torch.no_grad():
        for layer in model.world_conditioner.film.values():
            layer.weight.normal_(std=0.1)

    state = WorldState.baseline()
    state.values.requires_grad_(True)
    ids = torch.randint(0, 64, (1, 8))
    model(input_ids=ids, labels=ids, world_state=state).loss.backward()

    assert state.values.grad is not None
    assert torch.isfinite(state.values.grad).all()


def test_learnable_dynamics_expose_the_graph_as_parameters():
    dynamics = WorldDynamics(learnable=True)
    names = {name for name, _ in dynamics.named_parameters()}
    assert {"coupling", "log_tau", "log_tau_rise", "baseline"} <= names


def test_the_coupling_graph_is_documented_by_construction():
    """Every edge is inspectable, which is why `explain` can attribute anything."""
    assert len(EDGES) > 200
    assert all(isinstance(weight, float) for _, _, weight in EDGES)
