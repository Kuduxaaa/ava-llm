"""Automatic appraisal: prompt -> hidden states -> context -> world -> reply."""

import pytest
import torch

from ava import AvaForCausalLM, GenerationConfig
from ava.world import (
    NUM_CONTEXT,
    AvaPerception,
    DirectAppraisal,
    NeuralAppraisal,
    WorldEngine,
    encode_prompt,
    pool_hidden_states,
)

from .helpers import tiny

HIDDEN = 32


@pytest.fixture
def model():
    torch.manual_seed(0)
    return AvaForCausalLM(tiny(world_conditioning=True)).eval()


@pytest.fixture
def perception(model):
    torch.manual_seed(0)
    return AvaPerception(model, WorldEngine())


@pytest.fixture
def prompt():
    torch.manual_seed(0)
    return torch.randint(0, 64, (3, 10))


def loud(appraisal: NeuralAppraisal, std: float = 0.5) -> NeuralAppraisal:
    """An untrained head is deliberately silent; give it something to say."""
    with torch.no_grad():
        appraisal.head.weight.normal_(std=std)
        appraisal.head.bias.normal_(std=std)
    return appraisal


# --- shape and bounds --------------------------------------------------------


def test_output_has_one_channel_per_context_dimension(perception, prompt):
    output = perception.appraise(prompt)
    assert output.shape == (3, NUM_CONTEXT)
    assert NUM_CONTEXT == 23


def test_output_columns_are_the_engine_context_channels_in_order():
    """The head's 23 outputs have to land on the right 23 world channels.

    A silent transposition here would be invisible: the world would still move,
    just for the wrong reasons.
    """
    from ava.world import CONTEXT_NAMES, index_of
    from ava.world.appraisal import context_impulse

    for column, name in enumerate(CONTEXT_NAMES):
        intensities = torch.zeros(1, NUM_CONTEXT)
        intensities[0, column] = 1.0
        impulse = context_impulse(intensities, gain=1.0)

        assert impulse[0, index_of(f"context.{name}")] == 1.0
        assert impulse.sum() == 1.0


@pytest.mark.parametrize("std", [0.5, 3.0, 20.0])
def test_outputs_stay_inside_the_unit_interval(perception, prompt, std):
    loud(perception.appraisal, std)
    output = perception.appraise(prompt)
    assert (output >= 0).all() and (output <= 1).all()
    assert torch.isfinite(output).all()


def test_an_untrained_head_invents_nothing(perception, prompt):
    """It must be safe to attach before it is trained."""
    assert perception.appraise(prompt).max() < 0.05


# --- batching ----------------------------------------------------------------


def test_batch_rows_are_independent(perception):
    loud(perception.appraisal)
    torch.manual_seed(1)
    batch = torch.randint(0, 64, (4, 12))

    together = perception.appraise(batch)
    apart = torch.cat([perception.appraise(batch[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(together, apart, rtol=1e-4, atol=1e-5)


def test_padding_does_not_change_a_prompt(perception):
    """A short prompt batched with a long one must appraise the same either way.

    Mean-pooling without the mask drags it toward the pad embedding, so the
    result would depend on the company it kept.
    """
    loud(perception.appraisal)
    torch.manual_seed(2)
    short = torch.randint(1, 64, (1, 5))

    alone = perception.appraise(short, torch.ones_like(short))
    padded_ids = torch.cat([torch.zeros(1, 6, dtype=torch.long), short], dim=1)
    padded_mask = torch.cat(
        [torch.zeros(1, 6, dtype=torch.long), torch.ones_like(short)], dim=1
    )
    padded = perception.appraise(padded_ids, padded_mask)

    torch.testing.assert_close(alone, padded, rtol=1e-3, atol=1e-3)


def test_pooling_ignores_masked_positions():
    hidden = torch.randn(2, 6, HIDDEN)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[:, 3:] = 0

    torch.testing.assert_close(pool_hidden_states(hidden, mask), hidden[:, :3].mean(dim=1))


def test_runs_on_the_accelerator_if_there_is_one():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    device = torch.device("cuda")
    model = AvaForCausalLM(tiny(world_conditioning=True)).to(device).eval()
    perception = AvaPerception(model, WorldEngine(device=device)).to(device)

    output = perception.appraise(torch.randint(0, 64, (2, 8), device=device))
    assert output.device.type == "cuda"
    assert output.shape == (2, NUM_CONTEXT)


# --- gradients ---------------------------------------------------------------


def test_gradients_reach_the_head_but_not_the_frozen_model(perception, prompt):
    perception.appraise(prompt).sum().backward()

    assert perception.appraisal.head.weight.grad is not None
    assert torch.isfinite(perception.appraisal.head.weight.grad).all()
    assert all(p.grad is None for p in perception.model.parameters())


def test_trainable_parameters_are_only_the_new_ones(perception):
    trainable = list(perception.trainable_parameters())
    model_ids = {id(p) for p in perception.model.model.parameters()}
    assert trainable
    assert not any(id(p) in model_ids for p in trainable)


def test_freezing_ava_leaves_the_world_wiring_trainable(perception):
    """The conditioner is the other half of this wiring, not part of Ava.

    Freezing it with the model would leave the world computed but with no path
    by which it could ever come to matter.
    """
    conditioner = perception.model.world_conditioner
    assert all(p.requires_grad for p in conditioner.parameters())
    assert not any(p.requires_grad for p in perception.model.model.parameters())

    trainable = {id(p) for p in perception.trainable_parameters()}
    assert any(id(p) in trainable for p in conditioner.parameters())
    assert any(id(p) in trainable for p in perception.appraisal.parameters())


def test_the_conditioner_can_be_frozen_too(perception):
    perception.freeze_model(keep_conditioner=False)
    assert not any(p.requires_grad for p in perception.model.parameters())


def test_the_model_can_be_unfrozen_deliberately(perception, prompt):
    perception.unfreeze_model()
    perception.appraise(prompt).sum().backward()
    assert perception.model.get_input_embeddings().weight.grad is not None


def test_gradients_survive_the_world_dynamics(perception):
    """Training the head through the engine has to be possible, not just against
    labels: the impulse, the coupling and the integrator are all differentiable."""
    loud(perception.appraisal)
    torch.manual_seed(3)
    ids = torch.randint(0, 64, (1, 8))

    state = perception.perceive(ids, dt=60.0, detach=False)
    assert state.values.requires_grad

    state["output.valence"].sum().backward()
    assert perception.appraisal.head.weight.grad is not None
    assert perception.appraisal.head.weight.grad.abs().sum() > 0


def test_detached_by_default_so_turns_do_not_accumulate_a_graph(perception):
    ids = torch.randint(0, 64, (1, 8))
    state = perception.perceive(ids, dt=30.0)
    assert not state.values.requires_grad


# --- integration with the existing engine ------------------------------------


def test_perception_drives_the_existing_world(perception):
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 10))

    before = perception.engine.state.values.clone()
    perception.perceive(ids, dt=60.0)
    perception.engine.idle(600)

    assert not torch.allclose(before, perception.engine.state.values, atol=1e-3)


def test_the_expectation_tracker_still_runs(perception):
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 10))

    perception.perceive(ids, dt=30.0)
    assert perception.engine.expectations.expected is not None

    for _ in range(4):
        perception.perceive(ids, dt=30.0)
    assert perception.engine.expectations.expected.max() > 0.05


def test_an_explicit_event_still_overrides_the_learned_guess(perception):
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 10))

    perception.perceive(ids, dt=30.0, event={"social_rejection": 1.0})
    assert perception.engine.state.get("context.social_rejection") > 0.5


def test_relationships_still_track_per_person(perception):
    ids = torch.randint(0, 64, (1, 10))
    perception.perceive(ids, dt=120.0, person="nika")
    assert "nika" in perception.engine.relationships
    assert perception.engine.relationships.current == "nika"


def test_a_direct_appraisal_engine_is_upgraded_not_broken():
    model = AvaForCausalLM(tiny(world_conditioning=True)).eval()
    engine = WorldEngine()
    assert isinstance(engine.appraisal, DirectAppraisal)

    perception = AvaPerception(model, engine)
    assert isinstance(engine.appraisal, NeuralAppraisal)
    assert perception.appraisal is engine.appraisal


# --- ordering: appraisal happens before the reply ----------------------------


def test_generation_is_conditioned_on_the_post_perception_world(perception):
    """The reply must be written under the world the prompt produced.

    Appraising afterwards would leave the state one turn behind and Ava reacting
    to her own words instead of to yours.
    """
    loud(perception.appraisal, std=2.0)
    with torch.no_grad():
        for layer in perception.model.world_conditioner.film.values():
            layer.weight.normal_(std=0.2)

    ids = torch.randint(0, 64, (1, 8))
    stale = perception.engine.state.clone()

    config = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=None)
    tokens, used = perception.respond(ids, generation_config=config)

    assert not torch.allclose(used.values, stale.values, atol=1e-4), (
        "perception did not run before generation"
    )
    with torch.no_grad():
        expected = perception.model.generate(
            ids, generation_config=config, world_state=used
        )
        wrong = perception.model.generate(ids, generation_config=config, world_state=stale)
    assert torch.equal(tokens, expected)
    assert not torch.equal(tokens, wrong)


def test_respond_returns_the_state_the_reply_was_written_under(perception):
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 8))
    _, used = perception.respond(
        ids, generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=None)
    )

    perception.engine.idle(3600)
    assert not torch.allclose(used.values, perception.engine.state.values)


# --- no future-token leakage -------------------------------------------------


def test_hidden_states_never_depend_on_later_tokens(model):
    """Causal masking, checked on the exact path the appraisal reads from."""
    torch.manual_seed(4)
    ids = torch.randint(0, 64, (1, 12))

    with torch.no_grad():
        short = encode_prompt(model, ids[:, :6])
        long = encode_prompt(model, ids)

    torch.testing.assert_close(short, long[:, :6], rtol=1e-4, atol=1e-5)


def test_the_appraisal_cannot_see_the_reply_it_causes(perception):
    """Same prompt, same appraisal -- before and after a reply exists."""
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 8))

    before = perception.appraise(ids).clone()
    reply = perception.model.generate(
        ids, generation_config=GenerationConfig(max_new_tokens=6, eos_token_id=None)
    )
    after = perception.appraise(ids)

    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    # The assertion above is only meaningful if trailing tokens *would* have
    # changed the answer. They do -- which is exactly why they must be excluded.
    with_reply = perception.appraise(reply)
    assert not torch.allclose(before, with_reply, atol=1e-4)


def test_respond_appraises_the_prompt_and_not_the_generated_tokens(perception):
    loud(perception.appraisal, std=2.0)
    ids = torch.randint(0, 64, (1, 8))

    expected = perception.appraise(ids).clone()

    engine = perception.engine
    seen: list[torch.Tensor] = []
    original = engine.appraisal

    class Recording(NeuralAppraisal):
        pass

    def spy(state, hidden_states=None, event=None, relationship=None, attention_mask=None):
        seen.append(hidden_states)
        return original(
            state,
            hidden_states=hidden_states,
            event=event,
            relationship=relationship,
            attention_mask=attention_mask,
        )

    engine.appraisal = spy
    perception.respond(
        ids, generation_config=GenerationConfig(max_new_tokens=5, eos_token_id=None)
    )
    engine.appraisal = original

    assert len(seen) == 1, "appraisal ran more than once per turn"
    assert seen[0].shape[1] == ids.shape[1], (
        f"appraisal saw {seen[0].shape[1]} positions for an {ids.shape[1]}-token prompt"
    )
    assert expected.shape == (1, NUM_CONTEXT)
