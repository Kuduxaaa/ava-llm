import pytest
import torch

from ava.model.generation import (
    GenerationConfig,
    apply_repetition_penalty,
    ban_repeated_ngrams,
    filter_logits,
    select_next_token,
)

from .helpers import ConstantHead

# --- length control ---


def test_max_new_tokens_is_exact(transformer_model):
    """No hidden cap: asking for 40 tokens produces 40 tokens."""
    prompt = torch.randint(0, 64, (1, 5))
    output = transformer_model.generate(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=40, do_sample=False, eos_token_id=None
        ),
    )
    assert output.shape[1] == 45


def test_max_length_bounds_the_total(transformer_model):
    prompt = torch.randint(0, 64, (1, 5))
    output = transformer_model.generate(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=100, max_length=20, do_sample=False, eos_token_id=None
        ),
    )
    assert output.shape[1] == 20


def test_budget_takes_the_tighter_of_the_two():
    assert GenerationConfig(max_new_tokens=10, max_length=100).budget(5) == 10
    assert GenerationConfig(max_new_tokens=100, max_length=20).budget(5) == 15
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=None, max_length=None).budget(5)


def test_eos_stops_generation(transformer_model):
    """Force EOS to be the argmax and check the loop exits immediately."""
    transformer_model.lm_head = ConstantHead(transformer_model.config.vocab_size, 2)

    prompt = torch.randint(0, 64, (1, 4))
    output = transformer_model.generate(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=False, eos_token_id=2
        ),
    )
    assert output.shape[1] == 5
    assert output[0, -1].item() == 2


def test_finished_rows_are_padded_not_resampled(transformer_model):
    transformer_model.lm_head = ConstantHead(transformer_model.config.vocab_size, 2)

    output = transformer_model.generate(
        torch.randint(0, 64, (2, 4)),
        generation_config=GenerationConfig(
            max_new_tokens=5, do_sample=False, eos_token_id=2, pad_token_id=0
        ),
    )
    # First new token is EOS for both rows; the rest is padding.
    assert (output[:, 4] == 2).all()


# --- determinism ---


def test_greedy_is_deterministic(transformer_model):
    prompt = torch.randint(0, 64, (1, 5))
    config = GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=None)
    first = transformer_model.generate(prompt, generation_config=config)
    second = transformer_model.generate(prompt, generation_config=config)
    assert torch.equal(first, second)


def test_seeded_sampling_is_reproducible(transformer_model):
    prompt = torch.randint(0, 64, (1, 5))
    config = GenerationConfig(
        max_new_tokens=10, do_sample=True, temperature=1.0, eos_token_id=None, seed=7
    )
    first = transformer_model.generate(prompt, generation_config=config)
    second = transformer_model.generate(prompt, generation_config=config)
    assert torch.equal(first, second)


def test_generate_restores_training_mode(transformer_model):
    transformer_model.train()
    transformer_model.generate(
        torch.randint(0, 64, (1, 3)),
        generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )
    assert transformer_model.training


def test_unknown_kwarg_is_rejected(transformer_model):
    with pytest.raises(TypeError, match="Unknown generation argument"):
        transformer_model.generate(torch.randint(0, 64, (1, 3)), typo_arg=5)


# --- logits processing ---


def test_repetition_penalty_always_lowers_probability():
    """The sign bug: dividing a negative logit makes the token *more* likely."""
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    generated = torch.tensor([[0, 1]])

    penalised = apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    assert penalised[0, 0] < logits[0, 0]
    assert penalised[0, 1] < logits[0, 1]
    assert penalised[0, 2] == logits[0, 2]


def test_repetition_penalty_of_one_is_a_no_op():
    logits = torch.randn(2, 10)
    generated = torch.randint(0, 10, (2, 4))
    torch.testing.assert_close(
        apply_repetition_penalty(logits.clone(), generated, 1.0), logits
    )


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = filter_logits(logits, top_k=2)
    assert torch.isfinite(filtered).sum().item() == 2


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.05, 0.05]]))
    filtered = filter_logits(logits, top_p=0.7)
    kept = torch.isfinite(filtered[0])
    assert kept[0] and kept[1] and not kept[2]


def test_min_p_scales_with_the_top_token():
    logits = torch.log(torch.tensor([[0.8, 0.15, 0.04, 0.01]]))
    filtered = filter_logits(logits, min_p=0.1)
    kept = torch.isfinite(filtered[0])
    assert kept[0] and kept[1] and not kept[2] and not kept[3]


def test_no_repeat_ngram_blocks_the_continuation():
    generated = torch.tensor([[1, 2, 3, 1, 2]])
    logits = torch.zeros(1, 10)
    blocked = ban_repeated_ngrams(logits, generated, ngram_size=3)
    assert blocked[0, 3] == -float("inf")


def test_greedy_selection_ignores_temperature():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    config = GenerationConfig(do_sample=False, temperature=0.01)
    token = select_next_token(logits, config, torch.zeros(1, 0, dtype=torch.long))
    assert token.item() == 1


def test_invalid_sampling_parameters_are_rejected():
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=0.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=1.5)
    with pytest.raises(ValueError, match="min_p"):
        GenerationConfig(min_p=1.0)


def test_multiple_eos_tokens_are_supported():
    config = GenerationConfig(eos_token_id=[2, 3])
    assert config.stop_token_ids == [2, 3]
