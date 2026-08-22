"""The invariant that catches almost every decoding bug.

Running a sequence in one shot and running it token-by-token through the cache
must produce the same logits. RoPE applied at the wrong offset, keys rotated
twice, an SSM that forgets its state, an off-by-one in the causal mask -- all of
them break this equality and nothing else in a test suite notices.
"""

import pytest
import torch

from ava.model.cache import AttentionLayerCache, AvaCache, MambaLayerCache


@torch.no_grad()
def test_incremental_matches_full_forward(model, input_ids):
    reference = model(input_ids=input_ids).logits

    cache = AvaCache.from_config(model.config)
    incremental = []
    for position in range(input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, position : position + 1],
            cache=cache,
            use_cache=True,
        )
        incremental.append(step.logits)
        cache = step.cache

    incremental = torch.cat(incremental, dim=1)
    torch.testing.assert_close(incremental, reference, rtol=2e-4, atol=2e-4)


@torch.no_grad()
def test_prefill_then_decode_matches_full_forward(model, input_ids):
    """The realistic path: a chunked prefill, then single-token steps."""
    reference = model(input_ids=input_ids).logits

    cache = AvaCache.from_config(model.config)
    split = 7
    prefill = model(input_ids=input_ids[:, :split], cache=cache, use_cache=True)
    cache = prefill.cache

    pieces = [prefill.logits]
    for position in range(split, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, position : position + 1],
            cache=cache,
            use_cache=True,
        )
        pieces.append(step.logits)
        cache = step.cache

    torch.testing.assert_close(torch.cat(pieces, dim=1), reference, rtol=2e-4, atol=2e-4)


@torch.no_grad()
def test_cache_tracks_position_offset(model, input_ids):
    cache = AvaCache.from_config(model.config)
    model(input_ids=input_ids, cache=cache, use_cache=True)
    assert cache.seen_tokens == input_ids.shape[1]

    model(input_ids=input_ids[:, :1], cache=cache, use_cache=True)
    assert cache.seen_tokens == input_ids.shape[1] + 1


def test_cache_layout_matches_architecture(config):
    cache = AvaCache.from_config(config)
    assert len(cache) == config.num_hidden_layers
    for kind, layer in zip(config.layer_types(), cache.layers, strict=True):
        expected = AttentionLayerCache if kind == "attention" else MambaLayerCache
        assert isinstance(layer, expected)


@torch.no_grad()
def test_mamba_state_is_fixed_size(input_ids):
    """An SSM's decode state must not grow with context -- that is the point."""
    from ava import AvaForCausalLM

    from .helpers import tiny

    model = AvaForCausalLM(tiny(architecture_type="mamba")).eval()
    cache = AvaCache.from_config(model.config)

    model(input_ids=input_ids, cache=cache, use_cache=True)
    first = cache[0].ssm_state.shape

    for _ in range(5):
        model(input_ids=input_ids[:, :1], cache=cache, use_cache=True)
    assert cache[0].ssm_state.shape == first


@torch.no_grad()
def test_reorder_selects_batch_rows(transformer_model, input_ids):
    cache = AvaCache.from_config(transformer_model.config)
    transformer_model(input_ids=input_ids, cache=cache, use_cache=True)

    before = cache[0].keys.clone()
    cache.reorder(torch.tensor([1, 0]))
    torch.testing.assert_close(cache[0].keys[0], before[1])


@torch.no_grad()
def test_reset_clears_every_layer(model, input_ids):
    cache = AvaCache.from_config(model.config)
    model(input_ids=input_ids, cache=cache, use_cache=True)
    cache.reset()

    assert cache.seen_tokens == 0
    for layer in cache.layers:
        if isinstance(layer, AttentionLayerCache):
            assert layer.keys is None
        else:
            assert layer.ssm_state is None


@pytest.mark.parametrize("use_cache", [True, False])
@torch.no_grad()
def test_generate_is_cache_invariant(transformer_model, use_cache):
    """Greedy decoding must not depend on whether a cache was used."""
    from ava.model.generation import GenerationConfig

    prompt = torch.randint(0, 64, (1, 5))
    config = GenerationConfig(
        max_new_tokens=8, do_sample=False, eos_token_id=None, use_cache=use_cache
    )
    output = transformer_model.generate(prompt, generation_config=config)
    assert output.shape == (1, 13)


@torch.no_grad()
def test_generate_cached_equals_uncached(model):
    from ava.model.generation import GenerationConfig

    prompt = torch.randint(0, 64, (2, 5))
    cached = model.generate(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=8, do_sample=False, eos_token_id=None, use_cache=True
        ),
    )
    uncached = model.generate(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=8, do_sample=False, eos_token_id=None, use_cache=False
        ),
    )
    assert torch.equal(cached, uncached)
