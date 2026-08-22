import math

import pytest
import torch

from ava import AvaConfig, AvaForCausalLM
from ava.model.attention import build_causal_mask

from .helpers import tiny


def test_forward_shapes(model, input_ids):
    output = model(input_ids=input_ids)
    assert output.logits.shape == (2, 12, model.config.vocab_size)
    assert output.loss is None


def test_loss_is_finite_and_near_uniform_at_init(model, input_ids):
    output = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(output.loss)
    # An untrained model should sit close to ln(vocab_size).
    assert abs(output.loss.item() - math.log(model.config.vocab_size)) < 1.0


def test_backward_reaches_every_trainable_parameter(model, input_ids):
    model.train()
    model(input_ids=input_ids, labels=input_ids).loss.backward()

    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not missing, f"no gradient reached: {missing}"


def test_labels_are_shifted_not_aligned(transformer_model):
    """Predicting position t from position t means the loss can hit zero."""
    ids = torch.randint(0, 64, (1, 8))
    loss = transformer_model(input_ids=ids, labels=ids).loss
    assert loss.item() > 1.0


def test_ignore_index_masks_loss(transformer_model, input_ids):
    labels = input_ids.clone()
    labels[:, :] = -100
    labels[:, -3:] = input_ids[:, -3:]

    masked = transformer_model(input_ids=input_ids, labels=labels).loss
    full = transformer_model(input_ids=input_ids, labels=input_ids).loss
    assert not torch.isclose(masked, full)
    assert torch.isfinite(masked)


def test_z_loss_is_added_when_enabled():
    torch.manual_seed(0)
    ids = torch.randint(0, 64, (2, 8))

    plain = AvaForCausalLM(tiny(z_loss_coef=0.0)).eval()
    torch.manual_seed(0)
    regularised = AvaForCausalLM(tiny(z_loss_coef=1e-2)).eval()

    assert plain(input_ids=ids, labels=ids).z_loss is None
    output = regularised(input_ids=ids, labels=ids)
    assert output.z_loss is not None and output.z_loss.item() > 0


def test_tied_embeddings_share_storage():
    model = AvaForCausalLM(tiny(tie_word_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()

    # named_parameters() is the walk that exposes duplicates; parameters()
    # de-duplicates unconditionally and would report the same number we do.
    raw = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    assert model.num_parameters() < raw

    # And the saving is exactly one embedding table.
    untied = AvaForCausalLM(tiny(tie_word_embeddings=False))
    saved = untied.num_parameters() - model.num_parameters()
    assert saved == model.config.vocab_size * model.config.hidden_size


def test_num_logits_to_keep_returns_last_position_only(transformer_model, input_ids):
    full = transformer_model(input_ids=input_ids).logits
    last = transformer_model(input_ids=input_ids, num_logits_to_keep=1).logits
    assert last.shape[1] == 1
    torch.testing.assert_close(last[:, -1], full[:, -1])


def test_every_layer_moves_the_residual_stream(config, input_ids):
    """A residual branch that is ~zero at init leaves an identity network.

    It still trains eventually, but an identity stack with tied embeddings
    degenerates into "predict the current token" and starts the run well above
    ln(vocab_size). Checking the loss alone misses it; checking each layer's
    contribution to the stream does not.
    """
    torch.manual_seed(0)
    model = AvaForCausalLM(config).eval()
    with torch.no_grad():
        hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states

    # The last entry is post-final-norm, not a layer output.
    for index in range(len(hidden) - 2):
        before, after = hidden[index], hidden[index + 1]
        contribution = (after - before).norm() / before.norm()
        assert contribution > 0.01, (
            f"layer {index} ({config.layer_types()[index]}) moves the residual "
            f"stream by {contribution:.5f} -- its branch is effectively dead"
        )


def test_mamba_conv_keeps_its_fan_in_init():
    """initializer_range must not be applied to the depthwise convolution.

    Its fan-in is d_conv (4), so PyTorch's default sits near 0.29; overwriting
    it with 0.02 attenuates the whole SSM branch by more than 10x.
    """
    model = AvaForCausalLM(tiny(architecture_type="mamba"))
    conv = model.model.layers[0].ssm.conv1d
    assert conv.weight.std().item() > 5 * model.config.initializer_range


def test_gradient_checkpointing_matches_plain_forward(config, input_ids):
    torch.manual_seed(0)
    model = AvaForCausalLM(config)
    model.train()

    plain = model(input_ids=input_ids, labels=input_ids).loss
    model.gradient_checkpointing_enable()
    checkpointed = model(input_ids=input_ids, labels=input_ids).loss

    torch.testing.assert_close(plain, checkpointed, rtol=1e-4, atol=1e-4)


def test_resize_token_embeddings_preserves_old_rows(transformer_model):
    original = transformer_model.get_input_embeddings().weight[:64].clone()
    transformer_model.resize_token_embeddings(80)

    embeddings = transformer_model.get_input_embeddings()
    assert embeddings.num_embeddings == 80
    torch.testing.assert_close(embeddings.weight[:64], original)
    # New rows must match the configured init, not PyTorch's N(0, 1) default.
    assert embeddings.weight[64:].std().item() < 0.1
    assert transformer_model.config.vocab_size == 80


def test_resize_keeps_weights_tied():
    model = AvaForCausalLM(tiny(tie_word_embeddings=True))
    model.resize_token_embeddings(80)
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def test_scaled_residual_init_shrinks_output_projections():
    scaled = AvaForCausalLM(tiny(scaled_residual_init=True, num_hidden_layers=16))
    plain = AvaForCausalLM(tiny(scaled_residual_init=False, num_hidden_layers=16))

    scaled_std = scaled.model.layers[0].self_attn.o_proj.weight.std().item()
    plain_std = plain.model.layers[0].self_attn.o_proj.weight.std().item()
    assert scaled_std < plain_std * 0.5


def test_padding_mask_changes_output(transformer_model):
    ids = torch.randint(1, 64, (1, 8))
    mask = torch.ones_like(ids)
    mask[:, :3] = 0  # left padding

    unmasked = transformer_model(input_ids=ids).logits
    masked = transformer_model(input_ids=ids, attention_mask=mask).logits
    assert not torch.allclose(unmasked[:, -1], masked[:, -1])


def test_causal_mask_never_produces_an_all_masked_row():
    """A fully masked row makes softmax return NaN and poisons the whole batch."""
    attention_mask = torch.zeros(2, 6, dtype=torch.long)  # everything padded
    mask = build_causal_mask(
        attention_mask,
        batch=2,
        seq_len=6,
        past_length=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask is not None
    # Every query row must keep at least one attendable key.
    attendable = (mask == 0.0).sum(-1)
    assert (attendable >= 1).all()


def test_all_ones_mask_takes_the_fast_path():
    mask = build_causal_mask(
        torch.ones(2, 6, dtype=torch.long),
        batch=2,
        seq_len=6,
        past_length=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask is None, "an all-ones mask should fall through to SDPA is_causal"


def test_both_gqa_paths_produce_the_same_numbers(monkeypatch, input_ids):
    """SDPA is called two different ways depending on the PyTorch build.

    One passes ``enable_gqa=True``; the other repeats the KV heads first. The
    choice is a performance decision, so the two had better agree.
    """
    from ava.model import attention as attention_module

    torch.manual_seed(0)
    model = AvaForCausalLM(tiny(kv_heads=2)).eval()

    outputs = []
    for supported in (False, True):
        monkeypatch.setattr(
            attention_module,
            "sdpa_fused_supports_gqa",
            (lambda answer: lambda _device: answer)(supported),
        )
        with torch.no_grad():
            outputs.append(model(input_ids=input_ids).logits)

    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)


def test_rms_norm_reduces_in_fp32():
    """A bf16 mean-square is where precision quietly leaves a long run."""
    from ava.model.normalization import AvaRMSNorm

    torch.manual_seed(0)
    norm = AvaRMSNorm(512)
    x = (torch.randn(2, 8, 512) * 4).to(torch.bfloat16)

    reference = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-5)
    low_precision = x * torch.rsqrt(
        x.pow(2).mean(-1, keepdim=True).float().to(torch.bfloat16) + 1e-5
    )

    error = (norm(x).double() - reference).abs().max()
    bf16_error = (low_precision.double() - reference).abs().max()
    assert error < bf16_error


def test_rms_norm_keeps_the_fused_path_under_autocast():
    """fp32 gain + bf16 activations makes PyTorch refuse the fused kernel.

    That is the normal autocast configuration, so the fallback would be on for
    every layer of every step -- exactly where the fused path is worth having.
    """
    import warnings

    from ava.model.normalization import AvaRMSNorm

    norm = AvaRMSNorm(128)
    assert norm.weight.dtype == torch.float32

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        norm(torch.randn(2, 4, 128, dtype=torch.bfloat16))

    refused = [w for w in caught if "fused implementation" in str(w.message)]
    assert not refused, f"fell off the fused path: {refused[0].message}"


def test_output_attentions_returns_normalised_weights(transformer_model, input_ids):
    output = transformer_model(input_ids=input_ids, output_attentions=True)
    assert len(output.attentions) == transformer_model.config.num_hidden_layers
    weights = output.attentions[0]
    torch.testing.assert_close(
        weights.sum(-1), torch.ones_like(weights.sum(-1)), rtol=1e-4, atol=1e-4
    )


def test_output_hidden_states_has_one_entry_per_layer_plus_one(model, input_ids):
    output = model(input_ids=input_ids, output_hidden_states=True)
    assert len(output.hidden_states) == model.config.num_hidden_layers + 1


def test_rejects_both_ids_and_embeds(transformer_model, input_ids):
    with pytest.raises(ValueError, match="exactly one"):
        transformer_model(
            input_ids=input_ids,
            inputs_embeds=torch.zeros(2, 12, transformer_model.config.hidden_size),
        )


def test_save_and_load_roundtrip(tmp_path, model, input_ids):
    before = model(input_ids=input_ids).logits
    model.save_pretrained(tmp_path, safe=False)

    restored = AvaForCausalLM.from_pretrained(tmp_path).eval()
    after = restored(input_ids=input_ids).logits
    torch.testing.assert_close(before, after)


def test_config_roundtrip(tmp_path, config):
    config.save_pretrained(tmp_path)
    restored = AvaConfig.from_pretrained(tmp_path)
    assert restored.to_dict() == config.to_dict()


@pytest.mark.parametrize("preset", AvaConfig.available_presets())
def test_every_preset_is_valid(preset):
    config = AvaConfig.from_preset(preset)
    assert config.num_attention_heads % config.kv_heads == 0
    assert config.head_dim % 2 == 0
    assert config.estimate_parameters() > 0


def test_chunk_size_bounds_the_scan_slab():
    """The transient (chunk, inner, d_state) slab must not grow with the model."""
    for name in ("mamba-130m", "hybrid-130m", "hybrid-1b"):
        config = AvaConfig.from_preset(name)
        chunk = config.ssm_effective_chunk_size
        slab = chunk * config.ssm_inner_dim * config.d_state
        assert slab <= 2**24, f"{name}: scan slab of {slab} elements is unbounded"
        assert 16 <= chunk <= 64


def test_wide_state_shrinks_the_chunk():
    """A state width that would make the default window expensive lowers it."""
    wide = AvaConfig(architecture_type="mamba", hidden_size=8192, d_state=512)
    assert (
        wide.ssm_effective_chunk_size
        < AvaConfig.from_preset("hybrid-130m").ssm_effective_chunk_size
    )


def test_explicit_chunk_size_wins():
    assert AvaConfig(ssm_chunk_size=64).ssm_effective_chunk_size == 64
    with pytest.raises(ValueError, match="ssm_chunk_size"):
        AvaConfig(ssm_chunk_size=0)


def test_parameter_estimate_matches_reality():
    config = tiny(architecture_type="hybrid")
    model = AvaForCausalLM(config)
    actual = model.num_parameters()
    estimated = config.estimate_parameters()
    assert abs(actual - estimated) / actual < 0.02


def test_invalid_config_is_rejected_at_construction():
    with pytest.raises(ValueError, match="multiple"):
        AvaConfig(num_attention_heads=6, kv_heads=4, head_dim=8)
    with pytest.raises(ValueError, match="even"):
        AvaConfig(head_dim=7)
    with pytest.raises(ValueError, match="architecture_type"):
        AvaConfig(architecture_type="rnn")
