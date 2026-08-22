"""LoRA and quantisation."""

import pytest
import torch
import torch.nn as nn

from ava import AvaForCausalLM
from ava.model.lora import LoRALinear, apply_lora, lora_state_dict, merge_lora
from ava.model.quantization import QuantizedLinear, quantize_model

from .helpers import tiny

# --- LoRA ---


def test_adapter_is_a_no_op_at_initialisation(transformer_model, input_ids):
    before = transformer_model(input_ids=input_ids).logits
    apply_lora(transformer_model, rank=4)
    after = transformer_model(input_ids=input_ids).logits
    torch.testing.assert_close(before, after)


def test_only_adapter_parameters_are_trainable(transformer_model):
    apply_lora(transformer_model, rank=4)
    trainable = {
        name for name, param in transformer_model.named_parameters() if param.requires_grad
    }
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_adapter_targets_are_matched_by_leaf_name(transformer_model):
    """``o_proj`` must not accidentally capture Mamba's ``out_proj``."""
    apply_lora(transformer_model, target_modules=["o_proj"], rank=4)
    wrapped = [
        name
        for name, module in transformer_model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    assert wrapped
    assert all(name.endswith("o_proj") for name in wrapped)


def test_every_target_layer_is_wrapped_exactly_once(transformer_model):
    apply_lora(transformer_model, rank=4)
    count = sum(isinstance(module, LoRALinear) for module in transformer_model.modules())
    assert count == transformer_model.config.num_hidden_layers * 4

    nested = [
        module
        for module in transformer_model.modules()
        if isinstance(module, LoRALinear) and isinstance(module.base_layer, LoRALinear)
    ]
    assert not nested, "a layer was wrapped twice"


def test_merge_preserves_the_output(transformer_model, input_ids):
    apply_lora(transformer_model, rank=4)
    with torch.no_grad():
        for module in transformer_model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.05)

    before = transformer_model(input_ids=input_ids).logits
    merge_lora(transformer_model)
    after = transformer_model(input_ids=input_ids).logits

    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    assert not any(isinstance(module, LoRALinear) for module in transformer_model.modules())


def test_adapter_state_dict_is_small(transformer_model):
    apply_lora(transformer_model, rank=4)
    adapter = lora_state_dict(transformer_model)
    full = transformer_model.state_dict()
    assert adapter
    assert sum(t.numel() for t in adapter.values()) < 0.2 * sum(
        t.numel() for t in full.values()
    )


def test_unknown_target_raises_with_a_useful_message(transformer_model):
    with pytest.raises(ValueError, match=r"No nn\.Linear modules named"):
        apply_lora(transformer_model, target_modules=["does_not_exist"])


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(4, 4), rank=0)


# --- quantisation ---


def test_quantized_linear_constructs():
    """The old implementation raised KeyError before it could ever run."""
    layer = nn.Linear(16, 8)
    quantized = QuantizedLinear(layer.weight, layer.bias, bits=8)
    assert quantized.quantized_weight.dtype == torch.int8
    assert quantized.scale.shape == (8,)


@pytest.mark.parametrize("bits", [8, 4])
def test_quantisation_error_stays_bounded(bits):
    torch.manual_seed(0)
    layer = nn.Linear(64, 32, bias=False)
    quantized = QuantizedLinear(layer.weight, None, bits=bits)

    x = torch.randn(8, 64)
    error = (quantized(x) - layer(x)).abs().max() / layer(x).abs().max()
    assert error < (0.05 if bits == 8 else 0.5)


def test_per_channel_scales_survive_an_outlier_row():
    """One huge row must not set the step size for all the others."""
    weight = torch.randn(4, 16) * 0.01
    weight[0] *= 1000

    quantized = QuantizedLinear(weight, None, bits=8)
    restored = quantized.dequantize()
    small_row_error = (restored[1] - weight[1]).abs().max() / weight[1].abs().max()
    assert small_row_error < 0.02


def test_quantize_model_replaces_linears_but_skips_the_head():
    model = AvaForCausalLM(tiny(tie_word_embeddings=False))
    quantize_model(model, bits=8)

    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, QuantizedLinear)
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())


def test_quantized_model_still_runs(input_ids):
    model = AvaForCausalLM(tiny(tie_word_embeddings=False)).eval()
    reference = model(input_ids=input_ids).logits

    quantize_model(model, bits=8)
    with torch.no_grad():
        output = model(input_ids=input_ids).logits

    assert output.shape == reference.shape
    assert torch.isfinite(output).all()


def test_unsupported_bit_width_is_rejected():
    with pytest.raises(ValueError, match="4- and 8-bit"):
        QuantizedLinear(torch.randn(4, 4), None, bits=3)


def test_zero_weight_row_does_not_divide_by_zero():
    weight = torch.zeros(2, 8)
    quantized = QuantizedLinear(weight, None, bits=8)
    assert torch.isfinite(quantized.dequantize()).all()
