import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Quantized linear layer for inference"""

    def __init__(self, weight, bias=None, bits=8):
        super().__init__()
        self.bits = bits
        self.input_features = weight.shape[1]
        self.output_features = weight.shape[0]

        self.scale = weight.abs().max() / ((2 ** (bits - 1)) - 1)
        self.zero_point = 0

        quantized_weight = (
            (weight / self.scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        )
        self.register_buffer("quantized_weight", quantized_weight.to(torch.int8))
        self.register_buffer("scale", self.scale)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        weight_dequantized = self.quantized_weight.float() * self.scale
        return F.linear(x, weight_dequantized, self.bias)


def quantize_model(model, bits=8):
    """Quantize a model to int8 precision"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name

            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)

            quantized_module = QuantizedLinear(
                module.weight.data,
                module.bias.data if module.bias is not None else None,
                bits=bits,
            )

            setattr(parent, child_name, quantized_module)

    return model
