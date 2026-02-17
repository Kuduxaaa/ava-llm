import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.base_layer(x)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T

        return result + lora_output * self.scaling


def apply_lora_to_model(
    model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], rank=8, alpha=16
):

    """
    Apply LoRA fine-tuning to specific modules in the model
    """
    
    for name, module in model.named_modules():
        if any(target_name in name for target_name in target_modules) and isinstance(
            module, nn.Linear
        ):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name

            parent = model

            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)

            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True

    return model
