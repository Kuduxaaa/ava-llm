import torch
import torch.nn as nn


class AvaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-5):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        return (self.weight * hidden_states).to(original_dtype)
