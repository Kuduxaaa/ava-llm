import pytest
import torch

from ava import AvaConfig, AvaForCausalLM

from .helpers import tiny


@pytest.fixture(params=["transformer", "mamba", "hybrid"])
def architecture(request) -> str:
    return request.param


@pytest.fixture
def config(architecture) -> AvaConfig:
    return tiny(architecture_type=architecture)


@pytest.fixture
def model(config) -> AvaForCausalLM:
    torch.manual_seed(0)
    return AvaForCausalLM(config).eval()


@pytest.fixture
def transformer_model() -> AvaForCausalLM:
    torch.manual_seed(0)
    return AvaForCausalLM(tiny(architecture_type="transformer")).eval()


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 64, (2, 12))
