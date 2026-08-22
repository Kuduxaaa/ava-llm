"""The chunked scan must agree with the textbook recurrence it replaces."""

import pytest
import torch

from ava.model.cache import MambaLayerCache
from ava.model.mamba import SelectiveSSM, _prefix_scan, _scan_chunk

from .helpers import tiny


def reference_scan(a_bar, b_x, state):
    """h[t] = A[t] * h[t-1] + Bx[t], written out one step at a time."""
    states = []
    for t in range(a_bar.shape[1]):
        state = a_bar[:, t] * state + b_x[:, t]
        states.append(state)
    return torch.stack(states, dim=1), state


def test_prefix_scan_matches_the_recurrence():
    torch.manual_seed(0)
    gates = torch.rand(2, 16, 3, 4) * 0.9 + 0.05
    values = torch.randn(2, 16, 3, 4)

    scanned, cumulative = _prefix_scan(gates.clone(), values.clone())
    expected, _ = reference_scan(gates, values, torch.zeros(2, 3, 4))

    torch.testing.assert_close(scanned, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cumulative, gates.cumprod(dim=1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("length", [1, 2, 5, 8, 13, 16])
def test_prefix_scan_handles_non_power_of_two_lengths(length):
    torch.manual_seed(length)
    gates = torch.rand(1, length, 2, 2) * 0.9 + 0.05
    values = torch.randn(1, length, 2, 2)

    scanned, _ = _prefix_scan(gates.clone(), values.clone())
    expected, _ = reference_scan(gates, values, torch.zeros(1, 2, 2))
    torch.testing.assert_close(scanned, expected, rtol=1e-4, atol=1e-5)


def test_scan_chunk_folds_in_the_incoming_state():
    torch.manual_seed(0)
    gates = torch.rand(2, 8, 3, 4) * 0.9 + 0.05
    values = torch.randn(2, 8, 3, 4)
    state = torch.randn(2, 3, 4)

    states, last = _scan_chunk(gates.clone(), values.clone(), state)
    expected_states, expected_last = reference_scan(gates, values, state)

    torch.testing.assert_close(states, expected_states, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(last, expected_last, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 64])
def test_chunking_does_not_change_the_result(chunk_size):
    """Chunk size is a memory knob, not a numerical one."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 17, 32)

    torch.manual_seed(1)
    reference = SelectiveSSM(tiny(ssm_chunk_size=1024)).eval()
    torch.manual_seed(1)
    chunked = SelectiveSSM(tiny(ssm_chunk_size=chunk_size)).eval()

    with torch.no_grad():
        torch.testing.assert_close(chunked(hidden), reference(hidden), rtol=1e-4, atol=1e-5)


def test_recurrent_step_matches_the_parallel_pass():
    """Decoding one token at a time must reproduce the full-sequence output."""
    torch.manual_seed(0)
    ssm = SelectiveSSM(tiny()).eval()
    hidden = torch.randn(2, 10, 32)

    with torch.no_grad():
        reference = ssm(hidden)

        cache = MambaLayerCache()
        stepped = [ssm(hidden[:, i : i + 1], cache=cache) for i in range(10)]

    torch.testing.assert_close(torch.cat(stepped, dim=1), reference, rtol=2e-4, atol=2e-4)


def test_prefill_then_step_matches_the_parallel_pass():
    torch.manual_seed(0)
    ssm = SelectiveSSM(tiny()).eval()
    hidden = torch.randn(1, 12, 32)

    with torch.no_grad():
        reference = ssm(hidden)

        cache = MambaLayerCache()
        prefill = ssm(hidden[:, :7], cache=cache)
        rest = [ssm(hidden[:, i : i + 1], cache=cache) for i in range(7, 12)]

    torch.testing.assert_close(
        torch.cat([prefill, *rest], dim=1), reference, rtol=2e-4, atol=2e-4
    )


def test_conv_state_holds_the_lookback_window():
    ssm = SelectiveSSM(tiny()).eval()
    cache = MambaLayerCache()
    with torch.no_grad():
        ssm(torch.randn(2, 6, 32), cache=cache)

    assert cache.conv_state.shape == (2, ssm.inner_dim, ssm.d_conv - 1)
    assert cache.ssm_state.shape == (2, ssm.inner_dim, ssm.d_state)


def test_cached_state_does_not_alias_the_scan_slab():
    """A view of the last timestep would pin the whole chunk inside the cache.

    Shape alone cannot catch this: the state has the right shape either way, it
    just keeps a (batch, chunk, inner, d_state) slab alive behind it.
    """
    ssm = SelectiveSSM(tiny(ssm_chunk_size=8)).eval()
    cache = MambaLayerCache()
    with torch.no_grad():
        ssm(torch.randn(2, 24, 32), cache=cache)

    state = cache.ssm_state
    storage_bytes = state.untyped_storage().nbytes()
    assert storage_bytes == state.numel() * state.element_size(), (
        f"cached SSM state backs onto {storage_bytes} bytes for "
        f"{state.numel()} elements -- it is a view, not a copy"
    )


def test_short_prefill_left_pads_the_conv_state():
    """A prompt shorter than the conv kernel still needs a full-width window."""
    ssm = SelectiveSSM(tiny()).eval()
    cache = MambaLayerCache()
    with torch.no_grad():
        ssm(torch.randn(1, 1, 32), cache=cache)

    assert cache.conv_state.shape[-1] == ssm.d_conv - 1
    assert torch.all(cache.conv_state[..., : ssm.d_conv - 2] == 0)


def test_state_matrix_is_stable():
    """A = -exp(A_log) is negative by construction, so the system cannot blow up."""
    ssm = SelectiveSSM(tiny())
    assert torch.all(-torch.exp(ssm.A_log) < 0)


def test_dt_bias_spans_a_range_of_timescales():
    ssm = SelectiveSSM(tiny(hidden_size=256))
    dt = torch.nn.functional.softplus(ssm.dt_proj.bias)
    assert dt.min() < dt.max() / 10, "dt initialisation collapsed to one timescale"
    assert dt.min() > 0


def test_gradients_flow_through_the_chunked_scan():
    ssm = SelectiveSSM(tiny(ssm_chunk_size=4)).train()
    hidden = torch.randn(1, 12, 32, requires_grad=True)
    ssm(hidden).sum().backward()

    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, param in ssm.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
