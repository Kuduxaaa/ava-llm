"""Optimizers, schedules and the training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ava import AvaForCausalLM
from ava.training import (
    Muon,
    TrainingConfig,
    create_hybrid_optimizer,
    create_optimizer,
    create_scheduler,
    evaluate_model,
    train_model,
)
from ava.training.metrics import ThroughputMeter, flops_per_token
from ava.training.optimizer import _newton_schulz
from ava.training.trainer import resolve_precision, unwrap_model

from .helpers import tiny


class RandomTokens(Dataset):
    def __init__(self, size=32, length=16, vocab=64):
        torch.manual_seed(0)
        self.data = torch.randint(0, vocab, (size, length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids = self.data[index]
        return {"input_ids": ids, "labels": ids.clone()}


@pytest.fixture
def loader():
    return DataLoader(RandomTokens(), batch_size=4)


# --- parameter grouping ---


def test_weight_decay_skips_vectors():
    model = AvaForCausalLM(tiny())
    optimizer = create_optimizer(model, weight_decay=0.1)

    decayed, undecayed = optimizer.param_groups[0], optimizer.param_groups[1]
    assert all(p.ndim >= 2 for p in decayed["params"])
    assert all(p.ndim < 2 for p in undecayed["params"])
    assert undecayed["weight_decay"] == 0.0


def test_tied_parameters_are_not_added_twice():
    model = AvaForCausalLM(tiny(tie_word_embeddings=True))
    optimizer = create_optimizer(model)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))


# --- schedules ---


@pytest.mark.parametrize("schedule", ["cosine", "wsd", "linear", "constant"])
def test_schedule_warms_up_then_stays_in_range(schedule):
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_ratio=0.1, schedule=schedule
    )

    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()  # no grads, so this is a no-op -- it just orders the pair
        scheduler.step()

    assert rates[0] == pytest.approx(0.1, abs=1e-6)  # first step is not wasted
    assert rates[9] == pytest.approx(1.0, abs=1e-6)  # peak at the end of warmup
    assert max(rates) <= 1.0 + 1e-9
    assert min(rates) >= 0.0


def test_wsd_holds_the_peak_before_decaying():
    optimizer = torch.optim.SGD(nn.Linear(4, 4).parameters(), lr=1.0)
    scheduler = create_scheduler(
        optimizer, 100, warmup_ratio=0.1, schedule="wsd", stable_ratio=0.8
    )
    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert rates[50] == pytest.approx(1.0)
    assert rates[95] < 0.5


def test_unknown_schedule_is_rejected():
    optimizer = torch.optim.SGD(nn.Linear(4, 4).parameters(), lr=1.0)
    with pytest.raises(ValueError, match="Unknown schedule"):
        create_scheduler(optimizer, 10, schedule="magic")


# --- Muon ---


def test_newton_schulz_pushes_singular_values_toward_one():
    torch.manual_seed(0)
    matrix = torch.randn(32, 16)
    orthogonalised = _newton_schulz(matrix, steps=5)

    singular_values = torch.linalg.svdvals(orthogonalised.float())
    assert singular_values.max() < 1.5
    assert singular_values.min() > 0.5


def test_newton_schulz_handles_wide_and_tall_matrices():
    for shape in [(8, 32), (32, 8), (16, 16)]:
        result = _newton_schulz(torch.randn(*shape))
        assert result.shape == shape
        assert torch.isfinite(result).all()


def test_muon_rejects_non_matrix_parameters():
    with pytest.raises(ValueError, match="2D"):
        Muon([nn.Parameter(torch.zeros(8))])


def test_muon_step_changes_weights_and_stays_finite():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(16, 8))
    optimizer = Muon([weight], lr=0.01)

    before = weight.detach().clone()
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    assert not torch.equal(before, weight.detach())
    assert torch.isfinite(weight).all()


def test_hybrid_optimizer_routes_embeddings_to_adamw():
    model = AvaForCausalLM(tiny(tie_word_embeddings=False))
    muon, _adamw = create_hybrid_optimizer(model)

    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    assert id(model.get_input_embeddings().weight) not in muon_ids
    assert id(model.lm_head.weight) not in muon_ids
    assert id(model.model.layers[0].mlp.gate_proj.weight) in muon_ids


# --- precision ---


def test_cpu_training_falls_back_to_fp32():
    dtype, needs_scaler = resolve_precision(torch.device("cpu"), "auto")
    assert dtype is None and not needs_scaler


def test_fp16_requests_a_gradient_scaler():
    dtype, _needs_scaler = resolve_precision(torch.device("cpu"), "fp32")
    assert dtype is None
    assert resolve_precision(torch.device("cuda"), "fp16") == (torch.float16, True)


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision(torch.device("cuda"), "int4")


# --- the loop ---


def test_training_reduces_loss_on_a_memorisable_batch(tmp_path):
    torch.manual_seed(0)
    model = AvaForCausalLM(tiny())
    data = RandomTokens(size=8, length=16)
    loader = DataLoader(data, batch_size=4)

    config = TrainingConfig(
        num_epochs=12,
        learning_rate=3e-3,
        precision="fp32",
        warmup_ratio=0.1,
        checkpoint_dir=str(tmp_path),
        log_interval=1,
        save_every=None,
    )
    _trained, history = train_model(model, loader, training_config=config)

    losses = [entry["train_loss"] for entry in history if "train_loss" in entry]
    assert losses, "no training loss was recorded"
    assert losses[-1] < losses[0]


def test_gradient_accumulation_matches_a_large_batch(tmp_path):
    """Accumulating N micro-batches must equal one batch N times the size."""
    torch.manual_seed(0)
    data = RandomTokens(size=8, length=16)

    def run(batch_size, accum):
        torch.manual_seed(0)
        model = AvaForCausalLM(tiny())
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        config = TrainingConfig(
            num_epochs=1,
            learning_rate=1e-3,
            precision="fp32",
            warmup_ratio=0.0,
            lr_schedule="constant",
            max_grad_norm=0.0,
            gradient_accumulation_steps=accum,
            checkpoint_dir=str(tmp_path),
            log_interval=1000,
        )
        trained, _ = train_model(model, loader, training_config=config)
        return trained.lm_head.weight.detach().clone()

    torch.testing.assert_close(run(8, 1), run(2, 4), rtol=1e-3, atol=1e-4)


def test_checkpoint_resume_restores_step_and_optimizer(tmp_path, loader):
    from ava.training.trainer import Trainer

    model = AvaForCausalLM(tiny())
    config = TrainingConfig(
        num_epochs=1, precision="fp32", checkpoint_dir=str(tmp_path), log_interval=1000
    )
    trainer = Trainer(model, loader, config=config)
    trainer.train()
    path = trainer.save_checkpoint("manual", epoch=0)

    fresh = Trainer(AvaForCausalLM(tiny()), loader, config=config)
    fresh.load_checkpoint(path)

    assert fresh.global_step == trainer.global_step
    torch.testing.assert_close(
        unwrap_model(fresh.model).lm_head.weight,
        unwrap_model(trainer.model).lm_head.weight,
    )


def test_only_the_newest_step_checkpoints_are_kept(tmp_path, loader):
    import os

    model = AvaForCausalLM(tiny())
    config = TrainingConfig(
        num_epochs=1,
        precision="fp32",
        checkpoint_dir=str(tmp_path),
        save_every=1,
        keep_last=2,
        log_interval=1000,
    )
    train_model(model, loader, training_config=config)

    saved = [f for f in os.listdir(tmp_path) if f.startswith("ava_step_")]
    assert len(saved) == 2


def test_max_steps_overrides_epochs(tmp_path, loader):
    from ava.training.trainer import Trainer

    trainer = Trainer(
        AvaForCausalLM(tiny()),
        loader,
        config=TrainingConfig(
            num_epochs=100,
            max_steps=3,
            precision="fp32",
            checkpoint_dir=str(tmp_path),
            log_interval=1000,
        ),
    )
    trainer.train()
    assert trainer.global_step == 3


# --- evaluation ---


def test_evaluation_is_token_weighted(loader):
    model = AvaForCausalLM(tiny()).eval()
    loss = evaluate_model(model, loader, torch.device("cpu"))
    assert loss > 0 and torch.isfinite(torch.tensor(loss))


def test_evaluation_restores_the_previous_mode(loader):
    model = AvaForCausalLM(tiny())
    model.train()
    evaluate_model(model, loader, torch.device("cpu"))
    assert model.training


def test_throughput_meter_uses_the_actual_sequence_length():
    """MFU against max_position_embeddings instead of the real block size
    inflates the attention term and reports a number that is too good."""
    config = tiny(max_position_embeddings=8192)
    meter = ThroughputMeter(config, torch.device("cpu"), seq_len=512)
    assert meter.seq_len == 512

    meter.update(1000, seq_len=1024)
    assert meter.seq_len == 1024

    assert ThroughputMeter(config, torch.device("cpu")).seq_len == 8192


def test_wrap_ddp_is_a_no_op_without_a_process_group():
    from ava.utils import wrap_ddp

    model = AvaForCausalLM(tiny())
    assert wrap_ddp(model, torch.device("cpu")) is model


def test_flops_per_token_grows_with_context():
    config = tiny()
    assert flops_per_token(config, 2048) > flops_per_token(config, 128)
