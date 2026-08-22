"""The training loop."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import ThroughputMeter, evaluate_model
from .optimizer import create_hybrid_optimizer, create_optimizer, create_scheduler


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel off ``DistributedDataParallel`` and ``torch.compile`` wrappers."""
    while True:
        if hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def find_latest_checkpoint(directory: str | os.PathLike) -> str | None:
    """The newest step checkpoint in ``directory``, or ``None`` if there is none.

    Sessions on hosted notebooks are capped at a few hours, so a real run is a
    chain of them. Making "carry on from wherever you got to" a lookup rather
    than something the operator retypes each time removes the failure that costs
    the most: silently restarting from step zero.
    """
    directory = str(directory)
    if not os.path.isdir(directory):
        return None

    steps = []
    for name in os.listdir(directory):
        if name.startswith("ava_step_") and name.endswith(".pt"):
            try:
                steps.append((int(name[len("ava_step_") : -len(".pt")]), name))
            except ValueError:
                continue
    if not steps:
        return None
    return os.path.join(directory, max(steps)[1])


def has_hardware_bf16(device: torch.device) -> bool:
    """Real bf16 units, not emulation. Requires Ampere (sm_80) or newer.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True``
    and so answers yes on cards that have no bf16 hardware at all -- a Pascal
    P100 included. Believing it is expensive on a Turing T4, where the choice is
    between emulated bf16 and fp16 that actually reaches the tensor cores.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(index)
    return major >= 8


def check_device_is_supported(device: torch.device) -> None:
    """Fail early if this PyTorch build has no kernels for this GPU.

    Otherwise the first matmul raises ``no kernel image is available for
    execution on the device``, which names neither the GPU nor the fix. Hosted
    notebooks hand out older cards that current builds have dropped, so this is
    a real and confusing way to lose a session.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    arch_list = torch.cuda.get_arch_list()
    if not arch_list:
        return

    def parse(prefix: str) -> list[tuple[int, int]]:
        found = []
        for name in arch_list:
            if not name.startswith(prefix):
                continue
            digits = name[len(prefix) :].split("+", 1)[0]
            if digits.isdigit() and len(digits) >= 2:
                found.append((int(digits[:-1]), int(digits[-1])))
        return found

    # Cubins are binary compatible upward across *minor* versions within the
    # same major architecture, so sm_86 kernels run on an sm_89 card. They are
    # not compatible across majors, which is why a P100 (sm_60) finds nothing in
    # a build that starts at sm_70. PTX at or below this capability can be JIT
    # compiled forward regardless.
    if any(m == major and n <= minor for m, n in parse("sm_")):
        return
    if any((m, n) <= (major, minor) for m, n in parse("compute_")):
        return

    name = torch.cuda.get_device_name(index)
    raise RuntimeError(
        f"{name} is compute capability sm_{major}{minor}, and this PyTorch build "
        f"only has kernels for {', '.join(arch_list)}. Nothing will run on it. "
        "Choose a different accelerator, or install a build that supports this "
        "card."
    )


def resolve_precision(
    device: torch.device, requested: str
) -> tuple[torch.dtype | None, bool]:
    """Pick an autocast dtype and say whether a gradient scaler is needed.

    bf16 is preferred where the hardware has it, because it shares fp32's
    exponent range: no loss scaling, no inf/NaN skipped steps, no scaler state to
    checkpoint. Pre-Ampere cards get fp16 with a scaler, which is slower to
    babysit but is what actually reaches their tensor cores.
    """
    if requested == "fp32" or device.type == "cpu":
        return None, False
    if requested in ("bf16", "auto"):
        if has_hardware_bf16(device):
            return torch.bfloat16, False
        if requested == "bf16":
            raise RuntimeError(
                "bf16 was requested but this device has no bf16 hardware "
                "(it needs sm_80 or newer). Use precision='auto' for fp16."
            )
        return torch.float16, True  # auto -> fall back
    if requested == "fp16":
        return torch.float16, True
    raise ValueError(f"Unknown precision {requested!r}; use auto/bf16/fp16/fp32.")


@dataclass
class TrainingConfig:
    """Everything about *how* a model trains."""

    # --- schedule ---
    num_epochs: int = 1
    max_steps: int | None = None
    """Optimizer steps. Takes precedence over ``num_epochs`` when set -- which is
    what you want for pretraining, where "an epoch" is not a meaningful unit.

    When it asks for more steps than the corpus holds, the data is repeated.
    That is normal for a language without a large corpus, where the alternative
    is a model far below the size the available compute could support."""

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    lr_schedule: str = "cosine"
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # --- performance ---
    precision: str = "auto"
    compile_model: bool = False
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"
    """``adamw`` or ``muon`` (Muon on hidden matrices, AdamW on the rest)."""
    muon_lr: float = 0.02

    # --- bookkeeping ---
    checkpoint_dir: str = "checkpoints"
    save_every: int | None = None
    """Save every N optimizer steps. ``None`` saves once per epoch."""
    keep_last: int = 3
    eval_every: int | None = None
    log_interval: int = 10
    seed: int = 1337

    hub_repo: str | None = None
    """Mirror checkpoints to a Hugging Face repo as training runs.

    A hosted notebook's disk does not survive the session. Losing the machine
    before saving loses the run, and the only copy that is safe is the one that
    already left the machine."""
    hub_every: int | None = None
    """Steps between uploads. Defaults to four times ``save_every``, because a
    checkpoint with optimizer state is gigabytes and the upload is not free."""
    hub_private: bool = True

    max_hours: float | None = None
    """Stop cleanly after this long, saving first.

    Hosted notebooks are killed at a fixed wall-clock limit. Being killed loses
    everything since the last periodic save *and* leaves no final checkpoint;
    stopping ten minutes early loses ten minutes."""

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """A single-process or DDP training loop.

    Deliberately explicit rather than configurable-to-death: every knob it has
    changes numbers you can see in the log.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader

        check_device_is_supported(self.device)

        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.rank = torch.distributed.get_rank() if self.is_distributed else 0
        self.world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        self.is_main = self.rank == 0

        model = model.to(self.device)
        base = unwrap_model(model)
        if self.config.gradient_checkpointing:
            base.gradient_checkpointing_enable()
        self.model_config = base.config

        if self.config.compile_model:
            model = torch.compile(model)
        self.model = model

        self.autocast_dtype, needs_scaler = resolve_precision(
            self.device, self.config.precision
        )
        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=True) if needs_scaler else None
        )

        self.num_training_steps = self._plan_steps()
        self.optimizers = self._build_optimizers(base)
        self.schedulers = [
            create_scheduler(
                optimizer,
                self.num_training_steps,
                warmup_ratio=self.config.warmup_ratio,
                schedule=self.config.lr_schedule,
                min_lr_ratio=self.config.min_lr_ratio,
            )
            for optimizer in self.optimizers
        ]

        self.global_step = 0
        self.start_epoch = 0
        self.finished = False
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    # --- setup helpers ---

    def _plan_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        return max(1, per_epoch * self.config.num_epochs)

    def _build_optimizers(self, base: nn.Module) -> list[torch.optim.Optimizer]:
        if self.config.optimizer == "muon":
            muon, adamw = create_hybrid_optimizer(
                base,
                lr=self.config.learning_rate,
                muon_lr=self.config.muon_lr,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
            return [muon, adamw]
        if self.config.optimizer != "adamw":
            raise ValueError(
                f"Unknown optimizer {self.config.optimizer!r}; use 'adamw' or 'muon'."
            )
        return [
            create_optimizer(
                base,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        ]

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.autocast_dtype)

    def log(self, message: str) -> None:
        if self.is_main:
            print(message, flush=True)

    # --- the loop ---

    def train(self) -> tuple[nn.Module, list[dict[str, Any]]]:
        cfg = self.config
        accum = cfg.gradient_accumulation_steps

        tokens_per_step = self._tokens_per_optimizer_step()
        meter = ThroughputMeter(self.model_config, self.device)

        if cfg.hub_repo:
            self.log(f"mirroring checkpoints to {cfg.hub_repo}")
        self.log(
            f"steps={self.num_training_steps} lr={cfg.learning_rate:g} "
            f"schedule={cfg.lr_schedule} optim={cfg.optimizer} "
            f"precision={self.autocast_dtype or 'fp32'} accum={accum} "
            f"world_size={self.world_size} compile={cfg.compile_model}"
        )
        if tokens_per_step:
            total = tokens_per_step * self.num_training_steps
            self.log(f"tokens/optimizer-step: {tokens_per_step:,}")

            steps_per_pass = len(self.train_loader) // accum
            if steps_per_pass:
                passes = self.num_training_steps / steps_per_pass
                corpus = steps_per_pass * tokens_per_step
                self.log(
                    f"budget: {total:,} tokens over a {corpus:,}-token corpus "
                    f"= {passes:.1f} pass(es)"
                )
                if passes > 4:
                    self.log(
                        f"  note: {passes:.1f} passes is a lot of repetition. "
                        "Consider more data or a smaller model."
                    )

        self.model.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

        window_loss = torch.zeros((), device=self.device)
        window_batches = 0
        start_time = time.time()
        stop = False
        out_of_time = False
        is_step_boundary = True
        deadline = None if cfg.max_hours is None else start_time + cfg.max_hours * 3600.0

        epoch = self.start_epoch
        while True:
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch_index, batch in enumerate(self.train_loader):
                is_step_boundary = (batch_index + 1) % accum == 0
                loss = self._forward_backward(batch, accum, sync=is_step_boundary)

                window_loss += loss.detach()
                window_batches += 1

                if not is_step_boundary:
                    continue

                grad_norm = self._optimizer_step()
                self.global_step += 1
                meter.update(
                    self._batch_tokens(batch) * accum * self.world_size,
                    seq_len=batch["input_ids"].shape[1],
                )

                if self.global_step % cfg.log_interval == 0:
                    self._log_step(
                        window_loss, window_batches, grad_norm, meter, start_time
                    )
                    window_loss = torch.zeros((), device=self.device)
                    window_batches = 0

                if cfg.eval_every and self.global_step % cfg.eval_every == 0:
                    self._run_validation(epoch)

                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}", epoch)

                if deadline is not None and time.time() >= deadline:
                    self.log(
                        f"reached the {cfg.max_hours:g}h budget at step "
                        f"{self.global_step}; saving and stopping"
                    )
                    path = self.save_checkpoint(f"step_{self.global_step}", epoch)
                    # The final one always goes up, whatever the interval says.
                    if path:
                        self._upload_checkpoint(path)
                    stop = out_of_time = True
                    break

                if self.global_step >= self.num_training_steps:
                    stop = True
                    break

            # Any tail gradients from an incomplete accumulation window would
            # otherwise leak into the next epoch's first step.
            if window_batches and not is_step_boundary:
                self._optimizer_step()
                self.global_step += 1
                window_loss = torch.zeros((), device=self.device)
                window_batches = 0

            if not cfg.save_every:
                self.save_checkpoint(f"epoch_{epoch + 1}", epoch)
            if self.val_loader is not None:
                self._run_validation(epoch)

            epoch += 1
            if stop:
                break
            if cfg.max_steps is not None:
                # The step budget governs, so keep going over the data until it
                # is spent. Stopping at the end of one pass would silently
                # deliver a fraction of the requested training and report
                # success -- which is how a small corpus quietly halves a run.
                if self.global_step >= self.num_training_steps:
                    break
                self.log(f"pass {epoch + 1} over the corpus (step {self.global_step})")
            elif epoch >= cfg.num_epochs:
                break

        elapsed = time.time() - start_time
        self.log(
            f"Done in {elapsed / 60:.1f} min | {meter.tokens:,} tokens seen"
            + (
                f" | stopped early at step {self.global_step}/{self.num_training_steps},"
                " resume to continue"
                if out_of_time
                else ""
            )
        )
        self.finished = not out_of_time
        return unwrap_model(self.model), self.history

    def _forward_backward(self, batch: dict, accum: int, sync: bool) -> torch.Tensor:
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if key in ("input_ids", "attention_mask", "labels")
        }

        # DDP all-reduces on every backward by default; skipping the reduction
        # on non-boundary micro-batches is most of the point of accumulation.
        context = (
            self.model.no_sync()
            if (self.is_distributed and not sync and hasattr(self.model, "no_sync"))
            else contextlib.nullcontext()
        )

        with context:
            with self._autocast():
                output = self.model(**inputs)
                loss = output["loss"] / accum
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss * accum

    def _optimizer_step(self) -> torch.Tensor:
        grad_norm = torch.zeros((), device=self.device)
        for optimizer in self.optimizers:
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)

        if self.config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        for optimizer in self.optimizers:
            if self.scaler is not None:
                self.scaler.step(optimizer)
            else:
                optimizer.step()
        if self.scaler is not None:
            self.scaler.update()

        for scheduler in self.schedulers:
            scheduler.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

        return grad_norm

    def _log_step(
        self,
        window_loss: torch.Tensor,
        window_batches: int,
        grad_norm: torch.Tensor,
        meter: ThroughputMeter,
        start_time: float,
    ) -> None:
        # One synchronisation per log interval, not one per batch.
        loss = (window_loss / max(window_batches, 1)).item()
        lr = self.schedulers[0].get_last_lr()[0]
        stats = meter.report()
        self.log(
            f"step {self.global_step:>6} | loss {loss:.4f} | "
            f"ppl {math.exp(min(loss, 20)):>8.2f} | lr {lr:.2e} | "
            f"grad {float(grad_norm):.2f} | {stats['tokens_per_second']:,.0f} tok/s"
            + (f" | mfu {stats['mfu']:.1%}" if stats.get("mfu") else "")
            + f" | {time.time() - start_time:.0f}s"
        )
        self.history.append(
            {
                "step": self.global_step,
                "train_loss": loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                **stats,
            }
        )

    def _run_validation(self, epoch: int) -> float:
        if self.val_loader is None:
            return float("nan")
        val_loss = evaluate_model(
            self.model, self.val_loader, self.device, self.autocast_dtype
        )
        self.log(
            f"  validation | loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}"
        )
        self.history.append(
            {"step": self.global_step, "epoch": epoch + 1, "val_loss": val_loss}
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint("best", epoch, prune=False)
            self.log(f"  new best ({val_loss:.4f})")
        self.model.train()
        return val_loss

    # --- token accounting ---

    @staticmethod
    def _batch_tokens(batch: dict) -> int:
        return int(batch["input_ids"].numel())

    def _tokens_per_optimizer_step(self) -> int:
        try:
            sample = next(iter(self.train_loader))
        except (StopIteration, TypeError):
            return 0
        return (
            self._batch_tokens(sample)
            * self.config.gradient_accumulation_steps
            * self.world_size
        )

    # --- checkpointing ---

    def _upload_checkpoint(self, path: str) -> None:
        """Mirror one checkpoint to the Hub, under a fixed name.

        Always the same filename, so the repo holds the newest and nothing else:
        the point is a copy that survives the machine, not a history. A failure
        here is logged and ignored -- losing an upload is a setback, and
        stopping training over it would be the larger one.
        """
        repo = self.config.hub_repo
        if not repo:
            return
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo, private=self.config.hub_private, exist_ok=True)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo="checkpoint.pt",
                repo_id=repo,
                commit_message=f"step {self.global_step}",
            )
            self.log(f"  mirrored step {self.global_step} to {repo}")
        except Exception as exc:
            self.log(f"  hub upload failed ({type(exc).__name__}: {exc}); continuing")

    def save_checkpoint(self, tag: str, epoch: int, prune: bool = True) -> str | None:
        if not self.is_main:
            return None

        base = unwrap_model(self.model)
        path = os.path.join(self.config.checkpoint_dir, f"ava_{tag}.pt")
        torch.save(
            {
                "step": self.global_step,
                "epoch": epoch,
                "model": base.state_dict(),
                "optimizers": [o.state_dict() for o in self.optimizers],
                "schedulers": [s.state_dict() for s in self.schedulers],
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "best_val_loss": self.best_val_loss,
                "model_config": self.model_config.to_dict(),
                "training_config": self.config.to_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
            path,
        )
        with open(
            os.path.join(self.config.checkpoint_dir, "history.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.history, f, indent=2)

        if prune:
            self._prune_checkpoints()

        every = self.config.hub_every or ((self.config.save_every or 0) * 4)
        if self.config.hub_repo and every and self.global_step % every == 0:
            self._upload_checkpoint(path)
        return path

    def _prune_checkpoints(self) -> None:
        """Keep only the newest ``keep_last`` step checkpoints."""
        if self.config.keep_last <= 0:
            return
        directory = self.config.checkpoint_dir
        step_files = sorted(
            (
                name
                for name in os.listdir(directory)
                if name.startswith("ava_step_") and name.endswith(".pt")
            ),
            key=lambda name: int(name[len("ava_step_") : -len(".pt")]),
        )
        for name in step_files[: -self.config.keep_last]:
            os.remove(os.path.join(directory, name))

    def load_checkpoint(self, path: str | os.PathLike) -> None:
        """Resume optimizer, schedule and step counter -- not just the weights.

        Restoring weights alone restarts the learning-rate schedule from zero
        and throws away the Adam moments, which usually shows up as a loss spike
        that looks like a data problem.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        unwrap_model(self.model).load_state_dict(checkpoint["model"])

        for optimizer, state in zip(self.optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(self.schedulers, checkpoint["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        if self.scaler is not None and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = checkpoint.get("step", 0)
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        self.log(f"Resumed from {path} at step {self.global_step}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    device: torch.device | None = None,
    training_config: TrainingConfig | None = None,
    resume_from: str | os.PathLike | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Convenience wrapper around :class:`Trainer`."""
    trainer = Trainer(model, train_loader, val_loader, training_config, device)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    return trainer.train()
