import math
import os
import time

import torch

from .optimizer import create_optimizer, create_scheduler


class TrainingConfig:
    """Training hyperparameters and settings."""

    def __init__(
        self,
        num_epochs=3,
        learning_rate=5e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        max_grad_norm=1.0,
        use_amp=True,
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,
        checkpoint_dir="checkpoints",
        log_interval=10,
        compile_model=False,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.compile_model = compile_model


def train_model(
    model,
    train_loader,
    val_loader=None,
    device=None,
    training_config=None,
    # Legacy arguments for backward compatibility
    num_epochs=None,
    optimizer=None,
    checkpoint_dir=None,
    learning_rate=None,
):
    """Train model with AMP, gradient accumulation, LR scheduling, and torch.compile."""
    if training_config is None:
        training_config = TrainingConfig(
            num_epochs=num_epochs or 3,
            learning_rate=learning_rate or 5e-4,
            checkpoint_dir=checkpoint_dir or "checkpoints",
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = training_config
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Hardware-level speed optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model.to(device)

    # torch.compile for kernel fusion (~1.5-2x speedup)
    compiled = False
    if cfg.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        compiled = True

    if optimizer is None:
        params_model = model._orig_mod if compiled else model
        optimizer = create_optimizer(
            params_model, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=cfg.betas
        )

    num_training_steps = (
        len(train_loader) // cfg.gradient_accumulation_steps * cfg.num_epochs
    )
    scheduler = create_scheduler(optimizer, num_training_steps, cfg.warmup_ratio)

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    print(f"Training: {cfg.num_epochs} epochs, lr={cfg.learning_rate}, "
          f"AMP={'on' if use_amp else 'off'}, grad_accum={cfg.gradient_accumulation_steps}, "
          f"compile={'on' if compiled else 'off'}")
    print(f"Total training steps: {num_training_steps}")

    # Get config from compiled or original model
    model_config = (model._orig_mod if compiled else model).config

    best_val_loss = float("inf")
    start_time = time.time()
    global_step = 0
    total_tokens = 0
    history = []

    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            max_id = torch.max(input_ids).item()
            if max_id >= model_config.vocab_size:
                continue

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * cfg.gradient_accumulation_steps
            total_tokens += input_ids.numel()
            batch_count += 1

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    avg = total_loss / batch_count
                    ppl = math.exp(min(avg, 20))
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    tok_s = total_tokens / elapsed
                    print(
                        f"Step {global_step} | "
                        f"Epoch {epoch+1}/{cfg.num_epochs} | "
                        f"Loss: {avg:.4f} | PPL: {ppl:.1f} | "
                        f"LR: {lr:.2e} | "
                        f"{tok_s:.0f} tok/s | {elapsed:.1f}s"
                    )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(batch_count, 1)
        ppl = math.exp(min(avg_loss, 20))
        tok_s = total_tokens / (time.time() - start_time)
        print(f"Epoch {epoch+1}/{cfg.num_epochs} done in {epoch_time:.1f}s | "
              f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | {tok_s:.0f} tok/s")

        # Save checkpoint (use unwrapped model for state_dict)
        save_model = model._orig_mod if compiled else model
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"ava_model_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "config": model_config.to_dict(),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

        # Validation
        val_loss = None
        if val_loader:
            val_loss = _validate(model, val_loader, device, use_amp, model_config)
            print(f"Validation Loss: {val_loss:.4f} | PPL: {math.exp(min(val_loss, 20)):.1f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg.checkpoint_dir, "ava_model_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": save_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                        "config": model_config.to_dict(),
                    },
                    best_path,
                )
                print(f"New best model saved (val_loss={best_val_loss:.4f})")

            model.train()

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        })

    total_time = time.time() - start_time
    tok_s = total_tokens / total_time
    print(f"Training completed in {total_time:.1f}s | {tok_s:.0f} tok/s avg")

    return_model = model._orig_mod if compiled else model
    return return_model, history


def _validate(model, val_loader, device, use_amp=False, model_config=None):
    """Run validation loop and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    vocab_size = model_config.vocab_size if model_config else float("inf")

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if torch.max(input_ids).item() >= vocab_size:
                continue

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

    return total_loss / max(num_batches, 1)
