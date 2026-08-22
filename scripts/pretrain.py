"""Pretrain an Ava model on a packed token stream.

Single GPU:
    python scripts/pretrain.py --tokens data/tokens/train.bin --preset hybrid-130m

Multi-GPU (same flags, torchrun handles the rest):
    torchrun --nproc_per_node=4 scripts/pretrain.py --tokens data/tokens/train.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ava import AvaConfig, AvaForCausalLM
from ava.data import PackedDataset, collate_packed
from ava.tokenizer import AvaTokenizer
from ava.training import TrainingConfig, train_model
from ava.utils import cleanup_distributed, model_summary, setup_distributed, wrap_ddp


def build_loaders(args, rank: int, world_size: int):
    dataset = PackedDataset(args.tokens, block_size=args.block_size)
    train_set, val_set = dataset.split(val_fraction=args.val_fraction)

    def make(subset, shuffle: bool):
        sampler = (
            DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            if world_size > 1
            else None
        )
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            collate_fn=collate_packed,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    return make(train_set, True), make(val_set, False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument(
        "--preset", default="hybrid-130m", choices=AvaConfig.available_presets()
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--schedule", default="wsd", choices=["cosine", "wsd", "linear", "constant"]
    )
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument(
        "--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"]
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    vocab_size = None
    if args.tokenizer_dir.exists():
        vocab_size = len(AvaTokenizer.from_pretrained(args.tokenizer_dir))

    config = AvaConfig.from_preset(
        args.preset,
        max_position_embeddings=max(args.block_size, 2048),
        gradient_checkpointing=args.grad_checkpointing,
        **({"vocab_size": vocab_size} if vocab_size else {}),
    )
    model = AvaForCausalLM(config)

    if is_main:
        print(model_summary(model))
        print()

    model = wrap_ddp(model.to(device), device)
    train_loader, val_loader = build_loaders(args, rank, world_size)

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_schedule=args.schedule,
        optimizer=args.optimizer,
        precision=args.precision,
        compile_model=args.compile,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=str(args.checkpoint_dir),
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
    )

    trained, _ = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        training_config=training_config,
        resume_from=args.resume,
    )

    if is_main:
        output = args.checkpoint_dir / "final"
        trained.save_pretrained(output)
        print(f"Saved final model to {output}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
