"""Run the whole pipeline end to end on a tiny slice, in about a minute.

    python scripts/preflight.py                          # synthetic corpus
    python scripts/preflight.py --corpus data/corpus.txt # the first N lines of a real one

Run this on a rented machine *before* starting the real job. It exercises every
stage a pretraining run touches -- tokenizer, packing, dataset, model, optimizer,
checkpointing, resume, generation, and the world -- and fails loudly at whichever
one is broken. A GPU hour costs more than a minute of this does, and the failures
it catches are the boring ones that otherwise surface after the download.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ava import AvaConfig, AvaForCausalLM, GenerationConfig
from ava.data import PackedDataset, collate_packed, iter_text_file, pack_corpus
from ava.tokenizer import AvaTokenizer
from ava.training import Trainer, TrainingConfig
from ava.world import AvaPerception, WorldEngine

WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "while",
    "distant",
    "thunder",
    "rolls",
    "across",
    "an",
    "empty",
    "valley",
    "and",
    "someone",
    "somewhere",
    "decides",
    "that",
    "today",
    "is",
    "finally",
    "the",
    "day",
    "to",
    "begin",
    "again",
    "with",
    "whatever",
    "small",
    "thing",
    "happens",
    "to",
    "be",
    "nearest",
    "to",
    "hand",
]


class Check:
    """A pass/fail line with a timer, so a slow stage is visible as a slow stage."""

    def __init__(self) -> None:
        self.failures = 0

    def __call__(self, label: str, fn, detail=None):
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            print(f"  FAIL  {label:<34s} {type(exc).__name__}: {exc}")
            self.failures += 1
            return None
        elapsed = time.perf_counter() - start
        if detail is not None:
            note = "" if result is None else detail(result)
        else:
            note = "" if result is None else str(result)
        print(f"  ok    {label:<34s} {elapsed:6.2f}s  {note}")
        return result


def synthetic_corpus(path: Path, lines: int = 4000) -> Path:
    """Deterministic filler with enough variety to train a small vocabulary."""
    rows = []
    for index in range(lines):
        length = 12 + index % 25
        rows.append(" ".join(WORDS[(index * 7 + i) % len(WORDS)] for i in range(length)))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--lines", type=int, default=4000, help="Lines to read or generate."
    )
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep", action="store_true", help="Do not delete the workspace.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workspace = Path(tempfile.mkdtemp(prefix="ava-preflight-"))
    check = Check()

    print(f"device    {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"gpu       {properties.name}  {properties.total_memory / 1e9:.1f} GB")
    print(f"workspace {workspace}\n")

    # --- corpus ---
    corpus = workspace / "corpus.txt"
    if args.corpus:
        head = []
        for index, line in enumerate(iter_text_file(args.corpus)):
            if index >= args.lines:
                break
            head.append(line)
        corpus.write_text("\n".join(head), encoding="utf-8")
        check("corpus (first lines of yours)", lambda: f"{len(head):,} lines")
    else:
        check(
            "corpus (synthetic)",
            lambda: (
                f"{args.lines:,} lines" if synthetic_corpus(corpus, args.lines) else None
            ),
        )

    # --- tokenizer ---
    tokenizer = check(
        "tokenizer trains",
        lambda: AvaTokenizer.train(
            corpus,
            workspace / "tokenizer",
            vocab_size=args.vocab_size,
            character_coverage=1.0,
            input_sentence_size=50_000,
        ),
        detail=lambda t: f"{len(t):,} pieces",
    )
    if tokenizer is None:
        return report(check, workspace, args.keep)

    check(
        "tokenizer round-trips",
        lambda: _assert(
            tokenizer.decode(tokenizer.encode("the quick brown fox"))
            == "the quick brown fox",
            "decode(encode(x)) != x",
        ),
    )

    # --- packing ---
    tokens_path = workspace / "tokens" / "train.bin"
    corpus_info = check(
        "corpus packs",
        lambda: pack_corpus(iter_text_file(corpus), tokenizer, tokens_path, verbose=False),
        detail=lambda c: f"{c.num_tokens:,} tokens, {c.dtype}",
    )
    if corpus_info is None:
        return report(check, workspace, args.keep)

    block = 128
    dataset = check(
        "dataset memory-maps",
        lambda: PackedDataset(tokens_path, block_size=block),
        detail=lambda d: f"{len(d):,} blocks of {block}",
    )
    if dataset is None:
        return report(check, workspace, args.keep)

    train_set, val_set = dataset.split(val_fraction=0.05)
    loader = DataLoader(
        train_set, batch_size=8, shuffle=True, collate_fn=collate_packed, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=8, collate_fn=collate_packed)

    # --- model ---
    config = AvaConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=352,
        num_hidden_layers=4,
        num_attention_heads=4,
        kv_heads=2,
        head_dim=32,
        max_position_embeddings=block,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        world_conditioning=True,
    )
    built: list = []
    check(
        "model builds",
        lambda: (
            built.append(AvaForCausalLM(config).to(device))
            or f"{built[0].num_parameters():,} parameters"
        ),
    )
    if not built:
        return report(check, workspace, args.keep)
    model = built[0]

    # --- training ---
    trainer = Trainer(
        model,
        loader,
        val_loader,
        TrainingConfig(
            max_steps=args.steps,
            learning_rate=3e-3,
            warmup_ratio=0.1,
            checkpoint_dir=str(workspace / "checkpoints"),
            save_every=max(args.steps // 2, 1),
            log_interval=max(args.steps // 4, 1),
            precision="auto",
        ),
        device=device,
    )
    trained: list = []
    check(
        "training runs",
        lambda: trained.append(trainer.train()[1]) or f"{args.steps} steps",
    )
    if not trained:
        return report(check, workspace, args.keep)

    losses = [h["train_loss"] for h in trained[0] if "train_loss" in h]
    check(
        "loss goes down",
        lambda: (
            _assert(
                len(losses) >= 2 and losses[-1] < losses[0],
                f"loss did not fall: {losses[:1]} -> {losses[-1:]}",
            )
            or f"{losses[0]:.3f} -> {losses[-1]:.3f}"
        ),
    )

    # --- checkpoint round trip ---
    checkpoint = workspace / "checkpoints" / f"ava_step_{args.steps}.pt"
    check(
        "checkpoint resumes",
        lambda: Trainer(
            AvaForCausalLM(config).to(device), loader, config=trainer.config, device=device
        ).load_checkpoint(checkpoint),
    )
    check(
        "model saves and loads",
        lambda: (
            f"{AvaForCausalLM.from_pretrained(_save(model, workspace / 'final'), device=device).num_parameters():,} parameters"
        ),
    )

    # --- generation ---
    prompt = tokenizer.encode("the quick", return_tensors="pt").to(device)
    check(
        "generation runs",
        lambda: tokenizer.decode(
            model.generate(
                prompt,
                generation_config=GenerationConfig(
                    max_new_tokens=20,
                    temperature=0.8,
                    min_p=0.05,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )[0]
        )[:60],
    )

    # --- the world ---
    perception = check(
        "world attaches",
        lambda: AvaPerception(model, WorldEngine(device=device)),
        detail=lambda p: f"{sum(q.numel() for q in p.trainable_parameters()):,} trainable",
    )
    if perception is not None:
        check(
            "perceive then respond",
            lambda: tuple(
                perception.respond(
                    prompt,
                    generation_config=GenerationConfig(
                        max_new_tokens=8, do_sample=False, eos_token_id=None
                    ),
                )[0].shape
            ),
        )

    return report(check, workspace, args.keep)


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)
    return None


def _save(model, path: Path) -> Path:
    model.save_pretrained(path)
    return path


def report(check: Check, workspace: Path, keep: bool) -> int:
    if not keep:
        shutil.rmtree(workspace, ignore_errors=True)
    print()
    if check.failures:
        print(f"{check.failures} stage(s) failed -- do not start the real run yet.")
        return 1
    print("all stages passed. the pipeline is intact end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
