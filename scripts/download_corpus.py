"""Stream a text corpus from the Hugging Face Hub into a plain text file.

Dataset- and language-agnostic: point it at any dataset that has a text column.

    python scripts/download_corpus.py --dataset uonlp/CulturaX --config en
    python scripts/download_corpus.py --dataset HuggingFaceFW/fineweb --config sample-10BT
    python scripts/download_corpus.py --dataset wikimedia/wikipedia --config 20231101.fr

Several sources can be pooled into one corpus with --append, which is usually
necessary for a language that does not have a single large clean source:

    python scripts/download_corpus.py --dataset HuggingFaceFW/fineweb-2         --config kat_Geor --output data/corpus.txt
    python scripts/download_corpus.py --dataset wikimedia/wikipedia         --config 20231101.ka --output data/corpus.txt --append

Gated datasets need `hf auth login` first. On a hosted notebook that means an
HF token in the secrets store, not an interactive login.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ava.data import is_low_quality

#: Rough bytes of UTF-8 per token, by script. Latin text is about one byte per
#: character and a token is a few characters; Georgian, Cyrillic, Greek and the
#: Indic scripts are two or three bytes per character, so the same token budget
#: costs several times the disk. Used only for a progress estimate.
BYTES_PER_TOKEN = {"latin": 4.5, "two_byte": 8.0, "three_byte": 11.0}


def estimate_total(dataset: str, config: str | None, split: str) -> int | None:
    """Documents in this split, if the dataset card records it.

    Streaming hides the size, so "how long is this going to take" is otherwise
    unanswerable until it stops.
    """
    try:
        from datasets import load_dataset_builder

        info = load_dataset_builder(dataset, config).info
        entry = info.splits.get(split) if info.splits else None
        return getattr(entry, "num_examples", None) or None
    except Exception:
        return None


def bytes_per_token(sample: str) -> float:
    """Guess the disk cost of a token from the script the text is written in."""
    if not sample:
        return BYTES_PER_TOKEN["latin"]
    width = len(sample.encode("utf-8")) / len(sample)
    if width < 1.4:
        return BYTES_PER_TOKEN["latin"]
    if width < 2.5:
        return BYTES_PER_TOKEN["two_byte"]
    return BYTES_PER_TOKEN["three_byte"]


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def download(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    output = args.output
    if output.exists() and not (args.force or args.append):
        size_mb = output.stat().st_size / 1e6
        print(
            f"{output} already exists ({size_mb:.0f} MB). Pass --force to replace "
            "it or --append to add to it."
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and output.exists() else "w"
    print(f"Streaming {args.dataset}" + (f" [{args.config}]" if args.config else ""))

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )

    total = (
        None if args.no_estimate else estimate_total(args.dataset, args.config, args.split)
    )
    if total:
        print(f"{total:,} documents in this split")

    kept = skipped = 0
    started = time.perf_counter()
    token_cost = None
    with open(output, mode, encoding="utf-8") as handle:
        for record in dataset:
            text = (record.get(args.text_column) or "").strip()
            text = " ".join(text.split())  # collapse newlines: one doc per line

            if is_low_quality(text, args.min_length, args.max_symbol_ratio):
                skipped += 1
                continue

            handle.write(text + "\n")
            kept += 1
            if token_cost is None:
                token_cost = bytes_per_token(text)

            if kept % 50_000 == 0:
                handle.flush()
                size_mb = output.stat().st_size / 1e6
                elapsed = time.perf_counter() - started
                rate = kept / max(elapsed, 1e-9)

                line = (
                    f"  {kept:,} kept / {skipped:,} skipped ({size_mb:.0f} MB, "
                    f"~{size_mb * 1e6 / token_cost / 1e6:.0f}M tokens) "
                    f"| {rate:.0f} docs/s | {_duration(elapsed)} elapsed"
                )
                if args.max_docs:
                    remaining = (args.max_docs - kept) / max(rate, 1e-9)
                    line += f" | ~{_duration(remaining)} left"
                elif total:
                    line += (
                        f" | {kept / total:.0%} of split"
                        f" | ~{_duration((total - kept) / max(rate, 1e-9))} left"
                    )
                print(line, flush=True)

            if args.max_docs and kept >= args.max_docs:
                break

    size_mb = output.stat().st_size / 1e6
    elapsed = time.perf_counter() - started
    print(f"\nkept     {kept:,}")
    print(f"skipped  {skipped:,}")
    print(f"size     {size_mb:.0f} MB")
    print(f"tokens   ~{size_mb * 1e6 / (token_cost or 4.5) / 1e6:.0f}M (rough)")
    print(f"took     {_duration(elapsed)}")
    print(f"output   {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hub dataset id")
    parser.add_argument("--config", default=None, help="Dataset config / language")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.3)
    parser.add_argument(
        "--no-estimate",
        action="store_true",
        help="Skip the split-size lookup, which needs one extra Hub request.",
    )
    parser.add_argument("--force", action="store_true", help="Replace the output.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the output instead of replacing it. For a low-resource "
        "language, one source is rarely enough and several have to be pooled.",
    )
    args = parser.parse_args()

    try:
        download(args)
    except ImportError:
        print("This script needs the datasets extra: pip install 'ava-llm[data]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
