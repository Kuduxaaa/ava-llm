"""Stream a text corpus from the Hugging Face Hub into a plain text file.

Dataset- and language-agnostic: point it at any dataset that has a text column.

    python scripts/download_corpus.py --dataset uonlp/CulturaX --config en
    python scripts/download_corpus.py --dataset HuggingFaceFW/fineweb --config sample-10BT
    python scripts/download_corpus.py --dataset wikimedia/wikipedia --config 20231101.fr

Gated datasets need `hf auth login` first.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


def is_low_quality(text: str, min_length: int, max_symbol_ratio: float) -> bool:
    """Cheap, language-neutral quality filters.

    Every rule here is about *structure*, not vocabulary: run-length, symbol
    density and replacement characters flag boilerplate and mojibake in any
    script, whereas a stopword list would only work for the language it was
    written for.
    """
    if len(text) < min_length:
        return True
    if "�" in text:  # decoding already failed upstream
        return True
    if any(character * 10 in text for character in ".-_=*#~/\\|"):
        return True

    symbols = sum(
        1 for character in text if unicodedata.category(character).startswith(("P", "S"))
    )
    return symbols / len(text) > max_symbol_ratio


def download(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    output = args.output
    if output.exists() and not args.force:
        size_mb = output.stat().st_size / 1e6
        print(f"{output} already exists ({size_mb:.0f} MB). Pass --force to overwrite.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {args.dataset}" + (f" [{args.config}]" if args.config else ""))

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )

    kept = skipped = 0
    with open(output, "w", encoding="utf-8") as handle:
        for record in dataset:
            text = (record.get(args.text_column) or "").strip()
            text = " ".join(text.split())  # collapse newlines: one doc per line

            if is_low_quality(text, args.min_length, args.max_symbol_ratio):
                skipped += 1
                continue

            handle.write(text + "\n")
            kept += 1

            if kept % 50_000 == 0:
                size_mb = output.stat().st_size / 1e6
                print(f"  {kept:,} kept / {skipped:,} skipped ({size_mb:.0f} MB)")

            if args.max_docs and kept >= args.max_docs:
                break

    size_mb = output.stat().st_size / 1e6
    print(f"\nkept     {kept:,}")
    print(f"skipped  {skipped:,}")
    print(f"size     {size_mb:.0f} MB")
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        download(args)
    except ImportError:
        print("This script needs the datasets extra: pip install 'ava-llm[data]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
