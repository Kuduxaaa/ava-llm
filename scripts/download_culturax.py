"""
Download CulturaX Georgian (ka) subset to data/corpus_ka.txt

Usage:
    # First, authenticate with HuggingFace:
    huggingface-cli login

    # Then run:
    python scripts/download_culturax.py

    # Or with options:
    python scripts/download_culturax.py --max-docs 500000 --min-length 50 --output data/corpus_ka.txt
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def check_auth():
    try:
        api = HfApi()
        info = api.whoami()
        print(f"Authenticated as: {info['name']}")
        return True
    except Exception:
        return False


def download(output: Path, max_docs: int | None, min_length: int):
    if output.exists():
        print(f"File already exists: {output}")
        line_count = sum(1 for _ in open(output, encoding="utf-8"))
        size_mb = output.stat().st_size / 1e6
        print(f"  Lines: {line_count:,} | Size: {size_mb:.0f} MB")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CulturaX Georgian (ka) subset (streaming)...")
    ds = load_dataset(
        "uonlp/CulturaX",
        "ka",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    skipped = 0

    with open(output, "w", encoding="utf-8") as f:
        for doc in ds:
            text = doc["text"].strip()
            if len(text) < min_length:
                skipped += 1
                continue

            # Remove lines with too many repeated characters (low quality)
            if any(c * 10 in text for c in ".-_=*#~"):
                skipped += 1
                continue

            f.write(text + "\n")
            count += 1

            if count % 50_000 == 0:
                size_mb = output.stat().st_size / 1e6
                print(f"  {count:,} docs written ({size_mb:.0f} MB)...")

            if max_docs and count >= max_docs:
                break

    size_mb = output.stat().st_size / 1e6
    print(f"\nDone!")
    print(f"  Documents: {count:,}")
    print(f"  Skipped:   {skipped:,}")
    print(f"  Size:      {size_mb:.0f} MB")
    print(f"  Output:    {output}")


def main():
    parser = argparse.ArgumentParser(description="Download CulturaX Georgian corpus")
    parser.add_argument("--output", type=Path, default=Path("data/corpus_ka.txt"))
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of documents (default: all ~3.1M)")
    parser.add_argument("--min-length", type=int, default=20, help="Minimum document length in characters (default: 20)")
    args = parser.parse_args()

    if not check_auth():
        print("Error: HuggingFace authentication required.")
        print()
        print("CulturaX needs you to:")
        print("  1. Accept the license at https://huggingface.co/datasets/uonlp/CulturaX")
        print("  2. Run: hf auth login")
        sys.exit(1)

    download(args.output, args.max_docs, args.min_length)


if __name__ == "__main__":
    main()
