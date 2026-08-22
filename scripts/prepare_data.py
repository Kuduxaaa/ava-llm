"""Train a tokenizer and binarise a corpus into a packed token stream.

    python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000

Produces:
    data/tokenizer/tokenizer.model   the SentencePiece model
    data/tokens/train.bin            packed uint16/uint32 token stream
    data/tokens/meta.json            token count and dtype
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ava.data import iter_text_file, pack_corpus
from ava.tokenizer import AvaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokens"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="1.0 for small alphabets; ~0.9995 for a long tail of rare characters.",
    )
    parser.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="Skip training and load an existing tokenizer from --tokenizer-dir.",
    )
    args = parser.parse_args()

    if args.reuse_tokenizer:
        tokenizer = AvaTokenizer.from_pretrained(args.tokenizer_dir)
        print(f"Loaded tokenizer ({len(tokenizer):,} pieces)")
    else:
        print(f"Training a {args.model_type} tokenizer on {args.corpus}...")
        tokenizer = AvaTokenizer.train(
            args.corpus,
            args.tokenizer_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )
        print(f"Tokenizer saved to {args.tokenizer_dir} ({len(tokenizer):,} pieces)")

    sample = next(iter_text_file(args.corpus))
    pieces = tokenizer.tokenize(sample)
    print(f"\nsample     {sample[:70]}...")
    print(f"pieces     {pieces[:12]}")
    print(f"fertility  {len(pieces) / max(1, len(sample.split())):.2f} tokens/word\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack_corpus(
        iter_text_file(args.corpus),
        tokenizer,
        args.output_dir / "train.bin",
    )


if __name__ == "__main__":
    main()
