"""Train a tokenizer and binarise a corpus into a packed token stream.

From a local file:

    python scripts/prepare_data.py --corpus data/corpus.txt --vocab-size 32000

Or straight from the Hub, never landing the text on disk:

    python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-2 \
        --config kat_Geor --vocab-size 32000

The streaming path exists because the text is usually much larger than the
tokens it becomes. Georgian FineWeb-2 is 28.6 GB of text and 5.2 GB of tokens,
and a hosted notebook will not hold both. Streaming reads the source twice --
once to sample a tokenizer, once to pack -- spending bandwidth instead of disk.

Produces:
    data/tokenizer/tokenizer.model   the SentencePiece model
    data/tokens/train.bin            packed uint16/uint32 token stream
    data/tokens/meta.json            token count and dtype
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ava.data import iter_hub_dataset, iter_text_file, pack_corpus, write_sample
from ava.tokenizer import AvaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="A local one-document-per-line file.")
    source.add_argument("--dataset", help="A Hub dataset id, streamed rather than saved.")

    parser.add_argument("--config", default=None, help="Dataset config / language.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=400_000,
        help="Documents written to disk to train the tokenizer on when streaming. "
        "SentencePiece subsamples anyway, so this need not be the whole corpus.",
    )
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
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="Keep the tokenizer sample file instead of deleting it after training.",
    )
    args = parser.parse_args()

    def stream(label: str, max_docs: int | None):
        return iter_hub_dataset(
            args.dataset,
            config=args.config,
            split=args.split,
            text_column=args.text_column,
            min_length=args.min_length,
            max_docs=max_docs,
            every=100_000,
            label=label,
        )

    # --- tokenizer ---
    if args.reuse_tokenizer:
        tokenizer = AvaTokenizer.from_pretrained(args.tokenizer_dir)
        print(f"Loaded tokenizer ({len(tokenizer):,} pieces)")
    else:
        if args.corpus:
            training_text = args.corpus
        else:
            training_text = args.output_dir / "tokenizer_sample.txt"
            print(f"Sampling {args.sample_docs:,} documents for the tokenizer...")
            written = write_sample(stream("sample", None), training_text, args.sample_docs)
            print(f"  wrote {written:,} documents to {training_text}")

        print(f"Training a {args.model_type} tokenizer...")
        tokenizer = AvaTokenizer.train(
            training_text,
            args.tokenizer_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )
        print(f"Tokenizer saved to {args.tokenizer_dir} ({len(tokenizer):,} pieces)")

        if not args.corpus and not args.keep_sample:
            Path(training_text).unlink(missing_ok=True)

    # --- fertility, on a document from the real source ---
    first = next(iter_text_file(args.corpus) if args.corpus else stream("check", 1))
    pieces = tokenizer.tokenize(first)
    fertility = len(pieces) / max(1, len(first.split()))
    print(f"\nsample     {first[:70]}...")
    print(f"pieces     {pieces[:12]}")
    print(f"fertility  {fertility:.2f} tokens/word")
    if fertility > 3.0:
        print("           high -- the vocabulary does not fit this corpus well.")
    print()

    # --- pack ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents = (
        iter_text_file(args.corpus) if args.corpus else stream("packing", args.max_docs)
    )
    pack_corpus(documents, tokenizer, args.output_dir / "train.bin")


if __name__ == "__main__":
    main()
