"""Reading a corpus straight from the Hub, without landing it on disk first.

Writing the text out and packing it afterwards needs room for both. For English
that is an inconvenience; for a script where a token costs eleven bytes it is
often the thing that stops the job. Georgian FineWeb-2 is 28.6 GB of text and
5.2 GB of tokens, and a hosted notebook will not hold the pair.

Streaming the same source twice -- once to sample a tokenizer, once to pack --
costs bandwidth instead of disk, which is the cheaper of the two.

The quality filters live here rather than in a script so that both paths apply
exactly the same ones. Every rule is about *structure*: run-length, symbol
density, replacement characters. A stopword list would only work for the
language it was written for.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator


def is_low_quality(text: str, min_length: int = 200, max_symbol_ratio: float = 0.3) -> bool:
    """Cheap, language-neutral rejection tests."""
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


def clean(text: str | None) -> str:
    """One document per line, so the packer and SentencePiece agree on records."""
    return " ".join((text or "").split())


def iter_hub_dataset(
    dataset: str,
    config: str | None = None,
    split: str = "train",
    text_column: str = "text",
    min_length: int = 200,
    max_symbol_ratio: float = 0.3,
    max_docs: int | None = None,
    every: int | None = None,
    label: str = "streaming",
) -> Iterator[str]:
    """Yield cleaned documents from a Hub dataset, streaming.

    ``every`` prints progress after that many kept documents; ``None`` is silent.
    """
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)

    kept = skipped = 0
    for record in stream:
        text = clean(record.get(text_column))
        if is_low_quality(text, min_length, max_symbol_ratio):
            skipped += 1
            continue

        yield text
        kept += 1
        if every and kept % every == 0:
            print(f"  {label}: {kept:,} kept / {skipped:,} skipped", flush=True)
        if max_docs and kept >= max_docs:
            break


def write_sample(
    documents: Iterator[str], path, max_docs: int, encoding: str = "utf-8"
) -> int:
    """Write the first ``max_docs`` documents to ``path``, for tokenizer training.

    A tokenizer does not need the whole corpus -- SentencePiece subsamples its
    input anyway -- so this is the only part that has to touch the disk.
    """
    import os

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    written = 0
    with open(path, "w", encoding=encoding) as handle:
        for text in documents:
            handle.write(text + "\n")
            written += 1
            if written >= max_docs:
                break
    return written
