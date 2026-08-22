"""Corpus binarisation: text in, a flat token stream on disk out.

Padding every document to ``max_length`` is the default in tutorial code and it
is why small pretraining runs are so much slower than they should be: at a 512
block size and a median document of 90 tokens, roughly four out of five tokens
the GPU processes are padding. Packing concatenates documents into one
contiguous stream separated by EOS and cuts fixed-size blocks out of it, so
every position in every batch carries signal.

The result is a memory-mapped ``.bin`` file, which also means the dataset no
longer has to fit in RAM.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

HEADER_FILENAME = "meta.json"


def _dtype_for(vocab_size: int) -> np.dtype:
    """Smallest integer type that can hold every id in the vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


@dataclass
class PackedCorpus:
    """Metadata describing a binarised corpus."""

    path: str
    num_tokens: int
    dtype: str
    vocab_size: int
    num_documents: int

    def to_dict(self) -> dict:
        return {
            "path": os.path.basename(self.path),
            "num_tokens": self.num_tokens,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "num_documents": self.num_documents,
        }


def iter_text_file(path: str | os.PathLike, encoding: str = "utf-8") -> Iterator[str]:
    """Yield one document per non-empty line."""
    with open(path, encoding=encoding) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def pack_corpus(
    documents: Iterable[str],
    tokenizer,
    output_path: str | os.PathLike,
    append_eos: bool = True,
    batch_size: int = 1000,
    report_every: int = 100_000,
    verbose: bool = True,
) -> PackedCorpus:
    """Tokenise ``documents`` and stream them into a flat ``.bin`` file.

    Memory use is bounded by ``batch_size``, so this handles corpora far larger
    than RAM.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dtype = _dtype_for(len(tokenizer))
    eos = tokenizer.eos_token_id
    num_tokens = 0
    num_documents = 0
    buffer: list[str] = []

    def flush(handle, buffer: list[str]) -> int:
        if not buffer:
            return 0
        encoded = tokenizer.encode(buffer, add_special_tokens=False)
        flat: list[int] = []
        for ids in encoded:
            flat.extend(ids)
            if append_eos and eos >= 0:
                flat.append(eos)
        array = np.asarray(flat, dtype=dtype)
        handle.write(array.tobytes())
        return array.size

    meta_path = os.path.join(os.path.dirname(output_path) or ".", HEADER_FILENAME)

    def describe() -> PackedCorpus:
        return PackedCorpus(
            path=output_path,
            num_tokens=num_tokens,
            dtype=dtype.name,
            vocab_size=len(tokenizer),
            num_documents=num_documents,
        )

    def write_meta() -> None:
        """Keep the sidecar in step with the .bin, not just at the end.

        Packing a real corpus takes hours and streams from a remote host that
        drops connections. Writing the metadata only on success means an
        interruption leaves a large .bin that nothing can read -- the token
        dtype is not recoverable from the bytes -- and the whole run has to be
        repeated. Rewriting a few hundred bytes periodically makes any
        interruption leave a smaller but perfectly usable dataset.
        """
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(describe().to_dict(), f, indent=2)

    try:
        with open(output_path, "wb") as handle:
            for document in documents:
                buffer.append(document)
                num_documents += 1
                if len(buffer) >= batch_size:
                    num_tokens += flush(handle, buffer)
                    buffer.clear()
                    if num_tokens and num_tokens % report_every < batch_size:
                        handle.flush()
                        write_meta()
                        if verbose:
                            print(f"  {num_documents:,} docs -> {num_tokens:,} tokens")
            num_tokens += flush(handle, buffer)
    finally:
        write_meta()

    corpus = describe()

    if verbose:
        size_mb = os.path.getsize(output_path) / 1e6
        print(
            f"Packed {num_documents:,} documents -> {num_tokens:,} tokens "
            f"({size_mb:.1f} MB, {dtype.name}) at {output_path}"
        )
    return corpus


def load_packed(path: str | os.PathLike) -> tuple[np.memmap, PackedCorpus]:
    """Memory-map a packed corpus, reading its dtype from the sidecar metadata."""
    path = str(path)
    meta_path = os.path.join(os.path.dirname(path) or ".", HEADER_FILENAME)

    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        dtype = np.dtype(meta["dtype"])
    else:
        # Fall back to the smaller type; a mismatch here silently halves or
        # doubles the token count, so say so loudly rather than guessing twice.
        raise FileNotFoundError(
            f"{meta_path} is missing -- the token dtype cannot be inferred from "
            "the .bin alone. Re-run pack_corpus() to regenerate it."
        )

    tokens = np.memmap(path, dtype=dtype, mode="r")
    corpus = PackedCorpus(
        path=path,
        num_tokens=int(tokens.size),
        dtype=dtype.name,
        vocab_size=meta.get("vocab_size", 0),
        num_documents=meta.get("num_documents", 0),
    )
    return tokens, corpus
