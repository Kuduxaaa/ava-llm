"""Datasets and corpus packing, exercised with a stand-in tokenizer."""

import zlib

import numpy as np
import pytest
import torch

from ava.data import (
    ChatDataset,
    PackedDataset,
    PaddingCollator,
    collate_packed,
    load_packed,
    pack_corpus,
)


class WordTokenizer:
    """Whitespace tokenizer with the same surface as :class:`AvaTokenizer`."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 500

    def encode(self, text, add_special_tokens=True, **_):
        if not isinstance(text, str):
            return [self.encode(item, add_special_tokens) for item in text]
        # crc32 rather than hash(): str hashing is salted per process.
        ids = [(zlib.crc32(word.encode()) % 400) + 100 for word in text.split()]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids


@pytest.fixture
def tokenizer() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture
def packed_path(tmp_path, tokenizer):
    documents = [f"document number {i} with some words in it" for i in range(60)]
    pack_corpus(documents, tokenizer, tmp_path / "corpus.bin", verbose=False)
    return tmp_path / "corpus.bin"


# --- packing ---


def test_packing_writes_metadata_and_tokens(packed_path):
    tokens, corpus = load_packed(packed_path)
    assert corpus.num_documents == 60
    assert corpus.num_tokens == tokens.size > 0
    assert tokens.dtype == np.uint16


def test_packing_separates_documents_with_eos(tmp_path, tokenizer):
    pack_corpus(["alpha beta", "gamma"], tokenizer, tmp_path / "c.bin", verbose=False)
    tokens, _ = load_packed(tmp_path / "c.bin")
    assert (np.asarray(tokens) == tokenizer.eos_token_id).sum() == 2


def test_dtype_widens_for_large_vocabularies(tmp_path, tokenizer):
    class BigVocab(WordTokenizer):
        def __len__(self) -> int:
            return 100_000

    pack_corpus(["a b c"], BigVocab(), tmp_path / "big.bin", verbose=False)
    tokens, _ = load_packed(tmp_path / "big.bin")
    assert tokens.dtype == np.uint32


def test_an_interrupted_pack_still_leaves_a_usable_dataset(tmp_path, tokenizer):
    """Packing streams for hours from a host that drops connections.

    Writing the sidecar only on success leaves a large .bin that nothing can
    read -- the dtype is not recoverable from the bytes -- and the whole run has
    to be repeated.
    """

    def documents():
        for index in range(10_000):
            yield f"document number {index} with some words in it"
            if index == 3_000:
                raise ConnectionError("remote host went away")

    with pytest.raises(ConnectionError):
        pack_corpus(
            documents(), tokenizer, tmp_path / "c.bin", verbose=False, report_every=1000
        )

    tokens, corpus = load_packed(tmp_path / "c.bin")
    assert corpus.num_tokens > 0
    assert tokens.size >= corpus.num_tokens
    assert PackedDataset(tmp_path / "c.bin", block_size=64)[0]["input_ids"].shape == (64,)


def test_missing_metadata_is_an_explicit_error(tmp_path, tokenizer):
    pack_corpus(["a b c"], tokenizer, tmp_path / "c.bin", verbose=False)
    (tmp_path / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="dtype cannot be inferred"):
        load_packed(tmp_path / "c.bin")


# --- source filtering ---


def test_quality_filters_are_language_neutral():
    """Structural rules only: they must not favour the script they were written in."""
    from ava.data import is_low_quality

    georgian = "დღეს კარგი ამბავი მაქვს " * 20
    english = "the quick brown fox jumps over the lazy dog " * 20

    assert not is_low_quality(georgian)
    assert not is_low_quality(english)


def test_quality_filters_reject_what_they_should():
    from ava.data import is_low_quality

    assert is_low_quality("too short")
    assert is_low_quality("a" * 300 + "�")  # mojibake
    assert is_low_quality("=" * 300)  # a run of separators
    assert is_low_quality("!@#$%^&*() " * 40)  # mostly symbols


def test_min_length_counts_characters_not_bytes():
    """Georgian is three bytes per character; a byte threshold would keep junk."""
    from ava.data import is_low_quality

    text = "დღე" * 30  # 90 characters, 270 bytes
    assert is_low_quality(text, min_length=200)
    assert not is_low_quality(text, min_length=50)


def test_write_sample_stops_at_the_limit(tmp_path):
    from ava.data import write_sample

    written = write_sample((f"document {i}" for i in range(1000)), tmp_path / "s.txt", 40)
    assert written == 40
    assert len((tmp_path / "s.txt").read_text(encoding="utf-8").splitlines()) == 40


# --- packed dataset ---


def test_every_block_is_full_length(packed_path):
    dataset = PackedDataset(packed_path, block_size=16)
    assert len(dataset) > 0
    for index in range(min(5, len(dataset))):
        assert dataset[index]["input_ids"].shape == (16,)


def test_packed_blocks_contain_no_padding(packed_path):
    """The whole point of packing: no wasted positions."""
    dataset = PackedDataset(packed_path, block_size=16)
    batch = collate_packed([dataset[i] for i in range(4)])
    assert "attention_mask" not in batch
    assert batch["input_ids"].shape == (4, 16)


def test_labels_mirror_inputs_for_pretraining(packed_path):
    dataset = PackedDataset(packed_path, block_size=16)
    item = dataset[0]
    assert torch.equal(item["input_ids"], item["labels"])


def test_blocks_are_contiguous_slices_of_the_stream(packed_path):
    tokens, _ = load_packed(packed_path)
    dataset = PackedDataset(packed_path, block_size=8)
    expected = torch.from_numpy(np.asarray(tokens[8:16], dtype=np.int64))
    assert torch.equal(dataset[1]["input_ids"], expected)


def test_split_is_disjoint(packed_path):
    dataset = PackedDataset(packed_path, block_size=8)
    train, val = dataset.split(val_fraction=0.2)
    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1


def test_block_size_larger_than_corpus_is_rejected(packed_path):
    with pytest.raises(ValueError, match="block_size"):
        PackedDataset(packed_path, block_size=10**6)


# --- chat dataset ---


def conversation(turns=2):
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})
    return messages


def test_loss_is_masked_outside_assistant_turns(tokenizer):
    dataset = ChatDataset([conversation()], tokenizer, max_length=128)
    item = dataset[0]

    supervised = item["labels"] != -100
    assert supervised.any()
    assert not supervised.all(), "user turns must not be supervised"
    # Where it is supervised, the target is the input token itself.
    assert torch.equal(item["labels"][supervised], item["input_ids"][supervised])


def test_all_assistant_turns_are_supervised_by_default(tokenizer):
    both = ChatDataset([conversation(turns=2)], tokenizer)[0]
    last_only = ChatDataset(
        [conversation(turns=2)], tokenizer, train_on_last_turn_only=True
    )[0]
    assert (both["labels"] != -100).sum() > (last_only["labels"] != -100).sum()


def test_conversation_starts_with_bos_and_ends_with_eos(tokenizer):
    item = ChatDataset([conversation()], tokenizer, max_length=128)[0]
    assert item["input_ids"][0].item() == tokenizer.bos_token_id
    assert item["input_ids"][-1].item() == tokenizer.eos_token_id
    # EOS is supervised, otherwise the model never learns to stop.
    assert item["labels"][-1].item() == tokenizer.eos_token_id


def test_conversations_without_an_assistant_turn_are_dropped(tokenizer):
    dataset = ChatDataset(
        [[{"role": "user", "content": "hello"}], conversation()], tokenizer
    )
    assert len(dataset) == 1


def test_truncation_respects_max_length(tokenizer):
    dataset = ChatDataset([conversation(turns=20)], tokenizer, max_length=16)
    assert dataset[0]["input_ids"].shape[0] <= 16


# --- collator ---


def test_collator_pads_to_the_batch_maximum(tokenizer):
    collator = PaddingCollator(pad_token_id=0, pad_to_multiple_of=1)
    batch = collator(
        [
            {"input_ids": torch.arange(3), "labels": torch.arange(3)},
            {"input_ids": torch.arange(7), "labels": torch.arange(7)},
        ]
    )
    assert batch["input_ids"].shape == (2, 7)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert batch["labels"][0, 3:].tolist() == [-100] * 4


def test_collator_rounds_up_to_a_multiple():
    """Tensor-core kernels want a sequence length divisible by 8."""
    collator = PaddingCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = collator([{"input_ids": torch.arange(5), "labels": torch.arange(5)}])
    assert batch["input_ids"].shape[1] == 8


def test_collator_padding_is_excluded_from_the_loss():
    collator = PaddingCollator(pad_token_id=0)
    batch = collator([{"input_ids": torch.arange(4), "labels": torch.arange(4)}])
    padded = batch["attention_mask"] == 0
    assert (batch["labels"][padded] == -100).all()
