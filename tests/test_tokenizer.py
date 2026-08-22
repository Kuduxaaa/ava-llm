"""Tokenizer behaviour. Skipped when sentencepiece is not installed."""

import string

import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from ava.tokenizer import AvaTokenizer


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A synthetic corpus with enough distinct words to support a real vocab.

    SentencePiece will not pad a vocabulary out to the requested size: if the
    corpus cannot yield that many pieces, training fails. A handful of repeated
    words is not a corpus.
    """
    letters = string.ascii_lowercase
    words = [f"{a}{b}{c}" for a in letters for b in letters[:8] for c in letters[:6]][:1200]

    path = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    lines = [
        " ".join(words[(index * 7 + offset) % len(words)] for offset in range(14))
        for index in range(600)
    ]
    path.write_text(chr(10).join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, corpus):
    output = tmp_path_factory.mktemp("tokenizer")
    return AvaTokenizer.train(corpus, output, vocab_size=512, character_coverage=1.0)


def test_vocab_too_small_for_byte_fallback_is_a_clear_error(tmp_path, corpus):
    """SentencePiece reports this as an INTERNAL C++ assertion; we should not."""
    with pytest.raises(ValueError, match="too small for byte fallback"):
        AvaTokenizer.train(corpus, tmp_path, vocab_size=200)


def test_training_samples_rather_than_loading_the_whole_corpus(tmp_path, corpus):
    """SentencePiece loads every sentence into RAM unless told not to.

    Fine for a few hundred megabytes, fatal for the tens of gigabytes a real
    pretraining corpus runs to -- and the failure arrives after the download,
    on a rented machine.
    """
    tokenizer = AvaTokenizer.train(
        corpus,
        tmp_path,
        vocab_size=512,
        character_coverage=1.0,
        input_sentence_size=500,
    )
    assert len(tokenizer) == 512
    assert tokenizer.decode(tokenizer.encode("abc bcd")) == "abc bcd"


def test_vocab_too_large_for_the_corpus_is_a_clear_error(tmp_path, corpus):
    with pytest.raises(ValueError, match="larger than this corpus can support"):
        AvaTokenizer.train(corpus, tmp_path, vocab_size=50_000)


def test_special_token_ids_are_pinned(tokenizer):
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id) == (0, 1)
    assert (tokenizer.eos_token_id, tokenizer.unk_token_id) == (2, 3)


def test_encode_and_call_agree_on_special_tokens(tokenizer):
    """The mismatch that silently teaches a model never to emit EOS."""
    text = "abc bcd cde"
    assert tokenizer.encode(text) == tokenizer(text)["input_ids"]
    assert tokenizer.encode(text)[0] == tokenizer.bos_token_id
    assert tokenizer.encode(text)[-1] == tokenizer.eos_token_id


def test_add_special_tokens_false_is_bare(tokenizer):
    ids = tokenizer.encode("abc bcd", add_special_tokens=False)
    assert tokenizer.bos_token_id not in ids
    assert tokenizer.eos_token_id not in ids


def test_roundtrip_preserves_text(tokenizer):
    text = "abc bcd cde def"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_truncation_keeps_the_final_eos(tokenizer):
    """A truncated document that lost its EOS teaches the model not to stop."""
    ids = tokenizer.encode("abc bcd cde def efg fgh ghi", truncation=True, max_length=5)
    assert len(ids) == 5
    assert ids[-1] == tokenizer.eos_token_id


def test_batch_encoding_pads_to_the_longest(tokenizer):
    batch = tokenizer(["abc", "abc bcd cde def efg"], padding=True, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()


def test_left_padding_puts_content_last(tokenizer):
    """Generation needs the real last token to be in the last column."""
    batch = tokenizer(
        ["ab", "abc bcd cde def"], padding=True, padding_side="left", return_tensors="pt"
    )
    assert batch["attention_mask"][0, 0] == 0
    assert batch["attention_mask"][:, -1].all()


def test_pad_to_max_length(tokenizer):
    batch = tokenizer("abc bcd", padding="max_length", max_length=16, return_tensors="pt")
    assert batch["input_ids"].shape == (1, 16)


def test_max_length_padding_requires_max_length(tokenizer):
    with pytest.raises(ValueError, match="max_length"):
        tokenizer("abc", padding="max_length")


def test_ragged_batch_without_padding_is_an_error(tokenizer):
    with pytest.raises(ValueError, match="ragged"):
        tokenizer(["ab", "abc bcd cde def"], return_tensors="pt")


def test_batch_decode_skips_special_tokens(tokenizer):
    batch = tokenizer(["abc bcd", "cde def"], padding=True, return_tensors="pt")
    decoded = tokenizer.batch_decode(batch["input_ids"])
    assert decoded == ["abc bcd", "cde def"]


def test_decode_rejects_a_batch(tokenizer):
    with pytest.raises(ValueError, match="batch_decode"):
        tokenizer.decode([[1, 2], [3, 4]])


def test_save_and_load_roundtrip(tokenizer, tmp_path):
    tokenizer.save_pretrained(tmp_path)
    restored = AvaTokenizer.from_pretrained(tmp_path)

    assert len(restored) == len(tokenizer)
    assert restored.encode("abc bcd") == tokenizer.encode("abc bcd")
    assert restored.padding_side == tokenizer.padding_side


def test_chat_template_marks_the_generation_point(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert rendered.endswith("<|assistant|>\n")


def test_byte_fallback_handles_unseen_characters(tokenizer):
    """Trained on ASCII, asked about something else -- must not lose data."""
    assert tokenizer.decode(tokenizer.encode("abc éè bcd")) == "abc éè bcd"
