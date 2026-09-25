"""Chunking, which has to preserve offsets and never lose text between chunks."""

from __future__ import annotations

from itertools import pairwise

import pytest
from pipeline.retrieval.chunk import chunk_document, count_words

SENTENCE = "Unemployment fell to 4.1% in March and ministers called it encouraging. "


def _document(n: int) -> str:
    return (SENTENCE * n).strip()


def test_offsets_slice_back_to_the_chunk_text() -> None:
    """A chunk shown as evidence must be locatable in its source."""
    text = _document(40)
    for chunk in chunk_document(text):
        assert text[chunk.start : chunk.end] == chunk.text


def test_chunks_end_on_sentence_boundaries() -> None:
    text = _document(40)
    for chunk in chunk_document(text):
        assert chunk.text.endswith(".")


def test_a_decimal_does_not_split_a_chunk() -> None:
    """The bug that produced '1% in March' as a standalone claim."""
    text = _document(30)
    assert all("4.1%" in c.text or "4.1" not in c.text for c in chunk_document(text))
    for chunk in chunk_document(text):
        assert not chunk.text.startswith("1%")


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_document(_document(60))
    assert len(chunks) > 1
    for a, b in pairwise(chunks):
        assert b.start < a.end, "a gap here loses evidence that straddles the cut"


def test_no_text_falls_between_chunks() -> None:
    """Overlap means chunk n+1 starts before chunk n ends; nothing is skipped."""
    text = _document(60)
    chunks = chunk_document(text)
    assert chunks[0].start == 0
    covered = chunks[-1].end
    assert covered == len(text)


def test_chunks_reach_roughly_the_target() -> None:
    chunks = chunk_document(_document(80), target_tokens=300, overlap_tokens=50)
    # Every chunk but the last reaches the target; the last is whatever remains.
    for chunk in chunks[:-1]:
        assert count_words(chunk.text) >= 300


def test_a_sentence_longer_than_the_target_becomes_its_own_chunk() -> None:
    """Rather than an empty chunk or a loop that cannot advance."""
    long_sentence = " ".join(["word"] * 500) + "."
    chunks = chunk_document(long_sentence, target_tokens=300, overlap_tokens=50)
    assert len(chunks) == 1
    assert count_words(chunks[0].text) == 500


def test_a_document_of_long_sentences_still_terminates() -> None:
    """The step-back leaves at least one new sentence, or this never returns."""
    sentence = " ".join(["word"] * 400) + "."
    chunks = chunk_document((sentence + " ") * 5, target_tokens=300, overlap_tokens=290)
    assert 1 < len(chunks) <= 5


def test_text_with_no_terminator_is_one_chunk() -> None:
    chunks = chunk_document("a headline with no full stop")
    assert len(chunks) == 1
    assert chunks[0].text == "a headline with no full stop"


def test_blank_text_produces_nothing() -> None:
    assert chunk_document("   \n  ") == []


def test_overlap_at_or_above_target_is_refused() -> None:
    """It cannot advance, and failing at the call is better than hanging."""
    with pytest.raises(ValueError, match="cannot advance"):
        chunk_document(_document(10), target_tokens=100, overlap_tokens=100)


def test_a_custom_token_counter_is_used() -> None:
    """The sweep passes the real tokenizer; this module never imports one."""
    chunks = chunk_document(_document(20), target_tokens=300, count=lambda t: len(t))
    assert len(chunks) > 1


def test_chunks_are_numbered_in_order() -> None:
    chunks = chunk_document(_document(60))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
