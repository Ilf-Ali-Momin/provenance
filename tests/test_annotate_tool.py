"""The annotation tool stores what a human marked, and rejects what it cannot.

D035 says nothing in this project may supply the gold standard. These tests
cover the half the tool is allowed to do: recording spans faithfully, and
refusing spans the scorer could not use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.extraction.run_eval import NotAnnotatedError, load_annotations
from tools.annotate import load, validate, write

TEXT = "Unemployment fell to 4.1% in March. The policy was a disgrace."


def _row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "passage_id": "doc_a#0",
        "domain": "example.com",
        "stratum": "partisan",
        "split": "train",
        "text": TEXT,
        "spans": [],
        "notes": "",
        "reviewed": False,
    }
    row.update(over)
    return row


def test_a_valid_span_is_stored_with_its_offsets_intact() -> None:
    spans = [{"start": 0, "end": 35, "type": "factual_assertion"}]
    kept = validate(_row(spans=spans), TEXT)
    assert kept == spans
    assert TEXT[kept[0]["start"] : kept[0]["end"]] == "Unemployment fell to 4.1% in March."


def test_a_span_outside_the_passage_is_refused() -> None:
    """A gold span the scorer cannot slice is a defect found at scoring time."""
    with pytest.raises(ValueError, match="outside a passage"):
        validate(_row(spans=[{"start": 0, "end": 9999, "type": "opinion"}]), TEXT)


def test_an_inverted_span_is_refused() -> None:
    with pytest.raises(ValueError, match="outside a passage"):
        validate(_row(spans=[{"start": 20, "end": 5, "type": "opinion"}]), TEXT)


def test_a_type_outside_the_four_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown type"):
        validate(_row(spans=[{"start": 0, "end": 10, "type": "probably_true"}]), TEXT)


def test_overlapping_spans_are_refused() -> None:
    """One-to-one matching has no answer for a gold span covered twice."""
    spans = [
        {"start": 0, "end": 20, "type": "factual_assertion"},
        {"start": 10, "end": 30, "type": "opinion"},
    ]
    with pytest.raises(ValueError, match="overlapping spans"):
        validate(_row(spans=spans), TEXT)


def test_spans_come_back_in_reading_order(tmp_path: Path) -> None:
    spans = [
        {"start": 36, "end": 62, "type": "opinion"},
        {"start": 0, "end": 35, "type": "factual_assertion"},
    ]
    assert [s["start"] for s in validate(_row(spans=spans), TEXT)] == [0, 36]


def test_the_file_survives_a_write(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    rows = [_row(passage_id=f"doc_{n}#0") for n in range(3)]
    write(path, rows)
    assert [r["passage_id"] for r in load(path)] == ["doc_0#0", "doc_1#0", "doc_2#0"]


def test_a_reviewed_passage_with_no_claims_is_scored(tmp_path: Path) -> None:
    """The guideline calls an empty passage a data point, so it must reach the scorer.

    Selecting annotated rows by `spans` dropped exactly these, so an extractor
    that invented claims in empty commentary was never charged for them.
    """
    path = tmp_path / "passages.jsonl"
    rows = [
        _row(passage_id="doc_empty#0", reviewed=True, spans=[]),
        _row(passage_id="doc_unseen#0", reviewed=False, spans=[]),
    ]
    write(path, rows)
    loaded = load_annotations(path)
    assert [r["passage_id"] for r in loaded] == ["doc_empty#0"]


def test_a_file_nobody_has_opened_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    write(path, [_row(), _row(passage_id="doc_b#0")])
    with pytest.raises(NotAnnotatedError, match="none marked reviewed"):
        load_annotations(path)


def test_a_file_written_before_reviewed_existed_is_read_the_old_way(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    row = _row(spans=[{"start": 0, "end": 35, "type": "factual_assertion"}])
    del row["reviewed"]
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert len(load_annotations(path)) == 1
