"""The extraction evaluation runner. No network, no annotations required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.extraction.run_eval import NotAnnotatedError, evaluate, load_annotations

PASSAGE = "Some 2,400 homes were flooded across three counties on Tuesday."


def _row(spans: list[dict] | None = None) -> dict:
    return {"passage_id": "p1", "text": PASSAGE, "spans": spans or [], "notes": ""}


def test_an_unannotated_file_is_refused_not_scored(tmp_path: Path) -> None:
    """Empty spans would score as zero recall, which reads as a result."""
    path = tmp_path / "passages.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    with pytest.raises(NotAnnotatedError, match="has not been annotated"):
        load_annotations(path)


def test_a_missing_file_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(NotAnnotatedError, match=r"eval\.annotation\.sample"):
        load_annotations(tmp_path / "absent.jsonl")


def test_partly_annotated_files_score_only_the_annotated_passages(tmp_path: Path) -> None:
    path = tmp_path / "passages.jsonl"
    path.write_text(
        json.dumps(_row([{"start": 0, "end": 62, "type": "factual_assertion"}]))
        + "\n"
        + json.dumps(_row())
        + "\n"
    )
    assert len(load_annotations(path)) == 1


def test_the_fallback_can_be_scored_through_the_same_path() -> None:
    """D037 defers the parse-based fallback until there is a measured gap."""
    rows = [_row([{"start": 0, "end": 62, "type": "factual_assertion"}])]
    result = evaluate(rows, use_fallback=True)
    assert result["extractor_version"].startswith("fallback")  # type: ignore[union-attr]
    assert result["assertion"]["recall"] == 1.0  # type: ignore[index]


def test_an_extractor_failure_counts_as_finding_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skipping the passage would quietly raise recall on the rest."""
    from eval.extraction import run_eval as mod
    from pipeline.claims.extract import ExtractorUnavailableError

    def unavailable(text: str, settings: object) -> tuple[list, dict]:
        raise ExtractorUnavailableError("HTTP 503")

    monkeypatch.setattr(mod, "extract_claims", unavailable)
    rows = [_row([{"start": 0, "end": 62, "type": "factual_assertion"}])]
    result = evaluate(rows, use_fallback=False)
    assert result["extractor_failures"] == 1
    assert result["assertion"]["recall"] == 0.0  # type: ignore[index]
    assert result["passages"] == 1
