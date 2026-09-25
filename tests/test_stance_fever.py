"""The FEVER loader: label mapping, premise construction, and the caveat."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.stance.fever import LABELS, evidence_text, judgements, pairs
from eval.stance.score import NEUTRAL, REFUTE, SUPPORT


def _corpus(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "fever.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "1",
        "claim": "Unemployment fell in March.",
        "label": "SUPPORTS",
        "verifiable": "VERIFIABLE",
        "evidence": [["Page", "0", "Unemployment fell to 4.1% in March."]],
    }
    row.update(over)
    return row


def test_fever_labels_map_to_the_three_stances() -> None:
    """FEVER's labels are stance toward a specific evidence sentence, which is
    what makes the mapping a rename rather than an interpretation."""
    assert LABELS == {"SUPPORTS": SUPPORT, "REFUTES": REFUTE, "NOT ENOUGH INFO": NEUTRAL}


def test_the_evidence_is_the_premise_and_the_claim_is_the_hypothesis(tmp_path: Path) -> None:
    """NLI is directional and this is the direction the question asks."""
    row = pairs(_corpus(tmp_path, [_row()]))[0]
    assert row["premise"] == "Unemployment fell to 4.1% in March."
    assert row["hypothesis"] == "Unemployment fell in March."


def test_several_evidence_sentences_join_in_the_order_given() -> None:
    """Shuffling a premise changes what it entails."""
    evidence = [["P", "0", "First sentence."], ["P", "1", "Second sentence."]]
    assert evidence_text(evidence) == "First sentence. Second sentence."


def test_malformed_evidence_yields_no_premise() -> None:
    assert evidence_text(None) == ""
    assert evidence_text([["P", "0"]]) == ""


def test_a_row_with_no_premise_is_dropped(tmp_path: Path) -> None:
    """Scoring it would charge the classifier for an empty string."""
    assert pairs(_corpus(tmp_path, [_row(evidence=[])])) == []


def test_a_row_with_an_unknown_label_is_dropped(tmp_path: Path) -> None:
    assert pairs(_corpus(tmp_path, [_row(label="DISPUTED")])) == []


def test_not_verifiable_rows_are_flagged_as_retrieved_evidence(tmp_path: Path) -> None:
    """FEVER's NEI claims have no gold evidence by construction, so the neutral
    class is not the same measurement as the other two."""
    rows = pairs(
        _corpus(
            tmp_path,
            [_row(), _row(label="NOT ENOUGH INFO", verifiable="NOT VERIFIABLE", id="2")],
        )
    )
    assert [r["retrieved_evidence"] for r in rows] == [False, True]


def test_the_limit_is_respected(tmp_path: Path) -> None:
    rows = [_row(id=str(n)) for n in range(10)]
    assert len(pairs(_corpus(tmp_path, rows), limit=3)) == 3


def test_reported_speech_marks_the_attributed_subset(tmp_path: Path) -> None:
    """Only `attributed` is derivable from FEVER text, and as a surface cue."""
    rows = pairs(
        _corpus(
            tmp_path,
            [
                _row(evidence=[["P", "0", "The minister said the deficit had halved."]]),
                _row(id="2", evidence=[["P", "0", "The deficit halved."]]),
            ],
        )
    )
    marked = judgements(rows, ["support", "support"])
    assert marked[0].subsets == ("attributed",)
    assert marked[1].subsets == ()


def test_the_other_stress_subsets_are_not_guessed_at(tmp_path: Path) -> None:
    """They are properties FEVER does not label; guessing would put a number
    against a subset nobody defined."""
    rows = pairs(_corpus(tmp_path, [_row()]))
    subsets = judgements(rows, ["support"])[0].subsets
    assert "partial_overlap" not in subsets
    assert "numeric" not in subsets


def test_predictions_must_line_up_with_rows(tmp_path: Path) -> None:
    rows = pairs(_corpus(tmp_path, [_row(), _row(id="2")]))
    with pytest.raises(ValueError, match="shorter"):
        judgements(rows, ["support"])
