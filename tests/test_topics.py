"""Topic measurement reporting. No model load, no network."""

from __future__ import annotations

from eval.ood_collection import topics as mod


def _row(stratum: str = "partisan", topic: str = "unknown") -> dict[str, object]:
    return {"stratum": stratum, "topic": topic, "text": "body"}


def test_documents_below_the_floor_are_unclear_not_forced() -> None:
    """A low confidence document must not be pushed into the top label."""
    rows = [_row(), _row()]
    report = mod.report(rows, [("politics", 0.91), (mod.UNCLEAR, 0.31)])
    assert report["distribution"] == {"politics": 1, "unclear": 1}
    assert report["share_named"] == 0.5


def test_agreement_is_measured_only_where_the_path_gave_a_label() -> None:
    rows = [_row(topic="politics"), _row(topic="health"), _row(topic="unknown")]
    report = mod.report(rows, [("politics", 0.9), ("business", 0.8), ("science", 0.9)])
    agreement = report["path_label_agreement"]
    assert isinstance(agreement, dict)
    assert agreement["documents_with_a_path_label"] == 2
    assert agreement["agreed"] == 1
    assert agreement["rate"] == 0.5


def test_agreement_is_none_when_no_path_label_exists() -> None:
    report = mod.report([_row()], [("politics", 0.9)])
    agreement = report["path_label_agreement"]
    assert isinstance(agreement, dict)
    assert agreement["rate"] is None


def test_the_skew_is_reported_as_a_share_of_named_documents() -> None:
    rows = [_row() for _ in range(5)]
    inferred = [("politics", 0.9)] * 3 + [("health", 0.9), (mod.UNCLEAR, 0.2)]
    report = mod.report(rows, inferred)
    assert report["largest_topic_share_of_named"] == 0.75


def test_cells_are_reported_per_stratum() -> None:
    rows = [_row("wire"), _row("wire"), _row("partisan")]
    report = mod.report(rows, [("politics", 0.9), ("health", 0.9), ("politics", 0.9)])
    assert report["cells_stratum_topic"] == {
        "partisan": {"politics": 1},
        "wire": {"health": 1, "politics": 1},
    }
