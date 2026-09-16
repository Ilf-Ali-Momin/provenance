"""The extraction scorer. Pure arithmetic, no model, no annotations needed."""

from __future__ import annotations

import pytest
from eval.extraction.score import Span, iou, match_spans, score

A = "factual_assertion"
O = "opinion"  # noqa: E741 - a type name, not an ambiguous variable
ATTR = "attribution"


def test_identical_spans_have_iou_one() -> None:
    assert iou(Span(0, 10, A), Span(0, 10, A)) == 1.0


def test_disjoint_spans_have_iou_zero() -> None:
    assert iou(Span(0, 10, A), Span(20, 30, A)) == 0.0


def test_a_paragraph_wide_prediction_does_not_match_a_short_gold_span() -> None:
    """Under 'half the gold span' this would match, and precision would be unearnable."""
    matches, missed, spurious = match_spans([Span(10, 30, A)], [Span(0, 400, A)])
    assert matches == []
    assert len(missed) == 1
    assert len(spurious) == 1


def test_a_one_word_prediction_does_not_match_a_long_gold_span() -> None:
    """Under 'half the predicted span' this would match, and recall would be unearnable."""
    matches, _, _ = match_spans([Span(0, 200, A)], [Span(10, 18, A)])
    assert matches == []


def test_a_mostly_overlapping_span_matches() -> None:
    matches, missed, spurious = match_spans([Span(0, 100, A)], [Span(5, 100, A)])
    assert len(matches) == 1
    assert (missed, spurious) == ([], [])


def test_matching_is_one_to_one() -> None:
    """Otherwise one prediction satisfies several gold spans and inflates recall."""
    gold = [Span(0, 100, A), Span(0, 98, A)]
    matches, missed, _ = match_spans(gold, [Span(0, 100, A)])
    assert len(matches) == 1
    assert len(missed) == 1


def test_the_better_overlap_wins_the_match() -> None:
    gold = [Span(0, 100, A)]
    matches, _, spurious = match_spans(gold, [Span(0, 60, A), Span(0, 99, A)])
    assert matches[0].predicted == Span(0, 99, A)
    assert spurious == [Span(0, 60, A)]


def test_type_is_ignored_when_matching_so_a_mislabel_is_not_counted_twice() -> None:
    matches, missed, spurious = match_spans([Span(0, 100, A)], [Span(0, 100, O)])
    assert len(matches) == 1
    assert (missed, spurious) == ([], [])


def test_a_mislabelled_assertion_costs_both_precision_and_recall() -> None:
    result = score([([Span(0, 100, A)], [Span(0, 100, O)])])
    assert result["assertion"]["recall"] == 0.0  # type: ignore[index]
    assert result["confusion"] == {f"{A}->{O}": 1}


def test_a_perfect_run_meets_acceptance() -> None:
    passages = [([Span(0, 100, A)], [Span(0, 100, A)]) for _ in range(10)]
    result = score(passages)
    assert result["assertion"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["meets_acceptance"] is True


def test_acceptance_needs_both_thresholds() -> None:
    """recall > 0.80 AND precision > 0.70; one alone is not enough."""
    # Nine found of ten, but ten spurious assertions alongside.
    passages = [([Span(0, 100, A)], [Span(0, 100, A), Span(200, 300, A)]) for _ in range(9)]
    passages.append(([Span(0, 100, A)], []))
    result = score(passages)
    assert result["assertion"]["recall"] == 0.9  # type: ignore[index]
    assert result["assertion"]["precision"] == 0.5  # type: ignore[index]
    assert result["meets_acceptance"] is False


def test_a_passage_with_no_gold_and_no_prediction_is_not_an_error() -> None:
    """Commentary with no checkable claim is a data point, not a failure."""
    result = score([([], [])])
    assert result["assertion"]["precision"] == 0.0  # type: ignore[index]
    assert result["passages"] == 1


def test_a_spurious_claim_where_none_exists_costs_precision() -> None:
    result = score([([], [Span(0, 50, A)])])
    assert result["assertion"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


@pytest.mark.parametrize("threshold", [0.5, 0.75])
def test_the_threshold_is_a_parameter_not_a_literal(threshold: float) -> None:
    result = score([([Span(0, 100, A)], [Span(0, 70, A)])], threshold=threshold)
    assert result["iou_threshold"] == threshold
