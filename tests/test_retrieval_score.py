"""Retrieval metrics, pinned before an index exists to rank anything."""

from __future__ import annotations

from eval.retrieval.score import ClaimRetrieval, dcg, ndcg_at_k, recall_at_k, score


def _claim(
    retrieved: tuple[str, ...], relevance: dict[str, int], group: str = "all"
) -> ClaimRetrieval:
    return ClaimRetrieval("c1", retrieved, relevance, group)


def test_a_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k([3, 2, 1], [3, 2, 1]) == 1.0


def test_reversing_a_ranking_costs_ndcg() -> None:
    assert ndcg_at_k([1, 2, 3], [3, 2, 1]) < 1.0


def test_the_ideal_is_every_relevant_passage_not_the_retrieved_ones() -> None:
    """Otherwise finding one mediocre passage and ranking it first is perfect."""
    found_one_weak = ndcg_at_k([1], [3, 3, 1])
    assert found_one_weak < 1.0
    assert ndcg_at_k([1], [1]) == 1.0


def test_graded_relevance_separates_decisive_from_merely_topical() -> None:
    decisive_first = ndcg_at_k([3, 1], [3, 1])
    topical_first = ndcg_at_k([1, 3], [3, 1])
    assert decisive_first > topical_first


def test_dcg_discounts_by_rank() -> None:
    assert dcg([1, 0]) > dcg([0, 1])


def test_recall_counts_relevant_passages_found_in_the_window() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=20) == 0.5


def test_recall_ignores_anything_below_the_cutoff() -> None:
    assert recall_at_k(["x"] * 20 + ["a"], {"a"}, k=20) == 0.0


def test_a_claim_with_no_coverage_is_excluded_from_recall_not_scored_zero() -> None:
    """Scoring it measures the corpus, not the retriever."""
    result = score(
        [
            _claim(("a",), {"a": 3}),
            _claim((), {}),
        ]
    )
    assert result["claims"] == 2
    assert result["claims_with_coverage"] == 1
    assert result["claims_without_coverage"] == 1
    assert result["recall_at_20"] == 1.0


def test_coverage_is_reported_next_to_recall() -> None:
    """The denominator someone could shrink to raise the number."""
    result = score([_claim(("a",), {"a": 3}), _claim((), {}), _claim((), {})])
    assert result["coverage"] == round(1 / 3, 4)
    assert result["recall_at_20"] is not None


def test_acceptance_is_false_when_nothing_has_coverage() -> None:
    result = score([_claim((), {}), _claim((), {})])
    assert result["meets_acceptance"] is False
    assert result["recall_at_20"] is None


def test_acceptance_needs_recall_strictly_above_the_floor() -> None:
    at_floor = score([_claim(("a", "b", "c", "d", "e", "f", "g"), dict.fromkeys("abcdefghij", 1))])
    assert at_floor["recall_at_20"] == 0.7
    assert at_floor["meets_acceptance"] is False


def test_the_failure_rate_is_a_distribution_not_one_number() -> None:
    """EVALUATION.md: 40 percent on health and 5 on politics is a different system."""
    result = score(
        [
            _claim((), {"x": 1}, group="health"),
            _claim((), {"x": 1}, group="health"),
            _claim(("a",), {"a": 3}, group="politics"),
            _claim(("a",), {"a": 3}, group="politics"),
        ]
    )
    assert result["failure_rate_by_group"] == {"health": 1.0, "politics": 0.0}
    assert result["failure_rate_pooled"] == 0.5


def test_an_uncovered_claim_still_counts_toward_the_failure_rate() -> None:
    """Returning nothing is correct there, and the rate describes output not skill."""
    result = score([_claim((), {}), _claim(("a",), {"a": 1})])
    assert result["failure_rate_pooled"] == 0.5
