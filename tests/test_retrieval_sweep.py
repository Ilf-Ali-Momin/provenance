"""The sweep, and the two numbers it refuses to let anyone misreport."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.retrieval.judgements import (
    SCALE,
    ClaimJudgements,
    NotJudgedError,
    load,
    pool_candidates,
    write,
)
from eval.retrieval.sweep import DEFAULT_GRID, check_splits, sweep


def _claim(
    claim_id: str, grades: dict[str, int], split: str = "dev", **over: object
) -> ClaimJudgements:
    fields: dict[str, object] = {
        "claim_id": claim_id,
        "claim_text": "Unemployment fell to 4.1% in March.",
        "split": split,
        "grades": grades,
        "pool": list(grades),
        "reviewed": True,
    }
    fields.update(over)
    return ClaimJudgements(**fields)  # type: ignore[arg-type]


# --- pooling ---------------------------------------------------------------


def test_the_pool_is_the_union_of_every_configuration() -> None:
    """No configuration is judged on documents the others never surfaced."""
    pool = pool_candidates({"dense": [("a", 1.0)], "lexical": [("b", 5.0)]})
    assert set(pool) == {"a", "b"}


def test_the_pool_is_ordered_by_best_rank_across_configurations() -> None:
    """A judge works down a list starting with what everything agreed on."""
    runs = {"dense": [("x", 1.0), ("a", 0.9)], "lexical": [("a", 5.0), ("y", 1.0)]}
    assert pool_candidates(runs)[0] == "a"


def test_pool_depth_is_respected() -> None:
    ranking = [(f"c{n}", 1.0 - n / 100) for n in range(30)]
    assert len(pool_candidates({"dense": ranking}, depth=5)) == 5


def test_the_scale_is_graded_not_binary() -> None:
    """Collapsing 'settles the claim' into 'same topic' makes a retriever that
    ranks chatter first score the same as one that does not."""
    assert set(SCALE) == {0, 1, 2}


# --- the judgement file ----------------------------------------------------


def test_an_ungraded_pool_is_refused_not_scored(tmp_path: Path) -> None:
    path = tmp_path / "judgements.jsonl"
    write([_claim("c1", {}, reviewed=False)], path)
    with pytest.raises(NotJudgedError, match="none marked reviewed"):
        load(path)


def test_a_missing_file_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(NotJudgedError, match=r"run eval\.retrieval\.pool"):
        load(tmp_path / "absent.jsonl")


def test_a_claim_whose_pool_holds_nothing_relevant_is_still_a_data_point(tmp_path: Path) -> None:
    """It is what the retrieval failure rate is about; selecting on non-empty
    grades would drop exactly those."""
    path = tmp_path / "judgements.jsonl"
    write([_claim("c1", {"a": 0, "b": 0})], path)
    assert len(load(path)) == 1


def test_an_unexpected_field_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extra"):
        ClaimJudgements(
            claim_id="c1",
            claim_text="x",
            split="dev",
            relevance={},  # type: ignore[call-arg]
        )


def test_the_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "judgements.jsonl"
    write([_claim("c1", {"a": 2}), _claim("c2", {"b": 1})], path)
    assert [c.claim_id for c in load(path)] == ["c1", "c2"]
    assert json.loads(path.read_text().splitlines()[0])["grades"] == {"a": 2}


# --- the split guard -------------------------------------------------------


def test_the_frozen_holdout_may_not_be_swept_against() -> None:
    """A sweep inspects its set once per grid point; D009 allows two in total."""
    with pytest.raises(ValueError, match="holdout"):
        check_splits([_claim("c1", {"a": 2}, split="holdout")])


def test_train_is_refused_too() -> None:
    with pytest.raises(ValueError, match="may only run on"):
        check_splits([_claim("c1", {"a": 2}, split="train")])


def test_dev_passes() -> None:
    check_splits([_claim("c1", {"a": 2})])


# --- the sweep -------------------------------------------------------------


def _rankings() -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]]:
    # Dense ranks the relevant chunk first, lexical ranks it last.
    dense = {"c1": [("good", 0.9), ("bad", 0.5)]}
    lexical = {"c1": [("bad", 40.0), ("good", 1.0)]}
    return dense, lexical


def test_the_whole_curve_is_returned_not_only_the_winner() -> None:
    dense, lexical = _rankings()
    result = sweep([_claim("c1", {"good": 2, "bad": 0})], dense, lexical)
    assert [p.dense_weight for p in result.grid] == list(DEFAULT_GRID)


def test_the_selected_weight_is_the_cheapest_grid_point_that_scores_best() -> None:
    """Dense ranks the relevant chunk first here, so every weight above the
    crossover ties at the top and the tie-break takes the lowest of them."""
    dense, lexical = _rankings()
    result = sweep([_claim("c1", {"good": 2, "bad": 0})], dense, lexical)
    best = max(p.ndcg_at_10 or 0.0 for p in result.grid)
    tied = [p.dense_weight for p in result.grid if p.ndcg_at_10 == best]
    assert result.best_dev_score == best
    assert result.selected_weight == min(tied)


def test_the_dev_score_is_named_for_what_it_is() -> None:
    """Choosing the best of N on dev and reporting its dev score reports the
    maximum of N noisy estimates."""
    dense, lexical = _rankings()
    result = sweep([_claim("c1", {"good": 2, "bad": 0})], dense, lexical)
    assert result.best_dev_score is not None
    assert "biased upward" in result.reading
    assert not hasattr(result, "retrieval_performance")


def test_both_endpoints_are_reported_separately() -> None:
    """EVALUATION.md: the contribution of each side visible rather than assumed."""
    dense, lexical = _rankings()
    result = sweep([_claim("c1", {"good": 2, "bad": 0})], dense, lexical)
    assert result.local_only["ndcg_at_10"] > result.lexical_only["ndcg_at_10"]


def test_ties_break_toward_the_cheaper_side() -> None:
    """At equal measured performance the side that needs no model wins."""
    flat = {"c1": [("good", 1.0)]}
    result = sweep([_claim("c1", {"good": 2})], flat, flat)
    assert result.selected_weight == 0.0


def test_an_unjudged_pooled_candidate_counts_as_irrelevant() -> None:
    """The pooling bias judgements.py names, not a silent default."""
    dense = {"c1": [("unjudged", 0.9)]}
    result = sweep([_claim("c1", {"judged": 2})], dense, {})
    assert result.grid[0].recall_at_20 == 0.0


def test_a_claim_with_no_relevant_chunk_is_excluded_from_the_scores() -> None:
    dense, lexical = _rankings()
    judged = [_claim("c1", {"good": 2, "bad": 0}), _claim("c2", {"x": 0})]
    result = sweep(judged, dense, lexical)
    assert result.claims == 2
    assert result.grid[0].coverage == 0.5


def test_recall_at_20_can_be_the_objective_instead() -> None:
    dense, lexical = _rankings()
    result = sweep([_claim("c1", {"good": 2, "bad": 0})], dense, lexical, objective="recall_at_20")
    assert result.objective == "recall_at_20"
    assert result.selected_weight is not None


def test_the_pool_spans_every_grid_point_not_just_the_default() -> None:
    """Otherwise a configuration is judged on documents it never surfaced.

    `only_at_w0` is last on the dense side and first on the lexical one, so it
    reaches the pool through the lexical endpoint and through nothing else.
    """
    from eval.retrieval.pool import pool_for

    dense = [("dense_top", 0.9), ("middle", 0.6), ("only_at_w0", 0.1)]
    lexical = [("only_at_w0", 40.0), ("middle", 8.0), ("dense_top", 1.0)]
    pool = pool_for("a claim", dense, lexical)
    assert {"dense_top", "middle", "only_at_w0"} <= set(pool)


def test_depth_bounds_each_configuration_not_the_union() -> None:
    """The union of several top-5 lists is more than five, and that is the
    point: the pool is every configuration's candidates, not one's.

    What is bounded is the tail. A chunk no configuration put in its top
    `depth` is never judged and counts as not relevant, which is the pooling
    bias judgements.py states rather than leaving to be discovered.
    """
    from eval.retrieval.pool import pool_for

    dense = [(f"c{n}", 1.0 - n / 100) for n in range(30)]
    pool = pool_for("a claim", dense, [], depth=5)
    assert len(pool) >= 5
    assert len(pool) < 30, "the tail is bounded, or pooling saves no judging effort"
    assert pool[0] == "c0"


def test_pooling_is_deterministic() -> None:
    """A pool that changed between runs would make judgements unreusable."""
    from eval.retrieval.pool import pool_for

    dense = [("a", 0.9), ("b", 0.8)]
    lexical = [("b", 5.0), ("c", 1.0)]
    assert pool_for("claim", dense, lexical) == pool_for("claim", dense, lexical)
