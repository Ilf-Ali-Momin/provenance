"""Stance metrics, pinned before a classifier exists to be scored by them."""

from __future__ import annotations

from eval.stance.score import Judgement, compare, macro_f1, per_class, score


def _j(gold: str, predicted: str, *subsets: str) -> Judgement:
    return Judgement(gold, predicted, subsets)


def test_a_model_that_never_predicts_refute_shows_a_row_of_zeros() -> None:
    """The failure mode EVALUATION.md names. An absent row would read as an oversight."""
    rows = per_class([_j("refute", "neutral"), _j("support", "support")])
    assert "refute" in rows
    assert rows["refute"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_every_class_appears_even_when_absent_from_the_data() -> None:
    rows = per_class([_j("support", "support")])
    assert set(rows) == {"support", "refute", "neutral"}


def test_macro_f1_is_unweighted_so_the_rare_class_cannot_disappear() -> None:
    """Nine correct support calls do not compensate for missing every refute."""
    judgements = [_j("support", "support") for _ in range(9)] + [_j("refute", "neutral")]
    assert macro_f1(judgements) < 0.5


def test_a_perfect_classifier_scores_one() -> None:
    judgements = [_j("support", "support"), _j("refute", "refute"), _j("neutral", "neutral")]
    assert macro_f1(judgements) == 1.0


def test_the_four_stress_subsets_each_get_their_own_number() -> None:
    result = score(
        [
            _j("support", "support", "numeric"),
            _j("refute", "neutral", "attributed"),
            _j("support", "support", "partial_overlap"),
            _j("neutral", "neutral", "temporal_mismatch"),
        ]
    )
    subsets = result["subsets"]
    assert set(subsets) == {"attributed", "partial_overlap", "temporal_mismatch", "numeric"}
    assert subsets["attributed"]["pairs"] == 1
    assert subsets["numeric"]["macro_f1"] is not None


def test_a_subset_with_no_pairs_reports_none_rather_than_zero() -> None:
    """Zero would read as a measured failure; None reads as not measured."""
    result = score([_j("support", "support")])
    assert result["subsets"]["attributed"] == {"pairs": 0, "macro_f1": None, "per_class": None}


def test_a_pair_can_belong_to_several_subsets() -> None:
    """A numeric claim reported as someone's statement is both."""
    result = score([_j("support", "refute", "attributed", "numeric")])
    assert result["subsets"]["attributed"]["pairs"] == 1
    assert result["subsets"]["numeric"]["pairs"] == 1


def test_acceptance_ignores_the_subsets_including_the_failing_one() -> None:
    """A gate a subset could fail is pressure to drop the subset."""
    good = [_j("support", "support"), _j("refute", "refute"), _j("neutral", "neutral")] * 5
    bad_attributed = [_j("refute", "neutral", "attributed") for _ in range(5)]
    result = score(good + bad_attributed, "ood")
    assert result["subsets"]["attributed"]["macro_f1"] == 0.0
    assert result["meets_acceptance"] is True


def test_acceptance_needs_macro_f1_strictly_above_the_floor() -> None:
    assert score([])["meets_acceptance"] is False


def test_the_confusion_matrix_records_the_direction_of_the_error() -> None:
    result = score([_j("refute", "neutral"), _j("refute", "neutral"), _j("support", "support")])
    assert result["confusion"]["refute->neutral"] == 2
    assert result["confusion"]["support->support"] == 1


def test_the_three_slices_come_back_together_with_the_gap_computed() -> None:
    """The gap is the headline result; a caller should not have to subtract."""
    fever = score([_j("support", "support")] * 10, "fever_dev")
    multifc = score([_j("support", "support")] * 5 + [_j("refute", "neutral")] * 5, "multifc")
    ood = score([_j("support", "neutral")] * 10, "ood")
    out = compare(fever, multifc, ood)
    assert out["fever_dev"]["macro_f1"] == fever["macro_f1"]
    assert out["fever_minus_ood"] == round(fever["macro_f1"] - ood["macro_f1"], 4)
    assert out["fever_minus_ood"] > 0
