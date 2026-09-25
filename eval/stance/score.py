"""Score stance classification (EVALUATION.md §3, Stance classification).

Written before the classifier exists, per EVALUATION.md's opening principle.

Three requirements from that section shape this module, and each of them exists
to stop a specific way of reporting a good number for a useless model.

**Per class breakdown is mandatory.**

    Aggregate F1 hides the failure mode that matters, which is that refutation
    is rarer and harder than support and that neutral absorbs everything
    ambiguous. A model that never predicts refute can post a respectable macro
    F1 while being useless for the one output users most need.

So `per_class` is always present, and a class the model never predicted appears
with precision 0.0 rather than being absent from the table. An absent row reads
as an oversight; a zero reads as a result.

**The three numbers are always reported together.**

    Macro F1 ... on FEVER dev, then on MultiFC, then on the OOD slice ... The
    three numbers are always reported together, and the gap between the first
    and the third is the actual headline result of this component.

`compare` takes the three and returns them with the gap computed, because the
gap is the finding and a caller who has to subtract two numbers themselves will
eventually quote the higher one alone.

**The stress subsets are separate numbers, and one of them is allowed to fail.**

    Acceptance: macro F1 above 0.60 on the OOD slice, with the attributed
    claims subset reported separately and no requirement that it pass, because
    the honest result there is likely poor and hiding it would be worse than
    reporting it.

Acceptance is therefore computed on the OOD slice overall and never on a
subset. `subsets` carries the four named ones, each with its own per class
breakdown, and `meets_acceptance` deliberately ignores them: a gate that a
subset could fail would create pressure to quietly drop the subset.
"""

from __future__ import annotations

from collections import Counter
from typing import Final, NamedTuple

# The classifier's three outputs, in ARCHITECTURE.md's terms: "output is
# entailment, contradiction, or neutral". Named here as stance rather than as
# NLI labels, because the card speaks about evidence, not about inference.
SUPPORT: Final = "support"
REFUTE: Final = "refute"
NEUTRAL: Final = "neutral"
CLASSES: Final = (SUPPORT, REFUTE, NEUTRAL)

# EVALUATION.md names these four and requires each as its own number.
STRESS_SUBSETS: Final = ("attributed", "partial_overlap", "temporal_mismatch", "numeric")
MACRO_F1_FLOOR: Final = 0.60


class Judgement(NamedTuple):
    """One (claim, passage) pair: what a human said, and what the model said.

    `subsets` is a tuple because the four stress cases are not mutually
    exclusive - a numeric claim reported as someone's statement is both
    attributed and numeric, and forcing a single label would make one of those
    subsets quietly incomplete.
    """

    gold: str
    predicted: str
    subsets: tuple[str, ...] = ()


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def per_class(judgements: list[Judgement]) -> dict[str, dict[str, float]]:
    """Precision, recall and F1 for each of the three classes.

    Every class appears even when it was never predicted or never occurs. A
    model that never predicts refute is the failure mode the spec calls out by
    name, and it has to be visible as a row of zeros rather than as a gap.
    """
    out: dict[str, dict[str, float]] = {}
    for label in CLASSES:
        tp = sum(1 for j in judgements if j.gold == label and j.predicted == label)
        fp = sum(1 for j in judgements if j.gold != label and j.predicted == label)
        fn = sum(1 for j in judgements if j.gold == label and j.predicted != label)
        out[label] = {**_prf(tp, fp, fn), "support": sum(1 for j in judgements if j.gold == label)}
    return out


def macro_f1(judgements: list[Judgement]) -> float:
    """Unweighted mean of the three per class F1 scores.

    Unweighted is the point. Refutation is the rarest class and the one users
    most need; weighting by support would let the model's performance on it
    disappear into the majority class.
    """
    rows = per_class(judgements)
    return round(sum(rows[c]["f1"] for c in CLASSES) / len(CLASSES), 4)


def score(judgements: list[Judgement], slice_name: str = "unnamed") -> dict[str, object]:
    """One dataset's numbers: macro F1, per class, confusion, and each subset."""
    subsets: dict[str, object] = {}
    for name in STRESS_SUBSETS:
        rows = [j for j in judgements if name in j.subsets]
        subsets[name] = (
            {"pairs": len(rows), "macro_f1": macro_f1(rows), "per_class": per_class(rows)}
            if rows
            else {"pairs": 0, "macro_f1": None, "per_class": None}
        )

    confusion: Counter[tuple[str, str]] = Counter((j.gold, j.predicted) for j in judgements)
    overall = macro_f1(judgements)
    return {
        "slice": slice_name,
        "pairs": len(judgements),
        "macro_f1": overall,
        "per_class": per_class(judgements),
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
        # The four named subsets, each its own number. `meets_acceptance` does
        # not read them: a gate a subset could fail is pressure to drop the
        # subset, and EVALUATION.md says the attributed result is expected to be
        # poor and must be reported anyway.
        "subsets": subsets,
        "meets_acceptance": bool(judgements) and overall > MACRO_F1_FLOOR,
        "acceptance": (
            f"EVALUATION.md §3: macro F1 above {MACRO_F1_FLOOR} on the OOD slice; "
            "the attributed subset is reported separately and is not required to pass"
        ),
    }


def compare(
    fever: dict[str, object], multifc: dict[str, object], ood: dict[str, object]
) -> dict[str, object]:
    """The three numbers together, with the gap that is the actual result.

    EVALUATION.md: "the gap between the first and the third is the actual
    headline result of this component". Computed here so that it is present in
    the output rather than left to a reader who has both numbers in front of
    them and an interest in the larger one.
    """
    a, c = fever.get("macro_f1"), ood.get("macro_f1")
    gap = round(float(a) - float(c), 4) if isinstance(a, float) and isinstance(c, float) else None
    return {
        "fever_dev": fever,
        "multifc": multifc,
        "ood": ood,
        "fever_minus_ood": gap,
        "reading": (
            "The three are reported together by requirement. `fever_minus_ood` is the "
            "distribution shift this component exists to measure: a high FEVER number "
            "with a large gap describes a classifier that works on curated NLI pairs "
            "and not on retrieved news text, which is the case the system actually runs."
        ),
    }
