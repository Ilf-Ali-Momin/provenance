"""Score claim extraction against the hand annotated set (EVALUATION.md §3).

Written before the annotations exist, deliberately. EVALUATION.md's opening
principle is that evaluation is written before the model it evaluates, "because
when the metric is designed after the results are visible, the metric drifts
toward what the model happens to do well, and the drift is invisible even to the
person doing it". Fixing the matching rule now, with no numbers to look at, is
that principle applied to this metric.

**One ambiguity in the spec, resolved here and recorded as D036.**
EVALUATION.md says a match is counted "when predicted and gold spans overlap by
more than half". Half of what is not stated, and the readings differ:

* half the gold span - a predicted span covering the whole paragraph matches
  every gold span inside it, so precision becomes unearnable.
* half the predicted span - a one word prediction inside a long gold span
  matches, so recall becomes unearnable.
* half of their union, the intersection-over-union above 0.5 used in detection
  literature. Symmetric, and neither degenerate case passes.

The third is used. It is the strictest of the three and the only symmetric one,
and choosing the strictest reading of an ambiguous acceptance criterion is the
direction that cannot be accused of having been chosen for the result.

Matching is one to one and greedy by overlap: without that, one predicted span
could satisfy several gold spans at once and inflate recall.
"""

from __future__ import annotations

from collections import Counter
from typing import Final, NamedTuple

from pipeline.claims.schema import ClaimType

IOU_THRESHOLD: Final = 0.5
ASSERTION: Final = ClaimType.FACTUAL_ASSERTION.value


class Span(NamedTuple):
    start: int
    end: int
    type: str


class Match(NamedTuple):
    gold: Span
    predicted: Span
    iou: float


def iou(a: Span, b: Span) -> float:
    """Intersection over union of two character ranges."""
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return overlap / union if union else 0.0


def match_spans(
    gold: list[Span], predicted: list[Span], threshold: float = IOU_THRESHOLD
) -> tuple[list[Match], list[Span], list[Span]]:
    """Greedy one-to-one matching. Returns (matches, missed gold, spurious pred).

    Type is ignored when matching: a span found but mislabelled is a located
    claim with the wrong tag, and the confusion matrix is where that shows up.
    Folding type into the match would report it twice, as a miss and as a
    confusion.
    """
    candidates = sorted(
        (
            (iou(g, p), gi, pi)
            for gi, g in enumerate(gold)
            for pi, p in enumerate(predicted)
            if iou(g, p) > threshold
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for score, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matches.append(Match(gold[gi], predicted[pi], round(score, 4)))
    return (
        matches,
        [g for i, g in enumerate(gold) if i not in used_gold],
        [p for i, p in enumerate(predicted) if i not in used_pred],
    )


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score(
    per_passage: list[tuple[list[Span], list[Span]]], threshold: float = IOU_THRESHOLD
) -> dict[str, object]:
    """Assertion precision and recall, plus the four way confusion matrix."""
    assertion_tp = assertion_fp = assertion_fn = 0
    located_tp = located_fp = located_fn = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for gold, predicted in per_passage:
        matches, missed, spurious = match_spans(gold, predicted, threshold)
        located_tp += len(matches)
        located_fn += len(missed)
        located_fp += len(spurious)
        for match in matches:
            confusion[(match.gold.type, match.predicted.type)] += 1
            if match.gold.type == ASSERTION and match.predicted.type == ASSERTION:
                assertion_tp += 1
            elif match.predicted.type == ASSERTION:
                assertion_fp += 1
            elif match.gold.type == ASSERTION:
                assertion_fn += 1
        # A gold assertion nobody located is a miss; a predicted assertion
        # matching nothing is spurious. EVALUATION.md weights recall higher
        # because a missed claim means no evidence surfaces at all.
        assertion_fn += sum(1 for g in missed if g.type == ASSERTION)
        assertion_fp += sum(1 for p in spurious if p.type == ASSERTION)

    assertion = _prf(assertion_tp, assertion_fp, assertion_fn)
    return {
        "passages": len(per_passage),
        "iou_threshold": threshold,
        "assertion": assertion,
        "span_location_ignoring_type": _prf(located_tp, located_fp, located_fn),
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
        "meets_acceptance": assertion["recall"] > 0.80 and assertion["precision"] > 0.70,
        "acceptance": "EVALUATION.md §3: assertion recall > 0.80 with precision > 0.70",
    }
