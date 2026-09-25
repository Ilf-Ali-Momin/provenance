"""Sweep the dense/lexical weighting against dev.

ARCHITECTURE.md: the weighting "is a config value swept in evaluation, not a
magic constant". This is that sweep.

**Two guards, both about not lying with a number the sweep produced.**

*The holdout is not swept against.* D009 allows the frozen holdout to be
inspected twice in the project. A sweep inspects it once per grid point. The
sweep refuses a holdout claim by split rather than trusting the caller to have
filtered, which is the same enforcement D048 applies to index construction.

*The winning grid point's own dev score is not the system's retrieval
performance.* This is the subtler one and it is the reason `SweepResult` has the
shape it has. Choosing the best of N configurations on dev and then reporting
that configuration's dev score reports the maximum of N noisy estimates, which
is biased upward by roughly the spread between grid points - and with a coarse
grid over a few hundred claims that is not small. The result therefore carries
`selected_weight` and the whole curve, and `EVALUATION.md` numbers come from
running the selected weight against data the sweep never saw. `best_score` is
named `best_dev_score` so that a reader copying it into a report is copying
something that says what it is.

**Measured separately for each side, because the spec asks for that.**
"Measured separately for the local index and live retrieval, and for the hybrid
combination, so the contribution of each is visible rather than assumed." A
weight of 1.0 and 0.0 are the two endpoints, so a grid including them gives all
three from one sweep; `score_sides` runs them explicitly so the report has them
even when the grid does not reach the ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pipeline.retrieval.config import RetrievalSettings
from pipeline.retrieval.hybrid import fuse
from pydantic import BaseModel

from eval.retrieval.score import ClaimRetrieval, score

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from eval.retrieval.judgements import ClaimJudgements

# Coarse on purpose. A finer grid buys precision in a number that is selected on
# dev and therefore not reported, and costs a full pass per point.
DEFAULT_GRID: Final = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
SWEEPABLE_SPLITS: Final = ("dev",)


class GridPoint(BaseModel):
    model_config = {"frozen": True}

    dense_weight: float
    ndcg_at_10: float | None
    recall_at_20: float | None
    coverage: float | None
    failure_rate_pooled: float | None


class SweepResult(BaseModel):
    """The curve, the selection, and a name that says what the score is."""

    model_config = {"frozen": True}

    claims: int
    grid: list[GridPoint]
    selected_weight: float | None
    # Named for what it is. Selecting the best of N on dev and reporting its dev
    # score reports the maximum of N noisy estimates; the reportable number
    # comes from running `selected_weight` against data the sweep never saw.
    best_dev_score: float | None
    objective: str
    local_only: dict[str, object]
    lexical_only: dict[str, object]
    reading: str


def _retrieval_for(
    judged: ClaimJudgements,
    dense: dict[str, list[tuple[str, float]]],
    lexical: dict[str, list[tuple[str, float]]],
    settings: RetrievalSettings,
) -> ClaimRetrieval:
    fused = fuse(dense.get(judged.claim_id, []), lexical.get(judged.claim_id, []), settings)
    return ClaimRetrieval(
        claim_id=judged.claim_id,
        retrieved=tuple(r.doc_id for r in fused),
        # An unjudged pooled candidate is absent from `grades` and scores 0.
        # That is the pooling bias judgements.py names, not a silent default.
        relevance=judged.grades,
        group=judged.group,
    )


def check_splits(judged: Sequence[ClaimJudgements]) -> None:
    """Refuse to sweep against anything but dev.

    A sweep inspects its evaluation set once per grid point, and D009 allows the
    frozen holdout two inspections in the project.
    """
    offending = sorted({j.split for j in judged if j.split not in SWEEPABLE_SPLITS})
    if offending:
        msg = (
            f"claims from {offending} reached the sweep, which may only run on "
            f"{list(SWEEPABLE_SPLITS)}. A sweep inspects its set once per grid point and "
            "D009 allows the frozen holdout two inspections in the whole project."
        )
        raise ValueError(msg)


def sweep(
    judged: Sequence[ClaimJudgements],
    dense: dict[str, list[tuple[str, float]]],
    lexical: dict[str, list[tuple[str, float]]],
    grid: Sequence[float] = DEFAULT_GRID,
    objective: str = "ndcg_at_10",
    base: RetrievalSettings | None = None,
) -> SweepResult:
    """Score every grid point. Returns the curve and the selection, not a verdict.

    `dense` and `lexical` are precomputed rankings per claim, so the sweep costs
    one retrieval pass rather than one per grid point, and so every grid point
    is scored against exactly the same candidates.
    """
    check_splits(judged)
    base = base or RetrievalSettings()

    points: list[GridPoint] = []
    for weight in grid:
        settings = base.model_copy(update={"dense_weight": weight})
        result = score([_retrieval_for(j, dense, lexical, settings) for j in judged])
        points.append(
            GridPoint(
                dense_weight=weight,
                ndcg_at_10=result["ndcg_at_10"],  # type: ignore[arg-type]
                recall_at_20=result["recall_at_20"],  # type: ignore[arg-type]
                coverage=result["coverage"],  # type: ignore[arg-type]
                failure_rate_pooled=result["failure_rate_pooled"],  # type: ignore[arg-type]
            )
        )

    scored = [p for p in points if getattr(p, objective) is not None]
    # Ties break toward the lower weight: at equal measured performance the
    # cheaper side wins, and the dense side is the one that needs a model.
    best = (
        min(scored, key=lambda p: (-float(getattr(p, objective)), p.dense_weight))
        if scored
        else None
    )

    ends = base.model_copy(update={"dense_weight": 1.0})
    lexical_ends = base.model_copy(update={"dense_weight": 0.0})
    return SweepResult(
        claims=len(judged),
        grid=points,
        selected_weight=best.dense_weight if best else None,
        best_dev_score=float(getattr(best, objective)) if best else None,
        objective=objective,
        local_only=score([_retrieval_for(j, dense, lexical, ends) for j in judged]),
        lexical_only=score([_retrieval_for(j, dense, lexical, lexical_ends) for j in judged]),
        reading=(
            "best_dev_score is the maximum of the grid's dev scores and is biased upward "
            "as an estimate of performance: it is the largest of several noisy numbers, "
            "chosen after seeing them. Report it as what the sweep selected on, never as "
            "the system's retrieval performance. The reportable number comes from running "
            "selected_weight against data this sweep did not see. local_only and "
            "lexical_only are the endpoints EVALUATION.md asks to see separately, so the "
            "contribution of each side is visible rather than assumed."
        ),
    )
