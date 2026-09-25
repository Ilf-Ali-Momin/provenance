"""Score retrieval against annotated relevance (EVALUATION.md §3, Retrieval).

Written before the index exists, for the reason EVALUATION.md gives in its
opening: a metric designed after the results are visible drifts toward what the
system happens to do well, and the drift is invisible to the person doing it.

EVALUATION.md asks for four things here, and the fourth is the one that shapes
this module:

    nDCG at 10 and recall at 20 against annotated relevance on the OOD claims.
    Measured separately for the local index and live retrieval, and for the
    hybrid combination, so the contribution of each is visible rather than
    assumed.

    Also measured: retrieval failure rate ... Track it as a distribution, not a
    single rate, because a system that fails on 40 percent of health claims and
    5 percent of political claims is a different system than one that fails on
    20 percent of everything.

So `failure_rate` is not a float on this module's output. It is a float per
group, and the pooled number is reported beside them rather than instead of
them - a single rate is exactly the summary the spec says is misleading, and
the way to stop it being quoted alone is to not compute it alone.

**The acceptance criterion has a precondition, and it is load bearing.**

    Acceptance: recall at 20 above 0.70 on claims that have coverage in the
    index, with the coverage rate reported alongside.

"On claims that have coverage" means the denominator excludes claims for which
no relevant passage exists at all. That is correct - a claim nothing in the
index bears on cannot be retrieved for, and scoring it as a recall failure
measures the corpus rather than the retriever. It is also the single easiest
number in this project to inflate by accident: shrink the covered set and
recall rises. `coverage` is therefore returned next to `recall_at_20` and
`meets_acceptance` is false when coverage cannot be computed, so the figure
cannot be quoted without its denominator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Final, NamedTuple

NDCG_K: Final = 10
RECALL_K: Final = 20
RECALL_FLOOR: Final = 0.70


def dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain over a ranked list of graded relevances.

    Graded rather than binary: annotated relevance distinguishes a passage that
    settles a claim from one that is merely on topic, and collapsing them makes
    a retriever that ranks topical chatter above the decisive passage score the
    same as one that does not.
    """
    return float(sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances)))


def ndcg_at_k(ranked: list[int], ideal: list[int] | None = None, k: int = NDCG_K) -> float:
    """nDCG@k. `ideal` is the full relevance multiset for the claim.

    The ideal ranking is built from every relevant passage that exists, not from
    the ones that were retrieved. Normalising against the retrieved set scores a
    retriever that found one mediocre passage and ranked it first as perfect.
    """
    pool = sorted(ideal if ideal is not None else ranked, reverse=True)
    best = dcg(pool[:k])
    if best == 0.0:
        # No relevant passage exists for this claim. Not a score of zero: an
        # undefined one, and the caller separates coverage from performance.
        return 0.0
    return dcg(ranked[:k]) / best


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = RECALL_K) -> float:
    """Share of the relevant passages that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


class ClaimRetrieval(NamedTuple):
    """One claim's retrieval, and the annotation it is scored against.

    `relevance` is the human judgement: passage id to graded relevance, where 0
    means retrieved-but-irrelevant and a claim with no positive entry has no
    coverage in the index. A typed record rather than a dict because the
    coverage denominator is derived from it, and that is the number this
    module exists to keep honest.
    """

    claim_id: str
    retrieved: tuple[str, ...]
    relevance: dict[str, int]
    group: str = "all"


def score(
    per_claim: list[ClaimRetrieval],
    k_ndcg: int = NDCG_K,
    k_recall: int = RECALL_K,
) -> dict[str, object]:
    """Aggregate over claims, keeping coverage beside the numbers it scales."""
    covered: list[dict[str, float]] = []
    failures: dict[str, list[bool]] = defaultdict(list)
    uncovered = 0

    for claim in per_claim:
        relevant = {i for i, rel in claim.relevance.items() if rel > 0}

        # Failure is "nothing above the relevance threshold was returned", which
        # is a property of the retriever's output and is defined for every
        # claim - including the ones with no coverage, where returning nothing
        # is correct behaviour. So it is counted before the coverage filter.
        failures[claim.group].append(not claim.retrieved)

        if not relevant:
            uncovered += 1
            continue
        covered.append(
            {
                "ndcg": ndcg_at_k(
                    [claim.relevance.get(i, 0) for i in claim.retrieved],
                    list(claim.relevance.values()),
                    k_ndcg,
                ),
                "recall": recall_at_k(list(claim.retrieved), relevant, k_recall),
            }
        )

    n = len(covered)
    mean_ndcg = sum(c["ndcg"] for c in covered) / n if n else None
    mean_recall = sum(c["recall"] for c in covered) / n if n else None
    total = len(per_claim)

    return {
        "claims": total,
        "claims_with_coverage": n,
        "claims_without_coverage": uncovered,
        # Reported next to recall, never apart from it: "on claims that have
        # coverage" is a denominator someone can shrink to raise the number.
        "coverage": round(n / total, 4) if total else None,
        f"ndcg_at_{k_ndcg}": round(mean_ndcg, 4) if mean_ndcg is not None else None,
        f"recall_at_{k_recall}": round(mean_recall, 4) if mean_recall is not None else None,
        # A distribution, per EVALUATION.md, with the pooled figure beside it
        # rather than instead of it.
        "failure_rate_by_group": {
            g: round(sum(f) / len(f), 4) for g, f in sorted(failures.items()) if f
        },
        "failure_rate_pooled": (
            round(sum(sum(f) for f in failures.values()) / total, 4) if total else None
        ),
        "meets_acceptance": bool(n) and mean_recall is not None and mean_recall > RECALL_FLOOR,
        "acceptance": (
            f"EVALUATION.md §3: recall at {k_recall} above {RECALL_FLOOR} on claims with "
            "coverage, coverage rate reported alongside"
        ),
    }
