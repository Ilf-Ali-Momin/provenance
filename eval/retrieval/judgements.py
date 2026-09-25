"""Relevance judgements for retrieval evaluation, and how they are gathered.

EVALUATION.md asks for "nDCG at 10 and recall at 20 against annotated relevance
on the OOD claims". Annotated relevance means a person decided whether a
retrieved passage bears on a claim. Nothing in this project may supply that, for
the reason D035 gives about the extraction annotations: a system scored against
labels it generated measures itself.

**Pooling, because judging every chunk against every claim is not possible.**
The index holds tens of thousands of chunks and the corpus has hundreds of
claims. The standard answer, and the one used here, is to run several retrieval
configurations, union their top-k, and judge only that pool. Two properties make
it honest:

* every configuration under comparison contributes to the pool, so none is
  judged on documents the others never had a chance to surface;
* the pool is built before any judgement is made, so it cannot be widened after
  seeing which configuration is winning.

**The bias this leaves, stated rather than hidden.** A chunk no pooled
configuration retrieved is never judged, and an unjudged chunk counts as not
relevant. So recall is recall *within the pool*, and a configuration invented
later - one not in the pool - is scored unfairly against these judgements.
`Judgements.configurations` records which configurations built the pool so that
a later reader can tell whether a number is comparable.

**Graded, not binary.** EVALUATION.md's failure rate distinguishes "nothing
above the relevance threshold" from "something", and the extraction guideline's
lesson applies here too: a passage that settles a claim and a passage that
merely shares its topic are different, and collapsing them makes a retriever
that ranks topical chatter above the decisive passage score the same as one that
does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

JUDGEMENTS: Final = Path("eval/retrieval/judgements.jsonl")

# The scale a judge applies. Three points, because a finer scale asks for a
# precision human judgement does not have and nDCG's exponential gain already
# separates these strongly.
SCALE: Final = {
    0: "does not bear on the claim",
    1: "related: same topic or entity, does not address the claim itself",
    2: "bears on the claim: supports, refutes, or directly qualifies it",
}


class ClaimJudgements(BaseModel):
    """One claim, its pooled candidates, and a human's grade for each."""

    model_config = {"frozen": True, "extra": "forbid"}

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    # The split the claim came from. Recorded so the sweep can refuse a holdout
    # claim rather than trusting the caller to have filtered (D009, D048).
    split: str
    # The group the failure rate is broken down by. EVALUATION.md: a system that
    # fails on 40 percent of health claims and 5 percent of political claims is
    # a different system from one that fails on 20 percent of everything.
    group: str = "all"
    # chunk_id -> grade. A pooled candidate with no entry is unjudged.
    grades: dict[str, int] = Field(default_factory=dict)
    pool: list[str] = Field(default_factory=list)
    notes: str = ""
    reviewed: bool = False


class NotJudgedError(RuntimeError):
    """Candidates were pooled but nobody graded them."""


def load(path: Path = JUDGEMENTS) -> list[ClaimJudgements]:
    """Read the judgement file, refusing to score an ungraded one.

    Same refusal as the extraction harness, for the same reason: a file of empty
    grades scores as zero recall rather than as an absence, and someone reading
    the table later sees a number where there is none.

    `reviewed` rather than "has any grade", because a claim whose pool contains
    nothing relevant is a real data point - it is what the retrieval failure
    rate is about - and selecting on non-empty grades would drop exactly those.
    """
    if not path.exists():
        msg = f"no judgement file at {path}; run eval.retrieval.pool first"
        raise NotJudgedError(msg)
    rows = [
        ClaimJudgements.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judged = [row for row in rows if row.reviewed]
    if not judged:
        msg = (
            f"{path} holds {len(rows)} pooled claims and none marked reviewed. Relevance "
            "has not been judged yet, and an unjudged pool scores as zero recall rather "
            "than as an absence. Nothing in this project may supply these labels (D035)."
        )
        raise NotJudgedError(msg)
    return judged


def write(rows: Iterable[ClaimJudgements], path: Path = JUDGEMENTS) -> int:
    """One JSON object per line, ready for a human to grade."""
    body = [row.model_dump_json() for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return len(body)


def pool_candidates(
    runs: dict[str, list[tuple[str, float]]],
    depth: int = 20,
) -> list[str]:
    """Union the top `depth` of each configuration's ranking.

    Order is by best rank achieved across configurations, so a judge works down
    a list that starts with what every configuration agreed on. That ordering
    affects nothing in the scoring; it makes the judging less tedious, and a
    tired judge is a measurement problem.
    """
    best: dict[str, int] = {}
    for ranking in runs.values():
        for rank, (chunk_id, _score) in enumerate(ranking[:depth]):
            if rank < best.get(chunk_id, depth + 1):
                best[chunk_id] = rank
    return sorted(best, key=lambda chunk_id: (best[chunk_id], chunk_id))
