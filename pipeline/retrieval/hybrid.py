"""Combine dense and lexical rankings into one.

ARCHITECTURE.md: "Hybrid scoring: dense similarity combined with BM25 lexical
overlap, weighted, because pure dense retrieval loses on proper nouns and
numbers, which is precisely the vocabulary claims are made of. The weighting is
a config value swept in evaluation, not a magic constant."

**The two scores are not on the same scale, and adding them directly is the
usual way this goes wrong.** Cosine similarity is bounded in [-1, 1] and in
practice sits in a narrow band near the top; BM25 is unbounded above and grows
with query length and corpus statistics. A weighted sum of the raw numbers is
therefore not a weighting at all - it is whatever ratio the two happen to
produce on that query, and the config value barely moves the result.

So each side is normalised within the query's own candidate set before fusion.
Min-max over the candidates, which makes the weight mean what it says: at
`dense_weight` 1.0 the ranking is the dense ranking exactly, at 0.0 the lexical
one exactly, and in between the config value is the actual mix.

The cost of min-max, stated rather than hidden: it discards absolute score
magnitude, so a query where every candidate is poor looks the same after
normalisation as one where every candidate is good. That is why the score floor
is applied to the raw scores *before* normalisation, not to the fused score
after - the failure rate has to be a property of how well the retriever did,
not of how the survivors compare to each other.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

from pipeline.retrieval.config import RetrievalSettings


class Scored(NamedTuple):
    doc_id: str
    score: float


@runtime_checkable
class Searcher(Protocol):
    """One retrieval side: local dense, lexical, or live news.

    A Protocol for the reason D025 gave for the robots interface: the caller
    that owns the dependency supplies it, and the evaluation harness can drive
    the fusion with neither a database nor a model loaded. EVALUATION.md
    requires the local index, live retrieval and the hybrid to be measured
    separately, which is only expressible if each is a thing that can be
    passed in on its own.
    """

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]: ...


def normalise(results: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max a ranking into [0, 1] within its own candidate set.

    A single candidate, or several tied, would divide by zero. They score 1.0:
    the side ranked them top and has expressed no preference among them, which
    is what a tie is.
    """
    if not results:
        return {}
    scores = [s for _, s in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {d: 1.0 for d, _ in results}
    return {d: (s - low) / (high - low) for d, s in results}


def fuse(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    settings: RetrievalSettings | None = None,
) -> list[Scored]:
    """Weighted fusion of two rankings, best first.

    A document found by one side and not the other scores 0 on the side that
    missed it rather than being dropped. Dropping it would make the hybrid an
    intersection, which is the opposite of the point: the two sides exist
    because they fail on different queries.
    """
    settings = settings or RetrievalSettings()
    weight = settings.dense_weight

    kept_dense = [(d, s) for d, s in dense if s >= settings.score_floor]
    kept_lexical = [(d, s) for d, s in lexical if s >= settings.score_floor]

    normalised_dense = normalise(kept_dense)
    normalised_lexical = normalise(kept_lexical)

    fused = [
        Scored(
            doc_id,
            weight * normalised_dense.get(doc_id, 0.0)
            + (1 - weight) * normalised_lexical.get(doc_id, 0.0),
        )
        for doc_id in dict.fromkeys([*normalised_dense, *normalised_lexical])
    ]
    # Ties break by id so a sweep's runs are comparable to each other rather
    # than to whichever order the dicts happened to produce.
    return sorted(fused, key=lambda r: (-r.score, r.doc_id))[: settings.top_k]


def search_hybrid(
    query: str,
    dense: Searcher,
    lexical: Searcher,
    settings: RetrievalSettings | None = None,
) -> list[Scored]:
    """Run both sides and fuse. Each side retrieves more than `top_k`."""
    settings = settings or RetrievalSettings()
    n = settings.candidates_per_side
    return fuse(dense.search(query, n), lexical.search(query, n), settings)
