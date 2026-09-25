"""BM25 and the fusion, which is where the two scales get reconciled."""

from __future__ import annotations

from pipeline.retrieval.bm25 import Bm25Index, tokenise
from pipeline.retrieval.config import RetrievalSettings
from pipeline.retrieval.hybrid import fuse, normalise, search_hybrid

DOCS = {
    "a": "Unemployment fell to 4.1% in March, the statistics office said.",
    "b": "The minister said the deficit had halved over the year.",
    "c": "Rates will rise again before June, analysts said.",
    "d": "Ministers said the policy was working and said so again.",
}


def test_numbers_survive_tokenisation_whole() -> None:
    """4.1 is one token. Splitting it loses the vocabulary claims are made of."""
    assert "4.1" in tokenise("Unemployment fell to 4.1% in March")


def test_abbreviations_and_hyphens_stay_together() -> None:
    assert "u.s" in tokenise("the U.S. economy")
    assert "twenty-year" in tokenise("a twenty-year high")


def test_a_rare_term_ranks_its_document_first() -> None:
    index = Bm25Index(DOCS)
    assert index.search("unemployment")[0][0] == "a"


def test_a_term_in_most_documents_never_scores_negative() -> None:
    """The textbook IDF penalises a document for containing a common word."""
    index = Bm25Index(DOCS)
    assert index.idf["said"] > 0
    assert all(score > 0 for _, score in index.search("said"))


def test_documents_sharing_no_term_are_not_returned() -> None:
    """Padding the tail would make the retrieval failure rate unmeasurable."""
    index = Bm25Index(DOCS)
    assert index.search("photosynthesis chlorophyll") == []


def test_search_respects_k() -> None:
    assert len(Bm25Index(DOCS).search("said", k=2)) == 2


def test_an_empty_index_scores_nothing() -> None:
    assert Bm25Index({}).search("anything") == []


def test_normalise_maps_a_ranking_onto_zero_to_one() -> None:
    out = normalise([("a", 10.0), ("b", 5.0), ("c", 0.0)])
    assert out == {"a": 1.0, "b": 0.5, "c": 0.0}


def test_tied_scores_all_normalise_to_one() -> None:
    """A tie is the side expressing no preference, not a division by zero."""
    assert normalise([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}


def test_full_dense_weight_reproduces_the_dense_ranking() -> None:
    """What makes the config value mean what it says."""
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("c", 40.0), ("b", 20.0), ("a", 1.0)]
    out = fuse(dense, lexical, RetrievalSettings(dense_weight=1.0))
    assert [r.doc_id for r in out] == ["a", "b", "c"]


def test_zero_dense_weight_reproduces_the_lexical_ranking() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("c", 40.0), ("b", 20.0), ("a", 1.0)]
    out = fuse(dense, lexical, RetrievalSettings(dense_weight=0.0))
    assert [r.doc_id for r in out] == ["c", "b", "a"]


def test_the_unbounded_side_cannot_swamp_the_bounded_one() -> None:
    """Summing raw cosine and raw BM25 is not a weighting, it is whatever ratio
    the two happened to produce on that query."""
    dense = [("a", 0.99), ("b", 0.10)]
    lexical = [("b", 900.0), ("a", 1.0)]
    balanced = fuse(dense, lexical, RetrievalSettings(dense_weight=0.5))
    assert {r.score for r in balanced} == {0.5}


def test_a_document_only_one_side_found_is_kept() -> None:
    """Dropping it would make the hybrid an intersection, which is the opposite
    of why two sides exist."""
    out = fuse([("a", 1.0)], [("b", 5.0)], RetrievalSettings(dense_weight=0.5))
    assert {r.doc_id for r in out} == {"a", "b"}


def test_the_score_floor_applies_before_normalisation() -> None:
    """Otherwise the failure rate describes how survivors compare, not how the
    retriever did."""
    out = fuse([("a", 5.0), ("b", 0.5)], [], RetrievalSettings(dense_weight=1.0, score_floor=1.0))
    assert [r.doc_id for r in out] == ["a"]


def test_top_k_truncates_the_fused_ranking() -> None:
    dense = [(chr(97 + n), 1.0 - n / 100) for n in range(30)]
    out = fuse(dense, [], RetrievalSettings(dense_weight=1.0, top_k=20))
    assert len(out) == 20


def test_ties_break_by_id_so_sweep_runs_are_comparable() -> None:
    out = fuse([("z", 1.0), ("a", 1.0)], [], RetrievalSettings(dense_weight=1.0))
    assert [r.doc_id for r in out] == ["a", "z"]


def test_each_side_is_asked_for_more_candidates_than_top_k() -> None:
    """Fusion can only rank what it was given."""
    asked: list[int | None] = []

    class Spy:
        def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
            asked.append(k)
            return [("a", 1.0)]

    search_hybrid("claim", Spy(), Spy(), RetrievalSettings(top_k=20, candidates_per_side=100))
    assert asked == [100, 100]
