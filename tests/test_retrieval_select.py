"""Page selection, and the split it must never read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.annotation.sample import ELIGIBLE_SPLITS
from pipeline.retrieval.select import seed_terms, wikinews_titles, wikipedia_titles


def _corpus(tmp_path: Path, **splits: list[str]) -> Path:
    for split, texts in splits.items():
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in texts) + "\n", encoding="utf-8"
        )
    return tmp_path


def test_the_frozen_holdout_may_not_seed_the_index(tmp_path: Path) -> None:
    """An index built around the holdout's subject matter retrieves better on it
    for reasons unrelated to the retriever. That is the leakage M0 documented."""
    with pytest.raises(ValueError, match="frozen holdout may not seed"):
        seed_terms(tmp_path, splits=("train", "holdout"))


def test_the_holdout_file_is_not_read_even_when_present(tmp_path: Path) -> None:
    corpus = _corpus(
        tmp_path,
        train=["The Supreme Court ruled on Tuesday."],
        holdout=["The Strait of Hormuz remained open."],
    )
    seeds = seed_terms(corpus)
    assert any("Supreme Court" in s for s in seeds)
    assert not any("Hormuz" in s for s in seeds)


def test_the_eligible_splits_come_from_one_place() -> None:
    """Two copies of "which splits may be touched" drift, and only one gets
    updated (D009, D035)."""
    assert ELIGIBLE_SPLITS == ("train", "dev")
    assert "holdout" not in ELIGIBLE_SPLITS


def test_entities_are_ranked_by_frequency(tmp_path: Path) -> None:
    corpus = _corpus(
        tmp_path,
        train=[
            "The Supreme Court met. The Supreme Court ruled. The Supreme Court adjourned.",
            "The Postal Service replied.",
        ],
    )
    seeds = seed_terms(corpus)
    assert seeds[0].endswith("Supreme Court")


def test_a_sentence_initial_capital_does_not_become_an_entity(tmp_path: Path) -> None:
    """ "The Supreme Court" is the court; "The" is not part of its name."""
    corpus = _corpus(tmp_path, train=["The Supreme Court ruled."] * 3)
    assert all(not s.startswith("The ") for s in seeds_of(corpus))


def seeds_of(corpus: Path) -> list[str]:
    return seed_terms(corpus)


def test_single_words_are_not_seeds(tmp_path: Path) -> None:
    """A one word query returns everything and costs a fetch per page."""
    corpus = _corpus(tmp_path, train=["Washington reacted. Washington replied."])
    assert all(" " in s for s in seed_terms(corpus))


def test_a_missing_split_file_is_not_an_error(tmp_path: Path) -> None:
    assert seed_terms(_corpus(tmp_path, train=["The Supreme Court ruled."])) != []


def test_wikinews_selection_takes_article_namespace_only() -> None:
    """Portals and templates carry navigation text, not prose."""
    urls: list[str] = []

    def spy(url: str) -> dict[str, object]:
        urls.append(url)
        return {"query": {"categorymembers": [{"title": "Voting begins"}]}}

    assert wikinews_titles(spy, limit=1) == ["Voting begins"]
    assert "cmnamespace=0" in urls[0]
    assert "Category%3APublished" in urls[0]


def test_wikinews_selection_follows_continuation() -> None:
    pages = [
        {"query": {"categorymembers": [{"title": "A"}]}, "continue": {"cmcontinue": "next"}},
        {"query": {"categorymembers": [{"title": "B"}]}},
    ]
    assert wikinews_titles(lambda url: pages.pop(0), limit=10) == ["A", "B"]


def test_wikinews_selection_stops_at_the_limit() -> None:
    def endless(url: str) -> dict[str, object]:
        return {
            "query": {"categorymembers": [{"title": f"T{n}"} for n in range(500)]},
            "continue": {"cmcontinue": "more"},
        }

    assert len(wikinews_titles(endless, limit=3)) == 3


def test_wikipedia_selection_deduplicates_across_seeds() -> None:
    """Two seeds often hit the same article; indexing it twice doubles its
    weight in retrieval for no reason."""

    def search(url: str) -> dict[str, object]:
        return {"query": {"search": [{"title": "Shared page"}, {"title": "Other"}]}}

    assert wikipedia_titles(["a b", "c d"], search) == ["Shared page", "Other"]


def test_wikipedia_selection_preserves_seed_order() -> None:
    """A truncated build should be the most frequent seeds' pages."""
    results = {"first seed": "Page A", "second seed": "Page B"}

    def search(url: str) -> dict[str, object]:
        key = "first seed" if "first" in url else "second seed"
        return {"query": {"search": [{"title": results[key]}]}}

    assert wikipedia_titles(["first seed", "second seed"], search) == ["Page A", "Page B"]


def test_a_malformed_search_response_yields_nothing() -> None:
    assert wikipedia_titles(["a b"], lambda url: {}) == []
