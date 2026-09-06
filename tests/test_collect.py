"""Collector selection logic. No network; the fetcher is stubbed.

These assert the rules that decide what enters the corpus. Getting them wrong
does not crash anything, it produces a corpus with a silent defect, which is
the failure mode this whole milestone exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from eval.ood_collection.collect import (
    SNAPSHOT_DATE,
    Document,
    Rejection,
    collect_domain,
    document_id,
    summarise,
)
from pipeline.net.config import FetchSettings
from pipeline.net.feeds import FeedEntry
from pipeline.net.fetch import FetchResult
from pipeline.text.config import ExtractSettings

ARTICLE_BODY = " ".join(
    f"Sentence number {i} carries enough text to clear the extraction floor." for i in range(14)
)
ARTICLE_HTML = f"<html><body><article><h1>Head</h1><p>{ARTICLE_BODY}</p></article></body></html>"

PUBLISHER = {
    "domain": "example.com",
    "stratum": "partisan",
    "lean": "lean_right",
    "low_quality": False,
}


class StubRobots:
    """Allows everything, waits for nothing."""

    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow

    def allows(self, url: str) -> bool:
        return self.allow

    def wait_for(self, url: str) -> None:
        return None


def _entries(*specs: tuple[str, datetime | None]) -> list[FeedEntry]:
    return [FeedEntry(url=u, title="t", published=d) for u, d in specs]


def _install(monkeypatch: pytest.MonkeyPatch, entries: list[FeedEntry]) -> list[str]:
    """Stub feed discovery and fetching; return the list of URLs fetched."""
    from eval.ood_collection import collect as mod

    fetched: list[str] = []

    def fake_find_feed(
        domain: str, robots: object, settings: FetchSettings
    ) -> tuple[str, list[FeedEntry]]:
        return ("https://example.com/rss", entries)

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        fetched.append(url)
        return FetchResult(
            final_url=url, status_code=200, content_type="text/html", text=ARTICLE_HTML
        )

    monkeypatch.setattr(mod, "find_feed", fake_find_feed)
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    return fetched


def _run(
    monkeypatch: pytest.MonkeyPatch, entries: list[FeedEntry], limit: int = 10
) -> tuple[list[Document], Rejection, list[str]]:
    fetched = _install(monkeypatch, entries)
    rej = Rejection()
    docs = collect_domain(
        PUBLISHER,
        limit=limit,
        robots=StubRobots(),  # type: ignore[arg-type]
        fetch_settings=FetchSettings(),
        extract_settings=ExtractSettings(),
        rejections=rej,
    )
    return docs, rej, fetched


def test_articles_before_the_snapshot_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot boundary is the whole point of an OOD suite."""
    docs, rej, fetched = _run(
        monkeypatch,
        _entries(
            ("https://example.com/old", datetime(2025, 12, 31, tzinfo=UTC)),
            ("https://example.com/new", datetime(2026, 2, 1, tzinfo=UTC)),
        ),
    )
    assert [d.url for d in docs] == ["https://example.com/new"]
    assert rej.as_dict()["before_snapshot"] == 1
    assert "https://example.com/old" not in fetched, "must not fetch what it will discard"


def test_undated_entries_are_rejected_not_assumed_recent(monkeypatch: pytest.MonkeyPatch) -> None:
    docs, rej, _ = _run(monkeypatch, _entries(("https://example.com/x", None)))
    assert docs == []
    assert rej.as_dict()["undated"] == 1


def test_offsite_urls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A syndicated URL would put the wrong domain in a grouped split."""
    docs, rej, _ = _run(
        monkeypatch,
        _entries(("https://syndicated.example.org/a", datetime(2026, 3, 1, tzinfo=UTC))),
    )
    assert docs == []
    assert rej.as_dict()["offsite_url"] == 1


def test_www_prefix_does_not_count_as_offsite(monkeypatch: pytest.MonkeyPatch) -> None:
    docs, _, _ = _run(
        monkeypatch, _entries(("https://www.example.com/a", datetime(2026, 3, 1, tzinfo=UTC)))
    )
    assert len(docs) == 1


def test_limit_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _entries(
        *[(f"https://example.com/{i}", datetime(2026, 3, 1, tzinfo=UTC)) for i in range(10)]
    )
    docs, _, fetched = _run(monkeypatch, entries, limit=3)
    assert len(docs) == 3
    assert len(fetched) == 3, "must stop fetching once the limit is reached"


def test_publisher_provenance_travels_onto_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    docs, _, _ = _run(
        monkeypatch, _entries(("https://example.com/a", datetime(2026, 3, 1, tzinfo=UTC)))
    )
    doc = docs[0]
    assert doc.stratum == "partisan"
    assert doc.lean == "lean_right"
    assert doc.low_quality is False
    assert doc.text_sha256 and doc.doc_id.startswith("doc_")


def test_document_id_distinguishes_syndicated_copies() -> None:
    """Same text at two publishers must stay two documents."""
    text = "identical wire copy"
    assert document_id("https://a.com/x", text) != document_id("https://b.com/x", text)


def test_document_id_is_stable() -> None:
    assert document_id("https://a.com/x", "t") == document_id("https://a.com/x", "t")


def test_summary_reports_rejections_and_balance() -> None:
    docs = [
        Document(
            doc_id=f"doc_{i}",
            domain="a.com" if i < 2 else "b.com",
            stratum="partisan",
            lean="lean_left" if i < 2 else "right",
            low_quality=i == 3,
            url=f"https://x/{i}",
            title=None,
            published="2026-03-01T00:00:00+00:00",
            text="t",
            text_sha256="s",
            markers_removed={"dateline": 1} if i == 0 else {},
            had_invisible=i == 1,
            collected_at="2026-03-01T00:00:00+00:00",
        )
        for i in range(4)
    ]
    rej = Rejection()
    rej.record("before_snapshot")
    rej.record("before_snapshot")
    out = summarise(docs, rej)
    assert out["documents"] == 4
    assert out["domains"] == 2
    assert out["partisan_lean_balance"] == {"left_of_centre": 2, "right_of_centre": 2}
    assert out["rejected"] == {"before_snapshot": 2}
    assert out["publisher_markers_removed"] == {"dateline": 1}
    assert out["documents_with_invisible_chars"] == 1
    assert out["snapshot_date"] == SNAPSHOT_DATE.date().isoformat()
