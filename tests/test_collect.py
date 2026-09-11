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

    def __init__(self, *, allow: bool = True, sitemaps: list[str] | None = None) -> None:
        self.allow = allow
        self.sitemaps = sitemaps or []

    def sitemaps_for(self, url: str) -> list[str]:
        return self.sitemaps

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


# --------------------------------------------------------- refusals and status


def _statuses(monkeypatch: pytest.MonkeyPatch, by_url: dict[str, int]) -> list[str]:
    from eval.ood_collection import collect as mod

    fetched: list[str] = []

    def fake_find_feed(
        domain: str, robots: object, settings: FetchSettings
    ) -> tuple[str, list[FeedEntry]]:
        entries = [
            FeedEntry(url=u, title="t", published=datetime(2026, 3, 1, tzinfo=UTC)) for u in by_url
        ]
        return ("https://example.com/rss", entries)

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        fetched.append(url)
        return FetchResult(
            final_url=url, status_code=by_url[url], content_type="text/html", text=ARTICLE_HTML
        )

    monkeypatch.setattr(mod, "find_feed", fake_find_feed)
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    return fetched


def _collect(rej: Rejection) -> list[Document]:
    return collect_domain(
        PUBLISHER,
        limit=10,
        robots=StubRobots(),  # type: ignore[arg-type]
        fetch_settings=FetchSettings(),
        extract_settings=ExtractSettings(),
        rejections=rej,
    )


def test_refusal_mid_run_stops_asking_that_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 403 is a decision by the publisher, not a missing page to retry past."""
    fetched = _statuses(
        monkeypatch,
        {
            "https://example.com/1": 200,
            "https://example.com/2": 403,
            "https://example.com/3": 200,
        },
    )
    rej = Rejection()
    docs = _collect(rej)
    assert len(docs) == 1
    assert rej.as_dict()["host_refused"] == 1
    assert "https://example.com/3" not in fetched, "must not keep asking after a refusal"


def test_missing_page_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _statuses(monkeypatch, {"https://example.com/1": 404, "https://example.com/2": 200})
    rej = Rejection()
    docs = _collect(rej)
    assert len(docs) == 1
    assert rej.as_dict()["http_404"] == 1


def _feed_host(monkeypatch: pytest.MonkeyPatch, pages: dict[str, tuple[int, str]]) -> list[str]:
    from eval.ood_collection import collect as mod

    asked: list[str] = []

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        asked.append(url)
        status, body = pages.get(url, (404, "<!DOCTYPE html><html>not found</html>"))
        return FetchResult(final_url=url, status_code=status, content_type="text/html", text=body)

    monkeypatch.setattr(mod, "fetch", fake_fetch)
    return asked


def test_homepage_refusal_ends_discovery_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ood_collection.collect import find_feed

    asked = _feed_host(monkeypatch, {"https://example.com/": (403, "Forbidden")})
    result = find_feed("example.com", StubRobots(), FetchSettings())  # type: ignore[arg-type]
    assert result == "host_refused"
    assert asked == ["https://example.com/"], "one refusal is enough"


def test_feed_mislabelled_as_html_is_still_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ood_collection.collect import find_feed

    rss = (
        '<?xml version="1.0"?><rss><channel><item><link>https://example.com/a</link>'
        "<pubDate>Mon, 02 Mar 2026 10:00:00 +0000</pubDate></item></channel></rss>"
    )
    _feed_host(
        monkeypatch,
        {"https://example.com/": (200, "<html></html>"), "https://example.com/rss": (200, rss)},
    )
    result = find_feed("example.com", StubRobots(), FetchSettings())  # type: ignore[arg-type]
    assert isinstance(result, tuple)
    assert result[0] == "https://example.com/rss"


def test_html_pages_are_not_reported_as_hostile_feeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from eval.ood_collection.collect import find_feed

    _feed_host(monkeypatch, {"https://example.com/": (200, "<html></html>")})
    result = find_feed("example.com", StubRobots(), FetchSettings())  # type: ignore[arg-type]
    assert result == "no_feed"
    assert "declared a DTD" not in caplog.text


def test_robots_refusal_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ood_collection.collect import find_feed

    asked = _feed_host(monkeypatch, {})
    result = find_feed("example.com", StubRobots(allow=False), FetchSettings())  # type: ignore[arg-type]
    assert result == "robots_disallowed"
    assert asked == []


# ------------------------------------------- hosts, outlet names and sitemaps


def test_articles_on_the_advertised_feed_host_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dailymail.co.uk advertises a feed whose articles all live on dailymail.com."""
    from eval.ood_collection import collect as mod

    when = datetime(2026, 3, 1, tzinfo=UTC)
    entries = [
        FeedEntry(url="https://www.example.net/a", title="t", published=when),
        FeedEntry(url="https://unrelated.org/b", title="t", published=when),
    ]

    def fake_find_feed(
        domain: str, robots: object, settings: FetchSettings
    ) -> tuple[str, list[FeedEntry]]:
        return ("https://www.example.net/rss", entries)

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        return FetchResult(
            final_url=url, status_code=200, content_type="text/html", text=ARTICLE_HTML
        )

    monkeypatch.setattr(mod, "find_feed", fake_find_feed)
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    rej = Rejection()
    docs = _collect(rej)
    assert [d.url for d in docs] == ["https://www.example.net/a"]
    assert docs[0].domain == "example.com", "grouping stays on the publisher's domain"
    assert rej.as_dict()["offsite_url"] == 1


def test_outlet_names_reach_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entries(("https://example.com/a", datetime(2026, 3, 1, tzinfo=UTC))))
    docs = collect_domain(
        PUBLISHER,
        limit=5,
        robots=StubRobots(),  # type: ignore[arg-type]
        fetch_settings=FetchSettings(),
        extract_settings=ExtractSettings(),
        rejections=Rejection(),
        outlet_names=["Sentence number"],
    )
    assert "Sentence number" not in docs[0].text
    assert docs[0].markers_removed["outlet_name"] == 14


SITEMAP = (
    '<?xml version="1.0"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
    'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
    "<url><loc>https://example.com/politics/edited</loc><lastmod>2026-08-01</lastmod></url>"
    "<url><loc>https://example.com/politics/new</loc><news:news>"
    "<news:publication_date>2026-03-02T09:00:00Z</news:publication_date></news:news></url>"
    "</urlset>"
)


def test_sitemap_is_used_when_no_feed_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ood_collection.collect import find_feed

    _feed_host(
        monkeypatch,
        {
            "https://example.com/": (200, "<html></html>"),
            "https://example.com/news-sitemap.xml": (200, SITEMAP),
        },
    )
    robots = StubRobots(sitemaps=["https://example.com/news-sitemap.xml"])
    result = find_feed("example.com", robots, FetchSettings())  # type: ignore[arg-type]
    assert isinstance(result, tuple)
    url, entries = result
    assert url == "https://example.com/news-sitemap.xml"
    assert entries[0].url == "https://example.com/politics/new", "newest dated entry first"
    assert entries[0].published == datetime(2026, 3, 2, 9, 0, tzinfo=UTC)


def test_lastmod_does_not_count_as_a_publication_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """An old article edited this year must not pass the snapshot filter."""
    from eval.ood_collection.collect import find_feed

    _feed_host(
        monkeypatch,
        {
            "https://example.com/": (200, "<html></html>"),
            "https://example.com/news-sitemap.xml": (200, SITEMAP),
        },
    )
    robots = StubRobots(sitemaps=["https://example.com/news-sitemap.xml"])
    result = find_feed("example.com", robots, FetchSettings())  # type: ignore[arg-type]
    assert isinstance(result, tuple)
    edited = next(e for e in result[1] if e.url.endswith("/edited"))
    assert edited.published is None


# ------------------------------------------------------------------ language

SPANISH_BODY = " ".join(
    "El consejo dijo que votaría el presupuesto en marzo y los funcionarios no se "
    "sorprendieron por la decisión tomada durante la sesión extraordinaria."
    for _ in range(6)
)


def _language_run(
    monkeypatch: pytest.MonkeyPatch, entry: FeedEntry, html: str
) -> tuple[list[Document], Rejection, list[str]]:
    from eval.ood_collection import collect as mod

    fetched: list[str] = []

    def fake_find_feed(
        domain: str, robots: object, settings: FetchSettings
    ) -> tuple[str, list[FeedEntry]]:
        return ("https://example.com/news-sitemap.xml", [entry])

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        fetched.append(url)
        return FetchResult(final_url=url, status_code=200, content_type="text/html", text=html)

    monkeypatch.setattr(mod, "find_feed", fake_find_feed)
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    rej = Rejection()
    return _collect(rej), rej, fetched


def test_declared_non_english_entry_is_skipped_before_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BBC Mundo reached the pilot through a news sitemap that declared es."""
    entry = FeedEntry(
        url="https://example.com/mundo/a",
        title=None,
        published=datetime(2026, 3, 1, tzinfo=UTC),
        language="es",
    )
    docs, rej, fetched = _language_run(monkeypatch, entry, ARTICLE_HTML)
    assert docs == []
    assert rej.as_dict()["non_english"] == 1
    assert fetched == [], "no request for an article that will be discarded"


def test_page_declaring_another_language_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = FeedEntry(
        url="https://example.com/a", title=None, published=datetime(2026, 3, 1, tzinfo=UTC)
    )
    html = f'<html lang="es"><body><article><p>{SPANISH_BODY}</p></article></body></html>'
    docs, rej, _ = _language_run(monkeypatch, entry, html)
    assert docs == []
    assert rej.as_dict()["non_english"] == 1


def test_mislabelled_page_is_caught_by_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page can claim en and still be Spanish."""
    entry = FeedEntry(
        url="https://example.com/a", title=None, published=datetime(2026, 3, 1, tzinfo=UTC)
    )
    html = f'<html lang="en"><body><article><p>{SPANISH_BODY}</p></article></body></html>'
    docs, rej, _ = _language_run(monkeypatch, entry, html)
    assert docs == []
    assert rej.as_dict()["non_english"] == 1


def test_non_english_sitemap_is_passed_over_for_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.ood_collection.collect import find_feed

    def news(url: str, lang: str) -> str:
        return (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
            'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
            f"<url><loc>{url}</loc><news:news><news:publication><news:name>E</news:name>"
            f"<news:language>{lang}</news:language></news:publication>"
            "<news:publication_date>2026-03-02T09:00:00Z</news:publication_date>"
            "</news:news></url></urlset>"
        )

    mundo = "https://example.com/mundo-news.xml"
    english = "https://example.com/english-news.xml"
    _feed_host(
        monkeypatch,
        {
            "https://example.com/": (200, "<html></html>"),
            mundo: (200, news("https://example.com/mundo/a", "es")),
            english: (200, news("https://example.com/news/a", "en")),
        },
    )
    robots = StubRobots(sitemaps=[mundo, english])
    result = find_feed("example.com", robots, FetchSettings())  # type: ignore[arg-type]
    assert isinstance(result, tuple)
    assert result[0] == english
    assert [e.url for e in result[1]] == ["https://example.com/news/a"]


def test_publisher_domain_is_always_a_self_name() -> None:
    from eval.ood_collection.collect import load_outlet_names

    names = load_outlet_names()
    assert names["breitbart.com"][0] == "Breitbart News", "listed names keep their order"
    assert "breitbart.com" in names["breitbart.com"]
    assert names["vox.com"].count("vox.com") == 1


def test_future_dated_entries_are_rejected_before_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A date after collection is wrong; guessing which way it is wrong is worse."""
    from datetime import timedelta

    later = datetime.now(UTC) + timedelta(days=30)
    docs, rej, fetched = _run(monkeypatch, _entries(("https://example.com/future", later)))
    assert docs == []
    assert rej.as_dict()["future_dated"] == 1
    assert fetched == []


def test_dates_within_a_day_of_collection_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time zone differences must not read as future dating."""
    from datetime import timedelta

    soon = datetime.now(UTC) + timedelta(hours=10)
    docs, rej, _ = _run(monkeypatch, _entries(("https://example.com/tz", soon)))
    assert len(docs) == 1
    assert "future_dated" not in rej.as_dict()
