"""Archive backfill: spread, budget, month caps, page dates. No network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from eval.ood_collection import backfill as mod
from eval.ood_collection.collect import Rejection
from eval.ood_collection.config import CollectSettings
from pipeline.net.config import FetchSettings
from pipeline.net.feeds import FeedEntry
from pipeline.net.fetch import FetchResult
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import ExtractedArticle

PUBLISHER = {
    "domain": "example.com",
    "stratum": "regional",
    "lean": None,
    "low_quality": False,
}
BODY = " ".join(f"The council considered item {i} at length on the day." for i in range(20))


class StubRobots:
    def __init__(self, sitemaps: list[str] | None = None) -> None:
        self.sitemaps = sitemaps or []

    def allows(self, url: str) -> bool:
        return True

    def wait_for(self, url: str) -> None:
        return None

    def sitemaps_for(self, url: str) -> list[str]:
        return self.sitemaps


def test_spread_samples_the_whole_list() -> None:
    """Taking the first few children would sample one end of an archive."""
    items = [f"s{i}" for i in range(100)]
    picked = mod.spread(items, 5)
    assert picked == ["s0", "s20", "s40", "s60", "s80"]
    assert mod.spread(["a", "b"], 5) == ["a", "b"]
    assert mod.spread(items, 0) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-10", datetime(2026, 3, 10, tzinfo=UTC)),
        ("2026-03-10T09:15:00Z", datetime(2026, 3, 10, 9, 15, tzinfo=UTC)),
        ("2026-03-10T09:15:00+02:00", datetime(2026, 3, 10, 7, 15, tzinfo=UTC)),
        ("last Tuesday", None),
        (None, None),
    ],
)
def test_page_date(raw: str | None, expected: datetime | None) -> None:
    got = mod.page_date(raw)
    assert (got.astimezone(UTC) if got else None) == expected


def _run(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[FeedEntry],
    page_dates: dict[str, str | None],
    limits: CollectSettings | None = None,
) -> tuple[list, Rejection, list[str]]:
    fetched: list[str] = []

    def fake_candidates(
        domain: str, robots: object, settings: FetchSettings, lim: CollectSettings
    ) -> list[FeedEntry]:
        return candidates

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        fetched.append(url)
        return FetchResult(final_url=url, status_code=200, content_type="text/html", text=BODY)

    def fake_extract(html: str, **kwargs: object) -> ExtractedArticle:
        url = str(kwargs.get("url"))
        return ExtractedArticle(
            text=BODY,
            title=None,
            published=page_dates.get(url),
            removed={},
            had_invisible=False,
            declared_language="en",
            english_ratio=0.4,
        )

    monkeypatch.setattr(mod, "archive_candidates", fake_candidates)
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    monkeypatch.setattr(mod, "extract_article", fake_extract)

    rejections = Rejection()
    documents = mod.backfill_domain(
        PUBLISHER,
        robots=StubRobots(),  # type: ignore[arg-type]
        fetch_settings=FetchSettings(),
        extract_settings=ExtractSettings(),
        rejections=rejections,
        limits=limits or CollectSettings(),
        outlet_names=(),
    )
    return documents, rejections, fetched


def _entry(url: str, lastmod: datetime | None = None) -> FeedEntry:
    return FeedEntry(url=url, title=None, published=lastmod)


def test_month_cap_spreads_the_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the cap an archive listing newest first refills one fortnight."""
    candidates = [_entry(f"https://example.com/a{i}") for i in range(12)]
    dates = {c.url: f"2026-0{1 + i // 4}-1{i % 4}" for i, c in enumerate(candidates)}
    documents, rejections, _ = _run(monkeypatch, candidates, dates)

    months = sorted(d.published[:7] for d in documents)
    assert months == ["2026-01", "2026-01", "2026-02", "2026-02", "2026-03", "2026-03"]
    assert rejections.as_dict()["month_quota_full"] == 6


def test_fetch_budget_bounds_the_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [_entry(f"https://example.com/a{i}") for i in range(50)]
    dates = {c.url: None for c in candidates}
    limits = CollectSettings(backfill_fetch_budget=3)
    documents, rejections, fetched = _run(monkeypatch, candidates, dates, limits)

    assert documents == []
    assert len(fetched) == 3
    assert rejections.as_dict()["budget_exhausted"] == 1
    assert rejections.as_dict()["page_undated"] == 3


def test_lastmod_before_the_snapshot_is_decided_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _entry("https://example.com/old", datetime(2025, 6, 1, tzinfo=UTC))
    documents, rejections, fetched = _run(monkeypatch, [old], {})
    assert documents == []
    assert fetched == []
    assert rejections.as_dict()["before_snapshot"] == 1


def test_page_dated_before_the_snapshot_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """lastmod says this year; the page says 2019. The page wins."""
    entry = _entry("https://example.com/edited", datetime(2026, 8, 1, tzinfo=UTC))
    documents, rejections, fetched = _run(monkeypatch, [entry], {entry.url: "2019-04-02T10:00:00Z"})
    assert documents == []
    assert fetched == [entry.url]
    assert rejections.as_dict()["page_before_snapshot"] == 1


def test_kept_document_takes_its_date_from_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry("https://example.com/story", datetime(2026, 8, 1, tzinfo=UTC))
    documents, _, _ = _run(monkeypatch, [entry], {entry.url: "2026-04-05T08:00:00Z"})
    assert len(documents) == 1
    assert documents[0].published.startswith("2026-04-05")
    assert documents[0].domain == "example.com"
    assert documents[0].stratum == "regional"


def test_urls_already_collected_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry("https://example.com/dup")
    monkeypatch.setattr(mod, "archive_candidates", lambda *a, **k: [entry])
    monkeypatch.setattr(mod, "fetch", lambda *a, **k: pytest.fail("must not fetch a known URL"))
    documents = mod.backfill_domain(
        PUBLISHER,
        robots=StubRobots(),  # type: ignore[arg-type]
        fetch_settings=FetchSettings(),
        extract_settings=ExtractSettings(),
        rejections=Rejection(),
        outlet_names=(),
        already_have={entry.url},
    )
    assert documents == []
