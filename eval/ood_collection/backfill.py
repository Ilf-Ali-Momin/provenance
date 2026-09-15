"""Backfill the OOD suite from publisher archives (D022).

Feeds and news sitemaps list the last day or two, which is why the first frozen
suite was one day of news. Archive sitemaps list far more, but date entries only
by last modification, and a page edited this year may have been published years
ago. D019 therefore refuses `lastmod` as a publication date, and D022 resolves
the impasse: take candidates from the archive, fetch each one, and keep it only
if the page itself states a publication date after the snapshot.

Two limits make that affordable and make the result a spread rather than a pile.
A per publisher fetch budget bounds the cost of candidates that turn out to be
undated or too old, which most will be. A per month cap stops a publisher whose
archive lists newest first from filling its whole quota from one fortnight,
which is the failure this module exists to undo.

Candidates are shuffled with a fixed seed before the caps are applied, so the
sample spreads across whatever the archive holds instead of following its
ordering, and the same archive yields the same sample twice.
"""

from __future__ import annotations

import contextlib
import logging
import random
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import urlsplit
from xml.etree import ElementTree

from pipeline.errors import ProvenanceError
from pipeline.net.config import FetchSettings
from pipeline.net.feeds import FeedEntry, UnsafeFeedError
from pipeline.net.fetch import fetch
from pipeline.net.robots import RobotsWithSitemaps
from pipeline.net.sitemaps import parse_sitemap
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

from eval.ood_collection.collect import (
    BLOCKING_STATUS,
    FEED_CONTENT_TYPES,
    FUTURE_TOLERANCE,
    SNAPSHOT_DATE,
    Document,
    Rejection,
    _looks_like_xml,
    build_document,
    load_outlet_names,
)
from eval.ood_collection.config import CollectSettings

logger = logging.getLogger(__name__)

SEED = 20260101


def spread(items: list[str], count: int) -> list[str]:
    """Take ``count`` items spaced evenly across the list.

    Sitemap indexes are usually ordered by date, so taking the first few
    children would sample one end of the archive. Even spacing samples the
    whole of it for the same number of requests.
    """
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def page_date(raw: str | None) -> datetime | None:
    """A publication date stated by the page itself, or None."""
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    # An unparseable date is not an error to propagate: the article is simply
    # rejected as undated rather than kept with a guessed date.
    with contextlib.suppress(ValueError):
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _fetch_xml(
    domain: str, url: str, robots: RobotsWithSitemaps, settings: FetchSettings
) -> str | None:
    if not robots.allows(url):
        return None
    robots.wait_for(url)
    try:
        response = fetch(url, settings=settings)
    except ProvenanceError as exc:
        logger.debug("%s: %s unreachable (%s)", domain, url, type(exc).__name__)
        return None
    if response.status_code >= 400 or not _looks_like_xml(response.text):
        return None
    return response.text


def archive_candidates(
    domain: str,
    robots: RobotsWithSitemaps,
    settings: FetchSettings,
    limits: CollectSettings | None = None,
) -> list[FeedEntry]:
    """Candidate article URLs from every sitemap the host declares."""
    limits = limits or CollectSettings()
    feed_settings = settings.model_copy(update={"allowed_content_types": FEED_CONTENT_TYPES})
    declared = robots.sitemaps_for(f"https://{domain}/")
    found: list[FeedEntry] = []
    seen: set[str] = set()

    for sitemap_url in declared[: limits.backfill_sitemaps]:
        body = _fetch_xml(domain, sitemap_url, robots, feed_settings)
        if body is None:
            continue
        try:
            sitemap = parse_sitemap(body)
        except (UnsafeFeedError, ElementTree.ParseError) as exc:
            logger.info("%s: sitemap %s unusable (%s)", domain, sitemap_url, type(exc).__name__)
            continue

        groups = [sitemap]
        for child in spread(sitemap.children, limits.backfill_children):
            child_body = _fetch_xml(domain, child, robots, feed_settings)
            if child_body is None:
                continue
            try:
                groups.append(parse_sitemap(child_body))
            except (UnsafeFeedError, ElementTree.ParseError) as exc:
                logger.info("%s: sitemap %s unusable (%s)", domain, child, type(exc).__name__)

        for group in groups:
            for entry in group.entries:
                if entry.url in seen:
                    continue
                seen.add(entry.url)
                found.append(
                    FeedEntry(
                        url=entry.url,
                        title=None,
                        published=entry.published,
                        language=entry.language,
                    )
                )
        if len(found) >= limits.backfill_candidates:
            break

    rng = random.Random(SEED)  # noqa: S311 - reproducible sampling, not cryptographic
    rng.shuffle(found)
    logger.info("%s: %d archive candidates", domain, len(found))
    return found[: limits.backfill_candidates]


def backfill_domain(
    publisher: dict[str, object],
    *,
    robots: RobotsWithSitemaps,
    fetch_settings: FetchSettings,
    extract_settings: ExtractSettings,
    rejections: Rejection,
    limits: CollectSettings | None = None,
    snapshot: datetime = SNAPSHOT_DATE,
    outlet_names: tuple[str, ...] | None = None,
    already_have: set[str] | None = None,
    months_seen: Counter[str] | None = None,
) -> list[Document]:
    """Fetch archive candidates and keep those the page dates after the snapshot.

    ``months_seen`` carries the months this publisher has already contributed
    from the live window, so the per month cap applies to the publisher's whole
    sample rather than to the archive alone. Without it the cap cannot do the
    job it exists for: the live window is itself a month, and the first backfill
    run left the suite 459 documents of 528 in September because the archive was
    capped and the feed was not.
    """
    limits = limits or CollectSettings()
    domain = str(publisher["domain"])
    names = outlet_names if outlet_names is not None else load_outlet_names().get(domain, ())
    have = already_have or set()

    candidates = archive_candidates(domain, robots, fetch_settings, limits)
    month_counts: Counter[str] = Counter(months_seen or {})
    months_added: set[str] = set()
    documents: list[Document] = []
    fetched = 0
    now = datetime.now(UTC)

    for entry in candidates:
        if fetched >= limits.backfill_fetch_budget:
            rejections.record("budget_exhausted")
            break
        if entry.url in have:
            continue
        if entry.language and not entry.language.lower().startswith("en"):
            rejections.record("non_english")
            continue
        if urlsplit(entry.url).netloc.lower().removeprefix("www.") != domain.removeprefix("www."):
            rejections.record("offsite_url")
            continue
        if not robots.allows(entry.url):
            rejections.record("robots_disallowed")
            continue
        # A page last modified before the snapshot cannot have been published
        # after it, so this one is decidable without a request.
        if entry.published is not None and entry.published < snapshot:
            rejections.record("before_snapshot")
            continue

        robots.wait_for(entry.url)
        fetched += 1
        try:
            response = fetch(entry.url, settings=fetch_settings)
        except ProvenanceError as exc:
            rejections.record(f"fetch_{type(exc).__name__}")
            continue
        if response.status_code in BLOCKING_STATUS:
            rejections.record("host_refused")
            break
        if response.status_code >= 400:
            rejections.record(f"http_{response.status_code}")
            continue

        article = extract_article(
            response.text, url=entry.url, settings=extract_settings, outlet_names=names
        )
        if article is None:
            rejections.record("extraction_failed")
            continue
        declared = (article.declared_language or "").lower()
        if (declared and not declared.startswith("en")) or (
            article.english_ratio < extract_settings.min_english_stopword_ratio
        ):
            rejections.record("non_english")
            continue

        published = page_date(article.published)
        if published is None:
            rejections.record("page_undated")
            continue
        if published > now + FUTURE_TOLERANCE:
            rejections.record("future_dated")
            continue
        if published < snapshot:
            rejections.record("page_before_snapshot")
            continue

        month = published.strftime("%Y-%m")
        if month_counts[month] >= limits.backfill_per_month:
            rejections.record("month_quota_full")
            continue
        month_counts[month] += 1
        months_added.add(month)
        documents.append(build_document(publisher, entry.url, published, article))

    logger.info(
        "%s: backfill kept %d from %d fetches across %d months",
        domain,
        len(documents),
        fetched,
        len(months_added),
    )
    return documents
