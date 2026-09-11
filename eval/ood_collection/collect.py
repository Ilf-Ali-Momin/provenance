"""Collect the OOD suite: discover, fetch, extract, record.

Composes the pieces rather than reimplementing any of them. Every request goes
through the SSRF guarded client, every host is checked against its robots.txt
first and given its own crawl delay, every article is extracted and normalised
by the same pipeline that serves production, and every publisher marker is
stripped by the construction rule in `pipeline.text.extract`.

What this module adds is bookkeeping, and the bookkeeping is the point:

* **Selection is recorded, not just its result.** Every candidate URL that was
  seen and rejected is counted by reason. A corpus that reports only what it
  kept cannot be audited for what it silently dropped.
* **Failures are outcomes, not exceptions to swallow.** A paywall, a timeout, a
  robots refusal and an extraction miss are four different results and each is
  counted separately, because "we collected 400 articles" means nothing without
  knowing how many were attempted.
* **Nothing is written until the whole run is described.** The manifest carries
  per stratum, per lean and per cell counts, so a skew is visible in the
  artefact rather than discovered later in the results.

Article text does persist here, which makes this corpus the documented
exception to the project wide rule that it does not (SECURITY.md control 6).
The lawful basis, storage and retention obligations that come with it are
recorded in the collection plan, and this module writes only to the corpus
directory that plan describes.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit
from xml.etree import ElementTree

import yaml
from pipeline.errors import ProvenanceError
from pipeline.net.config import FetchSettings
from pipeline.net.feeds import FeedEntry, UnsafeFeedError, candidate_feed_urls, parse_feed
from pipeline.net.fetch import fetch
from pipeline.net.robots import RobotsCache
from pipeline.net.sitemaps import parse_sitemap, prefer_news
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

logger = logging.getLogger(__name__)

SNAPSHOT_DATE: Final = datetime(2026, 1, 1, tzinfo=UTC)

# A publication date later than the moment of collection is not a real date.
# The first frozen suite holds one: a cbsnews.com article dated 2026-10-03 and
# collected on 2026-09-11, which read day first would be 2026-03-10. One day of
# tolerance covers time zones; beyond that the entry is rejected rather than
# guessed at.
FUTURE_TOLERANCE: Final = timedelta(days=1)
SOURCES: Final = Path("eval/ood_collection/sources.yaml")
OUTLET_NAMES: Final = Path("eval/ood_collection/outlet_names.yaml")

# Politeness bounds on the sitemap fallback. A sitemap index can list hundreds
# of children; news sitemaps are asked for first, and only a few are fetched.
MAX_SITEMAPS: Final = 3
MAX_SITEMAP_CHILDREN: Final = 3

LEAN_BUCKETS: Final[tuple[str, ...]] = ("left", "lean_left", "center", "lean_right", "right")

# Feeds are served as XML, which the article path deliberately refuses. Widened
# here rather than in the default, because the article path's narrowness is a
# control and not an oversight: measured on 2026-09-06, the unwidened setting
# rejected every real feed as an unsupported type and the collector reported
# "no usable feed" for publishers that plainly have one.
FEED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/xml",
        "text/xml",
        "text/html",  # some hosts mislabel a feed; parsing decides, not the label
        "text/plain",
    }
)

# Statuses that mean the host has refused us, as opposed to a page being
# missing. Measured on 2026-09-11: chicagotribune.com answers every request
# from this collector with HTTP 403, and the collector, which did not look at
# status codes, went on to ask for sixteen candidate feed URLs on a host that
# had already said no, then logged the outcome as "no feed". A refusal ends
# collection from that host for the run. It is never worked around: changing
# the user agent or retrying to get past a block would be evading the
# publisher's decision, which this project does not do.
BLOCKING_STATUS: Final[frozenset[int]] = frozenset({401, 403, 407, 429, 451})


def _looks_like_xml(text: str) -> bool:
    """Whether a body is XML by its opening, whatever the label said."""
    head = text.lstrip()[:100].lower()
    return head.startswith(("<?xml", "<rss", "<feed", "<rdf", "<urlset", "<sitemapindex"))


@dataclass
class Rejection:
    """Why a candidate URL did not become a document."""

    counts: Counter[str] = field(default_factory=Counter)

    def record(self, reason: str) -> None:
        self.counts[reason] += 1

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(frozen=True)
class Document:
    """One collected article, publisher stripped and normalised."""

    doc_id: str
    domain: str
    stratum: str
    lean: str | None
    low_quality: bool | None
    url: str
    title: str | None
    published: str | None
    text: str
    text_sha256: str
    markers_removed: dict[str, int]
    had_invisible: bool
    collected_at: str


def document_id(url: str, text: str) -> str:
    """Stable id from the URL and the normalised text.

    Both, deliberately: the URL alone would collide across a republished
    article and the text alone would merge syndicated copies that this corpus
    wants to keep distinct by publisher.
    """
    digest = hashlib.sha256(f"{url}\n{text}".encode()).hexdigest()
    return f"doc_{digest[:16]}"


def load_publishers(path: Path = SOURCES) -> list[dict[str, object]]:
    """Read the source list, keeping only stratified publishers."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [p for p in data["publishers"] if p.get("stratum")]


def find_feed(
    domain: str, robots: RobotsCache, settings: FetchSettings
) -> tuple[str, list[FeedEntry]] | str:
    """Locate a usable feed and return its entries, or a reason there is none.

    Tries what the homepage advertises before the conventional paths, so a
    publisher who declares a feed is asked for exactly one URL. The reason is
    one of `host_refused`, `robots_disallowed`, `homepage_unreachable` or
    `no_feed`, because those are four different findings about a publisher
    and collapsing them into one hid a block behind "no feed".
    """
    feed_settings = settings.model_copy(update={"allowed_content_types": FEED_CONTENT_TYPES})

    # Try the www host as well as the bare one. Measured: theguardian.com
    # redirects into a chain that exhausts the hop limit, while
    # www.theguardian.com serves the homepage and advertises its feed directly.
    bases = [f"https://{domain}/"]
    if not domain.startswith("www."):
        bases.append(f"https://www.{domain}/")

    candidates: list[str] = []
    disallowed = False
    for base in bases:
        if not robots.allows(base):
            logger.info("%s: robots disallows %s", domain, base)
            disallowed = True
            continue
        robots.wait_for(base)
        try:
            home = fetch(base, settings=settings)
        except ProvenanceError as exc:
            logger.info("%s: %s unavailable (%s)", domain, base, type(exc).__name__)
            continue
        if home.status_code in BLOCKING_STATUS:
            logger.info("%s: %s refused with HTTP %d", domain, base, home.status_code)
            return "host_refused"
        for url in candidate_feed_urls(home.text, base):
            if url not in candidates:
                candidates.append(url)

    if not candidates:
        logger.info("%s: no homepage reachable and no feed candidates", domain)
        return "robots_disallowed" if disallowed else "homepage_unreachable"

    for url in candidates:
        if not robots.allows(url):
            logger.debug("%s: robots disallows %s", domain, url)
            continue
        robots.wait_for(url)
        # Each candidate that fails is logged with its reason rather than
        # skipped silently. Most publishers 404 several of the conventional
        # paths, which is expected, but a host failing *every* candidate for
        # the same reason is a finding about the collector, not the host.
        try:
            response = fetch(url, settings=feed_settings)
        except ProvenanceError as exc:
            logger.debug("%s: %s unreachable (%s)", domain, url, type(exc).__name__)
            continue
        if response.status_code in BLOCKING_STATUS:
            logger.info("%s: %s refused with HTTP %d", domain, url, response.status_code)
            return "host_refused"
        if response.status_code >= 400:
            logger.debug("%s: %s returned HTTP %d", domain, url, response.status_code)
            continue
        # Decide by the body, not the label. Some hosts serve a real feed as
        # text/html, which FEED_CONTENT_TYPES admits for that reason, and every
        # HTML page opens with a doctype that the DTD refusal would otherwise
        # report as an attack. An earlier version skipped anything labelled
        # HTML, which contradicted the admitted type and dropped mislabelled
        # feeds.
        if not _looks_like_xml(response.text):
            logger.debug("%s: %s is not XML", domain, url)
            continue
        try:
            entries = parse_feed(response.text)
        except UnsafeFeedError:
            logger.warning("%s: %s declared a DTD and was refused", domain, url)
            continue
        except ElementTree.ParseError as exc:
            logger.debug("%s: %s is not parseable XML (%s)", domain, url, exc)
            continue
        if entries:
            logger.info("%s: feed %s with %d entries", domain, url, len(entries))
            return url, entries

    from_sitemap = _entries_from_sitemaps(domain, robots, feed_settings)
    if from_sitemap is not None:
        return from_sitemap

    logger.info("%s: no usable feed or news sitemap found", domain)
    return "no_feed"


def _fetch_xml(domain: str, url: str, robots: RobotsCache, settings: FetchSettings) -> str | None:
    """Fetch a URL that should be XML, honouring robots and refusals."""
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


def _entries_from_sitemaps(
    domain: str, robots: RobotsCache, settings: FetchSettings
) -> tuple[str, list[FeedEntry]] | None:
    """Fall back to the sitemaps robots.txt declares.

    Measured on 2026-09-11: dallasnews.com advertises no feed and serves HTML
    at every conventional feed path, but lists its articles in sitemaps named
    in its robots.txt. Only entries dated by `news:publication_date` keep a
    date. A `lastmod` says when a page last changed, not when it was
    published, so an old article edited this year would otherwise pass the
    snapshot filter; such entries arrive undated and are rejected as undated.
    """
    base = f"https://{domain}/"
    declared = robots.sitemaps_for(base)
    if not declared:
        return None

    for sitemap_url in prefer_news(declared)[:MAX_SITEMAPS]:
        body = _fetch_xml(domain, sitemap_url, robots, settings)
        if body is None:
            continue
        try:
            sitemap = parse_sitemap(body)
        except (UnsafeFeedError, ElementTree.ParseError) as exc:
            logger.info("%s: sitemap %s unusable (%s)", domain, sitemap_url, type(exc).__name__)
            continue

        entries = list(sitemap.entries)
        for child in prefer_news(sitemap.children)[:MAX_SITEMAP_CHILDREN]:
            child_body = _fetch_xml(domain, child, robots, settings)
            if child_body is None:
                continue
            try:
                entries.extend(parse_sitemap(child_body).entries)
            except (UnsafeFeedError, ElementTree.ParseError) as exc:
                logger.info("%s: sitemap %s unusable (%s)", domain, child, type(exc).__name__)
                continue

        feed_entries = [
            FeedEntry(
                url=e.url,
                title=None,
                published=e.published if e.date_source == "publication_date" else None,
                language=e.language,
            )
            for e in entries
        ]
        # A sitemap whose entries all declare another language is a service in
        # that language: bbc.com lists BBC Mundo first, in Spanish. Move on to
        # the next declared sitemap rather than returning one that can only be
        # rejected, which left BBC yielding nothing once the gate existed.
        english = [e for e in feed_entries if not e.language or e.language.lower().startswith("en")]
        if feed_entries and not english:
            logger.info("%s: sitemap %s is not English, trying the next", domain, sitemap_url)
            continue
        feed_entries = english
        if feed_entries:
            # Newest first, undated last, so the per publisher limit is spent on
            # articles that can pass the snapshot filter.
            floor = datetime.min.replace(tzinfo=UTC)
            feed_entries.sort(key=lambda e: e.published or floor, reverse=True)
            logger.info("%s: sitemap %s with %d entries", domain, sitemap_url, len(feed_entries))
            return sitemap_url, feed_entries
    return None


@functools.cache
def load_outlet_names(path: Path = OUTLET_NAMES) -> dict[str, tuple[str, ...]]:
    """How each publisher names itself, for masking in its own copy."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # The bare domain is always a self name as well. The re-cleaned full run
    # still had "breitbart.com/downrange" in Breitbart copy after every listed
    # name was masked. Matched case sensitively like the rest, which is the
    # lower case form publishers print in URLs.
    return {
        str(domain): tuple(dict.fromkeys([*(str(n) for n in names), str(domain)]))
        for domain, names in data.items()
    }


def collect_domain(
    publisher: dict[str, object],
    *,
    limit: int,
    robots: RobotsCache,
    fetch_settings: FetchSettings,
    extract_settings: ExtractSettings,
    rejections: Rejection,
    snapshot: datetime = SNAPSHOT_DATE,
    outlet_names: Sequence[str] | None = None,
) -> list[Document]:
    """Collect up to ``limit`` articles from one publisher."""
    domain = str(publisher["domain"])
    names = outlet_names if outlet_names is not None else load_outlet_names().get(domain, ())
    found = find_feed(domain, robots, fetch_settings)
    if isinstance(found, str):
        rejections.record(found)
        return []

    feed_url, entries = found
    # Articles are accepted from the publisher's domain and from the host of
    # the feed the publisher itself advertised. Measured on 2026-09-11: the
    # dailymail.co.uk homepage advertises a feed on www.dailymail.com whose 150
    # entries all live there, and a same-domain-only rule rejected every one as
    # offsite. The document still records the publisher's domain, so grouping
    # for splits is unchanged.
    allowed_hosts = {
        domain.lower().removeprefix("www."),
        urlsplit(feed_url).netloc.lower().removeprefix("www."),
    }
    documents: list[Document] = []
    collected_now = datetime.now(UTC)

    for entry in entries:
        if len(documents) >= limit:
            break
        if entry.published is None:
            rejections.record("undated")
            continue
        if entry.published > collected_now + FUTURE_TOLERANCE:
            rejections.record("future_dated")
            continue
        if entry.published < snapshot:
            rejections.record("before_snapshot")
            continue
        # English only (BRIEF.md). A declared non-English entry is skipped
        # before any request is made for it.
        if entry.language and not entry.language.lower().startswith("en"):
            rejections.record("non_english")
            continue
        if not robots.allows(entry.url):
            rejections.record("robots_disallowed")
            continue
        # Only take articles served by the publisher itself; a feed may point
        # at a syndication partner, which would put the wrong domain in the
        # grouped split.
        if urlsplit(entry.url).netloc.lower().removeprefix("www.") not in allowed_hosts:
            rejections.record("offsite_url")
            continue

        robots.wait_for(entry.url)
        try:
            response = fetch(entry.url, settings=fetch_settings)
        except ProvenanceError as exc:
            rejections.record(f"fetch_{type(exc).__name__}")
            continue
        if response.status_code in BLOCKING_STATUS:
            # A publisher that starts refusing part way through has said stop.
            # Honour it for the rest of this host instead of asking per entry.
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
            # Checked on the page as well as the listing, because most
            # listings declare no language and some declarations are wrong.
            rejections.record("non_english")
            continue

        documents.append(
            Document(
                doc_id=document_id(entry.url, article.text),
                domain=domain,
                stratum=str(publisher["stratum"]),
                lean=publisher.get("lean"),  # type: ignore[arg-type]
                low_quality=publisher.get("low_quality"),  # type: ignore[arg-type]
                url=entry.url,
                title=article.title,
                published=entry.published.isoformat(),
                text=article.text,
                text_sha256=hashlib.sha256(article.text.encode()).hexdigest(),
                markers_removed=article.removed,
                had_invisible=article.had_invisible,
                collected_at=datetime.now(UTC).isoformat(),
            )
        )

    logger.info("%s: kept %d", domain, len(documents))
    return documents


def summarise(documents: list[Document], rejections: Rejection) -> dict[str, object]:
    """Describe the run, including everything it failed to collect."""
    by_stratum = Counter(d.stratum for d in documents)
    by_lean = Counter(d.lean for d in documents)
    by_domain = Counter(d.domain for d in documents)
    partisan_lean = Counter(d.lean for d in documents if d.stratum == "partisan")
    markers: Counter[str] = Counter()
    for doc in documents:
        markers.update(doc.markers_removed)

    return {
        "documents": len(documents),
        "domains": len(by_domain),
        "by_stratum": dict(by_stratum),
        "by_lean": {str(k): v for k, v in by_lean.items()},
        "partisan_lean_balance": {
            "left_of_centre": partisan_lean["left"] + partisan_lean["lean_left"],
            "right_of_centre": partisan_lean["right"] + partisan_lean["lean_right"],
        },
        "low_quality_flagged": sum(1 for d in documents if d.low_quality),
        "publisher_markers_removed": dict(markers),
        "documents_with_invisible_chars": sum(1 for d in documents if d.had_invisible),
        "rejected": rejections.as_dict(),
        "snapshot_date": SNAPSHOT_DATE.date().isoformat(),
    }


def write_corpus(documents: list[Document], summary: dict[str, object], out_dir: Path) -> None:
    """Write documents and the run summary. Not a freeze; see the plan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
