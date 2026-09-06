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

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

logger = logging.getLogger(__name__)

SNAPSHOT_DATE: Final = datetime(2026, 1, 1, tzinfo=UTC)
SOURCES: Final = Path("eval/ood_collection/sources.yaml")

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
) -> tuple[str, list[FeedEntry]] | None:
    """Locate a usable feed for a domain and return its entries.

    Tries what the homepage advertises before the conventional paths, so a
    publisher who declares a feed is asked for exactly one URL.
    """
    feed_settings = settings.model_copy(update={"allowed_content_types": FEED_CONTENT_TYPES})

    # Try the www host as well as the bare one. Measured: theguardian.com
    # redirects into a chain that exhausts the hop limit, while
    # www.theguardian.com serves the homepage and advertises its feed directly.
    bases = [f"https://{domain}/"]
    if not domain.startswith("www."):
        bases.append(f"https://www.{domain}/")

    candidates: list[str] = []
    for base in bases:
        if not robots.allows(base):
            logger.info("%s: robots disallows %s", domain, base)
            continue
        robots.wait_for(base)
        try:
            home = fetch(base, settings=settings)
        except ProvenanceError as exc:
            logger.info("%s: %s unavailable (%s)", domain, base, type(exc).__name__)
            continue
        for url in candidate_feed_urls(home.text, base):
            if url not in candidates:
                candidates.append(url)

    if not candidates:
        logger.info("%s: no homepage reachable and no feed candidates", domain)
        return None

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
        # An HTML error page is not a hostile feed, it is not a feed at all.
        # Checking the type before parsing keeps the DTD refusal meaningful:
        # every HTML page declares a doctype, so without this the log filled
        # with "declared a DTD and was refused" for ordinary 404s.
        if response.content_type in {"text/html", "application/xhtml+xml"}:
            logger.debug("%s: %s served HTML, not a feed", domain, url)
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

    logger.info("%s: no usable feed found", domain)
    return None


def collect_domain(
    publisher: dict[str, object],
    *,
    limit: int,
    robots: RobotsCache,
    fetch_settings: FetchSettings,
    extract_settings: ExtractSettings,
    rejections: Rejection,
    snapshot: datetime = SNAPSHOT_DATE,
) -> list[Document]:
    """Collect up to ``limit`` articles from one publisher."""
    domain = str(publisher["domain"])
    found = find_feed(domain, robots, fetch_settings)
    if found is None:
        rejections.record("no_feed")
        return []

    _, entries = found
    documents: list[Document] = []

    for entry in entries:
        if len(documents) >= limit:
            break
        if entry.published is None:
            rejections.record("undated")
            continue
        if entry.published < snapshot:
            rejections.record("before_snapshot")
            continue
        if not robots.allows(entry.url):
            rejections.record("robots_disallowed")
            continue
        # Only take articles served by the publisher itself; a feed may point
        # at a syndication partner, which would put the wrong domain in the
        # grouped split.
        if urlsplit(entry.url).netloc.lower().removeprefix("www.") != domain.removeprefix("www."):
            rejections.record("offsite_url")
            continue

        robots.wait_for(entry.url)
        try:
            response = fetch(entry.url, settings=fetch_settings)
        except ProvenanceError as exc:
            rejections.record(f"fetch_{type(exc).__name__}")
            continue

        article = extract_article(response.text, url=entry.url, settings=extract_settings)
        if article is None:
            rejections.record("extraction_failed")
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
