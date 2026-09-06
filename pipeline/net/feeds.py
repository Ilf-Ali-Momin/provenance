"""Feed discovery and parsing for article URL collection.

Feeds rather than crawling. A publisher's own feed is the list of articles they
intend to be read programmatically, it carries publication dates without
fetching each article first, and it costs one request instead of walking a
site. For a corpus that must select on publication date, that last point is
what makes collection affordable at all.

Parsed with the standard library. `feedparser` would be a dependency added for
a job `xml.etree` already does, and RSS and Atom differ in about six element
names between them.

**The XML is attacker controlled.** A feed is fetched from a remote host, so it
gets the same treatment as any other fetched byte (README, non negotiable
three). `defusedxml` is not available here, so the two attacks that matter
against `xml.etree` are handled directly: entity expansion is refused by
rejecting any document declaring a DTD, and the parser never resolves external
entities. Size is already bounded upstream by the fetcher's response cap.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urljoin
from xml.etree import ElementTree

# Paths worth trying when a page advertises no feed. Ordered by how commonly
# they are the real one, so the first hit is usually the first request.
COMMON_FEED_PATHS: Final[tuple[str, ...]] = (
    "/rss",
    "/feed",
    "/rss.xml",
    "/feed.xml",
    "/index.xml",
    "/atom.xml",
    "/feeds/all.atom.xml",
    "/news/rss",
)

_FEED_LINK = re.compile(
    r"""<link[^>]+(?:rel=["']alternate["'][^>]*type=["']application/(?:rss|atom)\+xml["']"""
    r"""|type=["']application/(?:rss|atom)\+xml["'][^>]*rel=["']alternate["'])[^>]*>""",
    re.IGNORECASE,
)
_HREF = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)

# A DTD in a fetched document is never legitimate here and is the entry point
# for entity expansion attacks, so its presence rejects the document outright.
_DOCTYPE = re.compile(r"<!DOCTYPE", re.IGNORECASE)
_ENTITY = re.compile(r"<!ENTITY", re.IGNORECASE)

_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%a, %d %b %Y %H:%M:%S %z",  # RFC 822, RSS
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%dT%H:%M:%S%z",  # RFC 3339, Atom
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
)


@dataclass(frozen=True)
class FeedEntry:
    """One article advertised by a feed."""

    url: str
    title: str | None
    published: datetime | None


class UnsafeFeedError(ValueError):
    """The feed declared a DTD or entity and was refused before parsing."""


def discover_feed_links(html: str, base_url: str) -> list[str]:
    """Feed URLs a page advertises in its head, resolved to absolute."""
    out: list[str] = []
    for tag in _FEED_LINK.findall(html):
        href = _HREF.search(tag)
        if href:
            absolute = urljoin(base_url, href.group(1))
            if absolute not in out:
                out.append(absolute)
    return out


def candidate_feed_urls(html: str, base_url: str) -> list[str]:
    """Advertised feeds first, then the conventional paths."""
    found = discover_feed_links(html, base_url)
    for path in COMMON_FEED_PATHS:
        guess = urljoin(base_url, path)
        if guess not in found:
            found.append(guess)
    return found


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        # A format that does not match is not an error, it is the next
        # candidate. An unparseable date returns None and is recorded as
        # undated rather than dropping the entry.
        with contextlib.suppress(ValueError):
            parsed = datetime.strptime(text, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _tag(element: ElementTree.Element) -> str:
    """Local tag name, namespace stripped."""
    return element.tag.rsplit("}", 1)[-1].lower()


def parse_feed(xml_text: str) -> list[FeedEntry]:
    """Parse an RSS or Atom feed into entries.

    Refuses any document declaring a DTD or entity before parsing begins,
    which is the entity expansion defence. Malformed XML raises
    ``ElementTree.ParseError`` and is a recorded collection failure, not a
    retry against a different parser.
    """
    if _DOCTYPE.search(xml_text) or _ENTITY.search(xml_text):
        raise UnsafeFeedError("Feed declares a DTD or entity and was refused.")

    root = ElementTree.fromstring(xml_text)  # noqa: S314 - guarded above
    entries: list[FeedEntry] = []

    for element in root.iter():
        name = _tag(element)
        if name not in {"item", "entry"}:
            continue

        url: str | None = None
        title: str | None = None
        published: str | None = None

        for child in element:
            child_name = _tag(child)
            if child_name == "link":
                # RSS puts the URL in the text, Atom in an href attribute.
                href = child.get("href")
                rel = (child.get("rel") or "alternate").lower()
                if href and rel == "alternate":
                    url = href
                elif child.text and not url:
                    url = child.text.strip()
            elif child_name == "title" and child.text:
                title = child.text.strip()
            elif child_name in {"pubdate", "published", "updated", "date"} and child.text:
                published = published or child.text.strip()

        if url:
            entries.append(FeedEntry(url=url, title=title, published=_parse_date(published)))

    return entries
