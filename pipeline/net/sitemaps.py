"""Sitemap parsing, the fallback when a publisher has no discoverable feed.

Measured on 2026-09-11: dallasnews.com serves its homepage, advertises no feed
in it, and answers every conventional feed path with an HTML page. Its
articles are still listed machine readably, in sitemaps declared through the
`Sitemap:` lines of robots.txt. A sitemap is the publisher's own statement of
what may be indexed, and news sitemaps carry a publication date per article,
which is exactly what selection on the snapshot date needs.

Two formats matter. A `urlset` lists pages; a `sitemapindex` lists further
sitemaps. Dates come from `news:publication_date` when present, else from
`lastmod`, and `lastmod` is recorded as the weaker of the two because a page
edited in March may have been published in 2019.

The XML is fetched from a remote host and gets the same refusal as feeds: any
document declaring a DTD or entity is rejected before parsing, which is the
entity expansion defence available without `defusedxml`.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from xml.etree import ElementTree

from pipeline.net.feeds import UnsafeFeedError

_DOCTYPE = re.compile(r"<!DOCTYPE", re.IGNORECASE)
_ENTITY = re.compile(r"<!ENTITY", re.IGNORECASE)


@dataclass(frozen=True)
class SitemapEntry:
    """One page listed by a sitemap."""

    url: str
    published: datetime | None
    date_source: str | None  # "publication_date", "lastmod", or None


@dataclass(frozen=True)
class Sitemap:
    entries: list[SitemapEntry]
    children: list[str]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    with contextlib.suppress(ValueError):
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def parse_sitemap(xml_text: str) -> Sitemap:
    """Parse a urlset or a sitemapindex.

    Raises UnsafeFeedError for a DTD or entity, and ElementTree.ParseError for
    malformed XML. Neither is swallowed: an empty result must mean an empty
    sitemap, never a failure that looked like one.
    """
    if _DOCTYPE.search(xml_text) or _ENTITY.search(xml_text):
        raise UnsafeFeedError("Sitemap declares a DTD or entity and was refused.")

    root = ElementTree.fromstring(xml_text)  # noqa: S314 - guarded above
    kind = _local(root.tag)
    entries: list[SitemapEntry] = []
    children: list[str] = []

    for node in root:
        name = _local(node.tag)
        loc: str | None = None
        lastmod: str | None = None
        publication: str | None = None
        for element in node.iter():
            local = _local(element.tag)
            text = (element.text or "").strip()
            if local == "loc" and text and loc is None:
                loc = text
            elif local == "lastmod" and text:
                lastmod = text
            elif local == "publication_date" and text:
                publication = text
        if not loc:
            continue
        if kind == "sitemapindex" and name == "sitemap":
            children.append(loc)
        elif name == "url":
            published = _parse_date(publication)
            source: str | None = "publication_date" if published else None
            if published is None:
                published = _parse_date(lastmod)
                source = "lastmod" if published else None
            entries.append(SitemapEntry(url=loc, published=published, date_source=source))

    return Sitemap(entries=entries, children=children)


def prefer_news(children: list[str]) -> list[str]:
    """Order child sitemaps with news sitemaps first.

    A sitemap index commonly lists dozens of children, most of them archives
    by year or month. Asking for the news ones first means the recent articles
    selection needs usually arrive in one or two requests rather than twenty.
    """

    def rank(url: str) -> int:
        lowered = url.lower()
        if "news" in lowered:
            return 0
        if "2026" in lowered:
            return 1
        return 2

    return sorted(children, key=rank)
