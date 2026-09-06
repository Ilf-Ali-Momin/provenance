"""Feed parsing, including the hostile cases. No network."""

from __future__ import annotations

from datetime import UTC, datetime
from xml.etree import ElementTree

import pytest
from pipeline.net.feeds import (
    UnsafeFeedError,
    candidate_feed_urls,
    discover_feed_links,
    parse_feed,
)

pytestmark = pytest.mark.adversarial

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Example</title>
<item><title>First</title><link>https://example.com/a</link>
<pubDate>Mon, 05 Jan 2026 10:00:00 +0000</pubDate></item>
<item><title>Second</title><link>https://example.com/b</link>
<pubDate>Tue, 06 Jan 2026 11:30:00 +0000</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Example</title>
<entry><title>Third</title><link rel="alternate" href="https://example.com/c"/>
<published>2026-01-07T09:15:00Z</published></entry>
</feed>"""

BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>
<rss><channel><item><link>https://example.com/x</link><title>&lol2;</title></item></channel></rss>"""

EXTERNAL_ENTITY = """<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<rss><channel><item><link>&xxe;</link></item></channel></rss>"""


def test_parses_rss() -> None:
    entries = parse_feed(RSS)
    assert [e.url for e in entries] == ["https://example.com/a", "https://example.com/b"]
    assert entries[0].title == "First"
    assert entries[0].published == datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


def test_parses_atom_href_link() -> None:
    entries = parse_feed(ATOM)
    assert entries[0].url == "https://example.com/c"
    assert entries[0].published == datetime(2026, 1, 7, 9, 15, tzinfo=UTC)


def test_entity_expansion_is_refused_before_parsing() -> None:
    """A feed is a fetched byte and every fetched byte is hostile."""
    with pytest.raises(UnsafeFeedError):
        parse_feed(BILLION_LAUGHS)


def test_external_entity_is_refused() -> None:
    with pytest.raises(UnsafeFeedError):
        parse_feed(EXTERNAL_ENTITY)


def test_malformed_xml_raises_rather_than_returning_nothing() -> None:
    """Silent empty results would look like a publisher with no articles."""
    with pytest.raises(ElementTree.ParseError):
        parse_feed("<rss><channel><item>")


def test_entry_without_link_is_skipped() -> None:
    feed = "<rss><channel><item><title>No link</title></item></channel></rss>"
    assert parse_feed(feed) == []


def test_undated_entry_is_kept_with_none() -> None:
    """Date filtering decides later; parsing does not silently drop."""
    feed = "<rss><channel><item><link>https://e.com/x</link></item></channel></rss>"
    entries = parse_feed(feed)
    assert len(entries) == 1
    assert entries[0].published is None


def test_discovers_advertised_feed() -> None:
    html = (
        '<html><head><link rel="alternate" type="application/rss+xml" '
        'href="/feeds/main.xml"></head><body></body></html>'
    )
    assert discover_feed_links(html, "https://example.com/") == [
        "https://example.com/feeds/main.xml"
    ]


def test_discovers_atom_with_attributes_reversed() -> None:
    html = '<link type="application/atom+xml" rel="alternate" href="https://e.com/atom">'
    assert discover_feed_links(html, "https://e.com/") == ["https://e.com/atom"]


def test_candidates_put_advertised_feeds_first() -> None:
    html = '<link rel="alternate" type="application/rss+xml" href="/real.xml">'
    candidates = candidate_feed_urls(html, "https://example.com/")
    assert candidates[0] == "https://example.com/real.xml"
    assert "https://example.com/rss" in candidates
