"""Sitemap parsing, including hostile input. No network."""

from __future__ import annotations

from datetime import UTC, datetime
from xml.etree import ElementTree

import pytest
from pipeline.net.feeds import UnsafeFeedError
from pipeline.net.robots_rules import sitemap_urls
from pipeline.net.sitemaps import parse_sitemap, prefer_news

pytestmark = pytest.mark.adversarial

NEWS_URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://example.com/news/politics/a</loc>
    <news:news>
      <news:publication><news:name>Example</news:name></news:publication>
      <news:publication_date>2026-03-02T09:00:00Z</news:publication_date>
    </news:news>
    <lastmod>2026-08-01T00:00:00Z</lastmod>
  </url>
  <url>
    <loc>https://example.com/business/b</loc>
    <lastmod>2026-04-05</lastmod>
  </url>
  <url>
    <loc>https://example.com/undated</loc>
  </url>
</urlset>"""

INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-2019.xml</loc></sitemap>
  <sitemap><loc>https://example.com/news-sitemap.xml</loc></sitemap>
  <sitemap><loc>https://example.com/sitemap-2026-03.xml</loc></sitemap>
</sitemapindex>"""


def test_publication_date_is_preferred_over_lastmod() -> None:
    """A page edited in August may have been published in March."""
    entries = parse_sitemap(NEWS_URLSET).entries
    assert entries[0].published == datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
    assert entries[0].date_source == "publication_date"


def test_lastmod_is_used_when_nothing_better_exists() -> None:
    entry = parse_sitemap(NEWS_URLSET).entries[1]
    assert entry.published == datetime(2026, 4, 5, tzinfo=UTC)
    assert entry.date_source == "lastmod"


def test_undated_url_is_kept_with_none() -> None:
    entry = parse_sitemap(NEWS_URLSET).entries[2]
    assert entry.published is None
    assert entry.date_source is None


def test_index_returns_children_not_entries() -> None:
    sitemap = parse_sitemap(INDEX)
    assert sitemap.entries == []
    assert len(sitemap.children) == 3


def test_news_children_are_requested_first() -> None:
    ordered = prefer_news(parse_sitemap(INDEX).children)
    assert ordered[0].endswith("news-sitemap.xml")
    assert ordered[-1].endswith("sitemap-2019.xml")


def test_dtd_is_refused_before_parsing() -> None:
    hostile = (
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]>'
        "<urlset><url><loc>&a;</loc></url></urlset>"
    )
    with pytest.raises(UnsafeFeedError):
        parse_sitemap(hostile)


def test_malformed_xml_raises() -> None:
    with pytest.raises(ElementTree.ParseError):
        parse_sitemap("<urlset><url>")


def test_sitemap_lines_are_read_from_robots() -> None:
    body = (
        "User-agent: *\nDisallow: /private/\n"
        "Sitemap: https://example.com/news-sitemap.xml\n"
        "sitemap:https://example.com/sitemap.xml  # trailing comment\n"
    )
    assert sitemap_urls(body) == [
        "https://example.com/news-sitemap.xml",
        "https://example.com/sitemap.xml",
    ]


def test_sitemap_lines_are_deduplicated() -> None:
    body = "Sitemap: https://e.com/s.xml\nSitemap: https://e.com/s.xml\n"
    assert sitemap_urls(body) == ["https://e.com/s.xml"]


def test_news_language_is_captured() -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
        "<url><loc>https://e.com/mundo/a</loc><news:news><news:publication>"
        "<news:name>E</news:name><news:language>es</news:language></news:publication>"
        "<news:publication_date>2026-03-02T09:00:00Z</news:publication_date>"
        "</news:news></url></urlset>"
    )
    assert parse_sitemap(xml).entries[0].language == "es"
