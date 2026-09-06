"""robots.txt rule matching, including what the standard library gets wrong."""

from __future__ import annotations

import urllib.robotparser

import pytest
from pipeline.net.robots_rules import compile_pattern, crawl_delay, parse, path_of

pytestmark = pytest.mark.adversarial

UA = "ProvenanceBot/0.1 (+https://example.invalid/bot)"

# Verbatim shape of the Associated Press robots.txt read on 2026-09-06.
AP_STYLE = """User-Agent: *
Disallow:
Disallow: *_ptid=*
Disallow: /zephr*
Disallow: /search?q=*
Disallow: /*.rss
Disallow: /gallery/
"""


def test_wildcard_disallow_is_honoured() -> None:
    """The bug this module exists for.

    AP disallow `/*.rss`. urllib.robotparser reports the path as allowed
    because it compares the pattern as a literal prefix. Reporting permission
    where a publisher stated a restriction is the one failure mode that is not
    acceptable here.
    """
    rules = parse(AP_STYLE, UA)
    assert rules.allows(path_of("https://apnews.com/index.rss")) is False

    stdlib = urllib.robotparser.RobotFileParser()
    stdlib.parse(AP_STYLE.splitlines())
    assert stdlib.can_fetch(UA, "https://apnews.com/index.rss") is True, (
        "if this fails the stdlib gained wildcard support and this module can be revisited"
    )


def test_ordinary_article_paths_still_allowed() -> None:
    """Over-blocking would be a different bug, not a safe default."""
    rules = parse(AP_STYLE, UA)
    assert rules.allows(path_of("https://apnews.com/article/politics-123")) is True


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://x.com/zephr/anything", False),
        ("https://x.com/gallery/a", False),
        ("https://x.com/search?q=cats", False),
        ("https://x.com/news/a?_ptid=abc", False),
        ("https://x.com/news/a", True),
    ],
)
def test_pattern_forms(url: str, allowed: bool) -> None:
    assert parse(AP_STYLE, UA).allows(path_of(url)) is allowed


def test_end_anchor() -> None:
    rules = parse("User-agent: *\nDisallow: /*.pdf$\n", UA)
    assert rules.allows("/report.pdf") is False
    assert rules.allows("/report.pdf.html") is True


def test_longest_match_wins_and_allow_breaks_ties() -> None:
    body = "User-agent: *\nDisallow: /news/\nAllow: /news/public/\n"
    rules = parse(body, UA)
    assert rules.allows("/news/private/a") is False
    assert rules.allows("/news/public/a") is True


def test_agent_specific_group_replaces_wildcard_group() -> None:
    body = "User-agent: *\nDisallow: /\n\nUser-agent: ProvenanceBot\nDisallow: /admin/\n"
    rules = parse(body, UA)
    assert rules.allows("/anything") is True, "our own group should replace the wildcard group"
    assert rules.allows("/admin/x") is False


def test_empty_disallow_constrains_nothing() -> None:
    assert parse("User-agent: *\nDisallow:\n", UA).allows("/anything") is True


def test_no_applicable_rules_allows() -> None:
    assert parse("User-agent: SomeoneElse\nDisallow: /\n", UA).allows("/x") is True


def test_regex_metacharacters_in_patterns_are_literal() -> None:
    """A path pattern must not be able to act as a regex."""
    rules = parse("User-agent: *\nDisallow: /a.b/\n", UA)
    assert rules.allows("/a.b/x") is False
    assert rules.allows("/axb/x") is True, "the dot must not match any character"


def test_comments_are_stripped() -> None:
    rules = parse("User-agent: *  # everyone\nDisallow: /x/  # nope\n", UA)
    assert rules.allows("/x/a") is False


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("User-agent: *\nCrawl-delay: 5\n", 5.0),
        ("User-agent: *\nCrawl-delay: 2.5\n", 2.5),
        ("User-agent: *\nCrawl-delay: soon\n", None),
        ("User-agent: *\n", None),
    ],
)
def test_crawl_delay_parsing(body: str, expected: float | None) -> None:
    assert crawl_delay(body, UA) == expected


def test_compile_pattern_is_anchored_at_the_start() -> None:
    assert compile_pattern("/news/").match("/news/a") is not None
    assert compile_pattern("/news/").match("/uk/news/a") is None
