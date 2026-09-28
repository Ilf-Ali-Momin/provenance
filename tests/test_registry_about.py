"""About-page extraction, and the claims it refuses to make.

This is the most defamation-exposed code in the project. Most of these tests are
about phrasing and scope rather than parsing: what an observation asserts, and
what it declines to.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pipeline.registry.about import (
    content_matches,
    find_candidates,
    looks_like_an_article,
    observe,
    page_summary,
    visible_text,
)
from pipeline.registry.schema import ObservationKind, Source

WHEN = datetime(2026, 9, 29, tzinfo=UTC)
BASE = "https://example.org/"

HOMEPAGE = """
<html><body>
  <nav>
    <a href="/">Home</a>
    <a href="/news/politics">Politics</a>
    <a href="#top">Top</a>
    <a href="mailto:news@example.org">Email us</a>
  </nav>
  <footer>
    <a href="/about-us">About us</a>
    <a href="/corrections">Corrections</a>
    <a href="/our-team">Our team</a>
    <a href="https://ipso.co.uk/complain">Complaints to IPSO</a>
  </footer>
</body></html>
"""

CORRECTIONS_PAGE = (
    "Corrections and clarifications. We correct errors promptly and note the "
    "correction at the foot of the article. Readers who believe an article is "
    "inaccurate should contact the editor; our complaints procedure is set out "
    "below and we aim to respond within five working days."
)
TEAM_PAGE = (
    "Our team. Jane Smith, Editor-in-Chief. Aziz Rahman, Managing Editor. "
    "Priya Nair, News Editor. Our reporters cover local government, health and "
    "transport, and our correspondents file from three regions."
)
ABOUT_PAGE = (
    "About us. The Example Herald is published by Example Media Limited, a "
    "registered company in England and Wales. The title is owned by the Example "
    "Trust and funded by subscriptions and philanthropic grants."
)
MARKETING_PAGE = "Welcome to our award-winning newsroom. Subscribe today for unlimited access."

PAGES = {
    "https://example.org/corrections": CORRECTIONS_PAGE,
    "https://example.org/our-team": TEAM_PAGE,
    "https://example.org/about-us": ABOUT_PAGE,
}


def fetch(url: str) -> str:
    if url not in PAGES:
        raise LookupError(url)
    return PAGES[url]


# --- link discovery --------------------------------------------------------


def test_the_three_kinds_are_found_from_the_homepage_links() -> None:
    kinds = {c.kind for c in find_candidates(HOMEPAGE, BASE)}
    assert kinds == {
        ObservationKind.CORRECTIONS_POLICY_PRESENT,
        ObservationKind.MASTHEAD_PRESENT,
        ObservationKind.OWNERSHIP_STATED,
    }


def test_an_offsite_regulator_link_is_not_this_publishers_policy() -> None:
    """Recording IPSO's complaints page as the publisher's corrections policy
    would attribute somebody else's document to them."""
    urls = {c.url for c in find_candidates(HOMEPAGE, BASE)}
    assert not any("ipso.co.uk" in u for u in urls)


def test_fragments_and_mailto_links_are_skipped() -> None:
    urls = {c.url for c in find_candidates(HOMEPAGE, BASE)}
    assert not any(u.startswith("mailto:") or u.endswith("#top") for u in urls)


def test_a_subdomain_counts_as_the_same_publisher() -> None:
    html = '<a href="https://corrections.example.org/policy">Corrections</a>'
    found = find_candidates(html, BASE)
    assert found and found[0].url.startswith("https://corrections.example.org/")


def test_the_path_is_matched_when_the_link_text_does_not() -> None:
    """Some publishers title the link "Editorial standards" and put it at
    /corrections, and some do the reverse."""
    html = '<a href="/corrections">Standards</a>'
    kinds = {c.kind for c in find_candidates(html, BASE)}
    assert ObservationKind.CORRECTIONS_POLICY_PRESENT in kinds


def test_malformed_html_yields_no_candidates_rather_than_raising() -> None:
    assert find_candidates("", BASE) == []
    assert find_candidates("<<<not html", BASE) == []


# --- content confirmation --------------------------------------------------


def test_a_corrections_page_matches_on_its_own_words() -> None:
    assert len(content_matches(CORRECTIONS_PAGE, ObservationKind.CORRECTIONS_POLICY_PRESENT)) >= 2


def test_a_marketing_page_yields_no_supporting_phrases() -> None:
    """Phrases are quoted to a reader where they appear. They decide nothing:
    the version that gated on them reported False for a publisher that plainly
    had the page."""
    assert content_matches(MARKETING_PAGE, ObservationKind.CORRECTIONS_POLICY_PRESENT) == []


def test_matches_come_back_as_quotes_not_a_count() -> None:
    """A bare True is an assertion; a True beside the sentence it rests on is
    something a reader can check."""
    quotes = content_matches(CORRECTIONS_PAGE, ObservationKind.CORRECTIONS_POLICY_PRESENT)
    assert any("correct" in q.lower() for q in quotes)


# --- what an observation asserts -------------------------------------------


def test_a_found_policy_records_the_page_it_was_found_on() -> None:
    found = {o.kind: o for o in observe(HOMEPAGE, BASE, WHEN, fetch)}
    policy = found[ObservationKind.CORRECTIONS_POLICY_PRESENT]
    assert policy.value is True
    assert policy.source_url == "https://example.org/corrections"
    assert policy.source is Source.ABOUT_PAGE
    assert policy.evidence and "correct" in policy.evidence.lower()


def test_the_observation_is_that_a_page_was_linked_not_that_it_qualifies() -> None:
    """ "Does this page constitute an ownership statement" is a judgement about a
    named publisher, and D007 puts judgements outside this system. That a
    homepage links a page with this title at this URL is checkable: a reader
    follows the link and decides.

    The version that gated on keyword matches reported False for masthead and
    ownership on the Guardian, whose /about and /index/contributors pages are
    plainly what they claim. The patterns had been written against fixtures
    written to match them.
    """
    thin = {**PAGES, "https://example.org/about-us": "About us. We publish news."}
    found = {o.kind: o for o in observe(HOMEPAGE, BASE, WHEN, lambda u: thin.get(u, ""))}
    ownership = found[ObservationKind.OWNERSHIP_STATED]
    assert ownership.value is True
    assert ownership.source_url == "https://example.org/about-us"
    assert ownership.evidence and "About us" in ownership.evidence


def test_evidence_carries_words_a_reader_can_read() -> None:
    """Matching ran against raw HTML at first, which put a title tag and a cache
    key into a quote meant for a person."""
    html = "<html><head><title>Corrections</title><script>var k='abc123';</script></head>"
    html += "<body><p>We correct errors promptly.</p></body></html>"
    assert "<title>" not in visible_text(html)
    assert "abc123" not in visible_text(html)
    assert "We correct errors promptly." in visible_text(html)


def test_a_page_summary_leads_with_the_title() -> None:
    summary = page_summary(
        "<html><head><title>About us</title></head><body>We publish.</body></html>"
    )
    assert summary.startswith("About us")


def test_an_article_url_is_not_a_policy_page() -> None:
    """Found by running this against a real homepage: four of nine candidates
    were articles whose slug contained "about" or "ownership"."""
    assert looks_like_an_article(
        "https://example.org/fashion/2026/sep/27/armani-plays-it-safe-ownership-milan"
    )
    assert looks_like_an_article("https://example.org/a/very-long-hyphenated-story-slug-here")
    assert not looks_like_an_article("https://example.org/about")
    assert not looks_like_an_article("https://example.org/info/complaints-and-corrections")


def test_evidence_names_the_link_text_that_led_there() -> None:
    found = {o.kind: o for o in observe(HOMEPAGE, BASE, WHEN, fetch)}
    assert 'Linked as "Corrections"' in (
        found[ObservationKind.CORRECTIONS_POLICY_PRESENT].evidence or ""
    )


def test_absence_is_recorded_as_what_was_searched_not_as_a_fact_about_them() -> None:
    """The harmful direction. "This publisher has no corrections policy" is an
    assertion this project cannot establish; "no linked page on this homepage
    matched these terms" is one it can.
    """
    bare = '<html><body><a href="/news">News</a></body></html>'
    found = {o.kind: o for o in observe(bare, BASE, WHEN, fetch)}
    absent = found[ObservationKind.CORRECTIONS_POLICY_PRESENT]
    assert absent.value is False
    assert absent.source_url == BASE
    assert absent.evidence is not None
    assert "No link on this homepage matched" in absent.evidence
    assert "not the same as absent from the publication" in absent.evidence


def test_an_unfetchable_page_is_not_recorded_as_absence_of_a_policy() -> None:
    """A 503 is not evidence that a publisher has no corrections policy."""

    def broken(url: str) -> str:
        raise TimeoutError(url)

    found = {o.kind: o for o in observe(HOMEPAGE, BASE, WHEN, broken)}
    absent = found[ObservationKind.CORRECTIONS_POLICY_PRESENT]
    assert absent.value is False
    assert absent.evidence is not None
    # The observation is about the homepage that was read, not the page that
    # would not load.
    assert absent.source_url == BASE


def test_every_kind_is_observed_even_when_nothing_is_found() -> None:
    """Silence about a kind would be indistinguishable from never looking, and
    DomainRecord.latest relies on that difference."""
    found = {o.kind for o in observe("<html></html>", BASE, WHEN, fetch)}
    assert len(found) == 3


def test_no_observation_carries_a_judgement() -> None:
    """Values are True or False about a page, and the evidence quotes the page.
    Nothing here says good, poor, trustworthy or compliant."""
    judgement_words = {"good", "poor", "trustworthy", "reliable", "credible", "compliant", "fails"}
    for observation in observe(HOMEPAGE, BASE, WHEN, fetch):
        assert isinstance(observation.value, bool)
        words = set((observation.evidence or "").lower().split())
        assert not (words & judgement_words)


def test_a_masthead_and_ownership_are_found_on_their_own_pages() -> None:
    found = {o.kind: o for o in observe(HOMEPAGE, BASE, WHEN, fetch)}
    assert found[ObservationKind.MASTHEAD_PRESENT].value is True
    assert found[ObservationKind.MASTHEAD_PRESENT].source_url.endswith("/our-team")
    assert found[ObservationKind.OWNERSHIP_STATED].value is True
    assert found[ObservationKind.OWNERSHIP_STATED].source_url.endswith("/about-us")


def test_evidence_stays_within_the_schema_bound() -> None:
    long_page = "We correct errors. " * 400
    pages = {**PAGES, "https://example.org/corrections": long_page}
    found = observe(HOMEPAGE, BASE, WHEN, lambda u: pages.get(u, ""))
    for observation in found:
        assert observation.evidence is None or len(observation.evidence) <= 500
