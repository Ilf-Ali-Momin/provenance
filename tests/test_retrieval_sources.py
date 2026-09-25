"""The Wikimedia sources, and the policy they refuse to breach."""

from __future__ import annotations

from datetime import date

import pytest
from pipeline.net.config import FetchSettings
from pipeline.retrieval.sources import (
    LICENCES,
    PLACEHOLDER_TOKENS,
    RESERVED_SUFFIXES,
    WIKINEWS,
    WIKIPEDIA,
    SourceUnavailableError,
    WikimediaSource,
    attribution_for,
    check_user_agent,
)

# A syntactically ordinary contact. Nothing here dials out, but the string has
# to pass the guard the way a real deployment's would.
REAL_AGENT = "ProvenanceBot/0.1 (+https://provenance-bot.dev/bot; bot@provenance-bot.dev)"


def _settings(agent: str = REAL_AGENT) -> FetchSettings:
    return FetchSettings(user_agent=agent)


def _payload(pages: list[dict[str, object]]) -> dict[str, object]:
    return {"query": {"pages": pages}}


def _source(pages: list[dict[str, object]], host: str = WIKIPEDIA) -> WikimediaSource:
    return WikimediaSource(
        host,
        ["Anything"],
        fetch_json=lambda url: _payload(pages),
        settings=_settings(),
        snapshot_date=date(2026, 9, 25),
        sleep=lambda _s: None,
    )


def test_a_reserved_domain_is_refused() -> None:
    """RFC 2606 and RFC 6761 names never resolve; the complaint reaches no one."""
    with pytest.raises(SourceUnavailableError, match="nobody can reach"):
        check_user_agent(_settings("ProvenanceBot/0.1 (+https://example.invalid/bot)"))


@pytest.mark.parametrize(
    "agent",
    [
        "ProvenanceBot/0.1 (+https://your-site/bot; you@your-domain)",
        "ProvenanceBot/0.1 (+https://yourdomain.com/bot)",
        "ProvenanceBot/0.1 (+https://changeme/bot)",
        "ProvenanceBot/0.1 (TODO: add contact)",
        "ProvenanceBot/0.1 (+https://bot.test/x)",
        "ProvenanceBot/0.1 (+https://bot.localhost/x)",
    ],
)
def test_template_text_left_in_place_is_refused(agent: str) -> None:
    """The gap the first guard had.

    It listed only example.*, so `https://your-site/bot; you@your-domain` — this
    project's own placeholder — passed it, and Wikimedia answered 200. The
    remote host does not catch this: a well formed agent naming a nonexistent
    domain is served normally, so the check has to be here or nowhere.
    """
    with pytest.raises(SourceUnavailableError, match="nobody can reach"):
        check_user_agent(_settings(agent))


def test_a_host_merely_containing_a_reserved_word_is_allowed() -> None:
    """`.test` is reserved; `testing-news.org` is somebody's actual domain."""
    check_user_agent(_settings("ProvenanceBot/0.1 (+https://testing-news.org/bot)"))


def test_the_error_names_what_it_objected_to() -> None:
    with pytest.raises(SourceUnavailableError, match="your-domain"):
        check_user_agent(_settings("ProvenanceBot/0.1 (+https://ok.org; you@your-domain)"))


def test_the_guard_covers_both_families() -> None:
    assert ".invalid" in RESERVED_SUFFIXES
    assert "your-site" in PLACEHOLDER_TOKENS


def test_the_projects_current_default_is_the_one_that_is_refused() -> None:
    """D030's open row. The obligation now blocks something."""
    with pytest.raises(SourceUnavailableError):
        check_user_agent(FetchSettings())


def test_a_real_contact_passes() -> None:
    check_user_agent(_settings())


def test_construction_checks_the_agent_before_any_request() -> None:
    with pytest.raises(SourceUnavailableError):
        WikimediaSource(WIKIPEDIA, ["Anything"], settings=FetchSettings())


def test_an_unknown_wiki_is_refused_rather_than_guessed() -> None:
    """Indexing text whose licence nobody read is the failure D046 prevents."""
    with pytest.raises(SourceUnavailableError, match="no licence recorded"):
        WikimediaSource("en.wikiquote.org", [], settings=_settings())


def test_the_recorded_licences_are_the_ones_the_wikis_declare() -> None:
    """Read from each wiki's siteinfo rightsinfo on 2026-09-25, not from memory."""
    assert LICENCES[WIKIPEDIA][0] == "CC-BY-SA-4.0"
    assert LICENCES[WIKINEWS][0] == "CC-BY-4.0"


def test_a_page_becomes_a_document_carrying_its_attribution() -> None:
    pages = [{"pageid": 42, "title": "Bangsamoro", "extract": "Some reference prose."}]
    documents = list(_source(pages).documents())
    assert len(documents) == 1
    document = documents[0]
    assert document.document_id == "reference_en_wikipedia_org:42"
    assert document.redistribution == "permitted_with_attribution"
    assert "CC-BY-SA-4.0" in (document.attribution or "")
    assert "Bangsamoro" in (document.attribution or "")
    assert document.source_url == "https://en.wikipedia.org/wiki/Bangsamoro"


def test_spaces_in_a_title_become_underscores_in_the_url() -> None:
    pages = [{"pageid": 7, "title": "Bangsamoro Autonomous Region", "extract": "Prose."}]
    document = next(iter(_source(pages).documents()))
    assert document.source_url.endswith("/wiki/Bangsamoro_Autonomous_Region")


def test_a_missing_page_yields_nothing() -> None:
    assert list(_source([{"title": "Nope", "missing": True}]).documents()) == []


def test_a_page_with_no_extract_yields_nothing() -> None:
    """An empty document would index as a chunkless row and count as coverage."""
    assert list(_source([{"pageid": 1, "title": "Stub", "extract": "  "}]).documents()) == []


def test_a_malformed_response_yields_nothing_rather_than_raising() -> None:
    source = WikimediaSource(
        WIKIPEDIA,
        ["Anything"],
        fetch_json=lambda url: {},
        settings=_settings(),
        sleep=lambda _s: None,
    )
    assert list(source.documents()) == []


def test_one_page_is_requested_per_call_because_the_api_caps_extracts() -> None:
    """A 50-title batch builds a corpus missing 98% of its text.

    `prop=extracts` with `explaintext` lowers `exlimit` to 1 for whole-article
    extracts: the other 49 pages come back with an empty extract and the cap is
    reported only in a `warnings` block. Every one of them then looks like a
    page that legitimately had no prose. Measured against en.wikinews.org on
    2026-09-25.
    """
    urls: list[str] = []

    def spy(url: str) -> dict[str, object]:
        urls.append(url)
        return _payload([])

    source = WikimediaSource(
        WIKIPEDIA,
        [f"Page {n}" for n in range(5)],
        fetch_json=spy,
        settings=_settings(),
        sleep=lambda _s: None,
    )
    list(source.documents())
    assert len(urls) == 5
    assert all("titles=Page" in url for url in urls)


def test_the_extract_limit_is_stated_in_the_request() -> None:
    """So raising MAX_TITLES_PER_REQUEST cannot silently lose the corpus."""
    urls: list[str] = []
    source = WikimediaSource(
        WIKIPEDIA,
        ["Anything"],
        fetch_json=lambda url: (urls.append(url), _payload([]))[1],
        settings=_settings(),
        sleep=lambda _s: None,
    )
    list(source.documents())
    assert "exlimit=1" in urls[0]


def test_requests_are_spaced_rather_than_issued_back_to_back() -> None:
    """DATA.md: well under anything that would look like scraping pressure."""
    waits: list[float] = []
    source = WikimediaSource(
        WIKIPEDIA,
        ["A", "B", "C"],
        fetch_json=lambda url: _payload([]),
        settings=_settings(),
        sleep=waits.append,
    )
    list(source.documents())
    assert len(waits) == 2
    assert all(w > 0 for w in waits)


def test_the_wikinews_attribution_names_wikinews_and_its_licence() -> None:
    line = attribution_for(WIKINEWS, "Voting begins", "https://en.wikinews.org/wiki/Voting_begins")
    assert "Wikinews" in line
    assert "CC-BY-4.0" in line


def test_the_corpus_id_is_stable_and_derived_from_the_host() -> None:
    assert _source([]).corpus_id == "reference_en_wikipedia_org"
