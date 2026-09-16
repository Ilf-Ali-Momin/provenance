"""The worker job. Stubs for robots, fetch and extract; no network."""

from __future__ import annotations

import pytest
from pipeline import jobs as mod
from pipeline.errors import UpstreamError
from pipeline.jobs import FetchOutcome, FetchRequest, analysis_id_for, fetch_and_extract
from pipeline.net.fetch import FetchResult
from pipeline.text.extract import ExtractedArticle

ARTICLE = "The committee met to consider the budget. " * 30
REQUEST = FetchRequest(analysis_id="an_test", url="https://example.com/a")


class StubRobots:
    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.waited = False

    def allows(self, url: str) -> bool:
        return self.allow

    def wait_for(self, url: str) -> None:
        self.waited = True

    def sitemaps_for(self, url: str) -> list[str]:
        return []


def _install(monkeypatch: pytest.MonkeyPatch, article: ExtractedArticle | None) -> None:
    monkeypatch.setattr(
        mod,
        "fetch",
        lambda url, **kw: FetchResult(
            final_url=url, status_code=200, content_type="text/html", text="<html></html>"
        ),
    )
    monkeypatch.setattr(mod, "extract_article", lambda *a, **k: article)


def _article() -> ExtractedArticle:
    return ExtractedArticle(
        text=ARTICLE,
        title="Budget",
        published="2026-03-01",
        removed={"byline": 1},
        had_invisible=False,
        declared_language="en",
        english_ratio=0.4,
    )


def test_the_outcome_carries_a_hash_and_never_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """RQ writes the return value to Redis, which is storage (D032)."""
    _install(monkeypatch, _article())
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]

    assert ARTICLE not in outcome.model_dump_json()
    assert outcome.content_sha256 is not None
    assert outcome.characters == len(ARTICLE)


def test_the_result_model_has_no_field_that_could_hold_text() -> None:
    """Enforced by the schema, the same way API.md keeps a verdict out."""
    forbidden = {"text", "content", "body", "html", "article"}
    assert forbidden.isdisjoint(FetchOutcome.model_fields)


def test_a_robots_refusal_is_an_outcome_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _article())
    outcome = fetch_and_extract(REQUEST, robots=StubRobots(allow=False))  # type: ignore[arg-type]
    assert outcome.status == "failed"
    assert outcome.error_code == "url_rejected"


def test_the_crawl_delay_is_honoured_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _article())
    robots = StubRobots()
    fetch_and_extract(REQUEST, robots=robots)  # type: ignore[arg-type]
    assert robots.waited is True


def test_a_fetch_failure_is_recorded_with_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **kwargs: object) -> FetchResult:
        raise UpstreamError("unreachable")

    monkeypatch.setattr(mod, "fetch", boom)
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]
    assert outcome.status == "failed"
    assert outcome.error_code == "upstream_error"


def test_a_page_with_no_article_is_reported_rather_than_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, None)
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]
    assert outcome.error_code == "no_article_text"
    assert outcome.http_status == 200


def test_the_analysis_id_is_stable_for_a_url() -> None:
    assert analysis_id_for("https://example.com/a  ") == analysis_id_for("https://example.com/a")
    assert analysis_id_for("https://example.com/a").startswith("an_")
    assert analysis_id_for("https://example.com/b") != analysis_id_for("https://example.com/a")


def test_the_queue_payload_is_a_model_not_a_dict() -> None:
    """RULES.md: pydantic across every process boundary, including queue payloads."""
    assert FetchRequest.model_validate({"analysis_id": "x", "url": "https://e.com"}).url
    with pytest.raises(ValueError, match="analysis_id"):
        FetchRequest.model_validate({"url": "https://e.com"})


# --- claim extraction inside the job (M2) -----------------------------------


def test_claims_reach_the_outcome_with_their_extractor_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATA.md permits derived claims in storage; this is that exception."""
    _install(monkeypatch, _article())
    monkeypatch.setattr(
        mod,
        "extract_claims",
        lambda text: (
            [
                mod.Claim(
                    claim_id="cl_1",
                    text="The committee met",
                    start=0,
                    end=17,
                    type="factual_assertion",
                    extractor_version="v1:abc",
                )
            ],
            {"extractor_version": "v1:abc"},
        ),
    )
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]
    assert len(outcome.claims) == 1
    assert outcome.extractor_version == "v1:abc"
    assert outcome.extractor_error is None


def test_an_unavailable_extractor_degrades_to_the_rule_based_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D004: the pipeline degrades rather than fails, and says which ran."""
    from pipeline.claims.extract import ExtractorUnavailableError

    # A passage the rule based extractor can actually work on. The default
    # fixture is one sentence repeated, with no number and no mid-sentence
    # proper noun, so the fallback correctly finds nothing in it - which makes
    # it the wrong text for testing that the fallback ran.
    findable = ExtractedArticle(
        text="Some 2,400 homes were flooded across three counties on Tuesday.",
        title="Floods",
        published="2026-03-01",
        removed={},
        had_invisible=False,
        declared_language="en",
        english_ratio=0.4,
    )
    _install(monkeypatch, findable)

    def unavailable(text: str) -> tuple[list, dict]:
        raise ExtractorUnavailableError("HTTP 503")

    monkeypatch.setattr(mod, "extract_claims", unavailable)
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]

    assert outcome.status == "complete"
    assert outcome.extractor_error == "extractor_unavailable"
    assert outcome.extractor_version == mod.FALLBACK_VERSION
    assert outcome.claims, "the fallback should find the numeric claim here"


def test_the_outcome_still_cannot_hold_the_article(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claims are spans. The document is not a span (D039)."""
    _install(monkeypatch, _article())
    monkeypatch.setattr(mod, "extract_claims", lambda text: ([], {"extractor_version": "v1:a"}))
    outcome = fetch_and_extract(REQUEST, robots=StubRobots())  # type: ignore[arg-type]
    assert ARTICLE not in outcome.model_dump_json()
