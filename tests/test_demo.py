"""The demonstration CLI. No network."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pipeline import demo as mod
from pipeline.errors import UpstreamError, UrlRejectedError
from pipeline.net.fetch import FetchResult

BODY = " ".join(f"The committee met on day {i} to consider the budget." for i in range(20))
HTML = f'<html lang="en"><body><article><h1>Budget</h1><p>{BODY}</p></article></body></html>'


class StubPolicy:
    fetched = True
    delay_s = 2.0
    sitemaps: ClassVar[list[str]] = []


class StubRobots:
    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.waited = False

    def allows(self, url: str) -> bool:
        return self.allow

    def wait_for(self, url: str) -> None:
        self.waited = True

    def policy_for(self, url: str) -> StubPolicy:
        return StubPolicy()


def _install(monkeypatch: pytest.MonkeyPatch, html: str = HTML, status: int = 200) -> list[str]:
    fetched: list[str] = []

    def fake_fetch(url: str, **kwargs: object) -> FetchResult:
        fetched.append(url)
        return FetchResult(final_url=url, status_code=status, content_type="text/html", text=html)

    monkeypatch.setattr(mod, "fetch", fake_fetch)
    return fetched


def test_reports_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    report = mod.inspect("https://example.com/a", robots=StubRobots())  # type: ignore[arg-type]

    assert report["fetch"]["status"] == 200  # type: ignore[index]
    assert report["extract"]["article"] is True  # type: ignore[index]
    assert report["extract"]["declared_language"] == "en"  # type: ignore[index]
    assert "committee met" in str(report["text"])


def test_robots_refusal_happens_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """No override flag exists, and none should."""
    fetched = _install(monkeypatch)
    with pytest.raises(UrlRejectedError):
        mod.inspect("https://example.com/a", robots=StubRobots(allow=False))  # type: ignore[arg-type]
    assert fetched == []


def test_crawl_delay_is_honoured_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    robots = StubRobots()
    mod.inspect("https://example.com/a", robots=robots)  # type: ignore[arg-type]
    assert robots.waited is True


def test_outlet_names_are_masked_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, HTML.replace("Budget", "Budget at The Example Times"))
    report = mod.inspect(
        "https://example.com/a",
        outlet_names=("The Example Times",),
        robots=StubRobots(),  # type: ignore[arg-type]
    )
    assert "The Example Times" not in str(report["text"])
    assert report["removed"]["outlet_name"] == 1  # type: ignore[index]


def test_page_without_an_article_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "<html><body><nav>Menu</nav></body></html>")
    report = mod.inspect("https://example.com/index", robots=StubRobots())  # type: ignore[arg-type]
    assert report["extract"]["article"] is False  # type: ignore[index]
    assert "paywall" in str(report["extract"]["why"])  # type: ignore[index]


def test_cli_renders_text(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(mod, "RobotsCache", StubRobots)
    assert mod.main(["https://example.com/a"]) == mod.EXIT_OK
    out = capsys.readouterr().out
    assert "robots.txt" in out and "HTTP 200" in out and "committee met" in out


def test_cli_json_is_parseable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(mod, "RobotsCache", StubRobots)
    assert mod.main(["https://example.com/a", "--json"]) == mod.EXIT_OK
    assert json.loads(capsys.readouterr().out)["fetch"]["status"] == 200


def test_cli_exit_codes_separate_refusal_from_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(mod, "RobotsCache", lambda: StubRobots(allow=False))
    assert mod.main(["https://example.com/a"]) == mod.EXIT_REFUSED
    assert "refused" in capsys.readouterr().err

    def boom(url: str, **kwargs: object) -> FetchResult:
        raise UpstreamError("the remote host could not be reached")

    monkeypatch.setattr(mod, "RobotsCache", StubRobots)
    monkeypatch.setattr(mod, "fetch", boom)
    assert mod.main(["https://example.com/a"]) == mod.EXIT_ERROR
    assert "upstream_error" in capsys.readouterr().err
