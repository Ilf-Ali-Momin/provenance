"""robots.txt handling. Fixture driven, no network."""

from __future__ import annotations

import pytest
from pipeline.net.robots import RobotsCache

pytestmark = pytest.mark.adversarial

PUBLIC = "93.184.216.34"

ALLOW_ALL = "User-agent: *\nDisallow:\n"
BLOCK_NEWS = "User-agent: *\nDisallow: /news/\n"
BLOCK_ALL = "User-agent: *\nDisallow: /\n"
WITH_DELAY = "User-agent: *\nDisallow:\nCrawl-delay: 9\n"


def test_disallowed_path_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, BLOCK_NEWS)
    cache = RobotsCache(min_delay_s=0.0)
    assert cache.allows("https://example.com/sport/x") is True
    assert cache.allows("https://example.com/news/x") is False


def test_allow_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, ALLOW_ALL)
    cache = RobotsCache(min_delay_s=0.0)
    assert cache.allows("https://example.com/anything") is True


def test_block_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, BLOCK_ALL)
    cache = RobotsCache(min_delay_s=0.0)
    assert cache.allows("https://example.com/") is False


def test_unfetchable_robots_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decision that matters: failure closes, it does not open.

    A host refusing connections is not a host inviting a crawler.
    """
    from pipeline.errors import UpstreamError
    from pipeline.net import robots as mod

    def boom(url: str, **kwargs: object) -> object:
        raise UpstreamError("unreachable")

    monkeypatch.setattr(mod, "fetch", boom)
    cache = RobotsCache(min_delay_s=0.0)
    assert cache.allows("https://example.com/anything") is False


def test_declared_crawl_delay_is_honoured_when_larger(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, WITH_DELAY)
    cache = RobotsCache(min_delay_s=2.0)
    assert cache.policy_for("https://example.com/").delay_s == 9.0


def test_our_floor_applies_when_host_declares_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, ALLOW_ALL)
    cache = RobotsCache(min_delay_s=2.0)
    assert cache.policy_for("https://example.com/").delay_s == 2.0


def test_policy_is_fetched_once_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, ALLOW_ALL)
    cache = RobotsCache(min_delay_s=0.0)
    for path in ("/a", "/b", "/c"):
        cache.allows(f"https://example.com{path}")
    assert calls["n"] == 1, "robots.txt must not be refetched per URL"


def _install(monkeypatch: pytest.MonkeyPatch, body: str) -> dict[str, int]:
    """Replace the module's fetch with one returning a fixed robots body."""
    from pipeline.net import robots as mod
    from pipeline.net.fetch import FetchResult

    calls = {"n": 0}

    def fake(url: str, **kwargs: object) -> FetchResult:
        calls["n"] += 1
        return FetchResult(final_url=url, status_code=200, content_type="text/plain", text=body)

    monkeypatch.setattr(mod, "fetch", fake)
    return calls
