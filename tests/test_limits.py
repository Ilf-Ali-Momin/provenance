"""Rate limiting. SECURITY.md control 5. No Redis, no clock patching games."""

from __future__ import annotations

import pytest
from app.config import AppSettings
from app.limits import (
    DomainRateLimitedError,
    MemoryWindow,
    RateLimitedError,
    RateLimiter,
    registrable_domain,
)


def _limiter(**overrides: object) -> RateLimiter:
    base = {
        "rate_limit_per_key": 3,
        "fetch_limit_per_key": 2,
        "rate_limit_period_s": 60,
        "fetches_per_domain_per_hour": 2,
        "domain_period_s": 3600,
    }
    return RateLimiter(MemoryWindow(), AppSettings(**{**base, **overrides}))  # type: ignore[arg-type]


def test_a_key_is_limited_after_its_allowance() -> None:
    limiter = _limiter()
    for _ in range(3):
        limiter.check_key("k1")
    with pytest.raises(RateLimitedError):
        limiter.check_key("k1")


def test_one_key_does_not_consume_another_keys_allowance() -> None:
    limiter = _limiter()
    for _ in range(3):
        limiter.check_key("k1")
    limiter.check_key("k2")


def test_the_fetch_path_carries_its_own_tighter_limit() -> None:
    """Control 5: separate, tighter limits on the expensive path."""
    limiter = _limiter()
    limiter.check_fetch("k1")
    limiter.check_fetch("k1")
    with pytest.raises(RateLimitedError):
        limiter.check_fetch("k1")
    # The general allowance is untouched by the fetch counter.
    limiter.check_key("k1")


def test_the_domain_limit_ignores_which_key_asked() -> None:
    """The clause that makes this control work: 'regardless of which key'."""
    limiter = _limiter()
    limiter.check_domain("https://example.com/a")
    limiter.check_domain("https://example.com/b")
    with pytest.raises(DomainRateLimitedError):
        limiter.check_domain("https://example.com/c")


def test_subdomains_cannot_evade_the_domain_limit() -> None:
    """Without the public suffix list this is an unlimited fetch loop."""
    limiter = _limiter()
    limiter.check_domain("https://a.example.com/1")
    limiter.check_domain("https://b.example.com/2")
    with pytest.raises(DomainRateLimitedError):
        limiter.check_domain("https://c.example.com/3")


def test_a_multi_part_suffix_is_one_publisher_not_two() -> None:
    limiter = _limiter()
    limiter.check_domain("https://news.bbc.co.uk/1")
    limiter.check_domain("https://www.bbc.co.uk/2")
    with pytest.raises(DomainRateLimitedError):
        limiter.check_domain("https://bbc.co.uk/3")


def test_separate_publishers_have_separate_allowances() -> None:
    limiter = _limiter()
    limiter.check_domain("https://example.com/a")
    limiter.check_domain("https://example.com/b")
    limiter.check_domain("https://other.org/a")


def test_the_error_names_a_retry_delay() -> None:
    limiter = _limiter()
    limiter.check_domain("https://example.com/a")
    limiter.check_domain("https://example.com/b")
    with pytest.raises(DomainRateLimitedError, match=r"retry after \d+s"):
        limiter.check_domain("https://example.com/c")


def test_the_window_slides_rather_than_resetting_on_a_boundary() -> None:
    """A fixed window permits 2N across a boundary; that burst is the failure."""
    window = MemoryWindow()
    limiter = RateLimiter(window, AppSettings(fetches_per_domain_per_hour=2, domain_period_s=10))
    limiter.check_domain("https://example.com/a")
    limiter.check_domain("https://example.com/b")
    # Age the recorded hits past the window instead of sleeping.
    key = next(iter(window._events))
    window._events[key] = [t - 11 for t in window._events[key]]
    limiter.check_domain("https://example.com/c")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://news.bbc.co.uk/a", "bbc.co.uk"),
        ("https://a.b.c.example.com/x", "example.com"),
        ("https://example.com", "example.com"),
    ],
)
def test_registrable_domain(url: str, expected: str) -> None:
    assert registrable_domain(url) == expected


def test_an_unplaceable_host_falls_back_to_the_host_itself() -> None:
    """Conservative direction: limit too narrowly rather than not at all."""
    assert registrable_domain("http://192.0.2.1/z") == "192.0.2.1"
