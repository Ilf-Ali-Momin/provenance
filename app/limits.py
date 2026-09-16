"""Rate limiting. SECURITY.md control 5.

Two limits with different purposes, and the difference decides their shape.

**Per key** protects this service. It is scoped to the caller, and the URL fetch
path carries its own tighter limit because it is the expensive path and the one
that generates outbound traffic attributable to our infrastructure.

**Per target domain** protects *publishers*, and applies "regardless of which
key requests them". That clause is the whole control: a limit scoped per key
lets N keys generate N times the load against one newspaper, which is the
distributed load generator the control exists to prevent. So the domain counter
is global.

**Sliding window, not fixed.** A fixed window permits 2N in an instant across a
boundary — N at 10:59:59 and N at 11:00:00. Against a publisher, that burst is
exactly the behaviour that earns a blocklist entry, which is the stated reason
this control exists. A sorted set of timestamps costs one more Redis operation
and does not have that edge.

**The domain is the registrable domain, via the public suffix list.**
ARCHITECTURE.md requires that rather than naive string splitting, "since
subdomain handling is a common source of both bugs and evasion". Here the
evasion is concrete: without it, `a.example.com`, `b.example.com` and so on are
unlimited requests against one publisher's servers.
"""

from __future__ import annotations

import time
from typing import Final, Protocol
from urllib.parse import urlsplit

from pipeline.errors import ProvenanceError
from tld import get_fld
from tld.exceptions import TldBadUrl, TldDomainNotFound

from app.config import AppSettings

KEY_SCOPE: Final = "rl:key"
FETCH_SCOPE: Final = "rl:fetch"
DOMAIN_SCOPE: Final = "rl:domain"


class RateLimitedError(ProvenanceError):
    code = "rate_limited"
    http_status = 429


class DomainRateLimitedError(ProvenanceError):
    code = "domain_rate_limited"
    http_status = 429


class Window(Protocol):
    """Storage for a sliding window of event timestamps."""

    def hits(self, key: str, since: float) -> list[float]: ...

    def record(self, key: str, at: float, ttl_s: int) -> None: ...


class MemoryWindow:
    """In process window. Used by tests, and by a single process deployment."""

    def __init__(self) -> None:
        self._events: dict[str, list[float]] = {}

    def hits(self, key: str, since: float) -> list[float]:
        kept = [t for t in self._events.get(key, []) if t >= since]
        self._events[key] = kept
        return kept

    def record(self, key: str, at: float, ttl_s: int) -> None:
        self._events.setdefault(key, []).append(at)


class RedisWindow:
    """Sorted set per key: score and member are both the timestamp."""

    def __init__(self, redis: object) -> None:
        self._redis = redis

    def hits(self, key: str, since: float) -> list[float]:
        pipe = self._redis.pipeline()  # type: ignore[attr-defined]
        pipe.zremrangebyscore(key, "-inf", since)
        pipe.zrange(key, 0, -1, withscores=True)
        _, rows = pipe.execute()
        return [score for _member, score in rows]

    def record(self, key: str, at: float, ttl_s: int) -> None:
        pipe = self._redis.pipeline()  # type: ignore[attr-defined]
        pipe.zadd(key, {str(at): at})
        pipe.expire(key, ttl_s)
        pipe.execute()


def registrable_domain(url: str) -> str:
    """The domain a publisher actually is, for limiting purposes.

    Falls back to the host for anything the public suffix list cannot place,
    which is an IP literal or an unknown suffix. Falling back to the host is the
    conservative direction: it limits more narrowly than intended rather than
    letting an unrecognised host escape the counter entirely.
    """
    try:
        return str(get_fld(url))
    except (TldDomainNotFound, TldBadUrl, ValueError):
        return urlsplit(url).netloc.lower() or url


class RateLimiter:
    def __init__(self, window: Window, settings: AppSettings | None = None) -> None:
        self._window = window
        self._settings = settings or AppSettings()

    def _check(self, key: str, limit: int, period_s: int, error: type[ProvenanceError]) -> None:
        now = time.time()
        hits = self._window.hits(key, now - period_s)
        if len(hits) >= limit:
            retry_after = max(1, int(period_s - (now - min(hits))) + 1)
            raise error(f"{limit} requests per {period_s}s exceeded; retry after {retry_after}s")
        self._window.record(key, now, period_s)

    def check_key(self, api_key: str) -> None:
        self._check(
            f"{KEY_SCOPE}:{api_key}",
            self._settings.rate_limit_per_key,
            self._settings.rate_limit_period_s,
            RateLimitedError,
        )

    def check_fetch(self, api_key: str) -> None:
        """The tighter limit on the expensive path (control 5)."""
        self._check(
            f"{FETCH_SCOPE}:{api_key}",
            self._settings.fetch_limit_per_key,
            self._settings.rate_limit_period_s,
            RateLimitedError,
        )

    def check_domain(self, url: str) -> None:
        """Global, not per key. That is the point of the control."""
        domain = registrable_domain(url)
        self._check(
            f"{DOMAIN_SCOPE}:{domain}",
            self._settings.fetches_per_domain_per_hour,
            self._settings.domain_period_s,
            DomainRateLimitedError,
        )
