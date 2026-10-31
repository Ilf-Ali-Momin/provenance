"""Sampling posts about a URL from Bluesky's public API.

Bluesky is first among the two samplers M4 asks for because its AppView is
readable without credentials: `api.bsky.app` serves the same search and profile
endpoints an app would use, with no token and no developer agreement.
That makes it the one social source this project can sample in CI, in a test,
and on a laptop without a secret in the environment.

**Nothing here stores a post.** The sampler hands back `SampledPost` values,
which `aggregate.aggregate` turns into counts; both live in memory for as long
as one call takes. The reason is written up in `aggregate`, and this module is
deliberately on the safe side of that boundary: it has no database handle, no
file handle, and no cache.

**The transport is injected**, the same arrangement `registry.rdap.lookup`
uses. The caller owns the HTTP client, its timeouts, its rate limiting and its
user agent, so everything in this file is testable against recorded payloads
and the one thing that touches a network stays in the caller's hands.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, Final, NamedTuple
from urllib.parse import urlencode

import structlog

from pipeline.circulation.aggregate import SampledPost
from pipeline.circulation.config import BlueskySettings
from pipeline.errors import UpstreamError
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import fetch

logger = structlog.get_logger(__name__)

# The unauthenticated AppView. The same lexicon methods an authenticated client
# calls, served without a token.
#
# `api.bsky.app` rather than the more widely documented `public.api.bsky.app`,
# because the two do not serve the same methods. Checked on 2026-10-31:
# `public.api.bsky.app` answers `app.bsky.actor.getProfiles` with 200 and JSON
# but refuses `app.bsky.feed.searchPosts` with a 403 HTML block page, while
# `api.bsky.app` answers both. Search is the method this module is for, so the
# host that serves it is the host used.
API_BASE: Final = "https://api.bsky.app/xrpc"
SEARCH_METHOD: Final = "app.bsky.feed.searchPosts"
PROFILES_METHOD: Final = "app.bsky.actor.getProfiles"

# The AppView answers in JSON, which the article fetch path does not accept.
# Widened here rather than in `FetchSettings`, the same way RDAP and feed
# discovery widen it: the article path is narrow because a fetch returning a
# PDF or an executable must never reach extraction, and that reason does not
# stop applying because this API exists.
#
# `text/html` is in the set only because of the order of checks inside
# `net.fetch`: it gates the content type before the caller ever sees the status
# code, and the AppView's refusals are HTML pages. Without HTML accepted, a 403
# arrives as "not an accepted type" and a blocked endpoint is indistinguishable
# from a malformed one - which is exactly how the paging limit below was nearly
# missed. Nothing is loosened by it in practice: `default_transport` requires a
# 2xx status and a JSON object, which is a narrower gate than the media type.
API_CONTENT_TYPES: Final = frozenset(
    {"application/json", "application/json; charset=utf-8", "text/html"}
)


def api_settings(settings: FetchSettings | None = None) -> FetchSettings:
    """Fetch settings that accept the AppView's media type and nothing more."""
    base = settings or FetchSettings()
    return base.model_copy(update={"allowed_content_types": frozenset(API_CONTENT_TYPES)})


def default_transport(
    settings: FetchSettings | None = None,
) -> Callable[[str], dict[str, Any]]:
    """A transport over the project's guarded fetcher.

    Routed through `net.fetch` rather than a bare HTTP call so that this path
    carries the same controls as every other outbound request: scheme and port
    checks, address validation, per-hop redirect revalidation and the response
    size caps. A fixed public host does not earn an exemption from them, and an
    unchecked second HTTP path is how one appears later.
    """
    resolved = api_settings(settings)

    def transport(url: str) -> dict[str, Any]:
        result = fetch(url, settings=resolved)
        # Checked here because `net.fetch` returns a non-2xx response's body
        # rather than raising: it is a fetcher, and some callers want the body
        # of an error page. An API client does not. Without this, a refusal
        # surfaces as whatever parsing its HTML happens to fail on, which is
        # how a blocked endpoint gets mistaken for a malformed one.
        if not 200 <= result.status_code < 300:
            msg = f"Bluesky answered {result.status_code} for {url.split('?')[0]}."
            raise UpstreamError(msg)

        body = result.text
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise UpstreamError("Bluesky returned a body that is not JSON.") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("Bluesky returned JSON that is not an object.")
        return payload

    return transport


class Sample(NamedTuple):
    """Posts from one run, and how the run ended.

    The second field is the point of the type. Unauthenticated search serves
    one page, so a run against a widely shared URL stops at 100 posts while the
    settings permit 500, and the shortfall is invisible in the posts alone. The
    three measures built on a sample are proportions rather than volumes, which
    survives truncation, but a caller still has to know a sample was cut short
    before describing it to a reader as how a story spread.

    `stop_reason` is one of:

    * `cursor_exhausted` - upstream said there is nothing more. The only value
      that means the sample is everything the search could offer.
    * `post_cap`, `page_cap` - our own limits stopped it.
    * `upstream_refused` - a later page failed. Pages already collected are
      kept; see the paging note on `sample`.
    """

    posts: list[SampledPost]
    pages: int
    stop_reason: str


def search_url(query: str, *, limit: int, cursor: str | None = None) -> str:
    """The search URL for one page of results."""
    params: dict[str, str] = {"q": query, "limit": str(limit)}
    if cursor:
        params["cursor"] = cursor
    return f"{API_BASE}/{SEARCH_METHOD}?{urlencode(params)}"


def profiles_url(actors: list[str]) -> str:
    """The profile URL for a batch of accounts."""
    return f"{API_BASE}/{PROFILES_METHOD}?{urlencode([('actors', a) for a in actors])}"


def _timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_posts(payload: dict[str, Any]) -> tuple[list[SampledPost], str | None]:
    """One page of search results, and the cursor for the next page.

    Two timestamps are available on every post and they are not equally
    trustworthy. `record.createdAt` is written by whoever posted, so a
    coordinated set of accounts can backdate it and spread a burst across a
    fake week; `indexedAt` is when the service saw the post and the poster
    cannot set it. Time clustering is one of the three measures, so this takes
    `indexedAt` and falls back to the claimed time only when the service did
    not report one.

    A post with no usable timestamp at all is dropped rather than given a
    guessed one. Inventing a time would put a fabricated point into a
    distribution this project then shows a reader.
    """
    posts: list[SampledPost] = []
    for entry in payload.get("posts", []) or []:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        record = entry.get("record")
        if not isinstance(author, dict) or not isinstance(record, dict):
            continue
        handle = str(author.get("did") or author.get("handle") or "")
        text = record.get("text")
        if not handle or not isinstance(text, str):
            continue
        posted_at = _timestamp(entry.get("indexedAt")) or _timestamp(record.get("createdAt"))
        if posted_at is None:
            continue
        posts.append(
            SampledPost(
                author=handle,
                # Search results carry the account's creation date only
                # sometimes. None means "we did not learn it", which
                # `aggregate` counts separately from a known age, and
                # `ages_by_actor` can fill in with a second request.
                author_created_at=_timestamp(author.get("createdAt")),
                posted_at=posted_at,
                text=text,
            )
        )
    cursor = payload.get("cursor")
    return posts, cursor if isinstance(cursor, str) and cursor else None


def ages_by_actor(payload: dict[str, Any]) -> dict[str, datetime]:
    """Account creation dates from a profile batch, keyed by DID and handle.

    Keyed both ways because search results identify an author by DID and a
    caller may hold either. An account whose profile reports no creation date
    is absent from the mapping rather than present with a placeholder.
    """
    ages: dict[str, datetime] = {}
    for profile in payload.get("profiles", []) or []:
        if not isinstance(profile, dict):
            continue
        created = _timestamp(profile.get("createdAt"))
        if created is None:
            continue
        for key in (profile.get("did"), profile.get("handle")):
            if isinstance(key, str) and key:
                ages[key] = created
    return ages


def _fill_ages(
    posts: list[SampledPost],
    fetch_json: Callable[[str], dict[str, Any]],
    settings: BlueskySettings,
) -> list[SampledPost]:
    """Look up the accounts whose age the search did not report.

    A failed profile batch leaves those posts with an unknown age. That is the
    truthful outcome: the account is not new, it is unmeasured, and
    `CirculationStats.accounts_with_unknown_age` says so separately from the
    age distribution.
    """
    unknown = sorted({p.author for p in posts if p.author_created_at is None})
    if not unknown:
        return posts

    ages: dict[str, datetime] = {}
    for start in range(0, len(unknown), settings.profile_batch):
        batch = unknown[start : start + settings.profile_batch]
        try:
            ages.update(ages_by_actor(fetch_json(profiles_url(batch))))
        except Exception as exc:
            logger.warning(
                "bluesky_profile_batch_failed",
                accounts=len(batch),
                error=type(exc).__name__,
            )

    return [
        p if p.author_created_at is not None else p._replace(author_created_at=ages.get(p.author))
        for p in posts
    ]


def sample(
    query: str,
    fetch_json: Callable[[str], dict[str, Any]],
    settings: BlueskySettings | None = None,
) -> Sample:
    """Posts mentioning `query`, and how the run ended.

    `query` is normally an article URL. Bluesky's search matches it as text, so
    this finds posts that wrote the link out and misses posts that only
    embedded a card; the shortfall is in one direction, which matters because
    every measure built on the sample is a proportion rather than a volume.

    **Unauthenticated search serves one page.** Checked against api.bsky.app on
    2026-10-31: the first request returns 200 and up to 100 posts with a
    cursor, and any request carrying that cursor returns 403 "Request forbidden
    by administrative rules", at limit=25 and limit=100 alike. A cursor-free
    repeat of the same query still returns 200, so it is a restriction on
    paging and not a rate limit. The paging loop stays because it is what a
    credentialed transport needs, and because the alternative is discovering
    this again the next time someone adds a token.

    A refusal on a later page keeps the pages already collected and reports
    `upstream_refused`. Discarding them would throw away a usable sample over a
    limit that is now expected; hiding the refusal would let 100 posts pass for
    the whole of what was shared. A failure on the *first* page raises, because
    there is nothing to hand back and an empty sample would read as "nobody
    posted this" when the truth is that we did not get to look.
    """
    settings = settings or BlueskySettings()
    collected: list[SampledPost] = []
    cursor: str | None = None
    pages = 0
    stop_reason = "page_cap"

    for _ in range(settings.max_pages):
        try:
            payload = fetch_json(search_url(query, limit=settings.page_size, cursor=cursor))
        except Exception as exc:
            if not pages:
                raise
            logger.warning(
                "bluesky_paging_refused",
                page=pages + 1,
                collected=len(collected),
                error=type(exc).__name__,
            )
            stop_reason = "upstream_refused"
            break

        pages += 1
        posts, cursor = parse_posts(payload)
        collected.extend(posts)
        if not cursor:
            stop_reason = "cursor_exhausted"
            break
        if len(collected) >= settings.max_posts:
            stop_reason = "post_cap"
            break

    collected = collected[: settings.max_posts]
    logger.info("bluesky_sampled", posts=len(collected), pages=pages, stop_reason=stop_reason)
    return Sample(
        posts=_fill_ages(collected, fetch_json, settings),
        pages=pages,
        stop_reason=stop_reason,
    )


__all__ = [
    "API_BASE",
    "API_CONTENT_TYPES",
    "PROFILES_METHOD",
    "SEARCH_METHOD",
    "Sample",
    "ages_by_actor",
    "api_settings",
    "default_transport",
    "parse_posts",
    "profiles_url",
    "sample",
    "search_url",
]
