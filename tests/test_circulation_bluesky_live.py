"""The Bluesky sampler against the real API.

Excluded by default and never a commit gate: a failure here can mean the
AppView changed, which is information rather than a broken build. Run it with
`pytest -m network tests/test_circulation_bluesky_live.py`.

It exists because every significant defect in this project so far was found by
running against real data, and this module is a case in point. Three things the
recorded-payload tests could not have shown, all checked on 2026-10-31:

* `public.api.bsky.app`, the host most documentation names, serves
  `getProfiles` but refuses `searchPosts` with a 403 HTML page. `api.bsky.app`
  serves both.
* Unauthenticated search serves one page. Any request carrying the returned
  cursor is refused, at every page size, while a cursor-free repeat succeeds.
* A bare domain query returns posts from very few accounts - 100 posts from 7
  accounts for theguardian.com - because automated link-mirroring accounts
  dominate it. Sample article URLs, not domains.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pipeline.circulation.aggregate import aggregate, is_storable
from pipeline.circulation.bluesky import default_transport, sample
from pipeline.circulation.config import BlueskySettings

pytestmark = pytest.mark.network


def test_a_live_sample_aggregates_and_keeps_no_identifier() -> None:
    result = sample("bbc.co.uk", default_transport(), BlueskySettings(page_size=25, max_pages=1))
    assert result.posts, "the AppView returned nothing for a widely shared domain"
    assert result.pages == 1

    stats = aggregate(result.posts, datetime.now(UTC))
    assert is_storable(stats)
    assert stats.posts == len(result.posts)
    dumped = repr(stats.model_dump())
    assert "did:plc:" not in dumped
    assert "bsky.social" not in dumped


def test_paging_past_the_first_page_is_still_refused_without_a_token() -> None:
    """If this ever fails, the restriction was lifted and the cap can go up."""
    result = sample("bbc.co.uk", default_transport(), BlueskySettings(page_size=100, max_pages=2))
    assert result.pages == 1
    assert result.stop_reason == "upstream_refused"
