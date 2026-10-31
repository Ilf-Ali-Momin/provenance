"""Tests for the Bluesky sampler.

Every test here runs against a recorded payload shape and a fake transport.
The one test that talks to Bluesky is in `test_bluesky_live.py` and is skipped
unless it is asked for, the same split the registry tests use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pipeline.circulation.aggregate import aggregate, is_storable
from pipeline.circulation.bluesky import (
    ages_by_actor,
    parse_posts,
    profiles_url,
    sample,
    search_url,
)
from pipeline.circulation.config import BlueskySettings

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
OLD = "2015-01-01T00:00:00Z"
NEW = "2026-10-20T00:00:00Z"


def post(
    did: str,
    text: str,
    indexed_at: str | None = "2026-10-31T11:00:00Z",
    created_at: str | None = "2026-10-31T10:00:00Z",
    author_created_at: str | None = None,
) -> dict[str, Any]:
    author: dict[str, Any] = {"did": did, "handle": f"{did.split(':')[-1]}.bsky.social"}
    if author_created_at:
        author["createdAt"] = author_created_at
    entry: dict[str, Any] = {"author": author, "record": {"text": text}}
    if indexed_at:
        entry["indexedAt"] = indexed_at
    if created_at:
        entry["record"]["createdAt"] = created_at
    return entry


def test_the_search_url_carries_the_query_and_the_cursor() -> None:
    url = search_url("https://example.com/story", limit=100, cursor="abc")
    assert "app.bsky.feed.searchPosts" in url
    assert "q=https%3A%2F%2Fexample.com%2Fstory" in url
    assert "limit=100" in url
    assert "cursor=abc" in url


def test_the_search_url_omits_a_cursor_on_the_first_page() -> None:
    assert "cursor" not in search_url("anything", limit=25)


def test_a_profile_request_repeats_the_actors_parameter() -> None:
    url = profiles_url(["did:plc:one", "did:plc:two"])
    assert url.count("actors=") == 2


def test_a_page_is_parsed_into_posts_and_a_cursor() -> None:
    payload = {"posts": [post("did:plc:a", "one"), post("did:plc:b", "two")], "cursor": "next"}
    posts, cursor = parse_posts(payload)
    assert [p.text for p in posts] == ["one", "two"]
    assert cursor == "next"


def test_the_last_page_reports_no_cursor() -> None:
    _, cursor = parse_posts({"posts": [post("did:plc:a", "one")]})
    assert cursor is None


def test_an_empty_cursor_is_treated_as_no_cursor() -> None:
    _, cursor = parse_posts({"posts": [], "cursor": ""})
    assert cursor is None


def test_the_service_timestamp_is_preferred_over_the_posters_claim() -> None:
    """A poster can backdate `createdAt`; they cannot set `indexedAt`.

    Time clustering is one of the three measures, so taking the claimed time
    would let a set of coordinated accounts spread a one minute burst across a
    fabricated week and read as organic.
    """
    posts, _ = parse_posts(
        {
            "posts": [
                post(
                    "did:plc:a",
                    "one",
                    indexed_at="2026-10-31T11:00:00Z",
                    created_at="2019-01-01T00:00:00Z",
                )
            ]
        }
    )
    assert posts[0].posted_at == datetime(2026, 10, 31, 11, 0, tzinfo=UTC)


def test_the_claimed_time_is_used_when_the_service_reported_none() -> None:
    posts, _ = parse_posts(
        {"posts": [post("did:plc:a", "one", indexed_at=None, created_at="2026-10-30T09:00:00Z")]}
    )
    assert posts[0].posted_at == datetime(2026, 10, 30, 9, 0, tzinfo=UTC)


def test_a_post_with_no_usable_time_is_dropped_rather_than_guessed() -> None:
    posts, _ = parse_posts({"posts": [post("did:plc:a", "one", indexed_at=None, created_at=None)]})
    assert posts == []


def test_an_unparseable_time_is_not_silently_turned_into_now() -> None:
    posts, _ = parse_posts(
        {"posts": [post("did:plc:a", "one", indexed_at="not a date", created_at=None)]}
    )
    assert posts == []


def test_a_malformed_entry_does_not_lose_the_rest_of_the_page() -> None:
    payload = {"posts": ["nonsense", {"author": {}}, post("did:plc:a", "kept")]}
    posts, _ = parse_posts(payload)
    assert [p.text for p in posts] == ["kept"]


def test_an_account_age_the_search_reported_is_kept() -> None:
    posts, _ = parse_posts(
        {"posts": [post("did:plc:a", "one", author_created_at="2020-05-05T00:00:00Z")]}
    )
    assert posts[0].author_created_at == datetime(2020, 5, 5, tzinfo=UTC)


def test_an_account_age_the_search_withheld_is_unknown_not_zero() -> None:
    posts, _ = parse_posts({"posts": [post("did:plc:a", "one")]})
    assert posts[0].author_created_at is None


def test_profile_ages_are_keyed_by_both_did_and_handle() -> None:
    ages = ages_by_actor(
        {
            "profiles": [
                {"did": "did:plc:a", "handle": "a.bsky.social", "createdAt": "2021-01-01T00:00:00Z"}
            ]
        }
    )
    assert ages["did:plc:a"] == ages["a.bsky.social"] == datetime(2021, 1, 1, tzinfo=UTC)


def test_a_profile_without_a_creation_date_is_absent_not_placeholdered() -> None:
    assert ages_by_actor({"profiles": [{"did": "did:plc:a", "handle": "a.bsky.social"}]}) == {}


class FakeTransport:
    """Answers search and profile URLs from prepared pages."""

    def __init__(self, pages: list[dict[str, Any]], profiles: dict[str, Any] | None = None) -> None:
        self.pages = pages
        self.profiles = profiles or {"profiles": []}
        self.urls: list[str] = []
        self.profile_calls = 0

    def __call__(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        if "getProfiles" in url:
            self.profile_calls += 1
            return self.profiles
        return self.pages[len([u for u in self.urls if "searchPosts" in u]) - 1]


def test_sampling_follows_the_cursor_across_pages() -> None:
    transport = FakeTransport(
        [
            {"posts": [post("did:plc:a", "one")], "cursor": "p2"},
            {"posts": [post("did:plc:b", "two")]},
        ]
    )
    result = sample("https://example.com/story", transport, BlueskySettings(page_size=1))
    assert [p.text for p in result.posts] == ["one", "two"]
    assert "cursor=p2" in transport.urls[1]
    assert result.pages == 2


def test_an_exhausted_cursor_is_the_one_stop_reason_that_means_complete() -> None:
    transport = FakeTransport([{"posts": [post("did:plc:a", "one")]}])
    assert sample("q", transport, BlueskySettings()).stop_reason == "cursor_exhausted"


def test_paging_stops_at_the_page_cap_even_with_a_live_cursor() -> None:
    """An upstream that always returns a cursor must not page without end."""
    endless = [{"posts": [post(f"did:plc:{i}", f"post {i}")], "cursor": "more"} for i in range(50)]
    transport = FakeTransport(endless)
    result = sample("q", transport, BlueskySettings(page_size=1, max_pages=3, max_posts=500))
    assert len([u for u in transport.urls if "searchPosts" in u]) == 3
    assert result.stop_reason == "page_cap"


def test_the_post_cap_stops_paging() -> None:
    transport = FakeTransport(
        [
            {"posts": [post("did:plc:a", "one"), post("did:plc:b", "two")], "cursor": "p2"},
            {"posts": [post("did:plc:c", "three")]},
        ]
    )
    result = sample("q", transport, BlueskySettings(page_size=2, max_posts=2))
    assert len(result.posts) == 2
    assert result.stop_reason == "post_cap"
    assert len([u for u in transport.urls if "searchPosts" in u]) == 1


def test_a_refusal_on_a_later_page_keeps_the_pages_already_collected() -> None:
    """The live constraint this was written against.

    Checked on 2026-10-31: unauthenticated search returns 100 posts and a
    cursor, then refuses any request carrying that cursor. Discarding the first
    page over that would throw away the only sample available.
    """
    calls = {"n": 0}

    def transport(url: str) -> dict[str, Any]:
        if "getProfiles" in url:
            return {"profiles": []}
        calls["n"] += 1
        if calls["n"] == 1:
            return {"posts": [post("did:plc:a", "one")], "cursor": "p2"}
        raise RuntimeError("403 Request forbidden by administrative rules")

    result = sample("q", transport, BlueskySettings(page_size=1, max_posts=500))
    assert [p.text for p in result.posts] == ["one"]
    assert result.pages == 1
    assert result.stop_reason == "upstream_refused"


def test_a_refusal_on_the_first_page_raises_rather_than_reading_as_silence() -> None:
    """An empty sample would say nobody posted this. We did not get to look."""

    def transport(url: str) -> dict[str, Any]:
        raise RuntimeError("403 Request forbidden by administrative rules")

    with pytest.raises(RuntimeError):
        sample("q", transport, BlueskySettings())


def test_unknown_account_ages_are_filled_from_a_profile_request() -> None:
    transport = FakeTransport(
        [{"posts": [post("did:plc:a", "one"), post("did:plc:b", "two")]}],
        profiles={
            "profiles": [
                {"did": "did:plc:a", "handle": "a", "createdAt": "2021-01-01T00:00:00Z"},
                {"did": "did:plc:b", "handle": "b", "createdAt": "2026-10-01T00:00:00Z"},
            ]
        },
    )
    posts = sample("q", transport, BlueskySettings()).posts
    assert transport.profile_calls == 1
    assert {p.author_created_at for p in posts} == {
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime(2026, 10, 1, tzinfo=UTC),
    }


def test_no_profile_request_is_made_when_the_search_reported_every_age() -> None:
    transport = FakeTransport(
        [{"posts": [post("did:plc:a", "one", author_created_at="2020-01-01T00:00:00Z")]}]
    )
    sample("q", transport, BlueskySettings())
    assert transport.profile_calls == 0


def test_a_failed_profile_request_leaves_the_age_unknown_rather_than_new() -> None:
    """An unmeasured account is not a new account.

    Treating a failed lookup as a recent creation date would manufacture the
    exact pattern the account age measure exists to show, so a failure leaves
    the age as None and `accounts_with_unknown_age` carries it.
    """

    def transport(url: str) -> dict[str, Any]:
        if "getProfiles" in url:
            raise RuntimeError("upstream down")
        return {"posts": [post("did:plc:a", "one"), post("did:plc:b", "two")]}

    posts = sample("q", transport, BlueskySettings()).posts
    assert all(p.author_created_at is None for p in posts)

    stats = aggregate(list(posts), NOW)
    assert stats.accounts_with_unknown_age == 2
    assert stats.account_age_days_median is None


def test_an_empty_result_is_an_empty_sample_not_an_error() -> None:
    result = sample("q", FakeTransport([{"posts": []}]), BlueskySettings())
    assert result.posts == []
    assert result.stop_reason == "cursor_exhausted"


def test_a_sample_aggregates_into_something_storable() -> None:
    """The whole point of the boundary: what comes out holds no identifier."""
    transport = FakeTransport(
        [
            {
                "posts": [
                    post("did:plc:a", "council votes to close the library", author_created_at=OLD),
                    post("did:plc:b", "council votes to close the library", author_created_at=NEW),
                    post("did:plc:c", "unrelated chatter about the weather", author_created_at=OLD),
                ]
            }
        ]
    )
    stats = aggregate(sample("q", transport, BlueskySettings()).posts, NOW)

    assert is_storable(stats)
    assert stats.posts == 3
    assert stats.distinct_authors == 3
    assert stats.largest_near_duplicate_cluster == 2
    assert stats.accounts_newer_than_30_days == 1
    dumped = repr(stats.model_dump())
    for identifier in ("did:plc:", "council votes", "bsky.social"):
        assert identifier not in dumped


@pytest.mark.parametrize("field", ["page_size", "profile_batch"])
def test_the_upstream_maximums_are_enforced_by_the_settings(field: str) -> None:
    with pytest.raises(ValueError):
        BlueskySettings(**{field: 1000})
