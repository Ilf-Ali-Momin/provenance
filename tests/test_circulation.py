"""Circulation statistics, and the test M4 names as a done-when condition:
"the circulation sampler passes the no persistence test"."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from pipeline.circulation.aggregate import (
    CirculationStats,
    SampledPost,
    aggregate,
    is_storable,
    similarity,
)
from pipeline.circulation.config import CirculationSettings

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
HANDLE = "alice.bsky.social"
SECRET_TEXT = "a sentence that must never reach storage anywhere at all"


def _post(
    n: int = 0, *, age_days: float | None = 400, text: str = "", offset: int = 0
) -> SampledPost:
    return SampledPost(
        author=f"{HANDLE}.{n}",
        author_created_at=None if age_days is None else NOW - timedelta(days=age_days),
        posted_at=NOW + timedelta(seconds=offset),
        text=text or f"an ordinary post number {n} about the local election result",
    )


# --- the no-persistence guarantee -----------------------------------------


def test_no_handle_or_text_survives_aggregation() -> None:
    """The guarantee ROADMAP M4 asks to be verified by a test.

    Serialising the output and searching it is the check that matters: a field
    added later that happened to carry a handle would fail here even if nobody
    remembered this rule.
    """
    posts = [_post(n, text=SECRET_TEXT) for n in range(5)]
    stats = aggregate(posts, NOW)
    serialised = json.dumps(stats.model_dump(mode="json"))
    assert HANDLE not in serialised
    assert SECRET_TEXT not in serialised
    for word in SECRET_TEXT.split():
        if len(word) > 6:
            assert word not in serialised


def test_the_stats_type_has_no_field_that_could_hold_an_identifier() -> None:
    """Not "does not currently hold one" - cannot."""
    numeric = {
        "posts",
        "distinct_authors",
        "accounts_newer_than_30_days",
        "accounts_with_unknown_age",
        "largest_near_duplicate_cluster",
    }
    for name, field in CirculationStats.model_fields.items():
        if name == "reading":
            continue
        annotation = str(field.annotation)
        assert "str" not in annotation, f"{name} can hold a string"
        assert name in numeric or "float" in annotation


def test_is_storable_accepts_stats_and_rejects_posts() -> None:
    assert is_storable(aggregate([_post()], NOW))
    assert not is_storable(_post())
    assert not is_storable({"author": HANDLE})


def test_an_empty_sample_produces_empty_stats() -> None:
    stats = aggregate([], NOW)
    assert stats.posts == 0
    assert stats.near_duplicate_ratio is None


# --- account age -----------------------------------------------------------


def test_new_accounts_are_counted() -> None:
    posts = [_post(n, age_days=3) for n in range(4)] + [_post(9, age_days=900)]
    stats = aggregate(posts, NOW)
    assert stats.accounts_newer_than_30_days == 4
    assert stats.account_age_days_median is not None


def test_unknown_account_age_is_counted_separately_not_treated_as_new() -> None:
    """A missing creation date is missing data, and folding it into the new
    account count would manufacture the pattern this is looking for."""
    stats = aggregate([_post(0, age_days=None), _post(1, age_days=900)], NOW)
    assert stats.accounts_with_unknown_age == 1
    assert stats.accounts_newer_than_30_days == 0


def test_quartiles_show_a_cluster_a_mean_would_hide() -> None:
    """Four new accounts among four old ones: the median moves, a mean would
    sit between and describe neither group."""
    posts = [_post(n, age_days=2) for n in range(4)] + [
        _post(n, age_days=2000) for n in range(4, 8)
    ]
    stats = aggregate(posts, NOW)
    assert stats.account_age_days_p25 is not None
    assert stats.account_age_days_p25 < 100
    assert stats.account_age_days_p75 is not None
    assert stats.account_age_days_p75 > 1000


# --- near duplicates -------------------------------------------------------


def test_identical_text_is_similar() -> None:
    assert similarity("the same words here every time", "the same words here every time") == 1.0


def test_unrelated_text_is_not_similar() -> None:
    assert similarity("unemployment fell in March", "the bridge carries traffic") < 0.2


def test_one_changed_word_still_counts_as_near_duplicate() -> None:
    """The coordination pattern is "same text with a word changed"."""
    a = "breaking this story is being suppressed by the mainstream media right now"
    b = "breaking this story is being suppressed by the mainstream press right now"
    assert similarity(a, b) > 0.5


def test_coordinated_posting_shows_as_a_duplicate_cluster() -> None:
    shared = "identical wording repeated across many different accounts today here"
    posts = [_post(n, text=shared) for n in range(6)] + [_post(9)]
    stats = aggregate(posts, NOW)
    assert stats.largest_near_duplicate_cluster == 6
    assert stats.near_duplicate_ratio is not None
    assert stats.near_duplicate_ratio > 0.8


ORGANIC = (
    "genuinely surprised by the turnout figure, my polling station was empty all morning",
    "anyone else think the council handled this badly? took them three weeks to respond",
    "the bit nobody mentions is how much the repairs will cost the taxpayer",
    "my grandmother voted for the first time since 1997 today which is lovely",
    "reading the full report now and the appendix is more interesting than the summary",
    "was there in person, queue went round the block by half past eight",
)


def test_organic_posting_shows_no_cluster() -> None:
    """People writing their own sentences."""
    posts = [_post(n, text=text) for n, text in enumerate(ORGANIC)]
    stats = aggregate(posts, NOW)
    assert stats.largest_near_duplicate_cluster == 1
    assert stats.near_duplicate_ratio == 0.0


def test_people_quoting_the_same_headline_cluster_which_is_the_false_positive() -> None:
    """The known weakness, asserted rather than left to be discovered.

    Organic spread of one story means many people pasting the same headline.
    Shingle overlap cannot tell that from coordination, because at the level of
    text there is no difference: the words really are the same. Anything built
    on `near_duplicate_ratio` has to survive this, and a card reporting it as
    coordination would be accusing ordinary readers of running a campaign.

    This is the argument for sweeping the threshold against real sampled posts
    rather than choosing it, and for showing the statistic beside the posts
    rather than summarising it into a judgement.
    """
    headline = "Council approves the bridge repair budget after three years of delay"
    posts = [_post(n, text=headline) for n in range(5)]
    stats = aggregate(posts, NOW)
    assert stats.largest_near_duplicate_cluster == 5
    assert stats.near_duplicate_ratio == 1.0


# --- time clustering -------------------------------------------------------


def test_a_burst_shows_as_a_narrow_window() -> None:
    posts = [_post(n, offset=n) for n in range(10)]
    stats = aggregate(posts, NOW)
    assert stats.window_seconds is not None
    assert stats.window_seconds < 60
    assert stats.busiest_minute_share == 1.0


def test_spread_out_posting_shows_a_wide_window() -> None:
    posts = [_post(n, offset=n * 3600) for n in range(10)]
    stats = aggregate(posts, NOW)
    assert stats.window_seconds is not None
    assert stats.window_seconds > 30000
    assert stats.busiest_minute_share is not None
    assert stats.busiest_minute_share < 0.2


def test_volume_is_not_one_of_the_measures() -> None:
    """ARCHITECTURE.md: "The measure is not volume. Volume is confounded by
    topic popularity." `posts` is a denominator, not a finding."""
    assert "volume" not in CirculationStats.model_fields
    assert "not a verdict" in CirculationStats(posts=0, distinct_authors=0).reading


def test_the_threshold_travels_with_the_ratio() -> None:
    """A near-duplicate ratio computed at 0.5 is not comparable with one
    computed at 0.8, and the threshold is unswept."""
    stats = aggregate([_post(0), _post(1)], NOW, CirculationSettings(duplicate_threshold=0.42))
    assert stats.duplicate_threshold == 0.42


def test_the_threshold_changes_what_counts_as_coordinated() -> None:
    """Which is why it is a config value rather than a literal."""
    a = "breaking this story is being suppressed by the mainstream media right now"
    b = "breaking this story is being suppressed by the mainstream press right now"
    posts = [_post(0, text=a), _post(1, text=b)]
    loose = aggregate(posts, NOW, CirculationSettings(duplicate_threshold=0.4))
    strict = aggregate(posts, NOW, CirculationSettings(duplicate_threshold=0.9))
    assert loose.largest_near_duplicate_cluster == 2
    assert strict.largest_near_duplicate_cluster == 1


def test_the_sample_is_capped_because_clustering_is_quadratic() -> None:
    stats = aggregate([_post(n) for n in range(60)], NOW, CirculationSettings(max_posts=20))
    assert stats.posts == 20
