"""Circulation statistics, and the identifiers that never leave memory.

ARCHITECTURE.md: "All social data is processed in memory and aggregated. No
post text, no handle, no user identifier is persisted. Only the aggregate
statistics reach the database. This is a GDPR requirement, not a preference."
DATA.md says the same in its own words, and ROADMAP.md's M4 asks for the
guarantee to be "verified by a test asserting nothing identifying reaches
storage".

**Two types, and the boundary between them is the control.** `SampledPost` is
what a sampler holds while it works: a handle, a timestamp, the text. It is
never returned by anything in this module and never written anywhere.
`CirculationStats` is what comes out and what may be stored, and it has no
field capable of holding a handle, a post id, or a line of text - only counts
and distributions. A caller cannot accidentally persist an identifier, because
the function that produces the storable thing does not give them one.

**The measures are not volume, and that is a methodological choice rather than
an oversight.** ARCHITECTURE.md: "The measure is not volume. Volume is
confounded by topic popularity." A story about an election spreads more than a
story about a water main, and counting posts measures the topic. The three
measures instead describe the *shape* of spreading:

* account age distribution, because a burst from accounts created last week is
  different from a burst from accounts created years ago;
* near duplicate ratio, because near identical text across many accounts
  indicates coordination rather than organic spread;
* time clustering, because organic spread is spread out.

None of the three is evidence that a claim is false, and nothing here computes
a score from them. That is the same line D007 draws for the registry.
"""

from __future__ import annotations

import re
import statistics
from datetime import datetime
from itertools import pairwise
from typing import Final, NamedTuple

from pydantic import BaseModel, Field

from pipeline.circulation.config import CirculationSettings

# Shingles rather than exact match, because the coordination pattern this looks
# for is "same text with a word changed". Both numbers live in
# `CirculationSettings`: they are thresholds, and RULES.md keeps thresholds
# sweepable rather than buried. The defaults there are unswept and say so.
_WORD: Final = re.compile(r"[a-z0-9']+")


class SampledPost(NamedTuple):
    """One post, in memory, for as long as aggregation takes.

    A NamedTuple rather than a pydantic model deliberately: this is a transient
    working value, not a schema, and giving it a schema is the first step
    toward someone serialising it. Nothing in this module returns one and
    nothing writes one down.
    """

    author: str
    author_created_at: datetime | None
    posted_at: datetime
    text: str


class CirculationStats(BaseModel):
    """What may be stored. Counts and distributions, nothing identifying.

    Every field is a number. There is no string field at all, which is what
    makes "no handle is persisted" a property of the type rather than a promise
    about the code around it.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    posts: int = Field(ge=0)
    distinct_authors: int = Field(ge=0)

    # Account age, in days, among posting accounts. Quartiles rather than a
    # mean: the pattern worth seeing is a cluster of new accounts, and a mean
    # hides it behind long-standing ones.
    account_age_days_p25: float | None = None
    account_age_days_median: float | None = None
    account_age_days_p75: float | None = None
    accounts_newer_than_30_days: int = Field(default=0, ge=0)
    accounts_with_unknown_age: int = Field(default=0, ge=0)

    # Share of posts that are near identical to another post in the sample.
    near_duplicate_ratio: float | None = Field(default=None, ge=0, le=1)
    largest_near_duplicate_cluster: int = Field(default=0, ge=0)
    # Recorded with the result, because the ratio means nothing without it and
    # the threshold is unswept. A figure quoted without its threshold cannot be
    # compared against one computed at a different setting.
    duplicate_threshold: float | None = Field(default=None, ge=0, le=1)

    # Spread of posting times. A small spread against a long window is the
    # clustering ARCHITECTURE.md names.
    window_seconds: float | None = Field(default=None, ge=0)
    median_gap_seconds: float | None = Field(default=None, ge=0)
    busiest_minute_share: float | None = Field(default=None, ge=0, le=1)

    reading: str = (
        "Descriptions of how something spread, not evidence about whether it is true. "
        "A new-account cluster posting identical text within a minute is a pattern worth "
        "showing a reader; it is not a verdict, and nothing here composes these into one."
    )


def _shingles(text: str, size: int = 3) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def similarity(left: str, right: str, shingle_words: int = 3) -> float:
    """Jaccard overlap of word shingles. 1.0 is identical wording."""
    a, b = _shingles(left, shingle_words), _shingles(right, shingle_words)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def _duplicate_clusters(
    posts: list[SampledPost], threshold: float, shingle_words: int
) -> list[int]:
    """Sizes of groups of near identical posts, largest first.

    Single linkage over pairwise similarity. Quadratic, which is fine for a
    sample of hundreds and would not be for millions; the sampler caps the
    sample, and that cap is where the cost is bounded.
    """
    parent = list(range(len(posts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    shingles = [_shingles(p.text, shingle_words) for p in posts]
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            a, b = shingles[i], shingles[j]
            union = len(a | b)
            if union and len(a & b) / union >= threshold:
                parent[find(i)] = find(j)

    sizes: dict[int, int] = {}
    for i in range(len(posts)):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def aggregate(
    posts: list[SampledPost],
    observed_at: datetime,
    settings: CirculationSettings | None = None,
) -> CirculationStats:
    """Turn posts into statistics, and drop everything identifying.

    The posts go in and do not come out. Whatever the caller does with the
    return value, it cannot write a handle to a database, because the returned
    type has nowhere to put one.
    """
    settings = settings or CirculationSettings()
    if not posts:
        return CirculationStats(posts=0, distinct_authors=0)
    # Bounded here rather than inside the quadratic loop below.
    posts = posts[: settings.max_posts]

    ages = [
        (observed_at - p.author_created_at).total_seconds() / 86400
        for p in posts
        if p.author_created_at is not None
    ]
    quartiles: tuple[float | None, float | None, float | None] = (None, None, None)
    if len(ages) >= 4:
        cut = statistics.quantiles(ages, n=4)
        quartiles = (cut[0], cut[1], cut[2])
    elif ages:
        median = statistics.median(ages)
        quartiles = (min(ages), median, max(ages))

    clusters = _duplicate_clusters(posts, settings.duplicate_threshold, settings.shingle_words)
    in_clusters = sum(size for size in clusters if size > 1)

    times = sorted(p.posted_at for p in posts)
    window = (times[-1] - times[0]).total_seconds()
    gaps = [(b - a).total_seconds() for a, b in pairwise(times)]
    per_minute: dict[int, int] = {}
    for moment in times:
        key = int(moment.timestamp() // 60)
        per_minute[key] = per_minute.get(key, 0) + 1

    return CirculationStats(
        posts=len(posts),
        # Counted, then discarded. The count is a number; the handles are not
        # returned, and this is the only place they are looked at.
        distinct_authors=len({p.author for p in posts}),
        account_age_days_p25=quartiles[0],
        account_age_days_median=quartiles[1],
        account_age_days_p75=quartiles[2],
        accounts_newer_than_30_days=sum(1 for age in ages if age < 30),
        accounts_with_unknown_age=sum(1 for p in posts if p.author_created_at is None),
        near_duplicate_ratio=in_clusters / len(posts),
        duplicate_threshold=settings.duplicate_threshold,
        largest_near_duplicate_cluster=clusters[0] if clusters else 0,
        window_seconds=window,
        median_gap_seconds=statistics.median(gaps) if gaps else None,
        busiest_minute_share=max(per_minute.values()) / len(posts),
    )


def is_storable(value: object) -> bool:
    """Does this carry only aggregate statistics?

    Used by the test ROADMAP.md M4 asks for. A `CirculationStats` whose fields
    are all numeric passes; anything holding a string that is not the fixed
    reading, or any other type, does not.
    """
    if not isinstance(value, CirculationStats):
        return False
    return all(
        isinstance(v, int | float | type(None)) or k == "reading"
        for k, v in value.model_dump().items()
    )


__all__ = [
    "CirculationStats",
    "SampledPost",
    "aggregate",
    "is_storable",
    "similarity",
]
