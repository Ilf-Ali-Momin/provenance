"""Circulation thresholds.

RULES.md: every threshold is a named config value so the evaluation suite can
sweep it, rather than a literal buried in a function. Both values here are
starting points with no measurement behind them, and they are the kind of
threshold that is dangerous to guess: too high and coordinated posting with a
word changed reads as organic, too low and people quoting the same headline
read as coordinated. The second error is worse - it is this system telling a
reader that ordinary spread looks manufactured.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CirculationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CIRCULATION_", frozen=True)

    # Word shingle length. Three rather than five because social posts are
    # short: on a twelve word post, five word shingles leave only eight of
    # them, so a single edited word moves the score by a fifth. Measured on
    # one edited word in a twelve word post: 0.538 at n=3 against 0.455 at n=5.
    shingle_words: int = Field(default=3, ge=2, le=10)

    # Jaccard overlap at or above which two posts are called near identical.
    #
    # **Unswept, and the default is a guess.** Measured against one base post:
    # identical 1.0, a hashtag appended 0.909, a reworded opening 0.818, one
    # word changed 0.538, two words changed 0.429, unrelated 0.000. Anything
    # between about 0.05 and 0.42 separates those cases, which says the metric
    # discriminates and says nothing about where the line belongs on real
    # posts, where people quoting the same headline share real phrasing.
    #
    # 0.5 catches an edited duplicate and not two words changed. It is defended
    # by nothing except that it sits in the gap, and it must be swept against
    # sampled posts before any number derived from it is reported.
    duplicate_threshold: float = Field(default=0.5, ge=0, le=1)

    # Cap on a sample. Clustering is quadratic in the sample size, so this is
    # where that cost is bounded rather than in the loop.
    max_posts: int = Field(default=500, gt=0)
