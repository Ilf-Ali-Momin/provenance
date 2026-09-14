"""Collection limits, as named settings rather than literals.

RULES.md requires every threshold to be a named config value so the evaluation
suite can sweep it. These are the ones that decide how much of a publisher's
archive is walked and how the sample is spread, which is what D022 turns on.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CollectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COLLECT_", frozen=True)

    # Articles per publisher taken from the live window: feeds and news
    # sitemaps, which list the last day or two.
    live_limit: int = Field(default=15, gt=0)

    # Backfill (D022). Archive sitemaps name candidates but date them only by
    # last modification, so each candidate is fetched and kept only if the page
    # states a publication date after the snapshot.
    #
    # The month cap is what stops the suite being one news cycle again: without
    # it a publisher whose archive lists newest first would fill its whole quota
    # from the same fortnight. The fetch budget bounds the cost of candidates
    # that turn out to be undated or too old, which most will be.
    backfill_per_month: int = Field(default=2, gt=0)
    backfill_fetch_budget: int = Field(default=60, gt=0)
    backfill_sitemaps: int = Field(default=4, gt=0)
    backfill_children: int = Field(default=6, gt=0)
    backfill_candidates: int = Field(default=400, gt=0)
