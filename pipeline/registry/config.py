"""Re-verification cadences.

RULES.md: every threshold is a named config value so the evaluation suite can
sweep it. These are cadences rather than thresholds - nothing downstream is
derived from them, so a wrong value costs a stale field or a wasted request
rather than a wrong number.

The two cadences differ because the fields differ in how they change. A
registration date does not change at all once a domain exists, and a registrar
changes when a domain is transferred, which is rare and which is also the
signal worth catching. An about page is prose on a website: a publisher can add
a corrections policy, move it, or take it down in an afternoon, and the link
this project recorded can rot without anything else changing.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReverificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGISTRY_", frozen=True)

    # RDAP fields. Long, because a registration date is immutable and a
    # registrar change is a transfer. Shorter would mean asking a registry the
    # same question repeatedly for an answer that almost never moves.
    rdap_recheck_days: int = Field(default=180, gt=0)

    # About-page fields. A month, because a publisher's own pages move on the
    # timescale a website changes, and because a dead link recorded as evidence
    # is worse than no observation - a reader following it sees a 404 and this
    # project looks like it made the page up.
    about_recheck_days: int = Field(default=30, gt=0)

    # Cap on one scheduler run. The registry grows with the corpus, and a run
    # that re-reads every domain at once is a burst of requests at publishers
    # who did not ask for it. Bounded here rather than in the loop.
    max_domains_per_run: int = Field(default=50, gt=0)
