"""Application configuration. No literals anywhere else in `app/`.

RULES.md: every host, identifier, threshold and timeout is a named setting read
from the environment, so the evaluation suite can sweep it and so nothing is
tuned by editing code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", frozen=True)

    # Identification
    service_name: str = "provenance-api"
    environment: str = "development"

    # Logging. `max_logged_chars` is the redaction floor: any string longer than
    # this is replaced with its length and hash before it reaches a handler.
    # SECURITY.md requires that no article text reaches a log at any level, and
    # requires it enforced by a filter rather than by developer discipline, so
    # this is deliberately a cap on *every* string rather than on named fields.
    log_level: str = "INFO"
    log_json: bool = True
    max_logged_chars: int = Field(default=200, gt=0)

    # Readiness. A check that cannot answer within this budget is reported as
    # failing rather than holding the probe open.
    readiness_timeout_s: float = Field(default=2.0, gt=0)

    # Queue. ARCHITECTURE.md: Redis backs the job queue, the response cache and
    # the rate limit counters.
    redis_url: str = "redis://127.0.0.1:6379/0"
    queue_name: str = "fetch"
    job_timeout_s: int = Field(default=60, gt=0)
    # How long a completed result stays readable. It holds derived facts and a
    # hash, never article text (D032), so this is a cache lifetime rather than
    # a retention decision about personal data.
    result_ttl_s: int = Field(default=1800, gt=0)

    # Rate limiting (SECURITY.md control 5). Tiered by key, with a separate
    # tighter limit on the URL fetch path because it is the expensive one and
    # the one that generates outbound traffic attributable to us.
    rate_limit_per_key: int = Field(default=120, gt=0)
    fetch_limit_per_key: int = Field(default=20, gt=0)
    rate_limit_period_s: int = Field(default=60, gt=0)
    # Per target domain, global across every key. Conservative by default:
    # DATA.md requires rate limits "well under anything that would look like
    # scraping pressure", and this is the control that keeps our address out of
    # publisher blocklists.
    fetches_per_domain_per_hour: int = Field(default=30, gt=0)
    domain_period_s: int = Field(default=3600, gt=0)

    # Authentication. API.md: every endpoint except /health takes a bearer key.
    # Empty means the API refuses every request rather than allowing every one,
    # because an unset secret must never be the permissive case.
    api_keys: tuple[str, ...] = ()
