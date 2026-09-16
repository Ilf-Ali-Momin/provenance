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
