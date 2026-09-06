"""Fetcher configuration.

RULES.md forbids literals for hosts, timeouts, thresholds and caps, so every
limit in SECURITY.md control 1 is a named setting the evaluation suite can
sweep. Defaults are the values control 1 specifies; they are defaults, not
constants.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FetchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FETCH_", frozen=True)

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    allowed_ports: frozenset[int] = frozenset({80, 443})

    # What the fetcher will accept. The default is the article path and is
    # deliberately narrow: a fetch that returns a PDF, an image or an
    # executable is not something the extraction pipeline should ever see.
    # Feed discovery widens this explicitly (see FEED_CONTENT_TYPES) rather
    # than the default being loosened for everyone, because the reason the
    # article path is narrow does not stop applying just because feeds exist.
    allowed_content_types: frozenset[str] = frozenset(
        {"text/html", "application/xhtml+xml", "text/plain"}
    )

    connect_timeout_s: float = Field(default=5.0, gt=0)
    total_timeout_s: float = Field(default=15.0, gt=0)

    max_redirects: int = Field(default=3, ge=0)
    max_response_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_decompressed_bytes: int = Field(default=5 * 1024 * 1024, gt=0)

    # Sent so that a complaint reaches a person instead of a blocklist
    # (SECURITY.md control 5).
    user_agent: str = "ProvenanceBot/0.1 (+https://example.invalid/bot)"
