"""Extractor configuration. RULES.md: no literals for hosts, models or budgets."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractorSettings(BaseSettings):
    # populate_by_name so the field is settable as `api_key` in a test while
    # still being read from GROQ_API_KEY in the environment.
    model_config = SettingsConfigDict(env_prefix="EXTRACTOR_", frozen=True, populate_by_name=True)

    # The key is read from the environment and never logged. `.env` is
    # gitignored and holds it locally.
    api_key: str = Field(default="", alias="GROQ_API_KEY")
    base_url: str = "https://api.groq.com/openai/v1"
    # Checked against the account's own /models listing rather than assumed:
    # the first default here was llama-3.3-70b-versatile, which this key cannot
    # reach, and the request failed with model_not_found.
    model: str = "openai/gpt-oss-120b"
    prompt_name: str = "claim_extraction.v1.md"
    # Deterministic by default: EVALUATION.md runs the same evaluation against
    # every prompt revision, and a sampled extractor would confound a prompt
    # change with sampling noise.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Generous, because gpt-oss models emit reasoning tokens before content.
    # At 512 the budget was spent reasoning and the reply was empty, which the
    # provider reports as json_validate_failed with an empty failed_generation
    # - a confusing error for a budget problem.
    max_tokens: int = Field(default=4096, gt=0)
    # Lower effort keeps latency and token spend down on what is a short,
    # mechanical task. Sent only when set, since not every model accepts it.
    reasoning_effort: str = "low"
    timeout_s: float = Field(default=20.0, gt=0)
    # A provider that answers 429 with Retry-After is telling us exactly how
    # long to wait. Ignoring it and retrying immediately is the behaviour this
    # project's own rate limiter exists to stop (D033); doing it to someone
    # else's API is the same mistake pointed outward.
    max_retries: int = Field(default=4, ge=0)
    retry_backoff_s: float = Field(default=2.0, gt=0)
    # A floor between calls, so a long evaluation sweep does not walk into the
    # limit in the first place.
    min_interval_s: float = Field(default=0.0, ge=0)
