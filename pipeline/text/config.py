"""Extraction thresholds.

RULES.md: every threshold is a named config value so the evaluation suite can
sweep it, rather than a literal buried in a function.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXTRACT_", frozen=True)

    # Measured need, not a guess: trafilatura returns navigation text such as
    # "Menu" when a page carries no article, and without a floor those
    # fragments enter the corpus as documents. A news article that survives
    # publisher stripping is far longer than this; the floor is deliberately
    # low so it removes non articles rather than short articles.
    min_article_chars: int = Field(default=400, gt=0)
