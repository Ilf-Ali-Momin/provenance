"""Retrieval parameters.

RULES.md: every threshold is a named config value so the evaluation suite can
sweep it, rather than a literal buried in a function. ARCHITECTURE.md says the
same thing about this component specifically - the dense/lexical weighting "is
a config value swept in evaluation, not a magic constant" - so the default here
is a starting point for the sweep and carries no claim to be right.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", frozen=True)

    # Share of the hybrid score from the dense side. 0.5 is an unswept
    # starting point and is deliberately not defended: EVALUATION.md sweeps it
    # against dev, and whatever that produces replaces this with a number that
    # has a measurement behind it.
    dense_weight: float = Field(default=0.5, ge=0, le=1)

    # Below this, a result is "nothing above the relevance threshold" and the
    # claim counts toward the retrieval failure rate. Also unswept. It is the
    # parameter that most directly trades failure rate against precision, so it
    # is reported alongside any failure rate rather than assumed.
    score_floor: float = Field(default=0.0, ge=0)

    # How many candidates each side contributes before fusion. Larger than the
    # k in recall@20 on purpose: fusion can only rank what it was given, and
    # cutting each side at 20 caps the hybrid's recall at each side's own.
    candidates_per_side: int = Field(default=100, gt=0)

    top_k: int = Field(default=20, gt=0)
