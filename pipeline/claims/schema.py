"""The extractor's output schema, which is SECURITY.md control 2's actual control.

Control 2 is explicit that instructing a model to ignore embedded instructions
is "a mitigation of unknown and unmeasurable strength. It is a layer, not a
control." The control is structural: the schema permits only character offsets
and enum tags, so a successful injection can at most change which offsets are
selected. It cannot write attacker chosen text into a card, because no field
exists for that text to occupy.

That property is worth stating as an invariant rather than leaving implicit in
the field list: **no field in `RawClaim` accepts a string from the model.** The
claim's text is sliced from the source by this project, using offsets the model
returned, after those offsets have been validated against the source.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field, model_validator

MAX_CLAIMS: Final = 5
ID_PREFIX: Final = "cl_"


class ClaimType(StrEnum):
    """ARCHITECTURE.md: factual assertion, opinion, prediction, attribution."""

    FACTUAL_ASSERTION = "factual_assertion"
    OPINION = "opinion"
    PREDICTION = "prediction"
    ATTRIBUTION = "attribution"


class RawClaim(BaseModel):
    """What the model is permitted to return. Two integers and one enum."""

    model_config = {"frozen": True, "extra": "forbid"}

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    type: ClaimType

    @model_validator(mode="after")
    def _ordered(self) -> RawClaim:
        if self.end <= self.start:
            msg = "end must be greater than start"
            raise ValueError(msg)
        return self


class RawExtraction(BaseModel):
    model_config = {"extra": "forbid"}

    claims: list[RawClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)


class Claim(BaseModel):
    """A validated claim. `text` is sliced from the source, never model output."""

    model_config = {"frozen": True}

    claim_id: str
    text: str
    start: int
    end: int
    type: ClaimType
    extractor_version: str


def claim_id(source_sha: str, start: int, end: int) -> str:
    digest = hashlib.sha256(f"{source_sha}:{start}:{end}".encode()).hexdigest()
    return f"{ID_PREFIX}{digest[:20]}"
