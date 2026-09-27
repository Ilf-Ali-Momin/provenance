"""What the source registry may hold, and what it cannot express.

D007 is marked legally load bearing and says the registry "stores dated
observations with recorded provenance. No computed score, no tier, no label."
ARCHITECTURE.md gives the reason twice over: composing observations into a
score "would recreate the truth verdict problem one level down, and it would
also make the registry a defamation surface, which is a real legal exposure and
not a hypothetical one".

**So the score is not omitted, it is unrepresentable.** `DomainRecord` has no
numeric field, no tier, no label, and `extra="forbid"`, so a caller adding one
gets a validation error rather than a column. This is the same enforcement the
API uses to keep a verdict out of a response and the job result uses to keep
article text out of Redis: make the unwanted state impossible to write down,
because a rule that depends on reviewers noticing is a rule that lasts until a
busy afternoon.

**Every observation carries its own provenance.** Not the record's - the
observation's. A registry row is assembled from several sources on several
dates: RDAP for registration age, the about page for a masthead, a transparency
dataset for a listing. "This domain was registered in 2019" is only meaningful
beside where that came from and when it was read, and a shared record-level
date would silently backdate a field observed months later.

**Observations are append only.** A field that changes does not overwrite; the
new observation is appended and the old one stays. A registry that quietly
revised itself would be unable to answer the question that matters when
something is disputed, which is what did we observe and when did we observe it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field


class ObservationKind(StrEnum):
    """What can be observed about a domain.

    A closed set, and closed on purpose: an open string field is where someone
    eventually writes "credibility" (D007). Each member names something a
    reader could verify for themselves from the recorded source.
    """

    REGISTERED_ON = "registered_on"
    REGISTRAR = "registrar"
    MASTHEAD_PRESENT = "masthead_present"
    CORRECTIONS_POLICY_PRESENT = "corrections_policy_present"
    OWNERSHIP_STATED = "ownership_stated"
    TRANSPARENCY_DATASET_LISTING = "transparency_dataset_listing"
    PUBLISHER_METADATA = "publisher_metadata"


class Source(StrEnum):
    """Where an observation came from. Also closed, for the same reason."""

    RDAP = "rdap"
    ABOUT_PAGE = "about_page"
    TRANSPARENCY_DATASET = "transparency_dataset"
    STRUCTURED_METADATA = "structured_metadata"


# Values an observation may carry. Deliberately not `float`: a number invites a
# threshold, a threshold invites a tier, and a tier is the score D007 forbids.
# `registered_on` needs a date and the rest are presence or a short string.
ObservationValue = bool | str | date


class Observation(BaseModel):
    """One dated, sourced fact about a domain.

    Frozen, because an observation that could be edited after the fact is not a
    record of what was seen.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    kind: ObservationKind
    value: ObservationValue
    source: Source
    # Where a reader can go and check. Required: an observation whose origin
    # cannot be revisited is an assertion by this project, which is the thing
    # D007 rules out.
    source_url: str = Field(min_length=1)
    observed_at: datetime
    # What the source actually said, quoted rather than paraphrased, when the
    # value is an interpretation of prose. Absent for a date from RDAP.
    evidence: str | None = Field(default=None, max_length=500)


class DomainRecord(BaseModel):
    """Everything the registry holds about one registrable domain.

    **There is no score field, and there is no room for one.** `extra="forbid"`
    means `DomainRecord(..., credibility=0.8)` raises rather than storing, and
    the type carries no number that a caller could aggregate into one. D007
    calls this legally load bearing; the schema is where that stops being a
    policy and becomes a fact about the program.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Registrable domain, resolved through the public suffix list rather than
    # by splitting on dots - ARCHITECTURE.md names subdomain handling as a
    # common source of both bugs and evasion.
    domain: str = Field(min_length=1)
    observations: tuple[Observation, ...] = ()

    def latest(self, kind: ObservationKind) -> Observation | None:
        """The most recent observation of one kind, or None if never observed.

        None rather than a default: "we have not looked" and "we looked and
        found nothing" are different answers, and a card that renders them the
        same way is misleading about its own coverage.
        """
        matching = [o for o in self.observations if o.kind == kind]
        return max(matching, key=lambda o: o.observed_at) if matching else None

    def history(self, kind: ObservationKind) -> tuple[Observation, ...]:
        """Every observation of one kind, oldest first.

        The registry is append only, so a field that changed has more than one
        entry, and the sequence is the answer to "what did we observe and
        when".
        """
        return tuple(
            sorted((o for o in self.observations if o.kind == kind), key=lambda o: o.observed_at)
        )

    def with_observation(self, observation: Observation) -> DomainRecord:
        """A new record with one more observation. Nothing is replaced."""
        return self.model_copy(update={"observations": (*self.observations, observation)})


# Names a caller might reach for when they want the thing D007 forbids. Checked
# by a test against the schema's fields, so that adding one is caught here
# rather than in review.
FORBIDDEN_FIELDS: Final = frozenset(
    {"score", "credibility", "rating", "tier", "trust", "reliability", "label", "rank", "grade"}
)
