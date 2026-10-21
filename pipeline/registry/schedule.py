"""When to look again, and what to write down about what changed.

ROADMAP.md asks M4 for a "re verification scheduler with change logging". The
scheduling part is arithmetic on dates. The part worth care is the logging,
and it has two failure modes that matter more than a missed cadence.

**A failed lookup is not a change.** An RDAP server timing out, or a homepage
answering 503, says nothing about the publisher. If a re-verification turned
"we could not reach it" into an absent field, this project would be publishing
a claim produced by its own network conditions. So a kind that was attempted
and returned nothing is recorded in the change log as `NOT_RETURNED` and
**nothing is appended to the record**: the previous observation stands, with its
own older date, which is the honest state. `about.observe` already draws the
same line one level down, where an unfetchable page is not recorded as absence.

**An unchanged re-observation is still appended.** This costs rows and it buys
the only honest answer to "when did we last look at this". A card rendering
`latest(kind).observed_at` as freshness would be silently backdated if
re-reading a field that had not moved wrote nothing - the record would say the
masthead link was last seen in March when it was seen again yesterday. The
cadences in `ReverificationSettings` bound the cost: at 30 days, an about-page
field accumulates twelve rows a year per domain, which is a registry that can
answer the question rather than a registry that has grown.

So the record and the log answer different questions. The record is append only
and says what was observed and when, every time. The log is the transitions
only, and is what a reviewer reads to find the week a publisher's corrections
page disappeared.

**The change log holds values, not evidence.** An observation's `evidence` is
prose quoted from a publisher's page and it stays on the observation, where its
`source_url` and date sit beside it. A log entry carries the old and new value
and where they came from, because a log is read in bulk and a bulk read of
quoted prose is where someone starts pattern matching on publishers' writing -
which is the judgement D007 keeps out of this component.

Nothing here fetches. `reverify` takes observations the caller has already
produced, so the cadence logic is testable without a network and the caller
keeps robots, rate limiting and the guarded transport.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import structlog
from pydantic import BaseModel, Field

from pipeline.registry.config import ReverificationSettings
from pipeline.registry.schema import (
    DomainRecord,
    Observation,
    ObservationKind,
    ObservationValue,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

logger = structlog.get_logger(__name__)

# Which observer produces each kind, which is also which cadence applies.
RDAP_KINDS: Final = frozenset(
    {ObservationKind.REGISTERED_ON, ObservationKind.REGISTRAR},
)
ABOUT_KINDS: Final = frozenset(
    {
        ObservationKind.MASTHEAD_PRESENT,
        ObservationKind.CORRECTIONS_POLICY_PRESENT,
        ObservationKind.OWNERSHIP_STATED,
    },
)
# Kinds with no observer built. Listed rather than omitted so that adding an
# observer means moving a name here, and so that a kind cannot be scheduled for
# re-verification by code that has no way to re-verify it.
SCHEDULABLE_KINDS: Final = RDAP_KINDS | ABOUT_KINDS


class ChangeKind(StrEnum):
    """What a re-verification found, relative to what was already recorded."""

    FIRST_OBSERVATION = "first_observation"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    # Attempted and the observer returned nothing for this kind. Not absence:
    # see the module docstring. No observation is appended for this.
    NOT_RETURNED = "not_returned"


class Change(BaseModel):
    """One transition in one field of one domain's record.

    Frozen and `extra="forbid"`, for the reason `DomainRecord` is: a log entry
    is a record of what happened and an editable one is not, and a log is the
    other place a numeric field could be smuggled into this component.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    domain: str = Field(min_length=1)
    kind: ObservationKind
    change: ChangeKind
    previous_value: ObservationValue | None = None
    current_value: ObservationValue | None = None
    previous_observed_at: datetime | None = None
    observed_at: datetime
    # The page or server the new value came from. None for NOT_RETURNED, where
    # there is no new value and so no new source.
    source_url: str | None = None


class DomainDue(BaseModel):
    """A domain with the kinds whose cadence has elapsed."""

    model_config = {"frozen": True, "extra": "forbid"}

    domain: str = Field(min_length=1)
    kinds: tuple[ObservationKind, ...]
    # How long the oldest due field has been stale, which is what the queue is
    # ordered by. Zero for a field never observed, which sorts first by the
    # tie-break below rather than by this.
    stale_for: timedelta


def cadence(kind: ObservationKind, settings: ReverificationSettings) -> timedelta:
    """How long an observation of this kind stays current.

    Raises for a kind with no observer rather than returning a default: a
    default would quietly schedule work nothing can do.
    """
    if kind in RDAP_KINDS:
        return timedelta(days=settings.rdap_recheck_days)
    if kind in ABOUT_KINDS:
        return timedelta(days=settings.about_recheck_days)
    msg = f"no observer for {kind.value}, so no cadence"
    raise ValueError(msg)


def due(
    record: DomainRecord,
    now: datetime,
    settings: ReverificationSettings | None = None,
    kinds: Iterable[ObservationKind] | None = None,
) -> tuple[ObservationKind, ...]:
    """The kinds that should be looked at again, never-observed ones included.

    A kind that has never been observed is due, because the first observation
    is the one that makes a field exist. `DomainRecord.latest` distinguishes
    never-observed from observed-and-absent, and both are scheduled: the second
    because a publisher who had no corrections page in March may have one now,
    and that is the direction of change this component exists to catch.
    """
    settings = settings or ReverificationSettings()
    wanted = sorted(set(kinds) & SCHEDULABLE_KINDS) if kinds else sorted(SCHEDULABLE_KINDS)
    ready: list[ObservationKind] = []
    for kind in wanted:
        latest = record.latest(kind)
        if latest is None or now - latest.observed_at >= cadence(kind, settings):
            ready.append(kind)
    return tuple(ready)


def next_due_at(
    kind: ObservationKind,
    record: DomainRecord,
    settings: ReverificationSettings | None = None,
) -> datetime | None:
    """When this kind falls due, or None if it is due now or never observed."""
    settings = settings or ReverificationSettings()
    latest = record.latest(kind)
    if latest is None:
        return None
    return latest.observed_at + cadence(kind, settings)


def plan(
    records: Sequence[DomainRecord],
    now: datetime,
    settings: ReverificationSettings | None = None,
) -> tuple[DomainDue, ...]:
    """The next run's work, oldest first and capped.

    Ordered so that a run hitting the cap does the most overdue work rather
    than an alphabetical prefix of it - which would mean a domain late in the
    alphabet never being re-read at all once the registry outgrows one run.
    Domains missing the most fields come first, since a field nobody has looked
    at is worse than one looked at a while ago, and ties break on staleness.

    `stale_for` on each item is the age of the oldest field that *has* been
    observed, and is zero for a record where nothing has. It is a description
    of the work, not a ranking of the domain.
    """
    settings = settings or ReverificationSettings()
    queue: list[tuple[int, timedelta, DomainDue]] = []
    for record in records:
        kinds = due(record, now, settings)
        if not kinds:
            continue
        ages = [now - latest.observed_at for kind in kinds if (latest := record.latest(kind))]
        never = len(kinds) - len(ages)
        stale_for = max(ages) if ages else timedelta(0)
        queue.append(
            (never, stale_for, DomainDue(domain=record.domain, kinds=kinds, stale_for=stale_for))
        )
    # Most never-observed fields first, then the stalest. A sort key rather than
    # a field on DomainDue: the caller is given work to do, not a reason this
    # project ranked one domain above another.
    #
    # Counting the never-observed kinds rather than flagging them matters once
    # the registry is realistic, where almost every record has some kind nobody
    # has looked at yet - a boolean flag is true for nearly all of them and
    # orders nothing.
    queue.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(item[2] for item in queue[: settings.max_domains_per_run])


def reverify(
    record: DomainRecord,
    observed: Sequence[Observation],
    attempted: Iterable[ObservationKind],
    now: datetime,
) -> tuple[DomainRecord, tuple[Change, ...]]:
    """The record with the new observations appended, and what changed.

    `attempted` is the kinds the caller actually tried to observe, which is not
    the same as the kinds that came back. The difference is the whole point: a
    kind in `attempted` with nothing in `observed` is logged `NOT_RETURNED` and
    **appends nothing**, because an RDAP timeout or a 503 is a fact about a run
    and not about a publisher. A kind the caller did not attempt is not in the
    log at all.

    `now` is passed in rather than read from the clock so that a run's
    observations and its log entries carry one timestamp, and so this is
    testable. Observations keep their own `observed_at`, which is when the page
    was read.
    """
    fresh = {o.kind: o for o in observed}
    changes: list[Change] = []
    additions: list[Observation] = []

    for kind in sorted(set(attempted)):
        previous = record.latest(kind)
        current = fresh.get(kind)

        if current is None:
            changes.append(
                Change(
                    domain=record.domain,
                    kind=kind,
                    change=ChangeKind.NOT_RETURNED,
                    previous_value=previous.value if previous else None,
                    previous_observed_at=previous.observed_at if previous else None,
                    observed_at=now,
                )
            )
            logger.info(
                "reverification_returned_nothing",
                domain=record.domain,
                kind=kind.value,
                previously_observed=previous is not None,
            )
            continue

        additions.append(current)
        if previous is None:
            kind_of_change = ChangeKind.FIRST_OBSERVATION
        elif previous.value == current.value:
            kind_of_change = ChangeKind.UNCHANGED
        else:
            kind_of_change = ChangeKind.CHANGED
            # The line a reviewer greps for. Values rather than evidence, for
            # the reason in the module docstring.
            logger.info(
                "registry_field_changed",
                domain=record.domain,
                kind=kind.value,
                previous_value=str(previous.value),
                current_value=str(current.value),
                source_url=current.source_url,
            )
        changes.append(
            Change(
                domain=record.domain,
                kind=kind,
                change=kind_of_change,
                previous_value=previous.value if previous else None,
                current_value=current.value,
                previous_observed_at=previous.observed_at if previous else None,
                observed_at=now,
                source_url=current.source_url,
            )
        )

    updated = record
    for observation in additions:
        updated = updated.with_observation(observation)
    return updated, tuple(changes)
