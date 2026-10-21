"""Re-verification cadences, and the claims the change log refuses to make.

The arithmetic here is simple. Most of these tests are about the two cases
where a wrong answer becomes a claim about a publisher: a lookup that failed,
and a field re-read without changing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pipeline.registry.config import ReverificationSettings
from pipeline.registry.schedule import (
    ABOUT_KINDS,
    RDAP_KINDS,
    SCHEDULABLE_KINDS,
    Change,
    ChangeKind,
    cadence,
    due,
    next_due_at,
    plan,
    reverify,
)
from pipeline.registry.schema import (
    FORBIDDEN_FIELDS,
    DomainRecord,
    Observation,
    ObservationKind,
    Source,
)
from pydantic import ValidationError

NOW = datetime(2026, 10, 21, tzinfo=UTC)
BASE = "https://example.org/"
SETTINGS = ReverificationSettings()


def masthead(value: bool, when: datetime, url: str = "https://example.org/about") -> Observation:
    return Observation(
        kind=ObservationKind.MASTHEAD_PRESENT,
        value=value,
        source=Source.ABOUT_PAGE,
        source_url=url,
        observed_at=when,
        evidence='Linked as "Our team".',
    )


def registered(when: datetime, on: date = date(2019, 3, 4)) -> Observation:
    return Observation(
        kind=ObservationKind.REGISTERED_ON,
        value=on,
        source=Source.RDAP,
        source_url="https://rdap.example/domain/example.org",
        observed_at=when,
    )


def test_a_kind_with_no_observer_is_not_schedulable() -> None:
    # Scheduling work nothing can do would fill the queue with domains that
    # never come off it.
    assert ObservationKind.TRANSPARENCY_DATASET_LISTING not in SCHEDULABLE_KINDS
    assert ObservationKind.PUBLISHER_METADATA not in SCHEDULABLE_KINDS
    with pytest.raises(ValueError, match="no observer"):
        cadence(ObservationKind.PUBLISHER_METADATA, SETTINGS)


def test_about_pages_are_rechecked_more_often_than_rdap() -> None:
    # A registration date is immutable; a website's prose is not.
    assert cadence(ObservationKind.MASTHEAD_PRESENT, SETTINGS) < cadence(
        ObservationKind.REGISTERED_ON, SETTINGS
    )


def test_a_never_observed_field_is_due() -> None:
    assert due(DomainRecord(domain="example.org"), NOW, SETTINGS) == tuple(
        sorted(SCHEDULABLE_KINDS)
    )


def test_a_fresh_observation_is_not_due_and_a_stale_one_is() -> None:
    fresh = DomainRecord(domain="example.org").with_observation(
        masthead(True, NOW - timedelta(days=1))
    )
    assert ObservationKind.MASTHEAD_PRESENT not in due(fresh, NOW, SETTINGS)

    stale = DomainRecord(domain="example.org").with_observation(
        masthead(True, NOW - timedelta(days=SETTINGS.about_recheck_days + 1))
    )
    assert ObservationKind.MASTHEAD_PRESENT in due(stale, NOW, SETTINGS)


def test_an_observed_absence_is_rescheduled() -> None:
    # A publisher who had no corrections page in March may have one now, and
    # that direction of change is what this component exists to catch.
    record = DomainRecord(domain="example.org").with_observation(
        masthead(False, NOW - timedelta(days=SETTINGS.about_recheck_days + 1))
    )
    assert ObservationKind.MASTHEAD_PRESENT in due(record, NOW, SETTINGS)


def test_next_due_at_is_none_for_a_field_never_observed() -> None:
    record = DomainRecord(domain="example.org")
    assert next_due_at(ObservationKind.MASTHEAD_PRESENT, record, SETTINGS) is None

    observed = record.with_observation(masthead(True, NOW))
    assert next_due_at(ObservationKind.MASTHEAD_PRESENT, observed, SETTINGS) == NOW + timedelta(
        days=SETTINGS.about_recheck_days
    )


def test_a_capped_run_does_the_most_overdue_work() -> None:
    # Alphabetical would mean a domain late in the alphabet never being re-read
    # once the registry outgrows one run.
    settings = ReverificationSettings(max_domains_per_run=2)
    records = [
        DomainRecord(domain=f"{letter}.example").with_observation(
            masthead(True, NOW - timedelta(days=30 + age))
        )
        for letter, age in (("a", 1), ("b", 400), ("c", 200))
    ]
    queued = plan(records, NOW, settings)
    assert [item.domain for item in queued] == ["b.example", "c.example"]


def test_a_never_observed_domain_is_queued_ahead_of_a_stale_one() -> None:
    settings = ReverificationSettings(max_domains_per_run=1)
    stale = DomainRecord(domain="old.example").with_observation(
        masthead(True, NOW - timedelta(days=4000))
    )
    unseen = DomainRecord(domain="new.example")
    assert [item.domain for item in plan([stale, unseen], NOW, settings)] == ["new.example"]


def test_a_domain_with_nothing_due_is_not_queued() -> None:
    record = DomainRecord(domain="example.org")
    for kind in SCHEDULABLE_KINDS:
        record = record.with_observation(
            Observation(
                kind=kind,
                value=True if kind in ABOUT_KINDS else "Example Registrar",
                source=Source.ABOUT_PAGE if kind in ABOUT_KINDS else Source.RDAP,
                source_url="https://example.org/about",
                observed_at=NOW,
            )
            if kind != ObservationKind.REGISTERED_ON
            else registered(NOW)
        )
    assert plan([record], NOW, SETTINGS) == ()


def test_a_first_observation_is_logged_as_one_and_appended() -> None:
    record = DomainRecord(domain="example.org")
    updated, changes = reverify(
        record, [masthead(True, NOW)], [ObservationKind.MASTHEAD_PRESENT], NOW
    )
    assert [c.change for c in changes] == [ChangeKind.FIRST_OBSERVATION]
    assert changes[0].previous_value is None
    assert changes[0].current_value is True
    assert len(updated.observations) == 1


def test_an_unchanged_reobservation_is_still_appended() -> None:
    # The cost is a row. What it buys is the only honest answer to "when did we
    # last look at this": suppressing it would leave `latest().observed_at`
    # reading as freshness while naming a date months old.
    earlier = NOW - timedelta(days=40)
    record = DomainRecord(domain="example.org").with_observation(masthead(True, earlier))
    updated, changes = reverify(
        record, [masthead(True, NOW)], [ObservationKind.MASTHEAD_PRESENT], NOW
    )
    assert [c.change for c in changes] == [ChangeKind.UNCHANGED]
    assert len(updated.observations) == 2
    latest = updated.latest(ObservationKind.MASTHEAD_PRESENT)
    assert latest is not None
    assert latest.observed_at == NOW


def test_a_changed_field_keeps_the_old_observation() -> None:
    earlier = NOW - timedelta(days=40)
    record = DomainRecord(domain="example.org").with_observation(masthead(True, earlier))
    updated, changes = reverify(
        record, [masthead(False, NOW)], [ObservationKind.MASTHEAD_PRESENT], NOW
    )
    assert changes[0].change is ChangeKind.CHANGED
    assert changes[0].previous_value is True
    assert changes[0].current_value is False
    assert changes[0].previous_observed_at == earlier
    history = updated.history(ObservationKind.MASTHEAD_PRESENT)
    assert [o.value for o in history] == [True, False]


def test_a_failed_lookup_is_not_recorded_as_absence() -> None:
    # The failure mode that would make this project publish a claim produced by
    # its own network conditions. The previous observation stands, with its own
    # older date.
    earlier = NOW - timedelta(days=200)
    record = DomainRecord(domain="example.org").with_observation(registered(earlier))
    updated, changes = reverify(record, [], [ObservationKind.REGISTERED_ON], NOW)

    assert [c.change for c in changes] == [ChangeKind.NOT_RETURNED]
    assert changes[0].current_value is None
    assert changes[0].source_url is None
    assert updated.observations == record.observations
    latest = updated.latest(ObservationKind.REGISTERED_ON)
    assert latest is not None
    assert latest.observed_at == earlier


def test_a_kind_that_was_not_attempted_is_absent_from_the_log() -> None:
    record = DomainRecord(domain="example.org")
    _, changes = reverify(record, [masthead(True, NOW)], [ObservationKind.MASTHEAD_PRESENT], NOW)
    assert {c.kind for c in changes} == {ObservationKind.MASTHEAD_PRESENT}


def test_the_change_log_carries_no_evidence_prose() -> None:
    # Evidence is prose quoted from a publisher's page and stays on the
    # observation beside its source and date. A log is read in bulk, and a bulk
    # read of quoted prose is where someone starts pattern matching on
    # publishers' writing.
    assert "evidence" not in Change.model_fields


def test_the_change_log_cannot_hold_a_score() -> None:
    assert FORBIDDEN_FIELDS.isdisjoint(Change.model_fields)
    with pytest.raises(ValidationError):
        Change(
            domain="example.org",
            kind=ObservationKind.MASTHEAD_PRESENT,
            change=ChangeKind.UNCHANGED,
            observed_at=NOW,
            credibility=0.8,  # type: ignore[call-arg]
        )


def test_rdap_and_about_kinds_do_not_overlap() -> None:
    assert RDAP_KINDS.isdisjoint(ABOUT_KINDS)


HOMEPAGE_BEFORE = """
<html><body><footer>
  <a href="/about-us">About us</a>
  <a href="/our-team">Our team</a>
</footer></body></html>
"""

HOMEPAGE_AFTER = """
<html><body><footer>
  <a href="/about-us">About us</a>
  <a href="/our-team">Our team</a>
  <a href="/corrections">Corrections</a>
</footer></body></html>
"""

PAGES = {
    "https://example.org/about-us": (
        "<html><title>About us</title><body>Owned by Example Ltd.</body></html>"
    ),
    "https://example.org/our-team": (
        "<html><title>Our team</title><body>Editor: A. Name.</body></html>"
    ),
    "https://example.org/corrections": (
        "<html><title>Corrections</title><body>We correct errors promptly.</body></html>"
    ),
}


def test_a_publisher_adding_a_corrections_page_shows_as_one_changed_field() -> None:
    # Real observations from `about.observe` rather than hand-built ones, so
    # the transition this logs is the one the observer actually produces -
    # including its `False` absence, which is an observation and not a gap.
    from pipeline.registry.about import observe

    first = NOW - timedelta(days=40)
    before = observe(HOMEPAGE_BEFORE, BASE, first, PAGES.__getitem__)
    record, opening = reverify(
        DomainRecord(domain="example.org"), before, [o.kind for o in before], first
    )
    assert {c.change for c in opening} == {ChangeKind.FIRST_OBSERVATION}
    corrections = record.latest(ObservationKind.CORRECTIONS_POLICY_PRESENT)
    assert corrections is not None
    assert corrections.value is False

    after = observe(HOMEPAGE_AFTER, BASE, NOW, PAGES.__getitem__)
    record, second = reverify(record, after, [o.kind for o in after], NOW)
    by_kind = {c.kind: c.change for c in second}
    assert by_kind[ObservationKind.CORRECTIONS_POLICY_PRESENT] is ChangeKind.CHANGED
    assert by_kind[ObservationKind.MASTHEAD_PRESENT] is ChangeKind.UNCHANGED
    assert by_kind[ObservationKind.OWNERSHIP_STATED] is ChangeKind.UNCHANGED

    # The old absence is still there, which is what makes the history worth
    # having: the week a corrections page appeared is answerable.
    history = record.history(ObservationKind.CORRECTIONS_POLICY_PRESENT)
    assert [o.value for o in history] == [False, True]
    changed = record.latest(ObservationKind.CORRECTIONS_POLICY_PRESENT)
    assert changed is not None
    assert changed.source_url == "https://example.org/corrections"
