"""The registry's schema, and the score it cannot express.

D007 is marked legally load bearing. These tests are where that stops being a
policy and becomes a property of the program.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pipeline.registry.schema import (
    FORBIDDEN_FIELDS,
    DomainRecord,
    Observation,
    ObservationKind,
    Source,
)
from pydantic import ValidationError

WHEN = datetime(2026, 9, 27, tzinfo=UTC)


def _observation(**over: object) -> Observation:
    fields: dict[str, object] = {
        "kind": ObservationKind.MASTHEAD_PRESENT,
        "value": True,
        "source": Source.ABOUT_PAGE,
        "source_url": "https://example.org/about",
        "observed_at": WHEN,
    }
    fields.update(over)
    return Observation(**fields)  # type: ignore[arg-type]


def test_the_record_has_no_field_that_could_hold_a_score() -> None:
    """D007: no computed score, no tier, no label."""
    fields = set(DomainRecord.model_fields)
    assert not (fields & FORBIDDEN_FIELDS), f"score-shaped field added: {fields & FORBIDDEN_FIELDS}"


def test_a_score_cannot_be_added_to_a_record() -> None:
    """extra=forbid, so this raises rather than storing quietly."""
    with pytest.raises(ValidationError):
        DomainRecord(domain="example.org", credibility=0.8)  # type: ignore[call-arg]


def test_no_observation_kind_is_a_judgement() -> None:
    """Each kind names something a reader could check from the recorded source."""
    for kind in ObservationKind:
        assert not (set(kind.value.split("_")) & FORBIDDEN_FIELDS)


def test_an_observation_without_provenance_is_refused() -> None:
    """An observation whose origin cannot be revisited is this project asserting
    something about a named organisation, which is what D007 rules out."""
    with pytest.raises(ValidationError):
        Observation(  # type: ignore[call-arg]
            kind=ObservationKind.MASTHEAD_PRESENT, value=True, observed_at=WHEN
        )


def test_an_empty_source_url_is_refused() -> None:
    with pytest.raises(ValidationError):
        _observation(source_url="")


def test_an_observation_cannot_be_edited_after_the_fact() -> None:
    observation = _observation()
    with pytest.raises(ValidationError):
        observation.value = False  # type: ignore[misc]


def test_observations_are_appended_never_replaced() -> None:
    """A registry that quietly revised itself could not answer what was observed
    and when, which is the question that matters when something is disputed."""
    first = _observation(value=False, observed_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = _observation(value=True, observed_at=datetime(2026, 6, 1, tzinfo=UTC))
    record = DomainRecord(domain="example.org").with_observation(first).with_observation(second)
    assert len(record.observations) == 2
    assert len(record.history(ObservationKind.MASTHEAD_PRESENT)) == 2


def test_latest_returns_the_most_recent_not_the_last_added() -> None:
    late = _observation(observed_at=datetime(2026, 6, 1, tzinfo=UTC), value=True)
    early = _observation(observed_at=datetime(2026, 1, 1, tzinfo=UTC), value=False)
    record = DomainRecord(domain="example.org").with_observation(late).with_observation(early)
    latest = record.latest(ObservationKind.MASTHEAD_PRESENT)
    assert latest is not None
    assert latest.value is True


def test_history_is_oldest_first() -> None:
    late = _observation(observed_at=datetime(2026, 6, 1, tzinfo=UTC))
    early = _observation(observed_at=datetime(2026, 1, 1, tzinfo=UTC))
    record = DomainRecord(domain="example.org").with_observation(late).with_observation(early)
    dates = [o.observed_at for o in record.history(ObservationKind.MASTHEAD_PRESENT)]
    assert dates == sorted(dates)


def test_never_observed_returns_none_not_a_default() -> None:
    """ "We have not looked" and "we looked and found nothing" are different
    answers, and a card rendering them alike misleads about its own coverage."""
    record = DomainRecord(domain="example.org")
    assert record.latest(ObservationKind.CORRECTIONS_POLICY_PRESENT) is None

    looked = record.with_observation(
        _observation(kind=ObservationKind.CORRECTIONS_POLICY_PRESENT, value=False)
    )
    found = looked.latest(ObservationKind.CORRECTIONS_POLICY_PRESENT)
    assert found is not None
    assert found.value is False


def test_a_registration_date_is_carried_as_a_date() -> None:
    observation = _observation(
        kind=ObservationKind.REGISTERED_ON,
        value=date(2019, 3, 14),
        source=Source.RDAP,
        source_url="https://rdap.example/domain/example.org",
    )
    assert observation.value == date(2019, 3, 14)


def test_evidence_is_bounded() -> None:
    """A quote, not a copy of the page."""
    with pytest.raises(ValidationError):
        _observation(evidence="x" * 501)


def test_an_unknown_observation_kind_is_refused() -> None:
    """The kinds are a closed set: an open string is where someone eventually
    writes "credibility"."""
    with pytest.raises(ValidationError):
        _observation(kind="credibility_rating")
