"""The rule based fallback. No network, no model."""

from __future__ import annotations

import pytest
from pipeline.claims.fallback import VERSION, classify, extract_claims_offline, sentence_spans
from pipeline.claims.schema import ClaimType


def test_a_decimal_point_does_not_end_a_sentence() -> None:
    """The first version cut here and reported the fragment as a claim."""
    text = "Unemployment fell to 4.1% in March. The rate held."
    assert [text[a:b] for a, b in sentence_spans(text)] == [
        "Unemployment fell to 4.1% in March.",
        "The rate held.",
    ]


def test_an_abbreviation_does_not_end_a_sentence() -> None:
    text = "Dr. Smith said the ward was full. It reopened on Tuesday."
    assert len(sentence_spans(text)) == 2


def test_an_initial_does_not_end_a_sentence() -> None:
    text = "J. Baker chaired the inquiry into the collapse. It reported late."
    assert len(sentence_spans(text)) == 2


def test_offsets_always_resolve_into_the_source() -> None:
    """Even degraded, the extractor cannot emit text the source lacks."""
    text = "Some 2,400 homes flooded across three counties. The policy was a disgrace."
    for claim in extract_claims_offline(text):
        assert text[claim.start : claim.end] == claim.text


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The minister said the deficit had halved.", ClaimType.ATTRIBUTION),
        ("According to the report, emissions rose.", ClaimType.ATTRIBUTION),
        ("Rates will rise again before June.", ClaimType.PREDICTION),
        ("The policy was a disgrace.", ClaimType.OPINION),
        ("Some 2,400 homes were flooded.", ClaimType.FACTUAL_ASSERTION),
    ],
)
def test_type_cues(sentence: str, expected: ClaimType) -> None:
    assert classify(sentence) == expected


def test_attribution_beats_assertion_when_a_sentence_reports_speech() -> None:
    """The guideline's hardest boundary, and the order the cues are tested in."""
    assert classify("The office said unemployment fell to 4.1% in March.") == ClaimType.ATTRIBUTION


def test_a_sentence_with_no_checkable_content_is_skipped() -> None:
    assert classify("And so it goes on and on without end.") is None


def test_short_fragments_are_skipped_but_short_claims_are_not() -> None:
    """A 40 character floor dropped 'The policy was a disgrace.' in the first version."""
    claims = extract_claims_offline("Yes. The policy was a disgrace.")
    assert [c.text for c in claims] == ["The policy was a disgrace."]


def test_no_more_than_five_claims() -> None:
    text = " ".join(f"Some {i},400 homes were flooded in the county." for i in range(9))
    assert len(extract_claims_offline(text)) == 5


def test_the_fallback_marks_its_own_version() -> None:
    """A card must show which extractor produced it; degraded is not silent."""
    claims = extract_claims_offline("Some 2,400 homes were flooded across the county.")
    assert claims[0].extractor_version == VERSION
    assert VERSION != "v1"


def test_a_sentence_initial_capital_is_not_evidence_of_a_claim() -> None:
    """Every sentence starts with one, so it cannot be the proper noun cue."""
    assert classify("And so it goes on and on without end.") is None
    assert classify("But nothing much happened after that at all.") is None


def test_a_mid_sentence_proper_noun_still_counts() -> None:
    assert classify("The inquiry into Brussels reopened today.") == ClaimType.FACTUAL_ASSERTION
