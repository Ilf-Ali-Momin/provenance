"""MBFC parsing and stratum derivation. Pure logic, no network.

The derivation rules matter more than the parsing: they decide what the OOD
strata actually mean, and a rule that quietly encodes political lean into a
stratum labelled "quality" would make every downstream number uninterpretable.
"""

from __future__ import annotations

import pytest
from eval.ood_collection.mbfc import (
    MbfcObservation,
    derive_lean,
    derive_stratum,
    parse_ratings,
    slug_for,
)


def _obs(bias: str | None, credibility: str | None, *, found: bool = True) -> MbfcObservation:
    return MbfcObservation(
        domain="example.com",
        slug="example",
        bias=bias,
        factual_reporting="HIGH",
        credibility=credibility,
        source_url="https://mediabiasfactcheck.com/example/",
        observed_at="2026-08-31",
        found=found,
    )


PAGE = (
    "<div><span>Bias Rating: LEFT-CENTER</span>"
    "<span>Factual Reporting: HIGH</span>"
    "<span>Credibility Rating: HIGH CREDIBILITY</span></div>"
)


def test_parse_ratings_extracts_labels_only() -> None:
    r = parse_ratings(PAGE)
    assert r == {
        "bias": "LEFT-CENTER",
        "factual_reporting": "HIGH",
        "credibility": "HIGH CREDIBILITY",
    }


def test_parse_ratings_returns_none_when_absent() -> None:
    assert parse_ratings("<p>Nothing here</p>") == {
        "bias": None,
        "factual_reporting": None,
        "credibility": None,
    }


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("www.reuters.com", "reuters"), ("theguardian.com", "theguardian"), ("bbc.co.uk", "bbc")],
)
def test_slug_for_normalises(domain: str, expected: str) -> None:
    assert slug_for(domain) == expected


def test_factual_category_wins_over_mbfc_label() -> None:
    """ "Reuters is a wire service" is observable; its bias rating is MBFC's."""
    obs = _obs("LEAST BIASED", "HIGH CREDIBILITY")
    assert derive_stratum(obs, "wire") == "wire"
    assert derive_stratum(obs, "regional") == "regional"


def test_unknown_domain_is_left_unstratified() -> None:
    """Never assign a label nobody published."""
    obs = _obs(None, None, found=False)
    assert derive_stratum(obs, None) is None
    assert derive_lean(obs) is None


def test_lean_is_recorded_independently_of_stratum() -> None:
    """The regression guard for the measured confound.

    A right leaning outlet MBFC rates low credibility lands in `low_quality`,
    but its lean must still be recorded. Without this, `low_quality` silently
    becomes a political category and nothing downstream can tell.
    """
    obs = _obs("RIGHT", "LOW CREDIBILITY")
    assert derive_stratum(obs, None) == "low_quality"
    assert derive_lean(obs) == "right", "lean must survive routing into low_quality"


@pytest.mark.parametrize(
    ("bias", "lean"),
    [
        ("EXTREME LEFT", "left"),
        ("LEFT", "left"),
        ("LEFT-CENTER", "lean_left"),
        ("LEAST BIASED", "center"),
        ("RIGHT-CENTER", "lean_right"),
        ("RIGHT", "right"),
        ("EXTREME RIGHT", "right"),
    ],
)
def test_lean_mapping_is_symmetric(bias: str, lean: str) -> None:
    """Asymmetric bucketing would bias the stratum before sampling begins."""
    assert derive_lean(_obs(bias, "HIGH CREDIBILITY")) == lean


def test_credibility_routing_is_what_pulls_lean_out_of_partisan() -> None:
    """Documents the mechanism behind the measured 15:4 skew.

    Same bias label, different credibility, different stratum. This is not a
    bug in the code; it is the property of the rule that made the partisan
    stratum one sided, and it is asserted here so the behaviour cannot change
    silently.
    """
    high = _obs("RIGHT", "HIGH CREDIBILITY")
    low = _obs("RIGHT", "LOW CREDIBILITY")
    assert derive_stratum(high, None) == "partisan"
    assert derive_stratum(low, None) == "low_quality"
    assert derive_lean(high) == derive_lean(low) == "right"
