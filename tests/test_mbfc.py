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
    is_low_quality,
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


def test_low_quality_is_a_flag_and_does_not_move_the_stratum() -> None:
    """The regression guard for D016's confound, resolved by D017.

    A right leaning outlet MBFC rates low credibility must stay in `partisan`
    and carry the flag. If credibility ever moves it again, `partisan` starts
    drifting left and `low_quality` becomes a political category.
    """
    obs = _obs("RIGHT", "LOW CREDIBILITY")
    assert derive_stratum(obs, None) == "partisan", "credibility must not move the stratum"
    assert is_low_quality(obs) is True
    assert derive_lean(obs) == "right"


def test_credibility_does_not_change_placement_either_way() -> None:
    """Same lean, opposite credibility, same stratum."""
    high = _obs("RIGHT", "HIGH CREDIBILITY")
    low = _obs("RIGHT", "LOW CREDIBILITY")
    assert derive_stratum(high, None) == derive_stratum(low, None) == "partisan"
    assert is_low_quality(high) is False
    assert is_low_quality(low) is True


def test_missing_credibility_is_not_a_good_rating() -> None:
    assert is_low_quality(_obs("LEFT", None)) is None


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


def test_non_lean_label_leaves_the_domain_unstratified() -> None:
    """QUESTIONABLE SOURCE is not a position on a spectrum.

    Better unstratified than placed somewhere nobody published.
    """
    obs = _obs("QUESTIONABLE SOURCE", "LOW CREDIBILITY")
    assert derive_stratum(obs, None) is None
    assert is_low_quality(obs) is True
