"""Publisher marker stripping: the OOD construction rule from DATA.md.

These are not tidying assertions. The dateline prefix is the single strongest
publisher tell in the corpus the leakage analysis criticises, and letting one
through into the OOD suite would reintroduce the leakage this project exists
to document. Each case is a form real wire copy actually uses.
"""

from __future__ import annotations

import pytest
from pipeline.text.extract import extract_article, strip_publisher_markers

DATELINES = [
    "LONDON (Reuters) - Body follows.",
    "WASHINGTON (AP) - Body follows.",
    "NEW YORK, Jan 5 (Reuters) - Body follows.",
    "PARIS, France (CNN) -- Body follows.",
    "SAN FRANCISCO (Bloomberg) - Body follows.",
    "BRUSSELS (AFP) - Body follows.",
]

BYLINES = [
    "By Jane Doe\nBody follows.",
    "By Jane Doe, Staff Writer\nBody follows.",
    "Reporting by Li Wei and Ahmed Khan\nBody follows.",
    "Written by Maria Garcia\nBody follows.",
]


@pytest.mark.parametrize("raw", DATELINES)
def test_datelines_are_removed(raw: str) -> None:
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Body follows."
    assert removed, "removal must be recorded, not silent"


@pytest.mark.parametrize("raw", BYLINES)
def test_bylines_are_removed(raw: str) -> None:
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Body follows."
    assert "byline" in removed


def test_em_and_en_dash_datelines_both_match() -> None:
    """Wire copy uses typographic dashes; ASCII-only regexes miss most of it."""
    for dash in ("-", chr(0x2013), chr(0x2014)):
        out, removed = strip_publisher_markers(f"LONDON (Reuters) {dash} Body follows.")
        assert out.strip() == "Body follows.", f"dash {hex(ord(dash))} not handled"
        assert removed


def test_ordinary_sentence_is_not_mistaken_for_a_dateline() -> None:
    """The stripper must not eat real content.

    A false positive here silently deletes the first sentence of an article,
    which would be a worse corpus defect than the one being removed.
    """
    for keeper in (
        "The FBI (Federal Bureau of Investigation) said the report was wrong.",
        "NASA said the launch would proceed.",
        "In 2024 the committee - which had met twice - reported back.",
    ):
        out, removed = strip_publisher_markers(keeper)
        assert out.strip() == keeper, f"content destroyed: {out!r}"
        assert not removed


def test_removal_counts_are_reported_per_document() -> None:
    """Prevalence per stratum has to be reportable, not assumed."""
    _, removed = strip_publisher_markers(
        "By Jane Doe\nLONDON (Reuters) - One.\nCopyright 2026 Example Corp."
    )
    assert removed.get("byline") == 1
    assert removed.get("dateline") == 1
    assert removed.get("boilerplate") == 1


def test_end_to_end_extraction_strips_what_trafilatura_leaves() -> None:
    """Regression guard for the measured trafilatura gap.

    trafilatura alone returns the byline and dateline inside the body text.
    If this test ever fails open, the OOD suite is being built with publisher
    identity intact.
    """
    # Distinct paragraphs, long enough to clear the article length floor, so
    # this exercises the stripping rather than the gate. They must differ:
    # trafilatura deduplicates identical blocks, which collapsed an earlier
    # repeated fixture below the floor.
    filler = "".join(
        f"<p>{sentence}</p>"
        for sentence in (
            "Officials said the vote had been expected for several weeks.",
            "Consultation with residents closed at the end of last month.",
            "Local business groups had asked for a longer transition period.",
            "The finance committee published its recommendations in June.",
            "Two amendments were withdrawn before the final reading began.",
            "Spending on transport rises by four percent under the plan.",
            "Opposition members requested a recorded division on the motion.",
            "The measures take effect at the start of the next financial year.",
        )
    )
    html = (
        "<html><body><nav>Subscribe</nav><article>"
        "<h1>Council approves budget</h1>"
        "<p>By Jane Doe, Staff Writer</p>"
        "<p>LONDON (Reuters) - The council approved the budget on Tuesday.</p>"
        f"{filler}"
        "<p>Our Standards: The Thomson Reuters Trust Principles.</p>"
        "</article><footer>Copyright 2026</footer></body></html>"
    )
    article = extract_article(html)
    assert article is not None
    assert "Reuters" not in article.text, "agency name survived into the corpus"
    assert "Jane Doe" not in article.text, "byline survived into the corpus"
    assert "Subscribe" not in article.text
    assert "council approved the budget" in article.text


def test_navigation_only_page_is_not_an_article() -> None:
    """Measured: trafilatura returns "Menu" for a nav-only page.

    Without a length floor that fragment enters the corpus as a document, so
    the floor is a corpus construction control, not a nicety.
    """
    assert extract_article("<html><body><nav>Menu</nav></body></html>") is None


def test_short_fragment_is_rejected_but_real_article_survives() -> None:
    body = "The council approved the budget on Tuesday evening. " * 12
    html = f"<html><body><article><h1>Budget</h1><p>{body}</p></article></body></html>"
    article = extract_article(html)
    assert article is not None, "a full length article must survive the floor"
    assert len(article.text) >= 400


CAPS_BYLINES = [
    "DAVID MARCUS: Congress set to protect AM radio.",
    "JANE DOE: The council met on Tuesday.",
    "MARY-ANN O'BRIEN: Spending rose again.",
]

NOT_BYLINES = [
    "BREAKING: The council met on Tuesday.",
    "UPDATE: Spending rose again.",
    "The council met: it approved the budget.",
    "NASA: the launch will proceed.",
]


@pytest.mark.parametrize("raw", CAPS_BYLINES)
def test_caps_name_prefix_byline_is_removed(raw: str) -> None:
    """Measured on live Fox News copy: the "By" rule left these in place."""
    out, removed = strip_publisher_markers(raw)
    assert not out.strip().startswith(("DAVID", "JANE", "MARY")), out
    assert "caps_byline" in removed


@pytest.mark.parametrize("raw", NOT_BYLINES)
def test_single_word_prefixes_are_not_treated_as_bylines(raw: str) -> None:
    """A false positive here deletes the opening of a real article."""
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw, f"content destroyed: {out!r}"
    assert "caps_byline" not in removed
