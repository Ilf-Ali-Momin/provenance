"""Corpus residue audit. No network."""

from __future__ import annotations

from eval.ood_collection.audit import audit
from eval.ood_collection.collect import Document


def _doc(domain: str, text: str, removed: dict[str, int] | None = None) -> Document:
    return Document(
        doc_id=f"doc_{domain}_{hash(text) & 0xFFFF}",
        domain=domain,
        stratum="partisan",
        lean="right",
        low_quality=False,
        url=f"https://{domain}/politics/a",
        title=None,
        published="2026-03-01T00:00:00+00:00",
        text=text,
        text_sha256="x",
        markers_removed=removed or {},
        had_invisible=False,
        collected_at="2026-09-11T00:00:00+00:00",
    )


NAMES = {"foxnews.com": ("Fox News",), "vox.com": ("Vox",)}


def test_clean_masked_text_reports_no_residue() -> None:
    report = audit(
        [_doc("foxnews.com", "[OUTLET] reported the vote passed.", {"outlet_name": 1})], NAMES
    )
    pub = report["by_publisher"]["foxnews.com"]  # type: ignore[index]
    assert pub["documents_with_residue"] == {}
    assert pub["removed_by_construction"] == {"outlet_name": 1}


def test_own_name_in_a_case_masking_missed_is_flagged() -> None:
    """Masking is case sensitive; the audit deliberately is not."""
    report = audit([_doc("foxnews.com", "WATCH: FOX NEWS SPECIAL tonight at nine.")], NAMES)
    assert report["documents_with_residue"]["own_name_unmasked"] == 1  # type: ignore[index]


def test_another_outlets_name_is_not_flagged() -> None:
    """Fox News inside a Vox article is content, not Vox's identity."""
    report = audit([_doc("vox.com", "Fox News aired the segment twice.")], NAMES)
    assert "own_name_unmasked" not in report["documents_with_residue"]  # type: ignore[operator]


def test_marker_residue_is_counted_per_document_with_snippets() -> None:
    docs = [
        _doc("vox.com", "WASHINGTON (AP) - Lawmakers met."),
        _doc("vox.com", "Photo: Jane Doe for the paper. Subscribe to our newsletter."),
        _doc("vox.com", "Nothing to see."),
    ]
    report = audit(docs, NAMES)
    residue = report["by_publisher"]["vox.com"]["documents_with_residue"]  # type: ignore[index]
    assert residue["wire_tag"] == 1
    assert residue["photo_credit"] == 1
    assert residue["newsletter_promo"] == 1
    snippets = report["by_publisher"]["vox.com"]["snippets"]  # type: ignore[index]
    assert all(len(s) <= 80 for group in snippets.values() for s in group)


def test_prose_that_looks_like_a_marker_is_not_flagged() -> None:
    """False positives from the pilot 2 review, each one prose rather than residue."""
    names = {"bostonglobe.com": ("The Boston Globe", "the Globe")}
    docs = [
        _doc("bostonglobe.com", "His image beamed around the globe within the hour."),
        _doc(
            "bostonglobe.com", "They were all showing the same image: a World Trade Center tower."
        ),
        _doc("bostonglobe.com", "The publisher has sued OpenAI for copyright violations."),
    ]
    assert audit(docs, names)["documents_with_residue"] == {}


def test_real_markers_are_still_flagged() -> None:
    names = {"bostonglobe.com": ("The Boston Globe", "the Globe")}
    docs = [
        _doc("bostonglobe.com", "As the Globe reported last week, the vote failed."),
        _doc("bostonglobe.com", "Photo by Jane Doe. Copyright 2026 Example Media."),
    ]
    residue = audit(docs, names)["documents_with_residue"]
    assert residue["own_name_unmasked"] == 1
    assert residue["photo_credit"] == 1
    assert residue["copyright"] == 1


# --- precision fixes from the spread run, 2026-09-15 ------------------------


def _hits(text: str, names: dict[str, tuple[str, ...]] | None = None) -> dict[str, int]:
    report = audit([_doc("example.com", text)], names or {})
    residue = report["documents_with_residue"]
    assert isinstance(residue, dict)
    return residue


def test_a_weekday_sentence_is_not_reported_as_a_byline() -> None:
    assert "by_line" not in _hits("By Friday afternoon, Hezbollah had issued a statement.")
    assert "by_line" in _hits("By Michaila Byrne\nThe report says.")


def test_an_acronym_expansion_is_not_reported_as_a_wire_tag() -> None:
    """Advanced Placement (AP) is not the Associated Press."""
    assert "wire_tag" not in _hits("Advanced Placement (AP) and Baccalaureate courses count.")
    assert "wire_tag" not in _hits("according to The Associated Press (AP).")
    assert "wire_tag" in _hits("LONDON (Reuters) - Shares fell on Tuesday.")


def test_prose_about_newsletters_is_not_reported_as_boilerplate() -> None:
    assert "newsletter_promo" not in _hits("It competes with notes from friends, newsletters.")
    assert "newsletter_promo" in _hits("Sign up for our newsletter to get it first.")


def test_a_document_that_is_itself_a_newsletter_is_reported_separately() -> None:
    """A collection defect, not a stripping one, so it gets its own name."""
    issue = "Read all the way through the newsletter. We won't have a newsletter next week."
    assert "newsletter_issue" in _hits(issue)
    assert "newsletter_issue" not in _hits("It competes with notes from friends, newsletters.")


def test_a_proper_compound_is_not_the_outlet_naming_itself() -> None:
    names = {"example.com": ("The Atlantic",)}
    assert "own_name_unmasked" not in _hits(
        "a year without a hurricane in the Atlantic Ocean.", names
    )
    assert "own_name_unmasked" in _hits("You can subscribe to the Atlantic today.", names)


def test_a_link_back_into_the_publishers_own_site_is_reported() -> None:
    """Checked against the document's host, so a cited third party link is not."""
    own = audit(
        [_doc("bbc.com", "More at https://www.bbc.co.uk/learningenglish/newsletters here.")], {}
    )
    other = audit([_doc("bbc.com", "The filing is at https://supremecourt.gov/o/1.pdf.")], {})
    assert "self_link" in own["documents_with_residue"]  # type: ignore[operator]
    assert "self_link" not in other["documents_with_residue"]  # type: ignore[operator]
