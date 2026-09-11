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
    assert residue["newsletter"] == 1
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
