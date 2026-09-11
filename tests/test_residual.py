"""Residual separability check. Tiny synthetic corpora, no network."""

from __future__ import annotations

from eval.ood_collection.collect import Document
from eval.ood_collection.residual import majority_baseline, residual_report


def _doc(domain: str, stratum: str, text: str, n: int) -> Document:
    return Document(
        doc_id=f"doc_{domain}_{n}",
        domain=domain,
        stratum=stratum,
        lean=None,
        low_quality=None,
        url=f"https://{domain}/{n}",
        title=None,
        published="2026-03-01T00:00:00+00:00",
        text=text,
        text_sha256=f"h{n}",
        markers_removed={},
        had_invisible=False,
        collected_at="2026-09-11T00:00:00+00:00",
    )


def test_majority_baseline() -> None:
    assert majority_baseline(["a", "a", "b", "c"]) == 0.5
    assert majority_baseline([]) == 0.0


def test_publisher_with_a_signature_token_is_identifiable() -> None:
    """If identity survives in the text, the check must say so."""
    docs = []
    for domain, token in (("a.com", "qqalpha"), ("b.com", "qqbravo"), ("c.com", "qqcharlie")):
        for i in range(8):
            docs.append(_doc(domain, "partisan", f"{token} the council met on day {i} again", i))
    report = residual_report(docs)
    identity = report["masked_text"]["publisher_identity"]  # type: ignore[index]
    assert identity["accuracy"] >= 0.9
    assert identity["majority_baseline"] < 0.4


def test_identical_shapes_are_not_separable_on_structure() -> None:
    """Different words, same shape: the structure-only view must see nothing."""
    words = {"a.com": "abcde fghij", "b.com": "klmno pqrst", "c.com": "uvwxy zabcd"}
    docs = [_doc(d, "partisan", w, i) for d, w in words.items() for i in range(8)]
    report = residual_report(docs)
    shape = report["structure_only"]["publisher_identity"]  # type: ignore[index]
    assert shape["accuracy"] <= 0.5


def test_thin_publishers_are_named_not_hidden() -> None:
    docs = [_doc("solo.com", "wire", "one lonely article", 0)]
    docs += [_doc("b.com", "wire", f"text {i}", i) for i in range(3)]
    report = residual_report(docs)
    assert report["publishers_excluded_for_too_few_documents"] == ["solo.com"]


def test_single_stratum_reports_none_rather_than_a_number() -> None:
    docs = [_doc(f"d{i}.com", "wire", f"body {j}", j) for i in range(3) for j in range(3)]
    report = residual_report(docs)
    grouped = report["masked_text"]["stratum_grouped_by_publisher"]  # type: ignore[index]
    assert grouped["accuracy"] is None
