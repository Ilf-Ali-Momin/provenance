"""Re-applying text rules to a stored run. No network."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.ood_collection import reclean as mod
from eval.ood_collection.collect import Document, Rejection
from eval.ood_collection.run import load_documents, record_domain

FILLER = " ".join(f"The committee said that item {i} was ready for a vote." for i in range(12))


def _doc(domain: str, text: str, n: int = 0) -> Document:
    return Document(
        doc_id=f"doc_{domain}_{n}",
        domain=domain,
        stratum="partisan",
        lean="right",
        low_quality=True,
        url=f"https://{domain}/politics/{n}",
        title=None,
        published="2026-03-01T00:00:00+00:00",
        text=text,
        text_sha256="old",
        markers_removed={"outlet_name": 2},
        had_invisible=False,
        collected_at="2026-09-11T00:00:00+00:00",
    )


def test_new_rules_apply_to_stored_text() -> None:
    stored = f"{FILLER} Follow us on TikTok and Instagram."
    cleaned, report = mod.reclean([_doc("cnbc.com", stored)], {})
    assert "TikTok" not in cleaned[0].text
    assert cleaned[0].markers_removed == {"outlet_name": 2, "follow_or_contact": 1}
    assert report["second_pass_removed"] == {"follow_or_contact": 1}


def test_hashes_and_ids_follow_the_new_text() -> None:
    original = _doc("cnbc.com", f"{FILLER} Follow us on TikTok.")
    cleaned, _ = mod.reclean([original], {})
    assert cleaned[0].text_sha256 != "old"
    assert cleaned[0].doc_id != original.doc_id


def test_provenance_fields_are_untouched() -> None:
    original = _doc("breitbart.com", FILLER)
    cleaned, _ = mod.reclean([original], {})
    kept = cleaned[0]
    assert (kept.domain, kept.stratum, kept.lean, kept.low_quality, kept.url, kept.published) == (
        original.domain,
        original.stratum,
        original.lean,
        original.low_quality,
        original.url,
        original.published,
    )


def test_documents_under_the_floor_after_cleaning_are_dropped_and_counted() -> None:
    mostly_boilerplate = "Short piece. Follow us on Facebook, Instagram and X for more."
    cleaned, report = mod.reclean([_doc("nbcnews.com", mostly_boilerplate)], {})
    assert cleaned == []
    assert report["dropped_below_floor"] == 1


def test_already_clean_text_is_unchanged() -> None:
    """Idempotent: a second pass over clean text removes nothing."""
    cleaned, report = mod.reclean([_doc("vox.com", FILLER)], {})
    assert cleaned[0].text == FILLER
    assert report["second_pass_removed"] == {}


def test_cli_writes_a_new_run_and_refuses_an_existing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    record_domain(source, "cnbc.com", [_doc("cnbc.com", f"{FILLER} Follow us on X.")], Rejection())
    (source / "pipeline_sha.txt").write_text("aaa1111\n")
    monkeypatch.setattr(mod, "_head_sha", lambda: "bbb2222")
    monkeypatch.setattr(mod, "load_outlet_names", lambda: {})

    dest = tmp_path / "clean"
    assert mod.main([str(source), str(dest)]) == 0
    assert (dest / "pipeline_sha.txt").read_text().strip() == "aaa1111+reclean:bbb2222"
    assert "Follow us" not in load_documents(dest)[0].text
    assert (dest / "progress.jsonl").exists()

    assert mod.main([str(source), str(dest)]) == 1
