"""Checkpoint and resume behaviour of the collection runner. No network."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.ood_collection import run as mod
from eval.ood_collection.collect import Document, Rejection


def _doc(domain: str, n: int) -> Document:
    return Document(
        doc_id=f"doc_{domain}_{n}",
        domain=domain,
        stratum="partisan",
        lean="left",
        low_quality=False,
        url=f"https://{domain}/{n}",
        title=None,
        published="2026-03-01T00:00:00+00:00",
        text="body",
        text_sha256="x",
        markers_removed={},
        had_invisible=False,
        collected_at="2026-09-11T00:00:00+00:00",
    )


def test_record_then_load_round_trips(tmp_path: Path) -> None:
    rej = Rejection()
    rej.record("before_snapshot")
    mod.record_domain(tmp_path, "a.com", [_doc("a.com", 1), _doc("a.com", 2)], rej)

    assert [d.doc_id for d in mod.load_documents(tmp_path)] == ["doc_a.com_1", "doc_a.com_2"]
    progress = mod.load_progress(tmp_path)
    assert progress["a.com"]["kept"] == 2
    assert progress["a.com"]["rejected"] == {"before_snapshot": 1}


def test_restart_skips_finished_publishers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart must never re-contact a host that was already collected."""
    publishers = [
        {"domain": "done.com", "stratum": "wire", "lean": None, "low_quality": None},
        {"domain": "todo.com", "stratum": "wire", "lean": None, "low_quality": None},
    ]
    mod.record_domain(tmp_path, "done.com", [_doc("done.com", 1)], Rejection())

    contacted: list[str] = []

    def fake_collect(publisher: dict[str, object], **kwargs: object) -> list[Document]:
        contacted.append(str(publisher["domain"]))
        return [_doc(str(publisher["domain"]), 1)]

    monkeypatch.setattr(mod, "load_publishers", lambda: publishers)
    monkeypatch.setattr(mod, "collect_domain", fake_collect)

    summary = mod.run(tmp_path, limit=5, domains=None)

    assert contacted == ["todo.com"]
    assert summary["documents"] == 2
    assert summary["domains"] == 2


def test_zero_yield_publishers_are_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publishers = [{"domain": "empty.com", "stratum": "wire", "lean": None, "low_quality": None}]
    monkeypatch.setattr(mod, "load_publishers", lambda: publishers)
    monkeypatch.setattr(mod, "collect_domain", lambda publisher, **kw: [])

    summary = mod.run(tmp_path, limit=5, domains=None)

    assert summary["publishers_yielding_nothing"] == ["empty.com"]
