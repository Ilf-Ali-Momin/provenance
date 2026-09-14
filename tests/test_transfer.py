"""Ablation 4 reporting. Pure logic; the ISOT fit is stubbed, no download."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eval.leakage import run_transfer as mod


def _row(domain: str, stratum: str, lean: str | None, low_quality: bool, text: str) -> dict:
    return {
        "domain": domain,
        "stratum": stratum,
        "lean": lean,
        "low_quality": low_quality,
        "text": text,
        "split": "train",
    }


def test_wire_share_counts_the_corpus_true_class() -> None:
    assert mod.wire_share(np.array([1, 1, 0, 0])) == 0.5
    assert mod.wire_share(np.array([1, 1])) == 1.0
    assert np.isnan(mod.wire_share(np.array([], dtype=int)))


def test_rates_by_groups_and_counts() -> None:
    rows = [
        _row("a.com", "wire", None, False, "x"),
        _row("b.com", "partisan", "left", False, "y"),
        _row("c.com", "partisan", "right", True, "z"),
    ]
    predictions = np.array([1, 0, 0])
    by_stratum = mod.rates_by(rows, predictions, "stratum")
    assert by_stratum["wire"] == {"documents": 1, "predicted_wire_half": 1.0}
    assert by_stratum["partisan"] == {"documents": 2, "predicted_wire_half": 0.0}


def test_load_frozen_reads_every_split(tmp_path: Path) -> None:
    for split in ("train", "dev", "holdout"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(_row(f"{split}.com", "wire", None, False, "t")) + "\n"
        )
    rows = mod.load_frozen(tmp_path)
    assert sorted(str(r["domain"]) for r in rows) == ["dev.com", "holdout.com", "train.com"]


def test_load_frozen_on_an_empty_directory(tmp_path: Path) -> None:
    assert mod.load_frozen(tmp_path) == []


def test_transfer_skips_when_there_is_no_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """No corpus must not mean a silently empty result."""
    monkeypatch.setattr(mod, "load_frozen", lambda *a, **k: [])
    report = mod.transfer("baseline", None)
    assert report["skipped"] == "no frozen corpus found"


def test_transfer_reports_shares_and_never_an_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row("wire.com", "wire", None, False, "wire copy"),
        _row("p1.com", "partisan", "left", False, "left copy"),
        _row("p2.com", "partisan", "right", True, "right copy"),
        _row("p3.com", "partisan", "right", True, "more right copy"),
    ]

    class FakeVectoriser:
        def transform(self, texts: list[str]) -> list[str]:
            return texts

    class FakeModel:
        def predict(self, texts: list[str]) -> np.ndarray:
            # Wire copy looks like the wire half; everything else does not.
            return np.array([1 if t.startswith("wire") else 0 for t in texts])

    monkeypatch.setattr(mod, "load_frozen", lambda *a, **k: rows)
    monkeypatch.setattr(mod, "fit_isot", lambda *a, **k: (FakeVectoriser(), FakeModel(), 0.9849))

    report = mod.transfer("baseline", None)

    assert report["ood_documents"] == 4
    assert report["ood_predicted_wire_half_overall"] == 0.25
    assert report["by_stratum"]["wire"]["predicted_wire_half"] == 1.0
    assert report["by_stratum"]["partisan"]["predicted_wire_half"] == 0.0
    assert report["by_low_quality_flag"]["True"]["documents"] == 2
    assert report["isot_held_out_accuracy"] == 0.9849
    assert not any("ood_accuracy" in key for key in report), "no OOD accuracy exists to report"
    assert "carries no veracity labels" in str(report["reading"])
