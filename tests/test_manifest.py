"""Appending the frozen suite to the manifest. No network."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from eval.ood_collection.manifest import ManifestError, append_entry, build_entry

HEADER = """# ---------------------------------------------------------------------------
# Corpus manifest. This comment must survive an append.
# ---------------------------------------------------------------------------

schema_version: 1

corpora:
  - id: isot_style_kaggle_mirror
    role: leakage_demonstration_only
    training_use: prohibited
"""

FROZEN = {
    "id": "ood_suite_2026",
    "snapshot_date": "2026-01-01",
    "record_count": 12,
    "sha256": "deadbeef",
    "frozen_at": "2026-09-11T00:00:00+00:00",
    "splits": {"method": "grouped_by_domain", "holdout_inspections": []},
    "training_use": "prohibited",
    "training_use_reason": "Evaluation only.",
    "evaluation_use": "permitted",
    "extraction_pipeline": "abc1234",
}


def test_entry_is_appended_and_comments_survive(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(HEADER)
    append_entry(build_entry(FROZEN), manifest)

    text = manifest.read_text()
    assert "This comment must survive an append." in text
    parsed = yaml.safe_load(text)
    assert [c["id"] for c in parsed["corpora"]] == ["isot_style_kaggle_mirror", "ood_suite_2026"]


def test_open_obligations_are_stated_not_blank(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(HEADER)
    append_entry(build_entry(FROZEN), manifest)
    entry = yaml.safe_load(manifest.read_text())["corpora"][1]
    assert entry["storage"]["region"].startswith("OPEN OBLIGATION")
    assert entry["personal_data"]["balancing_test"].startswith("OPEN OBLIGATION")
    assert entry["training_use"] == "prohibited"
    assert entry["redistribution"] == "prohibited"


def test_same_id_is_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(HEADER)
    append_entry(build_entry(FROZEN), manifest)
    with pytest.raises(ManifestError):
        append_entry(build_entry(FROZEN), manifest)


def test_empty_corpora_list_is_filled(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text("# header\nschema_version: 1\n\ncorpora: []\n")
    append_entry(build_entry(FROZEN), manifest)
    parsed = yaml.safe_load(manifest.read_text())
    assert [c["id"] for c in parsed["corpora"]] == ["ood_suite_2026"]
