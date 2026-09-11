"""Record a frozen OOD suite in data/MANIFEST.yaml.

The manifest is tracked in git and its header documents the schema in comments,
so the entry is appended as text under `corpora:` rather than by loading and
re-dumping the whole file, which would erase every comment. The result is then
parsed back to confirm the file is still valid YAML and the entry is present.

Fields the schema marks required are filled from the freeze, from what is known
about collection, or with an explicit statement that the obligation is open.
Nothing is left blank to look complete: the balancing test for legitimate
interest has not been written and the storage is not the EU region SECURITY.md
control 6 names for this corpus, and the entry says so.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.manifest data/ood/frozen
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

MANIFEST: Final = Path("data/MANIFEST.yaml")


class ManifestError(ValueError):
    """The manifest could not be updated safely."""


def build_entry(frozen: dict[str, object]) -> dict[str, object]:
    """The full manifest entry for a frozen suite."""
    entry: dict[str, object] = {
        "id": frozen["id"],
        "name": "Out of distribution evaluation suite, 2026",
        "role": "ood_suite",
        "purpose": (
            "Frozen English news articles published after the training snapshot, "
            "stratified by publisher type, for out of distribution evaluation only."
        ),
        "source_url": "eval/ood_collection/sources.yaml",
        "access_date": datetime.now(UTC).date().isoformat(),
        "snapshot_date": frozen["snapshot_date"],
        "publisher": "many; see sources.yaml and the document records",
        "licence": "publisher copyright; not licensed for redistribution",
        "licence_url": "per publisher",
        "redistribution": "prohibited",
        "terms_notes": (
            "Collected through publisher feeds and sitemaps, robots.txt honoured on every "
            "request, refusals recorded and never worked around (D019)."
        ),
        "training_use": frozen["training_use"],
        "training_use_reason": frozen["training_use_reason"],
        "evaluation_use": frozen["evaluation_use"],
        "checked_in": False,
        "retrieval": {
            "method": "download_script",
            "script": "eval/ood_collection/run.py",
            "notes": "Rerunning collects a different corpus; the frozen files are the suite.",
        },
        "storage": {
            "location": "data/ood/frozen on a FileVault encrypted local disk",
            "region": "OPEN OBLIGATION: not the EU region SECURITY.md control 6 requires",
            "encrypted_at_rest": True,
        },
        "record_count": frozen["record_count"],
        "sha256": frozen["sha256"],
        "frozen": True,
        "frozen_at": frozen["frozen_at"],
        "splits": frozen["splits"],
        "personal_data": {
            "contains": True,
            "lawful_basis": "legitimate interest, research",
            "balancing_test": "OPEN OBLIGATION: not yet written",
            "retention": "Held for the evaluation; deletion procedure not yet implemented.",
        },
        "normalisation": {
            "extraction_pipeline": frozen.get("extraction_pipeline"),
            "identical_across_strata": True,
            "publisher_markers_stripped": True,
            "notes": (
                "Datelines, bylines, boilerplate, newsletter promotions, own outlet names "
                "and self disclosures removed per D018; English only per D020."
            ),
        },
        "notes": json.dumps(
            {
                "duplicates_removed": frozen.get("duplicates_removed"),
                "partisan_balance": frozen.get("partisan_balance"),
                "topic_unknown_rate": frozen.get("topic_unknown_rate"),
                "cells_stratum_topic": frozen.get("cells_stratum_topic"),
            }
        ),
    }
    return entry


def append_entry(entry: dict[str, object], manifest: Path = MANIFEST) -> None:
    """Append under `corpora:` as text, then verify by parsing."""
    text = manifest.read_text(encoding="utf-8")
    current = yaml.safe_load(text) or {}
    corpora = current.get("corpora") or []
    if any(c.get("id") == entry["id"] for c in corpora):
        raise ManifestError(f"manifest already has an entry with id {entry['id']!r}")
    if "corpora:" not in text:
        raise ManifestError("manifest has no corpora list to append to")

    block = yaml.safe_dump([entry], sort_keys=False, width=88, allow_unicode=True)
    indented = "\n".join(("  " + line) if line else line for line in block.splitlines())
    if text.rstrip().endswith("corpora: []"):
        updated = text.rstrip()[: -len("[]")].rstrip() + "\n" + indented + "\n"
    else:
        updated = text.rstrip("\n") + "\n\n" + indented + "\n"

    parsed = yaml.safe_load(updated)
    ids = [c.get("id") for c in parsed.get("corpora", [])]
    if ids.count(entry["id"]) != 1:
        raise ManifestError("appended manifest did not parse back with exactly one new entry")
    manifest.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    frozen_dir = Path(args[0]) if args else Path("data/ood/frozen")
    frozen = json.loads((frozen_dir / "manifest_entry.json").read_text(encoding="utf-8"))
    try:
        append_entry(build_entry(frozen))
    except ManifestError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"recorded {frozen['id']} in {MANIFEST}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
