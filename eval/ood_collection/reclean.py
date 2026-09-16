"""Re-apply the text-level construction rules to an already collected run.

Collection discards raw HTML, so any rule that needs the page, extraction
itself or the `<html lang>` check, can only run at collection time. Every other
construction rule takes text in and gives text out: publisher marker stripping,
outlet name masking, promotion, contact and credit removal. Those can be run
again over stored documents when the audit finds residue, instead of fetching
every publisher a second time.

The result is written to a new run directory, never over the source run, and
records both pipeline commits: the one that collected and extracted, and the
one whose text rules made this pass. Every document goes through the pass,
whatever its stratum, so the single-pipeline requirement in DATA.md still holds.
Documents that fall under the article length floor after cleaning are dropped
and counted, because a document that was mostly boilerplate is not an article.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.reclean \\
        data/ood/run-2fbca13 data/ood/run-2fbca13-clean
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from pipeline.text.config import ExtractSettings
from pipeline.text.extract import _host_of, strip_publisher_markers
from pipeline.text.normalise import normalise

from eval.ood_collection.collect import Document, document_id, load_outlet_names
from eval.ood_collection.run import DOCUMENTS, PROGRESS, load_documents, record_domain


def reclean(
    documents: list[Document],
    names_by_domain: dict[str, tuple[str, ...]],
    settings: ExtractSettings | None = None,
) -> tuple[list[Document], dict[str, object]]:
    """Run the text rules again over every document."""
    settings = settings or ExtractSettings()
    cleaned: list[Document] = []
    second_pass: Counter[str] = Counter()
    dropped = 0

    for doc in documents:
        text, removed = strip_publisher_markers(
            doc.text, names_by_domain.get(doc.domain, ()), _host_of(doc.url)
        )
        text = normalise(text)
        second_pass.update(removed)
        if len(text.strip()) < settings.min_article_chars:
            dropped += 1
            continue
        merged = Counter(doc.markers_removed)
        merged.update(removed)
        cleaned.append(
            replace(
                doc,
                text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                doc_id=document_id(doc.url, text),
                markers_removed=dict(merged),
            )
        )

    return cleaned, {
        "documents_in": len(documents),
        "documents_out": len(cleaned),
        "dropped_below_floor": dropped,
        "second_pass_removed": dict(second_pass),
    }


def _head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        sys.stderr.write("usage: reclean SOURCE_RUN DEST_RUN\n")
        return 2
    source, dest = Path(args[0]), Path(args[1])
    if dest.exists():
        sys.stderr.write(f"refusing to write into existing directory {dest}\n")
        return 1

    cleaned, report = reclean(load_documents(source), load_outlet_names())
    dest.mkdir(parents=True)
    by_domain: dict[str, list[Document]] = {}
    for doc in cleaned:
        by_domain.setdefault(doc.domain, []).append(doc)
    with (dest / DOCUMENTS).open("w", encoding="utf-8") as handle:
        for doc_list in by_domain.values():
            for doc in doc_list:
                handle.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
    if (source / PROGRESS).exists():
        shutil.copyfile(source / PROGRESS, dest / PROGRESS)

    collected_with = (source / "pipeline_sha.txt").read_text().strip()
    (dest / "pipeline_sha.txt").write_text(f"{collected_with}+reclean:{_head_sha()}\n")
    report["source_run"] = str(source)
    (dest / "reclean.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


__all__ = ["main", "reclean", "record_domain"]


if __name__ == "__main__":
    raise SystemExit(main())
