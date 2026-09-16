"""Measure the topic distribution of the frozen suite, and check it (D029).

DATA.md asks for "topics spread across politics, health, science, and business
so that a single news cycle does not dominate". The freeze reads topic from the
article URL path, because a path segment such as `/politics/` is the publisher's
own filing decision and is observable rather than inferred. That is the right
default and it is sparse: 93.6% of the frozen suite resolves to `unknown`,
because most publishers do not put a section in the path.

So the requirement has never been checked. This module checks it, by zero-shot
entailment over the article text with the MNLI model ARCHITECTURE.md already
names for stance classification at M3. No new dependency, and no model trained
on this corpus, which matters: a keyword or fitted topic model would learn the
vocabulary that later components are evaluated on.

Two rules keep the inference honest.

The inferred label never replaces the URL path label. The path label is the
publisher's own categorisation and stays the recorded field; the inferred label
is this project's reading of the text, recorded separately and marked as such.

Where both exist, they are compared. Agreement on the 6.4% of documents the path
does resolve is the only evidence available that the inferred labels mean what
they say, and it is reported whatever it shows.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.topics data/ood/frozen-v4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

from transformers import pipeline

from eval.ood_collection.config import TopicSettings

logger = logging.getLogger(__name__)

TOPICS: Final[tuple[str, ...]] = ("politics", "health", "science", "business")
UNCLEAR: Final = "unclear"
HYPOTHESIS: Final = "This news article is about {}."


def load_records(corpus: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "dev", "holdout"):
        path = corpus / f"{split}.jsonl"
        if path.exists():
            rows += [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    return rows


def classify(texts: list[str], settings: TopicSettings | None = None) -> list[tuple[str, float]]:
    """Zero-shot topic per text, as (label, score). Below the floor: `unclear`."""
    settings = settings or TopicSettings()
    classifier = pipeline(
        "zero-shot-classification",
        model=settings.model_id,
        revision=settings.revision,
    )
    out: list[tuple[str, float]] = []
    for start in range(0, len(texts), settings.batch_size):
        batch = [t[: settings.max_characters] for t in texts[start : start + settings.batch_size]]
        results = classifier(
            batch,
            candidate_labels=list(TOPICS),
            hypothesis_template=HYPOTHESIS,
        )
        for result in results if isinstance(results, list) else [results]:
            label = str(result["labels"][0])
            score = float(result["scores"][0])
            out.append((label, score) if score >= settings.entailment_floor else (UNCLEAR, score))
        logger.info("classified %d/%d", min(start + settings.batch_size, len(texts)), len(texts))
    return out


def report(rows: list[dict[str, object]], inferred: list[tuple[str, float]]) -> dict[str, object]:
    """Distribution, per stratum cells, and agreement with the URL path label."""
    labels = [label for label, _ in inferred]
    distribution = Counter(labels)

    cells: dict[str, Counter[str]] = defaultdict(Counter)
    for row, label in zip(rows, labels, strict=True):
        cells[str(row["stratum"])][label] += 1

    # The only available check on the inferred labels: where the publisher filed
    # the article under a section, does the text agree with the publisher?
    both = [
        (str(row.get("topic")), label)
        for row, label in zip(rows, labels, strict=True)
        if row.get("topic") not in (None, "unknown")
    ]
    agree = sum(1 for path_topic, label in both if path_topic == label)

    named = sum(count for label, count in distribution.items() if label != UNCLEAR)
    largest = max((c for label, c in distribution.items() if label != UNCLEAR), default=0)

    return {
        "documents": len(rows),
        "model": TopicSettings().model_id,
        "revision": TopicSettings().revision,
        "entailment_floor": TopicSettings().entailment_floor,
        "distribution": dict(sorted(distribution.items())),
        "share_named": round(named / max(len(rows), 1), 4),
        "largest_topic_share_of_named": round(largest / max(named, 1), 4),
        "cells_stratum_topic": {
            stratum: dict(sorted(counts.items())) for stratum, counts in sorted(cells.items())
        },
        "path_label_agreement": {
            "documents_with_a_path_label": len(both),
            "agreed": agree,
            "rate": round(agree / len(both), 4) if both else None,
        },
        "reading": (
            "The inferred label is this project's reading of the text and never replaces "
            "the URL path label, which is the publisher's own filing decision. Topic is "
            "not rebalanced on the strength of this measurement: the suite is frozen."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the frozen suite's topic spread.")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = load_records(args.corpus)
    if not rows:
        sys.stderr.write(f"no frozen corpus at {args.corpus}\n")
        return 1

    inferred = classify([str(r["text"]) for r in rows])
    result = report(rows, inferred)
    destination = args.out or args.corpus / "topics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps({k: v for k, v in result.items() if k != "cells_stratum_topic"}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
