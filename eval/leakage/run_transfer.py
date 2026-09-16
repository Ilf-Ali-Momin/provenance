"""Ablation 4, reported as a prediction distribution rather than an accuracy.

EVALUATION.md asks for cross domain transfer: apply the unmodified ISOT style
model to the 2026 out of distribution slice and watch accuracy degrade toward
chance. That cannot be run as written. Accuracy needs veracity labels, and the
OOD suite deliberately has none: D001 forbids this project asserting truth, and
DATA.md allows fact checker assessments only as a reference point in a writeup.

What is measurable is what the model *says*. The ISOT corpus labels one class
the wire service half and the other the scraped half; the standard project reads
those as "real" and "fake". This module reports, for every group of OOD
publishers, the share the model assigns to the wire half. If the model were
detecting veracity, that share would not track publisher type. If it is
detecting publishing pipeline, it will.

Two warnings the output repeats, because a reader will otherwise supply the
wrong conclusion:

* No group here is established as more or less truthful than another. The
  strata are publisher types and the lean buckets are Media Bias/Fact Check's
  labels, neither of which is a truth claim (D007).
* A high "fake" share for low credibility outlets is not the model working. The
  credibility flag is confounded with political lean in this corpus (D017), and
  the model has never seen a truth label for any of these articles.

    PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from eval.leakage.ablations import Condition, apply_condition, load_corpus

logger = logging.getLogger(__name__)

FROZEN: Final = Path("data/ood/frozen-v4")
OUT: Final = Path("eval/leakage/results/transfer.json")
WIRE_HALF: Final = 1  # ablations.Split: y is 1 for the wire service half


def load_frozen(path: Path = FROZEN) -> list[dict[str, object]]:
    """Read the frozen suite. Its records carry split and topic, so not Document."""
    rows: list[dict[str, object]] = []
    for split in ("train", "dev", "holdout"):
        file = path / f"{split}.jsonl"
        if file.exists():
            rows += [
                json.loads(line)
                for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    return rows


def corpus_identity(path: Path = FROZEN) -> dict[str, object]:
    """The frozen suite's id and hash, read from the freeze's own manifest entry.

    Results are keyed by this hash. A prediction distribution is a statement
    about one corpus, and the same code over a different freeze is a different
    result; keying by the hash makes replacing the corpus without re-running
    visible instead of silent.
    """
    entry_file = path / "manifest_entry.json"
    if not entry_file.exists():
        return {"id": None, "sha256": None, "path": str(path)}
    entry = json.loads(entry_file.read_text(encoding="utf-8"))
    return {
        "id": entry.get("id"),
        "sha256": entry.get("sha256"),
        "record_count": entry.get("record_count"),
        "frozen_at": entry.get("frozen_at"),
        "path": str(path),
    }


def wire_share(predictions: np.ndarray) -> float:
    """Share predicted to be the corpus's wire service half."""
    if predictions.size == 0:
        return float("nan")
    return float(np.mean(predictions == WIRE_HALF))


def rates_by(
    rows: list[dict[str, object]], predictions: np.ndarray, field: str
) -> dict[str, dict[str, float | int]]:
    """Wire-half share for each value of a grouping field."""
    out: dict[str, dict[str, float | int]] = {}
    for value in sorted({str(r.get(field)) for r in rows}):
        mask = np.array([str(r.get(field)) == value for r in rows])
        out[value] = {
            "documents": int(mask.sum()),
            "predicted_wire_half": round(wire_share(predictions[mask]), 4),
        }
    return out


class Vectoriser(Protocol):
    """What this module needs of a fitted vectoriser.

    Declared here because scikit-learn ships no type stubs, so naming its
    classes in a signature would make the return type Any under mypy strict.
    A protocol keeps the contract explicit and lets a test pass a stub.
    """

    def transform(self, raw_documents: list[str]) -> object: ...


class Classifier(Protocol):
    def predict(self, matrix: object) -> np.ndarray: ...


def fit_isot(
    condition: Condition, train_subsample: int | None, seed: int = 0
) -> tuple[Vectoriser, Classifier, float]:
    """Fit the standard model on ISOT and report its own held out accuracy."""
    train, test = apply_condition(*load_corpus(), condition)
    if train_subsample and train_subsample < len(train.x):
        rng = np.random.default_rng(seed)
        index = rng.choice(len(train.x), train_subsample, replace=False)
        train_x = [train.x[i] for i in index]
        train_y = train.y[index]
        logger.info("training on a %d document subsample of %d", train_subsample, len(train.x))
    else:
        train_x, train_y = train.x, train.y

    vectoriser = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    matrix = vectoriser.fit_transform(train_x)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(matrix, train_y)
    in_domain = float(accuracy_score(test.y, model.predict(vectoriser.transform(test.x))))
    logger.info("%s: ISOT held out accuracy %.4f", condition, in_domain)
    return vectoriser, model, in_domain


def lean_gap(
    rows: list[dict[str, object]], predictions: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    """Lean-left minus lean-right share, within a subset of the corpus.

    D024 anticipated this: the cap on documents per month exists because one
    news cycle is one distribution, and the way to find out whether that matters
    is to report the same rates by month. Composition is held fixed here -- the
    same publishers appear in every month of one frozen suite -- so a gap that
    differs between periods is a property of the period, not of the sample.
    """
    shares: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for bucket in ("lean_left", "lean_right"):
        sel = np.array([r.get("lean") == bucket for r in rows]) & mask
        counts[bucket] = int(sel.sum())
        shares[bucket] = round(wire_share(predictions[sel]), 4) if sel.sum() else None

    left, right = shares["lean_left"], shares["lean_right"]
    return {
        "lean_left": {"documents": counts["lean_left"], "predicted_wire_half": left},
        "lean_right": {"documents": counts["lean_right"], "predicted_wire_half": right},
        "gap_percentage_points": (
            round((left - right) * 100, 1) if left is not None and right is not None else None
        ),
    }


def transfer(
    condition: Condition,
    train_subsample: int | None,
    seed: int = 0,
    corpus: Path = FROZEN,
) -> dict[str, object]:
    rows = load_frozen(corpus)
    if not rows:
        return {"skipped": "no frozen corpus found", "condition": condition}

    vectoriser, model, in_domain = fit_isot(condition, train_subsample, seed)
    texts = [str(r["text"]) for r in rows]
    if condition == "structure_only":
        from eval.leakage.ablations import structure_only

        texts = [structure_only(t) for t in texts]
    predictions = model.predict(vectoriser.transform(texts))

    return {
        "condition": condition,
        "corpus": corpus_identity(corpus),
        "isot_held_out_accuracy": round(in_domain, 4),
        "train_subsample": train_subsample,
        "ood_documents": len(rows),
        "ood_predicted_wire_half_overall": round(wire_share(predictions), 4),
        "by_stratum": rates_by(rows, predictions, "stratum"),
        "by_lean": rates_by(rows, predictions, "lean"),
        "by_low_quality_flag": rates_by(rows, predictions, "low_quality"),
        "by_publisher": rates_by(rows, predictions, "domain"),
        "by_month": rates_by(
            [{**r, "month": str(r.get("published") or "")[:7] or "unknown"} for r in rows],
            predictions,
            "month",
        ),
        # The hypothesis test, not a decoration. The superseded suite was 75%
        # one day; if the lean gap it showed was a property of that news cycle,
        # September inside this suite should still show it and the other eight
        # months should not.
        "lean_gap_by_period": {
            "september": lean_gap(
                rows,
                predictions,
                np.array([str(r.get("published") or "")[:7] == "2026-09" for r in rows]),
            ),
            "january_to_august": lean_gap(
                rows,
                predictions,
                np.array(
                    ["2026-01" <= str(r.get("published") or "")[:7] <= "2026-08" for r in rows]
                ),
            ),
            "whole_suite": lean_gap(rows, predictions, np.ones(len(rows), dtype=bool)),
        },
        "reading": (
            "Share of each group the ISOT model assigns to the corpus's wire service half, "
            "which that corpus treats as real. No accuracy is reported and none is available: "
            "the OOD suite carries no veracity labels (D001). No group here is established as "
            "more or less truthful than another, and the low_quality flag is confounded with "
            "political lean (D017), so a high fake share for flagged outlets is not evidence "
            "that the model works."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross domain transfer, ablation 4.")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--train-subsample", type=int, default=None)
    parser.add_argument("--corpus", type=Path, default=FROZEN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    report = transfer(args.condition, args.train_subsample, corpus=args.corpus)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    identity = corpus_identity(args.corpus)
    key = str(identity["sha256"] or args.corpus)
    bucket = existing.setdefault(key, {"corpus": identity, "conditions": {}})
    bucket["conditions"][str(args.condition)] = report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
