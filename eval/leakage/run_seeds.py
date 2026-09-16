"""Variance on the numbers the writeup asserts, rather than a hedge about them.

EVALUATION.md's leakage table is deliberately single seed: it reproduces the
standard approach with its defaults, and the defaults are single seed. That
stays. This module measures variance only for the rows something is claimed
about, which is what an interval is for.

Two sources of variance, kept apart because they answer different questions and
because reporting one as the other is how a spurious interval gets published.

**Seed variance** is what changes when the model is refit. For the linear probe
this is expected to be nil: `load_corpus` returns the mirror's own fixed train
and test split rather than resampling, `structure_only` is a deterministic text
transform, and `LogisticRegression` under lbfgs does not consult `random_state`.
It is measured rather than asserted, because "it should be deterministic" and
"it is deterministic" are different claims.

**Sampling variance** is what would change with a different sample of the same
kind. It is the interval that actually bears on a claim about publishers, and
it is estimated by resampling **publishers**, not documents. DATA.md makes the
publishing domain the unit throughout -- grouped splits exist because documents
from one outlet are not independent -- and a document level bootstrap over a
group of 6 publishers would report a confidence it has not earned. Both are
computed here so the difference is visible.

    PYTHONPATH=. .venv/bin/python -m eval.leakage.run_seeds --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np

from eval.leakage.ablations import Condition, structure_only
from eval.leakage.run_transfer import (
    FROZEN,
    corpus_identity,
    fit_isot,
    load_frozen,
    wire_share,
)

logger = logging.getLogger(__name__)

OUT: Final = Path("eval/leakage/results/seeds.json")
BOOTSTRAP_DRAWS: Final = 2000
PERCENTILES: Final = (2.5, 97.5)


def _rows_for(rows: list[dict[str, object]], lean: str) -> np.ndarray:
    return np.array([r.get("lean") == lean for r in rows])


def asserted_rows(rows: list[dict[str, object]], predictions: np.ndarray) -> dict[str, float]:
    """The four numbers the writeup makes a claim about, for one fitted model."""
    left = _rows_for(rows, "lean_left")
    right = _rows_for(rows, "lean_right")
    left_share = wire_share(predictions[left])
    right_share = wire_share(predictions[right])
    return {
        "whole_suite": wire_share(predictions),
        "lean_left": left_share,
        "lean_right": right_share,
        "lean_gap_pp": (left_share - right_share) * 100,
    }


def summarise(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "range": round(float(array.max() - array.min()), 4),
    }


def cluster_bootstrap(
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Resample publishers with replacement, not documents.

    Documents from one outlet share a house style, an editorial line and a
    topic mix, so they are not independent draws. Resampling the publisher
    carries all of its documents with it, which is the same unit DATA.md makes
    the splits on.
    """
    by_domain: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        by_domain[str(row["domain"])].append(position)
    domains = sorted(by_domain)
    rng = np.random.default_rng(seed)

    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        picked = rng.choice(len(domains), len(domains), replace=True)
        index: list[int] = []
        for choice in picked:
            index.extend(by_domain[domains[int(choice)]])
        sample = [rows[i] for i in index]
        values = asserted_rows(sample, predictions[np.asarray(index)])
        for key, value in values.items():
            if not np.isnan(value):
                collected[key].append(value)

    return {
        key: {
            "low": round(float(np.percentile(values, PERCENTILES[0])), 4),
            "high": round(float(np.percentile(values, PERCENTILES[1])), 4),
            "draws_usable": len(values),
        }
        for key, values in collected.items()
    }


def run(condition: Condition, seeds: int, corpus: Path) -> dict[str, object]:
    """Refit the ISOT model once per seed and report both kinds of variance."""
    rows = load_frozen(corpus)
    texts = [str(r["text"]) for r in rows]
    if condition == "structure_only":
        texts = [structure_only(t) for t in texts]

    in_domain: list[float] = []
    per_seed: dict[str, list[float]] = defaultdict(list)
    first_predictions: np.ndarray | None = None

    for seed in range(seeds):
        started = time.time()
        vectoriser, model, accuracy = fit_isot(condition, None, seed)
        predictions = model.predict(vectoriser.transform(texts))
        if first_predictions is None:
            first_predictions = predictions
        in_domain.append(accuracy)
        for key, value in asserted_rows(rows, predictions).items():
            per_seed[key].append(value)
        logger.info(
            "%s seed=%d isot=%.4f whole=%.4f (%.0fs)",
            condition,
            seed,
            accuracy,
            per_seed["whole_suite"][-1],
            time.time() - started,
        )

    assert first_predictions is not None  # noqa: S101 - seeds >= 1 is enforced by argparse
    return {
        "condition": condition,
        "seeds": seeds,
        "isot_held_out_accuracy": summarise(in_domain),
        "seed_variance": {key: summarise(values) for key, values in sorted(per_seed.items())},
        "publisher_bootstrap_95": cluster_bootstrap(rows, first_predictions),
        "reading": (
            "seed_variance is what refitting the model changes. "
            "publisher_bootstrap_95 resamples publishers with replacement and is the "
            "interval that bears on a claim about publishers; it is wide because the "
            "lean buckets hold few outlets, and that width is the honest width."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="structure_only")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--corpus", type=Path, default=FROZEN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    report = run(args.condition, args.seeds, args.corpus)
    identity = corpus_identity(args.corpus)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    key = str(identity["sha256"] or args.corpus)
    bucket = existing.setdefault(key, {"corpus": identity, "conditions": {}})
    bucket["conditions"][str(args.condition)] = report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
