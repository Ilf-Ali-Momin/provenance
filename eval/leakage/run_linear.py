"""Run the TF-IDF probe across every ablation condition and record results."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from eval.leakage.ablations import Condition, apply_condition, load_corpus, run_linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS: tuple[Condition, ...] = (
    "baseline",
    "first_sentence_removed",
    "structure_only",
    "length_matched",
)


def main() -> None:
    train, test = load_corpus()
    logger.info("corpus train=%d test=%d pos_rate=%.3f", len(train.x), len(test.x), test.y.mean())
    out: dict[str, dict[str, float | int]] = {}
    for cond in CONDITIONS:
        t0 = time.time()
        tr, te = apply_condition(train, test, cond)
        acc = run_linear(tr, te)
        out[cond] = {
            "accuracy": round(acc, 4),
            "n_train": len(tr.x),
            "n_test": len(te.x),
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%-24s acc=%.4f n_train=%d", cond, acc, len(tr.x))
        dest = Path("eval/leakage/results/linear.json")
        dest.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
