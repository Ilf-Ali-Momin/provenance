"""Score the stance classifier on FEVER dev.

    PYTHONPATH=. .venv/bin/python -m eval.stance.run_eval --limit 500
    PYTHONPATH=. .venv/bin/python -m eval.stance.run_eval

EVALUATION.md requires the three slices reported together and the per class
breakdown as mandatory. This produces the first of the three. The MultiFC slice
is not built (D051: its labels are veracity verdicts, not stance toward
evidence) and the OOD slice needs hand annotated stance labels, so a run of this
alone is one number of three and the result says so.

**fp32 is scored alongside int8 when asked.** The quantisation delta in the
model manifest is six probes and says so; `--compare-fp32` runs both over the
same rows, which is the measurement EVALUATION.md actually asks for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Final

from pipeline.stance.classify import OnnxStanceClassifier

from eval.stance.fever import LOCAL, judgements, pairs
from eval.stance.score import score

logger = logging.getLogger(__name__)
OUT: Final = Path("eval/stance/results.json")
MODEL_DIR: Final = Path("models/deberta-nli")
BATCH: Final = 16


def predict(
    classifier: OnnxStanceClassifier, rows: list[dict[str, object]], batch: int = BATCH
) -> list[str]:
    """Labels for every row, batched, with progress on a long run."""
    labels: list[str] = []
    started = time.monotonic()
    for start in range(0, len(rows), batch):
        window = rows[start : start + batch]
        labels += [
            label
            for label, _ in classifier.classify(
                [str(r["premise"]) for r in window], [str(r["hypothesis"]) for r in window]
            )
        ]
        if start and start % (batch * 25) == 0:
            done = start + len(window)
            rate = done / (time.monotonic() - started)
            logger.info("%d/%d rows, %.1f/s", done, len(rows), rate)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=LOCAL)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare-fp32", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.annotations.exists():
        raise SystemExit(
            f"no corpus at {args.annotations}; run "
            "`PYTHONPATH=. .venv/bin/python -m eval.stance.fever --download`"
        )
    rows = pairs(args.annotations, args.limit)
    if not rows:
        raise SystemExit(f"{args.annotations} yielded no usable pairs")
    logger.info("scoring %d pairs", len(rows))

    result = score(
        judgements(rows, predict(OnnxStanceClassifier(args.model_dir), rows)), "fever_dev"
    )
    result["rows"] = len(rows)
    result["retrieved_evidence_rows"] = sum(1 for r in rows if r["retrieved_evidence"])
    result["quantised"] = True
    result["caveat"] = (
        "FEVER's NOT ENOUGH INFO claims have no gold evidence by construction, so those "
        "rows carry retrieved evidence. The neutral class is therefore measured against "
        "passages a retriever thought relevant and nobody confirmed were, which is not the "
        "same measurement as the other two classes. This is one of the three slices "
        "EVALUATION.md requires reported together; MultiFC is not built (D051) and the OOD "
        "slice needs hand annotated stance labels, so this number does not stand alone."
    )

    if args.compare_fp32:
        full = score(
            judgements(rows, predict(OnnxStanceClassifier(args.model_dir, quantised=False), rows)),
            "fever_dev_fp32",
        )
        result["fp32"] = full
        full_f1 = full["macro_f1"]
        int8_f1 = result["macro_f1"]
        if isinstance(full_f1, float) and isinstance(int8_f1, float):
            # Signed: a positive value is accuracy fp32 has and int8 does not,
            # which is the direction EVALUATION.md's "delta" means.
            result["quantisation_macro_f1_delta"] = round(full_f1 - int8_f1, 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps({k: v for k, v in result.items() if k not in {"confusion", "fp32"}}, indent=2)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
