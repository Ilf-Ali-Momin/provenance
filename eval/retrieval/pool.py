"""Build the candidate pool a human then grades.

    PYTHONPATH=. .venv/bin/python -m eval.retrieval.pool --claims eval/retrieval/claims.jsonl

Runs each configuration under comparison, unions their top-k, and writes one
row per claim with an empty `grades` map for a person to fill. The pool is built
before any judgement is made, so it cannot be widened after seeing which
configuration is winning - see `judgements.py` for why that matters and for the
bias this leaves.

Claims come from a file rather than from the extractor, because the extractor's
precision and recall are unmeasured (M2 is parked, D042) and running it here
would make retrieval's numbers depend on an unquantified upstream. A claims file
is explicit about what was retrieved for.

**Dev only.** The sweep refuses anything else, so producing a pool from another
split would only waste judging effort.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from pipeline.retrieval.bm25 import Bm25Index
from pipeline.retrieval.config import RetrievalSettings
from pipeline.retrieval.hybrid import fuse

from eval.retrieval.judgements import JUDGEMENTS, ClaimJudgements, pool_candidates, write
from eval.retrieval.sweep import DEFAULT_GRID, SWEEPABLE_SPLITS

logger = logging.getLogger(__name__)
POOL_DEPTH: Final = 20


def pool_for(
    claim_text: str,
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    grid: tuple[float, ...] = DEFAULT_GRID,
    depth: int = POOL_DEPTH,
) -> list[str]:
    """Every grid point's top `depth`, unioned."""
    runs = {
        f"w{weight}": [
            (r.doc_id, r.score)
            for r in fuse(dense, lexical, RetrievalSettings(dense_weight=weight, top_k=depth))
        ]
        for weight in grid
    }
    return pool_candidates(runs, depth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=JUDGEMENTS)
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge"))
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [
        json.loads(line)
        for line in args.claims.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    offending = sorted({str(r.get("split")) for r in rows} - set(SWEEPABLE_SPLITS))
    if offending:
        raise SystemExit(
            f"claims from {offending} are in the claims file; the sweep only runs on "
            f"{list(SWEEPABLE_SPLITS)}, so pooling them would spend judging effort on "
            "claims nothing may be measured against (D009)."
        )

    from pipeline.retrieval.db import PostgresSettings, connect
    from pipeline.retrieval.embed import OnnxEmbedder
    from pipeline.retrieval.store import PgVectorStore

    if not (args.model_dir / "model.onnx").exists():
        raise SystemExit(f"no embedder at {args.model_dir}; run tools.export_embedder first")

    connection = connect(PostgresSettings())
    try:
        store = PgVectorStore(connection, OnnxEmbedder(args.model_dir))
        with connection.cursor() as cursor:
            cursor.execute("SELECT chunk_id, text FROM evidence_chunks")
            corpus = {str(chunk_id): str(text) for chunk_id, text in cursor.fetchall()}
        if not corpus:
            raise SystemExit("the index is empty; run tools.build_index first")
        lexical_index = Bm25Index(corpus)

        pooled: list[ClaimJudgements] = []
        for row in rows:
            claim_text = str(row["claim_text"])
            candidates = pool_for(
                claim_text,
                store.search(claim_text, k=args.depth * 2),
                lexical_index.search(claim_text, k=args.depth * 2),
                depth=args.depth,
            )
            pooled.append(
                ClaimJudgements(
                    claim_id=str(row["claim_id"]),
                    claim_text=claim_text,
                    split=str(row["split"]),
                    group=str(row.get("group") or "all"),
                    pool=candidates,
                    grades={},
                    reviewed=False,
                )
            )
    finally:
        connection.close()

    written = write(pooled, args.out)
    depth_total = sum(len(p.pool) for p in pooled)
    sys.stdout.write(
        f"{written} claims pooled to {args.out}, {depth_total} candidates to grade "
        f"({depth_total / written:.1f} per claim). Nothing is graded yet; the scorer "
        "refuses the file until a person marks it reviewed.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
