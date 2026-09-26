"""Measure whether fp32 stance classification is affordable.

    PYTHONPATH=. .venv/bin/python -m tools.bench_stance --pairs 24

D050 recorded int8 as costing 0.0583 macro F1 and 12.5 points of refute recall
against fp32, and left open whether fp32 is affordable. That question is not
answered by one session's latency, because the deployment runs several worker
containers on one box (ARCHITECTURE.md: "RQ workers run as separate containers
with independent scaling"). Two things therefore get measured:

**Latency against threads**, per config. How much faster one request gets when
a session is allowed more cores. This is what a user waiting on a card feels.

**Throughput against workers**, at one thread each. How many pairs per second
the whole box does when W processes each take a core. This is what decides
capacity, and it is not the reciprocal of the first: ONNX Runtime's intra-op
parallelism scales sublinearly, so N single-threaded workers usually beat one
N-threaded session on total work while being N times worse on each request.

The two together say what fp32 costs, in the units the decision is actually
made in. Whether that cost is acceptable is not a question this file answers.

Worker processes rather than threads, because Python's GIL would make a thread
pool measure the wrong thing: the deployment's parallelism is process level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

from eval.stance.fever import LOCAL, pairs

INT8: Final = Path("models/deberta-nli")
OUT: Final = Path("eval/stance/bench.json")
WARMUP: Final = 4


def _run_one(spec: tuple[str, bool, int, list[str], list[str]]) -> float:
    """Classify a batch in this process and return the seconds it took.

    Top level so it pickles for the process pool. Each call builds its own
    session, which is what a worker container does at start.
    """
    model_dir, quantised, threads, premises, hypotheses = spec
    from pipeline.stance.classify import OnnxStanceClassifier

    classifier = OnnxStanceClassifier(Path(model_dir), quantised=quantised, threads=threads)
    classifier.classify(premises[:WARMUP], hypotheses[:WARMUP])
    started = time.monotonic()
    classifier.classify(premises, hypotheses)
    return time.monotonic() - started


def latency_curve(
    premises: list[str], hypotheses: list[str], thread_counts: tuple[int, ...]
) -> list[dict[str, Any]]:
    """One session, increasing threads. What a waiting user feels."""
    rows: list[dict[str, Any]] = []
    for name, quantised in (("int8", True), ("fp32", False)):
        for threads in thread_counts:
            seconds = _run_one((str(INT8), quantised, threads, premises, hypotheses))
            rows.append(
                {
                    "config": name,
                    "threads": threads,
                    "pairs": len(premises),
                    "seconds": round(seconds, 3),
                    "ms_per_pair": round(1000 * seconds / len(premises), 1),
                    "pairs_per_second": round(len(premises) / seconds, 3),
                }
            )
            sys.stdout.write(f"  {name} {threads}t: {rows[-1]['ms_per_pair']} ms/pair\n")
    return rows


def throughput_curve(
    premises: list[str], hypotheses: list[str], worker_counts: tuple[int, ...]
) -> list[dict[str, Any]]:
    """W single-threaded processes at once. What decides capacity.

    Every worker classifies the same batch, so the reported figure is the whole
    box's rate: workers times batch, divided by the wall time of the slowest.
    """
    rows: list[dict[str, Any]] = []
    for name, quantised in (("int8", True), ("fp32", False)):
        for workers in worker_counts:
            spec = (str(INT8), quantised, 1, premises, hypotheses)
            started = time.monotonic()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                each = list(pool.map(_run_one, [spec] * workers))
            wall = time.monotonic() - started
            done = workers * len(premises)
            rows.append(
                {
                    "config": name,
                    "workers": workers,
                    "threads_each": 1,
                    "pairs_total": done,
                    # Wall time includes each worker building its own session, which
                    # a container also pays once. `slowest_batch` excludes it.
                    "wall_seconds": round(wall, 3),
                    "slowest_batch_seconds": round(max(each), 3),
                    "pairs_per_second": round(done / max(each), 3),
                    "batch_spread_seconds": round(max(each) - min(each), 3),
                }
            )
            sys.stdout.write(
                f"  {name} {workers}w: {rows[-1]['pairs_per_second']} pairs/s "
                f"(spread {rows[-1]['batch_spread_seconds']}s)\n"
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--annotations", type=Path, default=LOCAL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"no corpus at {args.annotations}; run eval.stance.fever --download")
    rows = pairs(args.annotations, args.pairs)
    premises = [str(r["premise"]) for r in rows]
    hypotheses = [str(r["hypothesis"]) for r in rows]

    sys.stdout.write(f"latency, one session, {len(rows)} pairs\n")
    latency = latency_curve(premises, hypotheses, tuple(args.threads))
    sys.stdout.write(f"\nthroughput, W single-threaded processes, {len(rows)} pairs each\n")
    throughput = throughput_curve(premises, hypotheses, tuple(args.workers))

    def best(config: str, rowset: list[dict[str, Any]]) -> float:
        return max(float(r["pairs_per_second"]) for r in rowset if r["config"] == config)

    peak = {"int8": best("int8", throughput), "fp32": best("fp32", throughput)}
    report: dict[str, Any] = {
        "pairs_per_batch": len(rows),
        "latency_by_threads": latency,
        "throughput_by_workers": throughput,
        "peak_pairs_per_second": peak,
        "reading": (
            "latency_by_threads is what a waiting user feels; throughput_by_workers is what "
            "decides capacity, and the two do not trade off symmetrically because intra-op "
            "parallelism scales sublinearly. Machine dependent: a figure here describes the "
            "box it ran on and says nothing about another. The accuracy side of this trade is "
            "in D050 - int8 costs 0.0583 macro F1 and 12.5 points of refute recall."
        ),
    }
    ratio = round(peak["fp32"] / peak["int8"], 3) if peak["int8"] else None
    report["fp32_throughput_ratio"] = ratio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"\npeak pairs/s: int8 {peak['int8']}, fp32 {peak['fp32']} "
        f"(fp32 is {ratio}x int8)\n{args.out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
