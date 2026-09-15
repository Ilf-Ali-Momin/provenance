"""Run OOD collection with per publisher checkpoints.

A full run is hours of rate limited fetching, and the last long job in this
project died when the session that launched it exited. So every publisher's
result is appended to disk the moment it finishes, and a restarted run skips
publishers already recorded instead of fetching them again. Restarting costs
nothing and never re-contacts a host that has already been asked.

Output goes under data/, which is gitignored: article text is the documented
exception to the rule that it never persists (SECURITY.md control 6) and it is
not redistributable (DATA.md), so it must never reach git.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.run --limit 12
    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.run --limit 10 \\
        --out data/ood/pilot --domains theguardian.com foxnews.com
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final

from pipeline.net.config import FetchSettings
from pipeline.net.robots import RobotsCache
from pipeline.text.config import ExtractSettings

from eval.ood_collection.backfill import backfill_domain
from eval.ood_collection.collect import (
    Document,
    Rejection,
    collect_domain,
    load_publishers,
    summarise,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT: Final = Path("data/ood/raw")
DOCUMENTS: Final = "documents.jsonl"
PROGRESS: Final = "progress.jsonl"


def load_progress(out_dir: Path) -> dict[str, dict[str, object]]:
    """Publishers already finished, keyed by domain."""
    path = out_dir / PROGRESS
    if not path.exists():
        return {}
    done: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[str(record["domain"])] = record
    return done


def load_documents(out_dir: Path) -> list[Document]:
    path = out_dir / DOCUMENTS
    if not path.exists():
        return []
    return [
        Document(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def months_of(documents: list[Document]) -> Counter[str]:
    """Months these documents fall in, keyed YYYY-MM.

    A document whose date the page never stated is not counted: it cannot be
    placed in a month, so it cannot fill a month's quota either.
    """
    return Counter(doc.published[:7] for doc in documents if doc.published)


def record_domain(
    out_dir: Path, domain: str, documents: list[Document], rejections: Rejection
) -> None:
    """Append one publisher's outcome. Documents first, progress last.

    If the process dies between the two writes the publisher is not marked
    done and is collected again on restart. Duplicates that produces are
    removed by document id when the corpus is frozen, which is cheaper than a
    silently missing publisher.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / DOCUMENTS).open("a", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    with (out_dir / PROGRESS).open("a", encoding="utf-8") as handle:
        entry = {"domain": domain, "kept": len(documents), "rejected": rejections.as_dict()}
        handle.write(json.dumps(entry) + "\n")


def run(
    out_dir: Path, limit: int, domains: list[str] | None, backfill: bool = False
) -> dict[str, object]:
    publishers = load_publishers()
    if domains:
        wanted = set(domains)
        publishers = [p for p in publishers if p["domain"] in wanted]

    done = load_progress(out_dir)
    robots = RobotsCache()
    fetch_settings = FetchSettings()
    extract_settings = ExtractSettings()

    remaining = [p for p in publishers if p["domain"] not in done]
    logger.info(
        "%d publishers, %d already done, %d to go", len(publishers), len(done), len(remaining)
    )

    for index, publisher in enumerate(remaining, start=1):
        domain = str(publisher["domain"])
        rejections = Rejection()
        logger.info("[%d/%d] %s", index, len(remaining), domain)
        documents = collect_domain(
            publisher,
            limit=limit,
            robots=robots,
            fetch_settings=fetch_settings,
            extract_settings=extract_settings,
            rejections=rejections,
        )
        if backfill:
            # The live window first, then the archive, so a story already taken
            # from the feed is not fetched again from a sitemap (D022). The
            # live window's months are handed on so the per month cap covers
            # the publisher's whole sample, not just its archive half.
            documents += backfill_domain(
                publisher,
                robots=robots,
                fetch_settings=fetch_settings,
                extract_settings=extract_settings,
                rejections=rejections,
                already_have={doc.url for doc in documents},
                months_seen=months_of(documents),
            )
        record_domain(out_dir, domain, documents, rejections)

    total = Rejection()
    for record in load_progress(out_dir).values():
        rejected = record["rejected"]
        if isinstance(rejected, dict):
            total.counts.update({str(k): int(v) for k, v in rejected.items()})

    summary = summarise(load_documents(out_dir), total)
    zero = sorted(d for d, r in load_progress(out_dir).items() if r["kept"] == 0)
    summary["publishers_yielding_nothing"] = zero
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12, help="articles per publisher")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="also walk archive sitemaps and confirm publication dates from the page (D022)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(args.out / "run.log"), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    summary = run(args.out, args.limit, args.domains, args.backfill)
    logger.info("done: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
