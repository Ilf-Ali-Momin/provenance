"""Freeze a collected corpus: dedupe, assign topic, balance, split, hash.

Everything here is a selection decision, so every rule is written down in this
module and every count it produces goes into the manifest entry. Nothing is
sampled by a rule that is not in this file.

Topic comes from the article URL path, not from reading the text. A path
segment such as `/politics/` or `/health/` is the publisher's own filing
decision and is observable; a classifier over the text would be this project's
judgement, and a keyword model would learn exactly the vocabulary later models
are evaluated on. Articles whose path names none of the four topics are kept
and marked `unknown`, and the unknown rate is reported rather than hidden.

Partisan balance follows D017: equal numbers from left of centre and right of
centre, drawn evenly across each side's publishers so one prolific outlet
cannot stand in for its whole side.

Splits are grouped by publishing domain (DATA.md): every domain lands in
exactly one split, so no outlet's house style leaks between train and test.
Assignment is deterministic, seeded, and made within each stratum so that
every stratum with three or more domains has a domain in every split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from eval.ood_collection.collect import SNAPSHOT_DATE, Document
from eval.ood_collection.run import load_documents

TOPICS: Final[dict[str, tuple[str, ...]]] = {
    "politics": ("politics", "election", "elections", "government", "congress", "white-house"),
    "health": ("health", "medicine", "wellness", "coronavirus", "healthcare"),
    "science": ("science", "environment", "climate", "space", "nature"),
    "business": ("business", "economy", "markets", "finance", "money", "economics"),
}

SPLIT_PATTERN: Final[tuple[str, ...]] = ("train", "dev", "holdout", "train", "train")
LEFT: Final = frozenset({"left", "lean_left"})
RIGHT: Final = frozenset({"right", "lean_right"})
SEED: Final = 20260101


def topic_from_url(url: str) -> str:
    """The first of the four topics named in the URL path, else `unknown`."""
    segments = [s.lower() for s in urlsplit(url).path.split("/") if s]
    for segment in segments:
        for topic, names in TOPICS.items():
            if segment in names:
                return topic
    return "unknown"


# Pages a publisher serves from the same feeds and sitemaps as its journalism,
# which are not journalism. A newsletter issue is a digest addressed to
# subscribers; a solicitation page asks the reader for money. Both are the
# publisher talking about itself, which is the one thing this corpus is built
# to be free of, and neither is an article within the meaning of DATA.md.
#
# Detected by rule and applied to every document before any selection, rather
# than by reading the audit and lifting out the ones that look wrong. Both
# rules need two independent signals so that an article *about* newsletters or
# *about* a charity appeal is not mistaken for one (D027).
_NEWSLETTER_MENTIONS: Final = re.compile(
    r"\bnewsletters?\b[\s\S]{0,400}?\bnewsletters?\b", re.IGNORECASE
)
# A digest addresses its reader about itself. An article about the newsletter
# industry does not, and the mention count alone cannot tell them apart:
# "Rivals expanded their own newsletter businesses" mentions it twice.
_NEWSLETTER_ADDRESSES_READER: Final = re.compile(
    r"\b(?:we|our|your|this)\b[^.\n]{0,40}\bnewsletters?\b", re.IGNORECASE
)
_SOLICITATION: Final = re.compile(
    r"\b(?:tax[- ]deductible|501\(c\)\(?3\)?|charitable (?:gift|contribution)|"
    r"donor[- ]advised|bequest)\b",
    re.IGNORECASE,
)
_ASKS_FOR_MONEY: Final = re.compile(
    r"\b(?:your (?:gift|donation|contribution)|make a (?:gift|donation)|"
    r"mail a check|donate (?:now|today|monthly)|become a (?:member|supporter|donor))\b",
    re.IGNORECASE,
)


def not_an_article(doc: Document) -> str | None:
    """Why this document is not an article, or None if it is one.

    Returns the reason rather than a boolean so the freeze can report what it
    excluded and by which rule, instead of a single opaque count.
    """
    if _NEWSLETTER_MENTIONS.search(doc.text) and _NEWSLETTER_ADDRESSES_READER.search(doc.text):
        return "newsletter_issue"
    if _SOLICITATION.search(doc.text) and _ASKS_FOR_MONEY.search(doc.text):
        return "solicitation"
    return None


def drop_non_articles(documents: list[Document]) -> tuple[list[Document], dict[str, int]]:
    """Remove documents that are not articles, counting them by reason."""
    kept: list[Document] = []
    reasons: Counter[str] = Counter()
    for doc in documents:
        reason = not_an_article(doc)
        if reason is None:
            kept.append(doc)
        else:
            reasons[reason] += 1
    return kept, dict(sorted(reasons.items()))


def dedupe(documents: list[Document]) -> tuple[list[Document], int]:
    """Drop repeated document ids and repeated text within one domain.

    Repeats across domains are kept on purpose: the same wire story carried
    by two publishers is two documents for this corpus, one per outlet.
    """
    seen_ids: set[str] = set()
    seen_text: set[tuple[str, str]] = set()
    kept: list[Document] = []
    for doc in documents:
        key = (doc.domain, doc.text_sha256)
        if doc.doc_id in seen_ids or key in seen_text:
            continue
        seen_ids.add(doc.doc_id)
        seen_text.add(key)
        kept.append(doc)
    return kept, len(documents) - len(kept)


def _round_robin(
    by_domain: dict[str, list[Document]], n: int, rng: random.Random
) -> list[Document]:
    """Take n documents cycling across domains, so no one domain dominates."""
    pools = {d: rng.sample(docs, len(docs)) for d, docs in sorted(by_domain.items())}
    order = sorted(pools)
    taken: list[Document] = []
    while len(taken) < n and any(pools.values()):
        for domain in order:
            if pools[domain] and len(taken) < n:
                taken.append(pools[domain].pop())
    return taken


def balance_partisan(
    documents: list[Document], rng: random.Random
) -> tuple[list[Document], dict[str, int]]:
    """Equal left and right of centre within `partisan`; other strata untouched."""
    partisan = [d for d in documents if d.stratum == "partisan"]
    others = [d for d in documents if d.stratum != "partisan"]

    left: dict[str, list[Document]] = defaultdict(list)
    right: dict[str, list[Document]] = defaultdict(list)
    for doc in partisan:
        if doc.lean in LEFT:
            left[doc.domain].append(doc)
        elif doc.lean in RIGHT:
            right[doc.domain].append(doc)

    per_side = min(sum(map(len, left.values())), sum(map(len, right.values())))
    chosen = _round_robin(left, per_side, rng) + _round_robin(right, per_side, rng)
    dropped = len(partisan) - len(chosen)
    return others + chosen, {"per_side": per_side, "dropped_for_balance": dropped}


def assign_splits(documents: list[Document], seed: int = SEED) -> dict[str, str]:
    """Map each domain to one split, within each stratum, deterministically."""
    domains_by_stratum: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        domains_by_stratum[doc.stratum].add(doc.domain)

    assignment: dict[str, str] = {}
    for stratum in sorted(domains_by_stratum):
        ordered = sorted(
            domains_by_stratum[stratum],
            key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest(),
        )
        for index, domain in enumerate(ordered):
            assignment[domain] = SPLIT_PATTERN[index % len(SPLIT_PATTERN)]
    return assignment


def corpus_hash(documents: list[Document]) -> str:
    """sha256 over the sorted per document text hashes."""
    lines = "\n".join(sorted(f"{d.doc_id}:{d.text_sha256}" for d in documents))
    return hashlib.sha256(lines.encode()).hexdigest()


def freeze(
    documents: list[Document], seed: int = SEED
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Return split records and the manifest entry describing them."""
    rng = random.Random(seed)  # noqa: S311 - seeded reproducible sampling, not cryptographic
    # Before dedupe and before balancing: a document that is not an article is
    # not eligible for selection at all, so the balanced draw comes from a pool
    # of articles rather than depending on whether the shuffle happened to
    # discard one (D027).
    articles, non_articles = drop_non_articles(documents)
    unique, duplicates = dedupe(articles)
    balanced, balance = balance_partisan(unique, rng)
    splits = assign_splits(balanced, seed)

    records: dict[str, list[dict[str, object]]] = {"train": [], "dev": [], "holdout": []}
    for doc in balanced:
        record = asdict(replace(doc))
        record["topic"] = topic_from_url(doc.url)
        record["split"] = splits[doc.domain]
        records[splits[doc.domain]].append(record)

    all_records = [r for rs in records.values() for r in rs]
    cells = Counter(f"{r['stratum']}/{r['topic']}" for r in all_records)
    domains_per_split = {name: len({str(r["domain"]) for r in rs}) for name, rs in records.items()}
    coverage = {
        stratum: sorted({str(r["split"]) for r in all_records if r["stratum"] == stratum})
        for stratum in sorted({str(r["stratum"]) for r in all_records})
    }
    leakage = {str(r["domain"]) for r in records["train"]} & {
        str(r["domain"]) for r in records["dev"] + records["holdout"]
    }

    entry: dict[str, object] = {
        "id": "ood_suite_2026",
        "role": "ood_suite",
        "snapshot_date": SNAPSHOT_DATE.date().isoformat(),
        "record_count": len(all_records),
        "sha256": corpus_hash(balanced),
        "frozen": True,
        "frozen_at": datetime.now(UTC).isoformat(),
        "duplicates_removed": duplicates,
        "non_articles_excluded": non_articles,
        "partisan_balance": balance,
        "splits": {
            "method": "grouped_by_domain",
            "seed": seed,
            "train": len(records["train"]),
            "dev": len(records["dev"]),
            "frozen_holdout": len(records["holdout"]),
            "domains_per_split": domains_per_split,
            "domains_in_more_than_one_split": sorted(leakage),
            # A stratum with fewer than three domains cannot reach every split
            # under grouping by domain. Named here rather than discovered when
            # an evaluation finds no wire copy in its holdout.
            "splits_per_stratum": coverage,
            "strata_missing_a_split": sorted(
                s for s, present in coverage.items() if len(present) < 3
            ),
            "holdout_inspections": [],
        },
        "cells_stratum_topic": dict(sorted(cells.items())),
        "topic_unknown_rate": round(
            sum(1 for r in all_records if r["topic"] == "unknown") / max(len(all_records), 1), 4
        ),
        "training_use": "prohibited",
        "training_use_reason": "Evaluation only. Never a training source (DATA.md, D009).",
        "evaluation_use": "permitted",
        "redistribution": "prohibited",
        "checked_in": False,
    }
    return records, entry


def write_frozen(
    records: dict[str, list[dict[str, object]]], entry: dict[str, object], out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in records.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "manifest_entry.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


DEFAULT_FROZEN: Final = Path("data/ood/frozen")


def main(argv: list[str] | None = None) -> int:
    """Freeze one collection run. Refuses to overwrite an existing freeze.

    The collection plan says a frozen suite does not change, and D009 limits
    the holdout to two inspections. Overwriting a freeze would silently reset
    both, so a second freeze into the same directory is an error rather than
    an update; a corrected corpus is a new directory and a new manifest entry.
    """
    parser = argparse.ArgumentParser(description="Freeze a collected OOD corpus.")
    parser.add_argument("corpus", type=Path, help="collection run directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if (args.out / "manifest_entry.json").exists():
        sys.stderr.write(f"refusing to overwrite the frozen corpus at {args.out}\n")
        return 1

    records, entry = freeze(load_documents(args.corpus), args.seed)
    sha_file = args.corpus / "pipeline_sha.txt"
    entry["extraction_pipeline"] = sha_file.read_text().strip() if sha_file.exists() else None
    entry["source_run"] = str(args.corpus)
    write_frozen(records, entry, args.out)
    sys.stdout.write(json.dumps(entry, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
