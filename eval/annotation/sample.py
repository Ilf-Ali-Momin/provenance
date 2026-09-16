"""Draw the 200 passage annotation set from the frozen suite.

DATA.md: "A hand annotated set of 200 passages drawn from the OOD collection
below, annotated by you against a written guideline." This module draws them.
It does not annotate them, and nothing in this project may: a generated gold
standard measures the generator, and every acceptance number in EVALUATION.md
§3 would be circular.

**Train and dev only, never the frozen holdout (D009, and D035).** Hand
annotating a passage means reading it closely, which is an inspection in the
only sense D009 cares about, and the holdout is permitted exactly two. Drawing
from it would spend one on a task that does not need it, and would put a human's
close reading of holdout text into the project's working memory for the rest of
the build.

Sampling is seeded and spread across publishers, so the set is reproducible and
no single outlet's house style dominates what the extractor is measured on.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

FROZEN: Final = Path("data/ood/frozen-v4")
OUT: Final = Path("eval/annotation/passages.jsonl")
SEED: Final = 20260101
TARGET: Final = 200
MIN_CHARS: Final = 400
# No publisher may supply more than this share of the set. Without a cap, the
# round robin hands every remaining slot to the largest publisher once the
# smaller pools are exhausted.
MAX_PUBLISHER_SHARE: Final = 0.10
MAX_CHARS: Final = 1200
# Splits a passage may come from. The holdout is absent on purpose.
ELIGIBLE_SPLITS: Final = ("train", "dev")

_PARA = re.compile(r"\n\s*\n")


def load_documents(corpus: Path = FROZEN) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ELIGIBLE_SPLITS:
        path = corpus / f"{split}.jsonl"
        if path.exists():
            rows += [
                {**json.loads(line), "split": split}
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    return rows


def passages_from(text: str) -> list[str]:
    """Paragraph sized chunks within the length band.

    A passage is a paragraph rather than a sentence or a whole article. A
    sentence gives the extractor no context to distinguish an attribution from
    an assertion; a whole article makes the offsets unwieldy to annotate by hand
    and pushes past what fits in one prompt comfortably.
    """
    out: list[str] = []
    for block in _PARA.split(text):
        chunk = " ".join(block.split())
        if MIN_CHARS <= len(chunk) <= MAX_CHARS:
            out.append(chunk)
        elif len(chunk) > MAX_CHARS:
            # Long block: take the first window ending at a sentence boundary.
            window = chunk[:MAX_CHARS]
            cut = window.rfind(". ")
            if cut >= MIN_CHARS:
                out.append(window[: cut + 1])
    return out


def sample(
    documents: list[dict[str, object]],
    target: int = TARGET,
    seed: int = SEED,
    max_share: float = MAX_PUBLISHER_SHARE,
) -> list[dict[str, object]]:
    """Round robin across publishers, with a hard cap per publisher.

    Round robin alone does not guarantee the property. When the smaller pools
    run out, the largest publisher fills every remaining slot: on a synthetic
    case one outlet took 10 of 20. The cap is what makes "no outlet dominates"
    true regardless of pool shape.

    When the cap makes the target unreachable, the set comes back short. That is
    the same choice the corpus work made: report the underfill rather than meet
    the number by skewing the sample.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptographic
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for doc in documents:
        for index, passage in enumerate(passages_from(str(doc["text"]))):
            by_domain[str(doc["domain"])].append(
                {
                    "passage_id": f"{doc['doc_id']}#{index}",
                    "domain": doc["domain"],
                    "stratum": doc["stratum"],
                    "split": doc["split"],
                    "text": passage,
                }
            )

    pools = {d: rng.sample(v, len(v)) for d, v in sorted(by_domain.items())}
    order = sorted(pools)
    cap = max(1, int(target * max_share))
    taken: dict[str, int] = defaultdict(int)
    picked: list[dict[str, object]] = []
    while len(picked) < target and any(pools[d] and taken[d] < cap for d in order):
        for domain in order:
            if pools[domain] and taken[domain] < cap and len(picked) < target:
                picked.append(pools[domain].pop())
                taken[domain] += 1
    return picked


def to_form(rows: list[dict[str, object]]) -> str:
    """One JSON object per line, with empty `spans` for a human to fill."""
    return "\n".join(
        json.dumps({**row, "spans": [], "notes": ""}, ensure_ascii=False) for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=FROZEN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()

    documents = load_documents(args.corpus)
    if not documents:
        raise SystemExit(f"no train or dev split under {args.corpus}")
    rows = sample(documents, args.target)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite annotation work")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_form(rows) + "\n", encoding="utf-8")

    publishers = len({r["domain"] for r in rows})
    print(f"wrote {len(rows)} passages from {publishers} publishers to {args.out}")  # noqa: T201
    if len(rows) < args.target:
        print(  # noqa: T201
            f"UNDERFILLED: {len(rows)} of {args.target}. The per publisher cap "
            f"({int(args.target * MAX_PUBLISHER_SHARE)}) was reached before the target."
        )
    print("annotate against docs/annotation-guideline.md; splits used: train, dev")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
