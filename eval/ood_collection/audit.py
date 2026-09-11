"""Audit a collected corpus for publisher identity that survived construction.

The construction rules in `pipeline.text.extract` strip what they were written
to strip. This module looks for what they were not written for, because every
defect the pilot runs found was one of those: a byline written as a name prefix,
an outlet's own name, a promotion split across two sentences. A rule's own
counters can only report what the rule matched; residue has to be looked for
independently.

Checks are deliberately broader than the rules they audit. Own names are
searched case insensitively, so "FOX NEWS" in a headline block is caught even
though masking is case sensitive. Every hit is a candidate for a human to look
at, not proof of a leak, which is why the output carries short context snippets
and why DATA.md's construction rules also require a hand check per stratum.

Output holds counts and at most a few short snippets per publisher, never whole
articles, and is written beside the corpus under the gitignored data/ tree.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.audit data/ood/pilot2
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

from eval.ood_collection.collect import Document, load_outlet_names
from eval.ood_collection.run import load_documents

SNIPPET_CHARS: Final = 80
MAX_SNIPPETS: Final = 3

RESIDUE: Final[dict[str, re.Pattern[str]]] = {
    "by_line": re.compile(r"^\s*By\s+[A-Z][a-z]+", re.MULTILINE),
    "caps_name_colon": re.compile(r"^[A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){1,3}:\s", re.MULTILINE),
    "wire_tag": re.compile(r"\((?:AP|Reuters|AFP|CNN|UPI|Bloomberg)\)"),
    "photo_credit": re.compile(r"\b(?:Photo|Image|Illustration)\s*(?::|by\b|credit\b)", re.I),
    "newsletter": re.compile(r"\bnewsletters?\b", re.I),
    "follow_us": re.compile(r"\bfollow (?:us|@\w+)\b", re.I),
    "copyright": re.compile(r"(?:©|\(c\)\s*\d{4}|\bcopyright\b)", re.I),
}


def _snippet(text: str, start: int) -> str:
    left = max(0, start - SNIPPET_CHARS // 2)
    return text[left : left + SNIPPET_CHARS].replace("\n", " | ")


def audit(documents: list[Document], outlet_names: dict[str, tuple[str, ...]]) -> dict[str, object]:
    """Residue counts per publisher, plus corpus wide totals."""
    by_domain: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_domain[doc.domain].append(doc)

    publishers: dict[str, object] = {}
    totals: Counter[str] = Counter()

    for domain, docs in sorted(by_domain.items()):
        hits: Counter[str] = Counter()
        snippets: dict[str, list[str]] = defaultdict(list)
        masked: Counter[str] = Counter()
        name_patterns = [
            re.compile(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", re.IGNORECASE)
            for n in outlet_names.get(domain, ())
        ]

        for doc in docs:
            masked.update(doc.markers_removed)
            for name, pattern in RESIDUE.items():
                match = pattern.search(doc.text)
                if match:
                    hits[name] += 1
                    if len(snippets[name]) < MAX_SNIPPETS:
                        snippets[name].append(_snippet(doc.text, match.start()))
            for pattern in name_patterns:
                match = pattern.search(doc.text)
                if match:
                    hits["own_name_unmasked"] += 1
                    if len(snippets["own_name_unmasked"]) < MAX_SNIPPETS:
                        snippets["own_name_unmasked"].append(_snippet(doc.text, match.start()))
                    break

        lengths = sorted(len(d.text) for d in docs)
        totals.update(hits)
        publishers[domain] = {
            "documents": len(docs),
            "median_chars": lengths[len(lengths) // 2],
            "documents_with_residue": dict(hits),
            "removed_by_construction": dict(masked),
            "snippets": dict(snippets),
        }

    return {
        "documents": len(documents),
        "publishers": len(by_domain),
        "documents_with_residue": dict(totals),
        "by_publisher": publishers,
    }


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ood/raw")
    report = audit(load_documents(corpus), load_outlet_names())
    (corpus / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps({k: v for k, v in report.items() if k != "by_publisher"}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
