"""Choose which pages go into the reference corpus.

Kept out of the fetch loop deliberately. Which pages belong in a reference
corpus is a corpus design decision, and hiding it inside a loop that also does
HTTP makes it invisible to the manifest and impossible to test.

**The trap this module exists to close.** A reference index is only useful if it
covers what claims are actually about, so the obvious move is to seed selection
from the evaluation corpus's own entities. That is legitimate for train and dev
and is leakage for the frozen holdout: an index built around the holdout's
subject matter retrieves better on the holdout for a reason that has nothing to
do with the retriever, and the resulting number would describe the corpus
construction rather than the system. This is the same failure M0 spent a
milestone demonstrating, one component along.

So `seed_terms` reads `ELIGIBLE_SPLITS` and the holdout file is never opened.
The constant is imported from `eval.annotation.sample` rather than redeclared,
because two copies of "which splits may be touched" drift and only one of them
is ever updated (D009, D035).

**Wikinews selection is by category, not by search.** `Category:Published` is
the wiki's own marker for a finished article, so selection does not depend on
this project's judgement of what counts as news. Date bounding is left to the
caller, since the manifest records a snapshot date and the two have to agree.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

from eval.annotation.sample import ELIGIBLE_SPLITS, FROZEN

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

# Capitalised multiword runs: the proper nouns and institutions a claim names,
# which are what a reference lookup is for. Deliberately not a parser - the
# seeds are search queries, and a wrong one costs a search, not a wrong answer.
_ENTITY: Final = re.compile(r"\b(?:[A-Z][\w'-]+(?:\s+(?:of|and|the|for|de|van))?\s*){2,6}")
_STOP_STARTS: Final = frozenset(
    {
        "The",
        "A",
        "An",
        "In",
        "On",
        "At",
        "But",
        "And",
        "It",
        "He",
        "She",
        "They",
        "This",
        "That",
        "We",
        "I",
        "If",
        "When",
        "After",
    }
)
MIN_SEED_LENGTH: Final = 6
DEFAULT_SEEDS: Final = 200
DEFAULT_PAGES_PER_SEED: Final = 3


def seed_terms(
    corpus: Path = FROZEN,
    limit: int = DEFAULT_SEEDS,
    splits: tuple[str, ...] = ELIGIBLE_SPLITS,
) -> list[str]:
    """The most frequent named entities in the eligible splits, most common first.

    `splits` defaults to train and dev. Passing the holdout is possible and is
    the thing this module's docstring says not to do; it is a parameter so the
    default is visible rather than buried, not so that it is routine.
    """
    if "holdout" in splits:
        msg = (
            "the frozen holdout may not seed the reference index. An index built "
            "around the holdout's subject matter retrieves better on it for reasons "
            "unrelated to the retriever, which is the leakage this project exists to "
            "document (D009)."
        )
        raise ValueError(msg)

    counts: Counter[str] = Counter()
    for split in splits:
        path = corpus / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("text") or "")
            for match in _ENTITY.finditer(text):
                term = " ".join(match.group(0).split())
                first = term.split()[0]
                if first in _STOP_STARTS:
                    # Drop the leading sentence-initial capital and keep the rest.
                    term = " ".join(term.split()[1:])
                if len(term) >= MIN_SEED_LENGTH and " " in term:
                    counts[term] += 1
    return [term for term, _ in counts.most_common(limit)]


def wikinews_titles(
    fetch_json: Callable[[str], dict[str, object]],
    host: str = "en.wikinews.org",
    category: str = "Category:Published",
    limit: int = 500,
) -> list[str]:
    """Article titles from a category, article namespace only.

    `cmnamespace=0` keeps portals, templates and talk pages out; they carry
    navigation text rather than prose and would index as evidence.
    """
    titles: list[str] = []
    continue_token: str | None = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": "0",
            "cmlimit": str(min(500, limit - len(titles))),
            "format": "json",
            "formatversion": "2",
        }
        if continue_token:
            params["cmcontinue"] = continue_token
        payload = fetch_json(f"https://{host}/w/api.php?{urlencode(params)}")
        query = payload.get("query")
        members = query.get("categorymembers", []) if isinstance(query, dict) else []
        titles += [str(m["title"]) for m in members if isinstance(m, dict) and m.get("title")]
        cont = payload.get("continue")
        continue_token = str(cont.get("cmcontinue")) if isinstance(cont, dict) else None
        if not continue_token or not members:
            break
    return titles[:limit]


def wikipedia_titles(
    seeds: list[str],
    fetch_json: Callable[[str], dict[str, object]],
    host: str = "en.wikipedia.org",
    pages_per_seed: int = DEFAULT_PAGES_PER_SEED,
) -> list[str]:
    """Search each seed and take the top pages, de-duplicated across seeds.

    Order is preserved so a truncated build is the highest-frequency seeds'
    pages rather than an arbitrary subset.
    """
    seen: dict[str, None] = {}
    for seed in seeds:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": seed,
            "srlimit": str(pages_per_seed),
            "srnamespace": "0",
            "format": "json",
            "formatversion": "2",
        }
        payload = fetch_json(f"https://{host}/w/api.php?{urlencode(params)}")
        query = payload.get("query")
        hits = query.get("search", []) if isinstance(query, dict) else []
        for hit in hits if isinstance(hits, list) else []:
            if isinstance(hit, dict) and hit.get("title"):
                seen.setdefault(str(hit["title"]), None)
    return list(seen)
