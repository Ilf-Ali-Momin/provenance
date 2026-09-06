"""Build `sources.yaml` by reading Media Bias/Fact Check for each candidate.

Run deliberately, not on import, and not in CI: it makes one network request
per candidate with a conservative delay. The candidate domains are chosen by
this project; every bias and credibility label in the output is MBFC's,
recorded with its source URL and the date it was read (D016).

    PYTHONPATH=. .venv/bin/python eval/ood_collection/build_sources.py
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Final

import yaml

from eval.ood_collection.mbfc import derive_lean, derive_stratum, look_up, polite_pause

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT: Final = Path("eval/ood_collection/sources.yaml")

# (domain, mbfc slug, factual category this project may assign itself)
#
# The factual categories are observable facts about what an outlet is: a wire
# service is a wire service. Everything else is left to MBFC, because it is an
# opinion about a named organisation and D007 forbids this project holding one.
CANDIDATES: Final[tuple[tuple[str, str, str | None], ...]] = (
    ("reuters.com", "reuters", "wire"),
    ("apnews.com", "associated-press", "wire"),
    ("afp.com", "agence-france-presse-afp", "wire"),
    ("news.yahoo.com", "yahoo-news", "aggregator"),
    ("msn.com", "msn-news", "aggregator"),
    ("chicagotribune.com", "chicago-tribune", "regional"),
    ("dallasnews.com", "the-dallas-morning-news", "regional"),
    ("seattletimes.com", "the-seattle-times", "regional"),
    ("manchestereveningnews.co.uk", "manchester-evening-news", "regional"),
    ("bbc.com", "bbc", None),
    ("theguardian.com", "the-guardian", None),
    ("nytimes.com", "new-york-times", None),
    ("wsj.com", "wall-street-journal", None),
    ("washingtonpost.com", "washington-post", None),
    ("npr.org", "npr", None),
    ("cnn.com", "cnn", None),
    ("nbcnews.com", "nbc-news", None),
    ("cbsnews.com", "cbs-news", None),
    ("usatoday.com", "usa-today", None),
    ("thehill.com", "the-hill", None),
    ("politico.com", "politico", None),
    ("bloomberg.com", "bloomberg", None),
    ("cnbc.com", "cnbc", None),
    ("forbes.com", "forbes", None),
    ("statnews.com", "stat-news", None),
    ("sciencedaily.com", "science-daily", None),
    ("foxnews.com", "fox-news", None),
    ("nationalreview.com", "national-review", None),
    ("reason.com", "reason-magazine", None),
    ("dailywire.com", "the-daily-wire", None),
    ("newsmax.com", "newsmax", None),
    ("breitbart.com", "breitbart", None),
    ("oann.com", "one-america-news-network", None),
    ("thegatewaypundit.com", "the-gateway-pundit", None),
    ("motherjones.com", "mother-jones", None),
    ("thedailybeast.com", "daily-beast", None),
    ("theintercept.com", "the-intercept", None),
    ("jacobin.com", "jacobin-magazine", None),
    ("dailykos.com", "daily-kos", None),
    ("infowars.com", "infowars", None),
    ("naturalnews.com", "natural-news", None),
)

HEADER: Final = """# Candidate publishers for the OOD suite.
#
# `stratum` and `lean` are DERIVED from Media Bias/Fact Check labels by the
# rules in eval/ood_collection/mbfc.py. The labels themselves are MBFC's,
# recorded with the URL they came from and the date read. This project does
# not rate publishers (D007, SECURITY.md control 8).
#
# `assigned_category` is the one field this project sets itself, and only for
# factual categories: a wire service is a wire service.
#
# `lean` is a separate axis and is never folded into `stratum`. See D016 for
# the measured reason, and for the stratum question that is still open.
#
# Regenerate with: PYTHONPATH=. .venv/bin/python eval/ood_collection/build_sources.py

"""


def build() -> list[dict[str, object]]:
    """Read every candidate. One request each, politely spaced."""
    records: list[dict[str, object]] = []
    for index, (domain, slug, assigned) in enumerate(CANDIDATES, start=1):
        obs = look_up(domain, slug=slug)
        stratum = derive_stratum(obs, assigned)
        lean = derive_lean(obs)
        records.append(
            {
                "domain": domain,
                "stratum": stratum,
                "lean": lean,
                "assigned_category": assigned,
                "mbfc": obs.as_record(),
            }
        )
        logger.info(
            "[%2d/%d] %-30s stratum=%-12s lean=%-11s bias=%s",
            index,
            len(CANDIDATES),
            domain,
            stratum,
            lean,
            obs.bias,
        )
        polite_pause()
    return records


def summarise(records: list[dict[str, object]]) -> None:
    """Log the distribution that D016 is about.

    Reported every run rather than computed once, because a one sided stratum
    is only a finding if somebody sees it.
    """
    logger.info("stratum counts: %s", dict(Counter(r["stratum"] for r in records)))
    for stratum in ("partisan", "low_quality"):
        leans = Counter(r["lean"] for r in records if r["stratum"] == stratum)
        left = leans["left"] + leans["lean_left"]
        right = leans["right"] + leans["lean_right"]
        logger.info(
            "%-12s left-of-centre=%d right-of-centre=%d %s", stratum, left, right, dict(leans)
        )
    misses = [r["domain"] for r in records if not r["mbfc"]["found"]]  # type: ignore[index]
    logger.info("no MBFC entry under the slug tried: %s", misses or "none")


def main() -> None:
    records = build()
    with OUT.open("w", encoding="utf-8") as handle:
        handle.write(HEADER)
        yaml.safe_dump({"publishers": records}, handle, sort_keys=False, width=88)
    summarise(records)
    logger.info("wrote %s (%d publishers)", OUT, len(records))


if __name__ == "__main__":
    main()
