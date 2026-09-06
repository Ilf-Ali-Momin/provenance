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

from eval.ood_collection.mbfc import (
    derive_lean,
    derive_stratum,
    is_low_quality,
    look_up,
    polite_pause,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT: Final = Path("eval/ood_collection/sources.yaml")

# (domain, mbfc slug, factual category this project may assign itself)
#
# The factual categories are observable facts about what an outlet is: a wire
# service is a wire service. Everything else is left to MBFC, because it is an
# opinion about a named organisation and D007 forbids this project holding one.
CANDIDATES: Final[tuple[tuple[str, str, str | None], ...]] = (
    # --- wire ---------------------------------------------------------------
    ("reuters.com", "reuters", "wire"),
    ("apnews.com", "associated-press", "wire"),
    ("afp.com", "afp", "wire"),
    ("upi.com", "united-press-international", "wire"),
    # --- aggregator ---------------------------------------------------------
    ("news.yahoo.com", "yahoo-news", "aggregator"),
    ("msn.com", "msn", "aggregator"),
    ("news.google.com", "google-news", "aggregator"),
    ("realclearpolitics.com", "real-clear-politics", "aggregator"),
    ("drudgereport.com", "drudge-report", "aggregator"),
    # --- regional -----------------------------------------------------------
    ("chicagotribune.com", "chicago-tribune", "regional"),
    ("dallasnews.com", "dallas-morning-news", "regional"),
    ("seattletimes.com", "seattle-times", "regional"),
    ("bostonglobe.com", "boston-globe", "regional"),
    ("sfchronicle.com", "san-francisco-chronicle", "regional"),
    ("denverpost.com", "denver-post", "regional"),
    ("miamiherald.com", "miami-herald", "regional"),
    ("manchestereveningnews.co.uk", "manchester-evening-news", "regional"),
    ("liverpoolecho.co.uk", "liverpool-echo", "regional"),
    # --- MBFC decides the rest ---------------------------------------------
    # Selected for depth in each lean bucket, not for a balanced pool. Sampling
    # draws equal numbers per bucket (D017), so what the pool needs is enough
    # domains on each side to make that possible, and the achieved counts are
    # reported rather than assumed.
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
    # Added to deepen `national`: four domains cannot support a grouped split
    # by domain across train, dev and frozen holdout.
    ("csmonitor.com", "christian-science-monitor", None),
    ("c-span.org", "c-span", None),
    ("nature.com", "nature", None),
    ("newscientist.com", "new-scientist", None),
    ("axios.com", "axios", None),
    ("pbs.org", "pbs-news-hour", None),
    ("aljazeera.com", "al-jazeera", None),
    ("marketwatch.com", "marketwatch", None),
    ("theatlantic.com", "the-atlantic", None),
    ("telegraph.co.uk", "telegraph", None),
    ("foxnews.com", "fox-news", None),
    ("nationalreview.com", "national-review", None),
    ("reason.com", "reason", None),
    ("washingtonexaminer.com", "washington-examiner", None),
    ("thedispatch.com", "the-dispatch", None),
    ("nypost.com", "new-york-post", None),
    ("spectator.co.uk", "the-spectator", None),
    ("nationalpost.com", "national-post", None),
    ("dailywire.com", "the-daily-wire", None),
    ("thefederalist.com", "the-federalist", None),
    ("dailymail.co.uk", "daily-mail", None),
    ("newsmax.com", "newsmax", None),
    ("breitbart.com", "breitbart", None),
    ("oann.com", "one-america-news-network", None),
    ("thegatewaypundit.com", "the-gateway-pundit", None),
    ("motherjones.com", "mother-jones", None),
    ("thedailybeast.com", "daily-beast", None),
    ("theintercept.com", "the-intercept", None),
    ("jacobin.com", "jacobin", None),
    ("dailykos.com", "daily-kos", None),
    ("vox.com", "vox", None),
    ("slate.com", "slate", None),
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
# `lean` and `low_quality` are separate axes and neither is folded into
# `stratum`. Credibility never decides placement. See D016 for the measurement
# and D017 for the resolution.
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
                "low_quality": is_low_quality(obs),
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
    logger.info("low_quality flag: %s", dict(Counter(r["low_quality"] for r in records)))
    for stratum in ("partisan", "national"):
        leans = Counter(r["lean"] for r in records if r["stratum"] == stratum)
        left = leans["left"] + leans["lean_left"]
        right = leans["right"] + leans["lean_right"]
        logger.info(
            "%-10s left-of-centre=%d right-of-centre=%d %s", stratum, left, right, dict(leans)
        )
    # The check D017 exists to make possible: the flag must not be one sided
    # within a stratum, or it has become a proxy for lean again.
    flagged = Counter(r["lean"] for r in records if r["low_quality"] is True)
    logger.info("lean of low_quality flagged domains: %s", dict(flagged))
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
