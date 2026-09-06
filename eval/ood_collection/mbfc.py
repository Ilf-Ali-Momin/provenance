"""Media Bias/Fact Check lookups for the OOD publisher strata.

Why this module exists at all. Two of the six publisher strata in the
collection plan cannot be assigned by this project. Placing a named outlet on
a political spectrum, or calling one low quality, is an opinion about a real
organisation. D007 settles that question for the source registry -- store dated
observations with recorded provenance, never a score this project computed --
and the same reasoning applies one level up, to how the evaluation corpus is
stratified. SECURITY.md control 8 gives the other half of the reason: an
opinion about a named publisher is a defamation surface, and a student project
should not carry one.

So the bias and credibility labels here are Media Bias/Fact Check's, recorded
with the URL they came from and the date they were read. The stratum is then
*derived* from those labels by a rule written down in `STRATUM_RULES`, so the
derivation is auditable and the label stays attributable to MBFC rather than to
us.

Access. https://mediabiasfactcheck.com/robots.txt was read on 2026-08-31 and
permits this: the only Disallow is a wpforms upload path. Requests go through
the SSRF guarded client like every other fetch, one at a time, with a
conservative delay. Only the short rating labels, the source URL and the
observation date are retained; no page prose is stored, because recording a
factual label with attribution is not redistributing their content.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Final

from pipeline.errors import ProvenanceError
from pipeline.net.fetch import fetch

MBFC_BASE: Final = "https://mediabiasfactcheck.com"
CRAWL_DELAY_S: Final = 3.0

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Capture a run of ALL CAPS tokens and stop at the first word containing a
# lowercase letter, which is where the next field's label begins. A greedy
# class over "[A-Z ]" instead runs past the value: measured, it turned
# "LEFT-CENTER" into "LEFT-CENTER F" by eating the F of "Factual", and
# corrupted one live record into "FAR RIGHT CONSPIRACY-PSEUSDOSCI".
_CAPS_RUN: Final = r"((?:[A-Z][A-Z\-]*\b)(?:\s+[A-Z][A-Z\-]*\b)*)"

_FIELDS: Final[dict[str, re.Pattern[str]]] = {
    "bias": re.compile(r"Bias Rating:\s*" + _CAPS_RUN),
    "factual_reporting": re.compile(r"Factual Reporting:\s*" + _CAPS_RUN),
    "credibility": re.compile(r"Credibility Rating:\s*" + _CAPS_RUN),
}

# How an MBFC bias label maps to a collection stratum. Written down so the
# derivation can be argued with, rather than buried in a conditional.
STRATUM_RULES: Final[dict[str, str]] = {
    "LEFT": "partisan",
    "LEFT-CENTER": "partisan",
    "RIGHT-CENTER": "partisan",
    "RIGHT": "partisan",
    "EXTREME LEFT": "partisan",
    "EXTREME RIGHT": "partisan",
    "LEAST BIASED": "national",
    "PRO-SCIENCE": "national",
    "QUESTIONABLE SOURCE": "low_quality",
    "CONSPIRACY-PSEUDOSCIENCE": "low_quality",
    "SATIRE": "low_quality",
}

# A credibility label at or below this is treated as low quality regardless of
# the bias label, because credibility is the axis the stratum is about.
LOW_CREDIBILITY: Final = frozenset({"LOW CREDIBILITY", "MIXED CREDIBILITY"})

# Political lean is recorded as its own axis and never collapsed into the
# stratum. Measured on 2026-08-31, deriving the stratum from credibility alone
# moved every low credibility right leaning outlet out of `partisan` while
# leaving left leaning ones in it, which would have produced a partisan
# stratum drawn from one side -- the exact confound the collection plan warns
# against, reintroduced by the classification rule rather than by the sample.
# Keeping the axes separate lets `partisan` be balanced by lean and lets the
# lean distribution of `low_quality` be reported rather than hidden.
LEAN_MAP: Final[dict[str, str]] = {
    "EXTREME LEFT": "left",
    "FAR LEFT": "left",
    "LEFT": "left",
    "LEFT-CENTER": "lean_left",
    "LEAST BIASED": "center",
    "PRO-SCIENCE": "center",
    "RIGHT-CENTER": "lean_right",
    "RIGHT": "right",
    "EXTREME RIGHT": "right",
    "FAR RIGHT": "right",
}


@dataclass(frozen=True)
class MbfcObservation:
    """One dated reading of an MBFC entry. Never a judgement of ours."""

    domain: str
    slug: str
    bias: str | None
    factual_reporting: str | None
    credibility: str | None
    source_url: str
    observed_at: str
    found: bool

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def slug_for(domain: str) -> str:
    """Best guess at the MBFC page slug for a domain.

    A guess, not a lookup: MBFC slugs are editorial. A miss is recorded as
    ``found: False`` and the domain is left unstratified rather than being
    assigned a label nobody published.
    """
    stem = domain.lower()
    for prefix in ("www.", "edition.", "news."):
        stem = stem.removeprefix(prefix)
    stem = re.sub(r"\.(com|org|net|co\.uk|us|news|info)$", "", stem)
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def parse_ratings(html: str) -> dict[str, str | None]:
    """Pull the short rating labels out of a page. No prose is retained."""
    text = _WS.sub(" ", _TAGS.sub(" ", html))
    out: dict[str, str | None] = {}
    for name, pattern in _FIELDS.items():
        match = pattern.search(text)
        out[name] = match.group(1).strip() if match else None
    return out


def look_up(domain: str, *, slug: str | None = None, today: str | None = None) -> MbfcObservation:
    """Read one MBFC entry. Network call, rate limited by the caller."""
    resolved = slug or slug_for(domain)
    url = f"{MBFC_BASE}/{resolved}/"
    observed = today or date.today().isoformat()

    try:
        response = fetch(url)
    except ProvenanceError:
        return MbfcObservation(
            domain=domain,
            slug=resolved,
            bias=None,
            factual_reporting=None,
            credibility=None,
            source_url=url,
            observed_at=observed,
            found=False,
        )

    ratings = parse_ratings(response.text)
    return MbfcObservation(
        domain=domain,
        slug=resolved,
        bias=ratings["bias"],
        factual_reporting=ratings["factual_reporting"],
        credibility=ratings["credibility"],
        source_url=url,
        observed_at=observed,
        found=ratings["bias"] is not None,
    )


def derive_lean(obs: MbfcObservation) -> str | None:
    """MBFC's bias label as a lean bucket, independent of the stratum.

    Recorded for every domain, including those in `low_quality`, so that a
    one sided stratum is visible in the manifest instead of being discovered
    later in the results.
    """
    if not obs.found or not obs.bias:
        return None
    label = obs.bias.upper()
    if label in LEAN_MAP:
        return LEAN_MAP[label]
    # MBFC emits compound labels such as "FAR RIGHT CONSPIRACY-PSEUDOSCIENCE".
    # Match on the longest known prefix rather than dropping the lean, which
    # would silently under-report one side of the low_quality stratum.
    for known in sorted(LEAN_MAP, key=len, reverse=True):
        if label.startswith(known):
            return LEAN_MAP[known]
    return None


def derive_stratum(obs: MbfcObservation, assigned: str | None) -> str | None:
    """Derive the publisher stratum from MBFC's labels.

    ``assigned`` is the factual category this project may set itself -- wire,
    regional, aggregator -- which wins, because "Reuters is a wire service" is
    an observable fact rather than an opinion. Only where no factual category
    applies does the MBFC label decide, and where MBFC has no entry the domain
    stays unstratified.
    """
    if assigned in {"wire", "regional", "aggregator"}:
        return assigned
    if not obs.found:
        return None
    if obs.credibility and obs.credibility.upper() in LOW_CREDIBILITY:
        return "low_quality"
    if obs.bias:
        return STRATUM_RULES.get(obs.bias.upper())
    return None


def polite_pause() -> None:
    """Conservative delay between requests (SECURITY.md control 5)."""
    time.sleep(CRAWL_DELAY_S)
