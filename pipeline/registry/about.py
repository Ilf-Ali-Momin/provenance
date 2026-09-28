"""What a publisher's own pages say about themselves, recorded as observations.

ARCHITECTURE.md asks the registry for "ownership and masthead presence from
structured extraction of the about page, corrections policy presence".

**This is the most defamation-exposed code in the project, and the exposure is
in the phrasing rather than the parsing.** "This publisher has no corrections
policy" is an assertion about a named organisation, and it is one this project
has no way to establish: a policy might sit behind a login, in a PDF, in a
footer this code did not read, or on a page whose link text nobody anticipated.
What *can* be established is narrower and checkable:

    on 2026-09-29, the homepage at https://example.org/ linked a page titled
    "Corrections" at https://example.org/corrections, and that page contains
    the sentence "We correct errors promptly and note them at the foot of the
    article."

That is what an observation here means. `source_url` names the page that was
read, `observed_at` names when, and `evidence` quotes what was matched, so a
reader can go and disagree. SECURITY.md control 8 and D007 both forbid the
version where this composes into a judgement about the publisher, and the
schema already makes that unrepresentable.

**A false negative is the harmful direction, so absence is recorded as what was
looked at rather than as a fact about the publisher.** `MASTHEAD_PRESENT: False`
with `source_url` pointing at the homepage and evidence saying which link texts
were searched for is a statement this project can defend. The same value with
no evidence would not be.

**Links are followed, not guessed.** Requesting /corrections, /about, /ethics
and so on against every domain generates 404s at publishers who have done
nothing wrong, and it finds pages nobody links to, which are not the pages a
reader could have found either. This reads the homepage's own links, which is
both more accurate and less rude.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, NamedTuple
from urllib.parse import urljoin, urlsplit

import structlog
from lxml import html as lxml_html

from pipeline.registry.schema import Observation, ObservationKind, Source

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from datetime import datetime

# Link text or href fragments that mark each kind of page. Matched against both,
# because some publishers title the link "Editorial standards" and put it at
# /corrections, and some do the reverse.
PAGE_PATTERNS: Final[dict[ObservationKind, tuple[str, ...]]] = {
    ObservationKind.CORRECTIONS_POLICY_PRESENT: (
        "correction",
        "accuracy",
        "editorial standard",
        "editorial policy",
        "editorial guideline",
        "complaint",
        "editorial code",
    ),
    ObservationKind.MASTHEAD_PRESENT: (
        "masthead",
        "our team",
        "editorial team",
        "our staff",
        "our people",
        "who we are",
        "meet the team",
        "contributors",
        "our journalists",
    ),
    ObservationKind.OWNERSHIP_STATED: (
        "about us",
        "about",
        "ownership",
        "who owns",
        "imprint",
        "impressum",
        "funding",
        "our funders",
    ),
}

# Phrases that, where they appear, are worth quoting to a reader alongside the
# page. They do not decide anything: see `observe` for why the version that
# gated on them reported False for a publisher that plainly had the page.
CONTENT_PATTERNS: Final[dict[ObservationKind, tuple[str, ...]]] = {
    ObservationKind.CORRECTIONS_POLICY_PRESENT: (
        r"\bcorrect(?:s|ed|ion|ions)\b",
        r"\berrors?\b",
        r"\bcomplaints? (?:procedure|policy|process)\b",
        r"\baccuracy\b",
    ),
    ObservationKind.MASTHEAD_PRESENT: (
        r"\bEditor(?:-in-Chief|ial Director| in Chief)?\b",
        r"\bManaging Editor\b",
        r"\bDeputy Editor\b",
        r"\bNews Editor\b",
        r"\bReporters?\b",
        r"\bCorrespondents?\b",
    ),
    ObservationKind.OWNERSHIP_STATED: (
        r"\bowned (?:and operated )?by\b",
        r"\bpublished by\b",
        r"\bparent company\b",
        r"\ba (?:division|subsidiary|part) of\b",
        r"\bfunded by\b",
        r"\bregistered (?:company|charity|in)\b",
    ),
}

logger = structlog.get_logger(__name__)

# A path that carries a date segment, or a long hyphenated slug, is an article.
# Found by running this against a real homepage: `/fashion/2026/sep/27/
# armani-plays-it-safe-ownership-milan-fashion-week` matched the ownership
# patterns on a substring, and `/stage/2026/sep/28/tru-review-...` matched
# "about". Four of nine candidates on that page were articles. Policy pages have
# short, flat paths; articles are dated and verbose.
_ARTICLE_DATE: Final = re.compile(r"/(19|20)\d{2}/")
_LONG_SLUG: Final = re.compile(r"/[^/]*?(?:-[^/]+){4,}")

MAX_LINKS: Final = 400
MAX_EVIDENCE_CHARS: Final = 300


class Candidate(NamedTuple):
    """A page the homepage linked, and the link text that suggested it."""

    kind: ObservationKind
    url: str
    link_text: str


def looks_like_an_article(url: str) -> bool:
    """Is this a story rather than a standing page?

    Matching patterns as substrings of a path finds articles whose slug happens
    to contain "about" or "ownership". A publisher's corrections policy does not
    live at a dated URL with a seven word slug.
    """
    path = urlsplit(url).path
    return bool(_ARTICLE_DATE.search(path) or _LONG_SLUG.search(path))


def _same_site(base: str, url: str) -> bool:
    """Is this the publisher's own page?

    A link to a press regulator's website is not this publisher's corrections
    policy, and recording it as one would attribute somebody else's document to
    them.
    """
    base_host = urlsplit(base).hostname or ""
    host = urlsplit(url).hostname or ""
    if not host:
        return False
    base_host, host = base_host.lower().removeprefix("www."), host.lower().removeprefix("www.")
    return host == base_host or host.endswith(f".{base_host}")


def find_candidates(homepage_html: str, base_url: str) -> list[Candidate]:
    """Pages the homepage links that look like the three kinds.

    Order follows the page, so the first match for a kind is the one a reader
    would most likely have clicked. Several kinds can point at one URL: an
    "About us" page often carries ownership and a masthead together.
    """
    try:
        tree = lxml_html.fromstring(homepage_html)
    except (ValueError, lxml_html.etree.ParserError):
        return []

    found: list[Candidate] = []
    seen: set[tuple[ObservationKind, str]] = set()
    for anchor in tree.xpath("//a[@href]")[:MAX_LINKS]:
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        text = " ".join(str(anchor.text_content() or "").split()).strip()
        url = urljoin(base_url, href)
        if not _same_site(base_url, url) or looks_like_an_article(url):
            continue
        haystack = f"{text.lower()} {urlsplit(url).path.lower()}"
        for kind, patterns in PAGE_PATTERNS.items():
            if any(pattern in haystack for pattern in patterns) and (kind, url) not in seen:
                seen.add((kind, url))
                found.append(Candidate(kind, url, text or urlsplit(url).path))
    return found


def visible_text(page: str) -> str:
    """The words a reader sees, with markup and scripts removed.

    Matching ran against raw HTML in the first version, which put
    `<title>Corrections and clarifications | The Guardian</title>` and a cache
    key into an evidence quote meant for a person to read. Evidence that a
    reader cannot parse is not evidence.
    """
    try:
        tree = lxml_html.fromstring(page)
    except (ValueError, lxml_html.etree.ParserError):
        return page
    for element in tree.xpath("//script | //style | //noscript"):
        element.getparent().remove(element)
    return " ".join(str(tree.text_content() or "").split())


def content_matches(page_text: str, kind: ObservationKind) -> list[str]:
    """Phrases on the page supporting what its link promised.

    Returned rather than counted, because the quotes become the observation's
    evidence. A bare True is an assertion; a True beside "We correct errors
    promptly" is something a reader can check.
    """
    matches: list[str] = []
    for pattern in CONTENT_PATTERNS[kind]:
        found = re.search(pattern, page_text, re.IGNORECASE)
        if not found:
            continue
        start = max(0, found.start() - 60)
        snippet = " ".join(page_text[start : found.end() + 120].split())
        matches.append(snippet[:MAX_EVIDENCE_CHARS])
    return matches


def page_summary(page: str, limit: int = 200) -> str:
    """A page's title and opening words, for a reader to judge it by."""
    text = visible_text(page)
    try:
        tree = lxml_html.fromstring(page)
        titles = tree.xpath("//title/text()")
        title = " ".join(str(titles[0]).split()) if titles else ""
    except (ValueError, IndexError, lxml_html.etree.ParserError):
        title = ""
    body = text[:limit].strip()
    return f"{title} — {body}" if title else body


def observe(
    homepage_html: str,
    base_url: str,
    observed_at: datetime,
    fetch_page: Callable[[str], str],
) -> list[Observation]:
    """What the homepage links, and what those pages say, as observations.

    `fetch_page` returns a page's HTML and is injected so the caller owns
    robots, rate limiting and the guarded transport. Markup is stripped here
    rather than by the caller, so an evidence quote is always words a reader
    can read.

    **What an observation asserts, narrowed after testing this against a real
    publisher.** The first version gated on keyword matches in the fetched page
    and only recorded `True` when enough fired. Run against the Guardian, it
    matched nothing on `/about` or `/index/contributors` - both plainly the
    pages they claim to be - and reported `False` for masthead and ownership on
    a publisher that has both. The patterns had been written against fixtures
    written to match them.

    The deeper problem is that the gate was the wrong idea rather than the wrong
    regex. "Does this page constitute an ownership statement" is a judgement
    about a named publisher, and D007 puts judgements outside this system. What
    is checkable is narrower: this homepage links a page with this title at this
    URL, and here is how it opens. A reader follows the link and decides.

    So the value records that such a page was linked, `source_url` names it, and
    `evidence` carries the link text, the page title and its opening words.
    Supporting phrases are still searched for and quoted when they appear,
    because they are useful to a reader - they no longer decide anything.
    """
    candidates = find_candidates(homepage_html, base_url)
    by_kind: dict[ObservationKind, list[Candidate]] = {}
    for candidate in candidates:
        by_kind.setdefault(candidate.kind, []).append(candidate)

    found: list[Observation] = []
    for kind in PAGE_PATTERNS:
        matched = False
        for candidate in by_kind.get(kind, []):
            try:
                page = fetch_page(candidate.url)
            except Exception as exc:
                # Unfetchable, so unobserved. Recording False here would turn a
                # server error into a claim about a named organisation. Logged
                # rather than passed over in silence: "the page would not load"
                # and "no such page is linked" produce the same observation and
                # are different facts about a run.
                logger.info(
                    "about_page_unreachable",
                    kind=kind.value,
                    url=candidate.url,
                    problem=type(exc).__name__,
                )
                continue
            quotes = content_matches(visible_text(page), kind)
            evidence = f'Linked as "{candidate.link_text}". {page_summary(page)}'
            if quotes:
                evidence += f" Phrases found: {quotes[0]}"
            found.append(
                Observation(
                    kind=kind,
                    value=True,
                    source=Source.ABOUT_PAGE,
                    source_url=candidate.url,
                    observed_at=observed_at,
                    evidence=evidence[:500],
                )
            )
            matched = True
            break
        if not matched:
            # What was looked for, on the page that was read. Not a statement
            # about the publisher: a page behind a login, in a PDF, or under a
            # link text nobody anticipated is invisible from here, and the
            # evidence names the search rather than the conclusion.
            searched = ", ".join(f'"{p}"' for p in PAGE_PATTERNS[kind][:4])
            found.append(
                Observation(
                    kind=kind,
                    value=False,
                    source=Source.ABOUT_PAGE,
                    source_url=base_url,
                    observed_at=observed_at,
                    evidence=(
                        f"No link on this homepage matched {searched}, and no such page was "
                        "reachable. Absent from this page, which is not the same as absent "
                        "from the publication."
                    )[:500],
                )
            )
    return found
