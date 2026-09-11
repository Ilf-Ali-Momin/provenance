"""Article extraction, and the publisher marker stripping it does not do.

trafilatura removes navigation, share widgets and footers, which is most of
the boilerplate problem. Measured on 2026-08-31, it does **not** remove two
things the OOD construction rules in DATA.md require gone:

    input : <p class="byline">By Jane Doe, Staff Writer</p>
            <p>LONDON (Reuters) - The first body sentence.</p>
    output: Headline here By Jane Doe, Staff Writer LONDON (Reuters) - The
            first body sentence.

Both the byline and the dateline survive into the extracted text. Shipping
that into the OOD suite would reintroduce precisely the leakage this project
exists to document: the dateline prefix is the single strongest publisher tell
in the corpus the leakage analysis criticises, and a byline identifies the
outlet's staff. Stripping them is therefore not a tidying step, it is the
construction rule, and it is applied identically to every stratum.

What is removed is recorded per document rather than discarded silently, so
the writeup can state how often each marker was present per stratum instead of
asserting that stripping worked.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import trafilatura

from pipeline.text.config import ExtractSettings
from pipeline.text.normalise import has_invisible, normalise

# A wire dateline: an upper case place, an optional date or region, then the
# agency in brackets, then a dash. Matches "LONDON (Reuters) - ",
# "WASHINGTON (AP) -- ", "NEW YORK, Jan 5 (Reuters) - ".
_DATELINE: Final = re.compile(
    r"^\s*"
    r"[A-Z][A-ZÀ-Þ .'’-]{1,40}"  # SHOUTED PLACE NAME
    r"(?:,\s*[^()\n]{1,40})?"  # optional ", Jan 5" or ", France"
    r"\s*\(\s*[A-Za-z][A-Za-z .&-]{1,30}\s*\)"  # (Reuters), (AP), (CNN)
    r"\s*[-–—]{1,3}\s*"
)

# A dateline without brackets: "LONDON, Jan 5 - " or "WASHINGTON -- ".
_BARE_DATELINE: Final = re.compile(
    r"^\s*[A-Z][A-ZÀ-Þ .'’-]{2,40}"
    r"(?:,\s*[A-Z][a-z]{2,9}\.?\s*\d{1,2}(?:,\s*\d{4})?)?"
    r"\s*[-–—]{2,3}\s*"
)

_BYLINE: Final = re.compile(
    r"^\s*(?:By|BY|By:|Written by|Reporting by)\s+"
    r"[A-Z][\w.'’-]*(?:\s+(?:and\s+)?[A-Z][\w.'’-]*){0,4}"
    r"(?:\s*,\s*[^\n]{0,60})?\s*$",
    re.MULTILINE,
)

# A byline written as a name prefix rather than with "By": some outlets open
# opinion pieces "DAVID MARCUS: Congress set to ...". Measured on live Fox News
# copy on 2026-09-06, where the "By" rule left the author name in the corpus.
# Two or more capitalised words are required so that single word openers such
# as "BREAKING:" or "UPDATE:" are not mistaken for an author.
_CAPS_BYLINE: Final = re.compile(
    r"^\s*[A-Z][A-Z'\u2019-]+(?:\s+[A-Z][A-Z'\u2019-]+){1,3}:\s+",
    re.MULTILINE,
)

_BOILERPLATE: Final = re.compile(
    r"^\s*(?:"
    r"(?:All rights reserved|Copyright|©)\b.*"
    r"|Sign up (?:for|to)\b.*"
    r"|Subscribe\b.*"
    r"|Follow us\b.*"
    r"|Read more\b.*"
    r"|Related(?: stories| articles)?\s*:?.*"
    r"|Share this\b.*"
    r"|Advertisement\s*"
    r"|Our Standards:.*"
    r"|This article was (?:originally )?published.*"
    r")$",
    re.MULTILINE | re.IGNORECASE,
)


# Newsletter and subscription promos sitting inside a paragraph rather than on
# their own line, which the line anchored boilerplate rule cannot see. Measured
# in the 2026-09-11 pilot: 6 of 10 Vox articles kept one. A promo names a house
# product, so it is publisher identity.
#
# The real shape spans two sentences ("...first featured in the Future Perfect
# newsletter. Sign up here."), so a single sentence regex requiring both words
# missed every one. The rule is by adjacency instead: a sentence mentioning a
# newsletter is removed together with a neighbouring sentence that says sign up
# or subscribe, or alone when it says both. "Subscribe" without a newsletter
# nearby is ordinary content and stays.
_SENTENCE_PART: Final = re.compile(r"[^.!?]+[.!?]*\s*")
_NEWSLETTER: Final = re.compile(r"\bnewsletters?\b", re.IGNORECASE)
_SIGNUP: Final = re.compile(r"\b(?:sign\s+up|subscribe)\b", re.IGNORECASE)


def strip_newsletter_promos(text: str) -> tuple[str, int]:
    """Remove promo sentences, returning the text and how many were removed."""
    lines_out: list[str] = []
    removed = 0
    for line in text.split("\n"):
        parts = _SENTENCE_PART.findall(line) or [line]
        drop: set[int] = set()
        for index, part in enumerate(parts):
            if not _NEWSLETTER.search(part):
                continue
            if _SIGNUP.search(part):
                drop.add(index)
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(parts) and _SIGNUP.search(parts[neighbour]):
                    drop.update({index, neighbour})
        if drop:
            removed += len(drop)
            line = "".join(p for k, p in enumerate(parts) if k not in drop).rstrip()
        lines_out.append(line)
    return "\n".join(lines_out), removed


OUTLET_MASK: Final = "[OUTLET]"


def mask_outlet_names(text: str, names: Sequence[str]) -> tuple[str, int]:
    """Replace a publisher's own names with a constant token.

    Only ever called with the names of the publisher whose article this is,
    so a mention of another outlet is left alone as content. Longest names go
    first so "Fox News Digital" is masked whole rather than leaving "Digital"
    behind. Matching is case sensitive on word boundaries; see
    eval/ood_collection/outlet_names.yaml for why each name is narrow.
    """
    total = 0
    for name in sorted(set(names), key=len, reverse=True):
        pattern = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\w])")
        text, count = pattern.subn(OUTLET_MASK, text)
        total += count
    return text, total


_HTML_LANG: Final = re.compile(
    r"<html\b[^>]*?\blang\s*=\s*[\"']?([A-Za-z]{2,3}(?:[-_][A-Za-z0-9]+)*)", re.IGNORECASE
)
_WORD: Final = re.compile(r"[A-Za-z']+")

# Common English function words. "a" is left out on purpose: it is also a
# Spanish and Portuguese preposition and would lift the ratio for exactly the
# languages this check exists to catch.
_ENGLISH_FUNCTION_WORDS: Final = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "that",
        "is",
        "was",
        "for",
        "on",
        "it",
        "with",
        "as",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "be",
        "by",
        "at",
        "this",
        "have",
        "from",
        "or",
        "had",
        "not",
        "are",
        "but",
        "an",
        "were",
        "which",
        "would",
        "will",
        "has",
        "been",
        "said",
        "who",
        "its",
        "we",
        "you",
        "there",
    ]
)


def declared_language(html: str) -> str | None:
    """The language the page declares on its html element, if any."""
    match = _HTML_LANG.search(html)
    return match.group(1) if match else None


def english_ratio(text: str, sample: int = 400) -> float:
    """Share of the first ``sample`` word tokens that are English function words.

    A declaration can be wrong or missing, so the text is checked as well.
    Digits and punctuation are not tokens here, which keeps a table of numbers
    from reading as a foreign language.
    """
    words = _WORD.findall(text.lower())[:sample]
    if not words:
        return 0.0
    return sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS) / len(words)


@dataclass(frozen=True)
class ExtractedArticle:
    """Extracted text plus what was removed to get there.

    ``removed`` exists so the writeup can report marker prevalence per
    stratum. A construction rule whose effect is never measured is a claim,
    not a control.
    """

    text: str
    title: str | None
    published: str | None
    removed: dict[str, int] = field(default_factory=dict)
    had_invisible: bool = False
    declared_language: str | None = None
    english_ratio: float = 1.0


def strip_publisher_markers(
    text: str, outlet_names: Sequence[str] = ()
) -> tuple[str, dict[str, int]]:
    """Remove datelines, bylines and residual boilerplate.

    Returns the cleaned text and a count of what each rule removed. Applied
    line by line for datelines, because trafilatura may join the byline and
    the dateline onto the same line as the headline.
    """
    counts: dict[str, int] = {}

    def _count(key: str, n: int) -> None:
        if n:
            counts[key] = counts.get(key, 0) + n

    text, n = _BYLINE.subn("", text)
    _count("byline", n)
    text, n = _CAPS_BYLINE.subn("", text)
    _count("caps_byline", n)
    text, n = _BOILERPLATE.subn("", text)
    _count("boilerplate", n)
    text, n = strip_newsletter_promos(text)
    _count("newsletter_promo", n)
    if outlet_names:
        text, n = mask_outlet_names(text, outlet_names)
        _count("outlet_name", n)

    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped, n1 = _DATELINE.subn("", line, count=1)
        if n1:
            _count("dateline", n1)
        else:
            stripped, n2 = _BARE_DATELINE.subn("", line, count=1)
            if n2:
                _count("bare_dateline", n2)
        cleaned.append(stripped)

    return "\n".join(cleaned), counts


def extract_article(
    html: str,
    *,
    url: str | None = None,
    settings: ExtractSettings | None = None,
    outlet_names: Sequence[str] = (),
) -> ExtractedArticle | None:
    """Extract article text from HTML, normalised and publisher stripped.

    Returns None when no article text could be recovered, which is a normal
    outcome for a paywall or an index page and is recorded as a collection
    failure rather than retried into a different extractor.
    """
    settings = settings or ExtractSettings()

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,  # SECURITY.md control 2: non rendered content
        include_tables=False,
        favor_precision=True,  # prefer dropping boilerplate over keeping it
    )
    if not text:
        return None

    # Measured on the extracted text, before normalisation removes it, not on
    # the raw page. The 2026-09-11 pilot flagged 24 of 50 documents from raw
    # HTML, where zero width characters are routine inside scripts and markup,
    # and 0 of 50 once measured on the text a model would actually receive.
    # Only the latter says anything about injection or evasion.
    invisible = has_invisible(text)

    meta = trafilatura.extract_metadata(html)
    text = normalise(text)
    text, removed = strip_publisher_markers(text, outlet_names)
    text = normalise(text)  # collapse whatever the removals left behind

    # A page with no article still yields its navigation text. Without this
    # floor those fragments enter the corpus as documents.
    if len(text.strip()) < settings.min_article_chars:
        return None

    return ExtractedArticle(
        text=text,
        title=normalise(meta.title) if meta and meta.title else None,
        published=meta.date if meta and meta.date else None,
        removed=removed,
        had_invisible=invisible,
        declared_language=declared_language(html),
        english_ratio=round(english_ratio(text), 4),
    )
