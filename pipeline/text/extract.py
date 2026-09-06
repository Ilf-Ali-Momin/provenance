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


def strip_publisher_markers(text: str) -> tuple[str, dict[str, int]]:
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
    text, n = _BOILERPLATE.subn("", text)
    _count("boilerplate", n)

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
    html: str, *, url: str | None = None, settings: ExtractSettings | None = None
) -> ExtractedArticle | None:
    """Extract article text from HTML, normalised and publisher stripped.

    Returns None when no article text could be recovered, which is a normal
    outcome for a paywall or an index page and is recorded as a collection
    failure rather than retried into a different extractor.
    """
    settings = settings or ExtractSettings()
    invisible = has_invisible(html)

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,  # SECURITY.md control 2: non rendered content
        include_tables=False,
        favor_precision=True,  # prefer dropping boilerplate over keeping it
    )
    if not text:
        return None

    meta = trafilatura.extract_metadata(html)
    text = normalise(text)
    text, removed = strip_publisher_markers(text)
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
    )
