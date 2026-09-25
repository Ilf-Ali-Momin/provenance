"""Sentence boundaries, found by scanning rather than by one clever regex.

Lifted out of `pipeline/claims/fallback.py`, where it was written and debugged,
because chunking needs the same boundaries: ARCHITECTURE.md specifies the local
index is "chunked at roughly 300 tokens with 50 token overlap on sentence
boundaries", and a second implementation would drift from this one.

The history is the argument for keeping it as a scan with named conditions. The
first version used `[^.!?]+` and cut "Unemployment fell to 4.1% in March" in
half at the decimal point, then reported the fragment "1% in March, the
statistics office said." as a claim - with offsets that resolved, so nothing
downstream caught it. The lookahead version that replaced it failed the same
way.
"""

from __future__ import annotations

import re
from typing import Final

# The abbreviations and the decimal rule are the named conditions.
_ABBREVIATIONS: Final = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sen",
        "rep",
        "gov",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "vs",
        "etc",
        "eg",
        "ie",
        "approx",
        "no",
        "u.s",
        "u.k",
        "e.u",
        "a.m",
        "p.m",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
)
_TERMINATOR: Final = re.compile(r"[.!?]")
_WORD_BEFORE: Final = re.compile(r"([A-Za-z.]+)$")


def _is_boundary(text: str, index: int) -> bool:
    """Is the terminator at `index` the end of a sentence?"""
    char = text[index]
    if char in "!?":
        return True
    # A decimal point: digit on both sides.
    if index and text[index - 1].isdigit() and index + 1 < len(text) and text[index + 1].isdigit():
        return False
    # A known abbreviation.
    word = _WORD_BEFORE.search(text[:index])
    if word and word.group(1).lower().strip(".") in _ABBREVIATIONS:
        return False
    # An initial: a single capital letter standing alone, as in "J. Baker".
    is_initial = bool(
        index and text[index - 1].isupper() and (index < 2 or not text[index - 2].isalpha())
    )
    return not is_initial


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, offsets into `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _TERMINATOR.finditer(text):
        if not _is_boundary(text, match.start()):
            continue
        end = match.end()
        while end < len(text) and text[end] in ".!?":
            end += 1
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        if chunk.strip():
            spans.append((start + lead, end))
        start = end
    return spans
