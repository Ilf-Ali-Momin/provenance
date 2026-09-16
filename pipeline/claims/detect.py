"""Injection detection: control 2's fourth layer. Logged, never blocking.

SECURITY.md is specific about why this does not block: "its false positive rate
on legitimate text about AI is unacceptable for blocking". A news article about
a chatbot quotes prompts. An article about this very project would trip every
pattern here. Blocking on that would make the system refuse exactly the
journalism most worth analysing.

So this returns a count and a set of matched categories, for logging and for
rate limiting against. Nothing in the pipeline branches on it.
"""

from __future__ import annotations

import re
from typing import Final

PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all)\b"
        r"[^.\n]{0,20}\b(?:instruction|prompt|direction|rule)s?\b",
        re.IGNORECASE,
    ),
    "role_address": re.compile(
        r"\b(?:you are|act as|pretend to be|behave as)\b[^.\n]{0,30}"
        r"\b(?:an? )?(?:ai|assistant|model|language model|system)\b",
        re.IGNORECASE,
    ),
    "system_impersonation": re.compile(
        r"^\s*(?:system|assistant|developer|operator)\s*[:>\]]",
        re.IGNORECASE | re.MULTILINE,
    ),
    "output_steering": re.compile(
        r"\b(?:respond|reply|answer|output|return|say)\b[^.\n]{0,30}"
        r"\b(?:only|exactly|instead|with the following)\b",
        re.IGNORECASE,
    ),
    "credibility_steering": re.compile(
        r"\b(?:mark|report|classify|rate|label)\b[^.\n]{0,30}"
        r"\b(?:credible|trustworthy|reliable|true|verified|accurate)\b",
        re.IGNORECASE,
    ),
    "delimiter_probe": re.compile(r"<<<[0-9a-f]{8,}>>>|```\s*system|\[/?INST\]"),
}


def detect(text: str) -> dict[str, int]:
    """Categories matched, with a count each. Empty means nothing matched."""
    return {
        name: len(pattern.findall(text))
        for name, pattern in PATTERNS.items()
        if pattern.search(text)
    }
