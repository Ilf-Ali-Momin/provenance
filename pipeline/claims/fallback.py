"""Rule based extractor, for when the LLM path is unavailable.

D004 gives the extractor a fallback so "the pipeline degrades rather than
fails". This is that fallback, and it is deliberately worse: it locates
sentences that look like claims and types them by surface cues.

**A stated deviation from ARCHITECTURE.md (D037).** That document specifies "a
rule based sentence selector using dependency parse features", which means
spaCy and a model download. This implementation uses surface cues instead and
adds no dependency. The reason is ordering rather than disagreement: the
fallback's value is unknown until it is scored against the annotation set, and
adding a heavy dependency to a degraded path before knowing whether the cheap
version is adequate is the wrong order. If the scored gap is large, the parse
based version is the fix and the dependency is justified by a number.

The type cues are the same distinctions the annotation guideline draws, in the
order that matters: attribution is tested before assertion, because a sentence
reporting speech is an attribution even when its content is checkable, and that
is the boundary the guideline calls the hardest part of the task.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

from pipeline.claims.schema import Claim, ClaimType, claim_id
from pipeline.text.sentences import sentence_spans

MAX_CLAIMS: Final = 5
# Low, because the first version used 40 and silently dropped "The policy was
# a disgrace." (26) and "Rates will rise again before June." (34) - both
# perfectly good claims of exactly the types this is meant to tell apart. The
# floor exists to skip fragments, not short sentences.
MIN_SENTENCE_CHARS: Final = 15
VERSION: Final = "fallback:v1"

# Reported speech. Tested first: the guideline's rule is that attribution beats
# assertion whenever the sentence reports speech.
_ATTRIBUTION = re.compile(
    r"\b(?:said|says|told|added|argued|claimed|denied|insisted|according to|"
    r"wrote|stated|reported(?:ly)?|acknowledged|warned|announced)\b",
    re.IGNORECASE,
)
_PREDICTION = re.compile(
    r"\b(?:will|won't|shall|expected to|forecast|projected|due to|set to|"
    r"is likely to|are likely to|by \d{4})\b",
    re.IGNORECASE,
)
_OPINION = re.compile(
    r"\b(?:disgrace|outrageous|beautiful|terrible|wonderful|shameful|absurd|"
    r"remarkable|disappointing|should|ought to|must be|deserves)\b",
    re.IGNORECASE,
)
# A checkable sentence usually carries a number, a quantity, or a proper noun.
# Not a parse, and it does not pretend to be one.
#
# The proper noun test deliberately skips the first word. The first version
# used \b[A-Z][a-z]{2,}\b, which matches the capital that starts every
# sentence, so "carries evidence" silently meant "is a sentence": "And so it
# goes on and on without end." was typed as a factual assertion.
_EVIDENCE = re.compile(r"\d|\b(?:percent|per cent|million|billion|thousand)\b")
_PROPER_NOUN = re.compile(r"(?<=[a-z,;:]\s)[A-Z][a-z]{2,}")


def classify(sentence: str) -> ClaimType | None:
    """Type one sentence, or None when it carries no checkable content."""
    if _ATTRIBUTION.search(sentence):
        return ClaimType.ATTRIBUTION
    if _PREDICTION.search(sentence):
        return ClaimType.PREDICTION
    if _OPINION.search(sentence):
        return ClaimType.OPINION
    if _EVIDENCE.search(sentence) or _PROPER_NOUN.search(sentence):
        return ClaimType.FACTUAL_ASSERTION
    return None


def extract_claims_offline(source: str, max_claims: int = MAX_CLAIMS) -> list[Claim]:
    """Locate up to `max_claims` claim-like sentences with real offsets.

    Offsets come from the match position, so they resolve into the source by
    construction. The degraded path still cannot emit text the source does not
    contain.
    """
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    claims: list[Claim] = []
    for start, end in sentence_spans(source):
        if len(claims) >= max_claims:
            break
        sentence = source[start:end].strip()
        if len(sentence) < MIN_SENTENCE_CHARS:
            continue
        claim_type = classify(sentence)
        if claim_type is None:
            continue
        claims.append(
            Claim(
                claim_id=claim_id(source_sha, start, start + len(sentence)),
                text=source[start : start + len(sentence)],
                start=start,
                end=start + len(sentence),
                type=claim_type,
                extractor_version=VERSION,
            )
        )
    return claims
