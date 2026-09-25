"""Chunk reference text for the local index.

ARCHITECTURE.md: "Chunked at roughly 300 tokens with 50 token overlap on
sentence boundaries." Each of those three is a decision with a failure mode.

**On sentence boundaries**, because a chunk cut mid-sentence gives the embedder
half a proposition and gives the NLI classifier half a premise. The boundaries
come from `pipeline.text.sentences`, the scanner already debugged for decimals
and abbreviations, rather than a second splitter that would drift from it.

**Roughly 300 tokens.** "Roughly" is doing real work and is kept rather than
tightened: the unit is sentences, so a chunk ends at the first boundary at or
past the target and overshoots by part of a sentence. Splitting a sentence to
hit exactly 300 would defeat the first decision.

**50 token overlap**, because a claim's evidence often straddles a boundary -
the figure in one sentence and the attribution in the next. Without overlap,
whichever side the cut falls on loses the other half, and no retrieval weighting
recovers it.

**Tokens are counted as whitespace-separated words, and that is an
approximation.** The embedder (`BAAI/bge-base-en-v1.5`) uses WordPiece, which
splits rarer words into several tokens, so a 300 word chunk is typically 350 to
420 model tokens. The model's limit is 512, so the approximation has headroom
and does not truncate. Counting exactly would mean loading the tokenizer here,
which makes chunking depend on the model being installed - and chunking is
also run by the evaluation harness, which does not serve the model. If the
sweep shows chunk size mattering near the boundary, the fix is to pass a
token counter in, not to import one.
"""

from __future__ import annotations

from typing import Final, NamedTuple, Protocol

from pipeline.text.sentences import sentence_spans

TARGET_TOKENS: Final = 300
OVERLAP_TOKENS: Final = 50


class Chunk(NamedTuple):
    """A span of a document, with offsets back into it.

    Offsets rather than text alone, for the reason claims carry them: a chunk
    shown as evidence has to be locatable in the source it came from, and a
    copy of the text cannot be checked against anything.
    """

    text: str
    start: int
    end: int
    # `ordinal`, not `index`: NamedTuple inherits tuple.index and shadowing it
    # replaces a method with an int.
    ordinal: int


class TokenCounter(Protocol):
    """How many tokens is this text?

    A Protocol so the sweep can pass the real tokenizer without this module
    importing it. Same reason the robots interface is one (D025): the caller
    that has the dependency supplies it.
    """

    def __call__(self, text: str) -> int: ...


def count_words(text: str) -> int:
    """The default counter. Whitespace words, per the module docstring."""
    return len(text.split())


def chunk_document(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    count: TokenCounter = count_words,
) -> list[Chunk]:
    """Split into overlapping chunks that begin and end at sentence boundaries."""
    if overlap_tokens >= target_tokens:
        msg = (
            f"overlap {overlap_tokens} must be below target {target_tokens} "
            "or chunking cannot advance"
        )
        raise ValueError(msg)

    spans = sentence_spans(text)
    if not spans:
        # No terminator in the text: one chunk, or nothing if it is blank.
        stripped = text.strip()
        if not stripped:
            return []
        lead = len(text) - len(text.lstrip())
        return [Chunk(stripped, lead, lead + len(stripped), 0)]

    sizes = [count(text[s:e]) for s, e in spans]
    chunks: list[Chunk] = []
    first = 0
    while first < len(spans):
        # Take sentences until the target is reached. At least one sentence is
        # always taken, so a single sentence longer than the target becomes its
        # own oversized chunk rather than an empty one or an infinite loop.
        last, total = first, 0
        while last < len(spans):
            total += sizes[last]
            last += 1
            if total >= target_tokens:
                break

        start, end = spans[first][0], spans[last - 1][1]
        body = text[start:end].strip()
        if body:
            chunks.append(Chunk(body, start, start + len(body), len(chunks)))

        if last >= len(spans):
            break

        # Step back over whole sentences until `overlap_tokens` is covered. The
        # step-back is bounded to leave at least one new sentence: without that,
        # a document of long sentences can re-emit the same window forever.
        back, carried = last, 0
        while back > first + 1 and carried < overlap_tokens:
            back -= 1
            carried += sizes[back]
        first = back

    return chunks
