"""BM25 lexical scoring, implemented rather than imported.

ARCHITECTURE.md: "dense similarity combined with BM25 lexical overlap,
weighted, because pure dense retrieval loses on proper nouns and numbers, which
is precisely the vocabulary claims are made of".

Written here, roughly forty lines, instead of adding a dependency. The brief
says to ask before adding one, and the case for asking is weak: BM25 is a
closed-form formula with two parameters, the alternative packages differ mainly
in which variant they implement, and a retrieval score that has to be swept and
explained is better as code that can be read than as a call into someone
else's default.

The variant is Okapi BM25 with the standard IDF that cannot go negative. The
textbook IDF, `log((N - df + 0.5) / (df + 0.5))`, is negative for a term in
more than half the documents, so a document is *penalised* for containing a
common word. On a news corpus "said" and "percent" clear that threshold easily,
and the penalty silently reorders results. The `+ 1` inside the log is the
usual fix and is applied here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Final

K1: Final = 1.5
B: Final = 0.75
_WORD: Final = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")


def tokenise(text: str) -> list[str]:
    """Lowercase word tokens, keeping the vocabulary claims are made of.

    Numbers survive whole: "4.1" is one token, not "4" and "1". The pattern
    keeps internal periods, apostrophes and hyphens for the same reason the
    dense side needs help here - "4.1%", "U.S." and "twenty-year" are exactly
    the tokens that distinguish one claim about a figure from another.
    """
    return _WORD.findall(text.lower())


class Bm25Index:
    """An in-memory BM25 index over a fixed set of documents."""

    def __init__(self, documents: dict[str, str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = list(documents)
        self.tokens = {i: tokenise(t) for i, t in documents.items()}
        self.lengths = {i: len(t) for i, t in self.tokens.items()}
        self.frequencies = {i: Counter(t) for i, t in self.tokens.items()}
        n = len(self.doc_ids)
        self.average_length = (sum(self.lengths.values()) / n) if n else 0.0
        document_frequency = Counter(term for t in self.tokens.values() for term in set(t))
        # The non-negative IDF. See the module docstring: the textbook form
        # penalises a document for containing a word most documents contain.
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        """BM25 of one document against the query."""
        if self.average_length == 0.0:
            return 0.0
        frequencies = self.frequencies[doc_id]
        length = self.lengths[doc_id]
        total = 0.0
        for term in tokenise(query):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
            total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return total

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Every document with a non-zero score, best first.

        Zero-scoring documents are dropped rather than returned at the tail.
        The retrieval failure rate is defined as "nothing above the relevance
        threshold was returned", and padding the list with documents that share
        no term with the query would make that rate unmeasurable.
        """
        scored = [(i, self.score(query, i)) for i in self.doc_ids]
        ranked = sorted((r for r in scored if r[1] > 0), key=lambda r: (-r[1], r[0]))
        return ranked[:k] if k is not None else ranked
