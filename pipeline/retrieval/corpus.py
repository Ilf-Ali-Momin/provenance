"""Build the reference corpus that the local index holds.

DATA.md: "Reference corpus for the local index. Built rather than downloaded,
from openly licensed reference material and from a snapshot of published fact
checks accessed through their APIs where terms permit. Snapshot date recorded.
This is what pgvector actually holds."

**Provenance travels with the document, not only with the corpus.** DATA.md
records licence, source URL and access date per corpus in `data/MANIFEST.yaml`,
and that is where the corpus-level decision lives. It is not sufficient here.
The index holds chunks from several sources at once, and a chunk that reaches a
card has to carry the attribution its own licence requires - CC BY-SA text
displayed without its attribution is a licence breach, and a manifest entry
sitting in a different file cannot prevent one. So `ReferenceDocument` makes the
licence fields required, and a document that does not state its terms cannot be
constructed, let alone indexed (D046).

That is the same enforcement strategy used for the verdict the API cannot
return and the article text the job result cannot hold: make the unwanted state
unrepresentable rather than rely on a reviewer noticing.

**What this module does not do.** It does not fetch. Sources are passed in, so
the robots-respecting, rate-limited, SSRF-guarded fetcher stays the one path to
a remote host and this module stays testable without a network. It also makes
no licence judgement: it records what the source declares and refuses what
declares nothing.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, Field

from pipeline.retrieval.chunk import OVERLAP_TOKENS, TARGET_TOKENS, chunk_document

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator

    from pipeline.retrieval.store import PgVectorStore

# DATA.md's vocabulary, not a new one.
REDISTRIBUTION: Final = frozenset({"permitted", "prohibited", "permitted_with_attribution"})


class ReferenceDocument(BaseModel):
    """One document destined for the index, with the terms it arrived under.

    Every rights field is required. A source that cannot say what licence its
    text carries is a source whose text must not enter an index that feeds a
    user-facing card, and the place to stop that is here rather than in review.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str | None = None

    source_url: str = Field(min_length=1)
    publisher: str | None = None
    published: str | None = None

    licence: str = Field(min_length=1)
    licence_url: str = Field(min_length=1)
    redistribution: str
    # Required when the licence requires it, which this module cannot decide.
    # The source declares it; `attribution_required` below is what enforces the
    # pairing DATA.md names.
    attribution: str | None = None

    accessed: date
    snapshot_date: date

    def model_post_init(self, _context: object) -> None:
        if self.redistribution not in REDISTRIBUTION:
            msg = (
                f"redistribution must be one of {sorted(REDISTRIBUTION)}, "
                f"not {self.redistribution!r}"
            )
            raise ValueError(msg)
        # DATA.md: "attribution: (required if the licence requires it)". The one
        # case the schema can check is the one that says so in its own name.
        if self.redistribution == "permitted_with_attribution" and not self.attribution:
            msg = (
                f"{self.document_id} declares redistribution permitted_with_attribution "
                "and carries no attribution string; a chunk of it on a card would breach "
                "its own licence"
            )
            raise ValueError(msg)

    @property
    def indexable(self) -> bool:
        """May a chunk of this be shown as evidence?

        Prohibited redistribution does not mean the text may not be *read* - it
        means this project may not republish it. A card quotes the passage it
        retrieved, which is republication, so those documents do not enter the
        index at all rather than entering it with a flag someone has to check.
        """
        return self.redistribution != "prohibited"


class Source(Protocol):
    """Where reference documents come from.

    A Protocol so the fetching lives with the thing that has the credentials
    and the rate limiter, and so the builder is testable with a list.
    """

    @property
    def corpus_id(self) -> str: ...

    def documents(self) -> Iterator[ReferenceDocument]: ...


class BuildReport(BaseModel):
    """What a build did, in terms the manifest entry needs."""

    model_config = {"frozen": True}

    corpus_id: str
    documents_seen: int
    documents_indexed: int
    documents_skipped_not_redistributable: int
    chunks_indexed: int
    licences: dict[str, int]
    snapshot_date: date | None
    accessed_earliest: date | None
    accessed_latest: date | None


def build_index(
    source: Source,
    store: PgVectorStore,
    documents: Iterable[ReferenceDocument] | None = None,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> BuildReport:
    """Chunk, embed and index one source. Returns what the manifest needs.

    `documents` overrides `source.documents()` so a caller can resume or
    re-index a subset without the source knowing about it.
    """
    seen = indexed = skipped = chunks = 0
    licences: dict[str, int] = {}
    snapshots: set[date] = set()
    accessed: list[date] = []

    for document in documents if documents is not None else source.documents():
        seen += 1
        if not document.indexable:
            # Counted, not silently dropped: a build that skipped most of its
            # corpus for rights reasons is a different build from one that
            # indexed it, and the number belongs in the report.
            skipped += 1
            continue
        pieces = chunk_document(document.text, target_tokens, overlap_tokens)
        if not pieces:
            continue
        chunks += store.add(
            document.document_id,
            pieces,
            source_url=document.source_url,
            published=document.published,
            licence=document.licence,
            attribution=document.attribution,
        )
        indexed += 1
        licences[document.licence] = licences.get(document.licence, 0) + 1
        snapshots.add(document.snapshot_date)
        accessed.append(document.accessed)

    return BuildReport(
        corpus_id=source.corpus_id,
        documents_seen=seen,
        documents_indexed=indexed,
        documents_skipped_not_redistributable=skipped,
        chunks_indexed=chunks,
        licences=licences,
        # DATA.md requires a snapshot date. Several would mean the corpus does
        # not reflect one moment, so the report says so by carrying None rather
        # than picking one and making the manifest entry a guess.
        snapshot_date=next(iter(snapshots)) if len(snapshots) == 1 else None,
        accessed_earliest=min(accessed) if accessed else None,
        accessed_latest=max(accessed) if accessed else None,
    )
