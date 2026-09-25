"""The local evidence index: pgvector, HNSW, cosine.

ARCHITECTURE.md: "The local index holds an embedded corpus of reference
material ... stored in pgvector with an HNSW index. Cosine distance."

**What is stored, and what DATA.md permits.** The index holds reference
material, not analysed articles. That distinction is the whole reason this is
allowed to be durable storage at all: SECURITY.md control 6 says article text
submitted for analysis is never persisted, and nothing here writes on the
analysis path. The rows come from the reference corpus builder, which reads a
corpus the project assembled, and the submitted URL's text never reaches this
module. Live retrieval embeds on the fly and is not written here either.

**Cosine on normalised vectors.** `vector_cosine_ops` is the operator class,
and `embed.py` returns unit vectors, so cosine distance is `1 - dot`. Both
halves matter: an unnormalised vector would make the neighbour graph reflect
magnitude as well as direction, and the graph is built once at index time, so
the error would be baked into every later search rather than showing up as a
bad score.

**The score returned is a similarity, not a distance.** Postgres returns
distance because that is what the index orders by, and every consumer here
wants larger-is-better: the hybrid fusion min-maxes both sides and would invert
the dense ranking if handed a distance. Converted once, at the boundary, rather
than in each caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from psycopg import sql

from pipeline.retrieval.embed import DIMENSIONS

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection

    from pipeline.retrieval.chunk import Chunk
    from pipeline.retrieval.embed import Embedder

TABLE: Final = "evidence_chunks"
# Composed rather than interpolated. The table name is a constant, so an
# f-string would be safe today and would be the thing someone later
# parameterises; ruff's S608 is right to refuse the pattern in a data access
# module regardless of who controls the value this week.
_TABLE: Final = sql.Identifier(TABLE)
# HNSW build parameters. pgvector's defaults, stated here rather than left
# implicit because they trade index build time against recall and EVALUATION.md
# measures recall - a number that moved because someone rebuilt the index with
# different parameters would be untraceable otherwise.
HNSW_M: Final = 16
HNSW_EF_CONSTRUCTION: Final = 64

SCHEMA: Final = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE} (
    chunk_id     TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset   INTEGER NOT NULL,
    text         TEXT NOT NULL,
    source_url   TEXT,
    published    TEXT,
    -- Rights travel with the chunk, not only with the corpus in
    -- data/MANIFEST.yaml. The index holds several sources at once and a card
    -- quotes one passage, so the attribution its licence demands has to be
    -- retrievable with that passage or it cannot be displayed (D046).
    licence      TEXT NOT NULL,
    attribution  TEXT,
    embedding    vector({DIMENSIONS}) NOT NULL
);

CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx
    ON {TABLE} USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS {TABLE}_document_idx ON {TABLE} (document_id);
"""


def create_schema(connection: Connection) -> None:
    """Idempotent. Safe to run on every worker start."""
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA)
    connection.commit()


class PgVectorStore:
    """Dense search over the reference index.

    Satisfies `hybrid.Searcher`, so the fusion and the evaluation harness can
    take it or a stub without knowing which.
    """

    def __init__(self, connection: Connection, embedder: Embedder) -> None:
        from pgvector.psycopg import register_vector

        register_vector(connection)
        self.connection = connection
        self.embedder = embedder

    def add(self, document_id: str, chunks: list[Chunk], **columns: str | None) -> int:
        """Index one document's chunks. Re-adding a document replaces it.

        `licence` is required and is a KeyError rather than a default, because a
        chunk with no recorded terms is one a card cannot legally display and a
        nullable column would make that state reachable.

        Replace rather than append, because a reference document that is
        re-fetched and re-chunked produces different boundaries: appending
        would leave the old chunks in the index, retrievable, with offsets into
        a version of the text that no longer exists.
        """
        if not chunks:
            return 0
        vectors = self.embedder.embed_passages([c.text for c in chunks])
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE document_id = %s").format(_TABLE),
                (document_id,),
            )
            cursor.executemany(
                sql.SQL(
                    """INSERT INTO {}
                       (chunk_id, document_id, ordinal, start_offset, end_offset,
                        text, source_url, published, licence, attribution, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                ).format(_TABLE),
                [
                    (
                        f"{document_id}#{chunk.ordinal}",
                        document_id,
                        chunk.ordinal,
                        chunk.start,
                        chunk.end,
                        chunk.text,
                        columns.get("source_url"),
                        columns.get("published"),
                        columns["licence"],
                        columns.get("attribution"),
                        vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        self.connection.commit()
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Nearest chunks, as (chunk_id, cosine similarity), best first."""
        vector = self.embedder.embed_query(query)
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """SELECT chunk_id, 1 - (embedding <=> %s) AS similarity
                       FROM {}
                       ORDER BY embedding <=> %s
                       LIMIT %s"""
                ).format(_TABLE),
                (vector, vector, k or 20),
            )
            return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]

    def get(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        """The stored rows for a set of chunk ids, for rendering evidence."""
        if not chunk_ids:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """SELECT chunk_id, document_id, ordinal, start_offset, end_offset,
                              text, source_url, published, licence, attribution
                       FROM {} WHERE chunk_id = ANY(%s)"""
                ).format(_TABLE),
                (chunk_ids,),
            )
            columns = [
                "chunk_id",
                "document_id",
                "ordinal",
                "start_offset",
                "end_offset",
                "text",
                "source_url",
                "published",
                "licence",
                "attribution",
            ]
            return {str(row[0]): dict(zip(columns, row, strict=True)) for row in cursor.fetchall()}
