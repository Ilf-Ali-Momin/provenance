"""The index schema and the store's contract, without a database.

The store is exercised against the compose stack under the `integration`
marker; these tests pin the decisions that are visible in the SQL itself,
because those are the ones that would otherwise only surface as a retrieval
number nobody could explain.
"""

from __future__ import annotations

import numpy as np
import pytest
from pipeline.retrieval.chunk import Chunk
from pipeline.retrieval.embed import DIMENSIONS
from pipeline.retrieval.store import HNSW_EF_CONSTRUCTION, HNSW_M, SCHEMA, TABLE


def test_the_index_is_hnsw_with_cosine_ops() -> None:
    """ARCHITECTURE.md names both. The operator class is chosen at build time,
    so getting it wrong bakes the error into the neighbour graph."""
    assert "USING hnsw" in SCHEMA
    assert "vector_cosine_ops" in SCHEMA


def test_the_vector_column_matches_the_embedder_dimension() -> None:
    assert f"vector({DIMENSIONS})" in SCHEMA


def test_the_hnsw_build_parameters_are_named_not_implicit() -> None:
    """A recall number that moved because someone rebuilt with other parameters
    would be untraceable."""
    assert f"m = {HNSW_M}" in SCHEMA
    assert f"ef_construction = {HNSW_EF_CONSTRUCTION}" in SCHEMA


def test_the_schema_is_idempotent() -> None:
    """Workers run it on start."""
    assert SCHEMA.count("IF NOT EXISTS") >= 3


def test_the_table_carries_offsets_so_evidence_stays_locatable() -> None:
    for column in ("start_offset", "end_offset", "document_id", "ordinal"):
        assert column in SCHEMA


def test_nothing_in_the_schema_holds_a_submitted_article() -> None:
    """SECURITY.md control 6. This index is reference material; the analysis
    path never writes here."""
    assert "analysis_id" not in SCHEMA
    assert "submitted" not in SCHEMA


class _FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.executed: list[str] = []
        self.params: list[object] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: object, params: object = None) -> None:
        self.executed.append(str(query))
        self.params.append(params)

    def executemany(self, query: object, params: object) -> None:
        self.executed.append(str(query))
        self.params.append(params)

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class _FakeConnection:
    def __init__(self, rows: list[tuple[object, ...]] | None = None) -> None:
        self.cur = _FakeCursor(rows or [])
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return self.cur

    def commit(self) -> None:
        self.commits += 1


class _StubEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(DIMENSIONS, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), DIMENSIONS), dtype=np.float32)


def _store(connection: object) -> object:
    pytest.importorskip("pgvector")
    from pipeline.retrieval.store import PgVectorStore

    # register_vector needs a real connection; bypass __init__ for the parts
    # under test, which are the SQL and the distance conversion.
    store = object.__new__(PgVectorStore)
    store.connection = connection  # type: ignore[attr-defined]
    store.embedder = _StubEmbedder()  # type: ignore[attr-defined]
    return store


def test_search_returns_similarity_not_distance() -> None:
    """The fusion min-maxes both sides and would invert the dense ranking if
    handed a distance."""
    connection = _FakeConnection([("a#0", 0.9), ("b#1", 0.4)])
    results = _store(connection).search("a claim", k=2)  # type: ignore[attr-defined]
    assert results == [("a#0", 0.9), ("b#1", 0.4)]
    assert "1 - (embedding <=> %s)" in connection.cur.executed[0]


def test_readding_a_document_replaces_its_chunks() -> None:
    """Re-chunking produces different boundaries; appending would leave old
    chunks retrievable with offsets into text that no longer exists."""
    connection = _FakeConnection()
    chunks = [Chunk("first sentence.", 0, 15, 0), Chunk("second sentence.", 16, 32, 1)]
    added = _store(connection).add("doc_a", chunks)  # type: ignore[attr-defined]
    assert added == 2
    assert "DELETE FROM" in connection.cur.executed[0]
    assert "INSERT INTO" in connection.cur.executed[1]


def test_adding_nothing_touches_no_sql() -> None:
    connection = _FakeConnection()
    assert _store(connection).add("doc_a", []) == 0  # type: ignore[attr-defined]
    assert connection.cur.executed == []


def test_chunk_ids_are_derived_from_the_document_and_ordinal() -> None:
    connection = _FakeConnection()
    _store(connection).add("doc_a", [Chunk("a sentence.", 0, 11, 3)])  # type: ignore[attr-defined]
    rows = connection.cur.params[1]
    assert rows[0][0] == "doc_a#3"


def test_get_with_no_ids_does_not_query() -> None:
    connection = _FakeConnection()
    assert _store(connection).get([]) == {}  # type: ignore[attr-defined]
    assert connection.cur.executed == []


def test_the_table_name_is_composed_not_interpolated() -> None:
    """S608 is right about the pattern regardless of who controls the value."""
    from pipeline.retrieval.store import _TABLE
    from psycopg import sql

    assert isinstance(_TABLE, sql.Identifier)
    assert TABLE == "evidence_chunks"
