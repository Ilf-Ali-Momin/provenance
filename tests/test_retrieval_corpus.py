"""The reference corpus builder, and the rights it refuses to lose."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import numpy as np
import pytest
from pipeline.retrieval.corpus import ReferenceDocument, build_index
from pipeline.retrieval.embed import DIMENSIONS
from pydantic import ValidationError

TEXT = " ".join(["Unemployment fell to 4.1% in March and ministers called it encouraging."] * 40)


def _doc(**over: object) -> ReferenceDocument:
    fields: dict[str, object] = {
        "document_id": "ref_a",
        "text": TEXT,
        "source_url": "https://example.org/a",
        "licence": "CC-BY-SA-4.0",
        "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "redistribution": "permitted_with_attribution",
        "attribution": "Example Encyclopaedia, CC BY-SA 4.0",
        "accessed": date(2026, 9, 25),
        "snapshot_date": date(2026, 9, 20),
    }
    fields.update(over)
    return ReferenceDocument(**fields)  # type: ignore[arg-type]


def test_a_document_without_a_licence_cannot_be_constructed() -> None:
    """The index feeds a user-facing card; unrecorded terms stop here."""
    with pytest.raises(ValidationError):
        ReferenceDocument(  # type: ignore[call-arg]
            document_id="ref_a",
            text=TEXT,
            source_url="https://example.org/a",
            accessed=date(2026, 9, 25),
            snapshot_date=date(2026, 9, 20),
        )


def test_attribution_is_required_when_the_licence_says_so() -> None:
    """A CC BY-SA chunk shown without its attribution breaches its own licence."""
    with pytest.raises(ValidationError, match="breach"):
        _doc(attribution=None)


def test_attribution_is_optional_when_the_licence_does_not_require_it() -> None:
    assert _doc(redistribution="permitted", attribution=None).attribution is None


def test_an_unknown_redistribution_value_is_refused() -> None:
    with pytest.raises(ValidationError, match="redistribution must be one of"):
        _doc(redistribution="probably fine")


def test_an_unexpected_field_is_refused() -> None:
    """extra=forbid, so a typo'd rights field is an error not a silent default."""
    with pytest.raises(ValidationError):
        _doc(licence_note="see terms")


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, str | None]]] = []

    def add(self, document_id: str, chunks: list[object], **columns: str | None) -> int:
        self.calls.append((document_id, len(chunks), columns))
        return len(chunks)


class _Source:
    corpus_id = "test_reference"

    def __init__(self, documents: list[ReferenceDocument]) -> None:
        self._documents = documents

    def documents(self) -> Iterator[ReferenceDocument]:
        return iter(self._documents)


def test_a_document_is_chunked_and_indexed_with_its_rights() -> None:
    store = _Store()
    report = build_index(_Source([_doc()]), store)  # type: ignore[arg-type]
    assert report.documents_indexed == 1
    assert report.chunks_indexed > 0
    _, _, columns = store.calls[0]
    assert columns["licence"] == "CC-BY-SA-4.0"
    assert columns["attribution"] == "Example Encyclopaedia, CC BY-SA 4.0"


def test_a_document_that_may_not_be_redistributed_never_enters_the_index() -> None:
    """A card quotes the passage it retrieved, which is republication."""
    store = _Store()
    report = build_index(_Source([_doc(redistribution="prohibited")]), store)  # type: ignore[arg-type]
    assert store.calls == []
    assert report.documents_indexed == 0
    assert report.documents_skipped_not_redistributable == 1


def test_skipped_documents_are_counted_not_silently_dropped() -> None:
    """A build that skipped most of its corpus is a different build."""
    documents = [_doc(document_id="a"), _doc(document_id="b", redistribution="prohibited")]
    report = build_index(_Source(documents), _Store())  # type: ignore[arg-type]
    assert report.documents_seen == 2
    assert report.documents_indexed == 1
    assert report.documents_skipped_not_redistributable == 1


def test_licences_are_tallied_for_the_manifest_entry() -> None:
    documents = [
        _doc(document_id="a"),
        _doc(document_id="b"),
        _doc(document_id="c", licence="CC0-1.0", redistribution="permitted", attribution=None),
    ]
    report = build_index(_Source(documents), _Store())  # type: ignore[arg-type]
    assert report.licences == {"CC-BY-SA-4.0": 2, "CC0-1.0": 1}


def test_one_snapshot_date_is_reported_and_several_are_not() -> None:
    """DATA.md requires a snapshot date; several means the corpus reflects no
    single moment, and picking one would make the manifest entry a guess."""
    same = [_doc(document_id="a"), _doc(document_id="b")]
    assert build_index(_Source(same), _Store()).snapshot_date == date(2026, 9, 20)  # type: ignore[arg-type]

    mixed = [_doc(document_id="a"), _doc(document_id="b", snapshot_date=date(2026, 1, 1))]
    assert build_index(_Source(mixed), _Store()).snapshot_date is None  # type: ignore[arg-type]


def test_the_access_window_is_reported() -> None:
    documents = [
        _doc(document_id="a", accessed=date(2026, 9, 20)),
        _doc(document_id="b", accessed=date(2026, 9, 25)),
    ]
    report = build_index(_Source(documents), _Store())  # type: ignore[arg-type]
    assert report.accessed_earliest == date(2026, 9, 20)
    assert report.accessed_latest == date(2026, 9, 25)


def test_a_document_with_no_chunkable_text_indexes_nothing() -> None:
    store = _Store()
    report = build_index(_Source([_doc(text="   ")]), store)  # type: ignore[arg-type]
    assert report.documents_indexed == 0
    assert store.calls == []


def test_an_explicit_document_list_overrides_the_source() -> None:
    """So a caller can resume or re-index a subset."""
    store = _Store()
    report = build_index(_Source([]), store, documents=[_doc()])  # type: ignore[arg-type]
    assert report.documents_indexed == 1


class _StubEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(DIMENSIONS, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), DIMENSIONS), dtype=np.float32)


def test_the_store_requires_a_licence_rather_than_defaulting_it() -> None:
    """A nullable licence column would make an unattributable chunk reachable."""
    from pipeline.retrieval.store import PgVectorStore

    class _Cursor:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, *a: object, **k: object) -> None:
            return None

        def executemany(self, *a: object, **k: object) -> None:
            return None

    class _Connection:
        def cursor(self):  # noqa: ANN202
            return _Cursor()

        def commit(self) -> None:
            return None

    from pipeline.retrieval.chunk import Chunk

    store = object.__new__(PgVectorStore)
    store.connection = _Connection()  # type: ignore[attr-defined]
    store.embedder = _StubEmbedder()  # type: ignore[attr-defined]
    with pytest.raises(KeyError, match="licence"):
        store.add("doc_a", [Chunk("a sentence.", 0, 11, 0)])
