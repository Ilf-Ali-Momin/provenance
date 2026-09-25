"""The embedder's contract, and the two bge details that fail silently."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pipeline.retrieval.embed import (
    DIMENSIONS,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_PREFIX,
    Embedder,
    l2_normalise,
)
from pipeline.retrieval.hybrid import Searcher

MODEL_DIR = Path("models/bge")


def test_normalisation_makes_every_row_unit_length() -> None:
    """Cosine is the dot product only if the vectors are unit length."""
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(l2_normalise(vectors), axis=1), 1.0)


def test_a_zero_vector_stays_zero_rather_than_becoming_nan() -> None:
    """A NaN row poisons the HNSW graph at build time, not at query time."""
    out = l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(out).any()
    assert out.sum() == 0.0


def test_normalisation_preserves_direction() -> None:
    vectors = np.array([[3.0, 4.0]], dtype=np.float32)
    assert np.allclose(l2_normalise(vectors), [[0.6, 0.8]])


def test_the_revision_is_pinned_to_something_sha_like() -> None:
    """RULES.md: a model id alone resolves to whatever the branch holds today."""
    assert len(MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_ID == "BAAI/bge-base-en-v1.5"


def test_query_and_passage_are_separate_methods_not_a_flag() -> None:
    """A flag defaults to something, and the default is wrong half the time."""
    assert QUERY_PREFIX.endswith(": ")
    assert {"embed_query", "embed_passages"} <= set(dir(Embedder))


@pytest.mark.model
def test_the_exported_model_agrees_with_torch() -> None:
    """The export check, as a test. A wrong opset or pooling returns 768 floats
    that retrieve worse, and nothing downstream can tell."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not (MODEL_DIR / "model.onnx").exists():
        pytest.skip("no exported model; run tools.export_embedder")

    from pipeline.retrieval.embed import OnnxEmbedder

    probes = ["Unemployment fell to 4.1% in March.", "The minister said the deficit had halved."]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = transformers.AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
    with torch.no_grad():
        batch = tokenizer(probes, padding=True, truncation=True, return_tensors="pt")
        reference = l2_normalise(model(**batch).last_hidden_state[:, 0].to(torch.float32).numpy())

    exported = OnnxEmbedder(MODEL_DIR).embed_passages(probes)
    assert np.abs(reference - exported).max() < 1e-3


@pytest.mark.model
def test_the_exported_model_has_the_declared_shape_and_is_normalised() -> None:
    if not (MODEL_DIR / "model.onnx").exists():
        pytest.skip("no exported model; run tools.export_embedder")

    from pipeline.retrieval.embed import OnnxEmbedder

    embedder = OnnxEmbedder(MODEL_DIR)
    passages = embedder.embed_passages(["a sentence", "another sentence"])
    assert passages.shape == (2, DIMENSIONS)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-4)
    assert embedder.embed_query("a question").shape == (DIMENSIONS,)


@pytest.mark.model
def test_the_query_prefix_changes_the_vector() -> None:
    """Evidence the asymmetric instruction is actually being applied."""
    if not (MODEL_DIR / "model.onnx").exists():
        pytest.skip("no exported model; run tools.export_embedder")

    from pipeline.retrieval.embed import OnnxEmbedder

    embedder = OnnxEmbedder(MODEL_DIR)
    text = "did unemployment fall in March"
    assert not np.allclose(embedder.embed_query(text), embedder.embed_passages([text])[0])


class _StubEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(DIMENSIONS, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), DIMENSIONS), dtype=np.float32)


def test_a_stub_satisfies_the_embedder_protocol() -> None:
    """The harness must drive retrieval with no model loaded."""
    assert isinstance(_StubEmbedder(), Embedder)


class _StubSearcher:
    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        return [("a", 1.0)]


def test_a_stub_satisfies_the_searcher_protocol() -> None:
    """EVALUATION.md measures local, live and hybrid separately; each has to be
    passable on its own."""
    assert isinstance(_StubSearcher(), Searcher)
