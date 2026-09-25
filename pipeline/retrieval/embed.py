"""Dense embeddings for the local index and for live retrieval.

ARCHITECTURE.md: the index is "built with `BAAI/bge-base-en-v1.5` at 768
dimensions, stored in pgvector with an HNSW index. Cosine distance."

**Why ONNX here and not torch.** The same document says "Models serve from ONNX
Runtime on CPU" and that this "removes the GPU from the deployment entirely".
The embedder is on the serving path, not only the indexing path - live
retrieval "embeds on the fly and scores the same way" - so it runs under the
same runtime as the other two models. torch stays in the optional `ml` group,
where the export script and the M0 ablations use it, and never enters the
serving container.

**Two details of bge that are not optional and are easy to get wrong.**

Pooling is CLS, not mean. bge-v1.5 is trained with the first token as the
sequence representation; mean pooling produces vectors that still look
plausible, still have 768 dimensions, and retrieve measurably worse. Nothing
downstream can detect the mistake.

Embeddings are L2 normalised, because the index uses cosine distance. With
normalised vectors cosine is the dot product, which is what pgvector's
`vector_cosine_ops` assumes it can rely on; unnormalised vectors make the HNSW
neighbour graph reflect magnitude as well as direction.

**The query prefix.** bge is trained with an asymmetric instruction: queries
carry "Represent this sentence for searching relevant passages: " and passages
carry nothing. Retrieval quality drops without it, so `embed_query` and
`embed_passages` are separate methods rather than one method with a flag - a
flag defaults to something, and the default would be silently wrong half the
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

MODEL_ID: Final = "BAAI/bge-base-en-v1.5"
# RULES.md: every model artefact carries a base model identifier pinned to a
# revision hash. tools/check_model_pins.py enforces this at the call site.
MODEL_REVISION: Final = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
DIMENSIONS: Final = 768
MAX_TOKENS: Final = 512
QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """What the index and the retriever need, without either owning a model."""

    def embed_query(self, text: str) -> NDArray[np.float32]: ...

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]: ...


def l2_normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Unit length rows. Zero rows stay zero rather than becoming NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit: NDArray[np.float32] = np.divide(
        vectors, norms, out=np.zeros_like(vectors), where=norms > 0
    )
    return unit


class OnnxEmbedder:
    """bge-base-en-v1.5 under ONNX Runtime, CPU.

    Takes a directory holding `model.onnx` and `tokenizer.json`, produced by
    `tools/export_embedder.py`. The directory rather than a hub id, because the
    serving container has no network and no transformers.
    """

    def __init__(self, model_dir: Path, max_tokens: int = MAX_TOKENS) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.max_tokens = max_tokens
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()
        # One thread by default: the worker runs several of these in parallel
        # containers, and letting each spawn a thread per core oversubscribes
        # the box and makes latency worse, not better.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        encoded = self.tokenizer.encode_batch(texts)
        feed: dict[str, NDArray[np.int64]] = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        # CLS pooling: the first token. See the module docstring - mean pooling
        # here produces vectors that look fine and retrieve worse.
        cls: NDArray[np.float32] = np.asarray(hidden[:, 0], dtype=np.float32)
        return l2_normalise(cls)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """One query vector. The bge instruction prefix is applied here."""
        vector: NDArray[np.float32] = self._encode([QUERY_PREFIX + text])[0]
        return vector

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        """Passage vectors. No prefix, by the model's asymmetric training."""
        return self._encode(texts)
