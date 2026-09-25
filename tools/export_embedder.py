"""Fetch the bge ONNX artefact for the serving container, and verify it.

ARCHITECTURE.md puts serving on ONNX Runtime CPU. This script produces the
directory `pipeline/retrieval/embed.py` loads, and it runs offline: the serving
container then needs neither torch nor transformers nor any network.

    PYTHONPATH=. .venv/bin/python -m tools.export_embedder --out models/bge

**Fetched, not exported locally, and that is a deliberate reversal.** The first
version called `torch.onnx.export`. torch 2.13 routes that through `onnxscript`,
which is a third dependency for a step that runs once, and exporting a
transformers model with dynamic axes is fragile across torch versions in a way
that produces a *working* model with subtly different behaviour rather than an
error. `BAAI/bge-base-en-v1.5` ships `onnx/model.onnx` in the same repository at
the same pinned revision as the weights, so it is the same trust boundary and
the same artefact everyone else evaluates.

**What makes fetching safe is the verification, not the source.** A wrong
pooling, opset or input signature yields a model that runs, returns 768 floats
and retrieves worse, with nothing downstream able to tell. So this script loads
the torch weights at the same revision, embeds a probe set under both, and
refuses to write the manifest when the maximum absolute difference exceeds
tolerance. EVALUATION.md asks the same of the NLI model ("Quantisation accuracy
delta measured"); for the embedder it is a correctness gate rather than a
reported number, because an embedder that disagrees with itself has no usable
interpretation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Final

import numpy as np
from pipeline.retrieval.embed import (
    DIMENSIONS,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_PREFIX,
    OnnxEmbedder,
    l2_normalise,
)

# Deliberately includes the vocabulary the dense side is weakest on - figures,
# proper nouns, reported speech - since a pooling or opset mistake shows up
# there first.
PROBES: Final = (
    "Unemployment fell to 4.1% in March, the statistics office said.",
    "The minister said the deficit had halved over the year.",
    "Rates will rise again before June.",
    QUERY_PREFIX + "did unemployment fall in March",
)
TOLERANCE: Final = 1e-3
ONNX_PATH_IN_REPO: Final = "onnx/model.onnx"


def fetch(out: Path, revision: str = MODEL_REVISION) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    shutil.copyfile(
        hf_hub_download(MODEL_ID, ONNX_PATH_IN_REPO, revision=revision),
        out / "model.onnx",
    )

    # The gate. Same revision on both sides, so a difference here is the
    # artefact and not a version skew.
    model = AutoModel.from_pretrained(MODEL_ID, revision=revision).eval()
    batch = tokenizer(list(PROBES), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        reference = l2_normalise(model(**batch).last_hidden_state[:, 0].to(torch.float32).numpy())
    exported = OnnxEmbedder(out).embed_passages(list(PROBES))

    delta = float(np.abs(reference - exported).max())
    if delta > TOLERANCE:
        msg = (
            f"the ONNX artefact disagrees with the torch weights by {delta:.2e} "
            f"(tolerance {TOLERANCE:.0e}). A model that runs and returns the right shape "
            "while producing different vectors degrades retrieval silently, so no manifest "
            "is written and the directory is not usable."
        )
        raise SystemExit(msg)

    (out / "model.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": revision,
                "source": ONNX_PATH_IN_REPO,
                "dimensions": DIMENSIONS,
                "max_tokens": MAX_TOKENS,
                "pooling": "cls",
                "normalised": True,
                "max_abs_delta_vs_torch": delta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"{out} verified against torch, max abs delta {delta:.2e}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/bge"))
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    fetch(args.out, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
