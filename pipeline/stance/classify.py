"""NLI stance classification, under ONNX Runtime.

ARCHITECTURE.md: "`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` exported to
ONNX for CPU inference. Premise is the retrieved passage, hypothesis is the
extracted claim, output is entailment, contradiction, or neutral." And: "Both
the NLI and rhetoric models are base sized and quantised to int8."

**The label order is read from the model, never assumed.** This model's
`id2label` is `{0: entailment, 1: neutral, 2: contradiction}`. Other NLI
checkpoints, including widely used ones, order it `{0: contradiction, 1:
neutral, 2: entailment}` - exactly reversed on the two labels that matter. A
hardcoded order produces a classifier that is confidently backwards: it reports
refutation where there is support, the argmax is high, nothing raises, and the
only symptom is a stance number that is bad in a way that looks like the task
being hard. So the mapping comes from the exported manifest, and a manifest
without it is refused.

**Premise and hypothesis are not interchangeable.** NLI is directional: "the
minister said the deficit halved" entails "a minister spoke about the deficit",
not the reverse. The retrieved passage is the premise and the claim is the
hypothesis, which is the direction the question "does this evidence bear on this
claim" actually asks. `classify` takes them as named arguments for that reason.

**What this does not do.** It returns a stance toward a passage, not a verdict
about the world. A claim contradicted by one retrieved passage is a claim one
source disagrees with, and D001 puts the aggregation of those into a truth
judgement outside this system entirely.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from numpy.typing import NDArray

MODEL_ID: Final = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# RULES.md: pinned to a revision hash. tools/check_model_pins.py enforces this
# at the call site; the value is resolved against the repository, not guessed.
MODEL_REVISION: Final = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
MAX_TOKENS: Final = 512
# See the note at the call site: parallel workers, not one big session.
DEFAULT_THREADS: Final = 1
# Sub-batch size for length bucketing. See `logits`: padding, not token count,
# is what per-pair cost tracks.
DEFAULT_BATCH: Final = 16

# The project's vocabulary, which is about evidence rather than inference. The
# mapping from the model's own labels is applied once, here.
SUPPORT: Final = "support"
REFUTE: Final = "refute"
NEUTRAL: Final = "neutral"
NLI_TO_STANCE: Final = {
    "entailment": SUPPORT,
    "contradiction": REFUTE,
    "neutral": NEUTRAL,
}


class StanceUnavailableError(RuntimeError):
    """The classifier could not be loaded or used."""


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted, dtype=np.float32)
    out: NDArray[np.float32] = np.asarray(
        exponentiated / exponentiated.sum(axis=-1, keepdims=True), dtype=np.float32
    )
    return out


class OnnxStanceClassifier:
    """DeBERTa MNLI under ONNX Runtime, CPU.

    Takes a directory holding `model.onnx` (or `model.int8.onnx`),
    `tokenizer.json` and `model.json`, produced by `tools/export_stance.py`.
    """

    def __init__(
        self,
        model_dir: Path,
        quantised: bool = True,
        max_tokens: int = MAX_TOKENS,
        threads: int = DEFAULT_THREADS,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        manifest_path = model_dir / "model.json"
        if not manifest_path.exists():
            msg = (
                f"no manifest at {manifest_path}. The label order cannot be assumed: this "
                "checkpoint is entailment/neutral/contradiction and others are the "
                "reverse, which produces a classifier that reports refutation where there "
                "is support and never raises."
            )
            raise StanceUnavailableError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        id2label = manifest.get("id2label")
        if not id2label:
            msg = f"{manifest_path} carries no id2label; refusing to guess the label order"
            raise StanceUnavailableError(msg)
        # Keys arrive as strings through JSON.
        self.labels = [str(id2label[str(i)]).lower() for i in range(len(id2label))]
        unknown = [label for label in self.labels if label not in NLI_TO_STANCE]
        if unknown:
            msg = f"{manifest_path} names labels this project has no stance for: {unknown}"
            raise StanceUnavailableError(msg)

        name = "model.int8.onnx" if quantised else "model.onnx"
        path = model_dir / name
        if not path.exists():
            raise StanceUnavailableError(f"no {name} in {model_dir}")

        self.max_tokens = max_tokens
        self.batch_size = max(1, batch_size)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        # One by default, because several of these run in parallel worker
        # containers and a session that spawns a thread per core in each of them
        # oversubscribes the box: the threads contend, every request slows, and
        # total throughput falls. Raise it only where one worker owns the
        # machine, and only against a measurement - tools/bench_stance.py is
        # that measurement.
        options.intra_op_num_threads = threads
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def logits(self, premises: list[str], hypotheses: list[str]) -> NDArray[np.float32]:
        """Raw model outputs, in the model's own label order.

        Sub-batched by token length. The tokenizer pads every sequence in a call
        to the longest one in it, and the model then computes over the padding,
        so one long premise makes every pair batched with it cost what it costs.
        Measured on FEVER dev: a 64-pair batch padded to 219 tokens against a
        mean of 48.9 real ones - 78% of the work discarded - at 520 ms per pair,
        where 16 pairs padded to 83 ran at 142. Grouping similar lengths is
        worth 1.5 to 1.6x on both precisions and changes no output, since each
        pair still sees exactly its own tokens.
        """
        if len(premises) != len(hypotheses):
            msg = f"{len(premises)} premises against {len(hypotheses)} hypotheses"
            raise StanceUnavailableError(msg)
        if not premises:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        encoded = self.tokenizer.encode_batch(list(zip(premises, hypotheses, strict=True)))
        if len(premises) > self.batch_size:
            order = sorted(range(len(encoded)), key=lambda i: sum(encoded[i].attention_mask))
            out = np.zeros((len(premises), len(self.labels)), dtype=np.float32)
            for start in range(0, len(order), self.batch_size):
                window = order[start : start + self.batch_size]
                # Re-encoded per window so padding is that window's longest and
                # not the whole call's; encode_batch is cheap beside inference.
                out[window] = self._forward(
                    [premises[i] for i in window], [hypotheses[i] for i in window]
                )
            return out
        return self._forward(premises, hypotheses)

    def _forward(self, premises: list[str], hypotheses: list[str]) -> NDArray[np.float32]:
        """One session call. Pair encoding, so the model sees
        [CLS] premise [SEP] hypothesis [SEP] and truncation removes the
        premise's tail rather than the claim."""
        encoded = self.tokenizer.encode_batch(list(zip(premises, hypotheses, strict=True)))
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        out: NDArray[np.float32] = np.asarray(
            self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0],
            dtype=np.float32,
        )
        return out

    def classify(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[tuple[str, dict[str, float]]]:
        """Stance of each passage toward its claim, with the full distribution.

        The distribution is returned alongside the label because an abstention
        threshold is a calibration decision (EVALUATION.md) and a caller that
        only has the argmax cannot make one.
        """
        probabilities = softmax(self.logits(premises, hypotheses))
        results: list[tuple[str, dict[str, float]]] = []
        for row in probabilities:
            scores = {
                NLI_TO_STANCE[label]: float(row[index]) for index, label in enumerate(self.labels)
            }
            results.append((max(scores, key=lambda k: scores[k]), scores))
        return results
