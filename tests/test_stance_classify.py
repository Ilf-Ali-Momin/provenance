"""The stance classifier's contract, and the silent failure it refuses to allow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pipeline.stance.classify import (
    MODEL_ID,
    MODEL_REVISION,
    NLI_TO_STANCE,
    OnnxStanceClassifier,
    StanceUnavailableError,
    softmax,
)

MODEL_DIR = Path("models/deberta-nli")


def test_the_revision_is_pinned_to_something_sha_like() -> None:
    assert len(MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_ID == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def test_every_nli_label_maps_to_a_project_stance() -> None:
    assert set(NLI_TO_STANCE) == {"entailment", "contradiction", "neutral"}
    assert set(NLI_TO_STANCE.values()) == {"support", "refute", "neutral"}


def test_entailment_is_support_and_contradiction_is_refute() -> None:
    """Reversing these is the failure the manifest exists to prevent."""
    assert NLI_TO_STANCE["entailment"] == "support"
    assert NLI_TO_STANCE["contradiction"] == "refute"


def testsoftmax_rows_sum_to_one() -> None:
    out = softmax(np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]], dtype=np.float32))
    assert np.allclose(out.sum(axis=-1), 1.0)


def testsoftmax_is_stable_on_large_logits() -> None:
    """Without the max subtraction this overflows to nan."""
    out = softmax(np.array([[1000.0, 999.0, 998.0]], dtype=np.float32))
    assert not np.isnan(out).any()


def test_a_directory_without_a_manifest_is_refused(tmp_path: Path) -> None:
    """The label order cannot be assumed: this checkpoint is
    entailment/neutral/contradiction and others are the reverse."""
    with pytest.raises(StanceUnavailableError, match="label order cannot be assumed"):
        OnnxStanceClassifier(tmp_path)


def test_a_manifest_without_id2label_is_refused(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text(json.dumps({"model_id": "x"}), encoding="utf-8")
    with pytest.raises(StanceUnavailableError, match="refusing to guess"):
        OnnxStanceClassifier(tmp_path)


def test_a_manifest_naming_unknown_labels_is_refused(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text(
        json.dumps({"id2label": {"0": "yes", "1": "maybe", "2": "no"}}), encoding="utf-8"
    )
    with pytest.raises(StanceUnavailableError, match="no stance for"):
        OnnxStanceClassifier(tmp_path)


def test_a_missing_model_file_is_named(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text(
        json.dumps({"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}),
        encoding="utf-8",
    )
    with pytest.raises(StanceUnavailableError, match=r"model\.int8\.onnx"):
        OnnxStanceClassifier(tmp_path)


# --- against the exported model -------------------------------------------


def _skip_without_model() -> None:
    if not (MODEL_DIR / "model.int8.onnx").exists():
        pytest.skip("no exported model; run tools.export_stance")


@pytest.mark.model
def test_the_exported_label_order_is_the_checkpoints_own() -> None:
    _skip_without_model()
    manifest = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    assert manifest["id2label"] == {"0": "entailment", "1": "neutral", "2": "contradiction"}


@pytest.mark.model
def test_support_refute_and_neutral_are_told_apart() -> None:
    _skip_without_model()
    classifier = OnnxStanceClassifier(MODEL_DIR)
    premises = [
        "Unemployment fell to 4.1% in March, the statistics office said.",
        "Unemployment rose to 4.8% in March, the statistics office said.",
        "The bridge carries 40,000 vehicles a day.",
    ]
    hypothesis = "Unemployment fell in March."
    labels = [label for label, _ in classifier.classify(premises, [hypothesis] * 3)]
    assert labels == ["support", "refute", "neutral"]


@pytest.mark.model
def test_the_direction_of_the_pair_matters() -> None:
    """Premise and hypothesis are not interchangeable; swapping them is a
    silent failure."""
    _skip_without_model()
    classifier = OnnxStanceClassifier(MODEL_DIR)
    specific = "The minister said the deficit had halved over the year."
    general = "A minister commented on the deficit."
    forward, _ = classifier.classify([specific], [general])[0]
    backward, _ = classifier.classify([general], [specific])[0]
    assert forward == "support"
    assert backward != "support"


@pytest.mark.model
def test_the_distribution_is_returned_with_the_label() -> None:
    """An abstention threshold is a calibration decision and a caller with only
    the argmax cannot make one."""
    _skip_without_model()
    label, scores = OnnxStanceClassifier(MODEL_DIR).classify(["A cat sat."], ["A cat sat."])[0]
    assert set(scores) == {"support", "refute", "neutral"}
    assert abs(sum(scores.values()) - 1.0) < 1e-4
    assert scores[label] == max(scores.values())


@pytest.mark.model
def test_mismatched_batch_lengths_are_refused() -> None:
    _skip_without_model()
    with pytest.raises(StanceUnavailableError, match="against"):
        OnnxStanceClassifier(MODEL_DIR).classify(["a", "b"], ["c"])


@pytest.mark.model
def test_an_empty_batch_returns_nothing() -> None:
    _skip_without_model()
    assert OnnxStanceClassifier(MODEL_DIR).classify([], []) == []


@pytest.mark.model
def test_the_quantisation_delta_is_recorded() -> None:
    """EVALUATION.md asks for it measured; a gate would hide the size of the
    trade behind a pass/fail."""
    _skip_without_model()
    manifest = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    quantisation = manifest["measured"]["quantisation"]
    assert 0.0 <= quantisation["agreement"] <= 1.0
    assert quantisation["max_abs_probability_delta"] >= 0.0
    assert manifest["measured"]["size_bytes"]["int8"] < manifest["measured"]["size_bytes"]["fp32"]


@pytest.mark.model
def test_fp32_is_indifferent_to_what_else_is_in_the_batch() -> None:
    """Padding must not reach the answer, and under fp32 it does not.

    This is the control for the test below. Identical to 1.8e-07 across batch
    layouts means the attention mask is doing its job and the model computes
    over real tokens only.
    """
    _skip_without_model()
    corpus = Path("data/stance/fever_dev.jsonl")
    if not corpus.exists():
        pytest.skip("no FEVER dev; run eval.stance.fever --download")

    from eval.stance.fever import pairs

    rows = pairs(corpus, 40)
    premises = [str(r["premise"]) for r in rows]
    hypotheses = [str(r["hypothesis"]) for r in rows]

    one = OnnxStanceClassifier(MODEL_DIR, quantised=False, batch_size=len(rows))
    split = OnnxStanceClassifier(MODEL_DIR, quantised=False, batch_size=8)
    a = one.classify(premises, hypotheses)
    b = split.classify(premises, hypotheses)

    assert [label for label, _ in a] == [label for label, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        for stance in x:
            assert abs(x[stance] - y[stance]) < 1e-5


@pytest.mark.model
def test_int8_answers_depend_on_batch_composition() -> None:
    """A property of dynamic quantisation, recorded rather than assumed away.

    Dynamic int8 derives activation scales at run time from the values actually
    in the tensor, and padding changes those values. So the same pair can get a
    different answer depending on which other pairs it was batched with.
    Measured on 40 FEVER dev rows: 1 label in 40 flips between a single batch
    and batches of 8, with a maximum probability shift of 0.22, while fp32 over
    the same rows moves by 1.8e-07.

    Length bucketing does not cause this - batching by arrival order varies the
    padding too - but it does make it visible. The consequence is that an int8
    result is not reproducible across batch layouts, which the FEVER numbers in
    eval/stance/results.json inherit.
    """
    _skip_without_model()
    corpus = Path("data/stance/fever_dev.jsonl")
    if not corpus.exists():
        pytest.skip("no FEVER dev; run eval.stance.fever --download")

    from eval.stance.fever import pairs

    rows = pairs(corpus, 40)
    premises = [str(r["premise"]) for r in rows]
    hypotheses = [str(r["hypothesis"]) for r in rows]

    one = OnnxStanceClassifier(MODEL_DIR, batch_size=len(rows))
    split = OnnxStanceClassifier(MODEL_DIR, batch_size=8)
    a = [label for label, _ in one.classify(premises, hypotheses)]
    b = [label for label, _ in split.classify(premises, hypotheses)]

    disagreements = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    # Small, and not zero. Both halves are the finding.
    assert disagreements <= len(rows) // 10


@pytest.mark.model
def test_the_model_itself_is_deterministic() -> None:
    """Twice through the same session with the same batch gives the same answer,
    so the variation above is the batch layout and not the runtime."""
    _skip_without_model()
    classifier = OnnxStanceClassifier(MODEL_DIR, batch_size=16)
    premises = ["Unemployment fell to 4.1% in March."] * 4
    hypotheses = ["Unemployment fell in March."] * 4
    first = classifier.classify(premises, hypotheses)
    second = classifier.classify(premises, hypotheses)
    assert first == second


@pytest.mark.model
def test_bucketing_preserves_the_callers_order() -> None:
    """Results are sorted by length internally and must come back unsorted."""
    _skip_without_model()
    classifier = OnnxStanceClassifier(MODEL_DIR, batch_size=2)
    premises = [
        "Unemployment fell to 4.1% in March, the statistics office said.",
        "The bridge carries 40,000 vehicles a day. " * 12,
        "Unemployment rose to 4.8% in March.",
    ]
    hypothesis = "Unemployment fell in March."
    labels = [label for label, _ in classifier.classify(premises, [hypothesis] * 3)]
    assert labels[0] == "support"
    assert labels[2] == "refute"
