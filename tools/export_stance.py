"""Export the NLI stance classifier to ONNX, quantise it, and measure the cost.

    PYTHONPATH=. .venv/bin/python -m tools.export_stance --out models/deberta-nli

EVALUATION.md names this as a deliverable: "NLI stance classifier exported to
ONNX, int8 quantised, served in the network isolated model container.
Quantisation accuracy delta measured."

**The delta is reported, not gated, and that difference is the point.** The
embedder's export check is a hard gate (D045) because an embedder that disagrees
with itself has no usable interpretation - there is no such thing as an
acceptably different vector. Quantisation is not that. It is a deliberate
trade: int8 is roughly four times smaller and several times faster on CPU, and
it costs some accuracy. A gate would hide the size of the trade behind a
pass/fail; a number lets the reader decide whether the trade was worth it, and
lets a later reader see it change.

So this writes both models and records, on a probe set:

* `agreement`: share of probes where int8 picks the same label as fp32
* `max_abs_probability_delta`: the largest single probability shift
* `mean_abs_probability_delta`

**The label order is exported, never assumed.** `id2label` is copied from the
model's own config into the manifest. This checkpoint is
entailment/neutral/contradiction; other NLI checkpoints reverse the two labels
that matter, and a hardcoded order produces a classifier that reports refutation
where there is support with high confidence and no error.

**The probe set is directional on purpose.** It includes pairs where premise and
hypothesis swapped would flip the answer, because getting that backwards is the
other silent failure available here.

**Per-channel quantisation is not an optimisation here, it is the difference
between a classifier and a coin.** Measured on the probe set, 2026-09-25:

    all ops, per-tensor        agreement 0.333  max prob delta 0.970  244MB
    all ops, per-channel       agreement 1.000  max prob delta 0.206  244MB
    MatMul only, per-tensor    agreement 0.333  max prob delta 0.961  539MB
    MatMul only, per-channel   agreement 1.000  max prob delta 0.037  540MB
    MatMul only, reduce_range  agreement 0.333  max prob delta 0.962  539MB

Per-tensor scales collapse this model onto `neutral` for every input - 0.333 is
three classes and chance. DeBERTa-v3's disentangled attention puts wide dynamic
range in single tensors, and one scale per tensor cannot hold it. Nothing errors;
the model returns a confident distribution that is the same distribution every
time, and the only symptom is a macro F1 near the majority class, which reads as
a hard task rather than a broken export.

All ops is kept over MatMul only: three times smaller against a probability
delta of 0.206 rather than 0.037, and ARCHITECTURE.md's stated reason for int8
is size and CPU latency. That delta is large enough to matter for calibration,
so EVALUATION.md's calibration work measures the quantised model, not this one.

**Six probes is a smoke test, not the measurement.** Agreement 1.000 over six
pairs is weak evidence and is recorded as such. The reportable delta comes from
running both models over FEVER dev.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
from pipeline.stance.classify import (
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    OnnxStanceClassifier,
    softmax,
)

# (premise, hypothesis, expected stance). Expectations are the obvious reading,
# used to report whether the export is behaving sanely at all - not as an
# accuracy measurement, which is EVALUATION.md's job against FEVER and MultiFC.
PROBES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "Unemployment fell to 4.1% in March, the statistics office said.",
        "Unemployment fell in March.",
        "support",
    ),
    (
        "Unemployment rose to 4.8% in March, the statistics office said.",
        "Unemployment fell in March.",
        "refute",
    ),
    (
        "The bridge carries 40,000 vehicles a day.",
        "Unemployment fell in March.",
        "neutral",
    ),
    (
        "The minister said the deficit had halved over the year.",
        "A minister commented on the deficit.",
        "support",
    ),
    # Directional: the reverse of the pair above does not hold, and a premise /
    # hypothesis swap is a silent failure this catches.
    (
        "A minister commented on the deficit.",
        "The minister said the deficit had halved over the year.",
        "neutral",
    ),
    (
        "Rates will rise again before June, analysts said.",
        "Analysts expect rates to fall.",
        "refute",
    ),
)
OPSET: Final = 14
# Not a default worth inheriting: see the module docstring. Per-tensor scales
# put this model at chance.
PER_CHANNEL: Final = True


def export(out: Path, revision: str = MODEL_REVISION) -> dict[str, object]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    # `dtype` is explicit and load bearing. transformers 5 loads a checkpoint in
    # the dtype it was saved in, and this one is fp16: the first export produced
    # a model whose 202 initializers were all FLOAT16 while every label on it
    # said fp32. Two consequences, neither of which announces itself. The
    # "fp32 versus int8" delta below would have compared int8 against fp16 and
    # been reported as the cost of quantisation. And quantize_dynamic derives
    # its scale dtype from the weights, so it emitted FLOAT16 scales, which
    # DequantizeLinear rejects - the only reason this was noticed at all.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float32
    ).eval()
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    premises = [p for p, _, _ in PROBES]
    hypotheses = [h for _, h, _ in PROBES]
    batch = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in batch]

    # dynamo=False keeps the legacy exporter, which needs no onnxscript. The
    # embedder's export was abandoned for that dependency (D045); here the
    # model is not published as ONNX, so exporting is the only route and the
    # legacy path is the one that avoids adding a package.
    torch.onnx.export(
        model,
        tuple(batch[n] for n in names),
        str(out / "model.onnx"),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in [*names, "logits"]},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    quantize_dynamic(
        out / "model.onnx",
        out / "model.int8.onnx",
        weight_type=QuantType.QInt8,
        per_channel=PER_CHANNEL,
    )

    id2label = {str(k): str(v) for k, v in model.config.id2label.items()}
    (out / "model.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": revision,
                "id2label": id2label,
                "max_tokens": MAX_TOKENS,
                "opset": OPSET,
                "exported_dtype": "float32",
                "quantisation": (
                    f"dynamic int8, weights only, per_channel={PER_CHANNEL}. Per-tensor "
                    "scales put this checkpoint at chance; see tools/export_stance.py."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with torch.no_grad():
        reference = softmax(model(**batch).logits.to(torch.float32).numpy())
    full = softmax(OnnxStanceClassifier(out, quantised=False).logits(premises, hypotheses))
    quantised = softmax(OnnxStanceClassifier(out, quantised=True).logits(premises, hypotheses))

    measured = {
        "probes": len(PROBES),
        "export_max_abs_delta_vs_torch": float(np.abs(reference - full).max()),
        "quantisation": {
            "per_channel": PER_CHANNEL,
            "agreement": float((full.argmax(-1) == quantised.argmax(-1)).mean()),
            "max_abs_probability_delta": float(np.abs(full - quantised).max()),
            "mean_abs_probability_delta": float(np.abs(full - quantised).mean()),
            "caveat": (
                "Six probes. Agreement over six pairs is weak evidence and this is a "
                "smoke test; the reportable delta comes from FEVER dev."
            ),
        },
        "size_bytes": {
            "fp32": (out / "model.onnx").stat().st_size,
            "int8": (out / "model.int8.onnx").stat().st_size,
        },
    }

    labels = [id2label[str(i)].lower() for i in range(len(id2label))]
    from pipeline.stance.classify import NLI_TO_STANCE

    measured["probe_predictions"] = [
        {
            "premise": premise[:60],
            "hypothesis": hypothesis[:60],
            "expected": expected,
            "int8": NLI_TO_STANCE[labels[int(row.argmax())]],
        }
        for (premise, hypothesis, expected), row in zip(PROBES, quantised, strict=True)
    ]

    manifest = json.loads((out / "model.json").read_text(encoding="utf-8"))
    manifest["measured"] = measured
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/deberta-nli"))
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    measured = export(args.out, args.revision)
    sys.stdout.write(json.dumps(measured, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
