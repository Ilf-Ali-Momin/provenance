"""FEVER dev as stance pairs.

EVALUATION.md: "Macro F1 across the three classes on FEVER dev, then on
MultiFC, then on the OOD slice with hand annotated stance labels."

**Why FEVER is a legitimate stance set and MultiFC is not.** FEVER's labels are
about a claim *against a specific evidence sentence*: SUPPORTS means this
evidence supports it, REFUTES means this evidence contradicts it. That is
exactly what the NLI classifier outputs, so the mapping is a rename.

MultiFC's labels are veracity verdicts from fact checking organisations - "true",
"mostly false", "pants on fire" and dozens more, varying per site. Scoring the
stance classifier against them would mean treating "is this claim true" as "does
this passage support this claim", which conflates veracity with stance and is the
one thing D001 puts outside this system. That is recorded in D051 rather than
worked around; this module covers FEVER only.

**The NOT ENOUGH INFO caveat, which travels with the number.** FEVER's NEI
claims have no gold evidence by construction - nobody annotated evidence for a
claim nothing bears on. This distribution supplies *retrieved* evidence for
those rows. So the neutral class measures the classifier against passages a
retriever thought were relevant and a human never confirmed were, which is
closer to deployment than gold evidence is and is not the same measurement as
the other two classes. `retrieved_evidence` marks those rows so the per class
number can be read with it in view.

    PYTHONPATH=. .venv/bin/python -m eval.stance.fever --download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from eval.stance.score import NEUTRAL, REFUTE, SUPPORT, Judgement

# DATA.md: corpora that cannot be checked in are referenced by download script.
DATASET: Final = "copenlu/fever_gold_evidence"
# No REVISION constant. An earlier draft carried one that was the real commit's
# first ten characters padded out to forty with plausible hex - a value that
# looks pinned, fails nothing locally, and resolves to a different snapshot than
# it claims. `download` returns the commit the server actually served, and the
# manifest records that.
DEV_FILE: Final = "valid.jsonl"
LOCAL: Final = Path("data/stance/fever_dev.jsonl")

LABELS: Final = {
    "SUPPORTS": SUPPORT,
    "REFUTES": REFUTE,
    "NOT ENOUGH INFO": NEUTRAL,
}


def download(out: Path = LOCAL, revision: str = "main") -> tuple[int, str]:
    """Fetch FEVER dev and return (rows, resolved commit).

    The commit is returned rather than assumed so the manifest records what was
    actually downloaded. A dataset id alone resolves to whatever the branch
    holds today, which is the same defect `tools/check_model_pins.py` exists to
    catch for models.
    """
    import httpx

    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{DEV_FILE}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=120.0,
        headers={"User-Agent": "provenance-eval/0.1"},
    ) as response:
        response.raise_for_status()
        commit = response.headers.get("x-repo-commit", "unknown")
        with out.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    rows = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows, commit


def evidence_text(evidence: object) -> str:
    """Join an evidence set into one premise.

    FEVER evidence is a list of [page, sentence_id, sentence]. Several sentences
    can be needed together, and they are joined in the order given rather than
    reordered: NLI reads a premise as prose, and shuffling it changes what it
    entails.
    """
    if not isinstance(evidence, list):
        return ""
    sentences: list[str] = []
    for item in evidence:
        if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
            sentences.append(item[2].strip())
    return " ".join(s for s in sentences if s)


def pairs(path: Path = LOCAL, limit: int | None = None) -> list[dict[str, object]]:
    """(premise, hypothesis, gold) rows, with the caveat flag attached."""
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold = LABELS.get(str(row.get("label")))
        premise = evidence_text(row.get("evidence"))
        claim = str(row.get("claim") or "").strip()
        if gold is None or not premise or not claim:
            continue
        out.append(
            {
                "id": str(row.get("id")),
                # Premise is the evidence, hypothesis is the claim. NLI is
                # directional and this is the direction the question asks.
                "premise": premise,
                "hypothesis": claim,
                "gold": gold,
                # NEI has no gold evidence by construction; this row's premise
                # was retrieved, not annotated.
                "retrieved_evidence": str(row.get("verifiable")) == "NOT VERIFIABLE",
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def judgements(rows: list[dict[str, object]], predicted: list[str]) -> list[Judgement]:
    """Pair gold with prediction, tagging the subsets that can be derived here.

    Only `attributed` is derivable from FEVER text without further annotation,
    and even that is a surface cue rather than a judgement. The other three
    stress subsets EVALUATION.md names are properties of the claim/passage
    relationship that FEVER does not label, so they are measured on the OOD
    slice rather than guessed at here.
    """
    reported = (" said ", " says ", " told ", "according to", " claimed ", " argued ")
    out: list[Judgement] = []
    for row, prediction in zip(rows, predicted, strict=True):
        premise = str(row["premise"]).lower()
        subsets = ("attributed",) if any(cue in premise for cue in reported) else ()
        out.append(Judgement(gold=str(row["gold"]), predicted=prediction, subsets=subsets))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, default=LOCAL)
    args = parser.parse_args()
    if not args.download:
        raise SystemExit("pass --download; this module loads a corpus, it does not make one")
    rows, commit = download(args.out)
    sys.stdout.write(
        f"{rows} rows to {args.out}\n"
        f"dataset {DATASET} at commit {commit}\n"
        "Record that commit in data/MANIFEST.yaml; a dataset id alone resolves to "
        "whatever the branch holds today.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
