"""Score an extractor against the hand annotated set (EVALUATION.md §3).

The harness, so that the moment annotations exist the number comes out and
nobody has to decide how to compute it with results already visible. The
matching rule is fixed in `score.py` and was committed before any annotation
existed (D036).

Two extractors can be scored through the same path: the prompted one, and the
rule based fallback. D037 deferred ARCHITECTURE.md's dependency-parse fallback
until there was a measured gap to justify the dependency, and this is what
measures it.

EVALUATION.md also requires prompt sensitivity: "The same evaluation is run
against every prompt revision and the results are tabulated, because prompt
changes are code changes and untracked prompt drift is a real source of silent
regression." Results are therefore keyed by extractor version, and running a
new prompt adds a row rather than replacing one.

    PYTHONPATH=. .venv/bin/python -m eval.extraction.run_eval --live
    PYTHONPATH=. .venv/bin/python -m eval.extraction.run_eval --fallback
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final

from pipeline.claims.config import ExtractorSettings
from pipeline.claims.extract import ExtractorUnavailableError, extract_claims
from pipeline.claims.fallback import VERSION as FALLBACK_VERSION
from pipeline.claims.fallback import extract_claims_offline
from pipeline.claims.prompt import load_prompt

from eval.extraction.score import Span, score

logger = logging.getLogger(__name__)

ANNOTATIONS: Final = Path("eval/annotation/passages.jsonl")
OUT: Final = Path("eval/extraction/results.json")


class NotAnnotatedError(RuntimeError):
    """The passages exist but carry no human spans."""


def load_annotations(path: Path = ANNOTATIONS) -> list[dict[str, object]]:
    """Read the annotation file, refusing to score an unannotated one.

    A file of empty `spans` would score as perfect precision on zero claims and
    zero recall, and someone reading the table later would see numbers rather
    than an absence. Refusing is the only honest behaviour: nothing in this
    project may supply the gold standard (D035).

    Annotated-ness is read from `reviewed`, not from `spans`. Selecting rows
    whose `spans` are non-empty drops exactly the passages the guideline calls
    a data point - "a passage with zero spans is a data point, and the
    extractor is scored on returning zero there too" - so an extractor that
    invented claims in empty commentary was scored only on the passages where
    claims existed, and its false positives there never counted.
    """
    if not path.exists():
        msg = f"no annotation file at {path}; run eval.annotation.sample first"
        raise NotAnnotatedError(msg)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    # `reviewed` is authoritative where it exists. A file written before the
    # field existed is read the old way rather than reported as unannotated.
    if any("reviewed" in r for r in rows):
        annotated = [r for r in rows if r.get("reviewed")]
    else:
        annotated = [r for r in rows if r.get("spans")]
    if not annotated:
        msg = (
            f"{path} holds {len(rows)} passages and none marked reviewed. It has "
            "not been annotated yet, and an unannotated set scores as zero recall "
            "rather than as an absence. See docs/annotation-guideline.md."
        )
        raise NotAnnotatedError(msg)
    if len(annotated) < len(rows):
        logger.warning(
            "scoring %d of %d passages; the rest are not marked reviewed",
            len(annotated),
            len(rows),
        )
    return annotated


def _gold(row: dict[str, object]) -> list[Span]:
    spans = row.get("spans")
    if not isinstance(spans, list):
        return []
    return [Span(int(s["start"]), int(s["end"]), str(s["type"])) for s in spans]


def evaluate(
    rows: list[dict[str, object]],
    use_fallback: bool = False,
    settings: ExtractorSettings | None = None,
) -> dict[str, object]:
    settings = settings or ExtractorSettings()
    version = FALLBACK_VERSION if use_fallback else load_prompt(settings.prompt_name)[1]

    per_passage: list[tuple[list[Span], list[Span]]] = []
    failures = 0
    for row in rows:
        text = str(row["text"])
        if use_fallback:
            predicted = extract_claims_offline(text)
        else:
            try:
                predicted, _ = extract_claims(text, settings)
            except ExtractorUnavailableError as exc:
                # Counted, not skipped. A passage the extractor could not
                # process is a passage where it found nothing, and dropping it
                # would quietly raise recall on the rest.
                logger.info("extractor unavailable on one passage: %s", exc)
                failures += 1
                predicted = []
        per_passage.append((_gold(row), [Span(c.start, c.end, c.type.value) for c in predicted]))

    result = score(per_passage)
    result["extractor_version"] = version
    result["extractor_failures"] = failures
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the prompted extractor")
    parser.add_argument("--fallback", action="store_true", help="score the rule based extractor")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not (args.live or args.fallback):
        raise SystemExit("pass --live or --fallback; neither alone measures anything")

    rows = load_annotations(args.annotations)
    result = evaluate(rows, use_fallback=args.fallback)

    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[str(result["extractor_version"])] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    logger.info("%s", json.dumps({k: v for k, v in result.items() if k != "confusion"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
