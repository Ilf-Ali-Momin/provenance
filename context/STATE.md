# STATE

Where the project actually is, for whoever picks it up next. Regenerate the
numbers rather than trusting them: this file is written by hand and drifts, and
`docs/m0-report.html` sections 9 and 10 are the generated ones.

Last written 2026-09-29, at commit `2a24f16`. 84 commits, 770 tests passing,
57 decisions in `context/DECISIONS.md`.

Read `context/BRIEF.md` and `context/RULES.md` first. This file says where the
work stopped, not what the project is.

## The three constraints that outrank everything

1. **The system never asserts veracity** (D001). If a task as specified needs it
   to, stop and say so rather than building it.
2. **Nothing in this project may supply its own ground truth** (D035). An
   extractor scored against labels it generated measures the generator, and
   every number downstream inherits that.
3. **The frozen holdout is inspected twice in the whole project** (D009). Sweeps
   run against dev. `eval/retrieval/sweep.py` and
   `pipeline/retrieval/select.py` both refuse the holdout by name.

## Milestones

**M0, evaluation before models — closed.** The leakage analysis is the project's
actual contribution and stands alone. Corpus frozen at 266 documents, 41
publishers, 9 months. `docs/m0-report.html` is the writeup; its sections 9 and
10 regenerate with `python -m tools.refresh_report`, the rest is written prose
and must not be generated.

**M1, security boundary and skeleton — closed.** SSRF-guarded fetcher, redaction
keyed on value length, RQ queue whose result type cannot hold article text,
sliding-window rate limits, containers on an internal network.

**M2, claim extraction — built, not validated (D042).** Five of six deliverables
done. The injection suite is complete: 33 payloads, 33 trials, escape rate 0.0,
perturbation 1.0, detector recall 0.4848. Extraction has **no measured precision
or recall** and no document may imply otherwise. It closes when somebody
annotates `eval/annotation/passages.jsonl`, which is at 0 of 200.

**M3, retrieval and stance — substantially built, one number measured.** Chunker,
BM25, hybrid fusion, bge embedder under ONNX, pgvector store, corpus builder,
Wikimedia sources, page selection, weight sweep, stance classifier. FEVER dev
measured at macro F1 **0.8319** on the fp32 configuration now served (D052).
MultiFC is not built and will not be: its labels are veracity verdicts, not
stance toward evidence (D051).

**M4, source registry and circulation — started.** Registry and circulation
schemas (D054), RDAP lookups (D055), about-page observations (D056). Remaining:
the re-verification scheduler with change logging, and the Bluesky and Reddit
samplers feeding `pipeline/circulation/aggregate.py`.

**M5 onwards — not started.**

## What blocks what

Three things need a person, and none of them can be generated.

**The 200 annotations.** Gates M2's acceptance number. Tooling is done:
`tools/annotate.py` serves them, `docs/annotation-brief.pdf` briefs an
annotator, the scorer refuses an unannotated file and reports its denominator on
a partial one. A 50-passage subset gives a real number with wide intervals and
is a legitimate stopping point.

**Retrieval relevance judgements.** Gates M3's nDCG and recall.
`eval/retrieval/pool.py` bounds the work to roughly 20 to 40 candidates per
claim rather than the whole index.

**OOD stance labels.** Gates the third of EVALUATION.md's three stance slices,
and the gap between FEVER and OOD is the headline result for that component.

Two smaller blockers:

**`FETCH_USER_AGENT` needs a contact that can receive mail** before a live
reference-corpus build. `pipeline/retrieval/sources.py` refuses placeholder
domains, and Wikimedia enforces it server-side as well (D047).

**No git remote**, so CI has never executed. The workflow exists and is
untested.

## Open rows

- `mypy tests/` reports 66 errors; tests are not in mypy's `files` (D025).
- Static or quantisation-aware int8 for the stance model would likely recover
  most of the 0.0583 macro F1 that dynamic int8 costs (D052). Unmeasured.
- The near-duplicate threshold in `CirculationSettings` is unswept and is the
  kind that is dangerous to guess (D054). Its known false positive — people
  quoting the same headline cluster — is asserted as a test.
- An intermittent `recursive_mutex` abort from ONNX Runtime at interpreter
  shutdown, after pytest prints its summary. Seen twice under load, exit code
  unaffected both times. Read the summary rather than only the exit code.

## Where to start

Finish M4. The re-verification scheduler is the next piece and needs nothing
from anyone: it re-reads observations on a cadence and logs what changed, which
is what makes the registry's append-only history worth having.

The samplers after that. Bluesky's public API needs no credentials; Reddit's
does, so check what is available before designing around it.

## What this project keeps getting wrong

Recorded because it has happened repeatedly and the pattern is worth knowing
before repeating it a seventh time.

Every significant defect so far was found by **running the thing against real
data**, not by review, and several were in code written to measure something
else. A fixture written to match the patterns it tests proves nothing
(D056). A smoke test that discriminates where the real measurement does not is
worse than no smoke test (D050). A generator whose output changes every run
cannot be compared against itself, and that property is worth checking before
its content (D053).
