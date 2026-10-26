# Provenance

Evidence surfacing for news claims. It tells you where a claim came from and
what other reporting says about it. It does not tell you whether the claim is
true.

## Why not a fake news classifier

The usual version of this project trains a classifier on a public fake news
dataset and reports accuracy in the high nineties. That number is an artefact
of how the dataset was built. In the most widely used corpus, the true articles
are mostly Reuters wire copy with a consistent dateline, and the false articles
come from a separate scrape with different punctuation and different average
length. The classifier learns publisher formatting. Point it at an article from
this week and it falls towards chance.

The measurement is in `docs/leakage-analysis.md` and the full writeup is
`docs/m0-report.html`.

There is also a conceptual problem. Truth is not a property of a text string.
Two articles can be identical in style and differ entirely in truth value,
which depends on facts about the world that no amount of surface text modelling
can reach. So this system answers four questions that are actually answerable
from data:

- **What is being claimed?** Extract discrete checkable claims, separating
  assertions from opinion, prediction and reported speech.
- **Who else is reporting it?** Retrieve related passages and classify each as
  supporting, refuting or neutral toward the claim.
- **Where did it come from?** Resolve the domain against a registry of dated,
  sourced observations — registration age, registrar, whether the homepage
  links a masthead or a corrections policy.
- **How is it moving?** Account age spread, near duplicate ratio and time
  clustering on open social platforms.

Every observation records where it came from and when it was read, so a reader
can follow the link and disagree.

## Status

Nine milestones. Two closed, one parked, one in progress, five untouched.

| Milestone | State |
| --- | --- |
| M0 evaluation before models | Closed. Corpus frozen at 266 documents, 41 publishers. |
| M1 security boundary | Closed. SSRF-guarded fetcher, redaction, rate limits, container isolation. |
| M2 claim extraction | Built, not validated. Injection suite passes: 33 payloads, escape rate 0.0. Precision and recall need hand annotation that is still in progress. |
| M3 retrieval and stance | Built. FEVER dev macro F1 0.8319. Two of three evaluation slices still unmeasured. |
| M4 source registry and circulation | In progress. Registry, RDAP, about-page observations and the re-verification scheduler are done. Social samplers remain. |
| M5–M8 | Not started. |

788 tests pass. Nothing here reports a number it has not measured, and the
places where a number is missing say so rather than filling it in.

## Running it

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env     # then fill in the values
.venv/bin/python -m pytest -q
```

The services:

```bash
docker compose up
```

Postgres with pgvector, Redis, the Flask API under gunicorn, and RQ workers on
an internal network. `docker-compose.dev.yml` exposes the databases for local
work; the production compose file does not.

## Layout

| Path | What is in it |
| --- | --- |
| `app/` | Flask API, request limits, error mapping, job queue |
| `pipeline/` | Fetching, text extraction, claim extraction, retrieval, stance, registry, circulation |
| `eval/` | Evaluation suites: leakage, injection, extraction, retrieval, stance |
| `tests/` | Unit, contract, adversarial and fixture-based tests |
| `docs/` | Reports and written analysis |
| `context/` | Design documents: brief, architecture, data, evaluation, security, API |
| `data/MANIFEST.yaml` | Every corpus with its licence, redistribution terms and storage |
| `tools/` | Local scripts, including the annotation server |

## What is not in this repository

The evaluation corpus and the annotation set are not here. `data/MANIFEST.yaml`
records the OOD suite as publisher copyright, not licensed for redistribution,
so the articles and the 200 annotated passages stay on local encrypted storage.
The manifest, the collection scripts and the evaluation code are all here, so
the measurements are reproducible by anyone who collects their own corpus — the
collector in `eval/ood_collection/` honours robots.txt and records refusals.

Model checkpoints are not here either. They live in object storage with a
manifest recording the base model revision, the training data snapshot, the
hyperparameters and the evaluation results on both distributions. Serving code
refuses a checkpoint without one.

## Scope

In scope: English text, the four questions above, rhetoric flags as a clearly
labelled secondary signal, calibration and abstention, a Flask API, a browser
extension, a minimal web client, and a published out-of-distribution evaluation
with a written analysis of where the system fails.

Out of scope: image, video and audio verification; firehose ingestion at scale;
any automated moderation or takedown action; user accounts beyond an API key
for rate limiting; and any claim of completeness. Absence of retrieved evidence
is reported as absence of retrieved evidence, never as evidence of absence.

## Licence

Code is unlicensed for now. Corpus terms are per source in
`data/MANIFEST.yaml`.
