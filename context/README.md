# Provenance

Working name. Rename freely. "Provenance" describes what the system actually does: it tells you where a claim came from and what other sources say about it. It does not tell you whether the claim is true.

## What these files are

The design documents for the project, written before the code they describe and kept in step with it. They are working documents rather than a tidy final writeup: BRIEF, RULES and ARCHITECTURE set the shape of the whole thing, and the rest are read when a task touches them.

The decision log these files refer to is kept on local storage and is not published here, so a reference to a decision number in a docstring or a document will not resolve in this repository.

## Read order

| File | Read when | Purpose |
| --- | --- | --- |
| BRIEF.md | Always | What is being built, for whom, and the boundaries |
| RULES.md | Always | How code gets written, reviewed, and committed |
| ARCHITECTURE.md | Always | Component map, data flow, service boundaries |
| DATA.md | Any dataset or training task | Sources, licences, leakage analysis, splits |
| EVALUATION.md | Any model or metric task | Eval protocol, OOD suite, acceptance thresholds |
| SECURITY.md | Any task touching network, input, storage, or models | Threat model and required controls |
| API.md | Any endpoint or client task | Contracts, error shapes, versioning |
| ROADMAP.md | Session planning | Milestones and definitions of done |

## Non negotiables

Three rules override everything else in this pack. They exist because breaking any of them turns the project from a defensible piece of work into a demo that collapses under a single question.

1. **The system never outputs a truth verdict.** No "fake" label, no truth score, no percentage of veracity. The output is evidence and provenance. This is a product decision, an ethical decision, and a technical decision at once, and it is not up for renegotiation mid build.
2. **Evaluation is written before the model.** The out of distribution suite in EVALUATION.md exists before the first training run. A number produced without a matching OOD number is not reportable.
3. **Every fetched byte is hostile.** Article text, social posts, and model checkpoints are all attacker controlled inputs. SECURITY.md governs how they are handled and there are no convenience exceptions.

