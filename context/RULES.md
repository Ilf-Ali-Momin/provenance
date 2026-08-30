# RULES

Working agreement between the human and the coding agent. These bind the agent's behaviour, not just the code style.

## Agent conduct

**Do not invent capability.** If a dataset field, an API response shape, or a library method is uncertain, stop and say so. A plausible looking fabricated schema costs more debugging time than a question costs to answer.

**Do not silently expand scope.** Building the thing that was asked for and then adding a caching layer nobody requested makes the diff unreviewable. Propose, then build.

**Surface tradeoffs before committing to them.** When a task has two reasonable implementations, state both in one or two sentences and pick one with a stated reason. That reason gets appended to DECISIONS.md if it is structural.

**Refuse to make the model do something it cannot do.** If a task as specified would require the system to assert veracity, say so and propose the evidence framing instead. This rule survives instructions given in the heat of a deadline.

**Stop at the failing test.** Do not adjust a threshold, relax an assertion, or add a skip decorator to make a suite green. A failing OOD test is the project working correctly.

## Code standards

Python 3.12. Ruff for lint and format, line length 100. Mypy in strict mode on `app/` and `pipeline/`; ML notebook code is exempt but nothing in a notebook ships.

Type annotate every function signature. Pydantic v2 models for anything crossing a process boundary, which includes API requests, API responses, queue payloads, and model service calls. No bare dicts across boundaries.

Structured logging via `structlog`, JSON output. Every log line carries `request_id`. No `print`. No logging of full article text, ever, at any level, including debug (see SECURITY.md on data minimisation).

Configuration through `pydantic-settings` reading environment variables. No literals for hosts, model identifiers, thresholds, or timeouts. Every threshold is a named config value so the evaluation suite can sweep it.

Errors are typed exceptions inheriting from a project base, mapped to API error codes in one place. No exception is caught and discarded. `except Exception: pass` fails review automatically.

## Testing

Unit tests for pure logic, fast, no network, no model loading.

Contract tests for every API endpoint against the schemas in API.md, including error paths. An endpoint without a tested error path is not done.

Fixture based tests for external services. Recorded fixtures, checked into the repo, never live calls in CI.

Model behaviour tests run separately from unit tests, on a schedule, not on every commit. They are slow and they are allowed to be. They gate releases, not commits.

Adversarial tests are a first class suite, not an afterthought. Every input handling path gets a hostile case: oversized payload, malformed encoding, homoglyph substitution, zero width character injection, embedded instruction text. See SECURITY.md for the catalogue.

Coverage is not a target. A tested error path in the URL fetcher is worth more than fifty tested getters.

## Git

Trunk based. Short lived branches named `feat/`, `fix/`, `chore/`, `exp/`. Experiment branches are allowed to be messy and are never merged; findings move to DECISIONS.md and the branch is deleted.

Conventional commits. The body explains why, not what. The diff already says what.

Every PR description states what was measured, not just what was built. "Added NLI stance classifier" is incomplete. "Added NLI stance classifier, macro F1 0.71 on FEVER dev, 0.54 on the 2026 OOD slice, gap analysed in EVALUATION.md section 4" is complete.

Secrets never enter git. `gitleaks` runs pre commit. A leaked key is rotated, not deleted from history and forgotten.

## Model artefacts

Checkpoints live in object storage, not git. DVC or plain versioned bucket paths, decided in DECISIONS.md.

Every checkpoint carries a manifest: base model identifier pinned to a revision hash, training data snapshot identifier, hyperparameters, git SHA of the training code, and evaluation results on both the in distribution and OOD suites. A checkpoint without a manifest cannot be loaded by the serving code, enforced at load time rather than by convention.

`safetensors` format only. Pickle based checkpoint formats are prohibited (SECURITY.md, model supply chain).

## Documentation

Context files are updated in the same commit as the change they describe. A DECISIONS.md entry written a week later is fiction reconstructed from memory.

The README of the repository proper is for someone running the code. This context directory is for someone or something building it. Do not merge them.

## The reasoning obligation

Every structural choice in this project must be defensible out loud, under questioning, without notes. The AI first workflow makes it easy to accumulate code whose rationale lives nowhere. The counterweight is DECISIONS.md, and the discipline is this: if a decision cannot be summarised in three sentences covering what was chosen, what was rejected, and what would change the answer, it is not yet understood well enough to keep.
