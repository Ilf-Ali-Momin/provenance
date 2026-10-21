# ROADMAP

**Current state is in `context/STATE.md`** — what is built, what is blocked and
on whom, and where to start. This file is the plan; that one is the position.

Sequenced so that every milestone produces something defensible on its own. If the project stops after any milestone, what exists is complete rather than half built.

The ordering is deliberate and it is not the ordering that feels natural. Evaluation infrastructure comes before models. The security boundary comes before the feature that needs it. Both inversions exist because the alternative order produces work that has to be redone.

---

## M0: Evaluation before models

**Why first.** This milestone produces the project's actual thesis. It is also the only milestone that is genuinely novel relative to every other project with this title, and completing it means that even a stalled project has a real contribution.

Build the OOD collection pipeline. Collect and freeze 400 to 600 articles per DATA.md, stratified by publisher type and topic. Strip publisher identifying markers.

Reproduce the standard approach on the ISOT style corpus. Report the accuracy it claims.

Run all four ablations from EVALUATION.md section 2.

Write `docs/leakage-analysis.md`, around 800 words with a results table.

**Done when.** The analysis document exists with real measured numbers, including any that contradict the expected result. The OOD set is frozen with a manifest.

**Defensible on its own as.** An empirical critique of a widely used benchmark, which is a more interesting artefact than most completed versions of this project.

---

## M1: Security boundary and skeleton

**Why here.** The SSRF guarded fetcher is the highest severity control in the system and building it after the feature means building the feature twice.

Flask application skeleton with configuration, structured logging with the redaction filter, error taxonomy, health endpoints.

Postgres with pgvector, Redis, RQ, Docker Compose, all containers non root with resource limits.

The SSRF guarded HTTP client, complete, with every control in SECURITY.md control 1 implemented and every corresponding adversarial test passing including DNS rebinding and redirect chains.

Article extraction and sanitisation, including zero width stripping, NFKC normalisation, and hidden text removal.

Rate limiting including per target domain limits.

CI: ruff, mypy strict, pytest, pip-audit, gitleaks, unpinned model reference check.

**Done when.** A URL submitted through the queue is fetched safely, extracted, and stored as a hash and text in memory, with the adversarial SSRF suite green.

---

## M2: Claim extraction

Prompted extraction over the Groq path, schema constrained, versioned prompts with hashes in output.

Offset validation dropping any claim whose span does not resolve.

The injection containment layer: random delimiters, sanitisation, injection detection logging.

The 200 passage annotation set built by hand against a written guideline.

The 30+ payload injection suite, run, escape rate measured and recorded.

Rule based fallback extractor.

**Done when.** Extraction meets its EVALUATION.md acceptance thresholds on the annotation set, and the injection suite reports a measured escape rate from the output schema, which should be zero by construction.

**Status: built, not validated. Parked 2026-09-25.** Five of six deliverables are done and the second done-when clause is met: the injection suite ran to completion, 33 payloads, 33 trials, escape rate 0.0, perturbation 1.0, injected span quoted 0.4848, detector recall 0.4848.

The first clause is unmet and no number is reported for it. The 200 passage annotation set is sampled but unannotated, and nothing in this project may supply those labels (D035) — an extractor scored against labels it generated measures itself. So extraction has **no measured precision or recall**, and any later document that needs one must say so rather than quote the fallback's behaviour as though it had been validated.

What is in place for whoever picks this up: the written guideline (`docs/annotation-guideline.md`), the sampler, the IoU scorer, the CLI, and an annotation tool (`tools/annotate.py`). The scorer refuses an unannotated file by design and reports the denominator when partly annotated, so the set can be done in pieces. D042 records the decision to park rather than close.

---

## M3: Retrieval and stance

Reference corpus built, chunked, embedded, indexed in pgvector with HNSW.

Hybrid dense and lexical retrieval with the weighting swept against dev.

Live news retrieval behind the same interface.

NLI stance classifier exported to ONNX, int8 quantised, served in the network isolated model container. Quantisation accuracy delta measured.

Stance evaluated on FEVER dev, MultiFC, and the OOD slice, with the four named stress subsets each reported separately.

**Done when.** The three stance numbers are recorded together with per class breakdowns, and the attributed claims subset is reported honestly whatever it shows.

**Status: substantially built, one of three stance numbers measured.** FEVER dev macro F1 0.8319 on the served fp32 configuration (D052). MultiFC is not built and will not be — its labels are veracity verdicts rather than stance toward evidence (D051), so the three-numbers-together requirement cannot be met as written. The OOD slice needs hand annotated stance labels. Retrieval has its metrics, sweep and pooling but no judged set, so nDCG and recall are unmeasured.

**Defensible on its own as.** A working evidence retrieval system with a measured distribution shift analysis. This is the point at which the project is fully presentable.

---

## M4: Source registry and circulation

Registry schema with per field provenance, public suffix list domain resolution, append only history.

RDAP lookups, structured extraction for masthead and corrections policy, all with recorded observation dates.

Re verification scheduler with change logging.

Bluesky and Reddit samplers with in memory aggregation and no identifier persistence, verified by a test asserting nothing identifying reaches storage.

Account age distribution, near duplicate ratio, time clustering.

**Done when.** A domain lookup returns dated observations with sources, and the circulation sampler passes the no persistence test.

**Status: started.** Registry and circulation schemas (D054), RDAP lookups verified against live servers (D055), about-page observations verified against a live publisher (D056), re-verification scheduler and change log verified on a live two-pass run (D057). The no-persistence test passes. Remaining: the Bluesky and Reddit samplers.

---

## M5: Composition and calibration

Composer with per branch deadlines and partial card assembly.

Temperature scaling per component, fitted on OOD dev, with reliability diagrams for both distributions.

Abstention thresholds derived from OOD reliability curves to hit the precision floor, with the resulting abstention rate recorded as an outcome.

Source diversity minimum enforced before any stance aggregate is displayed.

Full API surface per API.md including SSE streaming.

**Done when.** ECE meets the acceptance threshold on OOD after scaling, and abstention rates are documented per component.

---

## M6: Clients

Manifest V3 extension with `activeTab` only, explicit user invocation, isolated content script, strict CSP, pinned endpoint.

Card interface rendering all three analyser states distinguishably, with the rhetoric note verbatim.

Minimal web client for the demonstration path.

Interface language audit for implicature, not just literal accuracy.

**Done when.** The extension passes its own security checklist and the card correctly renders a partial result with two analysers abstained.

---

## M7: Adversarial and human evaluation

The full adversarial suite as a gating target.

Evasion measured per technique against the rhetoric model, results published.

Human evaluation, twenty cards, six evaluators, three questions, with the midpoint run already completed during M5.

Poisoning analysis written up including the attacks the system does not defend against.

**Done when.** Every adversarial result is recorded including failures, and the human evaluation's misleading rate is reported.

---

## M8: Writeup

`docs/leakage-analysis.md` finalised.

`docs/evaluation.md` with every component number, in distribution and OOD, with calibration plots and abstention rates.

`docs/security.md` as a public threat model with the measured injection and evasion rates.

`docs/limitations.md`, written properly rather than as a disclaimer paragraph. What the system cannot do, where it fails, what an attacker with resources defeats, and what it would take to fix each.

README with an honest results summary and no headline accuracy figure.

**Done when.** A reader can determine from the documentation alone whether to trust any given output of the system.

---

## Risk register

**The OOD collection is more work than expected.** Most likely schedule risk. Mitigation: it is M0, so the cost surfaces before anything is built on top of it. If it runs long, reduce to 300 articles rather than reducing stratification, because the stratification is what makes the analysis interpretable.

**Stance classification performs poorly on real news passages.** Likely, particularly on attributed claims. Not a failure. Measure it, report it, and analyse why. A documented negative result on a hard subproblem is a stronger artefact than an undocumented positive one.

**Weak supervision for rhetoric produces a weak model.** Likely. Mitigation: rule precision is measured first, and if it is too low to support a useful ceiling, labels are dropped or the whole component is cut. Cutting a component with a written reason is a legitimate outcome and should be recorded in DECISIONS.md.

**The LLM extraction path costs more than budgeted.** Mitigation: circuit breaker and daily token budget from M2, caching by content hash, fallback extractor already built.

**Scope creep toward a verdict.** The persistent risk, because a verdict demonstrates better. Mitigation: the schema has no field for it, D001 is marked load bearing, and RULES.md instructs the agent to refuse the task.

**Trying to do all eight milestones.** M0 through M3 is a complete, defensible project. M4 onward is genuinely optional. Deciding to stop at M3 and doing M0 through M3 thoroughly beats reaching M6 with an unmeasured pipeline, and that decision should be made deliberately at the M3 boundary rather than drifted into.
