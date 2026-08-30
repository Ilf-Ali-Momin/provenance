# EVALUATION

## Principle

Evaluation is written and implemented before the model it evaluates. This inverts the usual order deliberately. When the metric is designed after the results are visible, the metric drifts toward what the model happens to do well, and the drift is invisible even to the person doing it.

A number without a matching out of distribution number is not reportable, does not go in a README, does not go on a CV, and does not go in a PR description.

## Section 1: What is being measured

Not "is this article fake". That target is unmeasurable from text and the project does not claim it.

The measurable targets, each mapped to a component:

- Does claim extraction find the checkable assertions a human annotator would find, and does it correctly separate assertion from opinion?
- Does retrieval surface passages that genuinely bear on the claim?
- Does stance classification correctly identify whether a passage supports, refutes, or is neutral toward a claim?
- Do rhetoric flags fire on spans that a human agrees exhibit the flagged pattern?
- Does the system abstain when it should, and does calibration hold when the input distribution shifts?

Each of these is answerable. None requires adjudicating truth.

## Section 2: The leakage ablation

Run first, before any component training. This is a deliverable in its own right.

**Procedure.** Fine tune a standard transformer classifier on the ISOT style corpus with default settings, exactly as the common project does. Report the accuracy. Then run four ablations against the same held out split.

1. **First sentence removal.** Strip the leading sentence from every document, which removes the dateline prefix.
2. **Structure only.** Replace every content word with a single placeholder token, preserving punctuation, casing pattern, and length. Retrain.
3. **Length matched resample.** Resample both classes to identical length distributions. Retrain.
4. **Cross domain transfer.** Apply the unmodified model to the 2026 OOD slice.

**Expected shape of the result.** Ablations 1 through 3 each remove a large fraction of the accuracy despite removing no veracity relevant information. Ablation 4 degrades toward chance. Report the actual numbers, whatever they are, including the case where an ablation degrades less than expected, because that is a finding too.

**Deliverable.** A written analysis of roughly 800 words with a table, in `docs/leakage-analysis.md`. This is the artefact that demonstrates evaluation literacy, which is the scarce skill here, not the ability to call `Trainer.train()`.

## Section 3: Component metrics

### Claim extraction

Against the 200 passage hand annotated set. Precision and recall on assertion identification, with a match counted when predicted and gold spans overlap by more than half. Separately, confusion matrix on the assertion, opinion, prediction, attribution tagging.

Also measured: sensitivity to prompt version. The same evaluation is run against every prompt revision and the results are tabulated, because prompt changes are code changes and untracked prompt drift is a real source of silent regression.

Acceptance: assertion recall above 0.80 with precision above 0.70. Recall weighted higher because a missed claim means no evidence surfaces at all, while a spurious claim merely produces a low value card section.

### Retrieval

nDCG at 10 and recall at 20 against annotated relevance on the OOD claims. Measured separately for the local index and live retrieval, and for the hybrid combination, so the contribution of each is visible rather than assumed.

Also measured: retrieval failure rate, meaning the fraction of claims for which nothing above the relevance threshold is returned. This is expected to be substantial on genuinely novel claims and it is a feature, not a bug, provided the interface reports it honestly. Track it as a distribution, not a single rate, because a system that fails on 40 percent of health claims and 5 percent of political claims is a different system than one that fails on 20 percent of everything.

Acceptance: recall at 20 above 0.70 on claims that have coverage in the index, with the coverage rate reported alongside.

### Stance classification

Macro F1 across the three classes on FEVER dev, then on MultiFC, then on the OOD slice with hand annotated stance labels. The three numbers are always reported together, and the gap between the first and the third is the actual headline result of this component.

Per class breakdown is mandatory. Aggregate F1 hides the failure mode that matters, which is that refutation is rarer and harder than support and that neutral absorbs everything ambiguous. A model that never predicts refute can post a respectable macro F1 while being useless for the one output users most need.

Specific stress cases, each a named subset with its own number:

- Attributed claims, where the passage reports that someone said the claim rather than asserting it. This is the single largest error source in stance classification on real news text and it must be measured in isolation.
- Partial overlap, where the passage addresses part of a compound claim.
- Temporal mismatch, where the passage was true at the time of writing and the claim concerns a later period.
- Numeric claims, where the passage contains a different figure for the same quantity.

Acceptance: macro F1 above 0.60 on the OOD slice, with the attributed claims subset reported separately and no requirement that it pass, because the honest result there is likely poor and hiding it would be worse than reporting it.

### Rhetoric

Precision at the span level against human verification of a stratified sample, per label. Recall is not meaningfully estimable without exhaustive annotation and is not claimed.

The weak supervision rule precision, measured in DATA.md, is reported as the ceiling. If the rules are 0.78 precise and the model reaches 0.75, the model has learned the rules and generalised marginally, and that is what gets written down.

Acceptance: precision above 0.70 per label on the verified sample. Labels that cannot reach it are removed from the shipped model rather than shipped with a caveat.

### Calibration

Expected calibration error and reliability diagrams for every component that emits a confidence, on both in distribution and OOD data.

The core question is whether calibration survives distribution shift. It usually does not: models remain confident while becoming wrong, which is the worst possible failure mode for a system whose entire value proposition is honest uncertainty. Measuring the calibration gap under shift, and setting abstention thresholds from the OOD reliability curve rather than the in distribution one, is the mechanism that makes the abstention promise real.

Acceptance: ECE below 0.10 on OOD after temperature scaling fitted on OOD dev. Thresholds set so that shown outputs achieve at least 0.85 precision, with whatever abstention rate that implies. The abstention rate is an output of the requirement, never an input to it.

## Section 4: System level

**Latency.** p50 and p95 for the text selection path and the URL path, separately. Budget: p95 under 2.5 seconds for text selection with all analysers, since the extension is useless above that. The URL path is asynchronous and budgeted at p95 under 8 seconds end to end.

**Partial card rate.** Fraction of responses where at least one analyser missed its deadline or abstained, broken down by analyser. Expected to be high and that is acceptable; what is not acceptable is it being invisible.

**Adversarial robustness.** The SECURITY.md attack catalogue run as an evaluation suite. For each evasion technique, the change in analyser output. Paraphrase attacks against the rhetoric model are expected to succeed substantially and the measured evasion rate is reported, because a robustness claim without an attack measurement is marketing.

## Section 5: Human evaluation

Twenty cards, six evaluators, three questions each: does the card tell you something you did not know, does it lead you to a conclusion the evidence does not support, and would you trust it.

The middle question is the important one and it is the one automated metrics cannot reach. A card can be composed entirely of accurate components and still mislead through emphasis, ordering, or the implicature of showing a rhetoric flag next to a contradicting passage. That failure is only detectable by asking people what they concluded.

Run twice: once at the midpoint on an ugly interface, once at the end. The midpoint run is what makes the findings actionable.

## Section 6: What gets reported

The project's headline result is not an accuracy figure. It is a pair of statements.

First, the documented demonstration that the standard approach reports capability it does not have, with ablations quantifying the gap.

Second, per component measurements of the evidence pipeline on out of distribution data, with calibration under shift and explicit abstention rates, including the components that performed poorly.

A results section that contains no disappointing numbers is a results section that has been curated, and a reviewer who has done this work will read the absence correctly.
