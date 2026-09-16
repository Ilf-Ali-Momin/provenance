# DATA

## The leakage problem, stated precisely

This section exists because it is the technical core of the project's honesty and because an agent given a training task will otherwise reach for the easiest corpus and produce a meaningless number.

The most commonly used fake news corpus pairs a set of true articles scraped from a single wire service with a set of false articles scraped from a heterogeneous collection of low quality sites. The two halves differ systematically in ways that have nothing to do with veracity.

Wire copy carries a consistent dateline prefix at the head of the body text. It uses a house style guide, which means consistent punctuation around quotations, consistent title casing, consistent attribution formatting. Its average length and paragraph structure follow editorial standards. The scrape itself introduced artefacts: different HTML extraction produced different whitespace and entity handling across the two halves.

A model trained on this learns the union of those artefacts. Ablations confirm it. Truncating the first sentence of every article drops accuracy substantially. Reducing every document to its punctuation and casing pattern, with all content words replaced by a single token, still yields high accuracy, which is decisive: content carries little of the signal.

This is not a criticism of the dataset's creators, who documented their collection method. It is a criticism of the thousands of projects that report the resulting accuracy as a capability measurement.

**Required deliverable.** The leakage ablation is run and written up before any other training work. It appears in EVALUATION.md section 2 and it is the first thing shown in any demonstration of this project. A build that skips it has skipped the point.

## Corpora

### For evidence retrieval

**FEVER.** Claims paired with Wikipedia evidence sentences and support, refute, or not enough info labels. The closest available analogue to the retrieve then classify pipeline. Licence permits research use with attribution. Its weakness is domain: Wikipedia prose is unlike news prose, so it trains the mechanism and overstates the performance.

**MultiFC.** Claims from many fact checking organisations with metadata and evidence pages. Heterogeneous and messy, which is closer to reality. Useful for measuring how much FEVER performance was Wikipedia specific.

**Reference corpus for the local index.** Built rather than downloaded, from openly licensed reference material and from a snapshot of published fact checks accessed through their APIs where terms permit. Snapshot date recorded. This is what pgvector actually holds.

### For claim extraction

**ClaimBuckets and CheckThat! lab data** where accessible, for checkworthiness classification. Primarily used to validate the LLM extractor rather than to train a model, since the extractor is a prompted LLM.

**A hand annotated set of 200 passages** drawn from the OOD collection below, annotated by you against a written guideline. Small, but it is the only extraction evaluation set that matches the actual input distribution, and building it forces the annotation guideline to be explicit about the assertion versus opinion boundary, which is the hard part of the task.

### For rhetoric

This is where honest sourcing is hardest and the agent must not paper over it.

There is no clean labelled corpus for unsourced attribution or manufactured urgency at span level. The plan is weak supervision: high precision pattern rules generate candidate labels over a large unlabelled news corpus, a stratified sample of those candidates is hand verified to measure rule precision, and the model trains on rule labels with the measured precision reported as a ceiling on trustworthy performance.

The limitation is stated in the output, not buried. A model trained on rule labels can at best learn the rules plus some generalisation, and claiming otherwise would be exactly the overclaiming this project is built to avoid.

### For leakage demonstration only

The ISOT style Kaggle corpus. Used solely to reproduce and document the failure. It is never used to train a shipped component. The agent must not, under any framing, use it as a training source for a production model.

## Out of distribution suite

The most important dataset in the project, because it is the one that produces the honest number.

**Composition.** 400 to 600 articles published after the training snapshot date, from a stratified sample of publisher types: wire services, national outlets, regional outlets, partisan outlets across the spectrum, aggregators, and known low quality sites. Topics spread across politics, health, science, and business so that a single news cycle does not dominate.

**Topic spread: measured, not assumed (D029).** This requirement went unverified for the whole of M0 because the freeze reads topic from the article URL path, which resolves for only 6.4% of the corpus. It was measured on `ood_suite_2026_v4` by zero shot entailment over the article text, and **it holds at the corpus level**: all four topics are present and the largest, politics, is 37.0% of the 81.2% of documents that get a named topic. It does not hold per cell, and the cells are reported rather than rebalanced — the wire stratum is 73% business, the partisan stratum 48% politics. Neither topic method is a ground truth: where both exist they agree on 12 of 17 documents. The inferred label never replaces the URL path label, which is the publisher's own filing decision.

**Labels.** Where a claim has been assessed by an established fact checking organisation, that assessment is recorded with its source and date, and used only as a reference point in the writeup, never as ground truth for training. For retrieval and stance evaluation the labels are annotation of whether retrieved evidence genuinely bears on the claim, which is answerable without adjudicating truth.

**Construction rules.** Both halves come from the same extraction pipeline. Formatting is normalised identically. No systematic length difference between strata. Publisher identity is stripped from the text before any model sees it, including datelines, bylines, and boilerplate. If a model can still separate strata after that, the remaining signal is worth investigating, and if it cannot, that is the expected result.

**Timing.** The OOD set is collected and frozen before the first training run, then never inspected during development beyond the frozen holdout protocol below.

## Splits

Grouped splits by publishing domain, not random splits. Random splitting across a corpus where one domain contributes many documents leaks the domain's style across train and test, which is a smaller version of the same leakage this project is documenting. Every domain appears in exactly one split.

Temporal split as a second axis where dates exist. Train on earlier, evaluate on later. News language drifts and a random temporal split hides that drift.

Three way: train, dev, frozen holdout. The frozen holdout is looked at twice in the whole project, once at the midpoint and once at the end. Every threshold sweep, every architecture comparison, every prompt iteration happens against dev. This is the discipline that prevents the slow leak of test set information through repeated peeking, and it is worth stating in the writeup because most projects at this level do not do it.

## Licensing and terms

Every corpus records licence, source URL, access date, and any redistribution restriction in `data/MANIFEST.yaml`. Corpora that cannot be redistributed are referenced by download script rather than checked in.

Publisher content fetched for the OOD set is stored locally for evaluation and not redistributed. Robots directives are respected on every fetch. Rate limits are conservative, well under anything that would look like scraping pressure.

The distinction between "this is publicly accessible" and "I am permitted to redistribute this" is one the agent must not blur, and the manifest is where that distinction is made explicit.

## Personal data

News text contains personal data about named individuals, which places this project under GDPR as a controller.

Article text is not persisted. Only content hashes, derived claims, and aggregate statistics reach storage. Social data is aggregated in memory with no identifier retained. The lawful basis for the evaluation corpus is legitimate interest for research, recorded with a documented balancing test, and the corpus lives on encrypted storage in an EU region with access limited to the project.

The source registry stores facts about publishing organisations, not about individuals. Where a masthead names a person, only the presence or absence of a masthead is recorded, not its contents. This boundary matters legally and it is not a detail to be optimised away for a richer feature set.
