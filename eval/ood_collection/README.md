# OOD collection — procedure

**Status: plan only. No implementation exists in this directory.**

This document specifies the collection of the out of distribution suite before
any of it is built, for the same reason EVALUATION.md is written before the
models: a procedure designed after the data is visible drifts toward whatever
the data happens to contain, and the drift is invisible to the person causing
it. Everything below is derived from `context/DATA.md` and `context/EVALUATION.md`.
Where this document and those disagree, those win and this is the bug.

The OOD suite is the most important dataset in the project. It is the one that
produces the honest number.

---

## 1. What is being collected

400 to 600 articles published after the training snapshot date.

If collection runs long, reduce toward 300 articles. Do **not** reduce
stratification to hit the count. The stratification is what makes the analysis
interpretable; a larger unstratified set answers a question nobody asked
(ROADMAP.md, risk register).

Both the article text and a per article record of the fields in
`data/MANIFEST.yaml`. The suite gets one manifest entry with `role: ood_suite`.

## 2. Stratification

Two axes, sampled independently so that neither collapses into the other.

**Publisher type.** Six strata:

| Stratum | Description |
| --- | --- |
| `wire` | Wire services and their direct syndication |
| `national` | National outlets with a masthead and a corrections policy |
| `regional` | Regional and local outlets |
| `partisan` | Openly partisan outlets, sampled across the spectrum, not from one side |
| `aggregator` | Aggregators and republishers |
| `low_quality` | Sites with known quality problems, per published listings, recorded with source |

`partisan` is sampled across the spectrum deliberately. A partisan stratum drawn
from one side reintroduces publisher identity as a confound, which is the exact
failure this project exists to document.

`low_quality` membership is recorded as an external observation with its source
and date, in the same shape the source registry uses (D007). It is a note about
where the article came from, never a label on the article.

**Topic.** Four strata: `politics`, `health`, `science`, `business`.

Sampled so that no single news cycle dominates any cell. In practice this means
capping articles per event cluster and spreading collection across weeks rather
than pulling a fortnight's output at once.

**Cell targets.** Aim for even coverage across the 6 × 4 grid. Record actual
counts per cell in the manifest. Cells that cannot be filled are reported as
underfilled, not silently backfilled from an adjacent cell.

## 3. Timing

**Snapshot date: 2026-01-01.** Fixed on 2026-08-31, before any collection, and
recorded here rather than inferred afterwards from whatever was collected. It
sits comfortably after the ISOT style corpus was assembled, so no article in
this suite can have been seen during the leakage work, and it leaves a wide
collection window.

Every article in the suite is published **after** that date. Collection
completes and the set is frozen **before the first training run**.

## 4. Collection procedure

1. Record the snapshot date and the cell targets. Commit them.
2. For each publisher, check `robots.txt` and honour it. A disallowed path is
   not fetched, and the exclusion is recorded rather than worked around.
3. Fetch through the SSRF guarded client from `context/SECURITY.md` control 1.
   The OOD collector is not exempt from it: these are attacker influenced URLs
   like any other, and a separate unguarded fetch path is how the guarded one
   stops being the only one.
4. Conservative crawl delay, well under anything that would read as scraping
   pressure. A descriptive user agent with a contact URL (SECURITY.md control 5).
   Per domain limits apply to collection as they do to serving.
5. Extract and normalise per section 5, using one pipeline for every stratum.
6. Record the per article manifest fields, including the extraction pipeline
   version, at the time of extraction rather than reconstructed later.
7. Discard raw HTML after extraction. It is not retained.

## 5. Normalisation rules

These are the rules that stop the OOD suite reproducing the leakage it exists
to measure. Every one of them is applied identically to every stratum.

**One pipeline.** Both halves, all strata, the same extractor at the same
version. Different extraction across strata is precisely the artefact that makes
the ISOT style corpus separable, and it is trivially reintroduced by collecting
one stratum through a different tool because it was more convenient.

**Unicode.** NFKC normalisation. Zero width characters stripped. Bidirectional
override characters removed. Confusable characters mapped. Whitespace collapsed.
The same normalisation the serving path applies (SECURITY.md controls 2 and 4),
so evaluation input matches production input.

**Publisher identity stripped.** Before any model sees the text:

- Datelines, including the leading location and wire service prefix
- Bylines and author blocks
- Boilerplate: subscription prompts, related links, share widgets, footers
- Copyright lines and outlet names in running text where they are structural
  rather than part of a claim

Stripping is applied uniformly. A stratum whose boilerplate is harder to remove
does not get less stripping; it gets more work or it gets excluded, and the
exclusion is recorded.

**No systematic length difference between strata.** Check the length
distribution per stratum after normalisation. If strata differ systematically,
resample. Length was one of the four ablations for a reason.

**The residual check.** After all of the above, train a classifier to predict
the stratum from the text. If it succeeds, the remaining signal is worth
investigating and gets written up. If it fails, that is the expected result and
it is also worth stating. Either outcome is a finding; neither is a reason to
adjust the normalisation until the number comes out preferred.

## 6. Labels

The OOD suite carries **no veracity ground truth**, and this is not a gap to be
filled later.

- Where a claim has been assessed by an established fact checking organisation,
  the assessment is recorded with its source and date. It is used as a reference
  point in the writeup only. It is never a training target and never a
  correctness criterion for any component.
- The labels that are annotated are the answerable ones: whether a retrieved
  passage genuinely bears on a claim, and the stance of that passage toward it.
  Both are answerable without adjudicating truth.
- The 200 passage hand annotated extraction set is drawn from this suite,
  annotated against a written guideline that is committed before annotation
  begins (DATA.md, EVALUATION.md section 3).

## 7. Splits

Grouped by publishing domain. Never random. Every domain appears in exactly one
split, because a domain spread across splits leaks its house style between them,
which is a smaller version of the leakage this project is documenting.

Temporal as a second axis where dates exist: train earlier, evaluate later.

Three way: `train`, `dev`, `frozen_holdout`. All iteration — every threshold
sweep, every prompt revision, every architecture comparison — happens against
`dev`. The frozen holdout is inspected exactly twice in the whole project, at
the midpoint and at the end, and the midpoint inspection is recorded so the
final number can be read in context (D009). The manifest carries the inspection
dates and the list must never exceed two entries.

## 8. Freezing

The set is frozen once collection and normalisation are complete:

- `record_count` and `sha256` written to the manifest
- `frozen: true` and `frozen_at` written
- Per cell counts written, including underfilled cells
- The extraction pipeline git SHA written

After freezing, content does not change. A correction to a genuinely broken
article is a new manifest revision with the change logged, not an edit.

## 9. Storage and personal data

Article text in this corpus **does** persist, which makes it the exception to
the project wide rule that article text is never stored, and it is therefore
the part that carries the GDPR obligations (SECURITY.md control 6).

- Encrypted storage, EU region, access limited to the project
- Lawful basis: legitimate interest for research, with a written balancing test
  committed alongside the corpus
- Not redistributed, recorded as `redistribution: prohibited` in the manifest
- Retention and deletion recorded, with deletion implemented and tested rather
  than documented and assumed

## 10. Done when

- 400 to 600 articles (or a deliberate reduction to 300 with stratification
  intact), collected after the snapshot date
- Every 6 × 4 cell reported with an actual count, underfilled cells named
- One extraction pipeline, one normalisation, applied identically, verified by
  the residual stratum classifier check with its result written down
- Publisher markers stripped and spot verified by hand on a sample per stratum
- Grouped domain splits with no domain in two splits, verified by assertion
- Manifest entry complete, `frozen: true`, hash recorded
- The whole thing complete **before the first training run**

## 11. Not in this directory

No collector, no fetcher, no normaliser, no annotation tooling. Implementation
follows once this plan is agreed. The SSRF guarded client it depends on is M1
work, and building a second unguarded fetch path here to avoid waiting for it
would defeat the control.
