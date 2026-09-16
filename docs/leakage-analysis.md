# Leakage analysis

*All four ablations run, ablation 4 in both the baseline and structure only
conditions. Ablation 4 is reported as a prediction distribution rather than an
accuracy, for the reason given below. Every number here is
measured, not estimated or carried over from published work.*

## The claim being tested

The standard project trains a classifier on the ISOT style corpus, reports
accuracy in the high nineties, and presents that as an ability to detect false
news. DATA.md asserts the figure is an artefact of assembly: the true half is
wire copy from one agency, the false half a heterogeneous scrape, differing
systematically in ways unrelated to veracity. If accuracy survives the removal
of information that cannot carry veracity, the classifier was never measuring
veracity.

## Corpus, and a provenance finding

Only mirrors were reachable. The one used is recorded in `data/MANIFEST.yaml`
with `training_use: prohibited` (D010), and holds 21,416 wire documents against
22,851 others, matching the canonical true count almost exactly.

**One caveat changes one result.** In the original distribution essentially
every wire article opens with a `CITY (Reuters) - ` dateline. In all three
mirrors examined it survives in **0.3%** of wire documents: they arrive
pre-stripped. Ablation 1 targets the dateline, so it removes a first sentence
whose most obvious tell someone else already removed, making it a lower bound.

## Method

TF-IDF over unigrams and bigrams, 50,000 features, logistic regression, random
splits of 36,000 train and 8,267 held out — the standard approach under
criticism. Chance is 50%; the held out set is 48.2% positive.

**Structure only** replaces every alphanumeric run with a shape token, so
`Trump` becomes `Xxxxx` and `2017` becomes `0000`: punctuation, casing and
length survive, every content word is destroyed. **Length matched** resamples
both classes to a common length distribution across 40 quantile bins, removing
length without removing a word.

The same conditions ran against `distilbert-base-uncased`, one epoch, batch 16,
learning rate 5e-5, defaults throughout, but **reduced in scale** — 6,000
train, 2,000 held out, 128 tokens — because external load starved a full size
run to roughly 36 seconds per step (D014). Its absolute values are not a full
scale reproduction; what matters is whether the columns agree in shape.

## Results

| Condition | Linear, full | DistilBERT, reduced | Information removed |
| --- | --- | --- | --- |
| Baseline | **0.9849** | **0.9920** | none |
| First sentence removed | 0.9756 (−0.9 pp) | 0.9710 (−2.1 pp) | leading sentence (dateline already absent) |
| Length matched | 0.9824 (−0.3 pp) | 0.9700 (−2.2 pp) | document length |
| Structure only | **0.8184 (−16.7 pp)** | **0.8350 (−15.7 pp)** | **every content word** |

## Reading the numbers

Structure only is decisive. With no nouns, verbs, names or topic — nothing a
human could read — accuracy is **81.8%** and **83.5%** against 50% chance.
Punctuation, capitalisation and length alone carry roughly two thirds of the
distance from chance to the headline figure. That two unrelated model families
land within 1.7 points of each other there matters more than either number: the
signal belongs to the corpus, not the classifier. A comma does not move when a
claim becomes false. The model separates two publishing pipelines — one house
style guide, one HTML scrape — by their formatting residue.

Length matching costs 0.3 and 2.2 points: real but minor. First sentence
removal costs 0.9 and 2.1, against pre-stripped text, so it measures what a
*second* removal would cost.

The surviving signal is distributed, not concentrated in one sentence. A single
leaky feature could be deleted; a stylistic fingerprint cannot. Hence D010
prohibiting this corpus as a training source under augmentation, pretraining or
auxiliary use: the leakage is co-extensive with the labels.

## Ablation 4: what the model says about 2026 news

EVALUATION.md expects accuracy degrading toward chance on later text. That
number cannot exist: accuracy needs veracity labels, and the out-of-distribution
suite deliberately has none (D001). What can be reported is what the model
*says*. The baseline model, unchanged, was applied to the frozen suite; the
table gives the share of each group assigned to the corpus's wire half, the
class that corpus treats as real.

These figures are measured on `ood_suite_2026_v4` (266 documents, 41 publishers,
nine months, sha `1adb5673`). They supersede figures measured on the first
freeze, which was 384 documents with 289 of them published on one day. Both
runs are kept in `eval/leakage/results/transfer.json`, keyed by corpus hash.

| Group | Documents | Baseline | Structure only |
| --- | --- | --- | --- |
| **Whole suite** | 266 | **22.2%** | **56.4%** |
| wire | 30 | 36.7% | 90.0% |
| regional | 27 | 33.3% | 37.0% |
| national | 73 | 17.8% | 68.5% |
| partisan | 136 | 19.1% | 46.3% |
| lean left | 68 | 27.9% | 39.7% |
| lean right | 50 | 20.0% | 54.0% |
| flagged low credibility | 18 | 16.7% | 44.4% |

A model scoring 0.9849 on its own held-out split calls **78% of real 2026 news
false**, including 63% of actual wire copy — the half of its training data it is
supposed to recognise. On the superseded one-day suite that figure was 77.9%.
Across nine months and 41 publishers it is 77.8%. **This is the result the
analysis rests on, and it is the one that replicated.**

### The lean gap did not replicate

The first freeze showed the model calling 36.8% of lean-left documents real
against 10.5% of lean-right, a 26.3 point gap, and that gap was reported as a
finding. On a suite spread across nine months it is 7.9 points, and under the
structure-only model it reverses sign.

| | First freeze (384 docs, one day) | v4 (266 docs, nine months) |
| --- | --- | --- |
| Baseline, lean left | 36.8% | 27.9% |
| Baseline, lean right | 10.5% | 20.0% |
| **Baseline gap** | **+26.3 pp** | **+7.9 pp** |
| Structure-only gap | +10.5 pp | **&minus;14.3 pp** |

**The 26.3 point gap is withdrawn.** It was a property of that corpus, not a
measured property of the model, and reporting it as the latter was an error.

### The obvious explanation was tested and does not hold either

The natural reading is that the one-day suite produced the gap: on the September
11 anniversary, left and right outlets cover the same event in different
registers. That is testable inside v4, where publisher composition is fixed and
only the period varies.

| Period | lean left | lean right | Gap |
| --- | --- | --- | --- |
| September (n=55) | 37.8% | 33.3% | +4.5 pp |
| January–August (n=63) | 16.1% | 12.5% | +3.6 pp |
| Pooled (n=118) | 27.9% | 20.0% | +7.9 pp |

**September inside this suite shows no larger a lean gap than the other eight
months.** 4.5 points against 3.6 is not a difference this corpus can resolve. So
the news-cycle explanation is not supported, and the cause of the original 26.3
points is not established by these runs. What changed between the two suites was
the dates *and* the publisher mix at once, and this test rules out the first of
those, not the second.

Two further things the table says, both stated because they bear on how the
pooled number should be read.

The pooled gap, 7.9 points, is **larger than either period's gap**. That is
Simpson's paradox and it is a composition effect: every rate is higher in
September, and lean-left is 54% September while lean-right is 36%, so pooling
inflates a gap that is small within each period.

September's rates are roughly double the rest of the year in the baseline
condition, 32.0% against 4.8–34.8% elsewhere with most months near 15%. That is
not established as a property of September. September is the live-window month —
its documents came from feeds, the other eight months from archive backfill — so
period and collection path are confounded in this suite and cannot be separated
within it.

**The flagged row is not the model working.** Low credibility is Media
Bias/Fact Check's label, not a truth verdict, and in this corpus it sits
entirely on right of centre publishers (D017), so it cannot be separated from
the lean effect above it. At 18 documents it is also the thinnest group in the
suite. The model has never seen a truth label for any of these articles.

## What this does not establish

Nothing here establishes that any article in the out-of-distribution suite is
true or false, and none of it is evidence about any publisher's accuracy.

## Limitations

The transformer column is reduced scale. The mirror's provenance is weaker than
a first party download and its licence is unstated. Ablation 1 is a lower bound.
Ablation 4 inherits the frozen suite's remaining defects — an empty aggregator
stratum, two wire publishers, 266 documents against a 400–600 target (D028) —
and its September rates are confounded with the live collection path.
Single seed, one epoch, no hyperparameter search, no confidence intervals —
deliberately the defaults the criticised approach uses, so differences between
adjacent rows are not established as significant. Only the structure only gap is
too large for seed variance to explain.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python eval/leakage/run_linear.py
PYTHONPATH=. .venv/bin/python eval/leakage/run_transformer.py
PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer \
    --condition baseline --corpus data/ood/frozen-v4
PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer \
    --condition structure_only --corpus data/ood/frozen-v4
```

Results are written to `eval/leakage/results/linear.json` and
`transformer.json`; ablation 4 writes `transfer.json`, keyed by the hash of the
corpus it was measured on, so replacing the suite without re-running is visible
rather than silent. The transformer runner skips any condition already present
in its results file, so an interrupted run resumes rather than retraining.
