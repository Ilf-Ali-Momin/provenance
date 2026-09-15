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
*says*. The baseline model, unchanged, was applied to the 384 document frozen
suite; the table gives the share of each group assigned to the corpus's wire
half, the class that corpus treats as real.

| Group | Documents | Called "real" |
| --- | --- | --- |
| **Whole suite** | 384 | **22.1%** |
| wire | 15 | 40.0% |
| regional | 75 | 30.7% |
| national | 40 | 20.0% |
| partisan | 254 | 18.9% |
| lean left | 136 | 36.8% |
| left | 51 | 15.7% |
| lean right | 67 | 10.5% |
| right | 60 | 10.0% |
| flagged low credibility | 60 | 10.0% |

A model scoring 0.9849 on its own held out split calls 78% of real 2026 news
false, including 60% of actual wire copy — the half of its training data it is
supposed to recognise. Its output tracks publisher type and lean: 36.8% for lean
left against 10.5% for lean right, across articles where no difference in
truthfulness has been established or claimed.

**The flagged row is not the model working.** Low credibility is Media
Bias/Fact Check's label, not a truth verdict, and in this corpus it sits
entirely on right of centre publishers (D017), so it cannot be separated from
the lean effect above it. The model has never seen a truth label for any of
these articles.

### The same ablation with every content word destroyed

Repeating ablation 4 with the structure only model — the one trained on shape
tokens at 0.8184 — separates the formatting residue from the topic.

| Group | Baseline | Structure only |
| --- | --- | --- |
| **Whole suite** | **22.1%** | **40.9%** |
| wire | 40.0% | 60.0% |
| regional | 30.7% | 38.7% |
| national | 20.0% | 37.5% |
| partisan | 18.9% | 40.9% |
| lean left | 36.8% | 47.8% |
| lean right | 10.5% | 37.3% |
| flagged low credibility | 10.0% | 30.0% |

Two things move. The blanket rejection relaxes — 40.9% called real rather than
22.1%, and 60% of wire copy recognised rather than 40% — and the lean gap
shrinks from 26.3 points to 10.5. Most of the political skew therefore rides on
topic vocabulary, what 2026 articles are about, while a smaller stylistic
component survives the destruction of every content word. This locates the skew;
neither model is doing anything with veracity. Single seed, no intervals, so the
residual 10.5 points is not established as non-zero.

## What this does not establish

Nothing here establishes that any article in the out-of-distribution suite is
true or false, and none of it is evidence about any publisher's accuracy.

## Limitations

The transformer column is reduced scale. The mirror's provenance is weaker than
a first party download and its licence is unstated. Ablation 1 is a lower bound.
Ablation 4 inherits the frozen suite's defects — one dominant news cycle, an
empty aggregator stratum, a single wire publisher (`docs/ood-suite.md`) — so its
shares describe the model on that day, not across 2026; both variants are
repeated when the spread suite exists (D024).
Single seed, one epoch, no hyperparameter search, no confidence intervals —
deliberately the defaults the criticised approach uses, so differences between
adjacent rows are not established as significant. Only the structure only gap is
too large for seed variance to explain.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python eval/leakage/run_linear.py
PYTHONPATH=. .venv/bin/python eval/leakage/run_transformer.py
PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer --condition baseline
PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer --condition structure_only
```

Results are written to `eval/leakage/results/linear.json` and
`transformer.json`. The transformer runner skips any condition already present
in its results file, so an interrupted run resumes rather than retraining.
