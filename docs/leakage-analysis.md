# Leakage analysis

*Status: linear probe complete, transformer reproduction in progress. Every
number below is measured. No number in this document is estimated, rounded
from memory, or carried over from published work.*

## The claim being tested

The standard project on this topic trains a classifier on the ISOT style
corpus, reports accuracy in the high nineties, and presents that figure as a
measure of its ability to detect false news. DATA.md asserts that the figure is
an artefact of how the corpus was assembled: its true half is wire copy from a
single agency, its false half a heterogeneous scrape, and the two differ
systematically in ways unrelated to veracity.

That assertion is testable. If accuracy survives the removal of information
that cannot possibly carry veracity, the classifier was never measuring
veracity.

## Corpus, and a provenance finding

Only mirrors of the corpus were reachable; no first party download was
available. Three were examined. The one used is recorded in
`data/MANIFEST.yaml` with `training_use: prohibited` per D010, and it carries
21,416 wire service documents against 22,851 others, matching the canonical
true count almost exactly.

**A measured caveat that changes one result.** In the original distribution
essentially every wire article opens with a `CITY (Reuters) - ` dateline. In
all three mirrors, that prefix is present in **0.3%** of wire documents. The
mirrors have been pre-stripped. Ablation 1 targets the dateline specifically,
so what is reported below is a weaker test than the one EVALUATION.md
specifies: it removes the first sentence from text whose most obvious tell has
already been removed by someone else. This is stated rather than quietly
reinterpreted, and it makes ablation 1 a lower bound.

## Method

TF-IDF over unigrams and bigrams, 50,000 features, logistic regression. Random
splits, 36,000 train and 8,267 held out, matching the standard approach that is
under criticism. Chance is 50%; the held out set is 48.2% positive.

Four conditions. **Baseline**: unmodified text. **First sentence removed**:
leading sentence stripped. **Structure only**: every alphanumeric run replaced
by a shape token preserving length, capitalisation and digit positions, so
`Trump` becomes `Xxxxx` and `2017` becomes `0000` — punctuation, casing and
document length survive, every content word is destroyed. **Length matched**:
both classes resampled to a common length distribution across 40 quantile bins,
removing length as a signal without removing a single word.

## Results

| Condition | Accuracy | Δ vs baseline | Information removed |
| --- | --- | --- | --- |
| Baseline | **0.9849** | — | none |
| First sentence removed | 0.9756 | −0.9 pp | leading sentence (dateline already absent) |
| Length matched | 0.9824 | −0.3 pp | document length |
| Structure only | **0.8184** | −16.7 pp | **every content word** |

## Reading the numbers

The decisive row is the last one. With every content word destroyed — no nouns,
no verbs, no names, no topic, nothing a human could read — accuracy is **81.8%**
against a 50% chance baseline. Punctuation habits, capitalisation patterns and
document length alone carry roughly two thirds of the distance from chance to
the headline figure.

No account of this result attributes it to veracity detection. A document's
comma placement does not become different when its content becomes false. What
the model is separating is two publishing pipelines: one house style guide and
one HTML scrape, distinguishable by their formatting residue.

The other two ablations are individually undramatic and jointly informative.
Length alone is worth 0.3 points, so the length confound DATA.md anticipates is
real but small. First sentence removal costs 0.9 points, but against
pre-stripped text this measures only what a *second* sentence removal would
cost — it is a lower bound and it should not be read as evidence that the
dateline was unimportant in the original corpus.

That the surviving signal is distributed rather than concentrated in one
sentence is the more uncomfortable finding. A single leaky feature could be
deleted. A pervasive stylistic fingerprint cannot be, which is why D010
prohibits this corpus as a training source under augmentation, pretraining or
auxiliary use: the leakage is co-extensive with the labels.

## What this does and does not establish

It establishes that reported accuracy on this corpus is not a capability
measurement, and it quantifies how much of the figure survives the removal of
content.

It does not establish how such a model behaves on text published after the
corpus was built. That is ablation 4, cross-domain transfer, and it **has not
been run**: it requires the 2026 out-of-distribution slice, which does not yet
exist. The collection procedure is specified in `eval/ood_collection/README.md`
and the collector depends on the SSRF-guarded client from M1. Reporting a
transfer number here without that data would reproduce exactly the overclaiming
this document exists to criticise, so the row is absent rather than estimated.

## Limitations

A linear model on TF-IDF is not the transformer EVALUATION.md specifies; the
transformer reproduction is running and its numbers will be added to the table
above rather than replacing it. The mirror's provenance is weaker than a first
party download and its licence is unstated. Ablation 1 is a lower bound for the
reason given. Single seed throughout; no confidence intervals are claimed.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python eval/leakage/run_linear.py
```

Results are written to `eval/leakage/results/linear.json`.
