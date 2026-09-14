# Leakage analysis

*Status: all four ablations run. Ablation 4 is reported as a prediction
distribution rather than an accuracy, for the reason given below. Every number
in this document is measured. None is estimated, rounded from memory, or
carried over from published work.*

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

The same four conditions were then run against a fine tuned transformer, which
is what EVALUATION.md actually specifies: `distilbert-base-uncased`, one epoch,
batch 16, learning rate 5e-5, defaults throughout, since the point is to
reproduce what the common project reports rather than to tune it. That run is
**reduced in scale** — 6,000 training and 2,000 held out documents truncated to
128 tokens, against the linear probe's 36,000 and 8,267. The host was under
heavy external load that starved a full size run to roughly 36 seconds per
step. The reduction is stated here so the transformer column is never read as a
full scale reproduction. The two columns are not comparable in absolute terms;
what matters is whether they agree in shape.

## Results

| Condition | Linear, full | DistilBERT, reduced | Information removed |
| --- | --- | --- | --- |
| Baseline | **0.9849** | **0.9920** | none |
| First sentence removed | 0.9756 (−0.9 pp) | 0.9710 (−2.1 pp) | leading sentence (dateline already absent) |
| Length matched | 0.9824 (−0.3 pp) | 0.9700 (−2.2 pp) | document length |
| Structure only | **0.8184 (−16.7 pp)** | **0.8350 (−15.7 pp)** | **every content word** |
| Cross domain transfer | no accuracy exists | — | — (own section below) |

## Reading the numbers

The decisive row is structure only. With every content word destroyed — no
nouns, no verbs, no names, no topic, nothing a human could read — accuracy is
**81.8%** for the linear probe and **83.5%** for the transformer, against a 50%
chance baseline. Punctuation habits, capitalisation patterns and document
length alone carry roughly two thirds of the distance from chance to the
headline figure.

That two unrelated model families land within 1.7 points of each other on that
condition matters more than either number. A single result could be an artefact
of the probe; TF-IDF logistic regression and a fine tuned transformer agreeing
this closely, at different corpus scales, indicates the signal is a property of
the corpus rather than of the classifier.

No account of this result attributes it to veracity detection. A document's
comma placement does not become different when its content becomes false. What
the model is separating is two publishing pipelines: one house style guide and
one HTML scrape, distinguishable by their formatting residue.

The other two ablations are individually undramatic and jointly informative.
Length matching costs 0.3 points on the linear probe and 2.2 on the
transformer, so the length confound DATA.md anticipates is real but minor in
both. First sentence removal costs 0.9 and 2.1 points respectively — but
against pre-stripped text that measures what a *second* sentence removal would
cost, so it is a lower bound and not evidence that the dateline was
unimportant in the original corpus.

That the surviving signal is distributed rather than concentrated in one
sentence is the more uncomfortable finding. A single leaky feature could be
deleted. A pervasive stylistic fingerprint cannot be, which is why D010
prohibits this corpus as a training source under augmentation, pretraining or
auxiliary use: the leakage is co-extensive with the labels.

## Ablation 4: what the model says about 2026 news

The fourth ablation asks how the model behaves on text published after the
corpus was built. EVALUATION.md expects accuracy degrading toward chance. That
number cannot exist: accuracy needs veracity labels, and the out-of-distribution
suite deliberately has none (D001). What can be reported is what the model
*says*. The baseline model, unchanged, was applied to the 384 document frozen
suite, and the table gives the share of each group it assigns to the corpus's
wire service half, the class that corpus treats as real.

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

The same model scores 0.9849 on its own held out split. On real news published
in 2026 it calls 78% of everything false, including 60% of actual wire copy,
which is the half of its training data it is supposed to recognise. Its output
tracks publisher type and political lean: 36.8% for lean left against 10.5% for
lean right, across articles where no difference in truthfulness has been
established or claimed.

**The flagged row is not the model working.** Low credibility here is Media
Bias/Fact Check's label, not a truth verdict, and in this corpus that flag sits
entirely on right of centre publishers (D017), so it cannot be separated from
the lean effect above it. The model has never seen a truth label for any of
these articles.

Two limitations. The suite is one day of news, dominated by a single news cycle
(`docs/ood-suite.md`), so these shares describe the model's behaviour on that
day, not across 2026; the measurement is repeated when the spread suite exists.
And the structure only variant of this ablation has not been run yet.

## What this does and does not establish

It establishes that reported accuracy on this corpus is not a capability
measurement, that most of the figure survives the removal of content, and that
the resulting model, moved to news from a later year, sorts articles by
publisher type and political lean rather than by anything it could know about
truth.

It does not establish that any article in the out-of-distribution suite is true
or false, and nothing here should be read as evidence about any publisher's
accuracy.

## Limitations

The transformer column is a reduced scale run for the reason given in the
method, so its absolute values should not be quoted as a full corpus
reproduction. The mirror's provenance is weaker than a first party download and
its licence is unstated. Ablation 1 is a lower bound, because the mirrors
arrive pre-stripped. Ablation 4 reads the frozen suite as it stands, so its shares
inherit that suite's defects: one day of news, one dominant news cycle, an
empty aggregator stratum and a single wire publisher. Its structure only
variant has not been run. Single seed, one epoch, no hyperparameter search, no
confidence intervals — deliberately the defaults the criticised approach uses,
which means differences between adjacent rows are not established as
significant. Only the structure only gap is too large for seed variance to
explain.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python eval/leakage/run_linear.py
PYTHONPATH=. .venv/bin/python eval/leakage/run_transformer.py
PYTHONPATH=. .venv/bin/python -m eval.leakage.run_transfer --condition baseline
```

Results are written to `eval/leakage/results/linear.json` and
`transformer.json`. The transformer runner skips any condition already present
in its results file, so an interrupted run resumes rather than retraining.
