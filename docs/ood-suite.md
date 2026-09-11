# OOD suite: first freeze

*Status: frozen 2026-09-11 as `ood_suite_2026` in `data/MANIFEST.yaml`. It does
**not** meet the collection plan, for the reasons in "Where it falls short".
It is kept as a measured first attempt, not as the suite M0 requires. Every
number below is measured from the frozen files or the collection run logs.*

## What it is

384 English news articles published after the 2026-01-01 snapshot, collected
from publisher feeds and news sitemaps, with publisher identity stripped from
the text, split by publishing domain into train, dev and frozen holdout.

| | Documents | Publishers |
| --- | --- | --- |
| partisan | 254 | 22 |
| regional | 75 | 5 |
| national | 40 | 3 |
| wire | 15 | 1 (UPI) |
| aggregator | 0 | 0 |
| **total** | **384** | **31** |

Partisan is balanced at 127 left of centre and 127 right. Splits are 219 train
(18 domains), 93 dev (7) and 72 holdout (6); no domain appears in more than one
split. Sixty documents carry the `low_quality` flag, all from four right of
centre publishers, so that flag is confounded with lean (D017) and is not a
safe grouping variable. Median length 3,601 characters.

Pipeline: collected at `2fbca13`, text rules re-applied at `3489893` (D021).
Corpus sha256 begins `9c59a053dfdd2214`.

## How it was made

Sixty-nine candidate publishers, with bias and credibility labels taken from
Media Bias/Fact Check and recorded with source URL and date (D007, D016, D017).
Every request went through the SSRF guarded client (D015), robots.txt was
honoured with wildcard rules implemented because the standard library ignores
them, and refusals ended collection from a host without retry (D019).

Thirty-eight of the 69 publishers yielded nothing. The collection run rejected:

| Reason | Count |
| --- | --- |
| undated (sitemap entry had only `lastmod`) | 3,708 |
| extraction failed | 95 |
| HTTP 402 paywall | 75 |
| robots.txt disallowed | 31 |
| host refused (401, 403, 429) | 14 |
| published before the snapshot | 11 |
| article on another host | 11 |
| no feed or news sitemap | 8 |
| declared or detected non-English (D020) | 7 |
| homepage unreachable | 4 |

Publisher identity removed from the frozen documents, by rule: own outlet names
masked 268 times (D018), boilerplate 78, newsletter promotions 28, datelines 22,
email addresses 18, name-prefix bylines 14, follow and contact lines 8,
affiliate disclosures 7, contributor credits 6, self promotion 5, staff bios 4,
self disclosures 4, membership and sponsor lines 2, photo credits 2, bylines 1.

## Measured quality

**Residue audit.** Three rounds of audit and re-cleaning took identity residue
candidates from 18 to 5 to 1. The last is "By Thursday, he was...", which is
prose. A review of every sentence removed by the newest rules found only
boilerplate.

**Residual separability** (collection plan section 5), 420 documents before
balancing, 31 publishers, TF-IDF with logistic regression:

| Question | Masked text | Structure only | Baseline |
| --- | --- | --- | --- |
| Stratum, folds grouped by publisher | 69.0% | 68.7% | 69.1% |
| Publisher identity | 40.5% | 18.8% | 3.6% |

No signal about publisher type transfers to publishers unseen in training.
Individual publishers remain identifiable, mostly through content: on structure
alone, punctuation, casing and length, identity falls to 18.8%. That is the
reverse of the ISOT corpus, where structure alone carried 81.8% and 83.5%.

## Where it falls short of the plan

1. **It is one day of news.** 371 of 384 documents were published in a single
   week and 289 (75%) on 2026-09-11 itself, the 25th anniversary of the
   September 11 attacks. 104 documents (27%) mention the attacks, including all
   15 wire documents. The plan requires collection spread across weeks so that
   no single news cycle dominates. A one-shot run cannot meet that: feeds and
   news sitemaps list only the last day or two, and D019 correctly refuses
   archive sitemap entries dated only by `lastmod`.
2. **Topic is unknown for 88%.** Topic is read from the URL path, and most
   publishers do not file articles under `/politics/`, `/health/`, `/science/`
   or `/business/`. The topic axis of the stratification is effectively absent.
3. **Aggregator is empty and wire is one publisher**, present only in train.
4. **Below the 400 to 600 target** at 384 after balancing.
5. **One invalid date.** A CBS News article in dev is dated 2026-10-03, after
   its collection. Collection now rejects such entries; the frozen record stays.
6. **Open obligations**, recorded in the manifest: no legitimate interest
   balancing test is written, storage is an encrypted local disk rather than
   the EU region SECURITY.md control 6 requires, and deletion is not implemented.

## What this means for M0

M0 is not done on this freeze. Ablation 4 also remains unrun, for a separate
reason: EVALUATION.md asks for accuracy degrading toward chance, which needs
veracity labels this suite deliberately does not have. D022 sets out how to
reach beyond the live window and is awaiting a decision.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.run --limit 15 --out data/ood/run-<sha>
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.reclean data/ood/run-<sha> data/ood/run-<sha>-clean-<sha>
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.audit data/ood/run-<sha>-clean-<sha>
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.residual data/ood/run-<sha>-clean-<sha>
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.freeze data/ood/run-<sha>-clean-<sha>
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.manifest data/ood/frozen
```

A rerun collects a different corpus, because feeds move. The frozen files are
the suite.
