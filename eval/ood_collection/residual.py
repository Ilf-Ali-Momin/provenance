"""Residual separability check for the OOD corpus (collection plan, section 5).

After every construction rule has run, can a plain classifier still tell the
strata or the publishers apart from the text? That is the question this module
answers with numbers, and either answer is a finding: separable means signal
survived that is worth investigating, not separable is the expected result.

Four measurements, because each isolates something different.

* **Stratum, grouped by publisher.** Cross validation folds never share a
  domain, so a classifier cannot succeed by memorising one outlet's house style.
  Success here is signal about publisher *type* that transfers to outlets never
  seen in training, which is plausible and not by itself leakage: wire copy and
  opinion writing do read differently.
* **Publisher identity.** Folds may share a domain. This is the direct residue
  measure D018 points to: how identifiable an outlet still is from its own text
  after its name, byline and dateline were removed.
* **Both again on structure only.** Every letter mapped to its case and every
  digit to zero, as in the M0 ablation. Accuracy that survives this comes from
  punctuation, casing and length, which is the mechanism the leakage analysis
  measured in the ISOT corpus.

Every accuracy is reported beside its majority class baseline and its sample
size. On a small corpus these numbers are noisy; they are recorded anyway,
because an unreported check is indistinguishable from one that was never run.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from eval.leakage.ablations import structure_only
from eval.ood_collection.collect import Document
from eval.ood_collection.run import load_documents

FOLDS: Final = 5
SEED: Final = 20260101


def majority_baseline(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def _model() -> object:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )


def _grouped_accuracy(texts: list[str], labels: list[str], groups: list[str]) -> float | None:
    n_groups = len(set(groups))
    if n_groups < 2 or len(set(labels)) < 2:
        return None
    splitter = GroupKFold(n_splits=min(FOLDS, n_groups))
    scores = cross_val_score(_model(), texts, labels, groups=groups, cv=splitter)
    return float(np.mean(scores))


def _stratified_accuracy(texts: list[str], labels: list[str]) -> float | None:
    smallest = min(Counter(labels).values()) if labels else 0
    if len(set(labels)) < 2 or smallest < 2:
        return None
    splitter = StratifiedKFold(n_splits=min(FOLDS, smallest), shuffle=True, random_state=SEED)
    scores = cross_val_score(_model(), texts, labels, cv=splitter)
    return float(np.mean(scores))


def _measure(texts: list[str], strata: list[str], domains: list[str]) -> dict[str, object]:
    return {
        "stratum_grouped_by_publisher": {
            "accuracy": _grouped_accuracy(texts, strata, domains),
            "majority_baseline": round(majority_baseline(strata), 4),
            "classes": len(set(strata)),
        },
        "publisher_identity": {
            "accuracy": _stratified_accuracy(texts, domains),
            "majority_baseline": round(majority_baseline(domains), 4),
            "classes": len(set(domains)),
        },
    }


def residual_report(documents: list[Document]) -> dict[str, object]:
    """Separability of strata and publishers, on masked text and on shape."""
    usable = [d for d in documents if d.text.strip()]
    # Publishers with too few documents cannot be cross validated for identity
    # and are left out of that measure only, and named, not silently dropped.
    per_domain = Counter(d.domain for d in usable)
    thin = sorted(domain for domain, count in per_domain.items() if count < 2)
    kept = [d for d in usable if d.domain not in thin]

    texts = [d.text for d in kept]
    strata = [d.stratum for d in kept]
    domains = [d.domain for d in kept]

    report: dict[str, object] = {
        "documents": len(kept),
        "publishers": len(set(domains)),
        "strata": dict(Counter(strata)),
        "publishers_excluded_for_too_few_documents": thin,
    }
    if len(kept) < 4:
        report["skipped"] = "fewer than four usable documents"
        return report

    report["masked_text"] = _measure(texts, strata, domains)
    report["structure_only"] = _measure([structure_only(t) for t in texts], strata, domains)
    return report


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ood/raw")
    report = residual_report(load_documents(corpus))
    (corpus / "residual.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
