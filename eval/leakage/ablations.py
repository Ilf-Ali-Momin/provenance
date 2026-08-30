"""Leakage ablations for the ISOT style corpus (EVALUATION.md section 2).

Run before any component training. The purpose is not to build a classifier.
It is to demonstrate that the headline accuracy of the standard approach is an
artefact of corpus construction, by removing information that cannot carry
veracity and observing how much accuracy survives.

The corpus used here is prohibited as a training source for any shipped
component (DECISIONS.md D010). It appears in this module only.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from datasets import concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Mirror of the ISOT style Kaggle corpus. Every accessible mirror has had the
# "CITY (Reuters) - " dateline prefix removed; see docs/leakage-analysis.md.
CORPUS_ID = "ErfanMoosaviMonazzah/fake-news-detection-dataset-English"

Condition = Literal[
    "baseline",
    "first_sentence_removed",
    "structure_only",
    "length_matched",
]

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Split:
    """A materialised split. `y` is 1 for the wire service half."""

    x: list[str]
    y: np.ndarray


def _shape_token(match: re.Match[str]) -> str:
    """Map an alphanumeric run to its case and length shape.

    Preserves length, casing pattern and digit positions while destroying every
    content word. "Trump" becomes "Xxxxx", "2017" becomes "0000".
    """
    out = []
    for ch in match.group(0):
        if ch.isdigit():
            out.append("0")
        elif ch.isupper():
            out.append("X")
        else:
            out.append("x")
    return "".join(out)


def strip_first_sentence(text: str) -> str:
    """Ablation 1. Remove the leading sentence, which carries any dateline."""
    parts = _SENT_END.split(text.strip(), maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def structure_only(text: str) -> str:
    """Ablation 2. Keep punctuation, casing and length; destroy all content."""
    return _ALNUM_RUN.sub(_shape_token, text)


def length_matched_indices(
    y: np.ndarray, lengths: np.ndarray, rng: np.random.Generator, bins: int = 40
) -> np.ndarray:
    """Ablation 3. Indices resampling both classes to one length distribution.

    Bins documents by character length and, within each bin, truncates both
    classes to the smaller class count. Removes length as a usable signal
    without removing any word.
    """
    edges = np.quantile(lengths, np.linspace(0, 1, bins + 1))
    edges[-1] += 1.0
    keep: list[np.ndarray] = []
    for lo, hi in itertools.pairwise(edges):
        in_bin = np.where((lengths >= lo) & (lengths < hi))[0]
        if in_bin.size == 0:
            continue
        pos = in_bin[y[in_bin] == 1]
        neg = in_bin[y[in_bin] == 0]
        n = min(pos.size, neg.size)
        if n == 0:
            continue
        keep.append(rng.choice(pos, n, replace=False))
        keep.append(rng.choice(neg, n, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def load_corpus() -> tuple[Split, Split]:
    """Return (train, test). Random splits, matching the standard approach."""
    ds = load_dataset(CORPUS_ID)
    train = concatenate_datasets([ds["train"], ds["validation"]])
    test = ds["test"]
    return (
        Split(list(train["text"]), np.asarray(train["label"], dtype=int)),
        Split(list(test["text"]), np.asarray(test["label"], dtype=int)),
    )


def apply_condition(
    train: Split, test: Split, condition: Condition, seed: int = 0
) -> tuple[Split, Split]:
    """Return the train and test splits transformed for `condition`."""
    if condition == "baseline":
        return train, test
    if condition == "first_sentence_removed":
        return (
            Split([strip_first_sentence(t) for t in train.x], train.y),
            Split([strip_first_sentence(t) for t in test.x], test.y),
        )
    if condition == "structure_only":
        return (
            Split([structure_only(t) for t in train.x], train.y),
            Split([structure_only(t) for t in test.x], test.y),
        )
    rng = np.random.default_rng(seed)
    out: list[Split] = []
    for sp in (train, test):
        lengths = np.asarray([len(t) for t in sp.x], dtype=float)
        idx = length_matched_indices(sp.y, lengths, rng)
        out.append(Split([sp.x[i] for i in idx], sp.y[idx]))
    return out[0], out[1]


def run_linear(train: Split, test: Split, seed: int = 0) -> float:
    """TF-IDF plus logistic regression. Fast probe, reported alongside."""
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    xtr = vec.fit_transform(train.x)
    xte = vec.transform(test.x)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(xtr, train.y)
    return float(accuracy_score(test.y, clf.predict(xte)))
