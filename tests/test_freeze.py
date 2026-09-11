"""Freeze rules: dedupe, topic, partisan balance, grouped splits. No network."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from eval.ood_collection.collect import Document
from eval.ood_collection.freeze import (
    assign_splits,
    balance_partisan,
    corpus_hash,
    dedupe,
    freeze,
    topic_from_url,
)


def doc(
    domain: str,
    n: int,
    *,
    stratum: str = "partisan",
    lean: str | None = "left",
    url: str = "",
    text_hash: str | None = None,
) -> Document:
    return Document(
        doc_id=f"doc_{domain}_{n}",
        domain=domain,
        stratum=stratum,
        lean=lean,
        low_quality=False,
        url=url or f"https://{domain}/politics/{n}",
        title=None,
        published="2026-03-01T00:00:00+00:00",
        text="body",
        text_sha256=text_hash or f"h_{domain}_{n}",
        markers_removed={},
        had_invisible=False,
        collected_at="2026-09-11T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("url", "topic"),
    [
        ("https://x.com/politics/2026/story", "politics"),
        ("https://x.com/news/health/a", "health"),
        ("https://x.com/business/markets/a", "business"),
        ("https://x.com/environment/a", "science"),
        ("https://x.com/sport/a", "unknown"),
        ("https://x.com/politician-profile", "unknown"),
    ],
)
def test_topic_comes_from_whole_path_segments(url: str, topic: str) -> None:
    """Whole segments only: `politician-profile` is not the politics section."""
    assert topic_from_url(url) == topic


def test_dedupe_keeps_same_story_at_two_publishers() -> None:
    a = doc("a.com", 1, text_hash="same")
    b = doc("b.com", 1, text_hash="same")
    a_again = doc("a.com", 2, text_hash="same")
    kept, removed = dedupe([a, b, a_again, a])
    assert [d.doc_id for d in kept] == ["doc_a.com_1", "doc_b.com_1"]
    assert removed == 2


def test_partisan_sides_come_out_equal() -> None:
    docs = [doc("l1.com", i, lean="left") for i in range(8)]
    docs += [doc("l2.com", i, lean="lean_left") for i in range(8)]
    docs += [doc("r1.com", i, lean="right") for i in range(5)]
    docs += [doc("n.com", i, stratum="national", lean="center") for i in range(3)]

    out, info = balance_partisan(docs, random.Random(1))  # noqa: S311
    partisan = [d for d in out if d.stratum == "partisan"]
    left = [d for d in partisan if d.lean in {"left", "lean_left"}]
    right = [d for d in partisan if d.lean in {"right", "lean_right"}]

    assert len(left) == len(right) == 5
    assert info == {"per_side": 5, "dropped_for_balance": 11}
    assert sum(1 for d in out if d.stratum == "national") == 3, "other strata untouched"


def test_balance_spreads_across_domains_on_a_side() -> None:
    """One prolific outlet must not stand in for its whole side."""
    docs = [doc("big.com", i, lean="left") for i in range(20)]
    docs += [doc("small.com", i, lean="left") for i in range(3)]
    docs += [doc("r.com", i, lean="right") for i in range(6)]
    out, _ = balance_partisan(docs, random.Random(1))  # noqa: S311
    left_domains = [d.domain for d in out if d.lean == "left"]
    assert left_domains.count("small.com") == 3
    assert left_domains.count("big.com") == 3


def test_every_domain_lands_in_exactly_one_split() -> None:
    docs = [doc(f"d{i}.com", j, stratum="regional", lean=None) for i in range(9) for j in range(3)]
    assignment = assign_splits(docs)
    assert set(assignment) == {f"d{i}.com" for i in range(9)}
    assert set(assignment.values()) == {"train", "dev", "holdout"}


def test_stratum_with_three_domains_reaches_every_split() -> None:
    docs = [doc(f"w{i}.com", 0, stratum="wire", lean=None) for i in range(3)]
    assert set(assign_splits(docs).values()) == {"train", "dev", "holdout"}


def test_splits_are_deterministic() -> None:
    docs = [doc(f"d{i}.com", 0, stratum="national", lean="center") for i in range(7)]
    assert assign_splits(docs, seed=5) == assign_splits(docs, seed=5)


def test_corpus_hash_ignores_order() -> None:
    a, b = doc("a.com", 1), doc("b.com", 1)
    assert corpus_hash([a, b]) == corpus_hash([b, a])


def test_freeze_reports_no_domain_crossing_splits() -> None:
    docs = [
        doc(f"p{i}.com", j, lean="left" if i % 2 else "right") for i in range(10) for j in range(4)
    ]
    docs += [
        doc(f"r{i}.com", j, stratum="regional", lean=None, url=f"https://r{i}.com/health/{j}")
        for i in range(5)
        for j in range(2)
    ]
    records, entry = freeze(docs)

    assert entry["splits"]["domains_in_more_than_one_split"] == []  # type: ignore[index]
    assert entry["record_count"] == sum(len(r) for r in records.values())
    assert entry["training_use"] == "prohibited"
    assert entry["splits"]["holdout_inspections"] == []  # type: ignore[index]
    assert "regional/health" in entry["cells_stratum_topic"]  # type: ignore[operator]


def test_cli_freezes_a_run_and_refuses_to_overwrite(tmp_path: Path) -> None:
    """A frozen suite does not change; a second freeze must fail, not update."""
    from eval.ood_collection import freeze as freeze_mod
    from eval.ood_collection.collect import Rejection
    from eval.ood_collection.run import record_domain

    run_dir = tmp_path / "run"
    for i in range(6):
        domain = f"p{i}.com"
        lean = "left" if i % 2 else "right"
        record_domain(run_dir, domain, [doc(domain, j, lean=lean) for j in range(3)], Rejection())
    (run_dir / "pipeline_sha.txt").write_text("abc1234\n")
    out = tmp_path / "frozen"

    assert freeze_mod.main([str(run_dir), "--out", str(out)]) == 0
    entry = json.loads((out / "manifest_entry.json").read_text())
    assert entry["extraction_pipeline"] == "abc1234"
    assert entry["record_count"] == sum(
        len((out / f"{name}.jsonl").read_text().splitlines())
        for name in ("train", "dev", "holdout")
    )

    assert freeze_mod.main([str(run_dir), "--out", str(out)]) == 1


def test_strata_that_cannot_reach_every_split_are_named() -> None:
    """One wire publisher cannot be in train, dev and holdout at once."""
    docs = [doc("wire.com", j, stratum="wire", lean=None) for j in range(4)]
    docs += [doc(f"r{i}.com", j, stratum="regional", lean=None) for i in range(5) for j in range(2)]
    _, entry = freeze(docs)
    splits = entry["splits"]
    assert splits["strata_missing_a_split"] == ["wire"]  # type: ignore[index]
    assert splits["splits_per_stratum"]["regional"] == ["dev", "holdout", "train"]  # type: ignore[index]
