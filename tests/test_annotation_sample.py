"""The annotation sampler. No network; the corpus is synthesised."""

from __future__ import annotations

import json
from pathlib import Path

from eval.annotation.sample import load_documents, passages_from, sample, to_form

PARA = "The council met to consider the budget on Tuesday afternoon in the chamber. " * 8


def _doc(domain: str, n: int, split: str = "train") -> dict[str, object]:
    return {
        "doc_id": f"d{n}",
        "domain": domain,
        "stratum": "partisan",
        "text": f"{PARA}\n\n{PARA}",
        "split": split,
    }


def test_the_holdout_is_never_drawn_from(tmp_path: Path) -> None:
    """D009 permits two holdout inspections; hand annotation is an inspection."""
    for split in ("train", "dev", "holdout"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({**_doc(f"{split}.com", 1), "split": split}) + "\n"
        )
    splits = {d["split"] for d in load_documents(tmp_path)}
    assert splits == {"train", "dev"}
    assert "holdout" not in splits


def test_passages_stay_inside_the_length_band() -> None:
    for passage in passages_from(f"{PARA}\n\n{PARA}"):
        assert 400 <= len(passage) <= 1200


def test_a_short_paragraph_is_not_offered_for_annotation() -> None:
    assert passages_from("Too short.") == []


def test_a_long_paragraph_is_cut_at_a_sentence_boundary() -> None:
    passage = passages_from("Sentence one is here. " * 200)[0]
    assert passage.endswith(".")
    assert len(passage) <= 1200


def test_sampling_is_reproducible() -> None:
    docs = [_doc(f"pub{i}.com", i) for i in range(20)]
    assert [r["passage_id"] for r in sample(docs, 10)] == [
        r["passage_id"] for r in sample(docs, 10)
    ]


def test_no_single_publisher_dominates_the_set() -> None:
    """One outlet's house style must not be what the extractor is measured on.

    Round robin alone does not deliver this: when the small pools run out the
    biggest publisher takes every remaining slot. The cap does.
    """
    docs = [_doc("big.com", i) for i in range(50)] + [
        _doc(f"small{i}.com", 100 + i) for i in range(5)
    ]
    rows = sample(docs, 20, max_share=0.25)
    counts = {d: sum(1 for r in rows if r["domain"] == d) for d in {str(r["domain"]) for r in rows}}
    assert max(counts.values()) <= 5


def test_the_cap_underfills_rather_than_skewing() -> None:
    """Meeting the number by taking more from one outlet is the wrong trade."""
    docs = [_doc("big.com", i) for i in range(50)]
    rows = sample(docs, 100, max_share=0.05)
    assert len(rows) == 5
    assert {r["domain"] for r in rows} == {"big.com"}


def test_the_form_has_empty_spans_for_a_human_to_fill() -> None:
    """Nothing in this project may pre-fill them: a generated gold standard
    measures the generator."""
    rows = sample([_doc("a.com", 1)], 1)
    parsed = json.loads(to_form(rows).splitlines()[0])
    assert parsed["spans"] == []
    assert parsed["notes"] == ""
