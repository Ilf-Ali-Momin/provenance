"""Unit tests for the leakage ablation transforms.

Pure logic only: no network, no corpus download, no model loading (RULES.md).
These matter because a silently wrong transform does not crash, it produces a
plausible number, which is the failure mode this project exists to catch.
"""

from __future__ import annotations

import numpy as np
from eval.leakage.ablations import (
    Split,
    apply_condition,
    length_matched_indices,
    strip_first_sentence,
    structure_only,
)


class TestStripFirstSentence:
    def test_removes_leading_sentence(self) -> None:
        assert strip_first_sentence("First one. Second one. Third.") == "Second one. Third."

    def test_handles_question_and_exclamation(self) -> None:
        assert strip_first_sentence("Really? Yes.") == "Yes."
        assert strip_first_sentence("Stop! Go.") == "Go."

    def test_single_sentence_becomes_empty(self) -> None:
        # A document of one sentence has nothing left. It must not silently
        # return the original text, which would make the ablation a no-op.
        assert strip_first_sentence("Only one sentence here.") == ""

    def test_no_terminator_becomes_empty(self) -> None:
        assert strip_first_sentence("no terminator at all") == ""


class TestStructureOnly:
    def test_preserves_length(self) -> None:
        src = "Trump said on Friday, 2017: it's over!"
        assert len(structure_only(src)) == len(src)

    def test_maps_case_and_digits(self) -> None:
        assert structure_only("Trump") == "Xxxxx"
        assert structure_only("2017") == "0000"
        assert structure_only("NATO") == "XXXX"

    def test_preserves_punctuation_and_whitespace(self) -> None:
        assert structure_only("Hi, there!  Bye.") == "Xx, xxxxx!  Xxx."

    def test_destroys_every_content_word(self) -> None:
        out = structure_only("Reuters reported Washington policy")
        assert "Reuters" not in out
        assert set(out) <= {"X", "x", "0", " "}


class TestLengthMatched:
    def test_returns_balanced_classes(self) -> None:
        rng = np.random.default_rng(0)
        y = np.array([1] * 60 + [0] * 40)
        # Class 1 systematically longer, the confound being removed.
        lengths = np.concatenate([rng.normal(900, 20, 60), rng.normal(300, 20, 40)])
        idx = length_matched_indices(y, lengths, rng, bins=4)
        kept = y[idx]
        assert kept.sum() == len(kept) - kept.sum(), "classes must be equal after matching"

    def test_disjoint_lengths_yield_nothing(self) -> None:
        # No bin contains both classes, so nothing can be matched. Must return
        # empty rather than fabricating pairs across bins.
        rng = np.random.default_rng(0)
        y = np.array([1] * 20 + [0] * 20)
        lengths = np.concatenate([np.full(20, 1000.0), np.full(20, 10.0)])
        assert length_matched_indices(y, lengths, rng, bins=2).size == 0


class TestApplyCondition:
    def _split(self) -> tuple[Split, Split]:
        s = Split(["Alpha one. Beta two.", "Gamma three. Delta four."], np.array([1, 0]))
        return s, s

    def test_baseline_is_identity(self) -> None:
        tr, te = self._split()
        out_tr, out_te = apply_condition(tr, te, "baseline")
        assert out_tr.x == tr.x
        assert out_te.x == te.x

    def test_labels_survive_every_condition(self) -> None:
        tr, te = self._split()
        for cond in ("baseline", "first_sentence_removed", "structure_only"):
            out_tr, _ = apply_condition(tr, te, cond)  # type: ignore[arg-type]
            assert len(out_tr.x) == len(out_tr.y), f"{cond} desynced text and labels"
