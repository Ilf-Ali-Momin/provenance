"""The deletion procedure. Irreversible, so every guard is pinned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.ood_collection import erase as mod


def _corpus(root: Path) -> Path:
    target = root / "ood" / "pilot"
    target.mkdir(parents=True)
    (target / "documents.jsonl").write_text('{"text": "article"}\n', encoding="utf-8")
    (target / "summary.json").write_text("{}\n", encoding="utf-8")
    return target


def test_plan_reads_and_deletes_nothing(tmp_path: Path) -> None:
    target = _corpus(tmp_path)
    record = mod.plan(target, tmp_path)

    assert record["file_count"] == 2
    assert target.exists()
    assert {f["path"] for f in record["files"]} == {"documents.jsonl", "summary.json"}  # type: ignore[index,union-attr]


def test_erase_removes_the_corpus_and_writes_a_receipt(tmp_path: Path) -> None:
    target = _corpus(tmp_path)
    receipts = tmp_path / "deletions.jsonl"
    record = mod.erase(target, tmp_path, receipts)

    assert not target.exists()
    written = json.loads(receipts.read_text(encoding="utf-8").strip())
    assert written["file_count"] == 2
    assert written["erased_at"] == record["erased_at"]


def test_the_receipt_names_a_hash_per_file_so_it_can_be_checked(tmp_path: Path) -> None:
    """A deletion claim that cannot be verified is not a control."""
    target = _corpus(tmp_path)
    expected = {
        "documents.jsonl": mod._digest(target / "documents.jsonl"),
        "summary.json": mod._digest(target / "summary.json"),
    }
    receipts = tmp_path / "deletions.jsonl"
    mod.erase(target, tmp_path, receipts)

    written = json.loads(receipts.read_text(encoding="utf-8").strip())
    assert {f["path"]: f["sha256"] for f in written["files"]} == expected


def test_the_receipt_carries_no_article_text(tmp_path: Path) -> None:
    target = _corpus(tmp_path)
    receipts = tmp_path / "deletions.jsonl"
    mod.erase(target, tmp_path, receipts)
    assert "article" not in receipts.read_text(encoding="utf-8")


def test_a_target_outside_the_data_root_is_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / "not-data"
    outside.mkdir(exist_ok=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    root = tmp_path / "data"
    root.mkdir()

    with pytest.raises(mod.EraseRefusedError, match="outside the data root"):
        mod.erase(outside, root)
    assert (outside / "keep.txt").exists()


def test_a_traversal_path_is_refused_after_resolution(tmp_path: Path) -> None:
    """`data/../elsewhere` must not look contained just because it starts inside."""
    root = tmp_path / "data"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(mod.EraseRefusedError, match="outside the data root"):
        mod.erase(root / ".." / "elsewhere", root)
    assert (elsewhere / "keep.txt").exists()


def test_a_missing_target_is_refused_rather_than_reported_as_erased(tmp_path: Path) -> None:
    with pytest.raises(mod.EraseRefusedError, match="does not exist"):
        mod.plan(tmp_path / "ood" / "absent", tmp_path)


def test_cli_without_yes_deletes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    target = _corpus(tmp_path)
    code = mod.main([str(target), "--root", str(tmp_path)])

    assert code == mod.EXIT_OK
    assert target.exists()
    out = capsys.readouterr().out
    assert "would erase" in out and "pass --yes" in out


def test_cli_with_yes_deletes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    target = _corpus(tmp_path)
    receipts = tmp_path / "deletions.jsonl"
    code = mod.main([str(target), "--root", str(tmp_path), "--receipts", str(receipts), "--yes"])

    assert code == mod.EXIT_OK
    assert not target.exists()
    assert "erased" in capsys.readouterr().out
    assert receipts.exists()


def test_cli_refusal_is_its_own_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "data"
    root.mkdir()
    code = mod.main([str(tmp_path / "elsewhere"), "--root", str(root), "--yes"])

    assert code == mod.EXIT_REFUSED
    assert "refusing to erase" in capsys.readouterr().err
