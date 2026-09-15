"""Delete a collected corpus, and leave a receipt that says so.

DATA.md commits to retention limits and SECURITY.md's checklist requires that
"retention deletion actually deletes". Until now the manifest recorded
"deletion procedure not yet implemented", which makes the retention limit a
promise rather than a control. This is the control.

Deleting is the easy half. The receipt is the point: it records what was
removed, when, and the SHA-256 of every file removed, so a deletion claim can
be checked afterwards instead of believed. Content hashes are the one derived
form of article text DATA.md permits in storage, which is why the receipt can
be this specific without itself becoming a copy of what it erased.

Two guards, because this is the only irreversible operation in the project.
Nothing outside the data root is touched, whatever path is passed. And a run
without --yes reports exactly what it would delete and deletes nothing, so the
destructive form is never the one reached by accident.

    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.erase data/ood/pilot
    PYTHONPATH=. .venv/bin/python -m eval.ood_collection.erase data/ood/pilot --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

DATA_ROOT: Final = Path("data")
RECEIPTS: Final = Path("data/deletions.jsonl")
EXIT_OK: Final = 0
EXIT_REFUSED: Final = 2


class EraseRefusedError(Exception):
    """The target is outside the data root, or does not exist."""


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def plan(target: Path, root: Path = DATA_ROOT) -> dict[str, object]:
    """What erasing this target would remove. Reads; never deletes.

    Resolution happens before the containment check so that a symlink or a
    ``..`` segment cannot point the deletion outside the data root while
    looking like it stays inside it.
    """
    resolved = target.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        msg = f"refusing to erase {target}: outside the data root {root}"
        raise EraseRefusedError(msg)
    if not resolved.exists():
        msg = f"refusing to erase {target}: does not exist"
        raise EraseRefusedError(msg)

    files = sorted(p for p in resolved.rglob("*") if p.is_file())
    return {
        "target": str(resolved.relative_to(root_resolved.parent)),
        "files": [
            {
                "path": str(path.relative_to(resolved)),
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
            for path in files
        ],
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def erase(target: Path, root: Path = DATA_ROOT, receipts: Path = RECEIPTS) -> dict[str, object]:
    """Delete the target and append a receipt.

    The receipt is written after the deletion succeeds, so a receipt never
    claims an erasure that did not happen. It is written outside the target,
    which is why the receipts file lives at the data root rather than inside
    the corpus being removed.
    """
    record = plan(target, root)
    resolved = target.resolve()
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()

    record["erased_at"] = datetime.now(UTC).isoformat()
    receipts.parent.mkdir(parents=True, exist_ok=True)
    with receipts.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete a corpus and record the deletion.")
    parser.add_argument("target", type=Path)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually delete; without it the run only reports what would go",
    )
    args = parser.parse_args(argv)

    try:
        record = (
            plan(args.target, args.root)
            if not args.yes
            else erase(args.target, args.root, args.receipts)
        )
    except EraseRefusedError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_REFUSED

    verb = "erased" if args.yes else "would erase"
    sys.stdout.write(
        f"{verb} {record['target']}: {record['file_count']} files, {record['total_bytes']} bytes\n"
    )
    if args.yes:
        sys.stdout.write(f"receipt appended to {args.receipts}\n")
    else:
        sys.stdout.write("nothing was deleted; pass --yes to delete\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
