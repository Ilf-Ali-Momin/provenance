"""Fail when a model is loaded without a pinned revision.

ROADMAP.md M1 requires an "unpinned model reference check" in CI, and RULES.md
requires every model artefact to carry a "base model identifier pinned to a
revision hash". Without the check that rule is prose: a model id alone resolves
to whatever the default branch holds at the moment of the download, so two runs
of the same commit can load different weights and the second one silently
reports different numbers.

That is not hypothetical here. D029 pinned a topic model to a revision, and
nothing enforced it; the transformer ablations loaded `distilbert-base-uncased`
unpinned for the whole of M0.

**What is checked.** Calls to `*.from_pretrained(...)` and to transformers'
`pipeline(...)` must pass `revision=`. Where the revision is a string literal it
must look like a commit sha, since `revision="main"` is a pin in syntax only.

**What is not checked, and why it cannot be.** Where the revision comes from a
variable or a settings field, only its presence is visible to a static check;
its value is decided at runtime. That is the limit of an AST check, and the
alternative -- resolving settings at check time -- would make CI depend on the
environment it is meant to be independent of.

    PYTHONPATH=. .venv/bin/python -m tools.check_model_pins pipeline eval app
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Final

LOADERS: Final = frozenset({"from_pretrained"})
PIPELINE_CALLS: Final = frozenset({"pipeline"})
SHA_LIKE: Final = re.compile(r"^[0-9a-f]{7,40}$")
BRANCHES: Final = frozenset({"main", "master", "refs/heads/main", "HEAD"})


class Finding:
    def __init__(self, path: Path, line: int, call: str, problem: str) -> None:
        self.path = path
        self.line = line
        self.call = call
        self.problem = problem

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.call}() {self.problem}"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute) and node.func.attr in LOADERS:
        return node.func.attr
    if isinstance(node.func, ast.Name) and node.func.id in PIPELINE_CALLS:
        return node.func.id
    return None


def check_source(source: str, path: Path) -> list[Finding]:
    """Findings for one file. A file that does not parse is a finding itself."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(path, exc.lineno or 0, "parse", f"could not be parsed: {exc.msg}")]

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        revision = next((kw for kw in node.keywords if kw.arg == "revision"), None)
        if revision is None:
            findings.append(Finding(path, node.lineno, name, "has no revision= argument"))
            continue
        value = revision.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literal = value.value
            if literal in BRANCHES or not SHA_LIKE.match(literal):
                findings.append(
                    Finding(path, node.lineno, name, f"pins revision={literal!r}, not a commit sha")
                )
    return findings


def check_paths(roots: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for root in roots:
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for file in files:
            findings += check_source(file.read_text(encoding="utf-8"), file)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail on unpinned model references.")
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("pipeline"), Path("eval")])
    args = parser.parse_args(argv)

    findings = check_paths([r for r in args.roots if r.exists()])
    for finding in findings:
        sys.stderr.write(f"{finding}\n")
    if findings:
        sys.stderr.write(
            f"\n{len(findings)} unpinned model reference(s). "
            "RULES.md requires a base model identifier pinned to a revision hash.\n"
        )
        return 1
    sys.stdout.write("every model reference is pinned to a revision\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
