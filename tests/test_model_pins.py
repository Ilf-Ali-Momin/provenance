"""The unpinned model reference check. Pure AST, no network, no model load."""

from __future__ import annotations

from pathlib import Path

from tools.check_model_pins import check_paths, check_source

HERE = Path("x.py")
SHA = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def _problems(source: str) -> list[str]:
    return [f.problem for f in check_source(source, HERE)]


def test_an_unpinned_from_pretrained_is_a_finding() -> None:
    assert _problems("m = AutoModel.from_pretrained('distilbert-base-uncased')") == [
        "has no revision= argument"
    ]


def test_a_pinned_from_pretrained_passes() -> None:
    assert _problems(f"m = AutoModel.from_pretrained('x', revision='{SHA}')") == []


def test_a_branch_name_is_not_a_pin() -> None:
    """revision='main' is a pin in syntax only."""
    assert _problems("m = AutoModel.from_pretrained('x', revision='main')") == [
        "pins revision='main', not a commit sha"
    ]


def test_an_unpinned_transformers_pipeline_is_a_finding() -> None:
    assert _problems("c = pipeline('zero-shot-classification', model='x')") == [
        "has no revision= argument"
    ]


def test_a_revision_from_a_variable_is_accepted_as_present() -> None:
    """Its value is decided at runtime; presence is the limit of a static check."""
    assert _problems("c = pipeline('t', model=s.model_id, revision=s.revision)") == []


def test_sklearn_make_pipeline_is_not_a_model_download() -> None:
    """A false positive here would train the team to ignore the check."""
    assert _problems("p = make_pipeline(TfidfVectorizer(), LogisticRegression())") == []


def test_an_unparseable_file_is_itself_a_finding() -> None:
    assert check_source("def (", HERE)[0].call == "parse"


def test_the_repository_has_no_unpinned_model_reference() -> None:
    """The check runs against the real tree, so a regression fails here too."""
    roots = [Path(p) for p in ("pipeline", "eval", "app", "tools") if Path(p).exists()]
    assert [str(f) for f in check_paths(roots)] == []
