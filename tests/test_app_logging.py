"""The log redaction filter. SECURITY.md requires this enforced, not observed."""

from __future__ import annotations

import json

import pytest
import structlog
from app.config import AppSettings
from app.logging import configure, redact_long_strings

ARTICLE = "The council met on Tuesday to consider the budget. " * 40


@pytest.fixture
def captured() -> list[dict]:
    """Capture what would be rendered, after every processor has run."""
    configure(AppSettings(log_json=True, log_level="DEBUG"))
    entries: list[dict] = []
    structlog.configure(
        processors=[
            *structlog.get_config()["processors"][:-1],
            lambda _l, _n, ed: entries.append(dict(ed)) or "",
        ],
        wrapper_class=structlog.make_filtering_bound_logger(10),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return entries


def test_article_text_cannot_be_logged_under_any_field_name(captured: list[dict]) -> None:
    """A name-keyed filter would pass this; the rule is on the value."""
    structlog.get_logger().info("extracted", note=ARTICLE)
    assert ARTICLE not in json.dumps(captured)
    assert "redacted" in captured[0]["note"]


def test_article_text_cannot_be_logged_as_the_message_itself(captured: list[dict]) -> None:
    structlog.get_logger().info(ARTICLE)
    assert ARTICLE not in json.dumps(captured)


def test_article_text_cannot_hide_in_a_nested_structure(captured: list[dict]) -> None:
    structlog.get_logger().info("doc", payload={"inner": [{"deep": ARTICLE}]})
    assert ARTICLE not in json.dumps(captured)


def test_redaction_applies_at_debug_level_too(captured: list[dict]) -> None:
    """SECURITY.md says at any level, including debug."""
    structlog.get_logger().debug("dumping", body=ARTICLE)
    assert ARTICLE not in json.dumps(captured)


def test_short_strings_survive_so_logs_stay_useful(captured: list[dict]) -> None:
    structlog.get_logger().info("fetched", host="example.com", status=200)
    assert captured[0]["host"] == "example.com"
    assert captured[0]["status"] == 200


def test_the_redaction_records_length_and_hash_so_lines_can_be_compared() -> None:
    processor = redact_long_strings(10)
    first = processor(None, "info", {"a": ARTICLE})["a"]
    second = processor(None, "info", {"a": ARTICLE})["a"]
    different = processor(None, "info", {"a": ARTICLE + "x"})["a"]
    assert first == second
    assert first != different
    assert str(len(ARTICLE)) in first


def test_non_string_values_are_left_alone() -> None:
    processor = redact_long_strings(5)
    assert processor(None, "info", {"n": 12345678, "b": True, "x": None}) == {
        "n": 12345678,
        "b": True,
        "x": None,
    }
