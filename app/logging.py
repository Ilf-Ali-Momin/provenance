"""Structured logging, with the redaction filter SECURITY.md requires.

RULES.md mandates structlog with JSON output and a `request_id` on every line.
SECURITY.md control 6 goes further: no article text in any log at any level,
"enforced by a log redaction filter, not by developer discipline".

That last clause decides the design. A filter keyed on field names — dropping
`text`, `body`, `html` — is developer discipline wearing a filter's clothes: it
works until someone logs the same string under a different key, which is exactly
the mistake a tired developer makes at 2am while debugging an extraction bug.

So the rule is on the value, not the name. Any string longer than
`max_logged_chars` is replaced by its length and a content hash, wherever it
appears and whatever it is called, including inside nested structures. An
article cannot be logged because an article is long, and the filter does not
need to know that it is an article.

The hash is kept because it is the one derived form DATA.md permits in storage,
and because it lets two log lines be compared for identity without either
carrying content.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any, Final

import structlog

from app.config import AppSettings

REDACTED: Final = "[redacted {chars} chars sha256:{digest}]"
DIGEST_CHARS: Final = 12


def _redact_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:DIGEST_CHARS]
        return REDACTED.format(chars=len(value), digest=digest)
    if isinstance(value, dict):
        return {k: _redact_value(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v, limit) for v in value)
    return value


def redact_long_strings(limit: int) -> Any:
    """Build a structlog processor that truncates every long string by value.

    Applies to the event message itself as well as to every bound field, since
    `logger.info(article_text)` is as much a leak as `logger.info(text=...)`.
    """

    def processor(_logger: object, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return {key: _redact_value(value, limit) for key, value in event_dict.items()}

    return processor


def configure(settings: AppSettings | None = None) -> None:
    """Install the logging configuration. Idempotent."""
    settings = settings or AppSettings()
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=settings.log_level, force=True
    )
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Redaction runs last before rendering, so nothing added by an
            # earlier processor can slip past it.
            redact_long_strings(settings.max_logged_chars),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
