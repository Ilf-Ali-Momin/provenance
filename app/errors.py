"""One place where an exception becomes an HTTP response.

RULES.md requires typed exceptions mapped to API error codes in a single
location. `pipeline.errors` owns the taxonomy and each class carries its own
`code` and `http_status`; this module is the boundary that renders them, so a
new error type gets an HTTP response without anyone editing a handler.

An unexpected exception is rendered as `internal_error` with no detail. The
detail goes to the log, where the redaction filter has already run over it: an
exception message can carry a fragment of the document that caused it, and a
500 body is the easiest place in a system to leak one.
"""

from __future__ import annotations

from typing import Any

import structlog
from flask import Flask, jsonify
from pipeline.errors import ProvenanceError
from werkzeug.exceptions import HTTPException

logger = structlog.get_logger(__name__)


def _body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def register(app: Flask) -> None:
    @app.errorhandler(ProvenanceError)
    def _provenance(exc: ProvenanceError) -> tuple[Any, int]:
        logger.warning("request_failed", code=exc.code, detail=str(exc))
        return jsonify(_body(exc.code, str(exc))), exc.http_status

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException) -> tuple[Any, int]:
        code = (exc.name or "http_error").lower().replace(" ", "_")
        return jsonify(_body(code, exc.description or "")), exc.code or 500

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception) -> tuple[Any, int]:
        # Deliberately no detail in the body. The type reaches the log; the
        # message may quote the input that produced it.
        logger.error("unhandled_exception", exception_type=type(exc).__name__, detail=str(exc))
        return jsonify(_body("internal_error", "An unexpected error occurred.")), 500
