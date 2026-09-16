"""Liveness and readiness.

Split deliberately. Liveness answers "is this process running", and must not
consult a dependency: a readiness failure that answers on the liveness path
gets the container killed during a database blip rather than taken out of
rotation. Readiness answers "can this process serve", and does consult them.

Checks register themselves rather than being hardcoded here, so a dependency
added in a later milestone becomes ready-checked by declaring itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from flask import Blueprint, jsonify

from app.config import AppSettings

logger = structlog.get_logger(__name__)

# name -> callable returning None on success, raising or returning a string on failure
Check = Callable[[], str | None]
_checks: dict[str, Check] = {}


def register_check(name: str, check: Check) -> None:
    _checks[name] = check


def clear_checks() -> None:
    _checks.clear()


def run_checks(settings: AppSettings | None = None) -> tuple[bool, dict[str, Any]]:
    """Run every registered check. Never raises; a raising check is a failure."""
    settings = settings or AppSettings()
    results: dict[str, Any] = {}
    healthy = True
    for name, check in sorted(_checks.items()):
        started = time.monotonic()
        try:
            problem = check()
        except Exception as exc:
            problem = f"{type(exc).__name__}"
        elapsed = time.monotonic() - started
        if elapsed > settings.readiness_timeout_s and problem is None:
            problem = f"exceeded {settings.readiness_timeout_s}s budget"
        results[name] = {
            "ok": problem is None,
            "ms": int(elapsed * 1000),
            **({"problem": problem} if problem else {}),
        }
        healthy = healthy and problem is None
    return healthy, results


def blueprint(settings: AppSettings) -> Blueprint:
    bp = Blueprint("health", __name__)

    @bp.get("/health/live")
    def live() -> tuple[Any, int]:
        return jsonify({"status": "ok", "service": settings.service_name}), 200

    @bp.get("/health/ready")
    def ready() -> tuple[Any, int]:
        healthy, results = run_checks(settings)
        if not healthy:
            logger.warning("readiness_failed", checks=results)
        return jsonify({"status": "ok" if healthy else "degraded", "checks": results}), (
            200 if healthy else 503
        )

    return bp
