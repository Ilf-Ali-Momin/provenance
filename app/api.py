"""The `/v1` surface, as far as M1 builds it.

M1 is the security boundary and the skeleton, so this implements the URL path
of `POST /v1/analyses` and the poll endpoint, and nothing that needs an
analyser. The response carries the fetch outcome; the analyser sections arrive
with the analysers.

Two commitments from API.md are load bearing here rather than aspirational.

The URL path never blocks a request thread on a remote host: the handler
enqueues and returns 202 (D008). Nothing in this module fetches.

The response schema has no field capable of holding a verdict, and at this
layer no field capable of holding article text either. Both are enforced by the
model rather than by review.
"""

from __future__ import annotations

import contextlib
import hmac
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import structlog
from flask import Blueprint, jsonify, request
from pipeline.errors import ProvenanceError, UrlRejectedError
from pipeline.jobs import FetchRequest, analysis_id_for
from pipeline.net.addresses import parse_and_check
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import validate_url

from app.config import AppSettings
from app.limits import RateLimiter
from app.queue import Submitter

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class UnauthorisedError(ProvenanceError):
    code = "unauthorised"
    http_status = 401


class BadRequestError(ProvenanceError):
    code = "bad_request"
    http_status = 400


def _supplied_key() -> str:
    return request.headers.get("Authorization", "").removeprefix("Bearer ").strip()


def _requires_key(settings: AppSettings) -> Callable[[F], F]:
    def decorator(view: F) -> F:
        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            supplied = _supplied_key()
            # compare_digest against every configured key: a plain `in` leaks
            # key length and prefix through timing, and the set is small.
            if not supplied or not any(
                hmac.compare_digest(supplied, key) for key in settings.api_keys
            ):
                raise UnauthorisedError("A valid bearer key is required.")
            return view(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def blueprint(settings: AppSettings, queue: Submitter, limiter: RateLimiter) -> Blueprint:
    bp = Blueprint("api", __name__, url_prefix="/v1")
    needs_key = _requires_key(settings)

    @bp.post("/analyses")
    @needs_key
    def submit() -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        payload = body.get("input") or {}
        if payload.get("type") != "url":
            raise BadRequestError("M1 implements input.type 'url' only.")
        url = str(payload.get("context_url") or "").strip()
        if not url:
            raise BadRequestError("input.context_url is required for a url submission.")

        # Checked here as well as in the worker, and deliberately only as far
        # as it can be checked without the network. The worker's check is the
        # one that matters for safety: it resolves the name and validates every
        # returned address, pinned (D015).
        #
        # This one covers scheme and port, which need no lookup, and the
        # address itself when the host is an IP literal — `169.254.169.254`
        # requires no DNS to recognise. A hostname is *not* resolved here:
        # resolution in the request thread is exactly the blocking on attacker
        # controlled input that D008 exists to prevent, so a hostile name gets
        # a 202 and is refused by the worker a moment later.
        try:
            _, host, _ = validate_url(url, FetchSettings())
            with contextlib.suppress(ValueError):
                _, problem = parse_and_check(host)
                if problem is not None:
                    raise UrlRejectedError(f"{host}: {problem}")
        except UrlRejectedError as exc:
            raise BadRequestError(str(exc)) from exc

        # Order matters. The per key limits come first because they are about
        # this caller; the domain limit comes last because a rejection there is
        # about a publisher's servers and should not be reachable by a caller
        # who is already over their own budget.
        api_key = _supplied_key()
        limiter.check_key(api_key)
        limiter.check_fetch(api_key)
        limiter.check_domain(url)

        analysis_id = request.headers.get("Idempotency-Key") or analysis_id_for(url)
        queue.submit(FetchRequest(analysis_id=analysis_id, url=url))
        return (
            jsonify(
                {
                    "analysis_id": analysis_id,
                    "status": "processing",
                    "poll_url": f"/v1/analyses/{analysis_id}",
                }
            ),
            202,
        )

    @bp.get("/analyses/<analysis_id>")
    @needs_key
    def poll(analysis_id: str) -> tuple[Any, int]:
        limiter.check_key(_supplied_key())
        outcome = queue.outcome(analysis_id)
        if outcome is not None:
            return jsonify({"analysis_id": analysis_id, **outcome.model_dump()}), 200
        status = queue.status(analysis_id)
        if status == "unknown":
            raise BadRequestError("No such analysis.")
        return jsonify({"analysis_id": analysis_id, "status": status}), 200

    return bp
