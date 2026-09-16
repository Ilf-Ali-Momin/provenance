"""Flask application factory.

A factory rather than a module level `app`, because tests need an instance with
their own settings and an import time singleton cannot be given one. It also
keeps configuration out of import order, which is where Flask projects usually
acquire their first untestable dependency.

The request id is bound into structlog's context, not passed around. RULES.md
requires every log line to carry one, and a parameter threaded through call
sites is a requirement that holds until someone forgets. It is echoed back on
the response so a user reporting a failure can quote something that appears in
the logs, and so a log line can be tied to a request without the log carrying
anything about the request's content.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from flask import Flask, Response, g, request

from app import api, errors, health
from app.config import AppSettings
from app.limits import MemoryWindow, RateLimiter, RedisWindow, Window
from app.logging import configure
from app.queue import Submitter

REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_ID_CHARS = 64
logger = structlog.get_logger(__name__)


def sanitise_request_id(supplied: str) -> str:
    """Reduce a caller supplied request id to something safe to log and echo.

    Accepting the caller's value makes a request traceable across a proxy. The
    value is then written into every log line this request produces and into a
    response header, so it is length capped and reduced to url safe characters.
    A newline is the cheapest way to forge an audit trail entry, and while
    werkzeug rejects one in a header before it reaches here, this function is
    what makes that a defence rather than an accident of the server in front.
    """
    return "".join(c for c in supplied if c.isalnum() or c in "-_")[:MAX_REQUEST_ID_CHARS]


def _incoming_request_id() -> str:
    return sanitise_request_id(request.headers.get(REQUEST_ID_HEADER, "")) or uuid.uuid4().hex


def create_app(settings: AppSettings | None = None, queue: Submitter | None = None) -> Flask:
    settings = settings or AppSettings()
    configure(settings)

    app = Flask(__name__)
    app.config["SETTINGS"] = settings

    @app.before_request
    def _bind_request_id() -> None:
        g.request_id = _incoming_request_id()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=g.request_id,
            method=request.method,
            path=request.path,
        )

    @app.after_request
    def _echo_request_id(response: Response) -> Response:
        response.headers[REQUEST_ID_HEADER] = getattr(g, "request_id", "")
        return response

    @app.teardown_request
    def _unbind(_exc: BaseException | None) -> None:
        structlog.contextvars.clear_contextvars()

    errors.register(app)
    app.register_blueprint(health.blueprint(settings))

    # The queue is constructed here, not imported at module scope, so a test
    # can pass its own and the app never opens a Redis connection it does not
    # need. Readiness learns about Redis by the queue registering itself.
    redis_client: object | None = None
    if queue is None:
        from app.queue import RedisQueue

        live = RedisQueue(settings)
        health.register_check("redis", live.ping)
        redis_client = live.client
        queue = live
    # The limiter shares Redis with the queue when there is one, so the per
    # domain counter is global across every API process. A memory window in a
    # multi process deployment would limit per worker, which is not a limit.
    window: Window = MemoryWindow() if redis_client is None else RedisWindow(redis_client)
    app.register_blueprint(api.blueprint(settings, queue, RateLimiter(window, settings)))

    logger.info("app_created", service=settings.service_name, environment=settings.environment)
    return app


def __getattr__(name: str) -> Any:
    raise AttributeError(name)
