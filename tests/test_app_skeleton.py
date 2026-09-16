"""App factory: request ids, error mapping, health. No network, no database."""

from __future__ import annotations

import pytest
from app import REQUEST_ID_HEADER, create_app, health, sanitise_request_id
from app.config import AppSettings
from flask import Flask
from pipeline.errors import ProvenanceError, UrlRejectedError


@pytest.fixture
def app() -> Flask:
    health.clear_checks()
    application = create_app(AppSettings(environment="test"))

    @application.get("/boom/<kind>")
    def boom(kind: str) -> str:
        if kind == "typed":
            raise UrlRejectedError("that URL is not permitted")
        raise RuntimeError(f"leaky detail about {'x' * 50}")

    return application


def test_liveness_does_not_consult_dependencies(app: Flask) -> None:
    """A dependency blip must take the pod out of rotation, not kill it."""
    health.register_check("db", lambda: "unreachable")
    response = app.test_client().get("/health/live")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_readiness_reports_the_failing_check_by_name(app: Flask) -> None:
    health.register_check("db", lambda: "unreachable")
    health.register_check("cache", lambda: None)
    response = app.test_client().get("/health/ready")
    assert response.status_code == 503
    assert response.json["checks"]["db"]["ok"] is False
    assert response.json["checks"]["cache"]["ok"] is True


def test_a_raising_check_is_a_failure_not_a_500(app: Flask) -> None:
    """A probe that propagates turns a degraded service into an outage."""

    def explode() -> str | None:
        raise ConnectionError("no route")

    health.register_check("db", explode)
    response = app.test_client().get("/health/ready")
    assert response.status_code == 503
    assert response.json["checks"]["db"]["problem"] == "ConnectionError"


def test_readiness_with_no_checks_is_ready(app: Flask) -> None:
    assert app.test_client().get("/health/ready").status_code == 200


def test_a_typed_error_keeps_its_code_and_status(app: Flask) -> None:
    response = app.test_client().get("/boom/typed")
    assert response.status_code == 400
    assert response.json["error"]["code"] == "url_rejected"
    assert response.json["error"]["message"] == "that URL is not permitted"


def test_every_taxonomy_member_maps_to_a_code_and_a_status() -> None:
    """RULES.md wants one mapping place; this asserts it is complete."""

    def descendants(cls: type) -> list[type]:
        out = []
        for sub in cls.__subclasses__():
            out += [sub, *descendants(sub)]
        return out

    for cls in descendants(ProvenanceError):
        assert cls.code != "internal_error", f"{cls.__name__} has no code of its own"
        assert 400 <= cls.http_status <= 599


def test_an_unexpected_error_leaks_no_detail_into_the_body(app: Flask) -> None:
    """A 500 body is the easiest place in a system to leak document content."""
    app.config["PROPAGATE_EXCEPTIONS"] = False
    response = app.test_client().get("/boom/untyped")
    assert response.status_code == 500
    assert response.json["error"]["code"] == "internal_error"
    assert "leaky detail" not in response.get_data(as_text=True)


def test_a_missing_route_is_json_not_html(app: Flask) -> None:
    response = app.test_client().get("/nope")
    assert response.status_code == 404
    assert response.json["error"]["code"] == "not_found"


def test_a_request_id_is_minted_and_echoed(app: Flask) -> None:
    response = app.test_client().get("/health/live")
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_a_supplied_request_id_is_honoured_so_traces_survive_a_proxy(app: Flask) -> None:
    response = app.test_client().get("/health/live", headers={REQUEST_ID_HEADER: "abc-123"})
    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


def test_a_forged_request_id_cannot_inject_into_the_log_stream() -> None:
    """It is written into every line this request logs; a newline would forge one.

    Asserted on the function rather than through the test client, because
    werkzeug refuses to build a request with a newline header at all. That
    refusal is the server in front being careful; this is the project being
    careful, and the two are not the same defence.
    """
    forged = 'a\n{"level":"error","event":"fabricated"}'
    assert sanitise_request_id(forged) == "alevelerroreventfabricated"
    assert "\n" not in sanitise_request_id(forged)


def test_a_request_id_is_length_capped() -> None:
    assert len(sanitise_request_id("x" * 500)) == 64


def test_an_empty_or_unusable_request_id_is_replaced(app: Flask) -> None:
    """Punctuation-only input must not yield an empty id."""
    assert sanitise_request_id("!!!") == ""
    response = app.test_client().get("/health/live", headers={REQUEST_ID_HEADER: "!!!"})
    assert len(response.headers[REQUEST_ID_HEADER]) == 32
