"""The /v1 URL path: auth, enqueue, poll. No Redis, no network."""

from __future__ import annotations

import pytest
from app import create_app, health
from app.config import AppSettings
from flask import Flask
from pipeline.jobs import FetchOutcome, FetchRequest, analysis_id_for

KEY = "test-key-aaaaaaaaaaaaaaaaaaaa"
URL = "https://example.com/article"


class StubQueue:
    def __init__(self) -> None:
        self.submitted: list[FetchRequest] = []
        self.outcomes: dict[str, FetchOutcome] = {}
        self.statuses: dict[str, str] = {}

    def submit(self, payload: FetchRequest) -> str:
        self.submitted.append(payload)
        self.statuses.setdefault(payload.analysis_id, "processing")
        return payload.analysis_id

    def outcome(self, analysis_id: str) -> FetchOutcome | None:
        return self.outcomes.get(analysis_id)

    def status(self, analysis_id: str) -> str:
        return self.statuses.get(analysis_id, "unknown")


@pytest.fixture
def queue() -> StubQueue:
    return StubQueue()


@pytest.fixture
def app(queue: StubQueue) -> Flask:
    health.clear_checks()
    return create_app(AppSettings(api_keys=(KEY,), environment="test"), queue=queue)


def _post(app: Flask, body: dict, key: str | None = KEY, **kwargs: object) -> object:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    headers.update(kwargs.pop("headers", {}))  # type: ignore[arg-type]
    return app.test_client().post("/v1/analyses", json=body, headers=headers)


def test_a_url_submission_returns_202_and_never_fetches(app: Flask, queue: StubQueue) -> None:
    """D008: the URL path must not block a request thread on a remote host."""
    response = _post(app, {"input": {"type": "url", "context_url": URL}})
    assert response.status_code == 202
    assert response.json["status"] == "processing"
    assert response.json["poll_url"] == f"/v1/analyses/{response.json['analysis_id']}"
    assert [r.url for r in queue.submitted] == [URL]


def test_resubmitting_the_same_url_reuses_the_analysis_id(app: Flask, queue: StubQueue) -> None:
    first = _post(app, {"input": {"type": "url", "context_url": URL}})
    second = _post(app, {"input": {"type": "url", "context_url": URL}})
    assert first.json["analysis_id"] == second.json["analysis_id"] == analysis_id_for(URL)


def test_an_idempotency_key_overrides_the_derived_id(app: Flask) -> None:
    response = _post(
        app,
        {"input": {"type": "url", "context_url": URL}},
        headers={"Idempotency-Key": "caller-supplied"},
    )
    assert response.json["analysis_id"] == "caller-supplied"


def test_a_request_without_a_key_is_refused(app: Flask) -> None:
    response = _post(app, {"input": {"type": "url", "context_url": URL}}, key=None)
    assert response.status_code == 401
    assert response.json["error"]["code"] == "unauthorised"


def test_a_wrong_key_is_refused(app: Flask) -> None:
    assert _post(app, {"input": {"type": "url", "context_url": URL}}, key="nope").status_code == 401


def test_no_configured_keys_refuses_everything_rather_than_allowing_it(
    queue: StubQueue,
) -> None:
    """An unset secret must never be the permissive case."""
    health.clear_checks()
    app = create_app(AppSettings(api_keys=()), queue=queue)
    assert _post(app, {"input": {"type": "url", "context_url": URL}}).status_code == 401


def test_health_stays_open_without_a_key(app: Flask) -> None:
    assert app.test_client().get("/health/live").status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/admin",
        "file:///etc/passwd",
        "http://example.com:22/",
    ],
)
def test_a_hostile_url_is_rejected_at_submission(app: Flask, queue: StubQueue, url: str) -> None:
    """The worker's check is the one that matters; this one gives a reason."""
    response = _post(app, {"input": {"type": "url", "context_url": url}})
    assert response.status_code == 400
    assert queue.submitted == []


def test_a_text_submission_is_refused_at_m1(app: Flask) -> None:
    response = _post(app, {"input": {"type": "text", "content": "hello"}})
    assert response.status_code == 400


def test_a_missing_url_is_refused(app: Flask) -> None:
    assert _post(app, {"input": {"type": "url"}}).status_code == 400


def test_polling_an_unfinished_analysis_reports_processing(app: Flask, queue: StubQueue) -> None:
    _post(app, {"input": {"type": "url", "context_url": URL}})
    response = app.test_client().get(
        f"/v1/analyses/{analysis_id_for(URL)}", headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 200
    assert response.json["status"] == "processing"


def test_a_finished_analysis_returns_derived_facts_and_no_text(
    app: Flask, queue: StubQueue
) -> None:
    """The response schema cannot express article text, by construction."""
    analysis_id = analysis_id_for(URL)
    queue.outcomes[analysis_id] = FetchOutcome(
        analysis_id=analysis_id,
        status="complete",
        content_sha256="abc123",
        characters=4210,
        page_published="2026-03-01",
    )
    response = app.test_client().get(
        f"/v1/analyses/{analysis_id}", headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.json["content_sha256"] == "abc123"
    assert response.json["characters"] == 4210
    assert "text" not in response.json
    assert "content" not in response.json


def test_polling_an_unknown_analysis_is_a_400(app: Flask) -> None:
    response = app.test_client().get(
        "/v1/analyses/an_nothing", headers={"Authorization": f"Bearer {KEY}"}
    )
    assert response.status_code == 400


# --- rate limiting at the endpoint (SECURITY.md control 5) ------------------


@pytest.fixture
def tight(queue: StubQueue) -> Flask:
    health.clear_checks()
    return create_app(
        AppSettings(
            api_keys=(KEY, "second-key-bbbbbbbbbbbbbbbbbb"),
            fetch_limit_per_key=2,
            rate_limit_per_key=50,
            fetches_per_domain_per_hour=3,
        ),
        queue=queue,
    )


def test_the_fetch_path_is_limited_per_key_with_a_retry_after(tight: Flask) -> None:
    for n in range(2):
        assert (
            _post(tight, {"input": {"type": "url", "context_url": f"{URL}?{n}"}}).status_code == 202
        )
    response = _post(tight, {"input": {"type": "url", "context_url": f"{URL}?3"}})
    assert response.status_code == 429
    assert response.json["error"]["code"] == "rate_limited"
    assert int(response.headers["Retry-After"]) > 0


def test_the_domain_limit_holds_across_different_keys(tight: Flask, queue: StubQueue) -> None:
    """Control 5: no more than N fetches of one domain regardless of key."""
    other = "second-key-bbbbbbbbbbbbbbbbbb"
    assert _post(tight, {"input": {"type": "url", "context_url": f"{URL}?1"}}).status_code == 202
    assert _post(tight, {"input": {"type": "url", "context_url": f"{URL}?2"}}).status_code == 202
    assert (
        _post(tight, {"input": {"type": "url", "context_url": f"{URL}?3"}}, key=other).status_code
        == 202
    )
    blocked = _post(tight, {"input": {"type": "url", "context_url": f"{URL}?4"}}, key=other)
    assert blocked.status_code == 429
    assert blocked.json["error"]["code"] == "domain_rate_limited"
    assert len(queue.submitted) == 3


def test_a_rate_limited_submission_never_reaches_the_queue(tight: Flask, queue: StubQueue) -> None:
    for n in range(4):
        _post(tight, {"input": {"type": "url", "context_url": f"{URL}?{n}"}})
    assert len(queue.submitted) <= 2


def test_health_is_never_rate_limited(tight: Flask) -> None:
    """A probe blocked by a limit reports the service down when it is not."""
    for _ in range(30):
        assert tight.test_client().get("/health/live").status_code == 200
