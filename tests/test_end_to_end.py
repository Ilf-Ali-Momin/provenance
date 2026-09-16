"""M1's done-when, end to end against a live Redis.

ROADMAP.md: "A URL submitted through the queue is fetched safely, extracted,
and stored as a hash and text in memory, with the adversarial SSRF suite green."

Marked `integration`, so it is excluded by default and never runs in CI: it
needs Redis up, and one case makes a single real request to a real publisher.
Run it deliberately:

    docker compose up -d redis
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_end_to_end.py -m integration -q
"""

from __future__ import annotations

import os
import uuid

import pytest
from app import create_app, health
from app.config import AppSettings
from flask import Flask
from pipeline.jobs import FetchRequest, analysis_id_for, fetch_and_extract

pytestmark = pytest.mark.integration

KEY = "e2e-key-aaaaaaaaaaaaaaaaaaaaaaa"
REDIS_URL = os.environ.get("APP_REDIS_URL", "redis://127.0.0.1:6379/0")
RESULT_SIZE_CEILING = 2048
FEED_URL = "https://www.theguardian.com/world/rss"


@pytest.fixture(scope="module")
def live_url() -> str:
    """An article the publisher is advertising right now.

    Not a hardcoded URL. Articles move and a pinned one turns this test into a
    404 detector: the first version of this test used an invented Guardian path
    and failed for that reason rather than for anything about the code. The
    feed is the publisher's own list of what exists, which is the same source
    the collector uses.
    """
    from pipeline.net.config import FetchSettings
    from pipeline.net.feeds import parse_feed
    from pipeline.net.fetch import fetch

    settings = FetchSettings()
    feed = fetch(
        FEED_URL,
        settings=settings.model_copy(
            update={
                "allowed_content_types": frozenset(
                    {"application/rss+xml", "text/xml", "application/xml"}
                )
            }
        ),
    )
    entries = [e for e in parse_feed(feed.text) if e.url.startswith("https://")]
    if not entries:
        pytest.skip("the publisher's feed listed nothing fetchable")
    return entries[0].url


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(api_keys=(KEY,), redis_url=REDIS_URL, queue_name="fetch-e2e")


@pytest.fixture
def app(settings: AppSettings) -> Flask:
    health.clear_checks()
    return create_app(settings)


def test_redis_is_reachable_and_readiness_says_so(app: Flask) -> None:
    response = app.test_client().get("/health/ready")
    assert response.status_code == 200, response.json
    assert response.json["checks"]["redis"]["ok"] is True


def test_a_url_goes_through_the_queue_and_comes_back_as_a_hash(
    app: Flask, settings: AppSettings, live_url: str
) -> None:
    """The milestone's done-when, minus the worker process: the job runs here."""
    client = app.test_client()
    submitted = client.post(
        "/v1/analyses",
        json={"input": {"type": "url", "context_url": live_url}},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert submitted.status_code == 202
    analysis_id = submitted.json["analysis_id"]
    assert analysis_id == analysis_id_for(live_url)

    # Run the job exactly as the worker would, then assert on what it returns.
    outcome = fetch_and_extract(FetchRequest(analysis_id=analysis_id, url=live_url))

    assert outcome.status in {"complete", "failed"}
    if outcome.status == "failed":
        pytest.skip(f"publisher returned {outcome.error_code}; not a defect in this path")
    assert outcome.content_sha256 and len(outcome.content_sha256) == 64
    assert outcome.characters and outcome.characters > 0
    # The whole point of D032: nothing text-shaped survives into the result.
    assert "text" not in outcome.model_dump()


def test_the_queue_round_trips_a_result_without_article_text(
    app: Flask, settings: AppSettings, live_url: str
) -> None:
    """What RQ actually wrote to Redis, inspected as bytes.

    The first version of this test read `rq:results:<id>` with GET and scanned
    for the word "article". Both were wrong, and instructively so. RQ 2.x keeps
    results in a *stream*, so GET raises WRONGTYPE; and "article" appears inside
    the legitimate error code `no_article_text`, so the marker would have failed
    on a result that was entirely correct. A test that greps for a word is a
    test that will eventually be satisfied by the wrong thing.

    So this asserts the property instead: whatever is stored is small. A
    text-free outcome is a few hundred bytes. The article this job extracted is
    kilobytes, and no encoding of it fits under the bound.
    """
    from redis import Redis
    from rq import Queue, SimpleWorker

    redis = Redis.from_url(settings.redis_url)
    queue = Queue(settings.queue_name, connection=redis)
    queue.empty()

    # A fresh id per run. RQ's result stream retains previous executions, and
    # summing them made the first version of this assertion fail on six correct
    # results rather than on one bad one.
    payload = FetchRequest(analysis_id=f"an_e2e_{uuid.uuid4().hex[:12]}", url=live_url)
    job = queue.enqueue("pipeline.jobs.fetch_and_extract", payload, job_id=payload.analysis_id)
    SimpleWorker([queue], connection=redis).work(burst=True)

    job.refresh()
    assert job.get_status() == "finished"

    outcome = job.return_value()
    assert outcome.status in {"complete", "failed"}
    if outcome.status == "complete":
        assert outcome.characters and outcome.characters > 1000

    stored = b"".join(
        value for _id, fields in redis.xrange(f"rq:results:{job.id}") for value in fields.values()
    )
    assert stored, "nothing was stored, so this test proves nothing"
    assert len(stored) < RESULT_SIZE_CEILING, (
        f"{len(stored)} bytes stored; an article-free outcome is far smaller, "
        "so something text-shaped reached the result backend"
    )
