"""Job queue. RQ over Redis, behind a protocol so tests need neither.

The queue is reached through `Submitter`, not through an RQ import at the call
site. That keeps the API testable without a running Redis, and it keeps the
decision to use RQ in one place: ARCHITECTURE.md names RQ, and if that changed
the API would not notice.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog
from pipeline.jobs import FetchOutcome, FetchRequest

from app.config import AppSettings

logger = structlog.get_logger(__name__)


class Submitter(Protocol):
    """What the API needs of a queue: submit one, ask about one."""

    def submit(self, payload: FetchRequest) -> str: ...

    def outcome(self, analysis_id: str) -> FetchOutcome | None: ...

    def status(self, analysis_id: str) -> str: ...


class RedisQueue:
    """RQ backed submitter. Constructed by the app; never imported by tests."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        from redis import Redis
        from rq import Queue

        self._settings = settings or AppSettings()
        self._redis = Redis.from_url(self._settings.redis_url)
        self._queue = Queue(
            self._settings.queue_name,
            connection=self._redis,
            default_timeout=self._settings.job_timeout_s,
        )

    @property
    def client(self) -> object:
        """The connection, so the rate limiter shares it rather than opening a second."""
        return self._redis

    def ping(self) -> str | None:
        """Readiness probe. Returns a problem string, or None when healthy."""
        try:
            self._redis.ping()
        except Exception as exc:
            return type(exc).__name__
        return None

    def submit(self, payload: FetchRequest) -> str:
        # The analysis id is the job id, so resubmitting a URL already in
        # flight joins the existing job rather than fetching the host twice.
        # API.md's idempotency, expressed where the duplicate would occur.
        existing = self._fetch_job(payload.analysis_id)
        if existing is not None:
            logger.info("job_already_queued", analysis_id=payload.analysis_id)
            return payload.analysis_id
        self._queue.enqueue(
            "pipeline.jobs.fetch_and_extract",
            payload,
            job_id=payload.analysis_id,
            result_ttl=self._settings.result_ttl_s,
        )
        logger.info("job_submitted", analysis_id=payload.analysis_id)
        return payload.analysis_id

    def _fetch_job(self, analysis_id: str) -> Any | None:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job

        try:
            return Job.fetch(analysis_id, connection=self._redis)
        except NoSuchJobError:
            return None

    def status(self, analysis_id: str) -> str:
        job = self._fetch_job(analysis_id)
        if job is None:
            return "unknown"
        return {"finished": "complete", "failed": "failed"}.get(
            job.get_status(refresh=True), "processing"
        )

    def outcome(self, analysis_id: str) -> FetchOutcome | None:
        job = self._fetch_job(analysis_id)
        if job is None or job.return_value() is None:
            return None
        value = job.return_value()
        return value if isinstance(value, FetchOutcome) else FetchOutcome(**value)
