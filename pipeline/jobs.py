"""The fetch job, as it runs inside a worker.

D008 keeps the fetch out of the request thread: remote latency is unbounded and
attacker influenced, and the fetch must run in the network restricted worker
container, which the API container is not. This module is what the worker runs.

No Flask import, deliberately. The worker needs the pipeline and nothing else,
and importing the application package would drag a web framework into a
container whose job is to talk to hostile hosts.

**The result carries no article text, and that is the point of this module's
shape.** RQ stores a job's return value in Redis. Redis is storage. DATA.md
permits only content hashes, derived claims and aggregate statistics to reach
storage, and SECURITY.md control 6 says article text is never persisted:
extraction happens in memory, the text is discarded. So the job returns the
hash and the derived facts, the text goes out of scope when the function
returns, and the requirement holds by construction rather than by a reviewer
noticing (D032).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, Field

from pipeline.errors import ProvenanceError, UrlRejectedError
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import fetch
from pipeline.net.robots import RobotsCache, RobotsWithSitemaps
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

ID_PREFIX: Final = "an_"


class FetchRequest(BaseModel):
    """Queue payload. RULES.md: pydantic across every process boundary."""

    model_config = {"frozen": True}

    analysis_id: str
    url: str
    outlet_names: tuple[str, ...] = ()


class FetchOutcome(BaseModel):
    """What the worker returns, and therefore what reaches Redis.

    Every field here is a derived fact or a hash. There is no field capable of
    holding article text, which is the same enforcement strategy API.md uses to
    keep a verdict out of the response: make the schema unable to express it.
    """

    model_config = {"frozen": True}

    analysis_id: str
    status: str
    final_url: str | None = None
    http_status: int | None = None
    content_sha256: str | None = None
    characters: int | None = None
    page_published: str | None = None
    declared_language: str | None = None
    had_invisible: bool | None = None
    markers_removed: dict[str, int] = Field(default_factory=dict)
    error_code: str | None = None
    completed_at: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def analysis_id_for(url: str) -> str:
    """Stable id for a URL, so resubmitting one in flight joins it.

    API.md provides idempotency via the `Idempotency-Key` header or, absent
    that, a normalised content hash. A URL has no content until it is fetched,
    so the normalised URL is what is hashed.
    """
    digest = hashlib.sha256(url.strip().encode()).hexdigest()
    return f"{ID_PREFIX}{digest[:24]}"


def fetch_and_extract(
    payload: FetchRequest,
    robots: RobotsWithSitemaps | None = None,
    fetch_settings: FetchSettings | None = None,
    extract_settings: ExtractSettings | None = None,
) -> FetchOutcome:
    """Fetch one URL, extract it, and return derived facts only.

    A robots refusal and a fetch failure are both outcomes, not exceptions to
    propagate: the worker records them against the analysis so the API can
    report why, rather than the job vanishing into a failed queue entry.
    """
    robots = robots or RobotsCache()
    fetch_settings = fetch_settings or FetchSettings()
    extract_settings = extract_settings or ExtractSettings()
    try:
        if not robots.allows(payload.url):
            raise UrlRejectedError("robots.txt disallows this URL for our user agent.")
        robots.wait_for(payload.url)
        response = fetch(payload.url, settings=fetch_settings)
    except ProvenanceError as exc:
        return FetchOutcome(
            analysis_id=payload.analysis_id,
            status="failed",
            error_code=exc.code,
            completed_at=_now(),
        )

    article = extract_article(
        response.text,
        url=payload.url,
        settings=extract_settings,
        outlet_names=payload.outlet_names,
    )
    if article is None:
        return FetchOutcome(
            analysis_id=payload.analysis_id,
            status="failed",
            final_url=response.final_url,
            http_status=response.status_code,
            error_code="no_article_text",
            completed_at=_now(),
        )

    # `article.text` is read here and never returned. It leaves scope with this
    # function, which is what makes "extraction happens in memory" true.
    return FetchOutcome(
        analysis_id=payload.analysis_id,
        status="complete",
        final_url=response.final_url,
        http_status=response.status_code,
        content_sha256=hashlib.sha256(article.text.encode()).hexdigest(),
        characters=len(article.text),
        page_published=article.published,
        declared_language=article.declared_language,
        had_invisible=article.had_invisible,
        markers_removed=article.removed or {},
        completed_at=_now(),
    )
