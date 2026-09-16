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

Claims are the one exception, and they are the exception DATA.md names: "only
content hashes, **derived claims**, and aggregate statistics reach storage".
A claim is a span, not the document. That distinction is enforced by a length
cap rather than trusted - without one, a single claim of start=0 to
len(source) is valid and the whole article reaches Redis under the name of a
derived claim (D039).

Claim extraction runs here rather than in the API, because ARCHITECTURE.md puts
it first in the pipeline and because it needs the article text, which exists
only inside this function.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Final

import structlog
from pydantic import BaseModel, Field

from pipeline.claims.extract import extract_claims
from pipeline.claims.fallback import VERSION as FALLBACK_VERSION
from pipeline.claims.fallback import extract_claims_offline
from pipeline.claims.schema import Claim
from pipeline.errors import ProvenanceError, UrlRejectedError
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import fetch
from pipeline.net.robots import RobotsCache, RobotsWithSitemaps
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

ID_PREFIX: Final = "an_"
logger = structlog.get_logger(__name__)


class FetchRequest(BaseModel):
    """Queue payload. RULES.md: pydantic across every process boundary."""

    model_config = {"frozen": True}

    analysis_id: str
    url: str
    outlet_names: tuple[str, ...] = ()


class FetchOutcome(BaseModel):
    """What the worker returns, and therefore what reaches Redis.

    Every field is a derived fact, a hash, or a claim. There is no field
    capable of holding the article, which is the same enforcement strategy
    API.md uses to keep a verdict out of the response: make the schema unable
    to express it. Claims carry spans bounded by `max_claim_chars`, which is
    what keeps "derived claim" from meaning "the document" (D039).
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
    claims: list[Claim] = Field(default_factory=list)
    extractor_version: str | None = None
    extractor_error: str | None = None
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
    claim_extraction: bool = True,
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

    # Claim extraction, on text that is still in scope. A later component
    # needing the article fetches it again or runs here; nothing reads it back
    # out of a queue result, which is the constraint D032 chose and the
    # architecture had already chosen.
    claims: list[Claim] = []
    extractor_version: str | None = None
    extractor_error: str | None = None
    if claim_extraction:
        try:
            claims, claim_report = extract_claims(article.text)
            extractor_version = str(claim_report.get("extractor_version") or "")
        except ProvenanceError as exc:
            # D004: the pipeline degrades rather than fails. The rule based
            # extractor is worse and says so in its version string, which is
            # what makes the degradation visible on the card rather than
            # looking like an article that happened to contain no claims.
            extractor_error = exc.code
            claims = extract_claims_offline(article.text)
            extractor_version = FALLBACK_VERSION
            logger.info("extraction_degraded", code=exc.code, claims=len(claims))

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
        claims=claims,
        extractor_version=extractor_version,
        extractor_error=extractor_error,
        completed_at=_now(),
    )
