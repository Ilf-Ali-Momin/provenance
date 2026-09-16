"""Claim extraction over an OpenAI compatible endpoint, currently Groq.

D004 limits the LLM to this one task. ARCHITECTURE.md says why it is the right
tool here: the task is linguistic, low stakes, and the output is structurally
validated before use.

**Offset validation is where control 2 is enforced**, not in the prompt. Every
returned offset must resolve to a span in the source text; a claim whose span
does not resolve is dropped. The claim's text is then *sliced from the source*
using those offsets. The model never supplies a character of the output.

The HTTP call uses httpx directly rather than an SDK. httpx is already a
dependency, the request is one JSON POST to a fixed, configured host, and the
SDK would add a dependency to save very little. The endpoint is not attacker
controlled, so this does not go through the SSRF guarded client, which exists
for URLs a user submits.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Final, Protocol

import httpx
import structlog

from pipeline.claims.config import ExtractorSettings
from pipeline.claims.detect import detect
from pipeline.claims.prompt import build_messages, load_prompt, new_delimiter
from pipeline.claims.schema import Claim, ClaimType, RawExtraction, claim_id
from pipeline.errors import ProvenanceError

logger = structlog.get_logger(__name__)
HTTP_TOO_MANY_REQUESTS: Final = 429
HTTP_SERVER_ERROR: Final = 500

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "type"],
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                    "type": {"type": "string", "enum": [t.value for t in ClaimType]},
                },
            },
        }
    },
}


class ExtractorUnavailableError(ProvenanceError):
    code = "extractor_unavailable"
    http_status = 503


class Completion(Protocol):
    """The one call this module makes. A protocol so tests need no network."""

    def __call__(self, messages: list[dict[str, str]], settings: ExtractorSettings) -> str: ...


def groq_completion(messages: list[dict[str, str]], settings: ExtractorSettings) -> str:
    """One JSON POST. No tools, no function calling, no network reach (control 2)."""
    if not settings.api_key:
        msg = "No API key configured for the claim extractor."
        raise ExtractorUnavailableError(msg)
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claims", "schema": RESPONSE_SCHEMA, "strict": True},
        },
    }
    if settings.reasoning_effort:
        body["reasoning_effort"] = settings.reasoning_effort

    for attempt in range(settings.max_retries + 1):
        try:
            return _post_once(body, settings)
        except _RetryableError as transient:
            # Two different transient failures turned up in one evaluation
            # sweep: a 429 with Retry-After, and a bare transport drop
            # ("Server disconnected without sending a response"). The first
            # version retried only the 429, so the second still lost the run.
            # Anything the provider can recover from on a second attempt is
            # retried here; a 4xx that is not 429 is the caller's fault and is
            # not.
            if attempt == settings.max_retries:
                raise ExtractorUnavailableError(f"{transient.reason} after retries") from transient
            delay = transient.retry_after or settings.retry_backoff_s * (2**attempt)
            logger.info(
                "retrying_transient",
                reason=transient.reason,
                seconds=round(delay, 2),
                attempt=attempt + 1,
            )
            time.sleep(delay)
    raise ExtractorUnavailableError("retries exhausted")


class _RetryableError(Exception):
    """A failure a second attempt might not hit: 429, 5xx, or a transport drop."""

    def __init__(self, reason: str, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _post_once(body: dict[str, Any], settings: ExtractorSettings) -> str:
    if settings.min_interval_s:
        time.sleep(settings.min_interval_s)
    try:
        response = httpx.post(
            f"{settings.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=body,
            timeout=settings.timeout_s,
        )
        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            header = response.headers.get("Retry-After", "")
            raise _RetryableError(
                "HTTP 429 (rate_limit_exceeded)",
                float(header) if header.replace(".", "").isdigit() else None,
            )
        if response.status_code >= HTTP_SERVER_ERROR:
            raise _RetryableError(f"HTTP {response.status_code}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Carry the status and the provider's error code. Not the body: it can
        # echo the request, and the request contains the article.
        code = ""
        try:
            code = str(exc.response.json().get("error", {}).get("code", ""))
        except ValueError:
            code = ""
        raise ExtractorUnavailableError(
            f"HTTP {exc.response.status_code}{f' ({code})' if code else ''}"
        ) from exc
    except httpx.TransportError as exc:
        # Connection reset, disconnect without a response, read timeout. The
        # request may never have reached the provider.
        raise _RetryableError(type(exc).__name__) from exc
    except httpx.HTTPError as exc:
        raise ExtractorUnavailableError(f"{type(exc).__name__}") from exc
    content = response.json()["choices"][0]["message"]["content"]
    return str(content)


def validate_offsets(raw: RawExtraction, source: str, version: str) -> tuple[list[Claim], int]:
    """Keep claims whose spans resolve; drop the rest. Returns (claims, dropped).

    This is the boundary an injection would have to escape and cannot. The text
    is taken from `source`, so whatever the model returned, the output contains
    only characters this project already had.
    """
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    kept: list[Claim] = []
    dropped = 0
    for item in raw.claims:
        if item.end > len(source):
            dropped += 1
            continue
        raw_slice = source[item.start : item.end]
        text = raw_slice.strip()
        if not text:
            dropped += 1
            continue
        # Move the offsets to the stripped bounds rather than keeping the
        # model's and stripping the text. API.md gives each claim a span, and a
        # client highlighting it should highlight exactly the claim: offsets
        # that name more than the text is a mismatch a reader would see. It
        # also keeps the invariant exact - source[start:end] == text - which is
        # what the injection suite tests to prove nothing was authored.
        start = item.start + (len(raw_slice) - len(raw_slice.lstrip()))
        end = start + len(text)
        kept.append(
            Claim(
                claim_id=claim_id(source_sha, start, end),
                text=text,
                start=start,
                end=end,
                type=item.type,
                extractor_version=version,
            )
        )
    return kept, dropped


def extract_claims(
    source: str,
    settings: ExtractorSettings | None = None,
    completion: Completion | None = None,
) -> tuple[list[Claim], dict[str, object]]:
    """Extract claims from one passage. Returns the claims and a provenance record."""
    settings = settings or ExtractorSettings()
    completion = completion or groq_completion
    prompt_text, version = load_prompt(settings.prompt_name)
    delimiter = new_delimiter()

    # Fourth layer: recorded, never acted on. A news article about AI trips
    # these patterns, and refusing to analyse it would be the wrong failure.
    signals = detect(source)
    if signals:
        logger.info("injection_signals", categories=sorted(signals), version=version)

    body = completion(build_messages(source, prompt_text, delimiter), settings)
    try:
        raw = RawExtraction.model_validate_json(body)
    except ValueError as exc:
        # A model that cannot produce the schema produces nothing. There is no
        # salvage path that does not involve trusting free text.
        logger.warning("extraction_unparseable", version=version, detail=type(exc).__name__)
        return [], {"extractor_version": version, "claims": 0, "dropped": 0, "unparseable": True}

    claims, dropped = validate_offsets(raw, source, version)
    if dropped:
        logger.info("offsets_dropped", count=dropped, version=version)
    return claims, {
        "extractor_version": version,
        "claims": len(claims),
        "dropped": dropped,
        "injection_signals": signals,
    }
