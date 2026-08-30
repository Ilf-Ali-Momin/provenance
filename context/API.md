# API

Base path `/v1`. JSON only. All requests authenticated by `Authorization: Bearer <key>` except `/health`.

## Design commitments

The response schema has no field capable of holding a verdict. There is no `score`, no `verdict`, no `is_fake`, no `credibility`. This is enforced by the schema, not by convention, so a future change that wanted to add one would have to be a deliberate schema change visible in review. That is the point.

Every analyser section can independently be `available`, `abstained`, or `unavailable`, each with a reason. Clients must handle all three. A client that renders `abstained` as a neutral or passing result has misrepresented the system.

Asynchronous by default for anything involving a remote fetch. The URL path never blocks a request thread on an attacker controlled remote host.

## Endpoints

### `POST /v1/analyses`

Submit text or a URL for analysis.

```json
{
  "input": {
    "type": "text",
    "content": "The passage under analysis.",
    "context_url": "https://example.com/article"
  },
  "options": {
    "analysers": ["retrieval", "stance", "source", "circulation", "rhetoric"],
    "max_claims": 3
  }
}
```

`type` is `text` or `url`. For `url`, `content` is omitted and `context_url` carries the target. `context_url` on a text submission is optional and used only for source registry lookup; it is not fetched.

`content` maximum 50000 characters, rejected above that rather than truncated, since silent truncation produces analysis of a passage the user did not submit.

Idempotency via `Idempotency-Key` header, or automatically via normalised content hash when absent.

**202 Accepted** for the URL path, or when any analyser is expected to exceed the synchronous budget.

```json
{
  "analysis_id": "an_01J8X...",
  "status": "processing",
  "poll_url": "/v1/analyses/an_01J8X...",
  "stream_url": "/v1/analyses/an_01J8X.../events"
}
```

**200 OK** for cached or fast synchronous results, returning the full analysis object below.

### `GET /v1/analyses/{id}`

Returns the analysis object. `status` is `processing`, `complete`, or `failed`.

### `GET /v1/analyses/{id}/events`

Server sent events. Emits `analyser.complete` per analyser as each finishes, then `analysis.complete`. This is what makes the extension feel responsive: the source panel renders in 200ms while retrieval is still running.

Event types: `analyser.started`, `analyser.complete`, `analyser.failed`, `analysis.complete`, `error`. Heartbeat comment every 15 seconds. Reconnection via `Last-Event-ID`.

### `GET /v1/sources/{domain}`

Source registry lookup. Returns observations with provenance. Cacheable, long TTL.

### `GET /health`, `GET /health/ready`

Liveness and readiness. Readiness checks database, Redis, and model container socket.

## Analysis object

```json
{
  "analysis_id": "an_01J8X...",
  "status": "complete",
  "created_at": "2026-08-29T10:14:02Z",
  "input_hash": "sha256:...",
  "claims": [
    {
      "claim_id": "cl_01J8X...",
      "text": "The extracted claim as it appears in the source.",
      "span": { "start": 142, "end": 218 },
      "type": "factual_assertion",
      "extractor_version": "prompt_v4:a91f3c2"
    }
  ],
  "analysers": {
    "retrieval": {
      "state": "available",
      "per_claim": [
        {
          "claim_id": "cl_01J8X...",
          "passages": [
            {
              "passage_id": "ps_01J8X...",
              "text": "Retrieved passage excerpt.",
              "source_domain": "example.org",
              "published_at": "2026-08-20T00:00:00Z",
              "relevance": 0.81,
              "stance": {
                "label": "contradiction",
                "confidence": 0.74,
                "calibrated": true
              }
            }
          ],
          "distinct_domains": 4,
          "coverage": "found"
        }
      ]
    },
    "source": {
      "state": "available",
      "domain": "example.com",
      "observations": [
        {
          "field": "domain_registered",
          "value": "2024-11-03",
          "source": "rdap",
          "observed_at": "2026-08-01T00:00:00Z"
        },
        {
          "field": "masthead_present",
          "value": false,
          "source": "structured_extraction",
          "observed_at": "2026-08-01T00:00:00Z"
        }
      ]
    },
    "circulation": {
      "state": "available",
      "sampled_posts": 312,
      "account_age_median_days": 41,
      "account_age_p10_days": 9,
      "near_duplicate_ratio": 0.62,
      "platforms": ["bluesky", "reddit"],
      "sampled_at": "2026-08-29T10:13:40Z"
    },
    "rhetoric": {
      "state": "available",
      "note": "Observations about writing style. Not evidence about accuracy.",
      "flags": [
        {
          "label": "unsourced_attribution",
          "span": { "start": 88, "end": 141 },
          "confidence": 0.79
        }
      ]
    }
  },
  "notes": [
    {
      "code": "no_coverage_found",
      "message": "No related reporting was retrieved for one claim. This does not indicate the claim is false."
    }
  ]
}
```

## Abstention and unavailability

```json
{
  "state": "abstained",
  "reason": "below_calibrated_threshold",
  "detail": "Stance confidence fell below the threshold required for display."
}
```

```json
{
  "state": "unavailable",
  "reason": "deadline_exceeded",
  "detail": "Retrieval did not complete within the request budget."
}
```

Abstention reasons: `below_calibrated_threshold`, `insufficient_source_diversity`, `no_factual_assertions`, `out_of_supported_language`, `input_too_short`.

Unavailability reasons: `deadline_exceeded`, `upstream_error`, `rate_limited`, `disabled_by_request`.

The distinction matters. Abstained means the system ran and declined to report. Unavailable means it did not run. Collapsing them in the client is a misrepresentation.

## Errors

```json
{
  "error": {
    "code": "url_rejected",
    "message": "The submitted URL resolves to a non public address.",
    "request_id": "req_01J8X..."
  }
}
```

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Schema violation |
| 400 | `content_too_large` | Above 50000 characters |
| 400 | `url_rejected` | Failed SSRF validation |
| 400 | `unsupported_scheme` | Not http or https |
| 401 | `unauthenticated` | Missing or invalid key |
| 404 | `not_found` | Unknown analysis or domain |
| 409 | `idempotency_conflict` | Key reused with different body |
| 415 | `unsupported_content_type` | Fetched resource not HTML or text |
| 422 | `extraction_failed` | Could not extract usable text |
| 429 | `rate_limited` | With `Retry-After` |
| 429 | `domain_rate_limited` | Per target domain limit |
| 502 | `upstream_error` | External dependency failed |
| 503 | `capacity` | Circuit breaker open |
| 504 | `fetch_timeout` | Remote fetch exceeded budget |

Error codes are stable across versions. Messages are not, and clients must not parse them.

## Versioning

Path versioned. Within a version, additive changes only: new fields, new analysers, new enum members in non discriminating positions. Clients ignore unknown fields.

Breaking changes get `/v2` with `/v1` supported for 90 days and a `Sunset` header.

`analyser_versions` appears in every response, mapping each analyser to its model or prompt version, so a card can be reproduced and so a regression can be traced to a specific version change.

## Client obligations

Documented here because they are part of the security posture, not a suggestion.

Clients must not compute a summary score from the response. The schema is designed to make this awkward and doing it anyway defeats the design.

Clients must render abstention and unavailability visibly and distinguishably.

Clients must display the rhetoric `note` field verbatim wherever rhetoric flags appear.

Clients must not present the absence of retrieved evidence as a negative signal about the claim.
