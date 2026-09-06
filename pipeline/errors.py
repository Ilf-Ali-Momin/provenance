"""Typed exceptions, mapped to the API error codes in context/API.md.

RULES.md requires one place where exceptions map to codes. That place is
`CODE`, below: an exception without a code cannot be raised across a boundary,
and the mapping is asserted by a test rather than maintained by discipline.
"""

from __future__ import annotations


class ProvenanceError(Exception):
    """Base for every error this project raises deliberately."""

    code: str = "internal_error"
    http_status: int = 500


class UrlRejectedError(ProvenanceError):
    """The URL failed SSRF validation (SECURITY.md control 1)."""

    code = "url_rejected"
    http_status = 400


class UnsupportedSchemeError(UrlRejectedError):
    code = "unsupported_scheme"
    http_status = 400


class FetchTimeoutError(ProvenanceError):
    code = "fetch_timeout"
    http_status = 504


class ContentTooLargeError(ProvenanceError):
    code = "content_too_large"
    http_status = 400


class UnsupportedContentTypeError(ProvenanceError):
    code = "unsupported_content_type"
    http_status = 415


class UpstreamError(ProvenanceError):
    code = "upstream_error"
    http_status = 502
