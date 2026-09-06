"""Fetch level adversarial cases for SECURITY.md control 1.

No network. The resolver and the transport are both injected, which is what
makes DNS rebinding and redirect chains testable at all: a rebinding test needs
a resolver whose answer changes between calls, and that cannot be arranged
against real DNS.
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence

import httpx
import pytest
from pipeline.errors import (
    ContentTooLargeError,
    FetchTimeoutError,
    UnsupportedSchemeError,
    UrlRejectedError,
)
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import fetch

pytestmark = pytest.mark.adversarial

PUBLIC = "93.184.216.34"
INTERNAL = "169.254.169.254"


def const_resolver(*addrs: str) -> object:
    def _r(host: str, port: int) -> Sequence[str]:
        return list(addrs)

    return _r


def ok_transport(body: bytes = b"<html>hi</html>", **kw: object) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/html"}, **kw)

    return httpx.MockTransport(handler)


def client_for(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport, follow_redirects=False)


# --------------------------------------------------------------- scheme, port


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/1", "ftp://x/a", "dict://x/"])
def test_scheme_allowlist(url: str) -> None:
    with pytest.raises(UnsupportedSchemeError):
        fetch(url, resolver=const_resolver(PUBLIC))


@pytest.mark.parametrize("port", [22, 25, 8080, 6379, 11211])
def test_port_allowlist(port: int) -> None:
    with pytest.raises(UrlRejectedError):
        fetch(f"http://example.com:{port}/", resolver=const_resolver(PUBLIC))


# ------------------------------------------------------------------ addresses


@pytest.mark.parametrize(
    "addr", [INTERNAL, "127.0.0.1", "10.0.0.5", "::1", "::ffff:169.254.169.254", "100.64.0.1"]
)
def test_internal_resolution_is_rejected(addr: str) -> None:
    with pytest.raises(UrlRejectedError):
        fetch("http://evil.example/", resolver=const_resolver(addr))


def test_all_addresses_checked_not_just_the_first() -> None:
    """A name offering a public and an internal address must not be trusted
    because its first answer was clean."""
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.url.host)
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/html"})

    fetch(
        "http://mixed.example/",
        client=client_for(httpx.MockTransport(handler)),
        resolver=const_resolver(PUBLIC, INTERNAL),
    )
    assert used == [PUBLIC], "must connect to the validated address only"


# ------------------------------------------------------------- pinning + host


def test_connects_to_ip_and_sends_original_host() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host_hdr"] = request.headers["host"]
        seen["url_host"] = request.url.host
        seen["sni"] = str(request.extensions.get("sni_hostname"))
        return httpx.Response(200, content=b"<p>ok</p>", headers={"content-type": "text/html"})

    fetch(
        "https://example.com/article",
        client=client_for(httpx.MockTransport(handler)),
        resolver=const_resolver(PUBLIC),
    )
    assert seen["url_host"] == PUBLIC, "connection must target the validated address"
    assert seen["host_hdr"] == "example.com", "Host header must carry the original name"
    assert seen["sni"] == "example.com", "TLS must verify against the original name"


def test_dns_rebinding_cannot_open_a_window() -> None:
    """Resolver flips to an internal address after the first answer.

    The guard resolves once per hop and dials the address it validated, so the
    second answer is never consulted. Validating a name and reconnecting by
    name is the bug this asserts against.
    """
    answers = iter([[PUBLIC], [INTERNAL], [INTERNAL]])
    calls: list[str] = []

    def flipping(host: str, port: int) -> Sequence[str]:
        return next(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/html"})

    fetch(
        "http://rebind.example/",
        client=client_for(httpx.MockTransport(handler)),
        resolver=flipping,
    )
    assert calls == [PUBLIC], "the rebound address must never be dialled"


# -------------------------------------------------------------------- redirects


def test_redirect_chain_to_internal_address_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == PUBLIC:
            return httpx.Response(302, headers={"location": "http://metadata.internal/latest"})
        return httpx.Response(200, content=b"secrets", headers={"content-type": "text/html"})

    def resolver(host: str, port: int) -> Sequence[str]:
        return [INTERNAL] if host == "metadata.internal" else [PUBLIC]

    with pytest.raises(UrlRejectedError):
        fetch(
            "http://start.example/",
            client=client_for(httpx.MockTransport(handler)),
            resolver=resolver,
        )


def test_redirect_hop_limit_enforced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://example.com/next"})

    with pytest.raises(UrlRejectedError, match="redirect"):
        fetch(
            "http://example.com/",
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )


def test_redirect_to_disallowed_scheme_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "file:///etc/passwd"})

    with pytest.raises(UnsupportedSchemeError):
        fetch(
            "http://example.com/",
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )


# ------------------------------------------------------------------ size caps


def test_oversized_response_aborts_mid_stream() -> None:
    """The cap must abort while streaming, not after reading the whole body.

    The mock streams so the real code path runs: httpx only counts wire bytes
    when the response is genuinely streamed, and a mock built from a bytes
    literal is pre-read, which would silently bypass this check.
    """
    produced = 0

    def body() -> object:
        nonlocal produced
        for _ in range(50):
            produced += 1
            yield b"A" * 10_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/html"})

    settings = FetchSettings(max_response_bytes=1024, max_decompressed_bytes=10 * 1024 * 1024)
    with pytest.raises(ContentTooLargeError):
        fetch(
            "http://example.com/",
            settings=settings,
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )
    assert produced < 5, f"aborted only after {produced} chunks; cap is not mid stream"


def test_decompression_bomb_aborts() -> None:
    """A small body on the wire that expands far beyond the cap."""
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024))
    assert len(bomb) < 50_000, "the bomb must be small on the wire for this to mean anything"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=bomb,
            headers={"content-type": "text/html", "content-encoding": "gzip"},
        )

    settings = FetchSettings(max_decompressed_bytes=64 * 1024, max_response_bytes=10 * 1024 * 1024)
    with pytest.raises(ContentTooLargeError):
        fetch(
            "http://example.com/",
            settings=settings,
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )


def test_content_length_header_is_not_trusted() -> None:
    """A lying Content-Length must not let an oversized body through."""

    def body() -> object:
        for _ in range(40):
            yield b"B" * 10_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body(),
            headers={"content-type": "text/html", "content-length": "10"},
        )

    with pytest.raises(ContentTooLargeError):
        fetch(
            "http://example.com/",
            settings=FetchSettings(max_response_bytes=2048),
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )


# -------------------------------------------------------------------- timeouts


def test_slow_response_raises_typed_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(FetchTimeoutError):
        fetch(
            "http://example.com/",
            client=client_for(httpx.MockTransport(handler)),
            resolver=const_resolver(PUBLIC),
        )


# --------------------------------------------------------------- happy path


def test_successful_fetch_returns_text() -> None:
    result = fetch(
        "http://example.com/a",
        client=client_for(ok_transport(b"<html>body</html>")),
        resolver=const_resolver(PUBLIC),
    )
    assert result.status_code == 200
    assert "body" in result.text
    assert result.final_url == "http://example.com/a", "reports the name, not the pinned IP"
