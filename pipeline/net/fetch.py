"""The SSRF guarded HTTP client (SECURITY.md control 1).

Every control in that section is implemented here, and none of them is
optional. The ordering matters: scheme, then port, then explicit resolution,
then per address classification, then a connection pinned to an address that
already passed. Validating a name and then reconnecting by name is the standard
mistake, and it is what the pinning below exists to prevent.

The DNS rebinding window is closed by never resolving twice. The address that
was validated is the address dialled, because the request is issued against the
address literal while the original hostname is carried in the Host header and
in the TLS SNI extension, so certificate verification still runs against the
name the caller asked for.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

import httpx

from pipeline.errors import (
    ContentTooLargeError,
    FetchTimeoutError,
    UnsupportedContentTypeError,
    UnsupportedSchemeError,
    UpstreamError,
    UrlRejectedError,
)
from pipeline.net.addresses import IPAddress, rejection_reason
from pipeline.net.config import FetchSettings

Resolver = Callable[[str, int], Sequence[str]]

_DEFAULT_PORTS = {"http": 80, "https": 443}
_TEXTUAL_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


@dataclass(frozen=True)
class FetchResult:
    """A successfully fetched document. Raw HTML is never persisted from here."""

    final_url: str
    status_code: int
    content_type: str
    text: str


def system_resolver(host: str, port: int) -> Sequence[str]:
    """Resolve a hostname to every address it offers, A and AAAA alike.

    Every returned address is checked. Validating only the first is a bypass:
    a name may offer one public address and one internal one, and which is
    used is not the validator's choice.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejectedError("The submitted URL could not be resolved.") from exc
    return [str(info[4][0]) for info in infos]


def validate_url(url: str, settings: FetchSettings) -> tuple[str, str, int]:
    """Check scheme and port, returning (scheme, host, port).

    Runs before resolution, so a rejected scheme never reaches DNS.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in settings.allowed_schemes:
        raise UnsupportedSchemeError("Only http and https URLs are accepted.")

    host = parts.hostname
    if not host:
        raise UrlRejectedError("The submitted URL has no host.")

    try:
        port = parts.port or _DEFAULT_PORTS[scheme]
    except ValueError as exc:  # non numeric port
        raise UrlRejectedError("The submitted URL has an invalid port.") from exc
    if port not in settings.allowed_ports:
        raise UrlRejectedError("The submitted URL uses a disallowed port.")

    return scheme, host, port


def resolve_and_validate(
    host: str, port: int, settings: FetchSettings, resolver: Resolver
) -> IPAddress:
    """Resolve a host and return the first address that passes every check.

    An address literal in the URL takes the same path, so ``http://127.0.0.1``
    and a name resolving to 127.0.0.1 are rejected identically.
    """
    del settings  # limits live elsewhere; kept for a uniform signature
    candidates = resolver(host, port)
    if not candidates:
        raise UrlRejectedError("The submitted URL could not be resolved.")

    for literal in candidates:
        addr = ip_address(literal.split("%")[0])  # strip any zone index
        if rejection_reason(addr) is None:
            return addr

    # Deliberately does not say which category matched. Telling a prober how
    # their address was classified is free reconnaissance.
    raise UrlRejectedError("The submitted URL resolves to a non public address.")


def _host_header(host: str, port: int, scheme: str) -> str:
    return host if port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"


def _pinned_url(scheme: str, addr: IPAddress, port: int, parts: tuple[str, ...]) -> str:
    literal = f"[{addr}]" if addr.version == 6 else str(addr)
    netloc = literal if port == _DEFAULT_PORTS[scheme] else f"{literal}:{port}"
    return urlunsplit((scheme, netloc, parts[2], parts[3], ""))


def fetch(
    url: str,
    *,
    settings: FetchSettings | None = None,
    client: httpx.Client | None = None,
    resolver: Resolver | None = None,
) -> FetchResult:
    """Fetch a URL under every control in SECURITY.md control 1.

    ``resolver`` and ``client`` are injectable so the adversarial suite can
    drive rebinding and redirect cases without network access.
    """
    settings = settings or FetchSettings()
    resolver = resolver or system_resolver

    owns_client = client is None
    if client is None:
        client = httpx.Client(
            follow_redirects=False,  # followed manually, revalidated per hop
            timeout=httpx.Timeout(settings.total_timeout_s, connect=settings.connect_timeout_s),
            headers={"User-Agent": settings.user_agent},
        )

    try:
        current = url
        for _hop in range(settings.max_redirects + 1):
            scheme, host, port = validate_url(current, settings)
            addr = resolve_and_validate(host, port, settings, resolver)

            parts = urlsplit(current)
            pinned = _pinned_url(scheme, addr, port, tuple(parts))
            headers = {"Host": _host_header(host, port, scheme)}

            try:
                with client.stream(
                    "GET",
                    pinned,
                    headers=headers,
                    extensions={"sni_hostname": host},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise UpstreamError("Redirect without a location header.")
                        # Resolved against the *pre-pinning* URL, so a relative
                        # redirect cannot smuggle the address literal onward.
                        current = str(httpx.URL(current).join(location))
                        continue

                    return _read_body(response, current, settings)
            except httpx.TimeoutException as exc:
                raise FetchTimeoutError("The remote host did not respond in time.") from exc
            except httpx.HTTPError as exc:
                raise UpstreamError("The remote host could not be reached.") from exc

        raise UrlRejectedError("The submitted URL exceeded the redirect limit.")
    finally:
        if owns_client:
            client.close()


def _read_body(response: httpx.Response, final_url: str, settings: FetchSettings) -> FetchResult:
    """Stream the body, enforcing both caps mid stream.

    Two separate limits, because they defend against two different attacks.
    ``num_bytes_downloaded`` counts bytes actually taken off the wire, which
    catches an oversized response. The accumulated decoded length catches a
    small compressed body that expands to gigabytes. Content-Length is never
    consulted; it is attacker controlled.

    One trap, recorded because it silently disables the wire cap. httpx only
    increments ``num_bytes_downloaded`` when the response is genuinely
    streamed; for a response whose body was already read it yields from an
    internal buffer and the counter stays at zero. The cap is therefore live in
    production but invisible to any test whose fixture builds a response from a
    bytes literal. The adversarial suite streams its fixtures for exactly this
    reason, and asserts the abort happens part way through.
    """
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith(_TEXTUAL_TYPES):
        raise UnsupportedContentTypeError("The fetched resource is not HTML or text.")

    chunks: list[bytes] = []
    decoded = 0
    for chunk in response.iter_bytes():
        decoded += len(chunk)
        if decoded > settings.max_decompressed_bytes:
            raise ContentTooLargeError("The fetched resource exceeded the size limit.")
        if response.num_bytes_downloaded > settings.max_response_bytes:
            raise ContentTooLargeError("The fetched resource exceeded the size limit.")
        chunks.append(chunk)

    body = b"".join(chunks)
    return FetchResult(
        final_url=final_url,
        status_code=response.status_code,
        content_type=content_type,
        text=body.decode(response.encoding or "utf-8", errors="replace"),
    )
