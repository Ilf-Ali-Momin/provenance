"""Address classification for the SSRF guard (SECURITY.md control 1).

Deny by default. An address is forwarded only if it is globally routable and
matches none of the rejected categories. Anything unrecognised is rejected,
because the failure mode of allowing an unclassified address is credential
disclosure and the failure mode of rejecting one is a fetch that does not
happen.

Two categories are handled explicitly rather than delegated to the standard
library, because measurement showed the library does not cover them:

* Carrier grade NAT, 100.64.0.0/10, reports ``is_private is False`` on
  CPython 3.12. SECURITY.md names it, so it is checked by range.
* IPv6 transition addresses embed an IPv4 address that the outer address's
  flags say nothing about. ``64:ff9b::7f00:1`` wraps 127.0.0.1 and reports
  ``is_global is True``. Every embedding is unwrapped and the inner address
  re-checked, which is the ``::ffff:169.254.169.254`` case SECURITY.md calls
  out and three others it does not.
"""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address
from typing import Final

IPAddress = IPv4Address | IPv6Address

# Ranges the standard library does not flag, or flags inconsistently.
_EXTRA_V4_DENY: Final[tuple[IPv4Network, ...]] = (
    IPv4Network("100.64.0.0/10"),  # carrier grade NAT (RFC 6598)
    IPv4Network("192.0.0.0/24"),  # IETF protocol assignments
    IPv4Network("192.0.2.0/24"),  # TEST-NET-1
    IPv4Network("198.18.0.0/15"),  # benchmarking
    IPv4Network("198.51.100.0/24"),  # TEST-NET-2
    IPv4Network("203.0.113.0/24"),  # TEST-NET-3
)

_NAT64: Final[IPv6Network] = IPv6Network("64:ff9b::/96")


def _embedded_v4(addr: IPv6Address) -> IPv4Address | None:
    """Return the IPv4 address an IPv6 address carries, if it carries one."""
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr.sixtofour is not None:
        return addr.sixtofour
    if addr.teredo is not None:
        return addr.teredo[1]
    if addr in _NAT64:
        return IPv4Address(int(addr) & 0xFFFFFFFF)
    # ::a.b.c.d, deprecated IPv4 compatible form. Excludes :: and ::1.
    packed = int(addr)
    if 0 < packed <= 0xFFFFFFFF and packed != 1:
        return IPv4Address(packed)
    return None


def rejection_reason(addr: IPAddress) -> str | None:
    """Why this address must not be connected to, or None if it is allowed.

    The reason is a stable identifier suitable for logging and for tests. It
    is never surfaced to a caller, because telling a prober which category
    their address fell into is free reconnaissance.
    """
    if isinstance(addr, IPv6Address):
        inner = _embedded_v4(addr)
        if inner is not None:
            # The embedding itself is not the problem; what it wraps is.
            return rejection_reason(inner) or "ipv6_embedded_ipv4"

    if addr.is_unspecified:
        return "unspecified"
    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link_local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_reserved:
        return "reserved"
    if addr.is_private:
        return "private"
    if isinstance(addr, IPv4Address) and any(addr in net for net in _EXTRA_V4_DENY):
        return "special_use"
    if not addr.is_global:
        return "not_globally_routable"
    return None


def is_forwardable(addr: IPAddress) -> bool:
    """True only for addresses that survive every check above."""
    return rejection_reason(addr) is None


def parse_and_check(literal: str) -> tuple[IPAddress, str | None]:
    """Parse an address literal and classify it in one step."""
    addr = ip_address(literal)
    return addr, rejection_reason(addr)
