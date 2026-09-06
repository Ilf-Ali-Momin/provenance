"""Address classification cases for SECURITY.md control 1.

One case per blocked range named in that control, plus the IPv6 forms it calls
out and three embedding tricks it does not. Pure logic, no network.
"""

from __future__ import annotations

from ipaddress import ip_address

import pytest
from pipeline.net.addresses import is_forwardable, rejection_reason

pytestmark = pytest.mark.adversarial

BLOCKED = [
    ("127.0.0.1", "loopback"),
    ("127.16.44.9", "loopback"),
    ("0.0.0.0", "unspecified"),  # noqa: S104 — test data, not a bind address
    ("10.0.0.1", "private"),
    ("172.16.31.5", "private"),
    ("192.168.1.1", "private"),
    ("169.254.169.254", "link_local"),  # cloud metadata, the whole ballgame
    # CPython does not flag CGNAT as private, so our explicit range catches it.
    ("100.64.0.1", "special_use"),  # carrier grade NAT
    ("224.0.0.1", "multicast"),
    ("240.0.0.1", "reserved"),
    # CPython *does* flag these two as private, so that branch wins first. The
    # explicit ranges below them are belt and braces, not the active check.
    ("198.18.0.1", "private"),  # benchmarking
    ("192.0.2.5", "private"),  # TEST-NET-1
    ("::1", "loopback"),
    ("fe80::1", "link_local"),
    ("fc00::1", "private"),
    ("::", "unspecified"),
    ("ff02::1", "multicast"),
]

# Every one of these wraps an internal IPv4 address inside an IPv6 form.
EMBEDDED = [
    "::ffff:169.254.169.254",  # IPv4 mapped, named explicitly in SECURITY.md
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "64:ff9b::7f00:1",  # NAT64 wrapping 127.0.0.1 — reports is_global True
    "2002:c0a8:0101::1",  # 6to4 wrapping 192.168.1.1
    "::127.0.0.1",  # deprecated IPv4 compatible
]

ALLOWED = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2001:4860:4860::8888"]


@pytest.mark.parametrize(("literal", "reason"), BLOCKED)
def test_blocked_ranges(literal: str, reason: str) -> None:
    assert rejection_reason(ip_address(literal)) == reason
    assert not is_forwardable(ip_address(literal))


@pytest.mark.parametrize("literal", EMBEDDED)
def test_ipv6_embedded_ipv4_is_unwrapped(literal: str) -> None:
    # The outer address may look routable; what it wraps decides.
    assert not is_forwardable(ip_address(literal)), f"{literal} must not be forwardable"


@pytest.mark.parametrize("literal", ALLOWED)
def test_public_addresses_pass(literal: str) -> None:
    assert rejection_reason(ip_address(literal)) is None


def test_nat64_would_pass_a_naive_is_global_check() -> None:
    """Guards the specific bypass that motivated explicit unwrapping.

    If this ever fails, the standard library changed and the NAT64 branch may
    look redundant. It is not; this test is why it exists.
    """
    naive = ip_address("64:ff9b::7f00:1")
    assert naive.is_global is True, "stdlib no longer reports NAT64 as global"
    assert not is_forwardable(naive), "our guard must still reject it"
