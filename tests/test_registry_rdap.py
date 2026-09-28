"""RDAP extraction, and the personal data it walks past.

The protocol is easy. The care is in the payload: an RDAP response can carry
registrant name, postal address, email and telephone, and most of these tests
exist to pin that none of it survives.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from pipeline.registry.rdap import (
    RDAP_CONTENT_TYPES,
    RdapUnavailableError,
    bootstrap_settings,
    extract,
    lookup,
    parse_bootstrap,
    server_for,
)
from pipeline.registry.schema import ObservationKind, Source

WHEN = datetime(2026, 9, 28, tzinfo=UTC)
SOURCE = "https://rdap.example/domain/example.org"

BOOTSTRAP: dict[str, Any] = {
    "services": [
        [["org"], ["https://rdap.publicinterestregistry.org/rdap/"]],
        [["com", "net"], ["http://rdap.verisign.invalid/", "https://rdap.verisign.example/"]],
        [["nowhere"], ["http://only-insecure.example/"]],
    ]
}

# A response shaped like a real one, including the registrant entity that must
# not survive. The contact details here are invented and are the point.
RESPONSE: dict[str, Any] = {
    "objectClassName": "domain",
    "ldhName": "EXAMPLE.ORG",
    "events": [
        {"eventAction": "registration", "eventDate": "2019-03-14T09:20:00Z"},
        {"eventAction": "last changed", "eventDate": "2026-01-02T00:00:00Z"},
        {"eventAction": "expiration", "eventDate": "2027-03-14T09:20:00Z"},
    ],
    "entities": [
        {
            "roles": ["registrar"],
            "vcardArray": [
                "vcard",
                [["version", {}, "text", "4.0"], ["fn", {}, "text", "Example Registrar, Inc."]],
            ],
        },
        {
            "roles": ["registrant"],
            "vcardArray": [
                "vcard",
                [
                    ["fn", {}, "text", "Jane Q Registrant"],
                    ["email", {}, "text", "jane@personal.example"],
                    ["tel", {}, "text", "+44 7700 900123"],
                    ["adr", {}, "text", ["", "", "14 Rose Lane", "Bristol", "", "BS1 4XX", "GB"]],
                ],
            ],
        },
    ],
}

PERSONAL = ("Jane", "Registrant", "jane@personal.example", "7700 900123", "Rose Lane", "Bristol")


# --- what is kept ----------------------------------------------------------


def test_the_registration_date_becomes_a_dated_observation() -> None:
    found = {o.kind: o for o in extract(RESPONSE, SOURCE, WHEN)}
    registered = found[ObservationKind.REGISTERED_ON]
    assert registered.value == date(2019, 3, 14)
    assert registered.source is Source.RDAP
    assert registered.source_url == SOURCE
    assert registered.observed_at == WHEN


def test_the_registrar_is_kept_because_it_names_a_company() -> None:
    found = {o.kind: o for o in extract(RESPONSE, SOURCE, WHEN)}
    assert found[ObservationKind.REGISTRAR].value == "Example Registrar, Inc."


def test_only_the_registration_event_is_read() -> None:
    """Not `last changed`, not `expiration`."""
    found = {o.kind: o for o in extract(RESPONSE, SOURCE, WHEN)}
    assert found[ObservationKind.REGISTERED_ON].value == date(2019, 3, 14)


# --- what is not kept ------------------------------------------------------


def test_no_registrant_personal_data_survives_extraction() -> None:
    """The reason this module exists in the shape it does.

    DATA.md puts this project in scope as a controller and requires data
    minimisation. Registrant name, email, telephone and address are in the
    response and must be in none of the observations.
    """
    serialised = " ".join(
        f"{o.kind}{o.value}{o.source_url}{o.evidence or ''}"
        for o in extract(RESPONSE, SOURCE, WHEN)
    )
    for fragment in PERSONAL:
        assert fragment not in serialised


def test_a_registrant_name_is_not_mistaken_for_a_registrar() -> None:
    """Only the entity whose role is `registrar` is read at all."""
    only_registrant = {"entities": [RESPONSE["entities"][1]]}
    assert extract(only_registrant, SOURCE, WHEN) == []


def test_an_administrative_contact_is_ignored_too() -> None:
    response = {
        "entities": [
            {
                "roles": ["administrative"],
                "vcardArray": ["vcard", [["fn", {}, "text", "Some Person"]]],
            }
        ]
    }
    assert extract(response, SOURCE, WHEN) == []


def test_a_long_registrar_name_is_capped() -> None:
    """A hostile server should not get a paragraph into the registry under the
    name of an organisation."""
    response = {
        "entities": [
            {
                "roles": ["registrar"],
                "vcardArray": ["vcard", [["fn", {}, "text", "x" * 400]]],
            }
        ]
    }
    value = extract(response, SOURCE, WHEN)[0].value
    assert isinstance(value, str)
    assert len(value) == 120


# --- absent and malformed data ---------------------------------------------


def test_a_response_with_no_registration_event_yields_no_date() -> None:
    """ "We looked and the server said nothing" is a real answer."""
    response = {"events": [{"eventAction": "expiration", "eventDate": "2027-01-01T00:00:00Z"}]}
    assert [o.kind for o in extract(response, SOURCE, WHEN)] == []


def test_an_unparseable_event_date_is_dropped_rather_than_guessed() -> None:
    response = {"events": [{"eventAction": "registration", "eventDate": "sometime in 2019"}]}
    assert extract(response, SOURCE, WHEN) == []


def test_an_empty_response_yields_nothing() -> None:
    assert extract({}, SOURCE, WHEN) == []


def test_malformed_entities_do_not_raise() -> None:
    """A registry server is a third party and its output is untrusted input."""
    for response in (
        {"entities": "not a list"},
        {"entities": [None, 42, {"roles": None}]},
        {"entities": [{"roles": ["registrar"], "vcardArray": "wrong"}]},
        {"events": "not a list"},
    ):
        assert extract(response, SOURCE, WHEN) == []


# --- bootstrap -------------------------------------------------------------


def test_the_server_for_a_tld_comes_from_the_bootstrap_file() -> None:
    assert server_for("org", BOOTSTRAP) == "https://rdap.publicinterestregistry.org/rdap"


def test_https_is_preferred_over_http() -> None:
    """A registry offering both should be reached over the authenticated one."""
    assert server_for("com", BOOTSTRAP) == "https://rdap.verisign.example"


def test_a_tld_offering_only_http_is_refused() -> None:
    assert server_for("nowhere", BOOTSTRAP) is None


def test_an_unknown_tld_has_no_server() -> None:
    assert server_for("invalid", BOOTSTRAP) is None


def test_the_tld_lookup_is_case_insensitive() -> None:
    assert server_for("ORG", BOOTSTRAP) == server_for("org", BOOTSTRAP)


def test_a_malformed_bootstrap_entry_is_skipped() -> None:
    broken = {"services": [["org"], [["com"], ["https://ok.example/"]], "nonsense"]}
    assert server_for("com", broken) == "https://ok.example"


def test_bootstrap_json_is_validated() -> None:
    with pytest.raises(RdapUnavailableError, match="not JSON"):
        parse_bootstrap("<html>not json</html>")
    with pytest.raises(RdapUnavailableError, match="no services list"):
        parse_bootstrap('{"version": "1.0"}')


def test_rdap_content_types_are_widened_only_for_this_path() -> None:
    """The article path stays narrow: a fetch returning a PDF or an executable
    is not something extraction should ever see, and RDAP existing does not
    change that."""
    base = bootstrap_settings()
    assert base.allowed_content_types == frozenset(RDAP_CONTENT_TYPES)
    assert "text/html" not in base.allowed_content_types


# --- the lookup path -------------------------------------------------------


def test_a_lookup_returns_observations_for_a_known_tld() -> None:
    asked: list[str] = []

    def fetch_json(url: str) -> dict[str, Any]:
        asked.append(url)
        return RESPONSE

    found = lookup("https://www.example.org/article", fetch_json, BOOTSTRAP, WHEN)
    assert asked == ["https://rdap.publicinterestregistry.org/rdap/domain/example.org"]
    assert {o.kind for o in found} == {ObservationKind.REGISTERED_ON, ObservationKind.REGISTRAR}


def test_the_lookup_uses_the_registrable_domain_not_the_host() -> None:
    """ARCHITECTURE.md requires the public suffix list rather than splitting on
    dots, since subdomain handling is a common source of bugs and evasion."""
    asked: list[str] = []
    lookup(
        "https://news.sub.example.org/x",
        lambda url: (asked.append(url), RESPONSE)[1],
        BOOTSTRAP,
        WHEN,
    )
    assert asked[0].endswith("/domain/example.org")


def test_a_tld_with_no_rdap_server_returns_nothing_rather_than_raising() -> None:
    """The registry can hold "we looked and found nothing"; DomainRecord.latest
    already tells that apart from never having looked."""
    called = False

    def fetch_json(url: str) -> dict[str, Any]:
        nonlocal called
        called = True
        return RESPONSE

    assert lookup("https://example.invalid/", fetch_json, BOOTSTRAP, WHEN) == []
    assert not called, "no request should be made when no server is known"


def test_a_transport_failure_is_reported_not_swallowed() -> None:
    def fetch_json(url: str) -> dict[str, Any]:
        raise TimeoutError("connect timed out")

    with pytest.raises(RdapUnavailableError, match="TimeoutError"):
        lookup("https://example.org/", fetch_json, BOOTSTRAP, WHEN)


def test_the_error_names_no_response_content() -> None:
    """An error message is a place personal data leaks into logs."""

    def fetch_json(url: str) -> dict[str, Any]:
        raise ValueError("registrant jane@personal.example rejected the request")

    with pytest.raises(RdapUnavailableError) as raised:
        lookup("https://example.org/", fetch_json, BOOTSTRAP, WHEN)
    assert "jane@personal.example" not in str(raised.value)
