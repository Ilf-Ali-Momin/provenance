"""RDAP lookups, and the personal data they return that this never keeps.

ARCHITECTURE.md lists "domain registration age from RDAP" among the registry's
provenanced fields. RDAP is the protocol that replaced WHOIS: JSON over HTTPS,
with IANA publishing a bootstrap file mapping each TLD to the server that
answers for it.

**The thing to be careful about is not the protocol, it is the payload.** An
RDAP response for a domain can carry registrant name, postal address, email and
telephone in its `entities`, as vCard arrays. Most registries redact these
post-GDPR, some do not, and which is which is not something to rely on. DATA.md
puts this project in scope as a controller and requires data minimisation, so
this module extracts exactly two things and never holds the rest:

* the registration date, from the `registration` event;
* the registrar's name, which is an organisation rather than a person.

`extract` takes the parsed response and returns observations. It does not return
the response, and no caller is given one: `lookup` parses, extracts, and lets
the body go out of scope. The same shape as `FetchOutcome` having no field for
article text and `CirculationStats` having no field for a handle - the unwanted
data has nowhere to go rather than being carefully not written down.

**Registrar name is kept and a registrant name is not**, which is a line worth
stating rather than assuming. A registrar is a company performing a commercial
function, named in the public record for that reason. A registrant may be a
person, and "who owns this domain" is exactly the question that turns a registry
into a dossier. D007 already forbids composing observations into a judgement
about a publisher; keeping less makes that harder to undo later.

**Bootstrap, not a hardcoded server list.** IANA's `dns.json` maps TLDs to RDAP
base URLs and changes as registries move. The file is fetched through the same
guarded client as everything else, because its contents are URLs this code then
requests, and a bootstrap entry is as much an attacker-influenced URL as a link
in an article.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from app.limits import registrable_domain

from pipeline.errors import ProvenanceError
from pipeline.net.config import FetchSettings
from pipeline.registry.schema import Observation, ObservationKind, Source

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

BOOTSTRAP_URL: Final = "https://data.iana.org/rdap/dns.json"
# RDAP's own media type, plus the JSON some servers answer with instead.
RDAP_CONTENT_TYPES: Final = frozenset(
    {"application/rdap+json", "application/json", "application/rdap+json; charset=utf-8"}
)
# Registrar names are short. A cap stops a hostile server making the registry
# store a paragraph under the name of an organisation.
MAX_REGISTRAR_CHARS: Final = 120


class RdapUnavailableError(ProvenanceError):
    code = "rdap_unavailable"


def bootstrap_settings(settings: FetchSettings | None = None) -> FetchSettings:
    """Fetch settings that accept RDAP's media type and nothing more.

    Widened explicitly here rather than loosening `FetchSettings` for everyone:
    the article path is narrow because a fetch returning a PDF or an executable
    is not something extraction should ever see, and that reason does not stop
    applying because RDAP exists. Same approach feed discovery takes.
    """
    base = settings or FetchSettings()
    return base.model_copy(update={"allowed_content_types": frozenset(RDAP_CONTENT_TYPES)})


def server_for(tld: str, bootstrap: dict[str, Any]) -> str | None:
    """The RDAP base URL serving a TLD, from IANA's bootstrap document.

    The file's `services` is a list of [tlds, urls] pairs. Returns the first
    HTTPS URL offered, since a registry publishing both should be reached over
    the one that is authenticated.
    """
    wanted = tld.lower().lstrip(".")
    for entry in bootstrap.get("services", []):
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        tlds, urls = entry
        if not isinstance(tlds, list) or not isinstance(urls, list):
            continue
        if wanted not in {str(t).lower() for t in tlds}:
            continue
        https = [str(u) for u in urls if str(u).lower().startswith("https://")]
        chosen = https[0] if https else None
        return chosen.rstrip("/") if chosen else None
    return None


def _registration_date(response: dict[str, Any]) -> datetime | None:
    """The `registration` event's date, if the server reported one."""
    for event in response.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("eventAction", "")).lower() != "registration":
            continue
        raw = str(event.get("eventDate", ""))
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _registrar_name(response: dict[str, Any]) -> str | None:
    """The registrar's organisation name, from the entity holding that role.

    Reads only the `fn` (formatted name) field of the vCard, and only for the
    entity whose role is `registrar`. Every other entity is skipped without
    being looked at, because that is where a registrant's name and address sit.
    """
    for entity in response.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        roles = {str(r).lower() for r in entity.get("roles", []) or []}
        if "registrar" not in roles:
            continue
        vcard = entity.get("vcardArray")
        if not (isinstance(vcard, list) and len(vcard) == 2 and isinstance(vcard[1], list)):
            continue
        for field in vcard[1]:
            if isinstance(field, list) and len(field) >= 4 and str(field[0]).lower() == "fn":
                name = str(field[3]).strip()
                return name[:MAX_REGISTRAR_CHARS] or None
    return None


def extract(response: dict[str, Any], source_url: str, observed_at: datetime) -> list[Observation]:
    """The two fields this project keeps, as dated observations.

    Takes the parsed response and returns observations. It does not return the
    response and nothing here stores it: registrant name, address, email and
    telephone are read past rather than kept.
    """
    found: list[Observation] = []
    registered = _registration_date(response)
    if registered is not None:
        found.append(
            Observation(
                kind=ObservationKind.REGISTERED_ON,
                value=registered.date(),
                source=Source.RDAP,
                source_url=source_url,
                observed_at=observed_at,
            )
        )
    registrar = _registrar_name(response)
    if registrar:
        found.append(
            Observation(
                kind=ObservationKind.REGISTRAR,
                value=registrar,
                source=Source.RDAP,
                source_url=source_url,
                observed_at=observed_at,
            )
        )
    return found


def lookup(
    url_or_domain: str,
    fetch_json: Callable[[str], dict[str, Any]],
    bootstrap: dict[str, Any],
    observed_at: datetime | None = None,
) -> list[Observation]:
    """Observations about one domain, or an empty list where nothing is known.

    `fetch_json` is injected so the caller owns the transport, its rate limits
    and its user agent, and so this is testable without a network - the same
    reason the Wikimedia source takes one.

    An empty list rather than an exception when a TLD has no RDAP server or the
    server reports nothing: "we looked and found nothing" is a legitimate answer
    the registry can hold, and `DomainRecord.latest` already distinguishes it
    from never having looked.
    """
    observed_at = observed_at or datetime.now(UTC)
    domain = registrable_domain(url_or_domain)
    if "." not in domain:
        return []
    tld = domain.rsplit(".", 1)[-1]

    base = server_for(tld, bootstrap)
    if not base:
        return []

    source_url = f"{base}/domain/{domain}"
    try:
        response = fetch_json(source_url)
    except ProvenanceError:
        raise
    except Exception as exc:
        msg = f"RDAP lookup for {domain} failed: {type(exc).__name__}"
        raise RdapUnavailableError(msg) from exc

    return extract(response, source_url, observed_at)


def parse_bootstrap(body: str) -> dict[str, Any]:
    """IANA's bootstrap document, parsed and checked for the shape used here."""
    try:
        document = json.loads(body)
    except ValueError as exc:
        msg = "IANA RDAP bootstrap is not JSON"
        raise RdapUnavailableError(msg) from exc
    if not isinstance(document, dict) or not isinstance(document.get("services"), list):
        msg = "IANA RDAP bootstrap has no services list"
        raise RdapUnavailableError(msg)
    return document
