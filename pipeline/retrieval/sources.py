"""Wikimedia sources for the reference corpus.

DATA.md: the reference corpus is built "from openly licensed reference material
and from a snapshot of published fact checks accessed through their APIs where
terms permit". These are the openly licensed half. Fact check APIs are a
separate question per publisher and are not implemented here.

**Two wikis, chosen for different reasons.**

`en.wikinews.org` is news prose, which is the domain claims are made in, and is
the closer match to what the extractor produces. `en.wikipedia.org` carries the
background a news claim is usually checked against - the figures, the timelines,
the institutional detail that a news article assumes.

**Licences, read from each wiki's own API rather than from memory** (the
`rightsinfo` field of `action=query&meta=siteinfo`), on 2026-09-25:

* en.wikipedia.org: Creative Commons Attribution-Share Alike 4.0
* en.wikinews.org: Creative Commons Attribution 4.0

Both are `permitted_with_attribution`, so every document carries an attribution
string naming the article, the wiki and the licence, and D046's schema refuses
the document if it does not.

**Share-alike, stated rather than glossed.** Wikipedia's BY-SA obliges
derivative works to carry the same licence. A card quoting a retrieved passage
with attribution is a quotation, not a derivative of the article, so the
obligation does not propagate to the card's own text - but the passage itself
stays under BY-SA wherever it appears, which is why the licence is stored per
chunk and not inferred at render time.

**The user agent is a hard precondition, not a courtesy.** Wikimedia's user
agent policy requires a descriptive agent with a way to contact the operator,
and the project's default still points at `example.invalid` (D030's open row).
Fetching their API with a placeholder contact is a policy breach, so this module
refuses to start rather than doing it quietly. That is also what finally closes
D030: the obligation now blocks something.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

from pipeline.errors import ProvenanceError
from pipeline.net.config import FetchSettings
from pipeline.retrieval.corpus import ReferenceDocument

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

WIKIPEDIA: Final = "en.wikipedia.org"
WIKINEWS: Final = "en.wikinews.org"

# Read from each wiki's siteinfo rightsinfo on this date, not asserted from
# memory. Re-read before a rebuild: a wiki that relicenses invalidates every
# attribution string already in the index.
LICENCES: Final = {
    WIKIPEDIA: (
        "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
        "Wikipedia",
    ),
    WIKINEWS: (
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        "Wikinews",
    ),
}
LICENCES_READ_ON: Final = date(2026, 9, 25)

# Wikimedia asks for a contact in the agent string, and the requirement is that
# it be *reachable*. Two families of string cannot be, and neither Wikimedia's
# filter nor a first version of this guard caught the second.
#
# Reserved names, which by RFC 2606 and RFC 6761 never resolve to a host anyone
# operates. A complaint sent there is discarded by the DNS, not by a person.
RESERVED_SUFFIXES: Final = (".invalid", ".test", ".example", ".localhost")
# Template text left in place. The guard's first version listed only
# `example.*`, so `https://your-site/bot; you@your-domain` - the placeholder
# from this project's own instructions - passed it, and Wikimedia answered 200.
# An agent that satisfies both the server and the check while naming a contact
# nobody can reach is the exact failure SECURITY.md control 5 describes: it
# reads as compliance without being it.
PLACEHOLDER_TOKENS: Final = (
    "example.com",
    "example.org",
    "example.net",
    "your-site",
    "your-domain",
    "yourdomain",
    "your-company",
    "yourcompany",
    "changeme",
    "change-me",
    "todo",
    "fixme",
    "xxx",
)

# One page per request, and that is the API's limit rather than a politeness
# choice. `prop=extracts` with `explaintext` caps whole-article extracts at
# `exlimit=1`: asking for 50 titles returns 50 page objects of which exactly one
# carries an `extract` and the other 49 carry an empty string, with the cap
# reported only in a `warnings` block nobody reads. A 50-title batch therefore
# builds a corpus missing 98% of its text, and every skipped page looks like a
# page that legitimately had no prose. Measured against en.wikinews.org on
# 2026-09-25: three titles requested, one extract returned, warning
# "exlimit was too large for a whole article extracts request, lowered to 1".
MAX_TITLES_PER_REQUEST: Final = 1

# One request per page means politeness is now the caller's problem rather than
# the batch size's. DATA.md: "Rate limits are conservative, well under anything
# that would look like scraping pressure."
REQUEST_INTERVAL_S: Final = 0.5


class SourceUnavailableError(ProvenanceError):
    code = "reference_source_unavailable"


def check_user_agent(settings: FetchSettings) -> None:
    """Refuse to contact Wikimedia with a placeholder contact address.

    Their user agent policy asks for a way to reach the operator. Sending one
    that cannot be reached is worse than sending none: it looks like
    compliance. D030 left this row open on the basis that nothing depended on
    it yet; something does now.
    """
    agent = settings.user_agent
    lowered = agent.lower()

    # Reserved suffixes are checked against each token rather than the whole
    # string, so `a.invalid/path` is caught and a legitimate host whose name
    # merely contains "test" is not.
    tokens = re.split(r"[\s;,()<>]+", lowered)
    reserved = [
        token
        for token in tokens
        for host in [token.split("//")[-1].split("/")[0].split("@")[-1].rstrip(".")]
        if host.endswith(RESERVED_SUFFIXES)
    ]
    template = [token for token in PLACEHOLDER_TOKENS if token in lowered]

    if reserved or template:
        named = ", ".join(sorted({*reserved, *template}))
        msg = (
            f"the fetch user agent is {agent!r}, which names a contact nobody can "
            f"reach ({named}). Wikimedia's policy requires a real way to reach the "
            "operator, and an unreachable address is worse than none because it reads "
            "as compliance without being it (SECURITY.md control 5). Set "
            "FETCH_USER_AGENT to an agent carrying a working contact before building "
            "the reference corpus. Note that the remote host will not catch this: a "
            "well formed agent naming a nonexistent domain is answered 200."
        )
        raise SourceUnavailableError(msg)


def attribution_for(host: str, title: str, url: str) -> str:
    """The citation string a card must display beside a chunk of this article."""
    licence, licence_url, wiki = LICENCES[host]
    return f'"{title}", {wiki} ({url}), {licence} ({licence_url})'


class WikimediaSource:
    """Reference documents from one Wikimedia wiki, by page title.

    Titles are supplied by the caller rather than discovered here. Which pages
    belong in a reference corpus is a corpus design decision that belongs in the
    build script and the manifest entry, not hidden inside a fetch loop.
    """

    def __init__(
        self,
        host: str,
        titles: list[str],
        fetch_json: Callable[[str], dict[str, object]] | None = None,
        settings: FetchSettings | None = None,
        snapshot_date: date | None = None,
        request_interval_s: float = REQUEST_INTERVAL_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if host not in LICENCES:
            msg = f"no licence recorded for {host!r}; read its siteinfo rightsinfo first"
            raise SourceUnavailableError(msg)
        self.host = host
        self.titles = titles
        self.settings = settings or FetchSettings()
        check_user_agent(self.settings)
        self._fetch_json = fetch_json or self._default_fetch
        self.snapshot_date = snapshot_date or datetime.now(UTC).date()
        self.request_interval_s = request_interval_s
        self._sleep = sleep or time.sleep

    @property
    def corpus_id(self) -> str:
        return f"reference_{self.host.replace('.', '_')}"

    def _default_fetch(self, url: str) -> dict[str, object]:
        import httpx

        response = httpx.get(
            url,
            timeout=self.settings.total_timeout_s,
            headers={"User-Agent": self.settings.user_agent},
            follow_redirects=False,
        )
        response.raise_for_status()
        body: dict[str, object] = json.loads(response.text)
        return body

    def _batch_url(self, titles: list[str]) -> str:
        query = urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                # Stated rather than left to the server to impose silently. The
                # API lowers this to 1 for whole-article extracts either way;
                # writing it here is what stops a later reader raising
                # MAX_TITLES_PER_REQUEST and losing most of the corpus.
                "exlimit": "1",
                "titles": "|".join(titles),
                "format": "json",
                "formatversion": "2",
            }
        )
        return f"https://{self.host}/w/api.php?{query}"

    def documents(self) -> Iterator[ReferenceDocument]:
        """One `ReferenceDocument` per page that exists and has text."""
        licence, licence_url, wiki = LICENCES[self.host]
        accessed = datetime.now(UTC).date()

        for start in range(0, len(self.titles), MAX_TITLES_PER_REQUEST):
            if start:
                self._sleep(self.request_interval_s)
            batch = self.titles[start : start + MAX_TITLES_PER_REQUEST]
            payload = self._fetch_json(self._batch_url(batch))
            query = payload.get("query")
            pages = query.get("pages", []) if isinstance(query, dict) else []

            for page in pages if isinstance(pages, list) else []:
                if not isinstance(page, dict) or page.get("missing"):
                    continue
                text = str(page.get("extract") or "").strip()
                title = str(page.get("title") or "")
                if not text or not title:
                    continue
                url = f"https://{self.host}/wiki/{title.replace(' ', '_')}"
                yield ReferenceDocument(
                    document_id=f"{self.corpus_id}:{page.get('pageid')}",
                    text=text,
                    title=title,
                    source_url=url,
                    publisher=wiki,
                    licence=licence,
                    licence_url=licence_url,
                    redistribution="permitted_with_attribution",
                    attribution=attribution_for(self.host, title, url),
                    accessed=accessed,
                    snapshot_date=self.snapshot_date,
                )
