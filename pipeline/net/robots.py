"""robots.txt fetching and evaluation.

DATA.md commits this project to respecting robots directives on every fetch,
and SECURITY.md control 5 to conservative crawl delays and a descriptive user
agent. Neither is optional and neither is a courtesy: being reachable and well
behaved is what turns a complaint into an email rather than a block, and a
project about information integrity that ignores access terms to build itself
has undermined its own premise.

Two decisions worth naming.

A robots.txt that cannot be fetched is treated as **disallow**, not allow. The
permissive reading is the conventional one, and it is wrong here: a host that
is refusing connections or serving errors is not a host that wants to be
crawled, and this corpus can afford to lose a publisher far more easily than it
can afford to be a nuisance.

The crawl delay is the maximum of the host's declared value and our own floor,
so a host asking for more patience gets it and a host asking for none still
gets ours.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlsplit

from pipeline.errors import ProvenanceError
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import Resolver, fetch
from pipeline.net.robots_rules import RuleSet, crawl_delay, parse, path_of, sitemap_urls

DEFAULT_MIN_DELAY_S: Final = 2.0


@dataclass
class RobotsPolicy:
    """One host's robots.txt, parsed, with the delay we will actually use."""

    host: str
    fetched: bool
    delay_s: float
    rules: RuleSet | None = None
    sitemaps: list[str] = field(default_factory=list)

    def allows(self, url: str, user_agent: str) -> bool:
        """Whether this URL may be fetched. Unfetchable robots means no."""
        del user_agent  # the rule set was already selected for our agent
        if not self.fetched or self.rules is None:
            return False
        return self.rules.allows(path_of(url))


@dataclass
class RobotsCache:
    """Per host robots policies and last fetch times.

    Holds the crawl delay bookkeeping too, because the delay is per host and
    keeping it anywhere else invites two call sites disagreeing about when a
    host was last touched.
    """

    settings: FetchSettings = field(default_factory=FetchSettings)
    min_delay_s: float = DEFAULT_MIN_DELAY_S
    resolver: Resolver | None = None
    _policies: dict[str, RobotsPolicy] = field(default_factory=dict)
    _last_fetch: dict[str, float] = field(default_factory=dict)

    def policy_for(self, url: str) -> RobotsPolicy:
        """Fetch and cache the robots policy for this URL's host."""
        host = urlsplit(url).netloc.lower()
        cached = self._policies.get(host)
        if cached is not None:
            return cached

        scheme = urlsplit(url).scheme or "https"
        robots_url = f"{scheme}://{host}/robots.txt"

        try:
            response = fetch(robots_url, settings=self.settings, resolver=self.resolver)
        except ProvenanceError:
            # Unreachable, rejected, oversized or timed out. Treated as a
            # refusal, deliberately.
            policy = RobotsPolicy(host=host, fetched=False, delay_s=self.min_delay_s)
            self._policies[host] = policy
            return policy

        rules = parse(response.text, self.settings.user_agent)
        declared = crawl_delay(response.text, self.settings.user_agent)
        delay = max(float(declared or 0.0), self.min_delay_s)
        policy = RobotsPolicy(
            host=host,
            fetched=True,
            delay_s=delay,
            rules=rules,
            sitemaps=sitemap_urls(response.text),
        )
        self._policies[host] = policy
        return policy

    def allows(self, url: str) -> bool:
        return self.policy_for(url).allows(url, self.settings.user_agent)

    def sitemaps_for(self, url: str) -> list[str]:
        """Sitemaps this host declares in robots.txt. Empty if unfetchable."""
        return list(self.policy_for(url).sitemaps)

    def wait_for(self, url: str) -> None:
        """Block until this host may be contacted again.

        Per host rather than global: politeness is owed to each host
        separately, and a global sleep would make collection needlessly slow
        without making it any more polite.
        """
        host = urlsplit(url).netloc.lower()
        policy = self.policy_for(url)
        last = self._last_fetch.get(host)
        if last is not None:
            remaining = policy.delay_s - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_fetch[host] = time.monotonic()
