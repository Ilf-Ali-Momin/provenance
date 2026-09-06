"""robots.txt rule matching with wildcard support.

Written because the standard library is not sufficient here, and the way it
falls short under-blocks rather than over-blocks.

`urllib.robotparser` matches rule paths as literal prefixes. It has no support
for the `*` wildcard or the `$` end anchor, both of which are in near universal
use and are honoured by every major crawler. Measured against Associated Press
on 2026-09-06, whose robots.txt contains:

    Disallow: /*.rss

`urllib.robotparser.can_fetch` returns **True** for `https://apnews.com/index.rss`,
because it compares the literal prefix `/*.rss` against the path and finds no
match. A publisher had stated a restriction and the parser reported permission.

DATA.md commits this project to respecting robots directives on every fetch.
Honouring them only where the standard library happens to understand the syntax
is not respecting them, so the matching is implemented here instead: patterns
compile to anchored regexes, and the longest matching rule wins with `Allow`
taking precedence over an equally long `Disallow`, which is the conventional
resolution.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlsplit

_WILDCARD_STAR: Final = "\x00STAR\x00"


@dataclass(frozen=True)
class Rule:
    """One Allow or Disallow line, compiled."""

    allow: bool
    pattern: str
    matcher: re.Pattern[str]

    def matches(self, path: str) -> bool:
        return self.matcher.match(path) is not None


@dataclass
class RuleSet:
    """The rules that apply to one user agent."""

    rules: list[Rule] = field(default_factory=list)

    def allows(self, path: str) -> bool:
        """Longest match wins; Allow beats Disallow at equal length.

        An empty rule set allows everything, which is what a robots.txt with
        no applicable directives means.
        """
        best: Rule | None = None
        for rule in self.rules:
            if not rule.matches(path):
                continue
            if (
                best is None
                or len(rule.pattern) > len(best.pattern)
                or (len(rule.pattern) == len(best.pattern) and rule.allow)
            ):
                best = rule
        return best.allow if best is not None else True


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a robots path pattern to an anchored regex.

    `*` matches any run of characters and a trailing `$` anchors the end.
    Everything else is literal, so a path containing regex metacharacters
    cannot change the meaning of a rule.
    """
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    placeheld = body.replace("*", _WILDCARD_STAR)
    escaped = re.escape(placeheld).replace(re.escape(_WILDCARD_STAR), ".*")
    return re.compile(escaped + ("$" if anchored else ""))


def parse(text: str, user_agent: str) -> RuleSet:
    """Parse robots.txt and return the rules applying to ``user_agent``.

    A group naming our agent specifically wins over the `*` group entirely,
    which is how agent specific rules are meant to work: they replace the
    wildcard group rather than adding to it.
    """
    groups: dict[str, list[Rule]] = {}
    current: list[str] = []
    token = user_agent.split("/")[0].strip().lower()

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, _, value = line.partition(":")
        name = field_name.strip().lower()
        value = value.strip()

        if name == "user-agent":
            agent = value.lower()
            current = [agent]
            groups.setdefault(agent, [])
        elif name in {"allow", "disallow"} and current:
            if not value:
                # "Disallow:" with no value is an explicit allow-all and
                # carries no path, so it constrains nothing.
                continue
            rule = Rule(allow=name == "allow", pattern=value, matcher=compile_pattern(value))
            for agent in current:
                groups.setdefault(agent, []).append(rule)

    for agent, rules in groups.items():
        if agent and agent != "*" and (agent in token or token in agent):
            return RuleSet(rules)
    return RuleSet(groups.get("*", []))


def path_of(url: str) -> str:
    """The path and query a robots rule is matched against."""
    parts = urlsplit(url)
    path = parts.path or "/"
    return f"{path}?{parts.query}" if parts.query else path


def crawl_delay(text: str, user_agent: str) -> float | None:
    """The Crawl-delay declared for this agent, if any."""
    token = user_agent.split("/")[0].strip().lower()
    applicable: dict[str, float] = {}
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        name = name.strip().lower()
        value = value.strip()
        if name == "user-agent":
            current = [value.lower()]
        elif name == "crawl-delay" and current:
            # A malformed delay is not an error to propagate, it is a line to
            # ignore; our own floor still applies either way.
            with contextlib.suppress(ValueError):
                delay = float(value)
                for agent in current:
                    applicable[agent] = delay

    for agent, delay in applicable.items():
        if agent and agent != "*" and (agent in token or token in agent):
            return delay
    return applicable.get("*")
