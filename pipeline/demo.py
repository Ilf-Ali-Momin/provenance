"""Fetch one URL and show what the pipeline does to it.

A demonstration, not a product surface. It composes the parts that exist,
the SSRF guarded client, robots compliance, article extraction and publisher
marker stripping, and prints what each of them did, so the machinery can be
seen working before the API in M1 exists to wrap it.

It never bypasses anything. A URL that robots.txt disallows is refused here
exactly as it is refused in collection, and there is deliberately no flag to
override that.

    PYTHONPATH=. .venv/bin/python -m pipeline.demo https://example.com/article
    PYTHONPATH=. .venv/bin/python -m pipeline.demo <url> --outlet-name "The Guardian"
    PYTHONPATH=. .venv/bin/python -m pipeline.demo <url> --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from typing import Final
from urllib.parse import urlsplit

from pipeline.errors import ProvenanceError, UrlRejectedError
from pipeline.net.config import FetchSettings
from pipeline.net.fetch import fetch
from pipeline.net.robots import RobotsCache
from pipeline.text.config import ExtractSettings
from pipeline.text.extract import extract_article

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_REFUSED: Final = 2
PREVIEW_CHARS: Final = 800


def inspect(
    url: str,
    outlet_names: tuple[str, ...] = (),
    robots: RobotsCache | None = None,
    fetch_settings: FetchSettings | None = None,
    extract_settings: ExtractSettings | None = None,
) -> dict[str, object]:
    """Run one URL through the pipeline and report every step.

    Raises UrlRejectedError when robots.txt disallows the URL, and any other
    ProvenanceError the fetcher raises. Nothing is caught and turned into a
    partial result: a refusal is an outcome, not a degraded success.
    """
    robots = robots or RobotsCache()
    fetch_settings = fetch_settings or FetchSettings()
    extract_settings = extract_settings or ExtractSettings()

    started = time.monotonic()
    policy = robots.policy_for(url)
    if not robots.allows(url):
        raise UrlRejectedError("robots.txt disallows this URL for our user agent.")
    robots.wait_for(url)
    robots_ms = int((time.monotonic() - started) * 1000)

    started = time.monotonic()
    response = fetch(url, settings=fetch_settings, resolver=None)
    fetch_ms = int((time.monotonic() - started) * 1000)

    started = time.monotonic()
    article = extract_article(
        response.text, url=url, settings=extract_settings, outlet_names=outlet_names
    )
    extract_ms = int((time.monotonic() - started) * 1000)

    report: dict[str, object] = {
        "url": url,
        "host": urlsplit(url).netloc,
        "robots": {
            "fetched": policy.fetched,
            "crawl_delay_s": policy.delay_s,
            "sitemaps_declared": len(policy.sitemaps),
            "ms": robots_ms,
        },
        "fetch": {
            "status": response.status_code,
            "content_type": response.content_type,
            "html_bytes": len(response.text.encode()),
            "final_url": response.final_url,
            "ms": fetch_ms,
        },
        "extract": {"ms": extract_ms},
    }

    if article is None:
        report["extract"] = {
            **report["extract"],  # type: ignore[dict-item]
            "article": False,
            "why": (
                "no article text above the "
                f"{extract_settings.min_article_chars} character floor: paywall, "
                "index page, or a layout the extractor does not handle"
            ),
        }
        return report

    report["extract"] = {
        **report["extract"],  # type: ignore[dict-item]
        "article": True,
        "title": article.title,
        "page_published": article.published,
        "declared_language": article.declared_language,
        "english_function_word_ratio": article.english_ratio,
        "invisible_characters_present": article.had_invisible,
        "characters": len(article.text),
        "sha256": hashlib.sha256(article.text.encode()).hexdigest(),
    }
    report["removed"] = article.removed or {}
    report["text"] = article.text
    return report


def render(report: dict[str, object], preview_chars: int) -> str:
    """Human readable rendering. The JSON form is the machine readable one."""
    robots = report["robots"]
    fetched = report["fetch"]
    extract = report["extract"]
    lines = [
        f"URL         {report['url']}",
        f"robots.txt  fetched={robots['fetched']} "  # type: ignore[index]
        f"crawl_delay={robots['crawl_delay_s']}s "  # type: ignore[index]
        f"sitemaps={robots['sitemaps_declared']} ({robots['ms']} ms)",  # type: ignore[index]
        f"fetch       HTTP {fetched['status']} {fetched['content_type']} "  # type: ignore[index]
        f"{fetched['html_bytes']} bytes ({fetched['ms']} ms)",  # type: ignore[index]
    ]
    if fetched["final_url"] != report["url"]:  # type: ignore[index]
        lines.append(f"            redirected to {fetched['final_url']}")  # type: ignore[index]

    if not extract["article"]:  # type: ignore[index]
        lines.append(f"extract     no article ({extract['ms']} ms)")  # type: ignore[index]
        lines.append(f"            {extract['why']}")  # type: ignore[index]
        return "\n".join(lines)

    lines += [
        f"extract     {extract['characters']} characters ({extract['ms']} ms)",  # type: ignore[index]
        f"            title: {extract['title']}",  # type: ignore[index]
        f"            page date: {extract['page_published']}  "  # type: ignore[index]
        f"language: {extract['declared_language']}  "  # type: ignore[index]
        f"english ratio: {extract['english_function_word_ratio']}",  # type: ignore[index]
        f"            invisible characters: {extract['invisible_characters_present']}",  # type: ignore[index]
        f"            sha256: {extract['sha256'][:16]}...",  # type: ignore[index]
    ]

    removed = report.get("removed") or {}
    if removed:
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(removed.items()))  # type: ignore[union-attr]
        lines.append(f"stripped    {detail}")
    else:
        lines.append("stripped    nothing matched the publisher marker rules")

    text = str(report["text"])
    lines += ["", text[:preview_chars] + ("..." if len(text) > preview_chars else "")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch one URL and show what the pipeline does.")
    parser.add_argument("url")
    parser.add_argument(
        "--outlet-name",
        action="append",
        default=[],
        help="a name this publisher uses for itself, masked in its own text; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="machine readable output")
    parser.add_argument("--chars", type=int, default=PREVIEW_CHARS, help="preview length")
    parser.add_argument("--verbose", action="store_true", help="log each network step")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        report = inspect(args.url, tuple(args.outlet_name))
    except UrlRejectedError as exc:
        sys.stderr.write(f"refused: {exc} [{exc.code}]\n")
        return EXIT_REFUSED
    except ProvenanceError as exc:
        sys.stderr.write(f"failed: {exc} [{exc.code}]\n")
        return EXIT_ERROR

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(render(report, args.chars) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
