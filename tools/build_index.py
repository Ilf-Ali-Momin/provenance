"""Build the reference index: select pages, fetch, chunk, embed, store.

    PYTHONPATH=. .venv/bin/python -m tools.build_index --wikinews 200
    PYTHONPATH=. .venv/bin/python -m tools.build_index --wikipedia 100

Requires `FETCH_USER_AGENT` to name a contact that can actually receive mail.
The source refuses to start otherwise, and the remote host will not catch a
well formed agent naming a domain that does not exist (D047).

Requires Postgres with pgvector:

    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres

Wikipedia page selection is seeded from the OOD corpus's train and dev splits,
never the frozen holdout. `pipeline/retrieval/select.py` refuses the holdout by
name; the reason is in its docstring and in D009.

The report this prints is what the corpus's `data/MANIFEST.yaml` entry needs:
counts, licence tally, snapshot date and access window. It is printed rather
than written, because a manifest entry is a claim about a corpus and editing one
from a script makes it a side effect of a command someone ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from pipeline.net.config import FetchSettings
from pipeline.retrieval.corpus import build_index
from pipeline.retrieval.db import PostgresSettings, connect
from pipeline.retrieval.embed import OnnxEmbedder
from pipeline.retrieval.select import (
    DEFAULT_PAGES_PER_SEED,
    seed_terms,
    wikinews_titles,
    wikipedia_titles,
)
from pipeline.retrieval.sources import WIKINEWS, WIKIPEDIA, WikimediaSource, check_user_agent
from pipeline.retrieval.store import PgVectorStore, create_schema

MODEL_DIR: Final = Path("models/bge")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikinews", type=int, default=0, help="articles from Category:Published")
    parser.add_argument("--wikipedia", type=int, default=0, help="pages, seeded from train+dev")
    parser.add_argument("--seeds", type=int, default=50, help="entities to search Wikipedia with")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dry-run", action="store_true", help="select and count, index nothing")
    args = parser.parse_args()

    if not (args.wikinews or args.wikipedia):
        raise SystemExit("pass --wikinews N or --wikipedia N; neither alone builds anything")

    fetch_settings = FetchSettings()
    # Checked once, up front, rather than per source: a build that dies on the
    # second wiki after indexing the first leaves a half corpus whose manifest
    # entry would be a guess.
    check_user_agent(fetch_settings)

    if not (args.model_dir / "model.onnx").exists():
        raise SystemExit(
            f"no embedder at {args.model_dir}; run "
            "`PYTHONPATH=. .venv/bin/python -m tools.export_embedder` first"
        )

    plans: list[tuple[str, list[str]]] = []
    probe = WikimediaSource(WIKINEWS, [], settings=fetch_settings)

    if args.wikinews:
        plans.append((WIKINEWS, wikinews_titles(probe._fetch_json, limit=args.wikinews)))
    if args.wikipedia:
        seeds = seed_terms(limit=args.seeds)
        sys.stdout.write(f"{len(seeds)} seeds from train+dev, first five: {seeds[:5]}\n")
        titles = wikipedia_titles(seeds, probe._fetch_json, pages_per_seed=DEFAULT_PAGES_PER_SEED)
        plans.append((WIKIPEDIA, titles[: args.wikipedia]))

    for host, titles in plans:
        sys.stdout.write(f"{host}: {len(titles)} pages selected\n")
    if args.dry_run:
        return 0

    embedder = OnnxEmbedder(args.model_dir)
    connection = connect(PostgresSettings())
    try:
        create_schema(connection)
        store = PgVectorStore(connection, embedder)
        for host, titles in plans:
            source = WikimediaSource(host, titles, settings=fetch_settings)
            report = build_index(source, store)
            sys.stdout.write(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
