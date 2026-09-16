"""Worker entrypoint. Runs the fetch queue, and nothing else.

SECURITY.md control 1 requires the fetcher to run in a container whose egress is
restricted to the public internet, with no route to internal services. This is
that container's process. It imports the pipeline and never the application
package, so the image that talks to hostile hosts carries no web framework.

    PYTHONPATH=. .venv/bin/python -m pipeline.worker
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

QUEUE_ENV: Final = "APP_QUEUE_NAME"
REDIS_ENV: Final = "APP_REDIS_URL"


def main() -> int:
    from redis import Redis
    from rq import Queue, SimpleWorker

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    redis = Redis.from_url(os.environ.get(REDIS_ENV, "redis://127.0.0.1:6379/0"))
    queue = Queue(os.environ.get(QUEUE_ENV, "fetch"), connection=redis)

    # SimpleWorker: no fork per job. The fetch is IO bound and short, forking
    # buys process isolation this container already has from the outside, and a
    # forking worker inside a locked down container is the harder thing to
    # reason about when something goes wrong.
    SimpleWorker([queue], connection=redis).work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
