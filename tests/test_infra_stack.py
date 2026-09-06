"""Checks the local dependency stack matches what ARCHITECTURE.md assumes.

Marked ``integration``: it needs `docker compose up` and is excluded from the
default run. RULES.md forbids live external calls in CI, and this is the
boundary that rule is about — it verifies a local stack, never a remote one.
"""

from __future__ import annotations

import os
import socket

import pytest

pytestmark = pytest.mark.integration

PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


requires_stack = pytest.mark.skipif(
    not _listening(PG_PORT) or not _listening(REDIS_PORT),
    reason="docker compose stack is not running",
)


@requires_stack
def test_services_bind_loopback_only() -> None:
    """A database reachable off-box is the exposure this project prevents."""
    assert _listening(PG_PORT)
    assert _listening(REDIS_PORT)


@requires_stack
def test_pgvector_supports_the_architecture_dimension() -> None:
    psycopg = pytest.importorskip("psycopg")
    dsn = (
        f"host=127.0.0.1 port={PG_PORT} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']} "
        f"dbname={os.environ['POSTGRES_DB']}"
    )
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        assert cur.fetchone() is not None, "pgvector extension is not installed"
        # 768 dimensions, bge-base-en-v1.5, per ARCHITECTURE.md.
        cur.execute("CREATE TEMP TABLE probe (v vector(768))")
        cur.execute("INSERT INTO probe VALUES (array_fill(0.1::real, ARRAY[768])::vector)")
        cur.execute("SELECT v <=> v FROM probe")
        row = cur.fetchone()
        assert row is not None
        assert abs(row[0]) < 1e-6, "cosine distance operator is not behaving"
        cur.execute("SELECT amname FROM pg_am WHERE amname = 'hnsw'")
        assert cur.fetchone() is not None, "HNSW index type unavailable"
