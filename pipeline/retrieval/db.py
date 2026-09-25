"""Connection settings for the evidence index.

RULES.md forbids literals for hosts and credentials, and pydantic-settings is
the project's configuration mechanism. The compose file already reads
`POSTGRES_*` from `.env`, so this reads the same names rather than introducing a
second spelling of the same facts.

The password is a `SecretStr`, so a settings object that reaches a log line or a
traceback prints `**********` rather than the credential. The redaction
processor in `app/logging.py` covers structured fields; this covers the case
where the object is formatted by something that never heard of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", frozen=True, extra="ignore")

    user: str = "provenance"
    password: SecretStr = SecretStr("")
    db: str = "provenance"
    # Loopback by default. The deployed topology puts Postgres on an internal
    # compose network with no published port, so anything that connects from
    # the host is development, and development is where 127.0.0.1 is right.
    host: str = "127.0.0.1"
    port: int = Field(default=5432, gt=0, lt=65536)

    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )


def connect(settings: PostgresSettings | None = None) -> Connection:
    """Open a connection. Callers own closing it."""
    import psycopg

    settings = settings or PostgresSettings()
    return psycopg.connect(settings.dsn())
