"""Connection settings for the evidence index."""

from __future__ import annotations

from pipeline.retrieval.db import PostgresSettings


def test_the_password_does_not_print() -> None:
    """A settings object in a traceback must not carry the credential."""
    settings = PostgresSettings(password="hunter2")  # type: ignore[arg-type]
    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)


def test_the_dsn_carries_the_real_password() -> None:
    settings = PostgresSettings(user="u", password="p", db="d", host="h", port=6000)  # type: ignore[arg-type]
    assert settings.dsn() == "postgresql://u:p@h:6000/d"


def test_the_default_host_is_loopback() -> None:
    """The deployed topology publishes no port; anything connecting from the
    host is development."""
    assert PostgresSettings().host == "127.0.0.1"


def test_the_env_prefix_matches_what_compose_already_reads() -> None:
    assert PostgresSettings.model_config["env_prefix"] == "POSTGRES_"
