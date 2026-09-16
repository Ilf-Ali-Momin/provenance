"""Prompt loading, versioning, and the delimiter layer of control 2.

ARCHITECTURE.md requires the prompt versioned in `prompts/` with a hash
recorded in every output, "so a card can be traced to the exact prompt that
produced it". EVALUATION.md adds the reason: prompt changes are code changes,
and untracked prompt drift is a real source of silent regression.

The delimiter is the third layer of SECURITY.md control 2 and it is random per
request, "so it cannot be predicted and closed by injected text". A fixed
delimiter is worse than none once it is known: an attacker writes the closing
token into the article and everything after it reads as instructions.
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Final

PROMPT_DIR: Final = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_PROMPT: Final = "claim_extraction.v1.md"
DELIMITER_BYTES: Final = 16


def load_prompt(name: str = DEFAULT_PROMPT) -> tuple[str, str]:
    """Return the prompt text and its version string, e.g. `v1:a91f3c2`."""
    path = PROMPT_DIR / name
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:7]
    version = name.removeprefix("claim_extraction.").removesuffix(".md")
    return text, f"{version}:{digest}"


def new_delimiter() -> str:
    """An unguessable per request boundary token."""
    return f"<<<{secrets.token_hex(DELIMITER_BYTES)}>>>"


def build_messages(source: str, prompt_text: str, delimiter: str) -> list[dict[str, str]]:
    """Instructions above, untrusted data inside the delimited region.

    The source is never formatted into the instruction string. It sits between
    two copies of a token the caller generated a moment ago, which means an
    injected attempt to close the region has to guess 128 bits.
    """
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": f"{delimiter}\n{source}\n{delimiter}",
        },
    ]
