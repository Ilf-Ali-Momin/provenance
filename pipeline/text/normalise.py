"""Unicode normalisation applied identically to every stratum.

Serves two requirements that happen to want the same thing.

SECURITY.md controls 2 and 4: zero width characters, bidirectional overrides
and confusable glyphs are how an attacker perturbs a classifier, or hides an
injected instruction from a human reviewer while leaving it legible to a model.

DATA.md: the OOD suite is interpretable only if every stratum passes through
one pipeline. Normalising wire copy more carefully than scraped copy would
manufacture the very artefact this project documents.

The same functions are used for corpus construction and for serving, so
evaluation input matches production input.

Every codepoint below is declared as an integer range rather than written as
itself. Source containing literal invisible characters cannot be reviewed, and
a reviewer who cannot see a character cannot check it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# (first, last, why) -- inclusive ranges of characters that render as nothing
# but survive a naive strip().
_INVISIBLE_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x00AD, 0x00AD, "soft hyphen"),
    (0x061C, 0x061C, "arabic letter mark"),
    (0x180E, 0x180E, "mongolian vowel separator"),
    (0x200B, 0x200F, "ZWSP, ZWNJ, ZWJ, LRM, RLM"),
    (0x202A, 0x202E, "bidi embeddings and overrides"),
    (0x2060, 0x2064, "word joiner and invisible operators"),
    (0x2066, 0x2069, "bidi isolates"),
    (0xFEFF, 0xFEFF, "BOM / zero width no-break space"),
    (0xFFF9, 0xFFFB, "interlinear annotation"),
)

# C0 and C1 control characters, keeping tab (0x09) and newline (0x0A).
_CONTROL_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0000, 0x0008),
    (0x000B, 0x001F),
    (0x007F, 0x009F),
)

# Every horizontal space Unicode offers, collapsed to one ASCII space.
_SPACE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0009, 0x0009),
    (0x0020, 0x0020),
    (0x00A0, 0x00A0),
    (0x2000, 0x200A),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
)


def _char_class(ranges: tuple[tuple[int, int], ...]) -> str:
    """Build a regex character class from integer codepoint ranges."""
    parts = [f"\\U{lo:08X}-\\U{hi:08X}" if lo != hi else f"\\U{lo:08X}" for lo, hi in ranges]
    return "[" + "".join(parts) + "]"


_INVISIBLE = re.compile(_char_class(tuple((lo, hi) for lo, hi, _ in _INVISIBLE_RANGES)))
_CONTROL = re.compile(_char_class(_CONTROL_RANGES))
_WS_RUN = re.compile(_char_class(_SPACE_RANGES) + "+")
_BLANK_RUN = re.compile("\n{3,}")


def strip_invisible(text: str) -> str:
    """Remove characters that occupy no visual space.

    Runs before NFKC, because NFKC maps some of these to nothing and others to
    a space, and that difference would otherwise be silent.
    """
    return _CONTROL.sub("", _INVISIBLE.sub("", text))


def normalise(text: str) -> str:
    """Invisible removal, then NFKC, then whitespace collapse.

    NFKC is the compatibility form, which folds the lookalike codepoints an
    evader reaches for first: fullwidth Latin, mathematical alphanumerics,
    ligatures. It is lossy by design, and that loss is the point.
    """
    text = strip_invisible(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def has_invisible(text: str) -> bool:
    """Whether the input carried anything invisible.

    Logged and rate limited against, never used to block. SECURITY.md control 2
    is explicit that a detector with a false positive rate is a signal, not a
    gate.
    """
    return bool(_INVISIBLE.search(text) or _CONTROL.search(text))
