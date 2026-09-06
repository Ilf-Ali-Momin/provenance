"""Normalisation cases: the evasion vectors in SECURITY.md control 4.

Every codepoint is built with chr() rather than written literally, so the test
file stays reviewable. A test containing invisible characters cannot be read.
"""

from __future__ import annotations

import pytest
from pipeline.text.normalise import has_invisible, normalise, strip_invisible

pytestmark = pytest.mark.adversarial

ZWSP = chr(0x200B)
ZWNJ = chr(0x200C)
RLO = chr(0x202E)
PDF = chr(0x202C)
BOM = chr(0xFEFF)
NBSP = chr(0x00A0)
SOFT_HYPHEN = chr(0x00AD)


def fullwidth(ascii_text: str) -> str:
    """Build the fullwidth Latin form of an ASCII string.

    Constructed rather than pasted: fullwidth characters are by definition
    indistinguishable from their ASCII twins in a diff, which is exactly why
    they are an evasion vector and exactly why they must not sit literally in
    a test file.
    """
    out = []
    for ch in ascii_text:
        if "A" <= ch <= "Z":
            out.append(chr(ord(ch) - ord("A") + 0xFF21))
        elif "a" <= ch <= "z":
            out.append(chr(ord(ch) - ord("a") + 0xFF41))
        else:
            out.append(ch)
    return "".join(out)


@pytest.mark.parametrize("ch", [ZWSP, ZWNJ, RLO, PDF, BOM, SOFT_HYPHEN, chr(0x2060), chr(0x061C)])
def test_invisible_characters_are_removed(ch: str) -> None:
    assert strip_invisible(f"a{ch}b") == "ab"
    assert has_invisible(f"a{ch}b")


def test_zero_width_injection_between_letters_collapses() -> None:
    """The cheapest evasion: break a token so it no longer matches."""
    assert normalise(f"fa{ZWSP}ke ne{ZWSP}ws") == "fake news"


def test_bidi_override_is_stripped_not_rendered() -> None:
    assert normalise(f"safe{RLO}txet{PDF}") == "safetxet"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (fullwidth("Trump"), "Trump"),
        (chr(0x1D413) + "rump", "Trump"),  # mathematical bold capital T
        (chr(0xFB01) + "nance", "finance"),  # fi ligature
        (chr(0x2460), "1"),  # circled digit one
    ],
)
def test_nfkc_folds_lookalikes(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_whitespace_variants_collapse() -> None:
    assert normalise(f"a{NBSP}{NBSP}b\t\tc") == "a b c"


def test_normalisation_is_idempotent() -> None:
    """Applied at collection and again at serving; must not drift."""
    s = f"Fa{ZWSP}ke{NBSP}{fullwidth('News')}{BOM}"
    assert normalise(normalise(s)) == normalise(s)


def test_clean_text_is_left_alone() -> None:
    assert normalise("Ordinary sentence, with punctuation.") == (
        "Ordinary sentence, with punctuation."
    )
