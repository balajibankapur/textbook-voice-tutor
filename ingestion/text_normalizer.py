"""Clean up extracted/OCR'd text without changing its meaning.

Operations: de-hyphenate line-wrapped words, collapse runs of whitespace while
preserving paragraph breaks, strip control characters.
"""
from __future__ import annotations

import re
import unicodedata

_HYPHEN_LINEBREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_LINE_WS = re.compile(r"[ \t]+")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _LINE_WS.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_text_usable(text: str, min_alpha_chars: int = 50) -> bool:
    """Heuristic: a page's embedded text is usable if it has enough letters.

    Scanned pages often return a handful of selectable characters from
    headers/footers while the body is still an image — we want OCR for those.
    """
    if not text:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count >= min_alpha_chars
