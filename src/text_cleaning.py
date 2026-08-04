"""Cleaning pipeline for raw extracted document text.

Turns noisy extracted text (repeated headers/footers, encoding artifacts,
broken line wraps, runaway whitespace) into consistent, retrieval-ready text.
"""

import re
import unicodedata
from collections import Counter

PAGE_FOOTER_RE = re.compile(r"Page \d+ of \d+", re.IGNORECASE)
LINE_WRAP_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")


def fix_mojibake(text: str) -> str:
    """Repairs UTF-8 text that was mis-decoded as cp1252 (e.g. 'â€™' -> ''')."""
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def join_wrapped_lines(text: str) -> str:
    """Rejoins words that were hyphen-broken across a line wrap."""
    return LINE_WRAP_HYPHEN_RE.sub(r"\1\2", text)


def strip_repeated_lines(text: str, min_occurrences: int = 3, max_len: int = 80) -> str:
    """Removes lines that repeat often enough to be a header/footer/nav element."""
    lines = text.split("\n")
    counts = Counter(line.strip() for line in lines if line.strip())
    boilerplate = {
        line for line, count in counts.items()
        if count >= min_occurrences and len(line) <= max_len
    }
    return "\n".join(line for line in lines if line.strip() not in boilerplate)


def clean(text: str) -> str:
    # Applied per line: one line with an unrepairable byte sequence should not
    # block the mojibake fix for the rest of the document.
    text = "\n".join(fix_mojibake(line) for line in text.split("\n"))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = join_wrapped_lines(text)
    text = PAGE_FOOTER_RE.sub("", text)
    text = strip_repeated_lines(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
