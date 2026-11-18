"""Потоковий токенізатор з Unicode-нормалізацією та формалізованими правилами."""

from __future__ import annotations

from collections.abc import Iterator
import re
import unicodedata

APOSTROPHE_RE = re.compile(r"[’ʼ`]")

TOKEN_RE = re.compile(
    r"""
    [^\W\d_]+(?:['-][^\W\d_]+)*
    |
    \d+(?:[.,]\d+)*
    """,
    re.VERBOSE | re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Виконує NFKC-нормалізацію Unicode, уніфікацію апострофів і casefold()."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = APOSTROPHE_RE.sub("'", normalized)
    return normalized.casefold()


def tokenize(text: str) -> Iterator[str]:
    """Потоково повертає нормалізовані токени з тексту без створення проміжного списку."""
    normalized = normalize_text(text)
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)
