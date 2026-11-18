"""Тести потокового Unicode-токенізатора з використанням pytest.mark.parametrize."""

from __future__ import annotations

from collections.abc import Iterator
import types

import pytest

from findex.tokenizer import normalize_text, tokenize


def test_tokenize_returns_lazy_generator() -> None:
    stream = tokenize("Перший другий третій")
    assert isinstance(stream, Iterator)
    assert isinstance(stream, types.GeneratorType)
    assert next(stream) == "перший"


@pytest.mark.parametrize(
    ("raw_text", "expected_tokens"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("ﬁnance cafe\u0301 Straße", ["finance", "café", "strasse"]),
        ("м'яч м’яч мʼяч don't 'цитата'", ["м'яч", "м'яч", "м'яч", "don't", "цитата"]),
        ("світло-зелений state-of-the-art --тире--", ["світло-зелений", "state-of-the-art", "тире"]),
        ("Рік 2026. Пі = 3.14, сума 1,000,000.", ["рік", "2026", "пі", "3.14", "сума", "1,000,000"]),
        ("КИЇВ Україна PYTHON", ["київ", "україна", "python"]),
        ("   \n\t ... --- !!! ", []),
        ("", []),
    ],
)
def test_tokenize_parametrized_unicode_and_rules(
    raw_text: str,
    expected_tokens: list[str],
) -> None:
    assert list(tokenize(raw_text)) == expected_tokens


def test_normalize_text_idempotent() -> None:
    sample = "ﬁnance М’яч Straße 2026"
    once = normalize_text(sample)
    twice = normalize_text(once)
    assert once == twice == "finance м'яч strasse 2026"
