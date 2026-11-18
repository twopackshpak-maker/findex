"""Спільні фікстури pytest для тестування корпусу, індексу та ранжування."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import Index, build_inverted_index
from findex.storage import save_index


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    """Створює компактний тестовий корпус із 5 документів на диску."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "00_python.txt").write_text(
        "Python Async Guide\n"
        "Python supports async and await syntax with an event loop for concurrency.",
        encoding="utf-8",
    )
    (corpus / "01_search.txt").write_text(
        "Inverted Index Architecture\n"
        "An inverted index maps every normalized term to sorted postings.",
        encoding="utf-8",
    )
    (corpus / "02_ukrainian.txt").write_text(
        "Українська токенізація\n"
        "Слова з апострофом м'яч, під'їзд та дефісом світло-зелений індексуються коректно.",
        encoding="utf-8",
    )
    (corpus / "03_java.txt").write_text(
        "Java Virtual Machine\n"
        "Java threads and event loop frameworks differ from Python coroutines.",
        encoding="utf-8",
    )
    (corpus / "04_rare.txt").write_text(
        "Botany Notes\nA rare chrysanthemum flower blooms once in autumn.",
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    """Будує індекс без позицій на тестовому корпусі."""
    return build_inverted_index(sample_corpus_dir, include_positions=False)


@pytest.fixture
def sample_index_with_positions(sample_corpus_dir: Path) -> Index:
    """Будує індекс із позиціями токенів на тестовому корпусі."""
    return build_inverted_index(sample_corpus_dir, include_positions=True)


@pytest.fixture
def saved_index_path(tmp_path: Path, sample_index_with_positions: Index) -> Path:
    """Зберігає індекс із позиціями у файл index.bin."""
    out_path = tmp_path / "index.bin"
    save_index(sample_index_with_positions, out_path, fmt="bin")
    return out_path
