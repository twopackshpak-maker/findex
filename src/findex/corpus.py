"""Читання текстового корпусу як лінивого потоку документів."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md")


@dataclass(frozen=True, slots=True)
class Document:
    """Представлення окремого документа корпусу."""

    doc_id: str
    path: Path
    text: str


def collect_corpus_paths(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Повертає відсортований список шляхів до файлів корпусу."""
    if not root.exists():
        raise FileNotFoundError(f"Директорію корпусу не знайдено: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Очікувалась директорія, отримано файл: {root}")

    normalized_exts = {ext.lower() for ext in extensions}
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in normalized_exts
    )


def count_corpus_files(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> int:
    """Підраховує кількість файлів корпусу для відображення прогресу індексації."""
    return len(collect_corpus_paths(root, extensions=extensions))


def iter_documents(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> Iterator[Document]:
    """Ліниво ітерує документи з директорії `root` по одному файлу."""
    for path in collect_corpus_paths(root, extensions=extensions):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        doc_id = path.relative_to(root).as_posix()
        yield Document(doc_id=doc_id, path=path, text=text)
