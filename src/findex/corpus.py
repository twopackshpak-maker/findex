"""Читання текстового корпусу як лінивого потоку документів."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".jsonl")


@dataclass(frozen=True, slots=True)
class Document:
    """Представлення окремого документа корпусу."""

    doc_id: str
    path: Path
    text: str
    rel_path: str = ""
    title: str = ""


def collect_corpus_paths(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Повертає відсортований список шляхів до файлів корпусу."""
    if not root.exists():
        raise FileNotFoundError(f"Директорію корпусу не знайдено: {root}")

    normalized_exts = {ext.lower() for ext in extensions}
    if root.is_file():
        if root.suffix.lower() in normalized_exts:
            return [root]
        raise NotADirectoryError(f"Очікувалась директорія, отримано файл: {root}")

    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in normalized_exts)


def count_corpus_files(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> int:
    """Підраховує кількість файлів корпусу для відображення прогресу індексації."""
    return len(collect_corpus_paths(root, extensions=extensions))


def _iter_jsonl_documents(path: Path, base_id: str) -> Iterator[Document]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw_obj: Any = json.loads(stripped)
            if not isinstance(raw_obj, dict):
                continue
            record = cast(dict[str, Any], raw_obj)
            title = str(record.get("title", "")).strip()
            body = str(record.get("text", "")).strip()
            combined = (
                f"{title}\n{body}".strip() if title and title not in body else (body or title)
            )
            if not combined:
                continue
            doc_id = str(record.get("url") or f"{base_id}#{line_no}")
            yield Document(
                doc_id=doc_id,
                path=path,
                text=combined,
                rel_path=f"{base_id}::{line_no}",
                title=title or path.stem,
            )


def iter_documents(
    root: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> Iterator[Document]:
    """Ліниво ітерує документи з директорії або `.jsonl`-файлу по одному запису."""
    base_root = root.parent if root.is_file() else root
    for path in collect_corpus_paths(root, extensions=extensions):
        rel_id = path.relative_to(base_root).as_posix()
        if path.suffix.lower() == ".jsonl":
            yield from _iter_jsonl_documents(path, rel_id)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        yield Document(
            doc_id=rel_id,
            path=path,
            text=text,
            rel_path=rel_id,
            title=path.stem,
        )
