"""Серіалізація та десеріалізація інвертованого індексу (binary array та pickle)."""

from __future__ import annotations

from array import array
import logging
from pathlib import Path
import pickle
import struct
from typing import BinaryIO, Literal

from findex.decorators import timed
from findex.index import DocMeta, Index, Posting

logger = logging.getLogger("findex.storage")

StorageFormat = Literal["bin", "pickle"]

BIN_MAGIC = b"FIDXBIN1"
HEADER_STRUCT = struct.Struct("<8sIII?")
U32_STRUCT = struct.Struct("<I")
DOC_HEADER_STRUCT = struct.Struct("<II")


def save_index(index: Index, path: Path, fmt: StorageFormat = "bin") -> None:
    """Зберігає об'єкт `Index` у файл у форматі `bin` (array) або `pickle`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        _save_pickle(index, path)
    else:
        _save_binary_array(index, path)
    logger.debug("Індекс збережено у %s (формат=%s)", path, fmt)


@timed
def load_index(path: Path, *, corpus_root: Path | None = None) -> Index:
    """Автоматично визначає формат за заголовком файлу та завантажує `Index`."""
    if not path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Очікувався файл індексу, отримано директорію: {path}")

    with path.open("rb") as f:
        magic = f.read(len(BIN_MAGIC))

    if magic == BIN_MAGIC:
        idx = _load_binary_array(path)
    else:
        idx = _load_pickle(path)

    if corpus_root is not None:
        idx.corpus_root = corpus_root
    return idx


def _save_pickle(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Index:
    # БЕЗПЕКА: pickle.load() небезпечний для чужих файлів через опкод REDUCE,
    # який дозволяє виконати довільний код (__reduce__).
    with path.open("rb") as f:
        loaded: object = pickle.load(f)
    if not isinstance(loaded, Index):
        raise TypeError(f"Очікувався об'єкт Index, отримано {type(loaded)!r}")
    return loaded


def _write_str(f: BinaryIO, value: str) -> None:
    raw = value.encode("utf-8")
    f.write(U32_STRUCT.pack(len(raw)))
    f.write(raw)


def _read_str(f: BinaryIO) -> str:
    header = f.read(U32_STRUCT.size)
    if len(header) != U32_STRUCT.size:
        raise ValueError("Пошкоджений файл індексу: несподіваний кінець даних")
    (length,) = U32_STRUCT.unpack(header)
    raw = f.read(length)
    if len(raw) != length:
        raise ValueError("Пошкоджений файл індексу: неповний рядок UTF-8")
    return raw.decode("utf-8")


def _save_binary_array(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                BIN_MAGIC,
                index.num_docs,
                len(index),
                index.total_postings,
                index.has_positions,
            )
        )

        for doc in index.docs:
            f.write(DOC_HEADER_STRUCT.pack(doc.doc_id, doc.length))
            _write_str(f, doc.rel_path)
            _write_str(f, doc.title)

        for term, plist in index.items():
            _write_str(f, term)
            f.write(U32_STRUCT.pack(len(plist)))

            doc_ids = array("I", (p.doc_id for p in plist))
            freqs = array("I", (p.tf for p in plist))
            doc_ids.tofile(f)
            freqs.tofile(f)

            if index.has_positions:
                flat_positions = array("I")
                for p in plist:
                    flat_positions.extend(p.positions)
                flat_positions.tofile(f)


def _load_binary_array(path: Path) -> Index:
    with path.open("rb") as f:
        header = f.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            raise ValueError("Пошкоджений заголовок файлу індексу FIDXBIN1")
        magic, doc_count, vocab_size, _, has_positions = HEADER_STRUCT.unpack(header)
        if magic != BIN_MAGIC:
            raise ValueError("Некоректний заголовок файлу індексу FIDXBIN1")

        docs: list[DocMeta] = []
        for _ in range(doc_count):
            doc_hdr = f.read(DOC_HEADER_STRUCT.size)
            if len(doc_hdr) != DOC_HEADER_STRUCT.size:
                raise ValueError("Пошкоджений блок метаданих документа")
            doc_id, length = DOC_HEADER_STRUCT.unpack(doc_hdr)
            rel_path = _read_str(f)
            title = _read_str(f)
            docs.append(DocMeta(doc_id=doc_id, rel_path=rel_path, title=title, length=length))

        postings: dict[str, list[Posting]] = {}
        for _ in range(vocab_size):
            term = _read_str(f)
            count_raw = f.read(U32_STRUCT.size)
            if len(count_raw) != U32_STRUCT.size:
                raise ValueError("Пошкоджений лічильник постінгів")
            (n,) = U32_STRUCT.unpack(count_raw)

            doc_ids: array[int] = array("I")
            doc_ids.fromfile(f, n)
            freqs: array[int] = array("I")
            freqs.fromfile(f, n)

            if has_positions:
                total_pos = sum(freqs)
                flat_pos: array[int] = array("I")
                flat_pos.fromfile(f, total_pos)

                plist: list[Posting] = []
                cursor = 0
                for d_id, tf in zip(doc_ids, freqs, strict=True):
                    pos_tuple = tuple(flat_pos[cursor : cursor + tf])
                    cursor += tf
                    plist.append(Posting(doc_id=d_id, tf=tf, positions=pos_tuple))
                postings[term] = plist
            else:
                postings[term] = [
                    Posting(doc_id=d_id, tf=tf) for d_id, tf in zip(doc_ids, freqs, strict=True)
                ]

    return Index(postings=postings, docs=docs, has_positions=has_positions)
