"""Бінарна (FIDXBIN1 + NumPy int32) та pickle серіалізація інвертованого індексу."""

from __future__ import annotations

import json
import logging
import pickle
import struct
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np

from findex.decorators import timed
from findex.index import DocMeta, Index, Int32Array

logger = logging.getLogger("findex.storage")

StorageFormat = Literal["bin", "pickle"]

BIN_MAGIC = b"FIDXBIN1"
HEADER_STRUCT = struct.Struct("<8sIII?")
U32_STRUCT = struct.Struct("<I")
DOC_HEADER_STRUCT = struct.Struct("<II")


def save_index(index: Index, path: Path, fmt: StorageFormat = "bin") -> None:
    """Зберігає об'єкт `Index` у файл у форматі `bin` (NumPy/uint32) або `pickle`."""
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

    idx = _load_binary_array(path) if magic == BIN_MAGIC else _load_pickle(path)

    if corpus_root is not None:
        idx.corpus_root = corpus_root
    return idx


def save_index_memmap(index: Index, directory: Path) -> None:
    """Зберігає postings у двох пласких масивах `.npy`, придатних для mmap."""
    directory.mkdir(parents=True, exist_ok=True)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for term in index:
        arrays = index.get_posting_arrays(term)
        if arrays is None:
            continue
        count = int(arrays[0].size)
        offsets[term] = (cursor, cursor + count)
        cursor += count

    doc_ids = np.lib.format.open_memmap(
        directory / "doc_ids.npy", mode="w+", dtype=np.int32, shape=(cursor,)
    )
    tfs = np.lib.format.open_memmap(
        directory / "term_frequencies.npy", mode="w+", dtype=np.int32, shape=(cursor,)
    )
    for term, (start, stop) in offsets.items():
        arrays = index.get_posting_arrays(term)
        assert arrays is not None
        doc_ids[start:stop] = arrays[0]
        tfs[start:stop] = arrays[1]
    doc_ids.flush()
    tfs.flush()
    del doc_ids, tfs
    lengths = np.lib.format.open_memmap(
        directory / "doc_lengths.npy", mode="w+", dtype=np.int32, shape=(index.num_docs,)
    )
    lengths[:] = index.doc_lengths
    lengths.flush()
    del lengths

    metadata = {
        "has_positions": index.has_positions,
        "docs": [
            {"doc_id": d.doc_id, "rel_path": d.rel_path, "title": d.title, "length": d.length}
            for d in index.docs
        ],
        "offsets": offsets,
        "positions": {
            term: index.posting_positions[term]
            for term in offsets
            if term in index.posting_positions
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )


def load_index_memmap(directory: Path, *, corpus_root: Path | None = None) -> Index:
    """Відкриває масиви postings у режимі read-only mmap без їх копіювання в RAM."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    docs = [DocMeta(**item) for item in metadata["docs"]]
    doc_ids_all = np.load(directory / "doc_ids.npy", mmap_mode="r")
    tfs_all = np.load(directory / "term_frequencies.npy", mmap_mode="r")
    ids_map: dict[str, Int32Array] = {}
    tfs_map: dict[str, Int32Array] = {}
    for term, (start, stop) in metadata["offsets"].items():
        ids_map[term] = doc_ids_all[start:stop]
        tfs_map[term] = tfs_all[start:stop]
    positions = {
        term: tuple(tuple(p) for p in lists)
        for term, lists in metadata.get("positions", {}).items()
    }
    index = Index(
        postings={},
        docs=docs,
        has_positions=metadata["has_positions"],
        corpus_root=corpus_root,
        posting_arrays=(ids_map, tfs_map),
        posting_positions=positions,
    )
    index.doc_lengths = np.load(directory / "doc_lengths.npy", mmap_mode="r")
    return index


def _save_pickle(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Index:
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

        for term in index:
            _write_str(f, term)
            arrays = index.get_posting_arrays(term)
            if arrays is None:
                f.write(U32_STRUCT.pack(0))
                continue
            doc_ids_arr, tfs_arr = arrays
            f.write(U32_STRUCT.pack(int(doc_ids_arr.shape[0])))
            f.write(doc_ids_arr.astype("<u4").tobytes())
            f.write(tfs_arr.astype("<u4").tobytes())

            if index.has_positions:
                plist = index[term]
                flat_pos = np.array([pos for p in plist for pos in p.positions], dtype="<u4")
                f.write(flat_pos.tobytes())


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

        doc_ids_map: dict[str, Int32Array] = {}
        tfs_map: dict[str, Int32Array] = {}
        positions_map: dict[str, tuple[tuple[int, ...], ...]] = {}

        for _ in range(vocab_size):
            term = _read_str(f)
            count_raw = f.read(U32_STRUCT.size)
            if len(count_raw) != U32_STRUCT.size:
                raise ValueError("Пошкоджений лічильник постінгів")
            (n,) = U32_STRUCT.unpack(count_raw)

            raw_ids = f.read(n * 4)
            raw_freqs = f.read(n * 4)
            if len(raw_ids) != n * 4 or len(raw_freqs) != n * 4:
                raise ValueError("Пошкоджений блок постінгів у файлі індексу FIDXBIN1")

            doc_ids_arr = np.frombuffer(raw_ids, dtype="<u4").astype(np.int32)
            freqs_arr = np.frombuffer(raw_freqs, dtype="<u4").astype(np.int32)
            doc_ids_map[term] = doc_ids_arr
            tfs_map[term] = freqs_arr

            if has_positions:
                total_pos = int(freqs_arr.sum())
                raw_pos = f.read(total_pos * 4)
                if len(raw_pos) != total_pos * 4:
                    raise ValueError("Пошкоджений блок позицій у файлі індексу FIDXBIN1")
                flat_pos = np.frombuffer(raw_pos, dtype="<u4").tolist()
                pos_list: list[tuple[int, ...]] = []
                cursor = 0
                for tf_val in freqs_arr.tolist():
                    pos_list.append(tuple(flat_pos[cursor : cursor + tf_val]))
                    cursor += tf_val
                positions_map[term] = tuple(pos_list)

    return Index(
        postings={},
        docs=docs,
        has_positions=has_positions,
        posting_arrays=(doc_ids_map, tfs_map),
        posting_positions=positions_map if has_positions else None,
    )
