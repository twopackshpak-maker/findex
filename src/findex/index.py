"""Побудова та представлення інвертованого індексу на масивах NumPy int32."""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from findex.corpus import Document, collect_corpus_paths, iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize

ExecutorMode = Literal["serial", "threads", "processes"]
Int32Array = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Метадані проіндексованого документа."""

    doc_id: int
    rel_path: str
    title: str
    length: int


@dataclass(frozen=True, slots=True)
class Posting:
    """Запис інвертованого індексу для пари `(термін, документ)`."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PartialIndex:
    docs: list[DocMeta]
    postings: dict[str, list[Posting]]
    has_positions: bool


@dataclass(frozen=True, slots=True)
class BuildProfile:
    index: Index
    wall_time_sec: float
    merge_time_sec: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class PostingStorageBenchmark:
    variant: str
    total_postings: int
    elapsed_sec: float
    peak_memory_bytes: int

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)


class TokenizerFn(Protocol):
    """Протокол функції токенізації документа."""

    def __call__(self, text: str) -> Iterable[str]: ...


def _tokenize_document_worker(
    doc: Document,
    include_positions: bool,
) -> tuple[Document, list[str], dict[str, int], dict[str, tuple[int, ...]]]:
    """Воркер для паралельної токенізації документа (у потоках або процесах)."""
    tokens = list(tokenize(doc.text))
    tf_map: dict[str, int] = defaultdict(int)
    pos_map: dict[str, list[int]] = defaultdict(list)

    if include_positions:
        for pos, token in enumerate(tokens):
            tf_map[token] += 1
            pos_map[token].append(pos)
    else:
        for token in tokens:
            tf_map[token] += 1

    frozen_pos = {term: tuple(positions) for term, positions in pos_map.items()}
    return doc, tokens, dict(tf_map), frozen_pos


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс: постінги як масиви NumPy int32 та довжини документів int32."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: list[DocMeta],
        *,
        has_positions: bool = False,
        corpus_root: Path | None = None,
        posting_arrays: tuple[dict[str, Int32Array], dict[str, Int32Array]] | None = None,
        posting_positions: dict[str, tuple[tuple[int, ...], ...]] | None = None,
    ) -> None:
        self._docs: list[DocMeta] = docs
        self.doc_lengths: Int32Array = np.array([doc.length for doc in docs], dtype=np.int32)
        self.has_positions: bool = has_positions
        self.corpus_root: Path | None = corpus_root
        self._doc_text_cache: dict[int, str] = {}
        self._closed: bool = False
        self._posting_doc_ids: dict[str, Int32Array] = {}
        self._posting_tfs: dict[str, Int32Array] = {}
        self._posting_positions: dict[str, tuple[tuple[int, ...], ...]] = (
            dict(posting_positions) if posting_positions is not None else {}
        )
        self._postings_cache: dict[str, list[Posting]] = {}
        if posting_arrays is not None:
            self._posting_doc_ids, self._posting_tfs = posting_arrays
        else:
            for term, plist in postings.items():
                self._posting_doc_ids[term] = np.array([x.doc_id for x in plist], dtype=np.int32)
                self._posting_tfs[term] = np.array([x.tf for x in plist], dtype=np.int32)
                if has_positions:
                    self._posting_positions[term] = tuple(x.positions for x in plist)
                self._postings_cache[term] = plist

    def get_posting_arrays(self, term: str) -> tuple[Int32Array, Int32Array] | None:
        """Повертає пару масивів NumPy `(doc_ids, tfs)` для терміна або `None`."""
        doc_ids = self._posting_doc_ids.get(term)
        return (doc_ids, self._posting_tfs[term]) if doc_ids is not None else None

    def _materialize_postings(self, term: str) -> list[Posting]:
        cached = self._postings_cache.get(term)
        if cached is not None:
            return cached
        d_ids = self._posting_doc_ids[term].tolist()
        tfs = self._posting_tfs[term].tolist()
        if self.has_positions and term in self._posting_positions:
            plist = [
                Posting(doc_id=int(d), tf=int(f), positions=pos)
                for d, f, pos in zip(d_ids, tfs, self._posting_positions[term], strict=True)
            ]
        else:
            plist = [Posting(doc_id=int(d), tf=int(f)) for d, f in zip(d_ids, tfs, strict=True)]
        self._postings_cache[term] = plist
        return plist

    def __len__(self) -> int:
        return len(self._posting_doc_ids)

    def __contains__(self, term: object) -> bool:
        return term in self._posting_doc_ids

    def __getitem__(self, term: str) -> list[Posting]:
        if term not in self._posting_doc_ids:
            raise KeyError(term)
        return self._materialize_postings(term)

    def __iter__(self) -> Iterator[str]:
        return iter(self._posting_doc_ids)

    def __repr__(self) -> str:
        return (
            f"Index(num_docs={self.num_docs}, terms={len(self)}, "
            f"avg_doc_length={self.avg_doc_length:.1f}, has_positions={self.has_positions})"
        )

    @property
    def num_docs(self) -> int:
        return len(self._docs)

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def vocab_size(self) -> int:
        return len(self._posting_doc_ids)

    @property
    def docs(self) -> list[DocMeta]:
        return self._docs

    @property
    def postings(self) -> Mapping[str, list[Posting]]:
        return self

    @property
    def posting_positions(self) -> Mapping[str, tuple[tuple[int, ...], ...]]:
        """Positions by term in the current index, if phrase indexing was enabled."""
        return self._posting_positions

    @cached_property
    def avg_doc_length(self) -> float:
        return float(self.doc_lengths.mean()) if self.doc_lengths.size > 0 else 0.0

    @property
    def total_postings(self) -> int:
        return sum(int(arr.shape[0]) for arr in self._posting_doc_ids.values())

    @property
    def total_tokens(self) -> int:
        return int(self.doc_lengths.sum()) if self.doc_lengths.size > 0 else 0

    def df(self, term: str) -> int:
        arr = self._posting_doc_ids.get(term)
        return int(arr.shape[0]) if arr is not None else 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    def read_doc_text(self, doc_id: int) -> str:
        """Читає текст документа з диска за `corpus_root` та `rel_path` (із кешем у пам'яті)."""
        if self._closed:
            raise RuntimeError("Спроба читання з закритого об'єкта Index")
        cached = self._doc_text_cache.get(doc_id)
        if cached is not None:
            return cached
        if self.corpus_root is None:
            return ""
        meta = self._docs[doc_id]
        if "::" in meta.rel_path:
            file_part, line_str = meta.rel_path.rsplit("::", 1)
            target = self.corpus_root / file_part
            if not target.is_file():
                return ""
            try:
                line_idx = int(line_str) - 1
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                if 0 <= line_idx < len(lines):
                    record = json.loads(lines[line_idx])
                    text = f"{record.get('title', '')}\n{record.get('text', '')}".strip()
                    self._doc_text_cache[doc_id] = text
                    return text
            except Exception:
                return ""
            return ""
        target = self.corpus_root / meta.rel_path
        if not target.is_file():
            return ""
        text = target.read_text(encoding="utf-8", errors="replace")
        self._doc_text_cache[doc_id] = text
        return text

    def close(self) -> None:
        """Звільняє кешовані тексти документів та позначає індекс закритим."""
        self._doc_text_cache.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


# Public name retained for clients that describe the data structure by its role.
InvertedIndex = Index


@contextmanager
def open_index(
    path: Path | str,
    *,
    corpus_root: Path | None = None,
) -> Generator[Index, None, None]:
    """Менеджер контексту для безпечного відкриття індексу з гарантованим звільненням ресурсів."""
    from findex.storage import load_index

    index = load_index(Path(path), corpus_root=corpus_root)
    try:
        yield index
    finally:
        index.close()


@timed
def build_inverted_index(
    corpus_dir: Path | str,
    *,
    include_positions: bool = True,
    executor: ExecutorMode = "serial",
    workers: int | None = None,
    tokenizer_fn: TokenizerFn = tokenize,
    worker_fn: Any = None,
    on_document: Any = None,
) -> Index:
    """Будує інвертований індекс для всіх документів у директорії `corpus_dir`."""
    root = Path(corpus_dir)
    paths = collect_corpus_paths(root)
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Невідомий режим виконавця: {executor!r}")
    n_workers = max(1, workers or (os.cpu_count() or 1))
    if executor == "serial" or not paths:
        chunks: list[list[Path]] = [paths]
    else:
        chunk_size = (len(paths) + n_workers - 1) // n_workers
        chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
    tasks: list[tuple[list[Path], Path, int, bool, Any]] = []
    start = 0
    for chunk in chunks:
        tasks.append((chunk, root, start, include_positions, worker_fn))
        start += len(chunk)
    if executor == "threads" and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            partials: list[PartialIndex] = list(pool.map(_run_partial_task, tasks))
    elif executor == "processes" and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            partials = list(pool.map(_run_partial_task, tasks))
    else:
        partials = [_run_partial_task(task) for task in tasks]
    merged = merge(partials, corpus_root=root)
    if on_document is not None:
        for _ in merged.docs:
            on_document(0)
    if tokenizer_fn is not tokenize:
        raw_docs = list(iter_documents(root))
        tokenized = [
            _tokenize_default_or_custom(d, include_positions, tokenizer_fn) for d in raw_docs
        ]
        return _index_from_tokenized(tokenized, include_positions, root)
    return merged


def _index_from_tokenized(
    tokenized: list[tuple[Document, list[str], dict[str, int], dict[str, tuple[int, ...]]]],
    include_positions: bool,
    root: Path,
) -> Index:
    docs: list[DocMeta] = []
    postings: dict[str, list[Posting]] = defaultdict(list)
    for doc_id, (doc, tokens, tf_map, pos_map) in enumerate(tokenized):
        docs.append(DocMeta(doc_id, doc.rel_path, doc.title, len(tokens)))
        for term, tf in tf_map.items():
            postings[term].append(
                Posting(doc_id, tf, pos_map.get(term, ()) if include_positions else ())
            )
    return Index(dict(postings), docs, has_positions=include_positions, corpus_root=root)


def build_partial(
    paths: list[Path] | tuple[Path, ...],
    corpus_root: Path | None = None,
    *,
    start_doc_id: int = 0,
    include_positions: bool = False,
) -> PartialIndex:
    """Будує незалежний частковий індекс для відсортованого набору шляхів."""
    _ = corpus_root
    tokenized = [
        _tokenize_document_worker(doc, include_positions)
        for path in paths
        for doc in iter_documents(path)
    ]
    docs: list[DocMeta] = []
    postings: dict[str, list[Posting]] = defaultdict(list)
    for offset, (doc, tokens, tf_map, pos_map) in enumerate(tokenized):
        doc_id = start_doc_id + offset
        docs.append(DocMeta(doc_id, doc.rel_path, doc.title, len(tokens)))
        for term, tf in tf_map.items():
            postings[term].append(
                Posting(doc_id, tf, pos_map.get(term, ()) if include_positions else ())
            )
    return PartialIndex(docs, dict(postings), include_positions)


def _run_partial_task(
    task: tuple[list[Path], Path, int, bool, Any],
) -> PartialIndex:
    paths, root, start, include_positions, worker_fn = task
    if worker_fn is not None:
        return worker_fn(paths, root, start_doc_id=start, include_positions=include_positions)
    return build_partial(paths, root, start_doc_id=start, include_positions=include_positions)


def merge(partials: list[PartialIndex], *, corpus_root: Path | None = None) -> Index:
    """Зливає часткові індекси у щільну нумерацію документів із збереженням порядку."""
    docs: list[DocMeta] = []
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_id_map: dict[int, int] = {}
    for partial in partials:
        for meta in partial.docs:
            new_id = len(docs)
            doc_id_map[meta.doc_id] = new_id
            docs.append(DocMeta(new_id, meta.rel_path, meta.title, meta.length))
        for term, items in partial.postings.items():
            postings[term].extend(items)
    for term, items in postings.items():
        postings[term] = [Posting(doc_id_map[p.doc_id], p.tf, p.positions) for p in items]
        postings[term].sort(key=lambda p: p.doc_id)
    return Index(
        dict(postings),
        docs,
        has_positions=bool(partials and partials[0].has_positions),
        corpus_root=corpus_root,
    )


def get_peak_rss_bytes() -> int:
    """Поточний RSS процесу (Windows) або максимальний RSS із resource (Unix)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform != "darwin" else 1))


def build_index_profiled(
    corpus_dir: Path | str,
    *,
    include_positions: bool = True,
    workers: int | None = None,
    executor: ExecutorMode = "serial",
    on_document: Any = None,
) -> BuildProfile:
    start = time.perf_counter()
    before = get_peak_rss_bytes()
    index = build_inverted_index(
        corpus_dir,
        include_positions=include_positions,
        workers=workers,
        executor=executor,
        on_document=on_document,
    )
    elapsed = time.perf_counter() - start
    return BuildProfile(index, elapsed, 0.0, max(before, get_peak_rss_bytes()))


def benchmark_posting_storage(corpus_dir: Path | str) -> list[PostingStorageBenchmark]:
    """Порівнює object lists, NumPy arrays та компактні tuples за пік-алокаціями Python."""
    index = build_inverted_index(corpus_dir, include_positions=False)
    count = index.total_postings
    rows: list[PostingStorageBenchmark] = []
    for name, factory in (
        (
            "Python Posting objects",
            lambda: [Posting(p.doc_id, p.tf, p.positions) for term in index for p in index[term]],
        ),
        ("Python tuples", lambda: [(p.doc_id, p.tf) for term in index for p in index[term]]),
        ("NumPy int32 arrays", lambda: [index.get_posting_arrays(t) for t in index]),
    ):
        tracemalloc.start()
        started = time.perf_counter()
        value = factory()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del value
        rows.append(PostingStorageBenchmark(name, count, elapsed, peak))
    return rows


def _tokenize_default_or_custom(
    doc: Document,
    include_positions: bool,
    tokenizer_fn: TokenizerFn,
) -> tuple[Document, list[str], dict[str, int], dict[str, tuple[int, ...]]]:
    if tokenizer_fn is tokenize:
        return _tokenize_document_worker(doc, include_positions)

    tokens = list(tokenizer_fn(doc.text))
    tf_map: dict[str, int] = defaultdict(int)
    pos_map: dict[str, list[int]] = defaultdict(list)

    if include_positions:
        for pos, token in enumerate(tokens):
            tf_map[token] += 1
            pos_map[token].append(pos)
    else:
        for token in tokens:
            tf_map[token] += 1

    frozen_pos = {term: tuple(positions) for term, positions in pos_map.items()}
    return doc, tokens, dict(tf_map), frozen_pos
