"""Об'єктна модель інвертованого індексу, паралельна збірка (build_partial/merge) та open_index."""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import math
import multiprocessing
import os
import sys
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from findex.corpus import collect_corpus_paths, iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize

logger = logging.getLogger("findex.index")

ExecutorMode = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Метадані документа для відображення результатів пошуку."""

    doc_id: int
    rel_path: str
    title: str
    length: int


@dataclass(frozen=True, slots=True)
class Posting:
    """Елемент списку постінгів (компактний запис зі slots=True)."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PartialIndex:
    """Частковий індекс для окремого діапазону doc_id, придатний для передачі через pickle."""

    postings: dict[str, list[Posting]]
    docs: list[DocMeta]
    has_positions: bool = False


class PartialBuilder(Protocol):
    """Протокол воркера збірки часткового індексу."""

    def __call__(
        self,
        paths: Sequence[Path],
        corpus_root: Path | None = None,
        *,
        start_doc_id: int = 0,
        include_positions: bool = False,
    ) -> PartialIndex: ...


@dataclass(frozen=True)
class RegularPosting:
    """Варіант 1 для бенчмарку пам'яті: звичайний dataclass без slots=True."""

    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class ArrayPostings:
    """Варіант 3 для бенчмарку пам'яті: пара C-масивів uint32."""

    doc_ids: array[int]
    freqs: array[int]

    def __len__(self) -> int:
        return len(self.doc_ids)


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс як незмінне відображення `term -> list[Posting]`."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: list[DocMeta],
        *,
        has_positions: bool = False,
        corpus_root: Path | None = None,
    ) -> None:
        self._postings: dict[str, list[Posting]] = postings
        self._docs: list[DocMeta] = docs
        self.has_positions: bool = has_positions
        self.corpus_root: Path | None = corpus_root
        self._doc_text_cache: dict[int, str] = {}
        self._closed: bool = False

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

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
        return len(self._postings)

    @property
    def docs(self) -> list[DocMeta]:
        return self._docs

    @property
    def postings(self) -> Mapping[str, list[Posting]]:
        return self._postings

    @cached_property
    def avg_doc_length(self) -> float:
        """Середня довжина документа в токенах (кешується після першого обчислення)."""
        if not self._docs:
            return 0.0
        return sum(doc.length for doc in self._docs) / len(self._docs)

    @property
    def total_postings(self) -> int:
        return sum(len(plist) for plist in self._postings.values())

    @property
    def total_tokens(self) -> int:
        return sum(doc.length for doc in self._docs)

    def df(self, term: str) -> int:
        """Document frequency (кількість документів, у яких зустрічається `term`)."""
        plist = self._postings.get(term)
        return len(plist) if plist is not None else 0

    def read_doc_text(self, doc_id: int) -> str | None:
        """Читає текст документа з `corpus_root` (з кешуванням у межах сесії індексу)."""
        if self._closed:
            raise RuntimeError("Спроба читання із закритого об'єкта Index")
        if doc_id in self._doc_text_cache:
            return self._doc_text_cache[doc_id]
        if self.corpus_root is None:
            return None

        rel_path = self._docs[doc_id].rel_path
        if ".jsonl#" in rel_path:
            file_part, line_str = rel_path.rsplit("#", 1)
            jsonl_path = (
                self.corpus_root if self.corpus_root.is_file() else self.corpus_root / file_part
            )
            if jsonl_path.is_file() and line_str.isdigit():
                target_line = int(line_str)
                with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for idx, raw in enumerate(handle, start=1):
                        if idx == target_line:
                            raw_obj: Any = json.loads(raw)
                            if isinstance(raw_obj, dict):
                                rec = cast(dict[str, Any], raw_obj)
                                text = str(rec.get("text", ""))
                                self._doc_text_cache[doc_id] = text
                                return text
                            break
            return None

        file_path = self.corpus_root / rel_path
        if not file_path.is_file():
            return None
        text = file_path.read_text(encoding="utf-8", errors="replace")
        self._doc_text_cache[doc_id] = text
        return text

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Звільняє кешовані тексти документів та позначає індекс закритим."""
        self._doc_text_cache.clear()
        self._closed = True


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


@dataclass(frozen=True, slots=True)
class StorageBenchmarkRow:
    """Результат заміру одного способу зберігання постінгів під tracemalloc."""

    variant: str
    vocab_size: int
    total_postings: int
    elapsed_sec: float
    peak_memory_bytes: int

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class BuildProfile:
    """Детальний профіль збірки індексу (wall/CPU/merge час і піковий RSS батьківського процесу)."""

    index: Index
    wall_time_sec: float
    cpu_time_sec: float
    merge_time_sec: float
    peak_rss_bytes: int


def get_peak_rss_bytes() -> int:
    """Повертає піковий обсяг фізичної пам'яті (RSS) поточного процесу у байтах."""
    if sys.platform != "win32":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    kernel32 = ctypes.cdll.kernel32
    psapi = ctypes.cdll.psapi

    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p

    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_bool

    handle = get_current_process()
    if get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return 0


def _extract_title(rel_path: str, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:100]
    return Path(rel_path).stem


def _index_single_text(
    text: str,
    *,
    doc_id: int,
    rel_path: str,
    title: str,
    include_positions: bool,
    postings_map: dict[str, list[Posting]],
    docs_meta: list[DocMeta],
) -> None:
    if include_positions:
        pos_by_term: dict[str, list[int]] = defaultdict(list)
        doc_len = 0
        for doc_len, token in enumerate(tokenize(text), start=1):
            pos_by_term[token].append(doc_len - 1)

        for term, pos_list in pos_by_term.items():
            postings_map[term].append(
                Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
            )
    else:
        tf_counter = Counter(tokenize(text))
        doc_len = sum(tf_counter.values())
        for term, tf in tf_counter.items():
            postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

    docs_meta.append(
        DocMeta(
            doc_id=doc_id,
            rel_path=rel_path,
            title=title,
            length=doc_len,
        )
    )


def build_partial(
    paths: Sequence[Path],
    corpus_root: Path | None = None,
    *,
    start_doc_id: int = 0,
    include_positions: bool = False,
) -> PartialIndex:
    """Будує частковий індекс для заданого списку шляхів (функція рівня модуля для spawn)."""
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: list[DocMeta] = []
    local_offset = 0
    base_root = (
        corpus_root.parent if (corpus_root is not None and corpus_root.is_file()) else corpus_root
    )

    for path in paths:
        rel_path = path.relative_to(base_root).as_posix() if base_root is not None else path.name
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    raw_obj: Any = json.loads(stripped)
                    if not isinstance(raw_obj, dict):
                        continue
                    record = cast(dict[str, Any], raw_obj)
                    raw_title = str(record.get("title", "")).strip()
                    raw_body = str(record.get("text", "")).strip()
                    combined = (
                        f"{raw_title}\n{raw_body}".strip()
                        if raw_title and raw_title not in raw_body
                        else (raw_body or raw_title)
                    )
                    if not combined:
                        continue
                    doc_id = start_doc_id + local_offset
                    local_offset += 1
                    fallback_title = str(record.get("url", rel_path))
                    display_title = (raw_title or fallback_title)[:100]
                    _index_single_text(
                        combined,
                        doc_id=doc_id,
                        rel_path=f"{rel_path}#{line_no}",
                        title=display_title,
                        include_positions=include_positions,
                        postings_map=postings_map,
                        docs_meta=docs_meta,
                    )
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue

        doc_id = start_doc_id + local_offset
        local_offset += 1
        _index_single_text(
            text,
            doc_id=doc_id,
            rel_path=rel_path,
            title=_extract_title(rel_path, text),
            include_positions=include_positions,
            postings_map=postings_map,
            docs_meta=docs_meta,
        )

    return PartialIndex(
        postings=dict(postings_map),
        docs=docs_meta,
        has_positions=include_positions,
    )


def merge(
    partials: Iterable[PartialIndex],
    *,
    corpus_root: Path | None = None,
) -> Index:
    """Об'єднує впорядковану послідовність `PartialIndex` у єдиний `Index`."""
    partial_list = list(partials)
    if not partial_list:
        return Index(postings={}, docs=[], has_positions=False, corpus_root=corpus_root)

    has_positions = partial_list[0].has_positions
    if any(p.has_positions != has_positions for p in partial_list):
        raise ValueError("Неможливо об'єднати часткові індекси з різним режимом has_positions")

    merged_docs: list[DocMeta] = []
    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    id_remap: dict[int, int] = {}
    needs_remap = False

    for part in partial_list:
        for doc in part.docs:
            dense_id = len(merged_docs)
            if doc.doc_id != dense_id:
                needs_remap = True
            id_remap[doc.doc_id] = dense_id
            merged_docs.append(
                DocMeta(
                    doc_id=dense_id,
                    rel_path=doc.rel_path,
                    title=doc.title,
                    length=doc.length,
                )
                if doc.doc_id != dense_id
                else doc
            )

        if not needs_remap:
            for term, plist in part.postings.items():
                merged_postings[term].extend(plist)
        else:
            for term, plist in part.postings.items():
                merged_postings[term].extend(
                    Posting(
                        doc_id=id_remap[p.doc_id],
                        tf=p.tf,
                        positions=p.positions,
                    )
                    for p in plist
                )

    return Index(
        postings=dict(merged_postings),
        docs=merged_docs,
        has_positions=has_positions,
        corpus_root=corpus_root,
    )


def _partition_paths(paths: Sequence[Path], workers: int) -> list[tuple[int, list[Path]]]:
    """Розбиває шляхи на `min(workers, len(paths))` суміжних діапазонів `(start_doc_id, chunk)`."""
    if not paths:
        return []
    num_chunks = min(workers, len(paths))
    chunk_size = math.ceil(len(paths) / num_chunks)
    chunks: list[tuple[int, list[Path]]] = []
    for start in range(0, len(paths), chunk_size):
        chunks.append((start, list(paths[start : start + chunk_size])))
    return chunks


def _run_chunks(
    chunks: list[tuple[int, list[Path]]],
    corpus_dir: Path,
    *,
    include_positions: bool,
    executor: ExecutorMode,
    on_document: Callable[[int], None] | None,
    worker_fn: PartialBuilder,
) -> list[PartialIndex]:
    if not chunks:
        return []

    if executor == "serial" or len(chunks) == 1:
        partials: list[PartialIndex] = []
        for start_id, chunk_paths in chunks:
            part = worker_fn(
                chunk_paths,
                corpus_dir,
                start_doc_id=start_id,
                include_positions=include_positions,
            )
            partials.append(part)
            if on_document is not None:
                for doc in part.docs:
                    on_document(doc.doc_id)
        return partials

    ordered: list[PartialIndex | None] = [None] * len(chunks)
    pool: ProcessPoolExecutor | ThreadPoolExecutor
    if executor == "threads":
        pool = ThreadPoolExecutor(max_workers=len(chunks))
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp_ctx)

    futures_to_idx: dict[Future[PartialIndex], int] = {}
    try:
        for idx, (start_id, chunk_paths) in enumerate(chunks):
            fut = pool.submit(
                worker_fn,
                chunk_paths,
                corpus_dir,
                start_doc_id=start_id,
                include_positions=include_positions,
            )
            futures_to_idx[fut] = idx

        for fut in as_completed(futures_to_idx):
            part = fut.result()
            chunk_idx = futures_to_idx[fut]
            ordered[chunk_idx] = part
            if on_document is not None:
                for doc in part.docs:
                    on_document(doc.doc_id)
    except BaseException:
        for f in futures_to_idx:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    return [p for p in ordered if p is not None]


def build_index_profiled(
    corpus_dir: Path,
    *,
    include_positions: bool = False,
    workers: int | None = None,
    executor: ExecutorMode = "serial",
    on_document: Callable[[int], None] | None = None,
    worker_fn: PartialBuilder | None = None,
) -> BuildProfile:
    """Будує індекс у заданому режимі виконання та повертає заміри wall/CPU/merge часу і RSS."""
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Невідомий тип виконавця: {executor!r}")

    effective_workers = (
        (1 if executor == "serial" else (os.cpu_count() or 1)) if workers is None else workers
    )
    if effective_workers < 1:
        raise ValueError(f"Кількість воркерів має бути >= 1, отримано {effective_workers}")

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    paths = collect_corpus_paths(corpus_dir)
    chunks = _partition_paths(paths, effective_workers)
    fn: PartialBuilder = worker_fn if worker_fn is not None else build_partial

    partials = _run_chunks(
        chunks,
        corpus_dir,
        include_positions=include_positions,
        executor=executor,
        on_document=on_document,
        worker_fn=fn,
    )

    t_merge_start = time.perf_counter()
    index = merge(partials, corpus_root=corpus_dir)
    merge_time = time.perf_counter() - t_merge_start

    wall_time = time.perf_counter() - t_wall_start
    cpu_time = time.process_time() - t_cpu_start
    peak_rss = get_peak_rss_bytes()

    logger.debug(
        "Побудовано індекс (%s, workers=%d): docs=%d, vocab=%d, wall=%.3fs, merge=%.3fs",
        executor,
        effective_workers,
        index.num_docs,
        len(index),
        wall_time,
        merge_time,
    )

    return BuildProfile(
        index=index,
        wall_time_sec=wall_time,
        cpu_time_sec=cpu_time,
        merge_time_sec=merge_time,
        peak_rss_bytes=peak_rss,
    )


@timed
def build_inverted_index(
    corpus_dir: Path,
    *,
    include_positions: bool = False,
    workers: int | None = None,
    executor: ExecutorMode = "serial",
    on_document: Callable[[int], None] | None = None,
    worker_fn: PartialBuilder | None = None,
) -> Index:
    """Будує об'єкт `Index` через `build_partial` та `merge` у режимі serial/threads/processes."""
    return build_index_profiled(
        corpus_dir,
        include_positions=include_positions,
        workers=workers,
        executor=executor,
        on_document=on_document,
        worker_fn=worker_fn,
    ).index


def benchmark_posting_storage(corpus_dir: Path) -> list[StorageBenchmarkRow]:
    """Порівнює 3 способи зберігання постінгів на тому самому корпусі під tracemalloc."""
    results: list[StorageBenchmarkRow] = []

    gc.collect()
    tracemalloc.start(1)
    t0 = time.perf_counter()
    reg_map: dict[str, list[RegularPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        for term, tf in Counter(tokenize(doc.text)).items():
            reg_map[term].append(RegularPosting(doc_id=doc_id, tf=tf))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append(
        StorageBenchmarkRow(
            variant="1. Звичайний @dataclass (без slots)",
            vocab_size=len(reg_map),
            total_postings=sum(len(v) for v in reg_map.values()),
            elapsed_sec=elapsed,
            peak_memory_bytes=peak,
        )
    )
    del reg_map

    gc.collect()
    tracemalloc.start(1)
    t0 = time.perf_counter()
    slot_map: dict[str, list[Posting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        for term, tf in Counter(tokenize(doc.text)).items():
            slot_map[term].append(Posting(doc_id=doc_id, tf=tf))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append(
        StorageBenchmarkRow(
            variant="2. @dataclass(frozen=True, slots=True)",
            vocab_size=len(slot_map),
            total_postings=sum(len(v) for v in slot_map.values()),
            elapsed_sec=elapsed,
            peak_memory_bytes=peak,
        )
    )
    del slot_map

    gc.collect()
    tracemalloc.start(1)
    t0 = time.perf_counter()
    arr_map: dict[str, ArrayPostings] = defaultdict(lambda: ArrayPostings(array("I"), array("I")))
    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        for term, tf in Counter(tokenize(doc.text)).items():
            bucket = arr_map[term]
            bucket.doc_ids.append(doc_id)
            bucket.freqs.append(tf)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append(
        StorageBenchmarkRow(
            variant="3. Пари array('I') (doc_ids + freqs)",
            vocab_size=len(arr_map),
            total_postings=sum(len(v) for v in arr_map.values()),
            elapsed_sec=elapsed,
            peak_memory_bytes=peak,
        )
    )
    del arr_map

    return results
