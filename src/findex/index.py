"""Об'єктна модель інвертованого індексу (Mapping) та менеджер контексту open_index."""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize

logger = logging.getLogger("findex.index")


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
        file_path = self.corpus_root / self._docs[doc_id].rel_path
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


def _extract_title(rel_path: str, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:100]
    return Path(rel_path).stem


@timed
def build_inverted_index(
    corpus_dir: Path,
    *,
    include_positions: bool = False,
    on_document: Callable[[int], None] | None = None,
) -> Index:
    """Будує об'єкт `Index` одним проходом по лінивому пайплайну корпусу."""
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: list[DocMeta] = []

    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        if include_positions:
            pos_by_term: dict[str, list[int]] = defaultdict(list)
            doc_len = 0
            for doc_len, token in enumerate(tokenize(doc.text), start=1):
                pos_by_term[token].append(doc_len - 1)

            for term, pos_list in pos_by_term.items():
                postings_map[term].append(
                    Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
        else:
            tf_counter = Counter(tokenize(doc.text))
            doc_len = sum(tf_counter.values())
            for term, tf in tf_counter.items():
                postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

        docs_meta.append(
            DocMeta(
                doc_id=doc_id,
                rel_path=doc.doc_id,
                title=_extract_title(doc.doc_id, doc.text),
                length=doc_len,
            )
        )
        if on_document is not None:
            on_document(doc_id)

    logger.debug(
        "Побудовано індекс: docs=%d, vocab=%d, postings=%d, positions=%s",
        len(docs_meta),
        len(postings_map),
        sum(len(v) for v in postings_map.values()),
        include_positions,
    )

    return Index(
        postings=dict(postings_map),
        docs=docs_meta,
        has_positions=include_positions,
        corpus_root=corpus_dir,
    )


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
