"""Збір статистики корпусу: лінивий однопрохідний конвеєр vs жадібний підхід."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import time
import tracemalloc

from findex.corpus import iter_documents
from findex.tokenizer import tokenize

logger = logging.getLogger("findex.stats")


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Результати аналізу корпусу та заміри продуктивності."""

    mode: str
    doc_count: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]
    elapsed_sec: float
    peak_memory_bytes: int

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


def collect_stats_lazy(corpus_dir: Path, top_k: int = 50) -> CorpusStats:
    """Однопрохідний потоковий збір статистики корпусу."""
    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    doc_count = 0
    total_tokens = 0
    vocab: Counter[str] = Counter()

    for doc in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            vocab[token] += 1

    elapsed = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.debug(
        "Lazy stats: docs=%d, tokens=%d, vocab=%d, peak_mb=%.2f",
        doc_count,
        total_tokens,
        len(vocab),
        peak_bytes / (1024 * 1024),
    )

    return CorpusStats(
        mode="Lazy (генератори)",
        doc_count=doc_count,
        total_tokens=total_tokens,
        vocab_size=len(vocab),
        top_terms=vocab.most_common(top_k),
        elapsed_sec=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def collect_stats_eager(corpus_dir: Path, top_k: int = 50) -> CorpusStats:
    """Навмисно «жадібна» реалізація через матеріалізацію всіх списків у RAM."""
    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    docs = list(iter_documents(corpus_dir))
    tokenized_docs = [list(tokenize(doc.text)) for doc in docs]
    all_tokens = [token for doc_tokens in tokenized_docs for token in doc_tokens]
    vocab = Counter(all_tokens)

    elapsed = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.debug(
        "Eager stats: docs=%d, tokens=%d, vocab=%d, peak_mb=%.2f",
        len(docs),
        len(all_tokens),
        len(vocab),
        peak_bytes / (1024 * 1024),
    )

    return CorpusStats(
        mode="Eager (списки)",
        doc_count=len(docs),
        total_tokens=len(all_tokens),
        vocab_size=len(vocab),
        top_terms=vocab.most_common(top_k),
        elapsed_sec=elapsed,
        peak_memory_bytes=peak_bytes,
    )
