"""Дослідження Python 3.13t без GIL: індексація, гонитва даних (data race) та I/O-потоки."""

from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from findex.corpus import collect_corpus_paths
from findex.index import build_index_profiled


def ensure_corpus(preferred_dir: Path) -> Path:
    if preferred_dir.is_dir() and len(collect_corpus_paths(preferred_dir)) >= 200:
        return preferred_dir

    fallback_dir = Path("data/corpus_bench")
    if fallback_dir.is_dir() and len(collect_corpus_paths(fallback_dir)) >= 1000:
        return fallback_dir

    fallback_dir.mkdir(parents=True, exist_ok=True)
    vocab = [
        "python",
        "concurrency",
        "parallelism",
        "thread",
        "process",
        "interpreter",
        "lock",
        "memory",
        "pickle",
        "serialization",
        "amdahl",
        "speedup",
        "latency",
        "throughput",
        "inverted",
        "index",
        "posting",
        "document",
        "token",
        "ranking",
    ]
    paragraph = " ".join(vocab * 25) + ".\n"
    for i in range(1200):
        header = f"Document {i}: benchmark text for free-threaded Python\n"
        body = paragraph * (8 + (i % 5))
        (fallback_dir / f"doc_{i:04d}.txt").write_text(header + body, encoding="utf-8")
    return fallback_dir


def demo_data_race(num_threads: int = 8, increments_per_thread: int = 50_000) -> None:
    """Показує втрату оновлень у спільному словнику/лічильнику без блокування на 3.13t."""
    shared_counts: dict[str, int] = {"python": 0}
    barrier = threading.Barrier(num_threads)

    def unsafe_worker() -> None:
        barrier.wait()
        for _ in range(increments_per_thread):
            cur = shared_counts["python"]
            shared_counts["python"] = cur + 1

    threads = [threading.Thread(target=unsafe_worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = num_threads * increments_per_thread
    actual_unsafe = shared_counts["python"]

    safe_counts: dict[str, int] = {"python": 0}
    lock = threading.Lock()
    barrier_safe = threading.Barrier(num_threads)

    def safe_worker() -> None:
        barrier_safe.wait()
        for _ in range(increments_per_thread):
            with lock:
                safe_counts["python"] += 1

    threads_safe = [threading.Thread(target=safe_worker) for _ in range(num_threads)]
    for t in threads_safe:
        t.start()
    for t in threads_safe:
        t.join()

    print("=== Демонстрація гонитви даних (Data Race) ===")
    print(f"Очікуване значення: {expected:,}")
    print(f"Без Lock (race):    {actual_unsafe:,} (втрачено {expected - actual_unsafe:,} оновлень)")
    print(f"З threading.Lock:   {safe_counts['python']:,}")


def demo_io_scaling(corpus_dir: Path, workers: int) -> None:
    """Порівнює послідовне і багатопотокове читання файлів (I/O-bound)."""
    paths = collect_corpus_paths(corpus_dir)

    def read_one(p: Path) -> int:
        return len(p.read_bytes())

    t0 = time.perf_counter()
    serial_bytes = sum(read_one(p) for p in paths)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_bytes = sum(pool.map(read_one, paths))
    t_threads = time.perf_counter() - t0

    assert serial_bytes == parallel_bytes
    print("\n=== I/O-bound бенчмарк (читання файлів з диска) ===")
    print(f"Файлів: {len(paths)}, обсяг: {serial_bytes / (1024 * 1024):.2f} МБ")
    print(f"Serial I/O:         {t_serial:.4f} с")
    print(f"Threads ({workers}): {t_threads:.4f} с (прискорення {t_serial / t_threads:.2f}x)")


def main() -> None:
    gil_checker = getattr(sys, "_is_gil_enabled", None)
    gil_status = gil_checker() if callable(gil_checker) else True
    print(f"Python version: {sys.version}")
    print(f"sys._is_gil_enabled() -> {gil_status}\n")

    corpus_dir = ensure_corpus(Path("data/corpus"))
    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})

    print("=== Індексація threads на поточному інтерпретаторі ===")
    t1 = statistics.median(
        build_index_profiled(
            corpus_dir, include_positions=True, workers=1, executor="serial"
        ).wall_time_sec
        for _ in range(3)
    )
    for w in worker_counts:
        wall = statistics.median(
            build_index_profiled(
                corpus_dir, include_positions=True, workers=w, executor="threads"
            ).wall_time_sec
            for _ in range(3)
        )
        print(f"threads (workers={w}): wall={wall:.3f} с, speedup={t1 / wall:.2f}x")

    print()
    demo_data_race()
    demo_io_scaling(corpus_dir, workers=cpu_count)


if __name__ == "__main__":
    main()
