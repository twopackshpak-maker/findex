"""Бенчмарк паралельної індексації (Lab 5) та побудова графіка прискорення."""

from __future__ import annotations

import os
import platform
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from findex.corpus import collect_corpus_paths  # noqa: E402
from findex.index import ExecutorMode, build_index_profiled  # noqa: E402


def get_cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def ensure_benchmark_corpus(preferred_dir: Path) -> Path:
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
        "bm25",
        "tfidf",
        "search",
        "query",
        "executor",
        "worker",
        "spawn",
        "corpus",
    ]
    paragraph = " ".join(vocab * 25) + ".\n"
    for i in range(1200):
        header = f"Document {i}: benchmark text for parallel indexing\n"
        body = paragraph * (8 + (i % 5))
        (fallback_dir / f"doc_{i:04d}.txt").write_text(header + body, encoding="utf-8")
    return fallback_dir


def measure_cell(
    corpus_dir: Path,
    executor: ExecutorMode,
    workers: int,
    runs: int = 3,
) -> dict[str, float]:
    walls: list[float] = []
    cpus: list[float] = []
    merges: list[float] = []
    rss_mbs: list[float] = []

    for _ in range(runs):
        prof = build_index_profiled(
            corpus_dir,
            include_positions=True,
            workers=workers,
            executor=executor,
        )
        walls.append(prof.wall_time_sec)
        cpus.append(prof.cpu_time_sec)
        merges.append(prof.merge_time_sec)
        rss_mbs.append(prof.peak_rss_bytes / (1024 * 1024))

    return {
        "wall": statistics.median(walls),
        "cpu": statistics.median(cpus),
        "merge": statistics.median(merges),
        "rss_mb": statistics.median(rss_mbs),
    }


def main() -> None:
    corpus_dir = ensure_benchmark_corpus(Path("data/corpus"))
    num_docs = len(collect_corpus_paths(corpus_dir))
    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})
    gil_checker = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = gil_checker() if callable(gil_checker) else True

    print(f"CPU: {get_cpu_name()}")
    print(f"Логічних ядер (os.cpu_count): {cpu_count}")
    print(f"Python: {sys.version.split()[0]} | GIL enabled: {gil_enabled}")
    print(f"Корпус: {corpus_dir} ({num_docs} документів)\n")

    # Прогрів дискового кешу ОС
    build_index_profiled(corpus_dir, include_positions=True, workers=1, executor="serial")

    serial_base = measure_cell(corpus_dir, "serial", 1, runs=3)
    t1 = serial_base["wall"]

    rows: list[tuple[str, int, dict[str, float], float]] = [
        ("serial", 1, serial_base, 1.0),
    ]

    modes: tuple[ExecutorMode, ...] = ("threads", "processes")
    for mode in modes:
        for w in worker_counts:
            stats = measure_cell(corpus_dir, mode, w, runs=3)
            speedup = t1 / stats["wall"] if stats["wall"] > 0 else 0.0
            rows.append((mode, w, stats, speedup))

    print(
        "| Виконавець | Воркери | Wall time (с) | CPU time (с) "
        "| Пік RSS (МБ) | Merge (с) | Прискорення |"
    )
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode, w, st, sp in rows:
        print(
            f"| `{mode}` | {w} | {st['wall']:.3f} | {st['cpu']:.3f} | "
            f"{st['rss_mb']:.1f} | {st['merge']:.4f} | {sp:.2f}x |"
        )

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    chart_path = assets_dir / "speedup_lab5.png"

    thread_pts = [(w, sp) for m, w, _, sp in rows if m == "threads"]
    proc_pts = [(w, sp) for m, w, _, sp in rows if m == "processes"]

    plt.figure(figsize=(8.5, 5.2), dpi=140)
    max_w = max(worker_counts)
    ideal_x = list(range(1, max_w + 1))
    plt.plot(ideal_x, ideal_x, "--", color="gray", label="Ідеальне лінійне прискорення (y = x)")
    plt.plot(
        [p[0] for p in proc_pts],
        [p[1] for p in proc_pts],
        marker="o",
        linewidth=2,
        label="ProcessPoolExecutor (processes)",
    )
    plt.plot(
        [p[0] for p in thread_pts],
        [p[1] for p in thread_pts],
        marker="s",
        linewidth=2,
        label="ThreadPoolExecutor (threads, з GIL)",
    )
    plt.xticks(worker_counts)
    plt.xlabel("Кількість воркерів (N)")
    plt.ylabel("Прискорення (T_serial / T_N)")
    plt.title(f"Прискорення індексації findex ({num_docs} док., {cpu_count} ядер)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    print(f"\nГрафік збережено у {chart_path}")


if __name__ == "__main__":
    main()
