"""Reproducible Lab 8 scalar/NumPy and ten-times mmap benchmark."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

from findex.index import DocMeta, Index, get_peak_rss_bytes
from findex.search import clear_query_cache, search, search_scalar
from findex.storage import load_index, load_index_memmap, save_index_memmap


def median_ms(fn, index: Index, query: str, rounds: int = 9) -> float:
    fn(index, query, top_k=10)  # warm text cache and imports
    samples = []
    for _ in range(rounds):
        clear_query_cache()
        started = time.perf_counter()
        fn(index, query, top_k=10)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> None:
    base = load_index(Path("data/index.bin"), corpus_root=Path("data/corpus_bench"))
    terms = sorted(base, key=lambda term: (base.df(term), term))
    rare, frequent = terms[0], terms[-1]
    queries = [
        ("частий терм", frequent),
        ("рідкісний терм", rare),
        ("три терми", "whale OR sea OR captain"),
    ]
    rows = []
    for label, query in queries:
        scalar = median_ms(search_scalar, base, query)
        vector = median_ms(search, base, query)
        rows.append((label, query, scalar, vector, scalar / max(vector, 1e-9)))

    # Expand postings tenfold without duplicating the source files.
    docs = [
        DocMeta(replica * base.num_docs + d.doc_id, d.rel_path, d.title, d.length)
        for replica in range(10)
        for d in base.docs
    ]
    ids_map: dict[str, np.ndarray] = {}
    tfs_map: dict[str, np.ndarray] = {}
    for term in base:
        arrays = base.get_posting_arrays(term)
        assert arrays is not None
        ids_map[term] = np.concatenate(
            [arrays[0] + replica * base.num_docs for replica in range(10)]
        ).astype(np.int32, copy=False)
        tfs_map[term] = np.tile(arrays[1], 10).astype(np.int32, copy=False)
    expanded = Index({}, docs, posting_arrays=(ids_map, tfs_map), has_positions=False)
    mmap_dir = Path(".lab8-work/index-10x-mmap")
    save_index_memmap(expanded, mmap_dir)
    del expanded, ids_map, tfs_map, docs
    rss_before = get_peak_rss_bytes()
    mapped = load_index_memmap(mmap_dir)
    mmap_ms = median_ms(search, mapped, "whale OR sea OR captain")
    rss_after = get_peak_rss_bytes()
    mapped.close()
    total_bytes = sum(p.stat().st_size for p in mmap_dir.glob("*.npy"))

    report = [
        "# Lab 8: scorer and mmap measurements",
        "",
        f"Baseline: {base.num_docs:,} docs, {base.total_postings:,} postings; "
        f"Python {sys.version.split()[0]}, NumPy {np.__version__}.",
        "Medians of 9 runs after warmup; scorer-only API includes parsing, top-k, "
        "and result metadata.",
        "",
        "| Workload | Query | Scalar (ms) | NumPy (ms) | Speedup |",
        "|---|---|---:|---:|---:|",
    ]
    report.extend(
        f"| {label} | `{query}` | {scalar:.3f} | {vector:.3f} | {ratio:.2f}× |"
        for label, query, scalar, vector, ratio in rows
    )
    report.extend(
        [
            "",
            f"10× expanded index: {mapped.num_docs:,} docs, {mapped.total_postings:,} postings; "
            f"`.npy` payload {total_bytes / (1024 * 1024):.2f} MiB; "
            f"mapped search median {mmap_ms:.3f} ms; process peak RSS before/after map "
            f"{rss_before / (1024 * 1024):.1f}/{rss_after / (1024 * 1024):.1f} MiB.",
            "The mmap holds contiguous int32 posting IDs, term frequencies, and document lengths. "
            "A small-list query sees little benefit because Python call and query parsing overhead "
            "dominate; mmap avoids eagerly copying all posting data into Python-managed arrays.",
            "",
        ]
    )
    Path("docs/lab8_performance.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
