from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.index import Index, build_inverted_index
from findex.search import clear_query_cache, search, search_scalar
from findex.storage import load_index_memmap, save_index_memmap


def test_memmap_round_trip_preserves_postings_and_ranking(
    sample_index_with_positions: Index, tmp_path: Path
) -> None:
    target = tmp_path / "mmap-index"
    save_index_memmap(sample_index_with_positions, target)
    mapped = load_index_memmap(target, corpus_root=sample_index_with_positions.corpus_root)
    try:
        assert isinstance(mapped.doc_lengths, np.memmap)
        assert mapped.doc_lengths.dtype == np.int32
        python_arrays = mapped.get_posting_arrays("python")
        assert python_arrays is not None
        assert python_arrays[0].dtype == np.int32
        assert [p.doc_id for p in mapped["python"]] == [0, 3]
        for query in ("python", "chrysanthemum", "python OR index OR java"):
            clear_query_cache()
            assert search(mapped, query, top_k=5) == search_scalar(mapped, query, top_k=5)
    finally:
        mapped.close()


@pytest.mark.benchmark(group="lab8")
def test_benchmark_search(sample_index: Index, benchmark: BenchmarkFixture) -> None:
    benchmark(search, sample_index, "python OR index OR java", top_k=5)


@pytest.mark.benchmark(group="lab8")
def test_benchmark_build_index(sample_corpus_dir: Path, benchmark: BenchmarkFixture) -> None:
    benchmark(build_inverted_index, sample_corpus_dir, include_positions=False)
