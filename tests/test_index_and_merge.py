"""Тести моделі Index (Mapping), open_index, збірки індексу та операцій merge."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from findex.corpus import count_corpus_files, iter_documents
from findex.index import Index, benchmark_posting_storage, open_index
from findex.search import search_boolean
from findex.stats import collect_stats_eager, collect_stats_lazy


def test_index_mapping_protocol_and_dunders(sample_index: Index) -> None:
    assert isinstance(sample_index, Mapping)
    assert sample_index.num_docs == 5
    assert sample_index.doc_count == 5
    assert len(sample_index) == sample_index.vocab_size
    assert "python" in sample_index
    assert "nonexistent_xyz" not in sample_index
    assert set(sample_index) == set(sample_index.keys())
    assert sample_index.df("python") == 2
    assert sample_index.df("chrysanthemum") == 1
    assert sample_index.df("missing") == 0
    assert sample_index.avg_doc_length > 0
    assert "Index(num_docs=5" in repr(sample_index)


def test_postings_sorted_and_positions_recorded(
    sample_index_with_positions: Index,
) -> None:
    for postings in sample_index_with_positions.values():
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids)
        assert len(doc_ids) == len(set(doc_ids))

    loop_postings = sample_index_with_positions["loop"]
    assert [p.doc_id for p in loop_postings] == [0, 3]
    assert all(len(p.positions) == p.tf and p.tf >= 1 for p in loop_postings)


def test_open_index_context_manager_closes_on_exception(
    saved_index_path: Path,
    sample_corpus_dir: Path,
) -> None:
    holder: list[Index] = []
    with (
        pytest.raises(ValueError, match="test failure"),
        open_index(saved_index_path, corpus_root=sample_corpus_dir) as idx,
    ):
        holder.append(idx)
        assert not idx.is_closed
        text = idx.read_doc_text(0)
        assert text is not None
        raise ValueError("test failure")

    assert len(holder) == 1
    assert holder[0].is_closed
    with pytest.raises(RuntimeError):
        holder[0].read_doc_text(0)


def test_boolean_merge_matches_set_engine(sample_index_with_positions: Index) -> None:
    for query in [
        "python",
        "python AND loop",
        "async OR botany",
        "loop AND NOT java",
        '"event loop"',
        "NOT python",
    ]:
        res_merge = [
            h.doc.doc_id for h in search_boolean(sample_index_with_positions, query, engine="merge")
        ]
        res_set = [
            h.doc.doc_id for h in search_boolean(sample_index_with_positions, query, engine="set")
        ]
        assert res_merge == res_set


def test_corpus_helpers_and_stats(sample_corpus_dir: Path) -> None:
    assert count_corpus_files(sample_corpus_dir) == 5
    docs = list(iter_documents(sample_corpus_dir))
    assert len(docs) == 5

    lazy = collect_stats_lazy(sample_corpus_dir, top_k=5)
    eager = collect_stats_eager(sample_corpus_dir, top_k=5)
    assert lazy.doc_count == eager.doc_count == 5
    assert lazy.total_tokens == eager.total_tokens
    assert lazy.vocab_size == eager.vocab_size


@pytest.mark.slow
def test_three_storage_variants_benchmark_on_synthetic_corpus(tmp_path: Path) -> None:
    """Повільний тест (>1 с): перевіряє 3 варіанти зберігання постінгів на 250 документах."""
    big_dir = tmp_path / "big_corpus"
    big_dir.mkdir()
    sentence = "python search engine inverted index postings array memory benchmark " * 80
    for i in range(250):
        (big_dir / f"doc_{i:03d}.txt").write_text(
            f"Title {i}\n{sentence} unique_{i}", encoding="utf-8"
        )

    t0 = time.perf_counter()
    rows = benchmark_posting_storage(big_dir)
    _ = time.perf_counter() - t0
    assert len(rows) == 3
    assert rows[2].peak_memory_bytes <= rows[0].peak_memory_bytes
