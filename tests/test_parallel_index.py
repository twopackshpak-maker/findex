"""Тести паралельної індексації (build_partial, merge, threads, processes та обробка збоїв)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import collect_corpus_paths
from findex.index import (
    ExecutorMode,
    PartialIndex,
    build_inverted_index,
    build_partial,
    merge,
)
from findex.storage import load_index

runner = CliRunner()


def _failing_worker(
    paths: Sequence[Path],
    corpus_root: Path | None = None,
    *,
    start_doc_id: int = 0,
    include_positions: bool = False,
) -> PartialIndex:
    raise RuntimeError(f"Симуляція аварії воркера на діапазоні start_doc_id={start_doc_id}")


@pytest.fixture()
def sample_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = [
        ("01_alpha.txt", "Sherlock Holmes lived at Baker Street in London.\n"),
        ("02_beta.txt", "Doctor Watson assisted Holmes with every mystery.\n"),
        ("03_gamma.txt", "White whale Moby Dick sailed the deep ocean.\n"),
        ("04_delta.txt", "Captain Ahab hunted the white whale tirelessly.\n"),
        ("05_empty.txt", "   \n\t  "),
        ("06_epsilon.txt", "Python concurrency uses threads and processes.\n"),
        ("07_zeta.txt", "Global Interpreter Lock serializes Python bytecode.\n"),
    ]
    for name, content in docs:
        (corpus / name).write_text(content, encoding="utf-8")
    return corpus


@pytest.mark.parametrize("include_positions", [False, True])
def test_merge_single_build_partial_matches_lab4_serial(
    sample_corpus: Path,
    include_positions: bool,
) -> None:
    """Послідовний merge([build_partial(усі шляхи)]) дає ідентичний індекс до Лаб 4."""
    serial_index = build_inverted_index(
        sample_corpus,
        include_positions=include_positions,
        workers=1,
        executor="serial",
    )
    all_paths = collect_corpus_paths(sample_corpus)
    partial = build_partial(
        all_paths,
        sample_corpus,
        start_doc_id=0,
        include_positions=include_positions,
    )
    merged_index = merge([partial], corpus_root=sample_corpus)

    assert merged_index.num_docs == serial_index.num_docs
    assert merged_index.has_positions == serial_index.has_positions
    assert merged_index.docs == serial_index.docs
    assert dict(merged_index.postings) == dict(serial_index.postings)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_match_serial_index(
    sample_corpus: Path,
    executor: ExecutorMode,
    workers: int,
) -> None:
    """Усі виконавці (serial, threads, processes) будують ідентичний індекс."""
    baseline = build_inverted_index(
        sample_corpus,
        include_positions=True,
        workers=1,
        executor="serial",
    )
    parallel_idx = build_inverted_index(
        sample_corpus,
        include_positions=True,
        workers=workers,
        executor=executor,
    )

    assert parallel_idx.docs == baseline.docs
    assert [d.doc_id for d in parallel_idx.docs] == list(range(baseline.num_docs))
    assert dict(parallel_idx.postings) == dict(baseline.postings)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_worker_crash_propagates_without_hanging(
    sample_corpus: Path,
    executor: ExecutorMode,
) -> None:
    """Падіння воркера миттєво перериває збірку з оригінальним винятком, а не зависає."""
    with pytest.raises(RuntimeError, match="Симуляція аварії воркера"):
        build_inverted_index(
            sample_corpus,
            workers=2,
            executor=executor,
            worker_fn=_failing_worker,
        )


def test_cli_index_workers_and_executor(sample_corpus: Path, tmp_path: Path) -> None:
    """CLI підтримує --workers N та --executor {serial,threads,processes}."""
    out_serial = tmp_path / "serial.bin"
    out_proc = tmp_path / "proc.bin"

    res_serial = runner.invoke(
        app,
        ["index", str(sample_corpus), "--out", str(out_serial), "--executor", "serial", "-w", "1"],
    )
    assert res_serial.exit_code == 0, res_serial.output

    res_proc = runner.invoke(
        app,
        ["index", str(sample_corpus), "--out", str(out_proc), "--executor", "processes", "-w", "2"],
    )
    assert res_proc.exit_code == 0, res_proc.output

    idx_s = load_index(out_serial)
    idx_p = load_index(out_proc)
    assert idx_p.docs == idx_s.docs
    assert dict(idx_p.postings) == dict(idx_s.postings)
