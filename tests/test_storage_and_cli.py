"""Тести серіалізації save/load та CLI через typer.testing.CliRunner."""

from __future__ import annotations
from typing import Any, cast

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index
from findex.storage import load_index, save_index

runner = CliRunner()


@pytest.mark.parametrize("fmt", ["bin", "pickle"])
def test_save_and_load_roundtrip(
    tmp_path: Path,
    sample_index_with_positions: Index,
    fmt: str,
) -> None:
    out_file = tmp_path / f"idx_{fmt}.bin"
    save_index(sample_index_with_positions, out_file, fmt="bin" if fmt == "bin" else "pickle")
    restored = load_index(out_file)

    assert restored.num_docs == sample_index_with_positions.num_docs
    assert restored.has_positions == sample_index_with_positions.has_positions
    assert restored.docs == sample_index_with_positions.docs
    assert dict(restored.items()) == dict(sample_index_with_positions.items())


def test_cli_index_search_json_and_stats(
    tmp_path: Path,
    sample_corpus_dir: Path,
) -> None:
    idx_file = tmp_path / "cli_index.bin"

    # 1. findex index
    res_index = runner.invoke(
        app,
        ["-v", "index", str(sample_corpus_dir), "--out", str(idx_file), "--positions"],
    )
    assert res_index.exit_code == 0, res_index.output
    assert idx_file.exists()

    # 2. findex search (rich table)
    res_table = runner.invoke(
        app,
        [
            "search",
            str(idx_file),
            'python AND "event loop"',
            "--corpus-dir",
            str(sample_corpus_dir),
            "--top",
            "3",
        ],
    )
    assert res_table.exit_code == 0, res_table.output
    assert "00_python.txt" in res_table.stdout

    # 3. findex search --json (чистий JSON у stdout без діагностичного сміття)
    res_json = runner.invoke(
        app,
        [
            "-vv",
            "search",
            str(idx_file),
            "python",
            "--scorer",
            "tfidf",
            "--corpus-dir",
            str(sample_corpus_dir),
            "--json",
        ],
    )
    assert res_json.exit_code == 0, res_json.output
    parsed: Any = json.loads(res_json.stdout)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert {item["rel_path"] for item in parsed} == {"00_python.txt", "03_java.txt"}

    # 4. findex stats
    res_stats = runner.invoke(app, ["stats", str(sample_corpus_dir), "--top", "5"])
    assert res_stats.exit_code == 0, res_stats.output
    assert "Lazy" in res_stats.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["search", "nonexistent_index.bin", "python"],
        ["index", "nonexistent_corpus_dir"],
    ],
)
def test_cli_expected_errors_return_clean_nonzero_exit(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "Error:" in result.stderr
    assert "Traceback" not in result.output


def test_cli_broken_query_returns_clean_error(saved_index_path: Path) -> None:
    result = runner.invoke(app, ["search", str(saved_index_path), "(python AND"])
    assert result.exit_code != 0
    assert "Error:" in result.stderr
    assert "Traceback" not in result.output
