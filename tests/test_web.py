# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from findex.index import build_inverted_index
from findex.storage import save_index
from findex.web.app import create_app


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "a.txt").write_text(
        "Python Search Engine\nInverted index with BM25 ranking.", encoding="utf-8"
    )
    (c / "b.txt").write_text(
        "FastAPI Service\nWeb API running on Uvicorn.", encoding="utf-8"
    )
    idx_path = tmp_path / "index.bin"
    idx = build_inverted_index(c, include_positions=True)
    save_index(idx, idx_path, fmt="bin")
    idx.close()
    monkeypatch.setenv("INDEX_PATH", str(idx_path))
    monkeypatch.setenv("CORPUS_DIR", str(c))


def test_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "none.bin"))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "none"))
    client: Any = TestClient(create_app())
    with client:
        data: dict[str, Any] = client.get("/health").json()
        assert data["ready"] is False


def test_search_and_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(tmp_path, monkeypatch)
    client: Any = TestClient(create_app())
    with client:
        health: dict[str, Any] = client.get("/health").json()
        assert health["ready"] is True
        stats: dict[str, Any] = client.get("/stats").json()
        assert stats["corpus_files"] == 2
        r: dict[str, Any] = client.get(
            "/api/search", params={"q": "BM25", "scorer": "bm25"}
        ).json()
        assert r["total"] == 1
        r2: dict[str, Any] = client.post(
            "/api/search", json={"query": "FastAPI", "scorer": "tfidf"}
        ).json()
        assert r2["total"] == 1
        doc_id = str(r["results"][0]["doc_id"])
        doc: dict[str, Any] = client.get(f"/api/doc/{doc_id}").json()
        assert "BM25" in str(doc["content"])
        html: str = client.get("/", params={"q": "Python"}).text
        assert "Python Search Engine" in html
