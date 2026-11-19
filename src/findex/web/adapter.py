from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from findex.index import Index
from findex.search import make_scorer
from findex.search import search as core_search
from findex.storage import load_index

ScorerMode = Literal["bm25", "tfidf"]


@dataclass
class SearchHit:
    doc_id: str
    title: str
    score: float
    snippet: str
    path: str | None = None


class FindexEngineAdapter:
    def __init__(self, index_path: Path, corpus_dir: Path) -> None:
        self.index_path = Path(index_path)
        self.corpus_dir = Path(corpus_dir)
        self._index: Index | None = None
        self._engine_kind: str = "missing_index"
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            self._index = None
            self._engine_kind = "missing_index"
            return
        corpus_root = self.corpus_dir if self.corpus_dir.exists() else None
        self._index = load_index(self.index_path, corpus_root=corpus_root)
        self._engine_kind = "findex.storage.load_index (BM25 / TF-IDF)"

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    def stats(self) -> dict[str, Any]:
        index_size = self.index_path.stat().st_size if self.index_path.exists() else 0
        vocab_size = len(self._index.postings) if self._index is not None else 0
        doc_count = len(self._index.docs) if self._index is not None else 0
        has_positions = bool(self._index.has_positions) if self._index is not None else False
        return {
            "status": "ok" if self.is_ready else "degraded",
            "engine": self._engine_kind,
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
            "index_size_bytes": index_size,
            "corpus_dir": str(self.corpus_dir),
            "corpus_exists": self.corpus_dir.exists(),
            "corpus_files": doc_count,
            "vocab_size": vocab_size,
            "has_positions": has_positions,
        }

    def search(
        self, query: str, top_k: int = 10, scorer_name: str = "bm25"
    ) -> tuple[list[SearchHit], float]:
        t0 = time.perf_counter()
        q = query.strip()
        if not q or self._index is None:
            return [], 0.0
        mode: ScorerMode = "tfidf" if scorer_name.strip().lower() == "tfidf" else "bm25"
        scorer = make_scorer(mode)
        raw_results = core_search(
            self._index, q, scorer=scorer, top_k=top_k, highlight_ansi=False
        )
        hits = [
            SearchHit(
                doc_id=str(res.doc_id),
                title=res.title or Path(res.rel_path).name,
                score=round(float(res.score), 4),
                snippet=res.snippet or "(Фрагмент тексту недоступний)",
                path=res.rel_path,
            )
            for res in raw_results
        ]
        return hits, round((time.perf_counter() - t0) * 1000, 2)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        if self._index is None:
            return None
        meta = None
        if doc_id.isdigit():
            idx = int(doc_id)
            if 0 <= idx < len(self._index.docs) and self._index.docs[idx].doc_id == idx:
                meta = self._index.docs[idx]
            else:
                meta = next((d for d in self._index.docs if d.doc_id == idx), None)
        if meta is None:
            meta = next((d for d in self._index.docs if d.rel_path == doc_id), None)
        if meta is None:
            return None
        text = self._index.read_doc_text(meta.doc_id)
        if text is None and self.corpus_dir.exists():
            candidate = self.corpus_dir / meta.rel_path
            if candidate.exists() and candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace")
        return {
            "doc_id": meta.doc_id,
            "rel_path": meta.rel_path,
            "title": meta.title,
            "length": meta.length,
            "content": text or "",
        }
