"""Ранжований пошук на NumPy (BM25 / TF-IDF + np.argpartition), референс Лаби 3 та RRF."""

from __future__ import annotations

import heapq
import logging
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.decorators import timed
from findex.index import DocMeta, Index, Int32Array
from findex.query import parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, VectorizedScorer, build_snippet

logger = logging.getLogger("findex.search")

SearchEngineMode = Literal["merge", "set"]
ScorerName = Literal["bm25", "tfidf"]
SearchMode = Literal["lexical", "semantic", "hybrid"]
Float64Array = NDArray[np.float64]

_INDEX_REGISTRY: dict[int, weakref.ReferenceType[Index]] = {}


def make_scorer(name: ScorerName, *, k1: float = 1.5, b: float = 0.75) -> Scorer:
    """Фабрика скорерів за Literal-іменем (`bm25` або `tfidf`)."""
    if name == "bm25":
        return BM25(k1=k1, b=b)
    return TfIdf()


def _register_index(index: Index) -> int:
    token = id(index)
    existing = _INDEX_REGISTRY.get(token)
    if existing is None or existing() is not index:
        _INDEX_REGISTRY[token] = weakref.ref(
            index,
            lambda _ref, t=token: _INDEX_REGISTRY.pop(t, None),
        )
    return token


@lru_cache(maxsize=256)
def _resolve_query_doc_ids(index_token: int, raw_query: str) -> tuple[int, ...]:
    """Кешований резолвінг `запит -> відсортований кортеж doc_id` через AST."""
    ref = _INDEX_REGISTRY.get(index_token)
    index = ref() if ref is not None else None
    if index is None:
        raise RuntimeError("Індекс більше не існує в пам'яті")
    ast = parse_query(raw_query)
    matched_ids = ast.evaluate(index)
    return tuple(sorted(matched_ids))


def clear_query_cache() -> None:
    """Очищає LRU-кеш результатів запитів."""
    _resolve_query_doc_ids.cache_clear()


def get_query_cache_info() -> tuple[int, int]:
    """Повертає кортеж `(hits, misses)` для декоратора `@lru_cache`."""
    info = _resolve_query_doc_ids.cache_info()
    return (info.hits, info.misses)


def _select_topk_indices(
    doc_ids: Int32Array,
    scores: Float64Array,
    top_k: int,
) -> NDArray[np.intp]:
    """Відбирає топ-k індексів через `np.argpartition` із детермінованим tie-break як у Лабі 3."""
    n = int(doc_ids.shape[0])
    if n == 0 or top_k <= 0:
        return np.empty(0, dtype=np.intp)
    if top_k < n:
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        min_top_score = float(scores[part].min())
        cand_mask = scores >= min_top_score
        cand_idx = part if int(cand_mask.sum()) == top_k else np.flatnonzero(cand_mask)
    else:
        cand_idx = np.arange(n, dtype=np.intp)

    order = np.lexsort((-doc_ids[cand_idx], -scores[cand_idx]))[:top_k]
    return cand_idx[order]


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    top_k: int = 10,
    doc_texts: Mapping[int, str] | None = None,
    highlight_ansi: bool = False,
) -> list[SearchResult]:
    """Векторизований ранжований пошук на масивах NumPy int32 та `np.argpartition`."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    ast = parse_query(query)
    positive_terms = ast.positive_terms()

    index_token = _register_index(index)
    before = _resolve_query_doc_ids.cache_info()
    matched_doc_ids = _resolve_query_doc_ids(index_token, query)
    after = _resolve_query_doc_ids.cache_info()

    if after.hits > before.hits:
        logger.info("[cache hit] query=%r (hits=%d, misses=%d)", query, after.hits, after.misses)
    else:
        logger.info("[cache miss] query=%r (hits=%d, misses=%d)", query, after.hits, after.misses)

    if not matched_doc_ids or top_k <= 0:
        return []

    matched_arr = np.asarray(matched_doc_ids, dtype=np.int32)
    total_scores = np.zeros(matched_arr.shape[0], dtype=np.float64)
    num_docs = index.num_docs
    avgdl = index.avg_doc_length
    doc_lengths = index.doc_lengths

    if isinstance(active_scorer, VectorizedScorer):
        for term in dict.fromkeys(positive_terms):
            arrays = index.get_posting_arrays(term)
            if arrays is None:
                continue
            p_doc_ids, p_tfs = arrays
            term_scores = active_scorer.score_array(
                p_doc_ids, p_tfs, doc_lengths, num_docs, avgdl
            )
            if p_doc_ids.shape[0] == matched_arr.shape[0] and np.array_equal(
                p_doc_ids, matched_arr
            ):
                total_scores += term_scores
            else:
                pos = np.searchsorted(matched_arr, p_doc_ids)
                valid = pos < matched_arr.shape[0]
                v_pos = pos[valid]
                hit = matched_arr[v_pos] == p_doc_ids[valid]
                total_scores[v_pos[hit]] += term_scores[valid][hit]
    else:
        matched_set = set(matched_doc_ids)
        score_map: dict[int, float] = {doc_id: 0.0 for doc_id in matched_doc_ids}
        for term in dict.fromkeys(positive_terms):
            if term not in index:
                continue
            for posting in index[term]:
                if posting.doc_id in matched_set:
                    score_map[posting.doc_id] += active_scorer.score(term, posting, index)
        total_scores = np.array([score_map[d] for d in matched_doc_ids], dtype=np.float64)

    chosen = _select_topk_indices(matched_arr, total_scores, top_k)

    final_results: list[SearchResult] = []
    for idx_pos in chosen.tolist():
        doc_id = int(matched_arr[idx_pos])
        score_val = float(total_scores[idx_pos])
        meta = index.docs[doc_id]
        raw_text = (
            doc_texts[doc_id]
            if doc_texts is not None and doc_id in doc_texts
            else index.read_doc_text(doc_id)
        )
        snippet = (
            build_snippet(raw_text, positive_terms, window=80, highlight_ansi=highlight_ansi)
            if raw_text
            else ""
        )
        final_results.append(
            SearchResult(
                score=score_val,
                doc_id=doc_id,
                rel_path=meta.rel_path,
                title=meta.title,
                snippet=snippet,
            )
        )

    return final_results


def search_scalar(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    top_k: int = 10,
    doc_texts: Mapping[int, str] | None = None,
    highlight_ansi: bool = False,
) -> list[SearchResult]:
    """Еталонний об'єктний пошук із Лаби 3 (цикл по `Posting` + `heapq.nlargest`)."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    ast = parse_query(query)
    positive_terms = ast.positive_terms()
    matched_doc_ids = tuple(sorted(ast.evaluate(index)))

    if not matched_doc_ids or top_k <= 0:
        return []

    matched_set = set(matched_doc_ids)
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched_doc_ids}

    for term in dict.fromkeys(positive_terms):
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id in matched_set:
                scores[posting.doc_id] += active_scorer.score(term, posting, index)

    candidates: list[SearchResult] = []
    for doc_id, total_score in scores.items():
        meta = index.docs[doc_id]
        candidates.append(
            SearchResult(
                score=total_score,
                doc_id=doc_id,
                rel_path=meta.rel_path,
                title=meta.title,
                snippet="",
            )
        )

    top_raw = heapq.nlargest(top_k, candidates)

    final_results: list[SearchResult] = []
    for item in top_raw:
        raw_text = (
            doc_texts[item.doc_id]
            if doc_texts is not None and item.doc_id in doc_texts
            else index.read_doc_text(item.doc_id)
        )
        snippet = (
            build_snippet(raw_text, positive_terms, window=80, highlight_ansi=highlight_ansi)
            if raw_text
            else ""
        )
        final_results.append(
            SearchResult(
                score=item.score,
                doc_id=item.doc_id,
                rel_path=item.rel_path,
                title=item.title,
                snippet=snippet,
            )
        )

    return final_results


def reciprocal_rank_fusion(
    bm25_results: Sequence[SearchResult],
    semantic_results: Sequence[SearchResult],
    *,
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Зливає лексичний (BM25) та семантичний рейтинги через Reciprocal Rank Fusion (RRF)."""
    if top_k <= 0:
        return []

    rrf_scores: dict[int, float] = {}
    best_meta: dict[int, SearchResult] = {}

    for rank, item in enumerate(bm25_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (k_rrf + rank)
        best_meta[item.doc_id] = item

    for rank, item in enumerate(semantic_results, start=1):
        rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (k_rrf + rank)
        existing = best_meta.get(item.doc_id)
        if existing is None or (not existing.snippet and item.snippet):
            best_meta[item.doc_id] = item

    if not rrf_scores:
        return []

    doc_ids = np.array(list(rrf_scores.keys()), dtype=np.int32)
    scores = np.array([rrf_scores[int(d)] for d in doc_ids], dtype=np.float64)
    chosen = _select_topk_indices(doc_ids, scores, top_k)

    fused: list[SearchResult] = []
    for idx_pos in chosen.tolist():
        doc_id = int(doc_ids[idx_pos])
        ref = best_meta[doc_id]
        fused.append(
            SearchResult(
                score=float(scores[idx_pos]),
                doc_id=doc_id,
                rel_path=ref.rel_path,
                title=ref.title,
                snippet=ref.snippet,
            )
        )
    return fused


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Результат Boolean-пошуку (Лаб 2)."""

    doc: DocMeta
    matched_tf: int


def intersect_sorted(a: list[int], b: list[int]) -> list[int]:
    """Двовказівниковий перетин двох відсортованих списків doc_id."""
    result: list[int] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union_sorted(a: list[int], b: list[int]) -> list[int]:
    """Двовказівникове об'єднання двох відсортованих списків doc_id."""
    result: list[int] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def difference_sorted(left: list[int], excluded: list[int]) -> list[int]:
    """Двовказівникова різниця двох відсортованих списків doc_id."""
    result: list[int] = []
    i = 0
    j = 0
    while i < len(left) and j < len(excluded):
        if left[i] == excluded[j]:
            i += 1
            j += 1
        elif left[i] < excluded[j]:
            result.append(left[i])
            i += 1
        else:
            j += 1
    result.extend(left[i:])
    return result


def search_boolean(
    index: Index,
    query: str,
    *,
    engine: SearchEngineMode = "merge",
) -> list[SearchHit]:
    """Boolean-пошук по індексу за допомогою рушія `merge` або `set`."""
    ast = parse_query(query)
    matched_ids = _eval_ast_merge(ast, index) if engine == "merge" else sorted(ast.evaluate(index))

    if not matched_ids:
        return []

    matched_set = set(matched_ids)
    tf_by_doc: dict[int, int] = {d_id: 0 for d_id in matched_ids}
    for term in set(ast.positive_terms()):
        for p in index.get(term, ()):
            if p.doc_id in matched_set:
                tf_by_doc[p.doc_id] += p.tf

    hits = [SearchHit(doc=index.docs[d_id], matched_tf=tf_by_doc[d_id]) for d_id in matched_ids]
    hits.sort(key=lambda h: (-h.matched_tf, h.doc.doc_id))
    return hits


def _eval_ast_merge(node: object, index: Index) -> list[int]:
    from findex.query import And, Not, Or, Phrase, Term

    if isinstance(node, Term):
        arrays = index.get_posting_arrays(node.value)
        return arrays[0].tolist() if arrays is not None else []
    if isinstance(node, Phrase):
        return sorted(node.evaluate(index))
    if isinstance(node, And):
        left = _eval_ast_merge(node.left, index)
        if not left:
            return []
        right = _eval_ast_merge(node.right, index)
        return intersect_sorted(left, right)
    if isinstance(node, Or):
        left = _eval_ast_merge(node.left, index)
        right = _eval_ast_merge(node.right, index)
        return union_sorted(left, right)
    if isinstance(node, Not):
        universe = list(range(index.num_docs))
        excluded = _eval_ast_merge(node.operand, index)
        return difference_sorted(universe, excluded)
    raise TypeError(f"Непідтримуваний вузол AST: {type(node)!r}")