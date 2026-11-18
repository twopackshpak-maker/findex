"""Ранжований пошук (BM25 / TF-IDF), @lru_cache на резолвінг запитів та сніпети."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import heapq
import logging
from typing import Literal
import weakref

from findex.decorators import timed
from findex.index import DocMeta, Index
from findex.query import parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf

logger = logging.getLogger("findex.search")

SearchEngineMode = Literal["merge", "set"]
ScorerName = Literal["bm25", "tfidf"]

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
    """Виконує ранжований пошук по `Index` з використанням протоколу `Scorer` та `heapq.nlargest`."""
    from findex.scoring import build_snippet

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

    if not matched_doc_ids:
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
    if engine == "merge":
        matched_ids = _eval_ast_merge(ast, index)
    else:
        matched_ids = sorted(ast.evaluate(index))

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
        return [p.doc_id for p in index.get(node.value, ())]
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
