"""Тести парсера AST-дерев, скорерів TF-IDF / BM25, сніпетів та LRU-кешу."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from findex.index import Index, build_inverted_index
from findex.query import And, Not, Or, Phrase, Term, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, build_snippet
from findex.search import clear_query_cache, get_query_cache_info, search


@pytest.mark.parametrize(
    ("raw_query", "expected_tree"),
    [
        (
            'python AND (async OR await) NOT java "event loop"',
            Term("python")
            & (Term("async") | Term("await"))
            & ~Term("java")
            & Phrase(("event", "loop")),
        ),
        (
            "alpha OR beta AND NOT gamma",
            Or(Term("alpha"), And(Term("beta"), Not(Term("gamma")))),
        ),
        (
            "python index",
            And(Term("python"), Term("index")),
        ),
        (
            "(a OR b) AND (c OR d)",
            And(Or(Term("a"), Term("b")), Or(Term("c"), Term("d"))),
        ),
    ],
)
def test_parse_query_builds_expected_ast_trees(raw_query: str, expected_tree: object) -> None:
    assert parse_query(raw_query) == expected_tree


@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "   ",
        "(python AND async",
        "python AND OR async",
        '""',
        "python )",
    ],
)
def test_parse_query_rejects_invalid_syntax(bad_query: str) -> None:
    with pytest.raises(ValueError):
        parse_query(bad_query)


def test_phrase_evaluation_with_and_without_positions(
    sample_index: Index,
    sample_index_with_positions: Index,
) -> None:
    ast = parse_query('"event loop"')
    with pytest.raises(ValueError, match="позиціями"):
        ast.evaluate(sample_index)

    assert ast.evaluate(sample_index_with_positions) == {0, 3}


def test_scorers_satisfy_protocol_and_three_ranking_sanity_checks(tmp_path: Path) -> None:
    corpus = tmp_path / "rank_corpus"
    corpus.mkdir()
    (corpus / "00.txt").write_text("chrysanthemum common alpha beta", encoding="utf-8")
    (corpus / "01.txt").write_text("common alpha beta gamma", encoding="utf-8")
    (corpus / "02.txt").write_text("common " + " ".join(["darwin"] * 1), encoding="utf-8")
    (corpus / "03.txt").write_text("common " + " ".join(["darwin"] * 19), encoding="utf-8")
    (corpus / "04.txt").write_text(" ".join(["darwin"] * 20), encoding="utf-8")
    (corpus / "05.txt").write_text("whale short", encoding="utf-8")
    (corpus / "06.txt").write_text("whale " + " ".join(["filler"] * 200), encoding="utf-8")

    idx = build_inverted_index(corpus)
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()

    # Структурна відповідність протоколу typing.Protocol без наслідування
    assert isinstance(bm25, Scorer)
    assert isinstance(tfidf, Scorer)

    # 1. Рідкісний термін оцінюється вище за частий
    assert bm25.score("chrysanthemum", idx["chrysanthemum"][0], idx) > bm25.score(
        "common", idx["common"][0], idx
    )
    assert tfidf.score("chrysanthemum", idx["chrysanthemum"][0], idx) > tfidf.score(
        "common", idx["common"][0], idx
    )

    # 2. 20-те повторення у BM25 додає мізерно мало (ефект насичення)
    s_19 = bm25.score("darwin", idx["darwin"][1], idx)
    s_20 = bm25.score("darwin", idx["darwin"][2], idx)
    assert (s_20 - s_19) / s_19 < 0.02

    # 3. Короткий документ з 1 входженням вище за дуже довгий (обидва скорери)
    for scorer in (bm25, tfidf):
        ranked = search(idx, "whale", scorer=scorer, top_k=2)
        assert [r.doc_id for r in ranked] == [5, 6]
        assert ranked[0].score > ranked[1].score


def test_snippet_highlighting_and_lru_cache(
    sample_index_with_positions: Index,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clear_query_cache()
    with caplog.at_level(logging.INFO, logger="findex"):
        r1 = search(sample_index_with_positions, 'python AND "event loop"', top_k=2)
        r2 = search(sample_index_with_positions, 'python AND "event loop"', top_k=2)

    assert len(r1) == 2
    assert isinstance(r1[0], SearchResult)
    assert r1[0].doc_id == 0
    assert "**Python**" in r1[0].snippet
    assert "**event** **loop**" in r1[0].snippet

    hits, misses = get_query_cache_info()
    assert hits == 1
    assert misses == 1
    assert r1 == r2

    snippet_plain = build_snippet("Short text without target words", ["missing"], window=10)
    assert "Short text" in snippet_plain
