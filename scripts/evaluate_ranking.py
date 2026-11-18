"""Оцінка якості ранжування (Precision@5) на 10 розмічених запитах: TF-IDF vs BM25."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from findex.index import open_index
from findex.scoring import BM25, TfIdf
from findex.search import search


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    query: str
    expected_book_slug: str
    description: str


BENCHMARK_QUERIES: tuple[RelevanceJudgment, ...] = (
    RelevanceJudgment('"sherlock holmes" AND "baker street"', "sherlock_holmes", "Бейкер-стріт"),
    RelevanceJudgment('"white whale" AND harpoon', "moby_dick", "Мобі Дік"),
    RelevanceJudgment('"natural selection" AND species', "origin_of_species", "Походження видів"),
    RelevanceJudgment('"count dracula" OR transylvania', "dracula", "Дракула"),
    RelevanceJudgment('"victor frankenstein" OR monster', "frankenstein", "Франкенштейн"),
    RelevanceJudgment('"elizabeth bennet" AND darcy', "pride_and_prejudice", "Гордість і упередження"),
    RelevanceJudgment('"edmond dantes" OR "chateau d'if"', "monte_cristo", "Граф Монте-Крісто"),
    RelevanceJudgment("raskolnikov AND petersburg", "crime_and_punishment", "Злочин і кара"),
    RelevanceJudgment('"jean valjean" AND javert', "les_miserables", "Знедолені"),
    RelevanceJudgment('"sancho panza" AND windmill', "don_quixote", "Дон Кіхот"),
)


def precision_at_5(index_path: Path) -> None:
    tfidf = TfIdf()
    bm25 = BM25(k1=1.5, b=0.75)

    with open_index(index_path) as idx:
        sum_tfidf = 0.0
        sum_bm25 = 0.0
        for item in BENCHMARK_QUERIES:
            res_tfidf = search(idx, item.query, scorer=tfidf, top_k=5)
            res_bm25 = search(idx, item.query, scorer=bm25, top_k=5)
            sum_tfidf += sum(1 for r in res_tfidf if item.expected_book_slug in r.rel_path) / 5.0
            sum_bm25 += sum(1 for r in res_bm25 if item.expected_book_slug in r.rel_path) / 5.0


if __name__ == "__main__":
    precision_at_5(Path("index_pos.bin"))
