"""Workload entry point for Scalene's Python/native CPU attribution."""

from pathlib import Path

from findex.index import build_inverted_index
from findex.search import clear_query_cache, search
from findex.storage import load_index


def main() -> None:
    build_inverted_index(Path("data/corpus_bench"), include_positions=True, executor="serial")
    index = load_index(Path("data/index.bin"), corpus_root=Path("data/corpus_bench"))
    for _ in range(250):
        clear_query_cache()
        search(index, "whale OR sea OR captain", top_k=10)


if __name__ == "__main__":
    main()
