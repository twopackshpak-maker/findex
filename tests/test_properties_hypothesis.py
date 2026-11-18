"""Property-based тести (Hypothesis) для інвертованого індексу, серіалізації та merge."""

from __future__ import annotations

from pathlib import Path
import tempfile

from hypothesis import given, settings, strategies as st

from findex.index import build_inverted_index
from findex.search import difference_sorted, intersect_sorted, search_boolean, union_sorted
from findex.storage import load_index, save_index

VOCAB_STRATEGY = st.sampled_from(
    ["python", "search", "index", "async", "await", "memory", "array", "bm25", "tfidf", "token"]
)

DOC_TEXT_STRATEGY = st.lists(VOCAB_STRATEGY, min_size=1, max_size=25).map(" ".join)
CORPUS_STRATEGY = st.lists(DOC_TEXT_STRATEGY, min_size=1, max_size=12)
SORTED_INT_LIST = st.lists(st.integers(min_value=0, max_value=200), max_size=40).map(
    lambda xs: sorted(set(xs))
)


@given(docs=CORPUS_STRATEGY, include_positions=st.booleans())
@settings(max_examples=25, deadline=None)
def test_property_save_load_preserves_index(
    docs: list[str],
    include_positions: bool,
) -> None:
    """Властивість 1: load(save(index)) завжди повертає ідентичний індекс."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        for i, content in enumerate(docs):
            (corpus_dir / f"doc_{i:02d}.txt").write_text(f"Title {i}\n{content}", encoding="utf-8")

        original = build_inverted_index(corpus_dir, include_positions=include_positions)
        bin_path = tmp_path / "prop.bin"
        save_index(original, bin_path, fmt="bin")
        restored = load_index(bin_path)

        assert restored.num_docs == original.num_docs
        assert restored.has_positions == original.has_positions
        assert restored.docs == original.docs
        assert dict(restored.items()) == dict(original.items())


@given(docs=CORPUS_STRATEGY)
@settings(max_examples=25, deadline=None)
def test_property_postings_always_strictly_sorted(docs: list[str]) -> None:
    """Властивість 2: списки постінгів для кожного терміна завжди строго відсортовані за doc_id."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp)
        for i, content in enumerate(docs):
            (corpus_dir / f"doc_{i:02d}.txt").write_text(content, encoding="utf-8")

        index = build_inverted_index(corpus_dir, include_positions=True)
        for term, postings in index.items():
            doc_ids = [p.doc_id for p in postings]
            assert doc_ids == sorted(doc_ids), f"Несортовані doc_id для {term!r}: {doc_ids}"
            assert len(doc_ids) == len(set(doc_ids)), f"Дублікати doc_id для {term!r}: {doc_ids}"


@given(a=SORTED_INT_LIST, b=SORTED_INT_LIST)
@settings(max_examples=50, deadline=None)
def test_property_merge_matches_set_operations(a: list[int], b: list[int]) -> None:
    """Властивість 3: двовказівниковий merge завжди збігається з операціями над set."""
    assert intersect_sorted(a, b) == sorted(set(a) & set(b))
    assert union_sorted(a, b) == sorted(set(a) | set(b))
    assert difference_sorted(a, b) == sorted(set(a) - set(b))


@given(docs=CORPUS_STRATEGY, t1=VOCAB_STRATEGY, t2=VOCAB_STRATEGY)
@settings(max_examples=20, deadline=None)
def test_property_boolean_search_merge_equals_set(
    docs: list[str],
    t1: str,
    t2: str,
) -> None:
    """Додаткова властивість: Boolean-пошук через merge збігається з set на довільних корпусах."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp)
        for i, content in enumerate(docs):
            (corpus_dir / f"doc_{i:02d}.txt").write_text(content, encoding="utf-8")

        index = build_inverted_index(corpus_dir)
        query = f"({t1} OR {t2}) AND NOT {t2}" if t1 != t2 else t1
        merge_ids = [h.doc.doc_id for h in search_boolean(index, query, engine="merge")]
        set_ids = [h.doc.doc_id for h in search_boolean(index, query, engine="set")]
        assert merge_ids == set_ids
