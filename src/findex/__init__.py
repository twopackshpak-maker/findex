"""Пакет findex — навчальна пошукова система на Python."""

from findex.corpus import Document, iter_documents
from findex.decorators import TimedCallable, timed
from findex.index import DocMeta, Index, InvertedIndex, Posting, build_inverted_index, open_index
from findex.query import And, Not, Or, Phrase, QueryNode, Term, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, build_snippet
from findex.search import SearchEngineMode, SearchHit, search, search_boolean
from findex.stats import CorpusStats, collect_stats_eager, collect_stats_lazy
from findex.storage import StorageFormat, load_index, save_index
from findex.tokenizer import normalize_text, tokenize

__all__ = [
    "And",
    "BM25",
    "CorpusStats",
    "DocMeta",
    "Document",
    "Index",
    "InvertedIndex",
    "Not",
    "Or",
    "Phrase",
    "Posting",
    "QueryNode",
    "Scorer",
    "SearchEngineMode",
    "SearchHit",
    "SearchResult",
    "StorageFormat",
    "Term",
    "TfIdf",
    "TimedCallable",
    "build_inverted_index",
    "build_snippet",
    "collect_stats_eager",
    "collect_stats_lazy",
    "iter_documents",
    "load_index",
    "normalize_text",
    "open_index",
    "parse_query",
    "save_index",
    "search",
    "search_boolean",
    "timed",
    "tokenize",
]
