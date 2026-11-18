"""Протокол ранжування Scorer, моделі TfIdf та BM25, SearchResult і побудова сніпетів."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import re
from typing import Protocol, runtime_checkable

from findex.index import Index, Posting
from findex.tokenizer import APOSTROPHE_RE


@runtime_checkable
class Scorer(Protocol):
    """Протокол функції ранжування терміна в документі (duck typing без наслідування)."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Класичний TF-IDF зі згладженим IDF та довжинною нормалізацією TF."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0:
            return 0.0
        doc_len = max(index.docs[posting.doc_id].length, 1)
        tf = posting.tf / doc_len
        idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
        return tf * idf


@dataclass(frozen=True, slots=True)
class BM25:
    """Окапі BM25 з насиченням частоти терміна (k1) та нормалізацією довжини (b)."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0:
            return 0.0
        n = index.num_docs
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        doc_len = index.docs[posting.doc_id].length
        avgdl = max(index.avg_doc_length, 1.0)
        denom = posting.tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
        return idf * ((posting.tf * (self.k1 + 1.0)) / denom)


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Ранжований результат пошуку (`score` іде першим для `order=True` та `heapq.nlargest`)."""

    score: float
    doc_id: int
    rel_path: str
    title: str
    snippet: str = ""


def build_snippet(
    text: str,
    terms: Sequence[str],
    *,
    window: int = 80,
    highlight_ansi: bool = False,
) -> str:
    """Формує сніпет з вікном ±`window` символів навколо найкращого входження та підсвіткою."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return ""

    unique_terms = [t for t in dict.fromkeys(terms) if t]
    if not unique_terms:
        return clean_text[: window * 2] + ("..." if len(clean_text) > window * 2 else "")

    escaped = sorted((re.escape(t) for t in unique_terms), key=len, reverse=True)
    pattern_parts = [p.replace("'", "['’ʼ`]") for p in escaped]
    term_re = re.compile(
        rf"(?<!\w)(?:{'|'.join(pattern_parts)})(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    matches = list(term_re.finditer(clean_text))
    if not matches:
        return clean_text[: window * 2] + ("..." if len(clean_text) > window * 2 else "")

    best_match = matches[0]
    best_distinct = -1
    for m in matches:
        w_start = max(0, m.start() - window)
        w_end = min(len(clean_text), m.end() + window)
        distinct_in_win = {
            APOSTROPHE_RE.sub("'", hit.group(0)).casefold()
            for hit in matches
            if w_start <= hit.start() and hit.end() <= w_end
        }
        if len(distinct_in_win) > best_distinct:
            best_distinct = len(distinct_in_win)
            best_match = m
            if best_distinct == len(unique_terms):
                break

    start = max(0, best_match.start() - window)
    end = min(len(clean_text), best_match.end() + window)
    fragment = clean_text[start:end]

    def _highlight(match: re.Match[str]) -> str:
        word = match.group(0)
        if highlight_ansi:
            return f"\033[1;33m{word}\033[0m"
        return f"**{word}**"

    highlighted = term_re.sub(_highlight, fragment)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return f"{prefix}{highlighted}{suffix}"
