"""AST-вузли мови запитів (Term, Phrase, And, Or, Not) та парсер рекурсивного спуску."""

from __future__ import annotations

from dataclasses import dataclass
import re

from findex.index import Index
from findex.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class QueryNode:
    """Базовий вузол AST пошукового запиту з підтримкою операторів &, |, ~."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    """Одиночний нормалізований термін."""

    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}

    def positive_terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    """Точна послідовність слів у подвійних лапках, напр. `"event loop"`."""

    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)
        if not index.has_positions:
            raise ValueError(
                f"Фразовий запит {self.terms!r} потребує індексу з позиціями (--positions)."
            )

        postings_per_term: list[dict[int, tuple[int, ...]]] = []
        for term in self.terms:
            if term not in index:
                return set()
            postings_per_term.append({p.doc_id: p.positions for p in index[term]})

        common_docs = set(postings_per_term[0].keys())
        for pmap in postings_per_term[1:]:
            common_docs &= pmap.keys()
            if not common_docs:
                return set()

        matched_docs: set[int] = set()
        for doc_id in common_docs:
            base_positions = postings_per_term[0][doc_id]
            shifted_sets = [
                {pos - offset for pos in postings_per_term[offset][doc_id]}
                for offset in range(1, len(self.terms))
            ]
            if any(all(p in s for s in shifted_sets) for p in base_positions):
                matched_docs.add(doc_id)

        return matched_docs

    def positive_terms(self) -> tuple[str, ...]:
        return self.terms


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    """Кон'юнкція двох підзапитів (`left & right`)."""

    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        left_docs = self.left.evaluate(index)
        if not left_docs:
            return set()
        return left_docs & self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    """Диз'юнкція двох підзапитів (`left | right`)."""

    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    """Заперечення підзапиту (`~operand`)."""

    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        excluded = self.operand.evaluate(index)
        return set(range(index.num_docs)) - excluded

    def positive_terms(self) -> tuple[str, ...]:
        return ()


_LEX_RE = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|\b(AND|OR|NOT)\b|([^\s()"]+))')


def _lex(raw_query: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(raw_query):
        match = _LEX_RE.match(raw_query, pos)
        if not match:
            raise ValueError(f"Синтаксична помилка в запиті біля: {raw_query[pos:]!r}")
        pos = match.end()
        lparen, rparen, phrase, op, word = match.groups()
        if lparen:
            tokens.append(("LPAREN", "("))
        elif rparen:
            tokens.append(("RPAREN", ")"))
        elif phrase is not None:
            norm_words = tuple(tokenize(phrase))
            if not norm_words:
                raise ValueError("Порожня фраза у подвійних лапках")
            tokens.append(("PHRASE", " ".join(norm_words)))
        elif op:
            tokens.append((op, op))
        elif word:
            norm_words = list(tokenize(word))
            for w in norm_words:
                tokens.append(("TERM", w))

    if not tokens:
        raise ValueError("Порожній пошуковий запит")
    return tokens


def parse_query(raw_query: str) -> QueryNode:
    """Парсить рядок пошукового запиту в AST методом рекурсивного спуску."""
    tokens = _lex(raw_query)
    cursor = 0

    def peek() -> tuple[str, str] | None:
        return tokens[cursor] if cursor < len(tokens) else None

    def consume() -> tuple[str, str]:
        nonlocal cursor
        tok = tokens[cursor]
        cursor += 1
        return tok

    def parse_or() -> QueryNode:
        node = parse_and()
        while (tok := peek()) is not None and tok[0] == "OR":
            consume()
            node = Or(node, parse_and())
        return node

    def parse_and() -> QueryNode:
        node = parse_not()
        while (tok := peek()) is not None:
            if tok[0] == "AND":
                consume()
                node = And(node, parse_not())
            elif tok[0] in {"NOT", "TERM", "PHRASE", "LPAREN"}:
                node = And(node, parse_not())
            else:
                break
        return node

    def parse_not() -> QueryNode:
        if (tok := peek()) is not None and tok[0] == "NOT":
            consume()
            return Not(parse_not())
        return parse_primary()

    def parse_primary() -> QueryNode:
        tok = peek()
        if tok is None:
            raise ValueError("Неочікуваний кінець пошукового запиту")
        if tok[0] == "LPAREN":
            consume()
            inner = parse_or()
            closing = peek()
            if closing is None or closing[0] != "RPAREN":
                raise ValueError("Незакрита дужка '(' у запиті")
            consume()
            return inner
        if tok[0] == "TERM":
            consume()
            return Term(tok[1])
        if tok[0] == "PHRASE":
            consume()
            return Phrase(tuple(tok[1].split()))
        raise ValueError(f"Неочікуваний токен у запиті: {tok[1]!r}")

    ast = parse_or()
    if cursor != len(tokens):
        raise ValueError(f"Неочікуваний залишок запиту: {tokens[cursor][1]!r}")
    return ast
