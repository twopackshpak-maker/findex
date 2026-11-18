"""Завантаження публічних книг із Project Gutenberg та розбиття на ~1000 документів."""

from __future__ import annotations

from pathlib import Path
import re
import urllib.request

GUTENBERG_BOOKS: dict[str, str] = {
    "war_and_peace": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "moby_dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock_holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "frankenstein": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "dracula": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "monte_cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    "crime_and_punishment": "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "brothers_karamazov": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "les_miserables": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "great_expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "tale_of_two_cities": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "don_quixote": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    "ulysses": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "origin_of_species": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
}

START_MARKER = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*")
END_MARKER = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*")


def strip_gutenberg_boilerplate(raw: str) -> str:
    start = START_MARKER.search(raw)
    if start:
        raw = raw[start.end() :]
    end = END_MARKER.search(raw)
    if end:
        raw = raw[: end.start()]
    return raw.strip()


def build_corpus(output_dir: Path, target_docs: int = 1000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts: list[tuple[str, str]] = []

    for slug, url in GUTENBERG_BOOKS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "findex-lab/0.4"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        clean = strip_gutenberg_boilerplate(raw)
        texts.append((slug, clean))

    total_chars = sum(len(t) for _, t in texts)
    chunk_size = max(10_000, total_chars // target_docs)

    doc_idx = 0
    for slug, clean in texts:
        for offset in range(0, len(clean), chunk_size):
            fragment = clean[offset : offset + chunk_size].strip()
            if len(fragment) < 1_000:
                continue
            doc_path = output_dir / f"{doc_idx:04d}_{slug}.txt"
            doc_path.write_text(fragment, encoding="utf-8")
            doc_idx += 1


if __name__ == "__main__":
    build_corpus(Path("data/corpus"), target_docs=1000)
