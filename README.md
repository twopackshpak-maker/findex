# findex — Пошукова система на Python

![CI](https://github.com/twopackshpak-maker/findex/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Typing](https://img.shields.io/badge/pyright-strict-brightgreen)

Навчальна пошукова система, що реалізує повний цикл обробки тексту: лінивий потоковий парсер корпусу, Unicode-токенізатор, компактний інвертований індекс (`Mapping`), моделі ранжування `BM25` / `TF-IDF`, AST-парсер мови запитів і типізований CLI на базі `typer` та `rich`.

---

## Встановлення та швидкий старт

```bash
# 1. Встановлення залежностей розробки та фіксація uv.lock
uv sync --dev

# 2. Глобальне встановлення утиліти findex у PATH
uv tool install .
findex --help

# 3. Завантаження корпусу (~1 012 документів / ~42.8 МБ) у data/corpus
uv run python scripts/fetch_corpus.py

# 4. Побудова індексу з позиціями токенів (зі смугою прогресу rich)
findex index data/corpus --out index_pos.bin --positions

# 5. Пошук з таблицею rich або виводом JSON у stdout
findex search index_pos.bin 'holmes AND (watson OR moriarty) "baker street"' --corpus-dir data/corpus
findex search index_pos.bin "python AND async" --scorer bm25 --json > results.json

# 6. Статистика корпусу та порівняння пам'яті Lazy vs Eager
findex stats data/corpus
```

---

## Лабораторна робота №4: Типи, тести, пакування і CLI

### 1. Статична типізація (`pyright` у режимі `strict`)

- Усі публічні функції та класи в `src/findex/` повністю типізовані, у `pyproject.toml` увімкнено `typeCheckingMode = "strict"` (**0 помилок `pyright`**).
- `Scorer` оформлено як `typing.Protocol` (`@runtime_checkable`): класи `TfIdf` і `BM25` задовольняють контракт структурно **без наслідування** від `Scorer`.
- Потокові генератори анотовані через `collections.abc.Iterator[...]` та `Iterable[...]`.
- Рядкові режими типізовані через `Literal`: `SearchEngineMode = Literal["merge", "set"]`, `ScorerName = Literal["bm25", "tfidf"]`, `StorageFormat = Literal["bin", "pickle"]`.
- Декоратор `@timed` реалізовано через типізований generic-клас `TimedCallable[P, R]`, завдяки чому у коді **немає жодного `# type: ignore`**.

### 2. Тестування (`pytest` + `Hypothesis`) та покриття коду

Запуск швидких тестів (без `@pytest.mark.slow`) та повного набору з покриттям:

```bash
# Лише швидкі тести (< 1 с)
uv run pytest -v -m "not slow"

# Усі 34 тести (включно з Hypothesis та @pytest.mark.slow) + звіт покриття
uv run pytest -v --cov=findex --cov-report=term-missing
```

#### Звіт покриття (`pytest-cov`)

```text
Name                       Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------
src/findex/__init__.py         9      0      0      0   100%
src/findex/cli.py            156     11     24      4    91%   50-51, 134-136, 151-165, 250-252
src/findex/corpus.py          28      2     12      2    90%   22, 37
src/findex/decorators.py      22      0      0      0   100%
src/findex/index.py          167      9     44      5    94%   111, 128, 132, 136, 169-174
src/findex/query.py          138      7     48      5    94%   37, 63, 146, 200, 211
src/findex/scoring.py         75      4     26      3    94%   41, 56, 97, 131
src/findex/search.py         144      9     48      6    93%   52, 82, 206, 224, 240, 270
src/findex/stats.py           49      0      6      0   100%
src/findex/storage.py         99      8     28      5    90%   51, 61, 86, 91, 113, 136
src/findex/tokenizer.py       14      0      2      0   100%
----------------------------------------------------------------------
TOTAL                        901     50    238     30    93%
```

**Що навмисно залишилось поза покриттям (7%):**
1. `src/findex/__main__.py` (виключено через `omit` у `[tool.coverage.run]`) — тонка точка входу `if __name__ == "__main__": app()`.
2. Гілка `--benchmark-storage` у CLI (`src/findex/cli.py:151-165`), оскільки сам алгоритм `benchmark_posting_storage()` напряму протестовано в `@pytest.mark.slow` тесті `test_three_storage_variants_benchmark_on_synthetic_corpus`.
3. Захисні перевірки на бінарне пошкодження окремих байтів заголовка файлу (`_load_binary_array`) та ANSI-коди підсвітки у терміналі (`highlight_ansi=True`).

### 3. CI/CD (GitHub Actions) та реліз `v0.4.0`

Workflow `.github/workflows/ci.yml` автоматично виконується на кожен `push` і `pull_request`:
1. `uv sync --dev`
2. `uv run ruff check src tests`
3. `uv run pyright` (режим `strict`)
4. `uv run pytest -v --cov=findex --cov-report=term-missing`
5. `uv build` та перевірка встановлення `.whl` у чисте віртуальне середовище (`/tmp/clean-findex-env/bin/findex --help`).

### 4. Reflection (до захисту Лабораторної 4)

1. **Чим `typing.Protocol` кращий за базовий клас `ABC` для `Scorer`?**
   - `Protocol` реалізує **структурну типізацію** (статичний duck typing): `BM25` і `TfIdf` не змушені успадковувати спільний базовий клас, але `pyright` у режимі `strict` перевіряє точну відповідність сигнатури методу `score(self, term: str, posting: Posting, index: Index) -> float`. Будь-який сторонній клас або тестовий заглушковий скорер сумісний із `search()`, якщо має цей метод.
2. **Навіщо розділяти `stdout` і `stderr` у CLI-утиліті?**
   - Коли користувач викликає `findex -vv search index.bin "python" --json > out.json`, діагностичні логи (`logging`) та смуга прогресу (`rich.Progress`) спрямовуються у `stderr`, а у `stdout` потрапляє виключно валідний JSON. Це робить утиліту придатною для Unix-конвеєрів (`jq`, перенаправлення у файл).
3. **Що дають property-based тести `Hypothesis` порівняно зі звичайними прикладами у `@pytest.mark.parametrize`?**
   - Замість перевірки 3–5 ручно підібраних кейсів `Hypothesis` автоматично генерує десятки комбінацій корпусів, граничних списків і запитів, а у разі збою виконує **shrinking** — звужує контрприклад до мінімального випадку, який ламає інваріант (наприклад, рівність `load(save(index)) == index` або еквівалентність `merge` та `set`).
