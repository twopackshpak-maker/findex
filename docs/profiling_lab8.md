# Профілювання Lab 8

Команда `python scripts/profile_lab8_before.py` профілює індексацію корпусу `data/corpus_bench` і 100 запитів `whale OR sea OR captain`. cProfile інструментує інтерпретатор, тому абсолютний час побудови з цього запуску не порівнюється з часом без профайлера.

## До змін: гіпотези з профілю

| Місце | Спостереження cProfile | Гіпотеза / Python чи native |
|---|---:|---|
| `_tokenize_document_worker` | 6.277 s власного часу, 20.062 s сукупно | Python цикл підрахунку термів і позицій є головним гарячим місцем індексації. |
| `tokenize` | 8.721 s власного часу, 11.995 s сукупно | Python керує генератором; NFKC і регулярні вирази працюють у native C, тому частка їхнього часу окремо перевіряється Scalene. |
| `_save_binary_array` | 1.787 s сукупно | Запис байтів і виклики NumPy переважно обмежені native-кодом та I/O; це не цикл Python, який варто векторизувати. |

Профіль пошуку (100 запусків) зайняв 14 ms сумарно. Найпомітніші витрати: `_lex` — 8 ms сукупно, `tokenize` — 4 ms, `parse_or` — 4 ms. Парсер і обробка короткого запиту обмежені Python; пошук по малому набору постингів занадто короткий, щоб SIMD-виграш переважив виклики функцій.

Scalene snapshot зберігається у [scalene_lab8.json](scalene_lab8.json). `--profile-all` показав змішане навантаження: виклик побудови індексу відніс приблизно 58% до native-коду та 41% до системного часу; власне Python-код найчастіше потрапляв у `pos_map[token].append(pos)`. У токенізаторі `match.group(0)` значну частину часу виконує native regex engine. Це пояснює, чому один лише перенос циклу в NumPy не прибирає витрати читання файлів і regex.

`py-spy` SVG-флеймграфи: [індексація](flamegraph_index.svg) і [пошук](flamegraph_search.svg). Під час повторного запуску на CPython Windows отримано `Failed to find python version from target process`; ці SVG збережені від попереднього успішного запуску `py-spy` у цій робочій копії.

## Репродукція

```powershell
uv run python scripts/profile_lab8_before.py
uv run scalene run --cpu-only --profile-all --outfile docs/scalene_lab8.json scripts/profile_lab8_scalene.py
uv run scalene view --cli docs/scalene_lab8.json
```

Після векторизації порядок результатів перевіряють тестом `test_memmap_round_trip_preserves_postings_and_ranking` проти скалярного scorer-а з Лаби 3. Часи scorer-ів і 10× mmap вимірювання наведено у [lab8_performance.md](lab8_performance.md).
