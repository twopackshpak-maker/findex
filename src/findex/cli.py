"""CLI-інтерфейс пошукової системи findex на базі typer та rich."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.corpus import count_corpus_files
from findex.crawler import CrawlStats, DEFAULT_USER_AGENT, crawl
from findex.index import (
    ExecutorMode,
    benchmark_posting_storage,
    build_index_profiled,
    get_peak_rss_bytes,
    open_index,
)
from findex.search import ScorerName, make_scorer, search
from findex.stats import collect_stats_eager, collect_stats_lazy
from findex.storage import StorageFormat, save_index

app = typer.Typer(
    name="findex",
    help="findex — пошукова система з інвертованим індексом та ранжуванням BM25 / TF-IDF.",
    no_args_is_help=True,
    add_completion=False,
)

stdout_console = Console(stderr=False)
stderr_console = Console(stderr=True)
logger = logging.getLogger("findex.cli")


def _configure_logging(verbosity: int) -> None:
    """Налаштовує логування один раз у точці входу CLI; уся діагностика йде в stderr."""
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    root_logger = logging.getLogger("findex")
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.propagate = False

    handler = RichHandler(
        console=stderr_console,
        show_time=False,
        show_path=verbosity >= 2,
        rich_tracebacks=False,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_logger.addHandler(handler)


def _fail(message: str) -> None:
    """Виводить один зрозумілий рядок помилки у stderr і завершує команду з кодом 1."""
    sys.stderr.write(f"Error: {message}\n")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Рівень деталізації логів у stderr (-v = INFO, -vv = DEBUG).",
        ),
    ] = 0,
) -> None:
    """Глобальні параметри CLI findex."""
    _configure_logging(verbose)


@app.command("crawl")
def cmd_crawl(
    start_url: Annotated[
        str,
        typer.Argument(help="Початковий URL для обходу (напр. https://docs.python.org/3/)."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Шлях до вихідного JSONL-файлу."),
    ] = Path("data/crawled/pages.jsonl"),
    log_path: Annotated[
        Path,
        typer.Option("--log", help="Шлях до файлу журналу статусів URL."),
    ] = Path("crawl.log"),
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", "-n", min=1, help="Максимальна кількість сторінок."),
    ] = 50,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, help="Загальна кількість одночасних запитів."),
    ] = 5,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Ліміт одночасних запитів до одного хоста."),
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Мінімальна пауза (с) між запитами до одного хоста."),
    ] = 0.05,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Таймаут одного HTTP-запиту (с)."),
    ] = 10.0,
    allow_external: Annotated[
        bool,
        typer.Option("--allow-external", help="Дозволити перехід на зовнішні домени."),
    ] = False,
    ignore_robots: Annotated[
        bool,
        typer.Option("--ignore-robots", help="Ігнорувати robots.txt (лише для локальних тестів)."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Увімкнути debug-режим циклу подій asyncio."),
    ] = False,
) -> None:
    """Асинхронно обходить веб-сторінки та потоково зберігає їх у JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = CrawlStats()

    def _render_live_table() -> Table:
        tbl = Table(title=f"Краулер findex (concurrency={concurrency}, max={max_pages})")
        tbl.add_column("Метрика", style="bold cyan")
        tbl.add_column("Значення", justify="right")
        tbl.add_row("Зібрано сторінок", f"{stats.collected} / {max_pages}")
        tbl.add_row("Сторінок / сек", f"{stats.pages_per_sec:.2f}")
        tbl.add_row("Активні запити (in-flight)", str(stats.in_flight))
        tbl.add_row("Помилки", f"[red]{stats.errors}[/red]" if stats.errors else "0")
        tbl.add_row("Глибина черги", str(stats.queue_depth))
        tbl.add_row("Повторів (retries)", str(stats.retries))
        return tbl

    async def _run_crawl() -> float:
        if debug:
            loop = asyncio.get_running_loop()
            loop.slow_callback_duration = 0.1

        t0 = time.perf_counter()
        with (
            out.open("w", encoding="utf-8", buffering=1) as jsonl_file,
            log_path.open("w", encoding="utf-8", buffering=1) as log_file,
            Live(_render_live_table(), console=stderr_console, refresh_per_second=10) as live,
        ):

            def _write_log(entry: str) -> None:
                log_file.write(f"{entry}\n")
                live.update(_render_live_table())

            async for page in crawl(
                start_url,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_concurrency=per_host,
                min_host_delay=delay,
                timeout_sec=timeout,
                same_host_only=not allow_external,
                respect_robots=not ignore_robots,
                user_agent=DEFAULT_USER_AGENT,
                stats=stats,
                log_callback=_write_log,
            ):
                record = {
                    "doc_id": page.final_url,
                    "url": page.final_url,
                    "title": page.title,
                    "text": page.text,
                    "status": page.status_code,
                    "bytes": page.bytes_received,
                    "elapsed_sec": round(page.elapsed_sec, 4),
                    "fetched_at": page.fetched_at,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                live.update(_render_live_table())

        return time.perf_counter() - t0

    try:
        wall_sec = asyncio.run(_run_crawl(), debug=debug)
    except ValueError as exc:
        _fail(str(exc))
        return

    peak_rss_mb = get_peak_rss_bytes() / (1024 * 1024)
    summary = Table(title="Звіт асинхронного краулера findex")
    summary.add_column("Параметр", style="bold cyan")
    summary.add_column("Значення", justify="right")
    summary.add_row("Стартовий URL", start_url)
    summary.add_row("Concurrency / Per-host", f"{concurrency} / {per_host}")
    summary.add_row("Зібрано сторінок", str(stats.collected))
    summary.add_row("Час роботи (wall)", f"{wall_sec:.3f} с")
    summary.add_row("Швидкість", f"{stats.collected / max(wall_sec, 1e-6):.2f} стор./с")
    summary.add_row("Помилки / Повтори", f"{stats.errors} / {stats.retries}")
    summary.add_row("Пік RSS", f"{peak_rss_mb:.2f} МБ")
    summary.add_row("JSONL / Лог", f"{out} | {log_path}")
    stdout_console.print(summary)


@app.command("index")
def cmd_index(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Шлях до директорії з корпусом текстів або JSONL-файлу."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Шлях до вихідного файлу індексу."),
    ] = Path("index.bin"),
    fmt: Annotated[
        StorageFormat,
        typer.Option("--format", help="Формат серіалізації: bin або pickle."),
    ] = "bin",
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Зберігати позиції токенів для фразових запитів."),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Кількість воркерів (типово: os.cpu_count() для parallel або 1 для serial).",
        ),
    ] = None,
    executor: Annotated[
        ExecutorMode,
        typer.Option(
            "--executor",
            help="Виконавець індексації: serial, threads або processes.",
        ),
    ] = "processes",
    benchmark_storage: Annotated[
        bool,
        typer.Option("--benchmark-storage", help="Порівняти 3 варіанти зберігання постінгів."),
    ] = False,
) -> None:
    """Будує інвертований індекс з корпусу зі смугою прогресу rich та зберігає його на диск."""
    try:
        total_files = count_corpus_files(corpus_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        _fail(str(exc))
        return

    effective_workers = (
        (1 if executor == "serial" else (os.cpu_count() or 1)) if workers is None else workers
    )

    gc.collect()
    tracemalloc.start(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=stderr_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"Індексація ({executor}, workers={effective_workers})...",
            total=max(total_files, 1),
        )

        def _on_doc(_doc_id: int) -> None:
            progress.advance(task_id, 1)

        try:
            profile = build_index_profiled(
                corpus_dir,
                include_positions=positions,
                workers=effective_workers,
                executor=executor,
                on_document=_on_doc,
            )
            idx = profile.index
            save_index(idx, out, fmt=fmt)
        except (OSError, ValueError) as exc:
            tracemalloc.stop()
            _fail(str(exc))
            return
        except Exception as exc:
            tracemalloc.stop()
            traceback.print_exc(file=sys.stderr)
            _fail(f"Помилка воркера під час індексації: {exc}")
            return

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    file_mb = out.stat().st_size / (1024 * 1024)

    logger.info(
        "Індекс збережено: docs=%d, vocab=%d, postings=%d, peak=%.2f МБ",
        idx.num_docs,
        len(idx),
        idx.total_postings,
        peak_bytes / (1024 * 1024),
    )

    table = Table(title="Звіт індексації findex")
    table.add_column("Параметр", style="bold cyan")
    table.add_column("Значення", justify="right")
    table.add_row("Корпус", str(corpus_dir))
    table.add_row("Виконавець / Воркери", f"{executor} ({effective_workers})")
    table.add_row("Документів", f"{idx.num_docs:,}")
    table.add_row("Унікальних термів", f"{len(idx):,}")
    table.add_row("Усього постінгів", f"{idx.total_postings:,}")
    table.add_row("Позиції токенів", "так" if idx.has_positions else "ні")
    table.add_row(
        "Wall / Merge час",
        f"{profile.wall_time_sec:.3f} с / {profile.merge_time_sec:.3f} с",
    )
    table.add_row("Піковий RSS процесу", f"{profile.peak_rss_bytes / (1024 * 1024):.2f} МБ")
    table.add_row("Файл індексу", f"{out} ({file_mb:.2f} МБ, {fmt})")
    stdout_console.print(table)

    if benchmark_storage:
        bench_table = Table(title="Порівняння пам'яті зберігання постінгів (tracemalloc)")
        bench_table.add_column("Варіант зберігання", style="cyan")
        bench_table.add_column("Постінгів", justify="right")
        bench_table.add_column("Час (с)", justify="right")
        bench_table.add_column("Пік RAM (МБ)", justify="right", style="bold green")
        for row in benchmark_posting_storage(corpus_dir):
            bench_table.add_row(
                row.variant,
                f"{row.total_postings:,}",
                f"{row.elapsed_sec:.3f}",
                f"{row.peak_memory_mb:.2f}",
            )
        stdout_console.print(bench_table)


@app.command("search")
def cmd_search(
    index_path: Annotated[
        Path,
        typer.Argument(help="Шлях до файлу індексу (напр. index.bin)."),
    ],
    query: Annotated[
        str,
        typer.Argument(help="Пошуковий запит (напр. 'holmes AND (watson OR moriarty)')."),
    ],
    scorer: Annotated[
        ScorerName,
        typer.Option("--scorer", help="Модель ранжування: bm25 або tfidf."),
    ] = "bm25",
    k1: Annotated[float, typer.Option("--k1", help="Параметр k1 для BM25.")] = 1.5,
    b: Annotated[float, typer.Option("--b", help="Параметр b для BM25.")] = 0.75,
    top: Annotated[int, typer.Option("--top", "-k", help="Кількість топ-результатів.")] = 5,
    corpus_dir: Annotated[
        Path | None,
        typer.Option("--corpus-dir", help="Шлях до корпусу для генерації сніпетів."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Вивести результати у форматі JSON у stdout."),
    ] = False,
    repeat: Annotated[
        bool,
        typer.Option("--repeat", help="Повторити запит для демонстрації @lru_cache hit."),
    ] = False,
) -> None:
    """Виконує ранжований пошук по збереженому індексу та виводить таблицю або JSON."""
    active_scorer = make_scorer(scorer, k1=k1, b=b)

    try:
        with open_index(index_path, corpus_root=corpus_dir) as idx:
            results = search(
                idx,
                query,
                scorer=active_scorer,
                top_k=top,
                highlight_ansi=False,
            )
            if repeat:
                results = search(
                    idx,
                    query,
                    scorer=active_scorer,
                    top_k=top,
                    highlight_ansi=False,
                )
    except (FileNotFoundError, IsADirectoryError, ValueError, TypeError, OSError) as exc:
        _fail(str(exc))
        return

    if as_json:
        payload = [asdict(item) for item in results]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    if not results:
        stdout_console.print(f"[yellow]Нічого не знайдено за запитом:[/yellow] {query!r}")
        return

    table = Table(title=f"Результати пошуку ({scorer.upper()}): {query!r}")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Doc ID", justify="right", style="cyan")
    table.add_column("Файл", style="magenta")
    table.add_column("Заголовок / Сніпет")

    for rank, item in enumerate(results, start=1):
        detail = f"[bold]{item.title}[/bold]"
        if item.snippet:
            detail += f"\n[dim]{item.snippet}[/dim]"
        table.add_row(
            str(rank),
            f"{item.score:.4f}",
            str(item.doc_id),
            item.rel_path,
            detail,
        )

    stdout_console.print(table)


@app.command("stats")
def cmd_stats(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Шлях до директорії з корпусом текстів."),
    ] = Path("data/corpus"),
    skip_eager: Annotated[
        bool,
        typer.Option("--skip-eager", help="Пропустити жадібний прогін на списках."),
    ] = False,
    top: Annotated[
        int,
        typer.Option("--top", help="Кількість найчастіших термів у звіті."),
    ] = 15,
) -> None:
    """Збирає статистику корпусу за один прохід та порівнює Lazy vs Eager пам'ять."""
    try:
        lazy = collect_stats_lazy(corpus_dir, top_k=top)
        eager = None if skip_eager else collect_stats_eager(corpus_dir, top_k=top)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        _fail(str(exc))
        return

    summary = Table(title=f"Статистика корпусу: {corpus_dir}")
    summary.add_column("Підхід", style="bold cyan")
    summary.add_column("Документів", justify="right")
    summary.add_column("Токенів", justify="right")
    summary.add_column("Словник", justify="right")
    summary.add_column("Час (с)", justify="right")
    summary.add_column("Пік RAM (МБ)", justify="right", style="green")

    rows = [lazy] if eager is None else [lazy, eager]
    for s in rows:
        summary.add_row(
            s.mode,
            f"{s.doc_count:,}",
            f"{s.total_tokens:,}",
            f"{s.vocab_size:,}",
            f"{s.elapsed_sec:.2f}",
            f"{s.peak_memory_mb:.2f}",
        )
    stdout_console.print(summary)

    top_table = Table(title=f"Топ-{top} найчастіших термів")
    top_table.add_column("#", justify="right")
    top_table.add_column("Термін", style="bold")
    top_table.add_column("Частота", justify="right", style="cyan")
    for rank, (term, freq) in enumerate(lazy.top_terms, start=1):
        top_table.add_row(str(rank), term, f"{freq:,}")
    stdout_console.print(top_table)
