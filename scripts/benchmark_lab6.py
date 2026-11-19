"""Бенчмарк асинхронного краулера findex (Лаб 6) для concurrency 1, 5 і 20 на 200+ сторінках."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from findex.crawler import CrawlStats, crawl  # noqa: E402
from findex.index import get_peak_rss_bytes  # noqa: E402

TOTAL_MIRROR_PAGES = 260
MAX_PAGES = 200
NETWORK_LATENCY_SEC = 0.03


def _make_mirror_handler() -> httpx.MockTransport:
    retry_once_seen: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(NETWORK_LATENCY_SEC)
        path = request.url.path

        if path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nAllow: /docs/\nDisallow: /docs/private/\n",
            )

        if path == "/docs/page-7" and path not in retry_once_seen:
            retry_once_seen.add(path)
            return httpx.Response(
                429,
                headers={"Retry-After": "0.05"},
                text="Rate limit exceeded",
            )

        if path == "/docs/index.html":
            links = "".join(f"<a href='/docs/page-{i}'>Page {i}</a>" for i in range(1, 30))
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f"<title>Python Asyncio Mirror Index</title><p>Root documentation</p>{links}",
            )

        if path.startswith("/docs/page-"):
            idx = int(path.removeprefix("/docs/page-"))
            next_links = "".join(
                f"<a href='/docs/page-{nxt}'>Next {nxt}</a>"
                for nxt in range(idx + 1, min(idx + 8, TOTAL_MIRROR_PAGES + 1))
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    f"<title>Asyncio Chapter {idx}</title>"
                    f"<p>Event loop coroutines TaskGroup semaphore queue httpx chapter {idx}</p>"
                    f"{next_links}"
                ),
            )

        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


async def _run_case(
    concurrency: int,
    per_host: int,
    out_dir: Path,
) -> tuple[float, float, int, float, list[str]]:
    stats = CrawlStats()
    logs: list[str] = []
    jsonl_path = out_dir / f"crawl_c{concurrency}_h{per_host}.jsonl"

    t0 = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as fh:
        async for page in crawl(
            "https://docs.local.mirror/docs/index.html",
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host_concurrency=per_host,
            min_host_delay=0.0,
            base_backoff_sec=0.02,
            jitter_sec=0.01,
            transport=_make_mirror_handler(),
            stats=stats,
            log_callback=logs.append,
        ):
            fh.write(
                json.dumps(
                    {"url": page.final_url, "title": page.title, "text": page.text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    wall = time.perf_counter() - t0
    pps = stats.collected / max(wall, 1e-6)
    rss_mb = get_peak_rss_bytes() / (1024 * 1024)
    return wall, pps, stats.errors, rss_mb, logs


async def main() -> None:
    out_dir = Path("data/bench_lab6")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Початковий URL: https://docs.local.mirror/docs/index.html | max_pages={MAX_PAGES}\n")
    print("| Concurrency | Per-host | Wall time (с) | Сторінок / сек | Помилки | Пік RSS (МБ) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")

    sample_retry_log: list[str] = []
    for conc in (1, 5, 20):
        wall, pps, errors, rss_mb, logs = await _run_case(conc, conc, out_dir)
        print(f"| {conc} | {conc} | {wall:.3f} | {pps:.2f} | {errors} | {rss_mb:.1f} |")
        if conc == 5:
            sample_retry_log = [line for line in logs if "page-7" in line or "index.html" in line][
                :4
            ]

    # Додатковий замір із жорстким per-host=2 при concurrency=20 для порівняння ціни ліміту на хост
    wall_lim, pps_lim, err_lim, rss_lim, _ = await _run_case(20, 2, out_dir)
    print(f"| 20 (ліміт хоста) | 2 | {wall_lim:.3f} | {pps_lim:.2f} | {err_lim} | {rss_lim:.1f} |")

    print("\nПриклад рядків crawl.log (включно з обробкою 429 Retry-After):")
    for line in sample_retry_log:
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
