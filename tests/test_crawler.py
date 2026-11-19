"""Тести асинхронного краулера findex через httpx.MockTransport (без живої мережі)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import click
import httpx
from typer.main import get_command
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler import (
    CrawlStats,
    RobotsCache,
    canonicalize_url,
    crawl,
    extract_html_content,
    fetch,
)
from findex.index import build_inverted_index
from findex.search import search


def test_canonicalize_url_normalizes_scheme_host_fragment_slash_and_query() -> None:
    u1 = canonicalize_url("HTTP://Example.COM:80/Docs/Page/?b=2&a=1#section-3")
    u2 = canonicalize_url("http://example.com/Docs/Page?a=1&b=2")
    assert u1 == "http://example.com/Docs/Page?a=1&b=2"
    assert u1 == u2
    assert canonicalize_url("https://example.com:443/") == "https://example.com/"
    assert canonicalize_url("mailto:user@example.com") is None


def test_extract_html_content_strips_scripts_and_resolves_links() -> None:
    html = """
    <html>
      <head><title>  Asyncio   Guide </title><script>var secret = 1;</script></head>
      <body>
        <h1>Hello</h1><p>World</p>
        <a href="/docs/a#top">A</a>
        <a href="https://example.com/docs/a?x=1#frag">A2</a>
        <a href="javascript:void(0)">JS</a>
      </body>
    </html>
    """
    title, text, links = extract_html_content(html, base_url="https://example.com/index.html")
    assert title == "Asyncio Guide"
    assert text == "Hello World A A2 JS"
    assert "secret" not in text
    assert links == ["https://example.com/docs/a", "https://example.com/docs/a?x=1"]


def test_fetch_retries_429_and_5xx_with_retry_after_and_skips_404() -> None:
    async def _run() -> None:
        calls: dict[str, int] = {"/rate": 0, "/err503": 0, "/missing": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls[path] = calls.get(path, 0) + 1
            if path == "/rate":
                if calls[path] < 3:
                    return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Too many")
                return httpx.Response(200, text="OK after 429")
            if path == "/err503":
                if calls[path] == 1:
                    return httpx.Response(503, text="Unavailable")
                return httpx.Response(200, text="Recovered")
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            r429 = await fetch(
                client,
                "https://example.com/rate",
                max_retries=3,
                base_backoff_sec=0.005,
                jitter_sec=0.005,
            )
            assert r429.ok
            assert r429.attempts == 3
            assert r429.text == "OK after 429"

            r503 = await fetch(
                client,
                "https://example.com/err503",
                max_retries=3,
                base_backoff_sec=0.005,
                jitter_sec=0.005,
            )
            assert r503.ok
            assert r503.attempts == 2

            r404 = await fetch(
                client,
                "https://example.com/missing",
                max_retries=3,
                base_backoff_sec=0.005,
                jitter_sec=0.005,
            )
            assert not r404.ok
            assert r404.status_code == 404
            assert r404.attempts == 1
            assert calls["/missing"] == 1

    asyncio.run(_run())


def test_fetch_retries_timeout_and_stops_on_redirect_loop() -> None:
    async def _run() -> None:
        timeout_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal timeout_calls
            if request.url.path == "/slow":
                timeout_calls += 1
                if timeout_calls <= 2:
                    raise httpx.ReadTimeout("timed out", request=request)
                return httpx.Response(200, text="Finally loaded")
            if request.url.path == "/loop":
                return httpx.Response(302, headers={"Location": "https://example.com/loop"})
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            res_timeout = await fetch(
                client,
                "https://example.com/slow",
                timeout_sec=0.5,
                max_retries=3,
                base_backoff_sec=0.005,
                jitter_sec=0.005,
            )
            assert res_timeout.ok
            assert res_timeout.attempts == 3

            res_loop = await fetch(
                client,
                "https://example.com/loop",
                timeout_sec=0.5,
                max_retries=3,
            )
            assert not res_loop.ok
            assert res_loop.error == "TooManyRedirects"
            assert res_loop.attempts == 1

    asyncio.run(_run())


def test_robots_cache_fetches_once_per_host_and_enforces_disallow() -> None:
    async def _run() -> None:
        robots_hits = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal robots_hits
            if request.url.path == "/robots.txt":
                robots_hits += 1
                return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nAllow: /\n")
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(handler)
        cache = RobotsCache(user_agent="findex-crawler/0.5")
        async with httpx.AsyncClient(transport=transport) as client:
            a1, a2, a3 = await asyncio.gather(
                cache.is_allowed(client, "https://example.com/public/1"),
                cache.is_allowed(client, "https://example.com/private/secret"),
                cache.is_allowed(client, "https://example.com/public/2"),
            )
        assert a1 is True
        assert a2 is False
        assert a3 is True
        assert robots_hits == 1

    asyncio.run(_run())


def test_crawl_respects_max_pages_deduplicates_and_integrates_with_index(tmp_path: Path) -> None:
    async def _run() -> list[str]:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
            if path == "/":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text=(
                        "<title>Home</title>"
                        "<a href='/blocked'>B</a>"
                        "<a href='/p1'>1</a><a href='/p1#dup'>1dup</a>"
                        "<a href='/p2'>2</a>"
                        "<a href='https://other.org/ext'>Ext</a>"
                    ),
                )
            if path in ("/p1", "/p2", "/p3", "/p4"):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text=(
                        f"<title>Page {path}</title>"
                        f"<p>Asyncio keyword {path}</p>"
                        "<a href='/p3'>3</a><a href='/p4'>4</a>"
                    ),
                )
            return httpx.Response(404, text="missing")

        transport = httpx.MockTransport(handler)
        stats = CrawlStats()
        collected_urls: list[str] = []
        jsonl_file = tmp_path / "crawled.jsonl"

        with jsonl_file.open("w", encoding="utf-8") as fh:
            async for page in crawl(
                "https://example.com/",
                max_pages=3,
                concurrency=3,
                per_host_concurrency=2,
                min_host_delay=0.0,
                transport=transport,
                stats=stats,
            ):
                collected_urls.append(page.final_url)
                fh.write(
                    json.dumps(
                        {"url": page.final_url, "title": page.title, "text": page.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        assert len(collected_urls) == 3
        assert len(set(collected_urls)) == 3
        assert "https://example.com/blocked" not in collected_urls
        assert stats.skipped_robots == 1

        docs = list(iter_documents(tmp_path))
        assert len(docs) == 3

        idx = build_inverted_index(tmp_path, include_positions=True, executor="serial")
        results = search(idx, "asyncio", top_k=5)
        assert len(results) >= 1
        idx.close()
        return collected_urls

    asyncio.run(_run())


def test_cli_crawl_command_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["crawl", "--help"])
    assert res.exit_code == 0

    # Rich formats the help table differently across terminals/OSes. Verify
    # the actual Click command metadata instead of depending on rendered text.
    click_app = cast(click.Group, get_command(app))
    crawl_command = click_app.commands["crawl"]
    option_names = {
        option
        for param in crawl_command.params
        for option in cast(list[str], getattr(param, "opts", []))
    }
    assert {"--max-pages", "--concurrency"} <= option_names
