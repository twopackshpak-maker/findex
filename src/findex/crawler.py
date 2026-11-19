"""Асинхронний веб-краулер для findex на базі asyncio, httpx.AsyncClient та TaskGroup."""

from __future__ import annotations

import asyncio
import email.utils
import random
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

DEFAULT_USER_AGENT = (
    "findex-crawler/0.5 (+https://github.com/twopackshpak-maker/findex; "
    "contact: twopackshpak-maker@users.noreply.github.com)"
)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "td",
        "th",
        "section",
        "article",
        "header",
        "footer",
        "pre",
        "blockquote",
    }
)
_IGNORED_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "template", "svg"})


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Результат виконання одного HTTP-запиту (із урахуванням повторів)."""

    url: str
    final_url: str
    status_code: int | None
    content: bytes
    text: str
    content_type: str
    elapsed_sec: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300


@dataclass(frozen=True, slots=True)
class CrawledPage:
    """Успішно завантажена та розпарсена веб-сторінка."""

    url: str
    final_url: str
    status_code: int
    title: str
    text: str
    links: tuple[str, ...]
    bytes_received: int
    elapsed_sec: float
    attempts: int
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    """Живі лічильники стану краулера для індикації в CLI та профілювання."""

    collected: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    bytes_downloaded: int = 0
    retries: int = 0
    skipped_robots: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_sec(self) -> float:
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return self.collected / elapsed


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Канонізує URL: нижній регістр схеми/хоста, без #fragment, єдиний слеш і відсортований query."""
    cleaned = raw_url.strip()
    if not cleaned:
        return None

    resolved = urljoin(base_url, cleaned) if base_url is not None else cleaned
    parts = urlsplit(resolved)

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None

    hostname = (parts.hostname or "").lower()
    if not hostname:
        return None

    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = f"{hostname}:{port}" if port is not None else hostname

    path = re.sub(r"/+", "/", parts.path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    sorted_query = urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit((scheme, netloc, path, sorted_query, ""))


class _PageHTMLParser(HTMLParser):
    """Потоковий парсер HTML на базі стандартної бібліотеки html.parser."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._ignore_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._links: list[str] = []
        self._seen_links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in _IGNORED_TAGS:
            self._ignore_depth += 1
            return

        if tag_lower == "title" and self._ignore_depth == 0:
            self._in_title = True

        if tag_lower in _BLOCK_TAGS and self._ignore_depth == 0:
            self._text_parts.append(" ")

        if tag_lower == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    canon = canonicalize_url(attr_val, base_url=self._base_url)
                    if canon is not None and canon not in self._seen_links:
                        self._seen_links.add(canon)
                        self._links.append(canon)
                    break

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in _IGNORED_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1
            return

        if tag_lower == "title":
            self._in_title = False

        if tag_lower in _BLOCK_TAGS and self._ignore_depth == 0:
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        self._text_parts.append(data)

    def build_result(self) -> tuple[str, str, list[str]]:
        title = " ".join("".join(self._title_parts).split())
        text = " ".join("".join(self._text_parts).split())
        return title, text, self._links


def extract_html_content(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Витягує `(title, visible_text, canonical_links)` із HTML-документа."""
    parser = _PageHTMLParser(base_url=base_url)
    parser.feed(html)
    parser.close()
    return parser.build_result()


def parse_retry_after(header_value: str | None) -> float | None:
    """Розбирає HTTP-заголовок `Retry-After` (секунди або HTTP-дата) у секунди очікування."""
    if header_value is None:
        return None
    raw = header_value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        target_dt = email.utils.parsedate_to_datetime(raw)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=UTC)
        delta = (target_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_sec: float = 10.0,
    max_retries: int = 3,
    base_backoff_sec: float = 0.25,
    jitter_sec: float = 0.15,
    max_backoff_sec: float = 30.0,
    on_retry: Callable[[], None] | None = None,
) -> FetchResult:
    """Завантажує URL з `asyncio.timeout` та повторами (429, 5xx, таймаут) з backoff + jitter."""
    t_start = time.perf_counter()

    for attempt in range(max_retries + 1):
        attempts_made = attempt + 1
        try:
            async with asyncio.timeout(timeout_sec):
                response = await client.get(url)
        except (TimeoutError, httpx.TimeoutException):
            if attempt < max_retries:
                if on_retry is not None:
                    on_retry()
                delay = min(
                    max_backoff_sec,
                    base_backoff_sec * (2**attempt) + random.uniform(0.0, jitter_sec),
                )
                await asyncio.sleep(delay)
                continue
            return FetchResult(
                url=url,
                final_url=url,
                status_code=None,
                content=b"",
                text="",
                content_type="",
                elapsed_sec=time.perf_counter() - t_start,
                attempts=attempts_made,
                error="TimeoutError",
            )
        except httpx.TooManyRedirects:
            return FetchResult(
                url=url,
                final_url=url,
                status_code=None,
                content=b"",
                text="",
                content_type="",
                elapsed_sec=time.perf_counter() - t_start,
                attempts=attempts_made,
                error="TooManyRedirects",
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                url=url,
                final_url=url,
                status_code=None,
                content=b"",
                text="",
                content_type="",
                elapsed_sec=time.perf_counter() - t_start,
                attempts=attempts_made,
                error=f"{type(exc).__name__}: {exc}",
            )

        status = response.status_code
        final_url = str(response.url)
        content_type = response.headers.get("content-type", "")

        if _is_retryable_status(status):
            if attempt < max_retries:
                if on_retry is not None:
                    on_retry()
                exp_delay = base_backoff_sec * (2**attempt) + random.uniform(0.0, jitter_sec)
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                raw_delay = max(exp_delay, retry_after) if retry_after is not None else exp_delay
                delay = min(max_backoff_sec, raw_delay)
                await asyncio.sleep(delay)
                continue

            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=status,
                content=response.content,
                text="",
                content_type=content_type,
                elapsed_sec=time.perf_counter() - t_start,
                attempts=attempts_made,
                error=f"HTTP {status}",
            )

        if 200 <= status < 300:
            return FetchResult(
                url=url,
                final_url=final_url,
                status_code=status,
                content=response.content,
                text=response.text,
                content_type=content_type,
                elapsed_sec=time.perf_counter() - t_start,
                attempts=attempts_made,
                error=None,
            )

        return FetchResult(
            url=url,
            final_url=final_url,
            status_code=status,
            content=response.content,
            text="",
            content_type=content_type,
            elapsed_sec=time.perf_counter() - t_start,
            attempts=attempts_made,
            error=f"HTTP {status}",
        )

    return FetchResult(
        url=url,
        final_url=url,
        status_code=None,
        content=b"",
        text="",
        content_type="",
        elapsed_sec=time.perf_counter() - t_start,
        attempts=max_retries + 1,
        error="MaxRetriesExceeded",
    )


class _AllowAllParser(RobotFileParser):
    """Правило за замовчуванням, коли robots.txt відсутній (HTTP 404)."""

    def can_fetch(self, useragent: str, url: str) -> bool:
        return True


class _DenyAllParser(RobotFileParser):
    """Правило повної заборони, коли доступ до robots.txt закритий (HTTP 401/403)."""

    def can_fetch(self, useragent: str, url: str) -> bool:
        return False


class RobotsCache:
    """Асинхронний кеш правил `robots.txt` окремо для кожного хоста."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        *,
        timeout_sec: float = 10.0,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def is_allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        """Перевіряє дозвіл на обхід `url` згідно з кешованим `robots.txt` хоста."""
        parts = urlsplit(url)
        host_key = f"{parts.scheme.lower()}://{parts.netloc.lower()}"

        cached = self._cache.get(host_key)
        if cached is not None:
            return cached.can_fetch(self.user_agent, url)

        async with self._locks[host_key]:
            cached = self._cache.get(host_key)
            if cached is not None:
                return cached.can_fetch(self.user_agent, url)

            robots_url = f"{host_key}/robots.txt"
            res = await fetch(
                client,
                robots_url,
                timeout_sec=self.timeout_sec,
                max_retries=1,
            )

            parser: RobotFileParser
            if res.status_code == 200:
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(res.text.splitlines())
            elif res.status_code in (401, 403):
                parser = _DenyAllParser()
            else:
                parser = _AllowAllParser()

            self._cache[host_key] = parser
            return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """Повертає вказаний у `robots.txt` `Crawl-delay` для хоста, якщо він є."""
        parts = urlsplit(url)
        host_key = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
        parser = self._cache.get(host_key)
        if parser is None:
            return None
        try:
            delay = parser.crawl_delay(self.user_agent)
            return float(delay) if delay is not None else None
        except (AttributeError, TypeError, ValueError):
            return None


async def crawl(
    start_urls: str | Sequence[str],
    *,
    max_pages: int = 50,
    concurrency: int = 5,
    per_host_concurrency: int = 2,
    min_host_delay: float = 0.05,
    timeout_sec: float = 10.0,
    max_retries: int = 3,
    base_backoff_sec: float = 0.25,
    jitter_sec: float = 0.15,
    same_host_only: bool = True,
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> AsyncIterator[CrawledPage]:
    """Асинхронний генератор обходу сайтів із `TaskGroup`, семафорами та канонізацією URL."""
    if max_pages < 1:
        raise ValueError("max_pages має бути >= 1")
    if concurrency < 1:
        raise ValueError("concurrency має бути >= 1")
    if per_host_concurrency < 1:
        raise ValueError("per_host_concurrency має бути >= 1")

    raw_seeds = [start_urls] if isinstance(start_urls, str) else list(start_urls)
    seeds: list[str] = []
    seen: set[str] = set()
    allowed_hosts: set[str] = set()

    for raw in raw_seeds:
        canon = canonicalize_url(raw)
        if canon is not None and canon not in seen:
            seen.add(canon)
            seeds.append(canon)
            allowed_hosts.add(urlsplit(canon).netloc)

    if not seeds:
        return

    active_stats = stats if stats is not None else CrawlStats()
    active_stats.started_at = time.perf_counter()

    url_queue: asyncio.Queue[str] = asyncio.Queue()
    for seed in seeds:
        url_queue.put_nowait(seed)
    active_stats.queue_depth = url_queue.qsize()

    results_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(per_host_concurrency)
    )
    host_rate_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    host_last_request: dict[str, float] = {}
    robots_cache = RobotsCache(user_agent=user_agent, timeout_sec=timeout_sec)

    stop_event = asyncio.Event()
    collected_count = 0
    collect_lock = asyncio.Lock()

    def _emit_log(line: str) -> None:
        if log_callback is not None:
            log_callback(line)

    def _on_retry() -> None:
        active_stats.retries += 1

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal collected_count
        while not stop_event.is_set():
            url = await url_queue.get()
            active_stats.queue_depth = url_queue.qsize()
            try:
                if stop_event.is_set():
                    continue

                if respect_robots:
                    allowed = await robots_cache.is_allowed(client, url)
                    if not allowed:
                        active_stats.skipped_robots += 1
                        _emit_log(f"ROBOTS_DISALLOW	-	0	0.000s	{url}")
                        continue

                host = urlsplit(url).netloc
                robots_delay = robots_cache.crawl_delay(url) if respect_robots else None
                effective_delay = max(min_host_delay, robots_delay or 0.0)

                async with global_sem, host_sems[host]:
                    if stop_event.is_set():
                        continue

                    if effective_delay > 0.0:
                        async with host_rate_locks[host]:
                            now = time.monotonic()
                            last = host_last_request.get(host)
                            if last is not None:
                                wait_sec = effective_delay - (now - last)
                                if wait_sec > 0.0:
                                    await asyncio.sleep(wait_sec)
                            host_last_request[host] = time.monotonic()

                    if stop_event.is_set():
                        continue

                    active_stats.in_flight += 1
                    try:
                        res = await fetch(
                            client,
                            url,
                            timeout_sec=timeout_sec,
                            max_retries=max_retries,
                            base_backoff_sec=base_backoff_sec,
                            jitter_sec=jitter_sec,
                            on_retry=_on_retry,
                        )
                    finally:
                        active_stats.in_flight -= 1

                status_label = str(res.status_code) if res.status_code is not None else "ERR"
                if not res.ok:
                    active_stats.errors += 1
                    _emit_log(
                        f"ERROR({res.error})	{status_label}	{len(res.content)}B	"
                        f"{res.elapsed_sec:.3f}s	attempts={res.attempts}	{url}"
                    )
                    continue

                final_canon = canonicalize_url(res.final_url) or url
                seen.add(final_canon)

                is_html = (
                    "text/html" in res.content_type.lower()
                    or "application/xhtml" in res.content_type.lower()
                    or not res.content_type
                )
                if is_html:
                    title, visible_text, links = extract_html_content(
                        res.text, base_url=final_canon
                    )
                else:
                    title, visible_text, links = "", res.text, []

                page = CrawledPage(
                    url=url,
                    final_url=final_canon,
                    status_code=res.status_code or 200,
                    title=title or final_canon,
                    text=visible_text,
                    links=tuple(links),
                    bytes_received=len(res.content),
                    elapsed_sec=res.elapsed_sec,
                    attempts=res.attempts,
                    fetched_at=datetime.now(UTC).isoformat(),
                )

                should_yield = False
                async with collect_lock:
                    if collected_count < max_pages:
                        collected_count += 1
                        active_stats.collected = collected_count
                        active_stats.bytes_downloaded += page.bytes_received
                        should_yield = True
                        if collected_count >= max_pages:
                            stop_event.set()

                if should_yield:
                    _emit_log(
                        f"OK	{page.status_code}	{page.bytes_received}B	"
                        f"{page.elapsed_sec:.3f}s	attempts={page.attempts}	{page.final_url}"
                    )
                    await results_queue.put(page)

                if not stop_event.is_set():
                    for link in links:
                        if link in seen:
                            continue
                        if same_host_only and urlsplit(link).netloc not in allowed_hosts:
                            continue
                        seen.add(link)
                        url_queue.put_nowait(link)
                    active_stats.queue_depth = url_queue.qsize()
            finally:
                url_queue.task_done()
                active_stats.queue_depth = url_queue.qsize()

    async def _coordinator() -> None:
        try:
            async with (
                httpx.AsyncClient(
                    headers={"User-Agent": user_agent},
                    follow_redirects=True,
                    transport=transport,
                ) as client,
                asyncio.TaskGroup() as tg,
            ):
                worker_tasks: list[asyncio.Task[None]] = [
                    tg.create_task(_worker(client)) for _ in range(concurrency)
                ]

                stop_waiter: asyncio.Task[Any] = asyncio.create_task(stop_event.wait())
                join_waiter: asyncio.Task[Any] = asyncio.create_task(url_queue.join())
                waiters: set[asyncio.Task[Any]] = {stop_waiter, join_waiter}
                try:
                    _done, pending = await asyncio.wait(
                        waiters,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                finally:
                    stop_event.set()
                    for wt in worker_tasks:
                        wt.cancel()
        finally:
            await results_queue.put(None)

    coordinator_task = asyncio.create_task(_coordinator())
    try:
        while True:
            item = await results_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not coordinator_task.done():
            coordinator_task.cancel()
        await asyncio.gather(coordinator_task, return_exceptions=True)
