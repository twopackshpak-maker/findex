from __future__ import annotations

import asyncio
import statistics
import time
import httpx

QUERIES = [
    "concurrency",
    "python",
    '"inverted index"',
    "bm25",
    "latency",
    "pickle serialization",
]


async def run_benchmark(
    base_url: str = "http://127.0.0.1:8000",
    total_requests: int = 500,
    concurrency: int = 20,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    errors = 0

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=10.0) as client:
        await client.get("/health")

        async def worker(idx: int) -> None:
            nonlocal errors
            q = QUERIES[idx % len(QUERIES)]
            async with sem:
                t0 = time.perf_counter()
                try:
                    resp = await client.get("/api/search", params={"q": q, "top_k": 10})
                    if resp.status_code != 200:
                        errors += 1
                except Exception:
                    errors += 1
                finally:
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_start = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        wall_sec = time.perf_counter() - t_start

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[int(0.50 * (n - 1))]
    p95 = latencies_ms[int(0.95 * (n - 1))]
    p99 = latencies_ms[int(0.99 * (n - 1))]

    print(f"Requests:    {total_requests} (concurrency={concurrency})")
    print(f"Errors:      {errors}")
    print(f"Total time:  {wall_sec:.2f} s")
    print(f"Throughput:  {total_requests / wall_sec:.1f} req/s (RPS)")
    print(
        f"Latency:     avg={statistics.mean(latencies_ms):.2f} ms | "
        f"p50={p50:.2f} ms | p95={p95:.2f} ms | p99={p99:.2f} ms"
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())