# Lab 8: scorer and mmap measurements

Baseline: 1,200 docs, 41,999 postings; Python 3.12.5, NumPy 2.5.5.
Medians of 9 runs after warmup; scorer-only API includes parsing, top-k, and result metadata.

| Workload | Query | Scalar (ms) | NumPy (ms) | Speedup |
|---|---|---:|---:|---:|
| частий терм | `worker` | 60.441 | 46.538 | 1.30× |
| рідкісний терм | `0` | 1.980 | 2.023 | 0.98× |
| три терми | `whale OR sea OR captain` | 0.013 | 0.028 | 0.47× |

10× expanded index: 12,000 docs, 419,990 postings; `.npy` payload 3.25 MiB; mapped search median 0.066 ms; process peak RSS before/after map 373.5/373.5 MiB.
The mmap holds contiguous int32 posting IDs, term frequencies, and document lengths. A small-list query sees little benefit because Python call and query parsing overhead dominate; mmap avoids eagerly copying all posting data into Python-managed arrays.
