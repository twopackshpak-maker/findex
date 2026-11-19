# Stage 1: Builder
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md* ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev && \
    uv run python scripts/fetch_corpus.py || true && \
    if [ -d data/corpus ] && [ "$(ls -A data/corpus)" ]; then \
        uv run findex index data/corpus --out data/index.bin --positions; \
    else \
        mkdir -p data/corpus && \
        printf 'Moby Dick\nCall me Ishmael. White whale and harpoon on the high seas.\n' > data/corpus/0001_moby.txt && \
        printf 'Sherlock Holmes\nElementary deduction in London with Watson.\n' > data/corpus/0002_holmes.txt && \
        printf 'Information Retrieval\nInverted index with BM25 and TF-IDF ranking in Python.\n' > data/corpus/0003_ir.txt && \
        uv run findex index data/corpus --out data/index.bin --positions; \
    fi

# Stage 2: Runtime (non-root)
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid appgroup --create-home appuser

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
COPY --from=builder --chown=appuser:appgroup /app/data /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/data/index.bin \
    CORPUS_DIR=/app/data/corpus

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health')" || exit 1

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000}"]
