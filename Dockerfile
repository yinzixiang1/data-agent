FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.13-slim

ENV CODEX_HOME=/var/lib/lumen-codex \
    PATH=/app/.venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project \
    && .venv/bin/python -c "from codex_cli_bin import bundled_codex_path; from pathlib import Path; Path('/usr/local/bin/codex').symlink_to(bundled_codex_path())" \
    && mkdir -p /var/lib/lumen-codex \
    && chmod 700 /var/lib/lumen-codex

COPY app.py ./
COPY src/ ./src/

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9090/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
