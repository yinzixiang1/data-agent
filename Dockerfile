FROM python:3.13-slim

WORKDIR /app

# Subscription auth is supplied at runtime through a dedicated named volume.
# No Codex credentials are copied into the image.
ENV CODEX_HOME=/var/lib/lumen-codex

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装 PyTorch CUDA（最大的层，缓存住）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# 再装项目依赖（torch 已存在会跳过）
COPY pyproject.toml ./
RUN pip install --no-cache-dir . \
    pymilvus \
    langchain \
    langchain-openai \
    langchain-anthropic \
    anthropic

# The SDK bundles the matching Codex CLI but does not install a console script.
# Expose it for the one-time subscription device login performed in the container.
RUN python -c "from codex_cli_bin import bundled_codex_path; from pathlib import Path; Path('/usr/local/bin/codex').symlink_to(bundled_codex_path())"

# 复制源码
COPY app.py main.py ./
COPY src/ ./src/
RUN mkdir -p ./config /var/lib/lumen-codex \
    && chmod 700 /var/lib/lumen-codex

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
