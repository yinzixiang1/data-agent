FROM python:3.13-slim

WORKDIR /app

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

# 复制源码
COPY app.py main.py ./
COPY src/ ./src/
COPY config/ ./config/

EXPOSE 9090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
