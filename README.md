# NL2SQL Data Agent — RAG 引擎

自然语言转 SQL 的 RAG 检索引擎。接收用户自然语言问题，通过多路混合检索定位相关 Schema，组装 Prompt 交由 LLM 生成 SQL，再经 EXPLAIN 校验自动修复。

支持 HTTP 服务模式（对接管理后台）和 CLI 交互模式（本地调试）。

## 处理流程

```
用户提问: "这个月 CIMB 分 local/swift 的交易金额"
  │
  ├─❶ 多轮上下文压缩 (context_compressor)
  │    history_summary + 当前问题 → LLM 合并为完整独立问题
  │
  ├─❷ 业务术语解析 (glossary_resolver)
  │    Milvus Dense+BM25 混合检索 → enriched_query + business_context
  │    术语仅补充业务含义；具体表、字段、JSON 路径和枚举由当前 Schema 推理
  │
  ├─❸ Schema 混合检索 (hybrid_searcher)
  │    Qwen3-Embedding 编码 → Dense + BM25 双路召回
  │    ├── 表级: Milvus hybrid_search → Weighted/RRF 融合
  │    ├── 列级: hybrid_search → 归一化后反推表
  │    ├── 枚举反哺: 枚举命中按相对分数加权
  │    └── 关联补全: top 表的 relations 关联表加分 (×0.1)
  │
  ├─❹ Reranker 精排 (reranker)
  │    交叉编码器从 top N → top K
  │    + 被 Reranker 淘汰但与 top 表有关联的表补回
  │
  ├─❺ Schema Linking 值匹配 (value_indexer，精确命中表保底)
  │    jieba 实体抽取 → BM25 枚举值搜索
  │    "CIMB" → pmt_account.bank_name = 'CIMB'
  │
  ├─❻ 枚举值检索 (hybrid_searcher.search_enums)
  │    "LOCAL" → pmt_payment_beneficiary.payment_method = 1000
  │
  ├─❼ Few-shot 示例检索 (fewshot_selector)
  │    语义相似 + 表重叠加权 + MMR 多样性 → top 3
  │
  ├─❽ Prompt 组装 (schema_formatter)
  │    DDL + 列注释 + JOIN 提示 + 枚举映射 + 术语上下文 + Few-shot
  │
  └─❾ SQL 生成 + EXPLAIN 校验 (app.py)
       LLM 生成 SQL
       → EXPLAIN 语法校验（失败则带错误重试，最多 N 轮）
       → 执行计划交 LLM 分析 → LGTM 或自动优化
```

## 项目结构

```
data-agen/
├── app.py                               # FastAPI HTTP 服务入口
├── main.py                              # CLI 交互入口
├── .env                                 # 环境变量
├── docker-compose-milvus.yaml           # Milvus 本地部署
├── config/                              # 导出的本地配置文件（.gitignore）
│
├── src/retrieval/                       # RAG 检索核心
│   ├── config.py                        # 静态配置（.env fallback）
│   ├── config_export.py                 # 配置导出工具（MySQL → JSON）
│   ├── agent_config.py                  # 三层动态配置加载（MySQL / local）
│   ├── context_compressor.py            # 多轮对话上下文压缩
│   ├── query_logger.py                  # 查询日志（写 sys_query_log）
│   ├── embedding.py                     # Qwen3-Embedding（local + API 双模式）
│   ├── reranker.py                      # Reranker 精排（local + API 双模式）
│   ├── milvus_store.py                  # Milvus 封装（BM25 Function + HNSW Dense）
│   ├── schema_loader.py                 # MySQL 语义层 + Doris DDL 合并
│   ├── document_builder.py              # Schema → 表/列/枚举级检索文档
│   ├── index_manager.py                 # Milvus 索引管理（build/connect）
│   ├── hybrid_searcher.py               # 混合检索 + 枚举反哺 + 关联补全
│   ├── ranker_strategy.py               # per-collection 检索策略（CSC）
│   ├── value_indexer.py                 # Schema Linking（jieba 实体抽取 + BM25）
│   ├── fewshot_selector.py              # Few-shot（表重叠加权 + MMR 多样性）
│   ├── glossary_resolver.py             # 业务术语解析
│   ├── schema_formatter.py              # 检索结果 → DDL Prompt
│   ├── sql_validator.py                 # EXPLAIN 校验器
│   └── retriever.py                     # RAG 统一入口（串联全流程）
│
├── docs/                                # 设计文档
│   ├── RAG检索体系与向量化设计.md
│   ├── 数据交付规范.md
│   ├── 向量化文档格式示例_pmt_account.md
│   ├── 语义层生成_Claude_Prompt.md
│   └── table_template.yaml              # 语义层数据模板
│
└── tests/
    └── test_retrieval_offline.py
```

## 快速开始

### 前置依赖

| 依赖 | 说明 |
|------|------|
| Python 3.12+ | Conda 环境 `data_agen` |
| MySQL 8.0+ | 语义层元数据 + Agent 配置 |
| Milvus 2.5+ | 向量数据库（本地 Docker 或远程部署） |
| Doris | 目标查询数据库（远程） |

### 配置 `.env`

```bash
# Doris（目标查询数据库）
DORIS_HOST="your-doris-host"
DORIS_PORT="9030"
DORIS_USER="root"
DORIS_PASSWORD="your-password"
DORIS_DATABASE="dwd_banking"

# MySQL（语义层，由 admin-api 管理）
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
MYSQL_USER="root"
MYSQL_PASSWORD="your-password"
MYSQL_DATABASE="data_agent"

# LLM（.env 作为 fallback，优先使用 Agent 配置）
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_API_KEY="your-api-key"

# Milvus
MILVUS_URI="http://localhost:19530"
MILVUS_DB="nl2sql"

# 可选
HF_HUB_OFFLINE=1               # 跳过 HuggingFace 联网检查
DEFAULT_AGENT_TOKEN="your-token" # API 认证 Token
```

### 启动

**HTTP 服务模式（从 MySQL 加载配置）**

```bash
# 首次启动（构建索引）
CONFIG_SOURCE=mysql CONFIG_PROFILE=1 REBUILD_INDEX_ON_STARTUP=true \
  uvicorn app:app --host 0.0.0.0 --port 9090 --reload

# 日常启动（复用已有索引，秒级）
CONFIG_SOURCE=mysql CONFIG_PROFILE=1 \
  uvicorn app:app --host 0.0.0.0 --port 9090 --reload
```

**HTTP 服务模式（从本地文件加载配置）**

```bash
# 先导出配置到本地文件
python -m src.retrieval.config_export --agent-id 1

# 使用本地配置启动（无需 MySQL 中的 sys_config / agent_config 表）
CONFIG_SOURCE=local CONFIG_PROFILE=config/agent_config.json \
  uvicorn app:app --host 0.0.0.0 --port 9090 --reload
```

> 本地模式下，语义层数据（表/列/术语/枚举/Few-shot）仍从 MySQL 加载，只有启动配置（模型、检索参数等）从本地 JSON 读取。

**CLI 交互模式（本地调试）**

```bash
python main.py --agent 1           # 绑定 Agent
python main.py --agent 1 --debug   # 调试模式
```

CLI 命令：`/quit` 退出 | `/debug` 切换日志 | `/prompt` 显示 Prompt | `/config` 查看配置

### 关键环境变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `CONFIG_SOURCE` | 否 | 配置加载方式：`mysql`（默认，从数据库加载）或 `local`（从本地 JSON 文件加载） |
| `CONFIG_PROFILE` | 是 | mysql 模式下为 Agent ID（如 `1`），local 模式下为配置文件路径（如 `config/agent_config.json`） |
| `REBUILD_INDEX_ON_STARTUP` | 否 | `true` 全量重建索引，`false`（默认）复用已有 Collection |
| `DENSE_DEVICE` | 否 | 推理设备，Mac 默认 `mps`，GPU 服务器设为 `cuda` |

> `NL2SQL_ENV` 已废弃。模型、维度、检索参数全部从数据库或本地配置文件加载。

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/query` | NL2SQL 查询（支持 `agent_id` 动态切换） |
| POST | `/admin/index-rebuild` | 全量重建 Milvus 索引 |
| POST | `/admin/config-reload` | 重新加载 Agent 配置（不重建索引） |
| POST | `/evaluation/run` | 批量评估执行 |

### 查询请求

```bash
curl -X POST http://localhost:9090/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "目前有多少活跃商户",
    "agent_id": 1,
    "history_summary": "上一轮问题|||上一轮SQL"
  }'
```

### 查询响应

```json
{
  "sql": "SELECT COUNT(*) AS \"活跃商户数\" FROM pmt_account WHERE status = 1",
  "answer": "...",
  "error": null,
  "retry_count": 0,
  "execution_time_ms": 2350,
  "log_id": 42,
  "context_summary": "目前有多少活跃商户|||SELECT COUNT(*) ...",
  "trace": {
    "question": "目前有多少活跃商户",
    "effective_question": "目前有多少活跃商户",
    "steps": [...],
    "total_duration_ms": 2350
  }
}
```

## 配置体系

### 三层优先级

```
da_agent_config（Agent 专属）    ← 管理后台「Agent 配置」
        ↓ 覆盖
sys_config（全局默认）            ← 管理后台「系统配置」
        ↓ 覆盖
.env / 代码默认值               ← 本地环境变量
```

### Agent 级可覆盖项

| 配置段 | 字段 | 说明 |
|--------|------|------|
| **model** | model, temperature, base_url, api_key | LLM 模型和参数 |
| **model** | embedding_config | Embedding 模型/维度覆盖（启动时加载） |
| **model** | reranker_config | Reranker 模型覆盖（启动时加载） |
| **prompt** | system_prompt, compress_prompt | 提示词 |
| **retrieval** | table_search_top_k, fewshot_top_k, enable_reranker, enable_explain, max_fix_retries ... | 检索和校验参数 |

> Embedding/Reranker 是启动时加载的重模型，运行时无法动态切换。修改后需重启服务。

## Milvus Collections

数据库 `nl2sql`，共 6 个 Collection：

| Collection | 检索方式 | 内容 | 关键标量字段 |
|------------|---------|------|------------|
| `nl2sql_table` | Dense + BM25 | 表级 Schema 文档 | table_name, schema_json |
| `nl2sql_column` | Dense + BM25 | 列级 Schema 文档 | table_name, column_name |
| `nl2sql_enum` | Dense + BM25 | 枚举定义文档 | table_name, column_name, enum_label_cn |
| `nl2sql_value` | **纯 BM25** | 枚举值（Schema Linking） | table_name, column_name, sql_value |
| `nl2sql_fewshot` | Dense + BM25 | Few-shot SQL 示例 | question, sql, involved_tables |
| `nl2sql_glossary` | Dense + BM25 | 业务术语 | term, synonyms, definition |

所有 Dense Collection 使用 HNSW (COSINE) + BM25 SPARSE_INVERTED_INDEX。BM25 分词器类型为 `"chinese"`（Milvus v2.5）。检索策略（ranker 类型、权重、recall 数量、是否 rerank）通过 `COLLECTION_SEARCH_CONFIG` 按 collection 独立配置。

## 语义层数据来源

语义层存储在 MySQL `data_agent` 库，通过管理后台维护：

| MySQL 表 | 说明 |
|----------|------|
| `da_semantic_table` / `da_semantic_column` | 表和列的语义描述 |
| `da_semantic_relation` | 表关联关系（JOIN 提示） |
| `da_semantic_query` | 表级常见问题 |
| `da_semantic_glossary` | 业务术语 |
| `da_semantic_enum` / `da_semantic_enum_value` | 枚举定义和枚举值 |
| `da_semantic_fewshot` | Few-shot SQL 示例 |

### 语义检索原则

- 表召回由表、列的中文名、描述、业务逻辑及不同字段的语义覆盖度共同决定。
- 业务口径通过语义层声明；修复召回问题时完善语义定义并重建索引。
- 禁止为具体问题、表名或字段名增加硬编码分支、专属权重或强制召回规则。
- 通用检索算法和 SQL 校验可以演进，但必须对所有语义表使用相同规则。

### 变更后操作

| 变更内容 | 操作 |
|----------|------|
| 语义层数据（表/列/术语/枚举/Few-shot） | `POST /admin/index-rebuild` 或重启 `REBUILD_INDEX_ON_STARTUP=true` |
| Agent 配置（LLM/Prompt/检索参数） | `POST /admin/config-reload`（热加载，无需重启） |
| Agent Embedding/Reranker 模型 | 必须重启服务 |
| Doris 加表加列 | `POST /admin/index-rebuild` |

## 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 静态配置 | `config.py` | .env 环境变量和 fallback 默认值 |
| 动态配置 | `agent_config.py` | 三层配置加载（sys_config → agent_config → .env），支持 MySQL 和本地 JSON 两种来源 |
| 配置导出 | `config_export.py` | 从 MySQL 导出启动配置到本地 JSON（`python -m src.retrieval.config_export --agent-id 1`） |
| 上下文压缩 | `context_compressor.py` | 多轮对话历史 + 当前问题 → LLM 合并为独立完整问题 |
| Embedding | `embedding.py` | Qwen3-Embedding 封装，支持 local（SentenceTransformers）和 API 两种模式 |
| Reranker | `reranker.py` | 交叉编码器精排，支持 local 和 API 两种模式 |
| 向量存储 | `milvus_store.py` | Milvus Collection CRUD、BM25 Function、Dense/Sparse/Hybrid 检索 |
| 数据加载 | `schema_loader.py` | MySQL 语义层 + Doris DDL 合并 |
| 文档构建 | `document_builder.py` | Schema → 表级/列级/枚举级检索文档 |
| 索引管理 | `index_manager.py` | build()=全量重建 / connect()=复用已有 Collection |
| 混合检索 | `hybrid_searcher.py` | per-collection 策略检索 + 枚举反哺 + 关联表补全 |
| 检索策略 | `ranker_strategy.py` | Collection 粒度的 ranker/权重/recall 配置解析 |
| 值匹配 | `value_indexer.py` | jieba 实体抽取 → BM25 枚举值搜索（Schema Linking） |
| Few-shot | `fewshot_selector.py` | Dense 检索 + 表重叠加权 + MMR 多样性选择 |
| 术语解析 | `glossary_resolver.py` | Dense+BM25 混合检索术语，输出 enriched_query + business_context |
| Prompt 格式化 | `schema_formatter.py` | 检索结果 → CREATE TABLE DDL + 注释 + JOIN + 枚举 |
| SQL 校验 | `sql_validator.py` | 提取 SQL + Doris EXPLAIN 语法校验 |
| 查询日志 | `query_logger.py` | 写 sys_query_log，记录完整查询链路 |
| RAG 入口 | `retriever.py` | SchemaRetriever 统一入口，串联全部检索流程 |

## 与管理后台的对接

```
┌── dataAgent-admin-api（管理后台 :8090）──────────┐
│                                                  │
│  MySQL data_agent 库                             │
│    da_agent / da_agent_config / da_agent_ref     │
│    da_table / da_glossary / da_enum_* / da_fewshot│
│    res_resource / sys_config                     │
│                                                  │
│  代理转发:                                        │
│    POST /api/agents/:id/query  ──→  :9090/query  │
│    POST /system/index-rebuild  ──→  :9090/admin/ │
│                                                  │
└──────────────────────────────────────────────────┘
                    │
                    │ 共享 MySQL data_agent 库
                    ▼
┌── data-agen（本项目 :9090）───────────────────────┐
│                                                  │
│  启动时:                                          │
│    AgentConfigLoader.load()                      │
│      → CONFIG_SOURCE=mysql: 从 DB 加载            │
│      → CONFIG_SOURCE=local: 从 JSON 文件加载       │
│      → 驱动 Embedding / Reranker / LLM 选择      │
│                                                  │
│  每次查询:                                        │
│    SchemaRetriever.retrieve(question)            │
│      → 术语 → Schema 检索 → Reranker → 值匹配    │
│      → 枚举 → Few-shot → Prompt 组装             │
│    LLM 生成 SQL → EXPLAIN 校验 → 自动修复        │
│                                                  │
└──────────────────────────────────────────────────┘
```

admin-api 通过 `DATA_AGENT_BASE_URL=http://localhost:9090` 连接本服务。

## Codex 订阅模式

内部少量用户可把 Agent 的 `model` 分区切到 Codex，同一台 data-agen
实例复用一次 ChatGPT 订阅登录。Codex 不使用 `api_key` 或 `base_url`，也不会
在 DeepSeek/OpenAI 路径失败时自动回退。

```json
{
  "provider": "codex",
  "model": "从状态接口返回的模型 ID",
  "codex_reasoning_effort": "low",
  "codex_timeout_seconds": 90,
  "codex_max_concurrency": 1
}
```

首次部署或订阅失效时，在服务器上执行一次设备登录：

```bash
docker compose -f docker-compose.prod.yml run --rm data-agen codex login --device-auth
```

认证文件只保存在 `codex-home` 命名卷中，不进入镜像或 Agent 配置。Agent 绑定
Codex 后，可通过管理令牌检查实时状态和当前账号可用的模型列表：

```bash
curl -H 'Authorization: Bearer <ADMIN_TOKEN>' \
  http://127.0.0.1:9090/admin/codex/status
```

每个查询使用独立的临时 Codex 会话、空工作目录、只读沙盒且禁止工具与审批。
90 秒是包含排队和多轮 SQL 修复在内的整条查询累计预算；默认只允许一条 Codex
查询管道并发执行。修改 Agent 配置后调用 `/admin/config-reload` 即可切换。
