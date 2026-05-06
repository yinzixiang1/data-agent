# NL2SQL Data Agent

自然语言 → RAG 检索相关表/列/枚举 → 组装 Prompt → LLM 生成 SQL → EXPLAIN 校验自动修复。

支持 HTTP 服务模式（对接 admin 控制台）和 CLI 交互模式（本地调试）。

## 架构

```
用户提问: "这个月 CIMB 分 local/swift 的交易金额"
  │
  ├─❶ 业务术语解析 (glossary_resolver)
  │    MySQL glossary → enriched_query + business_context
  │    "SWIFT" → payment_method = 2000（大小写不敏感）
  │
  ├─❷ Schema 混合检索 (hybrid_searcher)
  │    BGE-M3 一次 encode → Dense + Sparse
  │    ├── 表级: Milvus hybrid_search → RRF 融合
  │    ├── 列级: hybrid_search → 反推表 (+0.01)
  │    ├── 枚举反哺: 枚举命中的表加分 (+0.02)
  │    └── 关联补全: top 表的 relations 关联表加分 (×0.1)
  │
  ├─❸ Reranker 精排 (reranker)
  │    BGE-Reranker-v2-M3 从 top 10 → top 5
  │    + 被 Reranker 淘汰但与 top 表有关联的表补回
  │
  ├─❹ 枚举值检索 (hybrid_searcher.search_enums)
  │    "LOCAL" → pmt_payment_beneficiary.payment_method = 1000
  │
  ├─❺ Few-shot 示例检索 (fewshot_selector)
  │    语义相似 + 表重叠加权 + MMR 多样性 → top 3
  │
  ├─❻ Prompt 组装 (schema_formatter)
  │    DDL + 列注释/description + JOIN 提示 + 枚举映射 + 术语上下文 + Few-shot
  │
  └─❼ SQL 生成 + EXPLAIN 校验 (多轮对话)
       LLM 生成 SQL
       → EXPLAIN 语法校验（失败则带错误重试，最多 N 轮，保持对话上下文）
       → 执行计划交 LLM 分析，LGTM 或自动优化
```

## 项目结构

```
data-agen/
├── app.py                               # FastAPI HTTP 服务入口
├── main.py                              # CLI 交互入口
├── .env                                 # 环境变量
├── docker-compose-milvus.yaml           # Milvus + Attu 部署
│
├── src/retrieval/                       # RAG 检索核心
│   ├── config.py                        # 静态配置（环境变量 + 默认值）
│   ├── agent_config.py                  # Agent 动态配置加载（MySQL）
│   ├── query_logger.py                  # 查询日志记录（写 sys_query_log）
│   ├── embedding.py                     # BGE-M3 封装（Dense + Sparse）
│   ├── milvus_store.py                  # Milvus 向量存储（hybrid_search + RRF）
│   ├── schema_loader.py                 # Doris DDL + MySQL 语义层合并
│   ├── document_builder.py              # Schema → 表级/列级/枚举级检索文档
│   ├── index_manager.py                 # 全量构建 Milvus 索引
│   ├── hybrid_searcher.py               # 混合检索 + 枚举反哺 + 关联表补全
│   ├── reranker.py                      # BGE-Reranker 精排
│   ├── fewshot_selector.py              # Few-shot 检索（Milvus + MMR）
│   ├── glossary_resolver.py             # 业务术语解析
│   ├── schema_formatter.py              # 检索结果 → DDL Prompt
│   ├── sql_validator.py                 # EXPLAIN 校验器
│   └── retriever.py                     # RAG 统一入口
│
├── docs/
│   ├── table_template.yaml              # 语义层数据模板
│   └── 数据交付规范.md                    # 填写指南
│
└── tests/
    └── test_retrieval_offline.py
```

## 快速开始

### 1. 启动 Milvus

```bash
docker compose -f docker-compose-milvus.yaml up -d
```

| 服务 | 端口 | 用途 |
|------|------|------|
| Milvus API | 19530 | 向量存储 |
| Attu GUI | 8000 | 可视化管理 |
| MinIO | 9001 | 对象存储 |

### 2. 配置 `.env`

```bash
# Doris
DORIS_HOST="your-doris-host"
DORIS_PORT="9030"
DORIS_USER="root"
DORIS_PASSWORD="your-password"
DORIS_DATABASE="dwd_banking"

# MySQL 语义层（data_agent 库，由 admin-api 管理）
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
DEFAULT_AGENT_ID=7              # HTTP 模式默认绑定的 Agent ID
```

### 3. 启动

#### HTTP 服务模式（生产部署，对接 admin 控制台）

```bash
# 默认配置启动
uvicorn app:app --host 0.0.0.0 --port 9090

# 绑定 Agent 启动
DEFAULT_AGENT_ID=7 uvicorn app:app --host 0.0.0.0 --port 9090
```

#### CLI 交互模式（本地调试）

```bash
# 默认配置
python main.py

# 绑定 Agent
python main.py --agent 7

# 调试模式（可叠加）
python main.py --agent 7 --debug
```

CLI 交互命令：`/quit` 退出 | `/debug` 切换日志 | `/prompt` 切换 Prompt 显示 | `/config` 查看当前配置

## 启动方式对比

|  | CLI 模式 (`main.py`) | HTTP 服务模式 (`app.py`) |
|--|---------------------|------------------------|
| **入口** | 终端 REPL 交互 | REST API |
| **绑定 Agent** | `--agent 7` | `DEFAULT_AGENT_ID=7` 或请求级 `agent_id` |
| **运行时切换 Agent** | 不支持 | `POST /admin/config-reload` 或请求级指定 |
| **查询日志** | 写 sys_query_log | 写 sys_query_log + 返回 log_id |
| **评估系统** | 不支持 | `POST /evaluation/run` |
| **索引重建** | 需重启 | `POST /admin/index-rebuild` |
| **控制台对接** | 不支持 | admin-api 调用 `/admin/*` 接口 |
| **适用场景** | 本地调试、效果验证 | 生产部署、对接前端 |

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（返回 initialized, agent, config_source） |
| POST | `/query` | NL2SQL 查询，支持 `agent_id` 动态切换 |
| POST | `/admin/index-rebuild` | 全量重建 Milvus 索引 |
| POST | `/admin/config-reload` | 重新加载 Agent 配置（不重建索引） |
| POST | `/evaluation/run` | 批量评估执行 |

### 查询示例

```bash
# 基本查询
curl -X POST http://localhost:9090/query \
  -H "Content-Type: application/json" \
  -d '{"question": "目前有多少活跃商户"}'

# 指定 Agent
curl -X POST http://localhost:9090/query \
  -H "Content-Type: application/json" \
  -d '{"question": "目前有多少活跃商户", "agent_id": 7}'

# 关闭 EXPLAIN 校验
curl -X POST http://localhost:9090/query \
  -H "Content-Type: application/json" \
  -d '{"question": "目前有多少活跃商户", "enable_explain": false}'
```

## 动态配置

### 配置优先级

```
da_agent_config（Agent 专属配置）    ← admin 控制台「Agent 配置」面板
        ↓ 覆盖
sys_config（全局默认值）             ← admin 控制台「系统配置」面板
        ↓ 覆盖
.env / 代码默认值                   ← 本地环境变量
```

### 与 admin 控制台的对接

```
┌── dataAgent-admin-api（控制台）──────────────────┐
│                                                  │
│  da_agent         → Agent 基础信息               │
│  da_agent_config  → 分段配置 (model/prompt/...)  │
│  da_agent_ref     → 资源引用 (provider/...)      │
│  res_resource     → 资源详情 (base_url/api_key)  │
│  sys_config       → 全局参数 (top_k/reranker)    │
│                                                  │
│  POST /system/index-rebuild ──────────────────────┼──→ data-agen /admin/index-rebuild
│                                                  │
└──────────────────────────────────────────────────┘
                        │
                        │ MySQL data_agent 库
                        ▼
┌── data-agen（RAG 引擎）─────────────────────────┐
│                                                  │
│  AgentConfigLoader.load(agent_id=7)              │
│    ① sys_config       → 全局默认值               │
│    ② da_agent         → Agent 名称/状态          │
│    ③ da_agent_config  → LLM/Prompt/检索参数      │
│    ④ da_agent_ref     → 关联的 provider 资源     │
│    ⑤ res_resource     → provider 的 base_url     │
│    → 合并为 AgentRuntimeConfig                   │
│                                                  │
│  配置驱动: LLM 客户端 / System Prompt /          │
│           检索参数 / EXPLAIN 参数                 │
│                                                  │
└──────────────────────────────────────────────────┘
```

admin-api 端需配置 `DATA_AGENT_BASE_URL=http://localhost:9090`。

### 可配置项

通过 admin 控制台的 Agent 配置（`da_agent_config`）可动态调整：

| 配置段 | 字段 | 说明 |
|--------|------|------|
| **model** | provider, model, temperature, api_key, base_url | LLM 供应商和模型参数 |
| **prompt** | system_prompt | 系统提示词 |
| **retrieval** | table_search_top_k, rerank_input_top_k, fewshot_top_k, rrf_k, mmr_lambda, enable_reranker, enable_explain, max_fix_retries | 检索和校验参数 |

## 代码导读

### 数据流：启动阶段

```
app.py / main.py
  → AgentConfigLoader.load()       # 加载 Agent 配置（MySQL）
  → print_config()                 # 打印配置信息
  → SchemaRetriever.initialize()
    → SchemaLoader.load_all()      # MySQL(da_*表) + Doris DDL → 合并 Schema
    → GlossaryResolver.load()      # 加载术语表
    → IndexManager.build()         # 全量构建 Milvus 索引
       → DocumentBuilder.build_all()
       → BGEEmbedding.encode()     # → Dense + Sparse
       → MilvusIndex.insert()      # 写入 4 个 Collection
    → HybridSearcher()
    → FewShotSelector()
```

### 数据流：每次查询

```
/query 或 CLI 输入
  → SchemaRetriever.retrieve()
    → GlossaryResolver.resolve()       # 术语匹配
    → HybridSearcher.search()          # 混合检索
    → Reranker.rerank()                # 精排
    → 补回被淘汰的关联表
    → HybridSearcher.search_enums()    # 枚举值映射
    → FewShotSelector.select()         # Few-shot
    → SchemaFormatter.format_all()     # 组装 Prompt
  → LLM 生成 SQL（多轮对话）
  → SQLValidator.validate()            # EXPLAIN 校验
  → 失败 → 追加错误重试（最多 N 轮）
  → 通过 → 执行计划分析 → LGTM 或优化
  → QueryLogger.log()                  # 记录查询日志
```

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **静态配置** | `config.py` | 环境变量和默认值（.env fallback） |
| **动态配置** | `agent_config.py` | 从 MySQL 加载 Agent 配置，合并为 AgentRuntimeConfig |
| **查询日志** | `query_logger.py` | 写 sys_query_log，对接 admin 控制台查询历史 |
| **Embedding** | `embedding.py` | BGE-M3 封装，`encode()` 同时返回 Dense + Sparse |
| **向量存储** | `milvus_store.py` | Milvus Collection CRUD + hybrid_search (RRF) |
| **数据加载** | `schema_loader.py` | Doris DDL + MySQL 语义层合并 |
| **文档构建** | `document_builder.py` | Schema → 三层文档（表/列/枚举） |
| **索引管理** | `index_manager.py` | 全量构建 4 个 Milvus Collection |
| **混合检索** | `hybrid_searcher.py` | 表级+列级检索 → 枚举反哺 → 关联补全 |
| **精排** | `reranker.py` | BGE-Reranker-v2-M3 交叉编码器 |
| **Few-shot** | `fewshot_selector.py` | Milvus 检索 + 表重叠加权 + MMR |
| **术语解析** | `glossary_resolver.py` | 大小写不敏感匹配，输出 enriched_query + business_context |
| **Prompt 格式化** | `schema_formatter.py` | CREATE TABLE DDL 格式组装 |
| **SQL 校验** | `sql_validator.py` | 提取 SQL + EXPLAIN + 执行计划 |
| **RAG 入口** | `retriever.py` | 串联全流程 + Reranker 后关联补回 |

## Milvus Collections（数据库: nl2sql）

| Collection | 内容 | 关键标量字段 |
|------------|------|------------|
| `nl2sql_table` | 表级文档 | table_name, schema_json |
| `nl2sql_column` | 列级文档 | table_name, column_name, is_enum |
| `nl2sql_enum` | 枚举值文档 | table_name, column_name, enum_label_cn, sql_value |
| `nl2sql_fewshot` | Few-shot 示例 | question, sql, involved_tables |

所有 Collection 包含 `dense_vec`(FLOAT_VECTOR 1024, COSINE) + `sparse_vec`(SPARSE_FLOAT_VECTOR, IP)，检索通过 `hybrid_search` + `RRFRanker(k=60)` 融合。高频过滤字段加 INVERTED 索引。

## 语义层维护

语义层数据存储在 MySQL `data_agent` 库，通过 dataAgent-admin-api 控制台管理。

### MySQL 表

| 表 | 说明 |
|---|------|
| `da_table` | 表语义（name, display_name, description, tags, query_tips） |
| `da_table_column` | 列语义（FK → da_table.id） |
| `da_table_relation` | 表关联关系（FK → da_table.id） |
| `da_table_query` | 表级常见问题（FK → da_table.id） |
| `da_glossary` | 业务术语 |
| `da_enum_def` / `da_enum_value` | 枚举字典（父子表） |
| `da_fewshot` | 全局 Few-shot 示例 |
| `da_agent` / `da_agent_config` / `da_agent_ref` | Agent 配置 |
| `res_resource` | 通用资源（provider / vector_db / tool） |
| `sys_config` | 全局系统配置 |
| `sys_query_log` | 查询日志 |
| `sys_eval_case` / `sys_eval_run` / `sys_eval_result` | 评估体系 |

### 变更感知

| 变更 | 操作 |
|------|------|
| 改语义层数据（表/列/术语/枚举） | `POST /admin/index-rebuild` 或重启 |
| 改 Agent 配置（LLM/Prompt/检索参数） | `POST /admin/config-reload` 或重启 |
| Doris 加表/加列 | `POST /admin/index-rebuild` 或重启 |
| 改检索逻辑/算法 | 重新部署 |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Embedding | BGE-M3 | 中文最佳，一次 encode 同时出 Dense + Sparse |
| 向量数据库 | Milvus 2.5 | 原生 Dense + Sparse 混合检索，内置 RRF |
| Reranker | BGE-Reranker-v2-M3 | 交叉编码器精排，与 BGE-M3 配套 |
| 融合算法 | RRF (k=60) | Milvus 内置，不同量纲天然适配 |
| Few-shot | MMR (λ=0.7) | 平衡相关性和多样性 |
| LLM | 动态配置 | 默认 DeepSeek，可通过控制台切换任意 provider |
| Schema 格式 | CREATE TABLE DDL | LLM 最熟悉，准确率最高 |
| 语义层存储 | MySQL (data_agent) | OLTP 适合 CRUD，admin API 管理 |
| HTTP 框架 | FastAPI | 异步，自带 OpenAPI 文档 |
