# NL2SQL RAG 检索系统

自然语言 → RAG 检索相关表/列/枚举 → 组装 Prompt → DeepSeek 生成 SQL → EXPLAIN 校验自动修复。

## 架构

```
用户提问: "这个月 CIMB 分 local/swift 的交易金额"
  │
  ├─❶ 业务术语解析 (glossary_resolver)
  │    glossary YAML → enriched_query + business_context
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
  └─❼ SQL 生成 + EXPLAIN 校验 (main.py, 多轮对话)
       DeepSeek 生成 SQL
       → EXPLAIN 语法校验（失败则带错误重试，最多 5 轮，保持对话上下文）
       → 执行计划交 LLM 分析，LGTM 或自动优化
```

## 项目结构

```
data-agen/
├── main.py                              # 交互入口 + 多轮 EXPLAIN 校验
├── .env                                 # 环境变量（Doris/DeepSeek/Milvus）
├── docker-compose-milvus.yaml           # Milvus + Attu 部署
│
├── src/retrieval/                       # RAG 检索核心
│   ├── config.py                        # 集中配置（环境变量 + 默认值）
│   ├── embedding.py                     # BGE-M3 封装（Dense + Sparse）
│   ├── milvus_store.py                  # Milvus 向量存储（hybrid_search + RRF）
│   ├── schema_loader.py                 # Doris DDL + 语义层 YAML + 枚举加载合并
│   ├── document_builder.py              # Schema → 表级/列级/枚举级检索文档
│   ├── index_manager.py                 # 索引生命周期（hash 变更检测 + 全量构建/加载）
│   ├── hybrid_searcher.py               # 混合检索 + 枚举反哺 + 关联表补全
│   ├── reranker.py                      # BGE-Reranker 精排
│   ├── fewshot_selector.py              # Few-shot 检索（Milvus + MMR）
│   ├── glossary_resolver.py             # 业务术语解析（大小写不敏感）
│   ├── schema_formatter.py              # 检索结果 → DDL Prompt（含 description 输出）
│   ├── sql_validator.py                 # EXPLAIN 校验器（提取 SQL + 执行 EXPLAIN）
│   └── retriever.py                     # 对外统一入口（串联全流程 + Reranker 后关联补回）
│
├── semantic_layer/                      # 语义层（人工维护的业务知识）
│   ├── tables/
│   │   ├── banking/                     # banking 业务线（15 张表）
│   │   │   ├── pmt_account.yaml
│   │   │   ├── pmt_finance_payout.yaml
│   │   │   ├── pmt_payment_beneficiary.yaml
│   │   │   └── ...
│   │   └── sys_exchange_rate.yaml       # 公共表
│   ├── glossary/
│   │   └── banking/glossary.yaml        # 业务术语表
│   └── enums/
│       ├── banking.yaml                 # 枚举字典（从 sys_dim_enum_dict 生成）
│       └── acquiring.yaml
│
├── docs/
│   ├── table_template.yaml              # 表 YAML 模板（含字段消费说明）
│   ├── RAG_检索体系设计方案.md
│   └── 数据交付规范.md
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

# DeepSeek LLM
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_API_KEY="your-api-key"

# Milvus
MILVUS_URI="http://localhost:19530"
MILVUS_DB="nl2sql"

# 可选：跳过 HuggingFace 联网检查（模型已缓存时）
HF_HUB_OFFLINE=1
```

### 3. 运行

```bash
python main.py                # 自动检测 Doris，连不上走离线模式
python main.py --offline      # 强制离线
python main.py --rebuild      # 强制重建索引
python main.py --debug        # 详细日志
```

交互命令：`/quit` 退出 | `/debug` 切换日志 | `/prompt` 切换 Prompt 显示

## 代码导读

### 数据流：启动阶段

```
main.py
  → SchemaRetriever.initialize()
    → SchemaLoader.load_all()         # 加载 YAML + Doris DDL + 枚举
    → GlossaryResolver.load()         # 加载术语表
    → IndexManager.need_rebuild()     # SHA-256 hash 比对
    → IndexManager.build_and_save()   # 变更时全量重建
       → DocumentBuilder.build_all()  # Schema → 文档（表/列/枚举 三层）
       → BGEEmbedding.encode()        # 一次 encode → Dense + Sparse
       → MilvusIndex.insert()         # 写入 5 个 Collection
    → HybridSearcher()                # 初始化检索器
    → FewShotSelector()               # 初始化 Few-shot
```

### 数据流：每次查询

```
main.py: 用户输入
  → SchemaRetriever.retrieve()
    → GlossaryResolver.resolve()       # "SWIFT" → payment_method=2000
    → HybridSearcher.search()          # 表级+列级+枚举反哺+关联补全
    → Reranker.rerank()                # 10 → 5
    → 补回被 Reranker 淘汰的关联表     # retriever.py
    → HybridSearcher.search_enums()    # 枚举值映射
    → FewShotSelector.select()         # Few-shot 示例
    → SchemaFormatter.format_all()     # 组装 Prompt
  → llm_chat(messages)                # 多轮对话生成 SQL
  → SQLValidator.validate()            # EXPLAIN 校验
  → 失败 → 追加错误到 messages → 重试（最多 5 轮）
  → 通过 → 执行计划分析 → LGTM 或优化
```

### 核心模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **配置** | `config.py` | 所有环境变量和默认值，检索参数（top_k、RRF_K、MMR_LAMBDA 等） |
| **Embedding** | `embedding.py` | BGE-M3 封装，`encode()` 一次调用同时返回 Dense 向量 + Sparse 词权重 |
| **向量存储** | `milvus_store.py` | `MilvusIndex`: 创建 Collection、插入、hybrid_search (Dense+Sparse→RRF)；`MilvusMetaStore`: KV 元数据 |
| **数据加载** | `schema_loader.py` | 在线: Doris DESCRIBE → DDL，合并语义层 YAML；离线: 纯 YAML 构建；枚举: 优先查 Doris，回退本地 YAML |
| **文档构建** | `document_builder.py` | Schema dict → 向量化文本。三层文档：表级（表名+描述+标签+关键列+常见问题）、列级（列名+类型+枚举）、枚举级（每个枚举值独立一条） |
| **索引管理** | `index_manager.py` | SHA-256 hash 变更检测；全量构建写入 Milvus；加载已有索引 + 重建 table_schemas 映射 |
| **混合检索** | `hybrid_searcher.py` | 表级+列级 hybrid_search → RRF 合并 → 枚举反哺(+0.02) → 关联表补全(×0.1) → 枚举值检索 |
| **精排** | `reranker.py` | BGE-Reranker-v2-M3 交叉编码器，对候选表按语义相关度精排 |
| **Few-shot** | `fewshot_selector.py` | Milvus Dense 检索 + 表重叠度加权 + MMR(λ=0.7) 多样性选择 |
| **术语解析** | `glossary_resolver.py` | 大小写不敏感匹配 glossary 术语，输出 enriched_query + business_context |
| **Prompt 格式化** | `schema_formatter.py` | CREATE TABLE DDL 格式，输出 display_name + description + 枚举值 + JOIN 提示 + query_tips |
| **SQL 校验** | `sql_validator.py` | 从 LLM 输出提取 SQL，执行 EXPLAIN，返回 {valid, error, sql, plan} |
| **统一入口** | `retriever.py` | 串联全流程，Reranker 后补回被淘汰的关联表 |
| **交互入口** | `main.py` | 多轮对话式 SQL 生成，EXPLAIN 校验失败带上下文重试，执行计划 LLM 分析 |

## Milvus Collections（数据库: nl2sql）

| Collection | 内容 | 关键标量字段 |
|------------|------|------------|
| `nl2sql_table` | 表级文档 | table_name, schema_json |
| `nl2sql_column` | 列级文档 | table_name, column_name, is_enum |
| `nl2sql_enum` | 枚举值文档 | table_name, column_name, enum_label_cn, sql_value |
| `nl2sql_fewshot` | Few-shot 示例 | question, sql, involved_tables |
| `nl2sql_metadata` | 元数据 KV | key, value (存 schema_hash) |

所有向量 Collection 包含 `dense_vec`(FLOAT_VECTOR 1024, COSINE) + `sparse_vec`(SPARSE_FLOAT_VECTOR, IP)，检索通过 `hybrid_search` + `RRFRanker(k=60)` 融合。高频过滤字段加 INVERTED 索引。

## 语义层维护

### 新增表

1. 参考 `docs/table_template.yaml` 模板，在 `semantic_layer/tables/` 下创建 YAML
2. 重启服务 → 自动检测 hash 变更 → 重建索引

### 关键字段优先级

**`description` 是最重要的字段** — 同时影响检索召回和 LLM Prompt。

- 表的 `description`: 写清楚业务含义、每行代表什么
- 列的 `description`: 列名/display_name 看不出含义时必填（如 `channel_code` 需要写明格式和示例值）
- `tags`: 补充同义词/别名，只影响检索
- `enum_values`: 明确的枚举映射，检索和 Prompt 都用
- `query_tips`: 常用过滤条件提示
- `relations`: JOIN 提示 + 关联表补全加分
- `common_queries`: question 影响表级召回，sql 作为 Few-shot 示例

### 变更感知

| 变更 | 操作 |
|------|------|
| 改 YAML（描述/标签/术语/示例/枚举） | 重启即可（hash 自动检测） |
| Doris 加表/加列 | 重启即可 |
| 改检索逻辑/算法 | 重新部署 |
| 强制重建 | `python main.py --rebuild` |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Embedding | BGE-M3 | 中文最佳，一次 encode 同时出 Dense + Sparse |
| 向量数据库 | Milvus 2.5 | 原生 Dense + Sparse 混合检索，内置 RRF |
| Reranker | BGE-Reranker-v2-M3 | 交叉编码器精排，与 BGE-M3 配套 |
| 融合算法 | RRF (k=60) | Milvus 内置，不同量纲天然适配 |
| Few-shot | MMR (λ=0.7) | 平衡相关性和多样性 |
| LLM | DeepSeek (deepseek-chat) | 中文 SQL 生成强，性价比高 |
| Schema 格式 | CREATE TABLE DDL | LLM 最熟悉，准确率最高 |
