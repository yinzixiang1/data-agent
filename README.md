# Data Agent — NL2SQL RAG 检索体系

基于 BGE-M3 Dense + Sparse 混合检索 + Milvus 向量数据库的 NL2SQL 系统。

用户输入自然语言问题 → RAG 检索相关表和列 → 组装 Prompt → DeepSeek LLM 生成 SQL。

## 架构

```
用户提问
  │
  ├─❶ 业务术语解析
  │    glossary YAML → enriched_query + business_context
  │
  ├─❷ Schema 混合检索（Milvus）
  │    BGE-M3 一次 encode → Dense + Sparse
  │    ├── 表级: Milvus hybrid_search (Dense + Sparse → RRF)
  │    ├── 列级: Milvus hybrid_search → 反推表
  │    └── 合并去重 → top 10 候选
  │
  ├─❸ Reranker 精排
  │    BGE-Reranker-v2-M3 交叉编码 → top 5
  │
  ├─❹ Few-shot 示例检索
  │    Milvus Dense 检索 + 表重叠度加权 + MMR 多样性 → top 3
  │
  ├─❺ Prompt 组装
  │    DDL Schema + 列注释 + JOIN 提示 + Few-shot + 业务上下文
  │
  └─❻ SQL 生成
       DeepSeek LLM → 可执行的 Doris SQL
```

## 项目结构

```
data-agen/
├── main.py                           # 交互式入口（RAG + LLM 生成 SQL）
├── .env                              # 环境变量（Doris/DeepSeek/Milvus 配置）
├── pyproject.toml                    # 依赖管理
├── docker-compose-milvus.yaml        # Milvus + Attu 部署
│
├── src/retrieval/                    # RAG 检索核心模块
│   ├── config.py                    # 集中配置
│   ├── embedding.py                 # BGE-M3 封装（Dense + Sparse 混合编码）
│   ├── milvus_store.py              # Milvus 向量存储（Dense + Sparse 混合索引）
│   ├── schema_loader.py             # Doris DDL + 语义层 YAML 加载合并
│   ├── document_builder.py          # Schema → 表级/列级检索文档
│   ├── hybrid_searcher.py           # Milvus 混合检索 + 表/列联合
│   ├── reranker.py                  # BGE-Reranker 精排
│   ├── fewshot_selector.py          # Few-shot 示例检索（Milvus + MMR）
│   ├── glossary_resolver.py         # 业务术语解析
│   ├── schema_formatter.py          # 检索结果 → DDL Prompt 文本
│   ├── index_manager.py             # 索引生命周期管理（Milvus 版）
│   └── retriever.py                 # 对外统一入口
│
├── semantic_layer/                   # 语义层（人工维护的业务知识）
│   ├── tables/
│   │   ├── pmt_account.yaml         # 商户账户表
│   │   └── pmt_finance_payout.yaml  # 账户代付表
│   └── glossary/
│       └── glossary.yaml            # 业务术语表（8 条）
│
├── index_store/                      # 本地元数据（table_docs.json 等）
│
├── milvus/                           # Milvus 数据持久化目录（Docker 挂载）
│
├── tests/
│   └── test_retrieval_offline.py    # 离线测试
│
└── docs/
    ├── RAG_检索体系设计方案.md
    ├── 向量化文档格式示例_pmt_account.md
    └── 数据交付规范_给数仓同学.md
```

## 快速开始

### 1. 启动 Milvus

```bash
docker compose -f docker-compose-milvus.yaml up -d
```

启动后的服务：

| 服务 | 端口 | 地址 |
|------|------|------|
| Milvus API | 19530 | localhost:19530 |
| Attu (Milvus GUI) | 8000 | http://localhost:8000 |
| MinIO Console | 9001 | http://localhost:9001 |
| Milvus Health | 9091 | http://localhost:9091/healthz |

### 2. 安装依赖

```bash
pip install -e ".[dev]"
pip install pymilvus openai
```

核心依赖：

| 包 | 用途 |
|---|------|
| FlagEmbedding | BGE-M3 Dense + Sparse 编码 |
| pymilvus | Milvus 向量数据库客户端 |
| sentence-transformers | BGE-Reranker 精排 |
| openai | DeepSeek LLM 调用 |
| sqlalchemy + pymysql | Doris 连接 |
| pyyaml | 语义层 YAML 解析 |

> BGE-M3 模型约 2GB，首次运行会自动下载到 `~/.cache/huggingface/`。

### 3. 配置环境变量

编辑 `.env`：

```bash
# Doris 连接
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

# 模型（可选，有默认值）
EMBEDDING_MODEL="BAAI/bge-m3"
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
ENABLE_RERANKER="true"
```

### 4. 准备语义层

在 `semantic_layer/tables/` 下为每张表创建 YAML 文件，参考已有的 `pmt_account.yaml`。

在 `semantic_layer/glossary/glossary.yaml` 中添加业务术语。

> 详细格式见 `docs/数据交付规范_给数仓同学.md`。

### 5. 运行

```bash
python main.py             # 自动检测 Doris，连不上走离线模式
python main.py --offline   # 强制离线（仅从 YAML 加载 Schema）
```

交互命令：

| 命令 | 作用 |
|------|------|
| 直接输入问题 | RAG 检索 → LLM 生成 SQL |
| `/prompt` | 开启/关闭完整 Prompt 显示 |
| `/debug` | 开启/关闭详细日志 |
| `/quit` | 退出 |

示例：

```
> 活跃商户有多少

[术语匹配] 活跃商户

[命中表] pmt_account, pmt_finance_payout

[Prompt] (4518 字符)
------------------------------------------------------------
## 可用数据表
CREATE TABLE `dwd_banking`.`pmt_account` (
  ...
);
## 业务上下文
- 活跃商户 = account_status = 1 且 is_delete = 0 的账户
## 参考示例
...
------------------------------------------------------------

[生成 SQL]
SELECT COUNT(*) AS "活跃商户数量"
FROM dwd_banking.pmt_account
WHERE account_status = 1 AND is_delete = 0
```

### 6. 编程调用

```python
from src.retrieval.retriever import SchemaRetriever

retriever = SchemaRetriever()
retriever.initialize()  # 启动时调用一次

result = retriever.retrieve("目前有多少活跃商户")

print(result.relevant_tables)     # 命中的表
print(result.relevant_examples)   # Few-shot 示例
print(result.business_context)    # 业务术语解释
print(result.prompt_text)         # 组装好的完整 Prompt
print(result.matched_terms)       # 匹配的术语
```

## 检索流程详解

### 离线阶段（启动时执行一次）

```
Doris DDL + 语义层 YAML
  → 合并为完整 Schema dict
  → 构建表级/列级文档
  → BGE-M3 encode（一次调用同时输出 Dense + Sparse）
  → 写入 Milvus（3 个 Collection: table / column / fewshot）
  → 元数据保存到 index_store/
  → 下次启动如果 Schema 未变则直接使用 Milvus 中的索引
```

### 在线阶段（每次用户提问）

| 步骤 | 模块 | 说明 |
|------|------|------|
| ❶ 业务术语解析 | `glossary_resolver` | "活跃商户" → `account_status=1 AND is_delete=0` |
| ❷ 混合检索 | `hybrid_searcher` | Milvus hybrid_search: Dense + Sparse → RRF 融合 |
| ❸ Reranker 精排 | `reranker` | 交叉编码器从 top 10 精排到 top 5 |
| ❹ Few-shot 检索 | `fewshot_selector` | Milvus Dense 检索 + 表重叠度加权 + MMR 多样性 |
| ❺ Prompt 组装 | `schema_formatter` | CREATE TABLE DDL + 列注释 + JOIN 提示 + Few-shot |
| ❻ SQL 生成 | `main.py` | DeepSeek LLM 生成可执行 SQL |

## Milvus 向量存储

### Collections（数据库: nl2sql）

| Collection | 数据量 | 字段 | 用途 |
|------------|--------|------|------|
| nl2sql_table | 表数 | id, dense_vec(1024), sparse_vec, doc_json | 表级混合检索 |
| nl2sql_column | 列数 | id, dense_vec(1024), sparse_vec, doc_json | 列级混合检索 → 反推表 |
| nl2sql_fewshot | 示例数 | id, dense_vec(1024), sparse_vec, doc_json | Few-shot 示例检索 |

每个 Collection 同时存储 Dense 向量（FLAT 索引）和 Sparse 向量（SPARSE_INVERTED_INDEX），检索时通过 Milvus 内置的 `hybrid_search` + `RRFRanker` 完成融合。

### 管理

- **Attu GUI**: http://localhost:8000 — 可视化查看 Collection、数据、执行搜索
- **Schema 变更检测**: 基于 SHA-256 hash，YAML 变更后重启自动重建索引

## 语义层维护

### 新增一张表

1. 在 `semantic_layer/tables/` 下创建 `your_table.yaml`（参考 `pmt_account.yaml`）
2. 重启服务（或调 `retriever.initialize(force_rebuild=True)`）
3. 系统自动检测 Schema 变更 → 重建 Milvus 索引

### 修改表描述 / 加术语 / 加示例 SQL

直接编辑对应 YAML → 重启服务。不需要改代码。

### 变更感知机制

| 变更 | 感知方式 | 操作 |
|------|---------|------|
| Doris 加表/加列 | 启动时自动 DESCRIBE | 重启即可 |
| 改 YAML（描述/标签/术语/示例） | Schema hash 对比 | 重启即可 |
| 改检索逻辑/算法 | 代码变更 | 需重新部署 |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Embedding | BGE-M3 | 中文最佳；一次 encode 同时出 Dense + Sparse |
| 向量数据库 | Milvus 2.5 | 原生支持 Dense + Sparse 混合检索；内置 RRF 融合 |
| Reranker | BGE-Reranker-v2-M3 | 与 BGE-M3 配套；交叉编码器精排 |
| 融合算法 | RRF (k=60) | Milvus 内置；无需调参；不同量纲天然适配 |
| Few-shot 选择 | MMR (lambda=0.7) | 平衡相关性和多样性 |
| LLM | DeepSeek (deepseek-chat) | 中文 SQL 生成能力强；性价比高 |
| Schema 格式 | CREATE TABLE DDL | LLM 最熟悉；准确率最高 |
| GUI | Attu | Milvus 官方可视化工具 |

### 为什么用 BGE-M3 Learned Sparse 而不是传统 BM25

| 维度 | 传统 BM25 | BGE-M3 Learned Sparse |
|------|----------|----------------------|
| 稀疏表示 | 词频统计 | 模型学到的词汇权重 |
| 同义词 | 无法处理 | 模型理解 "利润" ≈ "profit" |
| 中文分词 | 依赖 jieba | 模型 tokenizer 自带 |
| 编码次数 | 需额外处理 | 与 Dense 共享一次编码 |

## 相关文档

- [RAG 检索体系设计方案](docs/RAG_检索体系设计方案.md) — 完整架构设计
- [向量化文档格式示例](docs/向量化文档格式示例_pmt_account.md) — 以 pmt_account 为例的各层格式
- [数据交付规范](docs/数据交付规范_给数仓同学.md) — 给数仓/业务同学的填写指南
