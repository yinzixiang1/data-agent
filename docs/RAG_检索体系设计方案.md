# Data Agent RAG 检索体系设计方案

**基于 BGE-M3 Dense + Sparse 混合检索的 NL2SQL Schema 检索系统**

| 项目 | 信息 |
|------|------|
| 版本 | v1.0 |
| 日期 | 2026-04-27 |
| 状态 | Draft |
| 依赖文档 | Data_Agent_技术设计文档_v5_自研方案.md §12.2 |

---

## 目录

- [1. 设计目标](#1-设计目标)
- [2. 与通用 RAG 的核心区别](#2-与通用-rag-的核心区别)
- [3. 全局架构](#3-全局架构)
  - [3.1 离线阶段（Index Building）](#31-离线阶段index-building)
  - [3.2 在线阶段（Query Retrieval）](#32-在线阶段query-retrieval)
- [4. 离线阶段详解](#4-离线阶段详解)
  - [4.1 数据源加载](#41-数据源加载)
  - [4.2 文档构建策略](#42-文档构建策略)
  - [4.3 BGE-M3 混合编码](#43-bge-m3-混合编码)
  - [4.4 索引构建](#44-索引构建)
  - [4.5 索引持久化](#45-索引持久化)
- [5. 在线阶段详解](#5-在线阶段详解)
  - [5.1 业务术语解析](#51-业务术语解析)
  - [5.2 Schema 混合检索](#52-schema-混合检索)
  - [5.3 RRF 融合算法](#53-rrf-融合算法)
  - [5.4 Reranker 精排](#54-reranker-精排)
  - [5.5 Few-shot 示例检索](#55-few-shot-示例检索)
  - [5.6 Prompt 组装](#56-prompt-组装)
- [6. 对外接口设计](#6-对外接口设计)
- [7. 文件结构与模块职责](#7-文件结构与模块职责)
- [8. 技术选型](#8-技术选型)
- [9. 关键设计决策](#9-关键设计决策)
- [附录 A: BGE-M3 Dense vs Sparse vs 传统 BM25 对比](#附录-a-bge-m3-dense-vs-sparse-vs-传统-bm25-对比)
- [附录 B: 依赖清单](#附录-b-依赖清单)

---

## 1. 设计目标

将 Doris 的 Schema 元数据（表名、列定义、注释、关联关系、业务术语）构建为可检索的混合向量索引，使用户的自然语言提问能精准召回相关表和列信息，组装为高质量的 SQL 生成 Prompt。

**核心指标**：

| 指标 | 目标 |
|------|------|
| Top-5 表召回率 | >= 85% |
| Top-3 表精确率 | >= 75% |
| 检索延迟（P95） | < 200ms |
| 索引构建时间（100 张表） | < 60s |

---

## 2. 与通用 RAG 的核心区别

NL2SQL 的 RAG 与通用问答 RAG 有本质不同：

| 维度 | 通用问答 RAG | NL2SQL RAG |
|------|-------------|-----------|
| 检索对象 | 非结构化文档 | **结构化 Schema 元数据** |
| 分块策略 | 按段落/token 切分 | **按表/列粒度组织** |
| 检索目标 | 找到相关文本段落 | **找到正确的表和列** |
| 输出格式 | 文本片段 | **DDL + 列注释 + JOIN 提示** |
| Few-shot | 通常不需要 | **关键（+10-20% 准确率）** |
| 语义层 | 无 | **业务术语→计算口径映射** |

---

## 3. 全局架构

### 3.1 离线阶段（Index Building）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        离线阶段（Offline）                           │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Doris   │───→│ Schema 加载  │───→│  文档构建    │               │
│  │  DDL     │    │              │    │  ├─ 表级文档  │               │
│  └──────────┘    │  合并语义层  │    │  └─ 列级文档  │               │
│  ┌──────────┐    │  信息        │    └──────┬───────┘               │
│  │ 语义层   │───→│              │           │                       │
│  │ YAML     │    └──────────────┘           ▼                       │
│  └──────────┘                      ┌────────────────┐               │
│                                    │ BGE-M3 Encode  │               │
│                                    │ dense + sparse  │               │
│                                    └───────┬────────┘               │
│                                            │                        │
│                              ┌─────────────┼─────────────┐         │
│                              ▼             ▼             ▼         │
│                     ┌──────────────┐ ┌───────────┐ ┌───────────┐   │
│                     │ FAISS Dense  │ │  Sparse   │ │ Few-shot  │   │
│                     │ 表级 + 列级  │ │  倒排索引  │ │ 示例索引  │   │
│                     └──────────────┘ └───────────┘ └───────────┘   │
│                                                                     │
│  持久化 → index_store/ 目录                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 在线阶段（Query Retrieval）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        在线阶段（Online）                            │
│                                                                     │
│  用户提问: "上个月华东区各品类的退货率"                                 │
│      │                                                              │
│      ▼                                                              │
│  ❶ 业务术语解析                                                      │
│      "退货率" → 退货订单数/总订单数                                    │
│      "华东区" → region IN ('上海','江苏',...)                         │
│      → enriched_query + business_context                            │
│      │                                                              │
│      ▼                                                              │
│  ❷ Schema 混合检索                                                   │
│      BGE-M3 Encode(enriched_query) → query_dense + query_sparse     │
│      │                                                              │
│      ├── 表级检索                                                    │
│      │    ├── Dense → FAISS → top 20                                │
│      │    ├── Sparse → 倒排索引 → top 20                            │
│      │    └── RRF 融合 → top 10                                     │
│      │                                                              │
│      ├── 列级检索                                                    │
│      │    ├── Dense → FAISS → top 20                                │
│      │    ├── Sparse → 倒排索引 → top 20                            │
│      │    ├── RRF 融合 → 命中列                                     │
│      │    └── 反推表集合                                             │
│      │                                                              │
│      └── 合并：表级结果 ∪ 列级反推表 → 去重排序 → top 10             │
│      │                                                              │
│      ▼                                                              │
│  ❸ Reranker 精排                                                    │
│      BGE-Reranker-v2-M3 交叉编码                                    │
│      top 10 → 精排 → top 5                                         │
│      │                                                              │
│      ▼                                                              │
│  ❹ Few-shot 示例检索                                                │
│      语义相似度 + 表重叠度加权 + MMR 多样性                            │
│      → top 3 示例                                                   │
│      │                                                              │
│      ▼                                                              │
│  ❺ Prompt 组装                                                      │
│      DDL Schema + 列注释 + JOIN 提示                                 │
│      + Few-shot 示例                                                │
│      + 业务上下文                                                    │
│      → 交给 SQL 生成 LLM                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 离线阶段详解

### 4.1 数据源加载

从两个来源加载 Schema 信息，合并后得到每张表的完整元数据。

#### 4.1.1 Doris 元数据

```
Doris FE
  ├── SHOW TABLES                    → 表名列表
  ├── DESCRIBE `table`               → 列名、类型、是否可空、Key 类型
  ├── SHOW CREATE TABLE `table`      → 原始 DDL（含引擎、分区等信息）
  └── SELECT * FROM `table` LIMIT 3  → 样例数据（帮助理解字段含义）
```

输出结构：

```python
{
    "table_name": "dwd_order_fact",
    "columns": [
        {"name": "order_id", "type": "BIGINT", "nullable": "NO", "key": "UNI"},
        {"name": "amount", "type": "DECIMAL(18,2)", "nullable": "YES", "key": ""},
        ...
    ],
    "create_ddl": "CREATE TABLE ...",
    "sample_rows": [["10001", "299.00", ...], ...],
}
```

#### 4.1.2 语义层 YAML（人工维护）

```yaml
# semantic_layer/tables/dwd_order_fact.yaml
table:
  name: dwd_order_fact
  display_name: 订单事实表
  description: 记录所有订单的明细数据，包含金额、状态、时间等
  tags: [交易, 订单, 电商]

columns:
  - name: order_id
    display_name: 订单ID
    type: BIGINT
    description: 唯一订单标识
  - name: amount
    display_name: 订单金额
    type: DECIMAL(18,2)
    description: 订单实付金额（人民币）
  - name: status
    display_name: 订单状态
    type: VARCHAR(20)
    description: 订单当前状态
    enum_values: [pending, paid, refunded, cancelled]
  - name: create_date
    display_name: 创建日期
    type: DATE
    description: 订单创建日期
    is_partition: true
  - name: region
    display_name: 区域
    type: VARCHAR(50)
    description: 下单用户所在区域

relations:
  - column: product_id
    target_table: dim_product
    target_column: id
    join_type: LEFT JOIN
  - column: region_code
    target_table: dim_region
    target_column: code
    join_type: LEFT JOIN

common_queries:
  - question: 上个月的销售总额
    sql: |
      SELECT SUM(amount) AS "销售总额"
      FROM dwd_order_fact
      WHERE create_date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01')
        AND create_date < DATE_FORMAT(CURDATE(), '%Y-%m-01')
      LIMIT 1
    tables: [dwd_order_fact]
    difficulty: easy
  - question: 各区域的订单量分布
    sql: |
      SELECT region AS "区域", COUNT(*) AS "订单量"
      FROM dwd_order_fact
      WHERE create_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
      GROUP BY region
      ORDER BY COUNT(*) DESC
      LIMIT 100
    tables: [dwd_order_fact]
    difficulty: easy
```

业务术语表：

```yaml
# semantic_layer/glossary/glossary.yaml
glossary:
  - term: 退货率
    definition: 退货订单数 / 总订单数
    related_columns: [dwd_order_fact.status]
    sql_hint: "COUNT(CASE WHEN status = 'refunded' THEN 1 END) / COUNT(*)"

  - term: 华东区
    definition: 上海、江苏、浙江、安徽、福建、山东、江西
    related_columns: [dwd_order_fact.region]
    sql_hint: "region IN ('上海','江苏','浙江','安徽','福建','山东','江西')"

  - term: GMV
    definition: 成交总额（Gross Merchandise Volume），包含未付款订单
    related_columns: [dwd_order_fact.amount]
    sql_hint: "SUM(amount)"
```

#### 4.1.3 合并策略

```
Doris DDL（基础）
    │
    ▼
语义层 YAML（覆盖补充）
    ├── 有 display_name → 覆盖
    ├── 有 description → 覆盖
    ├── 有 tags → 新增
    ├── 有 enum_values → 新增
    ├── 有 relations → 新增
    ├── 有 common_queries → 新增
    └── 列级信息 → 按 column.name 匹配合并
    │
    ▼
完整 Schema dict（每张表一个）
```

### 4.2 文档构建策略

将每张表的 Schema dict 转换为可被 BGE-M3 编码的文本文档。采用**双粒度**策略：

#### 4.2.1 表级文档

每张表构建一个检索文档，内容涵盖表的全貌：

```
表名: dwd_order_fact
中文名: 订单事实表
描述: 记录所有订单的明细数据，包含金额、状态、时间等
业务标签: 交易, 订单, 电商
主要字段: order_id(订单ID), user_id(用户ID), amount(订单金额),
          status(订单状态), region(区域), create_date(创建日期)
常见问题: 上个月的销售总额是多少
常见问题: 各区域的订单量分布
关联: dim_product ON dwd_order_fact.product_id = dim_product.id
关联: dim_region ON dwd_order_fact.region_code = dim_region.code
```

**设计考量**：文档内容直接决定"什么样的问题能匹配到这张表"。把常见问题、业务标签写入文档，可以让语义上不直接相关的提问也能命中（如用户问"营收"→ 标签"交易"→ 命中订单表）。

#### 4.2.2 列级文档

每列构建一个独立文档，用于精确匹配指标/维度：

```
表: dwd_order_fact(订单事实表)
列名: refund_rate
中文名: 退货率
类型: DECIMAL(10,4)
描述: 退货订单数占总订单数的比例
业务逻辑: refund_count / total_count
```

**为什么需要列级文档**：

| 用户提问 | 表级检索 | 列级检索 | 更准的 |
|---------|---------|---------|--------|
| "查一下订单数据" | 命中 `dwd_order_fact` | 命中太多列 | 表级 |
| "利润率是多少" | 可能找不到表 | 精确命中 `profit_margin` 列 → 反推表 | **列级** |
| "华东区的 GMV" | 可能命中 | 命中 `region` 列 + `gmv` 列 | **联合** |

两种粒度互补：表级解决宽泛提问，列级解决指标精确提问。

### 4.3 BGE-M3 混合编码

使用 BGE-M3 **一次 encode 同时输出** Dense 和 Sparse 两种向量：

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

output = model.encode(
    sentences,
    return_dense=True,      # 1024-dim 稠密向量
    return_sparse=True,     # 词汇级稀疏权重
    return_colbert_vecs=False,
)

dense_vecs = output['dense_vecs']       # np.ndarray (N, 1024)
sparse_vecs = output['lexical_weights'] # list[dict{token_id: weight}]
```

**为什么不用传统 BM25**：

| 维度 | 传统 BM25 (rank-bm25) | BGE-M3 Learned Sparse |
|------|----------------------|----------------------|
| 稀疏表示 | 词频统计（TF-IDF） | **模型学到的词汇权重** |
| 同义词 | 无法处理 | 模型理解 "利润" ≈ "profit" |
| 中文分词 | 依赖 jieba 等外部分词 | **模型 tokenizer 自带** |
| 编码次数 | 需额外分词+统计 | **与 Dense 共享一次编码** |
| 额外依赖 | rank-bm25 + jieba | 无 |
| 精度 | 中 | **高** |

### 4.4 索引构建

共构建 **5 个索引**：

```
表级文档 × N 张表
    │
    BGE-M3.encode(return_dense=True, return_sparse=True)
    │
    ├── dense_vecs (N, 1024) → FAISS IndexFlatIP    [① 表级 Dense 索引]
    └── lexical_weights (N 个 dict) → 倒排索引       [② 表级 Sparse 索引]

列级文档 × M 列
    │
    BGE-M3.encode(return_dense=True, return_sparse=True)
    │
    ├── dense_vecs (M, 1024) → FAISS IndexFlatIP    [③ 列级 Dense 索引]
    └── lexical_weights (M 个 dict) → 倒排索引       [④ 列级 Sparse 索引]

Few-shot 示例 × K 条
    │
    BGE-M3.encode(return_dense=True)
    │
    └── dense_vecs (K, 1024) → FAISS IndexFlatIP    [⑤ Few-shot Dense 索引]
```

#### Dense 索引（FAISS）

使用 `IndexFlatIP`（内积 = 余弦相似度，向量已归一化）：

```python
import faiss
import numpy as np

embeddings = np.array(dense_vecs, dtype=np.float32)
index = faiss.IndexFlatIP(embeddings.shape[1])  # 1024
index.add(embeddings)
```

#### Sparse 倒排索引

BGE-M3 的 `lexical_weights` 输出为 `dict{token_id: weight}`，构建为倒排索引：

```python
# 结构: {token_id: [(doc_idx, weight), (doc_idx, weight), ...]}
inverted_index = {}

for doc_idx, weights in enumerate(lexical_weights):
    for token_id, weight in weights.items():
        if token_id not in inverted_index:
            inverted_index[token_id] = []
        inverted_index[token_id].append((doc_idx, weight))
```

查询时：遍历 query 的 sparse weights，从倒排索引中取出候选文档，累加得分：

```python
def sparse_search(query_weights: dict, top_k: int) -> list[tuple[int, float]]:
    scores = {}
    for token_id, q_weight in query_weights.items():
        if token_id in inverted_index:
            for doc_idx, d_weight in inverted_index[token_id]:
                scores[doc_idx] = scores.get(doc_idx, 0) + q_weight * d_weight
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]
```

### 4.5 索引持久化

```
index_store/
├── table_dense.faiss              # FAISS 表级 Dense 索引
├── table_sparse.json              # 表级 Sparse 倒排索引
├── table_docs.json                # 表级文档列表
├── column_dense.faiss             # FAISS 列级 Dense 索引
├── column_sparse.json             # 列级 Sparse 倒排索引
├── column_docs.json               # 列级文档列表
├── fewshot_dense.faiss            # FAISS Few-shot Dense 索引
├── fewshot_examples.json          # Few-shot 示例列表
├── table_schemas.json             # table_name → 完整 Schema dict
└── schema_hash.txt                # Schema 内容 hash（增量判断）
```

启动流程：

```
启动 → 加载 Doris DDL + 语义层 YAML
     → 计算 Schema 内容 hash
     → 对比 schema_hash.txt
         ├── 一致 → 从磁盘加载已有索引（秒级）
         └── 不一致 → 全量重建索引 → 写入磁盘 → 更新 hash
```

---

## 5. 在线阶段详解

### 5.1 业务术语解析

从语义层 glossary 中匹配用户问题里的业务术语，输出两部分：

- **enriched_query**：原始问题 + 术语展开词，用于检索增强
- **business_context**：术语→口径映射，注入 Prompt

```
输入: "上个月华东区各品类的退货率"

语义层 glossary 匹配:
  "退货率" → 计算口径: "退货订单数 / 总订单数"
             → sql_hint: "COUNT(CASE WHEN status='refunded' THEN 1 END) / COUNT(*)"
  "华东区" → 过滤条件: "region IN ('上海','江苏','浙江','安徽','福建','山东','江西')"
  "品类"   → 关联维度: "dim_product.category"

输出:
  enriched_query: "上个月华东区各品类的退货率 退货订单数 总订单数 region dim_product category"
  business_context: |
    - 退货率 = 退货订单数 / 总订单数, SQL: COUNT(CASE WHEN status='refunded' THEN 1 END) / COUNT(*)
    - 华东区 = region IN ('上海','江苏','浙江','安徽','福建','山东','江西')
    - 品类 → 关联 dim_product.category
```

匹配策略：遍历 glossary 中每个 term，检查是否出现在 user_query 中（精确子串匹配，后续可升级为模糊匹配）。

### 5.2 Schema 混合检索

对 enriched_query 进行 Dense + Sparse 双路检索，分别在表级和列级执行：

```
enriched_query
    │
    BGE-M3.encode(return_dense=True, return_sparse=True)
    │
    ├── query_dense (1, 1024)
    └── query_sparse (dict{token_id: weight})

=== 表级检索 ===
query_dense  → 表级 FAISS.search(top_k=20)  → dense_results
query_sparse → 表级倒排索引.search(top_k=20) → sparse_results
RRF(dense_results, sparse_results)           → table_candidates (top 10)

=== 列级检索 ===
query_dense  → 列级 FAISS.search(top_k=20)  → dense_results
query_sparse → 列级倒排索引.search(top_k=20) → sparse_results
RRF(dense_results, sparse_results)           → column_candidates
    → 反推表集合: column_tables = {命中列所属的表}

=== 合并 ===
table_candidates ∪ column_tables → 去重 → 按 RRF 总分排序 → top 10 候选
    （列级反推的表在合并时给予 bonus 分，解决"表级检索没排前面
      但用户确实在问那个列"的问题）
```

### 5.3 RRF 融合算法

Reciprocal Rank Fusion（RRF）将多路检索结果融合为统一排序：

```
公式: RRF_score(d) = Σ  1 / (k + rank_i(d))

参数: k = 60（论文推荐默认值，防止排名靠后的文档被过度惩罚）
```

示例：

```
Dense 结果: [table_A(rank1), table_B(rank2), table_C(rank3)]
Sparse 结果: [table_C(rank1), table_A(rank2), table_D(rank3)]

table_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
table_B: 1/(60+2)                                  = 0.01613
table_C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
table_D:            1/(60+3)                        = 0.01587

最终排序: table_A > table_C > table_B > table_D
```

列级反推表的 bonus：

```python
# 列级命中的表，在 RRF 合并时给额外加分
if table_name in column_reverse_tables:
    rrf_score += 0.01  # 小 bonus，不覆盖主排序
```

### 5.4 Reranker 精排

初始检索（Dense + Sparse + RRF）是"召回"阶段，追求不漏。Reranker 是"精排"阶段，用交叉编码器精细打分。

```
输入: top 10 候选表
模型: BGE-Reranker-v2-M3（交叉编码器）

对每个候选:
    pair = (user_query, 表级文档文本)
    score = CrossEncoder.predict(pair)

按 score 重排序 → 输出 top 5
```

**为什么需要 Reranker**：

| 对比 | Bi-Encoder (BGE-M3) | Cross-Encoder (Reranker) |
|------|---------------------|------------------------|
| 编码方式 | query 和 doc 分别编码 | query 和 doc **拼接后联合编码** |
| 速度 | 快（可预计算 doc 向量） | 慢（需在线计算每个 pair） |
| 精度 | 中 | **高** |
| 用法 | 召回阶段（从全量中选候选） | 精排阶段（从候选中选最优） |

典型提升：

| 指标 | 无 Reranker | 有 Reranker | 提升 |
|------|-----------|-----------|------|
| Top-5 表召回率 | ~75% | ~88% | +13% |
| Top-3 表精确率 | ~60% | ~78% | +18% |
| 额外延迟 | — | +50-100ms | 可接受 |

### 5.5 Few-shot 示例检索

Few-shot 示例是 NL2SQL 准确率提升最明显的单一因素（通常 +10-20%）。

#### 示例来源

按优先级排序：

1. 语义层 YAML 中的 `common_queries`（人工编写，质量最高）
2. 用户确认正确的历史 query-SQL 对（数据飞轮积累）
3. 人工标注的评测题库

#### 检索策略

```
user_query → BGE-M3.encode(return_dense=True)
           → Few-shot FAISS 检索 → 语义相似度 top 10

加权调整:
  对每条候选示例:
    表重叠度 = |示例涉及的表 ∩ Schema 检索命中的 5 张表|
    最终分 = 语义相似度 + 0.1 × 表重叠度

MMR 多样性选择（从 top 10 中选 3 条）:
  第 1 条: 选最终分最高的
  第 2-3 条: 兼顾相关性和多样性
    mmr_score = λ × relevance - (1-λ) × max_similarity_to_selected
    λ = 0.7（偏向相关性）
```

**为什么需要 MMR 多样性**：避免选出的 3 条示例全是 "查XX的销售额" 这类同质化示例，让示例覆盖不同 SQL 模式（聚合、JOIN、子查询、时间范围等）。

### 5.6 Prompt 组装

将检索结果格式化为 SQL 生成 Prompt。Schema 使用 **CREATE TABLE DDL** 格式（LLM 训练数据中大量出现，最熟悉）。

#### 输出格式

```sql
## 可用数据表

-- 订单事实表：记录所有订单的明细数据
CREATE TABLE `dwd_order_fact` (
  `order_id` BIGINT            -- 订单ID
  `user_id` BIGINT             -- 用户ID
  `amount` DECIMAL(18,2)       -- 订单金额（人民币）
  `status` VARCHAR(20)         -- 订单状态 [可选值: pending, paid, refunded, cancelled]
  `region` VARCHAR(50)         -- 区域
  `create_date` DATE           -- 创建日期（分区键，查询时建议带上过滤条件）
);
-- JOIN 提示: dwd_order_fact.product_id = dim_product.id
-- JOIN 提示: dwd_order_fact.region_code = dim_region.code

-- 商品维度表
CREATE TABLE `dim_product` (
  `id` BIGINT                  -- 商品ID
  `name` VARCHAR(200)          -- 商品名称
  `category` VARCHAR(100)      -- 品类
);

## 业务上下文
- 退货率 = 退货订单数 / 总订单数, SQL: COUNT(CASE WHEN status='refunded' THEN 1 END) / COUNT(*)
- 华东区 = region IN ('上海','江苏','浙江','安徽','福建','山东','江西')

## 参考示例

### 示例 1
问题：上个月的销售总额
SQL：
SELECT SUM(amount) AS "销售总额"
FROM dwd_order_fact
WHERE create_date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01')
  AND create_date < DATE_FORMAT(CURDATE(), '%Y-%m-01')
LIMIT 1

### 示例 2
问题：各区域的订单量分布
SQL：
SELECT region AS "区域", COUNT(*) AS "订单量"
FROM dwd_order_fact
WHERE create_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY region
ORDER BY COUNT(*) DESC
LIMIT 100
```

#### Schema 格式对比

| Prompt 中 Schema 格式 | 准确率 (基准测试) | Token 消耗 |
|----------------------|-----------------|-----------|
| 纯表名+列名列表 | ~65% | 最少 |
| Markdown 表格 | ~72% | 中等 |
| **CREATE TABLE DDL** | **~78%** | 中等 |
| **DDL + 列注释 + JOIN 提示** | **~82%** | 稍多（推荐） |

---

## 6. 对外接口设计

`retriever.py` 暴露统一入口，供 LangGraph `schema_retriever` Node 调用：

```python
from pydantic import BaseModel

class RetrievalResult(BaseModel):
    """检索结果"""
    relevant_tables: list[dict]     # top 5 表的完整 Schema dict
    relevant_examples: list[dict]   # top 3 Few-shot 示例 {question, sql, tables}
    business_context: str           # 业务术语解释文本
    prompt_text: str                # 组装好的完整 Prompt 文本（DDL + Few-shot + Context）

class SchemaRetriever:
    """RAG 检索体系统一入口"""

    def __init__(self, index_dir: str, semantic_layer_dir: str):
        """
        Args:
            index_dir: 索引持久化目录
            semantic_layer_dir: 语义层 YAML 目录
        """
        ...

    async def initialize(self, doris_connection_string: str):
        """启动初始化：加载或重建索引"""
        ...

    async def retrieve(self, user_query: str) -> RetrievalResult:
        """
        完整检索流程：
        1. 业务术语解析
        2. Schema 混合检索（Dense + Sparse + RRF）
        3. 表级 + 列级合并
        4. Reranker 精排
        5. Few-shot 示例检索（语义相似 + 表重叠 + MMR）
        6. Prompt 组装（DDL 格式）
        """
        ...
```

---

## 7. 文件结构与模块职责

```
src/retrieval/
├── __init__.py
├── schema_loader.py          # 数据源加载
│                              # - Doris DDL 加载（SHOW TABLES / DESCRIBE / SHOW CREATE TABLE）
│                              # - 语义层 YAML 加载（表/列/glossary）
│                              # - 两者合并为完整 Schema dict
│
├── document_builder.py        # 文档构建
│                              # - Schema dict → 表级检索文档（text）
│                              # - Schema dict → 列级检索文档（text）
│                              # - 控制文档内容：什么信息写入文档、什么不写
│
├── embedding.py               # BGE-M3 封装
│                              # - 单例模式，全局复用一个模型实例
│                              # - encode() 一次调用返回 dense + sparse
│                              # - 统一 batch_size / max_length / use_fp16 配置
│
├── dense_index.py             # FAISS Dense 索引
│                              # - 构建：np.ndarray → IndexFlatIP
│                              # - 检索：query_vec → top_k (ids, scores)
│                              # - 持久化：faiss.write_index / faiss.read_index
│
├── sparse_index.py            # Sparse 倒排索引
│                              # - 构建：list[dict] → inverted_index
│                              # - 检索：query_weights → top_k (ids, scores)
│                              # - 持久化：JSON 序列化 / 反序列化
│
├── hybrid_searcher.py         # 混合检索
│                              # - 串联 dense_index + sparse_index
│                              # - RRF 融合算法
│                              # - 表级检索 + 列级检索 + 合并去重
│
├── reranker.py                # Reranker 精排
│                              # - BGE-Reranker-v2-M3 CrossEncoder
│                              # - (query, doc) pair → rerank_score
│                              # - top N → 精排 → top K
│
├── fewshot_selector.py        # Few-shot 示例检索
│                              # - 示例库加载（YAML common_queries + 历史对）
│                              # - Dense 相似度检索
│                              # - 表重叠度加权
│                              # - MMR 多样性选择
│
├── glossary_resolver.py       # 业务术语解析
│                              # - glossary YAML 加载
│                              # - user_query 中术语匹配
│                              # - 输出 enriched_query + business_context
│
├── schema_formatter.py        # Prompt 格式化
│                              # - Schema dict → CREATE TABLE DDL 文本
│                              # - 列注释、JOIN 提示、分区键提示
│                              # - Few-shot 示例格式化
│                              # - 业务上下文拼接
│                              # - 最终输出完整 Prompt 文本
│
├── index_manager.py           # 索引生命周期管理
│                              # - 启动初始化：判断是否需要重建
│                              # - Schema hash 计算与对比
│                              # - 全量构建流程编排
│                              # - 索引加载 / 持久化
│
└── retriever.py               # 对外统一入口
                               # - 组合以上所有模块
                               # - 暴露 initialize() + retrieve() 接口
                               # - 供 LangGraph schema_retriever Node 调用
```

### 模块调用关系

```
retriever.py（对外唯一入口）
    │
    ├── glossary_resolver.py          ← glossary YAML
    │
    ├── hybrid_searcher.py
    │     ├── embedding.py            ← BGE-M3（全局单例）
    │     ├── dense_index.py          ← FAISS
    │     └── sparse_index.py         ← 倒排索引
    │
    ├── reranker.py                   ← BGE-Reranker-v2-M3
    │
    ├── fewshot_selector.py
    │     └── embedding.py            ← 复用同一个 BGE-M3
    │
    └── schema_formatter.py

index_manager.py（启动时调用）
    ├── schema_loader.py              ← Doris + YAML
    ├── document_builder.py
    ├── embedding.py                  ← BGE-M3
    ├── dense_index.py                ← 构建 + 写磁盘
    └── sparse_index.py              ← 构建 + 写磁盘
```

---

## 8. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Embedding 模型 | **BGE-M3** (BAAI/bge-m3) | 中文最佳；原生 Dense + Sparse 混合编码；一次推理两种向量 |
| Dense 索引 | **FAISS** (IndexFlatIP) | 零外部依赖；精确检索；100 张表规模内存完全够用 |
| Sparse 索引 | **自建倒排索引** (dict) | 轻量；与 BGE-M3 lexical_weights 直接对接；无需额外依赖 |
| Reranker | **BGE-Reranker-v2-M3** | 与 BGE-M3 配套；中文效果好；交叉编码器精排 |
| 融合算法 | **RRF** (k=60) | 无需调参；对不同量纲的分数天然适配 |
| Few-shot 选择 | **MMR** (λ=0.7) | 平衡相关性和多样性 |
| Schema 格式 | **CREATE TABLE DDL + 注释** | LLM 最熟悉的格式；准确率最高 |
| 持久化 | **本地文件** (FAISS + JSON) | 简单可靠；启动加载秒级 |
| BGE-M3 SDK | **FlagEmbedding** >= 1.3 | 官方包；API 最完整 |

---

## 9. 关键设计决策

### 9.1 为什么选 BGE-M3 Learned Sparse 而不是传统 BM25

- 一次 encode 同时产出 Dense 和 Sparse，无需两个独立系统
- Learned Sparse 理解同义词（"利润" ≈ "profit"），BM25 只做字面匹配
- 不依赖外部中文分词（jieba），减少一个故障点
- 减少依赖：去掉 rank-bm25 和 jieba 两个包

### 9.2 为什么需要双粒度索引（表级 + 列级）

- 表级：解决 "查订单数据" 这类宽泛提问
- 列级：解决 "退货率" "利润率" 这类指标精确提问，命中列后反推表
- 单一粒度无法同时覆盖两类场景

### 9.3 为什么 Schema 用 DDL 格式注入 Prompt

- LLM 训练数据中 CREATE TABLE 语句大量出现，模型最熟悉
- 比 Markdown 表格准确率高约 10%
- 列注释和 JOIN 提示以 SQL 注释形式内联，LLM 自然理解

### 9.4 为什么 Few-shot 需要 MMR 多样性选择

- 纯相似度 top-3 容易选出同质化示例（都是简单聚合查询）
- MMR 保证示例覆盖不同 SQL 模式（聚合/JOIN/子查询/时间范围）
- 实际效果比纯 top-k 更稳定

### 9.5 Reranker 是否必要

- 是。Dense + Sparse 召回阶段追求不漏掉相关表，但排序不够精确
- Reranker 用交叉编码器做精细打分，Top-3 精确率提升 +18%
- 额外延迟 50-100ms，对 NL2SQL 场景（总延迟目标 < 15s）完全可接受

---

## 附录 A: BGE-M3 Dense vs Sparse vs 传统 BM25 对比

| 用户问法 | Dense（语义） | Sparse（学习稀疏） | BM25（词频） | 最佳策略 |
|---------|-------------|-------------------|-------------|---------|
| "帮我看看销售情况" | 强（理解语义） | 中 | 弱（无精确关键词） | Dense 主导 |
| "查 dwd_order_fact 表" | 弱（表名非语义） | **强**（精确匹配） | 强 | Sparse 主导 |
| "上个月华东区退货率" | 中 | 中 | 中 | **混合最优** |
| "利润率同比" | 中（列级命中） | 中 | 弱 | Dense + 列级 |
| "GMV 趋势" | 强（理解缩写） | 中 | 弱（GMV 可能未分词） | Dense 主导 |

**结论**：没有单一策略在所有场景都最优，Dense + Sparse 混合 + RRF 融合是最稳健的选择。

---

## 附录 B: 依赖清单

```
# 核心依赖
FlagEmbedding>=1.3              # BGE-M3 Dense + Sparse 编码
faiss-cpu>=1.9                  # Dense 向量索引
sentence-transformers>=3.0      # BGE-Reranker CrossEncoder
numpy>=1.26                     # 向量计算
pydantic>=2.10                  # 数据模型

# 数据源
sqlalchemy>=2.0                 # Doris 连接（已有）
pymysql>=1.1                    # MySQL 协议（已有）
pyyaml>=6.0                     # 语义层 YAML 解析
```
