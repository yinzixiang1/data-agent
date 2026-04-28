# NL2SQL Milvus 设计方案

> 针对支付/银行业务场景（以 `pmt_account` 表为例），将数据库元数据向量化到 Milvus，配合 LLM 实现自然语言转 SQL。

---

## 一、整体架构

NL2SQL 的核心思路：**把数据库元数据向量化** → 用户提问时**检索相关元数据** → 拼装 prompt 交给 LLM 生成 SQL。

### Collection 划分（4 层）

| Collection | 内容 | 作用 |
|---|---|---|
| `nl2sql_tables` | 表元数据 | 检索"用户问题涉及哪些表" |
| `nl2sql_columns` | 字段元数据 | 检索"涉及哪些字段" |
| `nl2sql_enums` | 枚举值字典 | 把"持牌商户"映射到 `account_type=2000` |
| `nl2sql_examples` | Few-shot 示例 | 提供"问题→SQL"参考样本 |

> 枚举值是否独立成 Collection 看场景：枚举密集型业务（如支付、风控）建议独立；普通业务可塞在字段 Collection 的 `enum_values` 字段里。

---

## 二、Collection Schema 设计

### 1. 表元数据 `nl2sql_tables`

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

table_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="db_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="table_cn_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="table_comment", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="ddl", dtype=DataType.VARCHAR, max_length=16384),
    FieldSchema(name="business_domain", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
```

**向量化文本拼接**（关键：不要只用 comment）：

```
表: dwd_banking.pmt_account（支付商户账户表）
业务域: 支付、账户管理、KYC认证、风控
描述: 存储商户账户信息，包含商户ID、账户类型、KYC状态、风险等级、地址、邮箱等
关键字段: customer_id（客户ID）, account_type（账户类型）, verification_status（KYC状态）, risk_rating_level（风险等级）, country（国家）
```

---

### 2. 字段元数据 `nl2sql_columns`

```python
column_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="db_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="column_cn_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="column_type", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="column_comment", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="enum_values", dtype=DataType.VARCHAR, max_length=2048),  # 简短摘要
    FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sample_values", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="is_enum", dtype=DataType.BOOL),
    FieldSchema(name="is_pii", dtype=DataType.BOOL),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
```

**向量化文本示例**（pmt_account.verification_status）：

```
表: pmt_account（支付商户账户表）
字段: verification_status
中文名: KYC认证状态
含义: 商户的 KYC/KYB 认证审核状态
枚举值: -1=拒绝, 1=已通过, 2=审核中, 0=默认未提交
同义词: KYC状态、认证状态、审核状态、是否认证、是否通过审核
```

---

### 3. 枚举值字典 `nl2sql_enums`（可选，但推荐）

```python
enum_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="db_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="enum_code", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="enum_label_cn", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="enum_label_en", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="enum_abbr", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sql_value", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="value_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
```

**单条枚举数据示例**（LPSP）：

```python
{
  "table_name": "pmt_account",
  "column_name": "account_type",
  "enum_code": "2000",
  "enum_label_cn": "持牌支付服务商",
  "enum_label_en": "License Payment Service Provider",
  "enum_abbr": "LPSP",
  "description": "拥有支付牌照的服务提供商账户",
  "synonyms": "持牌商户,持牌PSP,LPSP,有牌照的支付服务商",
  "sql_value": "2000",
  "value_type": "INT"
}
```

**向量化文本**：

```
枚举值: 持牌支付服务商
英文: License Payment Service Provider
缩写: LPSP
对应字段: pmt_account.account_type
实际取值: 2000
说明: 拥有支付牌照的服务提供商账户
同义词: 持牌商户,持牌PSP,LPSP,有牌照的支付服务商
```

---

### 4. Few-shot 示例 `nl2sql_examples`

```python
example_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="sql", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="involved_tables", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="business_domain", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
```

**只对 question 做向量化**，SQL 不参与向量化。

---

## 三、索引配置

### 向量索引

```python
# 密集向量（语义检索）
collection.create_index(
    field_name="dense_vec",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",  # BGE 系列模型推荐
        "params": {"M": 16, "efConstruction": 200}
    }
)

# 稀疏向量（关键词检索）
collection.create_index(
    field_name="sparse_vec",
    index_params={
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
)
```

### 标量索引（提升过滤性能）

```python
collection.create_index(field_name="db_name", index_params={"index_type": "INVERTED"})
collection.create_index(field_name="table_name", index_params={"index_type": "INVERTED"})
collection.create_index(field_name="business_domain", index_params={"index_type": "INVERTED"})
```

---

## 四、运行时检索流程

**用户问题**：「新加坡的高风险持牌商户有多少个」

```python
query = "新加坡的高风险持牌商户有多少个"
query_dense, query_sparse = bge_m3_encode(query)

# 1. 检索相关表
tables = tables_collection.hybrid_search(
    [query_dense, query_sparse], limit=3
)
# → 命中 pmt_account

# 2. 检索相关字段（限定到上一步的表）
columns = columns_collection.hybrid_search(
    [query_dense, query_sparse],
    expr=f'table_name == "pmt_account"',
    limit=10
)
# → 命中 country, risk_rating_level, account_type, verification_status

# 3. 检索相关枚举值
enums = enums_collection.hybrid_search(
    [query_dense, query_sparse],
    expr=f'table_name == "pmt_account"',
    limit=8
)
# → "持牌商户" → account_type=2000 (LPSP)
# → "高风险"   → risk_rating_level='HIGH'/'VERY_HIGH'

# 4. 检索 Few-shot 示例
examples = examples_collection.hybrid_search(
    [query_dense, query_sparse], limit=3
)
```

---

## 五、Prompt 模板

```text
你是 SQL 专家，根据以下信息生成 SQL：

【相关表】
{tables}

【相关字段】
{columns}

【枚举值映射】（重要：用户的语义化描述对应的实际取值）
- "持牌商户" / "LPSP" → pmt_account.account_type = 2000
- "高风险" → pmt_account.risk_rating_level = 'HIGH'
- "极高风险" → pmt_account.risk_rating_level = 'VERY_HIGH'

【相似示例】
{examples}

【业务约定】
- 查询活跃数据时必须带 is_delete = 0

【用户问题】
{user_question}

请生成 SQL：
```

---

## 六、技术选型建议

| 组件 | 推荐 | 备注 |
|---|---|---|
| 向量数据库 | Milvus 2.4+ | 支持稀疏向量和混合检索 |
| Embedding 模型 | BGE-M3 | 同时输出密集+稀疏向量，多语言 |
| 检索策略 | 混合检索（Hybrid） | Dense + Sparse + 标量过滤 |
| 融合算法 | RRF (Reciprocal Rank Fusion) | Milvus 内置支持 |
| 可视化工具 | Attu | Milvus 官方 GUI |

---

## 七、实施 Checklist

按优先级动手：

- [ ] **1. 补全字段 comment**——空 comment 的字段必须人工或 LLM 辅助标注
- [ ] **2. 整理枚举字典**——把 DDL comment 里 `1000=xxx, 2000=yyy` 解析成结构化数据
- [ ] **3. 标注同义词**——业务黑话、缩写、中英文对照（如 LPSP/持牌支付服务商）
- [ ] **4. 准备 Few-shot 示例**——从历史 BI 报表、运营查询里捞 30~50 个真实样本
- [ ] **5. 选定 Embedding 模型**——推荐 BGE-M3
- [ ] **6. 部署 Milvus + Attu**——本地 Docker Compose 起步
- [ ] **7. 实现向量化 + 入库脚本**
- [ ] **8. 实现混合检索 + Prompt 拼装**
- [ ] **9. 上线后回流——好的 Q&A 自动加进 examples**

---

## 八、常见坑点

1. **字段 comment 不能空**——这是 NL2SQL 准确率的最大决定因素
2. **枚举值必须穷举**——否则 LLM 写出 `WHERE status='已付款'` 这种错的 SQL
3. **混合检索胜过纯向量**——专有名词、订单号、商户编号靠稀疏向量才准
4. **业务约定要进 prompt**——比如逻辑删除字段 `is_delete = 0`
5. **Few-shot 质量 > 数量**——100 个精选示例胜过 1000 个杂乱示例
6. **多表共享枚举要去重**——独立枚举 Collection 时尤其注意
7. **PII 字段要标记**——身份证号、银行卡号等敏感字段做查询限制

---

## 九、参考命令

### 启动 Milvus（Docker Compose）

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### 启动 Attu（可视化）

```bash
docker run -p 8000:3000 \
  -e MILVUS_URL=host.docker.internal:19530 \
  zilliz/attu:latest
```

访问 http://localhost:8000

### Python SDK 快速验证

```python
from pymilvus import connections, utility, Collection

connections.connect(host='localhost', port='19530')
print(utility.list_collections())

col = Collection("nl2sql_tables")
col.load()
print("数据量:", col.num_entities)
```

---

## 十、扩展方向

- **业务术语库**：把 GMV、DAU 等业务黑话和 SQL 计算口径单独建库
- **JOIN 关系库**：表之间的常见关联写法
- **拒答样本库**：哪些问题不该回答（涉及隐私、跨租户等）
- **SQL 校验**：生成后用 EXPLAIN 或 dry-run 验证语法
- **结果缓存**：相同问题 + 相同表结构直接复用历史 SQL
