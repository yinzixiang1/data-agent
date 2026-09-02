# NL2SQL Data Agent

面向 Doris 的自然语言转 SQL HTTP 服务。服务从 MySQL 读取 Agent 配置和语义层，
通过 Milvus 混合检索定位相关 Schema，再由 LLM 生成 SQL，并在返回或执行前完成
确定性安全校验。

整体架构、RAG 和统一接入见[项目文档中心](../docs/README.md)；Agent 专属镜像与运行约定见[本项目文档](docs/README.md)。本 README 只保留本地开发入口。

## 查询流程

1. 使用模型合并滚动查询状态和上一轮验证 SQL，必要时返回结构化澄清问题。
2. 解析业务术语和实体证据，以完整问题和粗粒度状态驱动检索。
3. 对表、字段、枚举、值和 Few-shot 进行 Dense + BM25 混合检索。
4. 使用 Reranker、Join 路径补全和字段裁剪构造最小 Schema 上下文。
5. 调用配置的 LLM 生成 SQL。
6. 校验只读语句、Schema 引用、数据库授权、实体条件和追问继承契约。
7. 使用 Doris EXPLAIN 校验语法及执行计划，按配置进行有限次数修复。
8. 可选执行 SQL、限制返回行数，并规划注册的结果工具调用。

## 项目结构

```text
data-agen/
├── app.py                       FastAPI 入口、生命周期和 HTTP 路由
├── pyproject.toml               Python 项目及依赖声明
├── uv.lock                      可复现依赖锁文件
└── src/
    ├── api/                     HTTP DTO
    ├── runtime/                 数据库和日志运行时边界
    ├── tools/                   查询结果工具执行
    ├── request_params.py        请求级可覆盖参数定义
    └── retrieval/
        ├── agent_config.py      Agent 动态配置与资源绑定
        ├── query_preparation.py 轮次预理解和澄清准备
        ├── query_pipeline.py    检索、生成、校验、执行编排
        ├── config.py            环境变量和启动配置校验
        ├── schema_loader.py     MySQL 语义层与 Doris Schema 合并
        ├── index_manager.py     Milvus 索引构建、连接和局部重建
        ├── hybrid_searcher.py   表、字段和枚举混合召回
        ├── retriever.py         RAG 检索统一入口
        ├── context_*.py         多轮状态、Join 补全和字段裁剪
        ├── sql_validator.py     SQL 安全、语义、EXPLAIN 和执行校验
        ├── llm_factory.py       LLM provider 创建入口
        └── codex_chat_model.py  Codex SDK 的 LangChain 适配器
```

## 基础设施依赖

- Python 3.11+
- MySQL：Agent 配置、资源绑定、语义层和查询日志
- Milvus 2.5+：Dense、BM25 和混合检索索引
- Doris：Schema 读取、EXPLAIN 和可选 SQL 执行
- 一个受支持的 LLM provider，或已完成认证的 Codex 运行环境

## 安装

项目使用 `uv` 管理依赖：

```bash
uv sync
```

仅安装生产依赖：

```bash
uv sync --no-dev
```

仅在本地运行 Embedding 或 Reranker 模型时安装模型依赖：

```bash
uv sync --extra local-models
```

## 配置

启动参数统一维护在 `configs/config.yaml`。生产环境推荐使用 MySQL 配置来源，
至少需要填写以下项目：

- `CONFIG_PROFILE`：当前进程绑定的 Agent ID
- `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE`
- `MILVUS_URI`、`MILVUS_DB`，以及按需配置的认证参数
- `DEFAULT_AGENT_TOKEN`：Agent 未单独配置查询令牌时的默认值
- `AGENT_ADMIN_TOKEN`：Admin 控制面调用管理接口的独立令牌，至少 32 字符

`CONFIG_PROFILE` 是当前进程绑定的 Agent ID。一个运行实例只服务一个 Agent；
请求中的 `agent_id` 与实例绑定不一致时返回 `409`。

使用本地 JSON 配置时，在 `configs/config.yaml` 中将 `CONFIG_SOURCE` 设为
`local`，并将 `CONFIG_PROFILE` 设为本地 JSON 文件路径。真实密钥不得提交。

## 启动

默认复用已有 Milvus 索引：

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 9090
```

语义层发生变更时，将 `configs/config.yaml` 中的
`REBUILD_INDEX_ON_STARTUP` 临时设为 `true` 后重启。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务及 RAG 就绪状态 |
| POST | `/query` | NL2SQL 查询、校验和可选执行 |
| POST | `/query/execute-saved` | 执行已验证的只读 SQL |
| POST | `/admin/index-rebuild` | 全量或局部重建 Milvus 索引 |
| POST | `/admin/config-reload` | 热加载当前 Agent 配置 |
| GET | `/admin/codex/status` | 获取不含身份信息的 Codex 状态 |
| POST | `/admin/codex/test` | 测试指定 Codex 模型 |
| POST | `/evaluation/run` | 批量执行评估用例 |

除 `/health` 外，业务接口使用 Agent Token，管理接口使用独立的
`AGENT_ADMIN_TOKEN`。Token 通过 `Authorization: Bearer <token>` 传递；
两个令牌不得复用。

## 索引与数据边界

每个 Agent 使用独立的 Milvus Collection 后缀。索引内容包括：

- table：表及用途；
- column：字段 Schema 和业务含义；
- enum：枚举定义；
- value：可过滤实体值；
- glossary：业务术语；
- fewshot：经审核的 SQL 示例。

MySQL 语义层决定当前 Agent 可见的数据范围；SQL 执行前还会再次根据
`da_agent_exec_db` 验证数据库授权。授权信息无法确认时，服务拒绝执行，不回退到
环境变量权限。

## 变更生效方式

| 变更 | 操作 |
| --- | --- |
| Prompt、LLM、检索参数 | `POST /admin/config-reload` |
| 表、字段、术语、枚举、Few-shot | `POST /admin/index-rebuild` |
| Embedding 或 Reranker 模型 | 重启服务 |
| Doris 表结构 | 重建索引 |

镜像构建、Compose 启动和健康检查见[Agent 镜像与运行](docs/部署.md)；跨服务发布顺序见[项目部署与运维](../docs/deployment/部署与运维.md)。
