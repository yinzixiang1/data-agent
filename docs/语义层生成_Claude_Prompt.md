# 语义层 YAML 生成 — Claude Prompt 模板

> **使用方式**：将下面「Prompt 模板」部分完整复制到 Claude 对话中，按提示粘贴 DDL 和业务代码即可。
>
> **前置准备**：
> 1. 导出整库 DDL：`mysqldump -h HOST -u USER -p --no-data 库名 > 库名_ddl.sql`
>    或单表：`SHOW CREATE TABLE 库名.表名;`
> 2.（可选）准备相关业务代码片段（Go/Java/Python 的 entity 定义、enum 常量、常见查询）

---

## Prompt 模板

下面是完整 prompt，从 `---开始---` 到 `---结束---` 之间的内容**整体复制**粘贴到 Claude：

---开始---

你是一个 NL2SQL 语义层数据工程师。请根据我提供的「建表 DDL」和「业务代码/文档」，为 DDL 中的**每张表**各生成一份独立的语义层 YAML。

## 输出格式要求

**每张表输出一个独立的 yaml 代码块**，文件名标注在代码块上方。严格按照以下 YAML 结构，不要输出其他内容：

`表名.yaml`
```yaml
table:
  name: "<MySQL 表名>"
  database: "<MySQL 库名>"
  display_name: "<中文表名，2-8个字>"
  description: |
    <2-4 句话描述这张表：记录什么业务数据、每行代表什么、常用于什么查询场景>
  tags: ["<中文关键词>", "<英文别名>", "<业务术语>", "<用户口语表达>"]
  query_tips: "<查询注意事项，如：时间范围优先使用 create_time 字段>"

columns:
  # 对每一列输出以下结构（所有列都要列出）
  - name: "<列名>"
    type: "<数据类型>"
    display_name: "<中文列名，2-6个字>"
    description: "<列含义说明>"
    # 以下字段仅在需要时添加：
    enum_values:          # 仅枚举列需要
      - {value: <码值>, label: "<含义>"}
    sensitive: true        # 仅敏感列需要（证件号、卡号、密码、密钥等PII数据）
    skip_index: true       # 仅技术列需要（JSON blob、元数据、内部配置、大文本）
    business_logic: "<计算逻辑说明>"  # 仅派生/计算列需要

relations:
  # 根据 DDL 中的外键约束或业务代码中的 JOIN 关系填写，不确定就写空数组 []
  - column: "<本表列名>"
    target_table: "<目标表名>"
    target_column: "<目标表列名>"
    join_type: "LEFT JOIN"

common_queries:
  # 生成 3-5 条冷启动种子查询，覆盖该表最核心的查询场景
  # 系统上线后会通过用户反馈自动积累更多语料，无需一次写全
  - question: "<用户最可能怎么问，用口语化表达>"
    sql: |
      <完整可执行的 MySQL SQL>
    tables: ["<涉及的表名>"]
    difficulty: "<easy / medium / hard>"
```

## 生成规则

### 表级
1. `display_name` 简短有辨识度，如"发卡卡片表""商户账户表"
2. `description` 写清楚：记录什么、每行代表什么、常用场景
3. `tags` 至少 4 个：中文关键词 + 英文名 + 业务术语 + 表名本身

### 列级
4. **所有列都要列出**，即使没有额外信息也要给 name + type + display_name + description
5. `display_name` 翻译为简短中文，如 "card_id" → "卡片ID"
6. 枚举列必须列出所有 `enum_values`，优先从业务代码的 enum/const 定义获取
7. 敏感列标记 `sensitive: true`：卡号、CVV、密码、PIN、证件号、加密字段、token 密钥
8. 技术列标记 `skip_index: true`：JSON 元数据、大文本配置、内部控制 blob
9. DDL COMMENT 含义不清的列，结合代码推断并补充 description

### 关联关系
10. 根据 DDL 外键约束自动提取；也可结合业务代码中的 JOIN 关系补充
11. 没有可确认的关联就写 `relations: []`

### 冷启动种子查询（common_queries）
12. 只需 **3-5 条**，聚焦该表最核心、最高频的查询场景
13. 选取原则（按优先级）：
    - **必选**：1 条简单计数/汇总（"XX有多少条"）— easy
    - **必选**：1 条按核心维度分组统计（"按XX分布"）— easy/medium
    - **必选**：1 条带枚举值的条件筛选（体现 CASE WHEN 用法）— medium
    - 可选：1 条时间范围查询（"最近30天的XX"）— easy
    - 可选：1 条涉及该表特有业务逻辑的查询 — medium/hard
14. SQL 必须是可直接在 **MySQL** 上执行的语法
15. 表名必须带库名前缀：`FROM 库名.\`表名\``
16. 列别名用单引号：`COUNT(*) AS '数量'`
17. 枚举值查询用 CASE WHEN 翻译为中文展示
18. 时间过滤用 `DATE_SUB(CURRENT_DATE(), INTERVAL N DAY)`
19. 所有 SELECT 都加 `LIMIT`（聚合加 LIMIT 1，列表加 LIMIT 100）
20. difficulty 判定：单表简单聚合 = easy，CASE WHEN/多条件 = medium，多表 JOIN/窗口函数 = hard

### 多表处理
21. **自动跳过非业务表**：表名匹配以下模式的直接忽略，不生成 YAML：
    - 备份表：含 `_bak`、`_backup`、`_back`、`_copy`、`_old`、`_tmp`、`_temp` 后缀
    - 日期快照表：含日期后缀如 `_20250101`、`_250101`、`_0712`、`_2024-12-31`
    - 日志/归档表：含 `_log`、`_archive`、`_hist`、`_history` 后缀（审计日志表除外）
    - 测试表：含 `_test`、`test_`、`_dev` 前缀或后缀
    - 如有疑似备份但不确定的表，在输出末尾列出跳过的表名清单供人工确认
22. DDL 中包含多张表时，**每张业务表各生成一个独立的 yaml 代码块**
23. 跨表的 `relations` 可以根据 DDL 中的外键约束或字段命名规律（如 `xxx_id` 对应 `xxx` 表）推断
24. `common_queries` 中如涉及多表 JOIN，`tables` 数组列出所有涉及的表名
25. 如果表数量较多（>10 张），先输出所有表，每张表都不能省略

## 输入数据

### 建表 DDL

```sql
<在这里粘贴整库 DDL（mysqldump --no-data 输出）或多张表的 SHOW CREATE TABLE 输出>
```

### 业务代码/文档（可选，有就贴）

```
<在这里粘贴相关的 entity 定义、enum 常量、查询代码、接口文档等>
```

---结束---

---

## 使用示例

### 场景 1：整库 DDL（最常见）

```bash
# 导出整库 DDL
mysqldump -h 127.0.0.1 -u root -p --no-data 库名 > 库名_ddl.sql
```

将导出的 SQL 文件内容粘贴到 prompt 模板的「建表 DDL」处。Claude 会自动识别每张表并逐一生成 YAML。

### 场景 2：DDL + 业务代码

1. 先让 Claude 扫描业务项目：

```
请扫描当前项目代码，提取与数据库 <库名> 相关的：
1. 枚举/常量定义（Go const、Java enum 等）
2. 常见查询模式（WHERE 条件、GROUP BY、JOIN）
3. 字段中文含义（注释、Swagger、i18n）
4. 敏感字段（有加密/脱敏处理的）

输出为结构化文本。
```

2. 拿到输出后，和 DDL 一起粘贴到 prompt 模板中。

### 场景 3：表太多时分批

如果整库超过 20 张表，建议分批（每批 5-10 张），避免单次输出过长：

```
先处理以下表（其余后续再给）：
pmt_account, pmt_finance_transactions, pmt_finance_payout, pmt_conversion, pmt_account_balance

DDL 如下：
<只粘贴这几张表的 DDL>
```

---

## 拿到 YAML 后

1. **人工审核**：重点检查 enum_values 完整性、SQL 可执行性、sensitive 标记
2. **保存为** `表名.yaml`，放入 `data-agen/semantic_layer/tables/<业务线>/` 目录
3. **运行导入**：`cd dataAgent-admin-api && python scripts/init_from_yaml.py`
4. **触发索引重建**：通过管理后台或调用 `/system/index-rebuild` 接口

## 关于语料积累

YAML 中的 `common_queries` 只是**冷启动种子**（3-5 条），无需追求覆盖面。系统上线后会通过以下方式自动扩充语料库：

1. **查询日志**：每次用户提问都会记录到 `sys_query_log`（含问题、生成 SQL、执行结果、用户反馈）
2. **反馈转化**：运营人员在管理后台查看日志，将用户反馈为 good 的查询一键转为 Few-shot 语料
3. **持续优化**：随着语料积累，系统的 SQL 生成质量会自然提升

因此，YAML 编写时**重心放在表/列/枚举的语义描述上**，common_queries 写几条典型查询即可。
