# 向量化文档格式示例 — pmt_account

基于真实 Doris DDL `dwd_banking.pmt_account` 生成。

---

## 1. 数据源：Doris DDL 原始解析结果

`schema_loader.py` 从 Doris 加载后输出的 Schema dict：

```json
{
  "database": "dwd_banking",
  "table_name": "pmt_account",
  "table_comment": "普通账户表",
  "unique_key": ["paid"],
  "columns": [
    {"name": "paid", "type": "INT", "nullable": false, "key": "UNI", "comment": ""},
    {"name": "customer_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "comment": "客户ID"},
    {"name": "direct_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "default": "0", "comment": "父ID（常用于机构模式）"},
    {"name": "account_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "comment": "系统内部账户ID"},
    {"name": "short_account_id", "type": "VARCHAR(60)", "nullable": true, "key": "", "comment": "商户短号"},
    {"name": "account_type", "type": "INT", "nullable": true, "key": "", "default": "1000", "comment": "1000=普通账户；2000=LPSP；2001=TPSP；3000=RPSP"},
    {"name": "account_name", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "商户短名，VM中显示在客户账单上用"},
    {"name": "nick_name", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "商户昵称"},
    {"name": "legal_entity_type", "type": "INT", "nullable": true, "key": "", "comment": "1000=individual 2000=company"},
    {"name": "legal_entity_name", "type": "VARCHAR(750)", "nullable": true, "key": "", "comment": "当legal_entity_type=company时的名称"},
    {"name": "legal_reg_name", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "当legal_entity_type=company时除英文外名字"},
    {"name": "legal_first_name", "type": "VARCHAR(360)", "nullable": true, "key": "", "comment": "legal_entity_type=individual时填写"},
    {"name": "legal_last_name", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "legal_entity_type=individual时填写"},
    {"name": "legal_date_of_birth", "type": "DATE", "nullable": true, "key": "", "comment": "legal_entity_type=individual时填写"},
    {"name": "identification_type", "type": "INT", "nullable": true, "key": "", "comment": "1000-passport, 1001-Drivers License; 1002=National ID, 2000-Incorporation number"},
    {"name": "identification_value", "type": "VARCHAR(144)", "nullable": true, "key": "", "comment": ""},
    {"name": "identification_issue_date", "type": "VARCHAR(30)", "nullable": true, "key": "", "comment": "对应证书签发时间 Format：YYYY-MM-DD"},
    {"name": "identification_expiry_date", "type": "VARCHAR(30)", "nullable": true, "key": "", "comment": "对应证书过期时间 Format：YYYY-MM-DD"},
    {"name": "incorporate_date", "type": "VARCHAR(30)", "nullable": true, "key": "", "comment": "legal_entity_type=company时必填写 Format：YYYY-MM-DD"},
    {"name": "industry_mcc_info", "type": "STRING", "nullable": true, "key": "", "comment": "行业属性"},
    {"name": "industry_mcc_code", "type": "VARCHAR(18)", "nullable": true, "key": "", "comment": "自动取industry_mcc表"},
    {"name": "risk_rating_level", "type": "VARCHAR(72)", "nullable": true, "key": "", "comment": "risk level: LOW MEDIUM HIGH VERY_HIGH MANUAL_REVIEW NONE NORMAL"},
    {"name": "risk_cra_level", "type": "INT", "nullable": true, "key": "", "comment": ""},
    {"name": "risk_score", "type": "INT", "nullable": true, "key": "", "comment": ""},
    {"name": "metadata", "type": "STRING", "nullable": true, "key": "", "comment": "元数据"},
    {"name": "business_scope", "type": "STRING", "nullable": true, "key": "", "comment": "业务范围"},
    {"name": "business_owners", "type": "STRING", "nullable": true, "key": "", "comment": "企业权益人"},
    {"name": "website_url", "type": "VARCHAR(750)", "nullable": true, "key": "", "comment": ""},
    {"name": "purpose", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "用途"},
    {"name": "country", "type": "VARCHAR(9)", "nullable": true, "key": "", "comment": ""},
    {"name": "city", "type": "VARCHAR(300)", "nullable": true, "key": "", "comment": ""},
    {"name": "state", "type": "VARCHAR(300)", "nullable": true, "key": "", "comment": ""},
    {"name": "address", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": ""},
    {"name": "postal_code", "type": "VARCHAR(48)", "nullable": true, "key": "", "comment": ""},
    {"name": "monthly_revenue", "type": "VARCHAR(288)", "nullable": true, "key": "", "comment": "月收入/营业额"},
    {"name": "trading_address", "type": "STRING", "nullable": true, "key": "", "comment": ""},
    {"name": "create_time", "type": "DATETIME", "nullable": true, "key": "", "comment": ""},
    {"name": "update_time", "type": "DATETIME", "nullable": true, "key": "", "comment": ""},
    {"name": "ref_bnb_sub_uid", "type": "VARCHAR(480)", "nullable": true, "key": "", "comment": ""},
    {"name": "ref_bnb_sub_email", "type": "VARCHAR(600)", "nullable": true, "key": "", "comment": ""},
    {"name": "ref_metacomp_uid", "type": "VARCHAR(30)", "nullable": true, "key": "", "default": "0", "comment": "临时字段"},
    {"name": "ref_cc_account_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "comment": "关联CC ID"},
    {"name": "ref_dbs_account_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "comment": "关联DBS ID"},
    {"name": "verification_status", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "KYC/KYB 认证状态 -1=拒绝, 2=pending, 1=approved"},
    {"name": "api_trading_status", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "1：启用；-1:禁用"},
    {"name": "payment_authorisation", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "支付授权多签配置"},
    {"name": "automatic_settlement_config", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "自动结算配置"},
    {"name": "allow_verification_subaccount", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "默认不允许直接审核下级商户"},
    {"name": "allow_follow_direct_account", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "允许跟随主账户信息，包含基本信息以及kyc等"},
    {"name": "account_status", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "0:default, 1:active, 2:inactive"},
    {"name": "main_email", "type": "VARCHAR(360)", "nullable": true, "key": "", "comment": "开户邮箱"},
    {"name": "sync_to_uq", "type": "INT", "nullable": true, "key": "", "default": "0", "comment": "同步到UQ表标记：0=未同步，1=已同步，2=暂不处理"},
    {"name": "tags", "type": "VARCHAR(765)", "nullable": true, "key": "", "comment": "tags标记"},
    {"name": "temp_product", "type": "SMALLINT", "nullable": true, "key": "", "default": "0", "comment": "模板account"},
    {"name": "white_label_status", "type": "TINYINT", "nullable": true, "key": "", "default": "0", "comment": "白标状态：0=未开通，1=已开通"},
    {"name": "restrict_id", "type": "INT", "nullable": true, "key": "", "comment": "约束ID：找1001类型的"},
    {"name": "is_delete", "type": "INT", "nullable": true, "key": "", "default": "0", "comment": "是否删除：0否，1是"},
    {"name": "fx_management_id", "type": "VARCHAR(108)", "nullable": true, "key": "", "comment": "换汇规则配置ID"}
  ]
}
```

---

## 2. 语义层 YAML（人工补充）

DDL 自带的 comment 已经比较丰富，语义层主要补充：**中文表名、业务标签、关联关系、常见问题、业务术语**。

```yaml
# semantic_layer/tables/pmt_account.yaml
table:
  name: pmt_account
  database: dwd_banking
  display_name: 商户账户表
  description: |
    支付系统的核心账户表，记录所有商户/个人的账户信息，
    包含基本信息、法人实体信息、KYC认证状态、风险评级、
    账户状态、结算配置等。每行代表一个独立账户。
  tags: [账户, 商户, KYC, 风险, 支付, 开户]

columns:
  - name: paid
    display_name: 账户主键ID
  - name: customer_id
    display_name: 客户ID
    description: 关联客户维度表的唯一标识
  - name: account_type
    display_name: 账户类型
    enum_values:
      - {value: 1000, label: 普通账户}
      - {value: 2000, label: "LPSP(License Payment Service Provider)"}
      - {value: 2001, label: "TPSP(Technical Payment Service Provider)"}
      - {value: 3000, label: "RPSP(Referral Payment Service Provider)"}
  - name: legal_entity_type
    display_name: 法人实体类型
    enum_values:
      - {value: 1000, label: 个人(individual)}
      - {value: 2000, label: 企业(company)}
  - name: risk_rating_level
    display_name: 风险等级
    enum_values: [LOW, MEDIUM, HIGH, VERY_HIGH, MANUAL_REVIEW, NONE, NORMAL]
  - name: verification_status
    display_name: KYC/KYB认证状态
    enum_values:
      - {value: -1, label: 拒绝}
      - {value: 0, label: 默认/未认证}
      - {value: 1, label: 已通过}
      - {value: 2, label: 审核中(pending)}
  - name: account_status
    display_name: 账户状态
    enum_values:
      - {value: 0, label: 默认}
      - {value: 1, label: 活跃(active)}
      - {value: 2, label: 停用(inactive)}
  - name: country
    display_name: 国家
    description: 商户所在国家代码
  - name: monthly_revenue
    display_name: 月营业额
    description: 商户月收入/营业额
  - name: white_label_status
    display_name: 白标状态
    enum_values:
      - {value: 0, label: 未开通}
      - {value: 1, label: 已开通}
  - name: is_delete
    display_name: 是否删除
    description: 软删除标记，查询时通常需要 is_delete = 0
    enum_values:
      - {value: 0, label: 未删除}
      - {value: 1, label: 已删除}

relations:
  - column: customer_id
    target_table: dim_customer
    target_column: customer_id
    join_type: LEFT JOIN
    description: 关联客户维度表获取客户详细信息

common_queries:
  - question: 目前有多少活跃商户
    sql: |
      SELECT COUNT(*) AS "活跃商户数"
      FROM dwd_banking.pmt_account
      WHERE account_status = 1
        AND is_delete = 0
      LIMIT 1
    tables: [pmt_account]
    difficulty: easy

  - question: 各国家的商户数量分布
    sql: |
      SELECT country AS "国家", COUNT(*) AS "商户数"
      FROM dwd_banking.pmt_account
      WHERE is_delete = 0
      GROUP BY country
      ORDER BY COUNT(*) DESC
      LIMIT 100
    tables: [pmt_account]
    difficulty: easy

  - question: KYC认证通过率
    sql: |
      SELECT
        COUNT(CASE WHEN verification_status = 1 THEN 1 END) AS "已通过",
        COUNT(*) AS "总数",
        ROUND(COUNT(CASE WHEN verification_status = 1 THEN 1 END) / COUNT(*) * 100, 2) AS "通过率(%)"
      FROM dwd_banking.pmt_account
      WHERE is_delete = 0
      LIMIT 1
    tables: [pmt_account]
    difficulty: medium

  - question: 各风险等级的商户数量
    sql: |
      SELECT risk_rating_level AS "风险等级", COUNT(*) AS "商户数"
      FROM dwd_banking.pmt_account
      WHERE is_delete = 0
        AND risk_rating_level IS NOT NULL
      GROUP BY risk_rating_level
      ORDER BY COUNT(*) DESC
      LIMIT 100
    tables: [pmt_account]
    difficulty: easy

  - question: 最近一个月新开户的商户数
    sql: |
      SELECT COUNT(*) AS "新开户数"
      FROM dwd_banking.pmt_account
      WHERE create_time >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        AND is_delete = 0
      LIMIT 1
    tables: [pmt_account]
    difficulty: easy

  - question: 各账户类型的商户占比
    sql: |
      SELECT
        CASE account_type
          WHEN 1000 THEN '普通账户'
          WHEN 2000 THEN 'LPSP'
          WHEN 2001 THEN 'TPSP'
          WHEN 3000 THEN 'RPSP'
          ELSE '其他'
        END AS "账户类型",
        COUNT(*) AS "数量",
        ROUND(COUNT(*) / SUM(COUNT(*)) OVER() * 100, 2) AS "占比(%)"
      FROM dwd_banking.pmt_account
      WHERE is_delete = 0
      GROUP BY account_type
      ORDER BY COUNT(*) DESC
      LIMIT 100
    tables: [pmt_account]
    difficulty: medium
```

---

## 3. 合并后的完整 Schema dict

`schema_loader.py` 将 DDL 解析结果和语义层 YAML 合并后的最终结构：

```json
{
  "database": "dwd_banking",
  "table_name": "pmt_account",
  "table_comment": "普通账户表",
  "display_name": "商户账户表",
  "description": "支付系统的核心账户表，记录所有商户/个人的账户信息，包含基本信息、法人实体信息、KYC认证状态、风险评级、账户状态、结算配置等。每行代表一个独立账户。",
  "tags": ["账户", "商户", "KYC", "风险", "支付", "开户"],
  "unique_key": ["paid"],
  "columns": [
    {"name": "paid", "type": "INT", "display_name": "账户主键ID", "comment": ""},
    {"name": "customer_id", "type": "VARCHAR(108)", "display_name": "客户ID", "comment": "关联客户维度表的唯一标识"},
    {"name": "account_type", "type": "INT", "display_name": "账户类型", "comment": "1000=普通账户；2000=LPSP；2001=TPSP；3000=RPSP", "enum_values": [{"value": 1000, "label": "普通账户"}, {"value": 2000, "label": "LPSP"}, {"value": 2001, "label": "TPSP"}, {"value": 3000, "label": "RPSP"}]},
    {"name": "verification_status", "type": "TINYINT", "display_name": "KYC/KYB认证状态", "comment": "KYC/KYB 认证状态", "enum_values": [{"value": -1, "label": "拒绝"}, {"value": 0, "label": "默认/未认证"}, {"value": 1, "label": "已通过"}, {"value": 2, "label": "审核中"}]},
    "...（其余列同理）"
  ],
  "relations": [
    {"column": "customer_id", "target_table": "dim_customer", "target_column": "customer_id", "join_type": "LEFT JOIN"}
  ],
  "common_queries": ["...（6条示例）"]
}
```

---

## 4. 向量化文档：表级文档

`document_builder.py` 将合并后的 Schema dict 转换为一段纯文本，交给 BGE-M3 编码。

**这段文本直接决定了"什么样的用户提问能匹配到这张表"**。

```
表名: pmt_account
中文名: 商户账户表
数据库: dwd_banking
描述: 支付系统的核心账户表，记录所有商户/个人的账户信息，包含基本信息、法人实体信息、KYC认证状态、风险评级、账户状态、结算配置等。每行代表一个独立账户。
业务标签: 账户, 商户, KYC, 风险, 支付, 开户
主要字段: paid(账户主键ID), customer_id(客户ID), account_id(系统内部账户ID), short_account_id(商户短号), account_type(账户类型), account_name(商户短名), legal_entity_type(法人实体类型), legal_entity_name(企业名称), risk_rating_level(风险等级), risk_score(风险评分), country(国家), monthly_revenue(月营业额), verification_status(KYC/KYB认证状态), account_status(账户状态), white_label_status(白标状态), create_time(创建时间)
常见问题: 目前有多少活跃商户
常见问题: 各国家的商户数量分布
常见问题: KYC认证通过率
常见问题: 各风险等级的商户数量
常见问题: 最近一个月新开户的商户数
常见问题: 各账户类型的商户占比
关联: dim_customer ON pmt_account.customer_id = dim_customer.customer_id
```

> 注意：主要字段只挑**业务含义明确的关键列**，跳过内部技术字段（metadata、sync_to_uq、ref_metacomp_uid 等），避免噪音干扰检索质量。

---

## 5. 向量化文档：列级文档

每列一个独立文档。只对**有业务含义的列**建索引，纯技术/内部字段跳过。

### 5.1 高价值列（建索引）

```
--- 列文档 1 ---
表: pmt_account(商户账户表)
列名: account_type
中文名: 账户类型
类型: INT
描述: 账户类型编码
可选值: 1000=普通账户, 2000=LPSP(License Payment Service Provider), 2001=TPSP(Technical Payment Service Provider), 3000=RPSP(Referral Payment Service Provider)

--- 列文档 2 ---
表: pmt_account(商户账户表)
列名: verification_status
中文名: KYC/KYB认证状态
类型: TINYINT
描述: KYC/KYB 认证状态
可选值: -1=拒绝, 0=默认/未认证, 1=已通过, 2=审核中(pending)

--- 列文档 3 ---
表: pmt_account(商户账户表)
列名: risk_rating_level
中文名: 风险等级
类型: VARCHAR(72)
描述: 商户风险评级等级
可选值: LOW, MEDIUM, HIGH, VERY_HIGH, MANUAL_REVIEW, NONE, NORMAL

--- 列文档 4 ---
表: pmt_account(商户账户表)
列名: account_status
中文名: 账户状态
类型: TINYINT
描述: 账户当前状态
可选值: 0=默认, 1=活跃(active), 2=停用(inactive)

--- 列文档 5 ---
表: pmt_account(商户账户表)
列名: legal_entity_type
中文名: 法人实体类型
类型: INT
描述: 账户所属法人实体的类型
可选值: 1000=个人(individual), 2000=企业(company)

--- 列文档 6 ---
表: pmt_account(商户账户表)
列名: country
中文名: 国家
类型: VARCHAR(9)
描述: 商户所在国家代码

--- 列文档 7 ---
表: pmt_account(商户账户表)
列名: monthly_revenue
中文名: 月营业额
类型: VARCHAR(288)
描述: 商户月收入/营业额

--- 列文档 8 ---
表: pmt_account(商户账户表)
列名: industry_mcc_code
中文名: MCC行业代码
类型: VARCHAR(18)
描述: 行业分类码，自动取industry_mcc表

--- 列文档 9 ---
表: pmt_account(商户账户表)
列名: white_label_status
中文名: 白标状态
类型: TINYINT
描述: 白标功能开通状态
可选值: 0=未开通, 1=已开通

--- 列文档 10 ---
表: pmt_account(商户账户表)
列名: create_time
中文名: 创建时间
类型: DATETIME
描述: 账户创建时间，用于统计开户趋势

--- 列文档 11 ---
表: pmt_account(商户账户表)
列名: risk_score
中文名: 风险评分
类型: INT
描述: 商户风险量化评分

--- 列文档 12 ---
表: pmt_account(商户账户表)
列名: main_email
中文名: 开户邮箱
类型: VARCHAR(360)
描述: 商户开户时使用的邮箱
```

### 5.2 跳过的列（不建索引）

以下列属于内部技术字段，用户不会直接提问，不建列级索引以减少噪音：

| 列名 | 跳过原因 |
|------|---------|
| `metadata` | JSON 元数据，非查询字段 |
| `business_scope` | JSON，非结构化 |
| `business_owners` | JSON，非结构化 |
| `payment_authorisation` | JSON 配置，非查询字段 |
| `automatic_settlement_config` | JSON 配置 |
| `ref_bnb_sub_uid` | 内部关联ID |
| `ref_bnb_sub_email` | 内部关联 |
| `ref_metacomp_uid` | 注释明确写了"临时字段" |
| `sync_to_uq` | 数据同步标记，纯技术字段 |
| `temp_product` | 模板标记 |
| `restrict_id` | 内部约束 |
| `trading_address` | 无注释，含义不明 |
| `identification_value` | 证件号码，敏感字段不应被检索 |

> 判断规则：无 comment + 无业务含义 → 跳过；JSON 配置类 → 跳过；敏感字段 → 跳过。

---

## 6. Prompt 注入格式：DDL

`schema_formatter.py` 将检索命中的表格式化为 LLM 最终看到的 Prompt 内容：

```sql
-- 商户账户表：支付系统的核心账户表，记录所有商户/个人的账户信息，
-- 包含基本信息、法人实体信息、KYC认证状态、风险评级、账户状态、结算配置等。
CREATE TABLE `dwd_banking`.`pmt_account` (
  `paid` INT                                -- 账户主键ID (UNIQUE KEY)
  `customer_id` VARCHAR(108),               -- 客户ID
  `direct_id` VARCHAR(108),                 -- 父ID（常用于机构模式）
  `account_id` VARCHAR(108),                -- 系统内部账户ID
  `short_account_id` VARCHAR(60),           -- 商户短号
  `account_type` INT,                       -- 账户类型 [1000=普通账户, 2000=LPSP, 2001=TPSP, 3000=RPSP]
  `account_name` VARCHAR(765),              -- 商户短名
  `nick_name` VARCHAR(765),                 -- 商户昵称
  `legal_entity_type` INT,                  -- 法人实体类型 [1000=个人, 2000=企业]
  `legal_entity_name` VARCHAR(750),         -- 企业名称（legal_entity_type=2000时）
  `legal_reg_name` VARCHAR(765),            -- 企业非英文名称
  `legal_first_name` VARCHAR(360),          -- 个人名（legal_entity_type=1000时）
  `legal_last_name` VARCHAR(765),           -- 个人姓（legal_entity_type=1000时）
  `legal_date_of_birth` DATE,               -- 出生日期（个人时）
  `identification_type` INT,                -- 证件类型 [1000=passport, 1001=Drivers License, 1002=National ID, 2000=Incorporation number]
  `identification_issue_date` VARCHAR(30),  -- 证书签发时间
  `identification_expiry_date` VARCHAR(30), -- 证书过期时间
  `incorporate_date` VARCHAR(30),           -- 公司成立日期（企业时）
  `industry_mcc_info` STRING,               -- 行业属性
  `industry_mcc_code` VARCHAR(18),          -- MCC行业代码
  `risk_rating_level` VARCHAR(72),          -- 风险等级 [LOW, MEDIUM, HIGH, VERY_HIGH, MANUAL_REVIEW, NONE, NORMAL]
  `risk_cra_level` INT,                     -- 风险CRA等级
  `risk_score` INT,                         -- 风险评分
  `website_url` VARCHAR(750),               -- 商户网站
  `purpose` VARCHAR(765),                   -- 账户用途
  `country` VARCHAR(9),                     -- 国家代码
  `city` VARCHAR(300),                      -- 城市
  `state` VARCHAR(300),                     -- 州/省
  `address` VARCHAR(765),                   -- 地址
  `postal_code` VARCHAR(48),                -- 邮编
  `monthly_revenue` VARCHAR(288),           -- 月营业额
  `create_time` DATETIME,                   -- 创建时间
  `update_time` DATETIME,                   -- 更新时间
  `verification_status` TINYINT,            -- KYC/KYB认证状态 [-1=拒绝, 0=默认, 1=已通过, 2=审核中]
  `api_trading_status` TINYINT,             -- API交易状态 [1=启用, -1=禁用]
  `account_status` TINYINT,                 -- 账户状态 [0=默认, 1=活跃, 2=停用]
  `main_email` VARCHAR(360),                -- 开户邮箱
  `tags` VARCHAR(765),                      -- 标签
  `white_label_status` TINYINT,             -- 白标状态 [0=未开通, 1=已开通]
  `is_delete` INT DEFAULT 0                 -- 软删除 [0=未删除, 1=已删除] ⚠️ 查询时通常需要 is_delete=0
);
-- JOIN 提示: pmt_account.customer_id = dim_customer.customer_id (LEFT JOIN)
-- 注意: is_delete=0 过滤已删除记录
```

> 关键细节：
> - enum 字段用 `[value=label]` 内联标注，LLM 生成 WHERE 条件时直接引用
> - `is_delete` 字段加 ⚠️ 提示，避免 LLM 忘记过滤
> - 跳过了 `metadata`、`business_scope`、`payment_authorisation` 等 JSON 配置字段，减少 token 消耗
> - 跳过了 `identification_value`（敏感数据）
> - 跳过了 `ref_*` 和 `sync_to_uq` 等纯技术字段

---

## 7. 检索效果验证示例

以下模拟不同用户提问命中 `pmt_account` 的路径：

| 用户提问 | 命中路径 | 说明 |
|---------|---------|------|
| "有多少活跃商户" | 表级 Dense（语义匹配"商户账户表"）+ 常见问题匹配 | 直接命中 |
| "KYC通过率" | 列级 Dense 命中 `verification_status`(KYC认证状态) → 反推 `pmt_account` | 列级检索价值 |
| "pmt_account" | 表级 Sparse 精确匹配表名 | Sparse 价值 |
| "高风险商户有哪些" | 列级 Dense 命中 `risk_rating_level`(风险等级) → 反推 `pmt_account` | 列级检索价值 |
| "各国家的开户数" | 表级 Dense（"开户"标签）+ 列级命中 `country`(国家) | 混合命中 |
| "白标商户数量" | 列级 Dense 命中 `white_label_status`(白标状态) → 反推 `pmt_account` | 列级检索价值 |
| "LPSP 和 TPSP 分别有多少" | 列级 Sparse 命中 `account_type` 的枚举值文本 | Sparse 精确匹配 |
| "上个月新注册的企业账户" | 表级 Dense + 列级命中 `create_time` + `legal_entity_type` | 多列联合 |

---

## 8. 总结：单表需要产出的向量化产物

| 产物 | 数量 | 用途 |
|------|------|------|
| 表级文档 | 1 个 | Dense + Sparse 编码 → 表级索引 |
| 列级文档 | 约 12 个（跳过技术/敏感字段） | Dense + Sparse 编码 → 列级索引 |
| Few-shot 示例 | 6 个（来自 common_queries） | Dense 编码 → 示例索引 |
| 完整 Schema dict | 1 个 | 检索命中后取完整信息用于 DDL 格式化 |
| DDL Prompt 文本 | 1 个（运行时生成） | 注入 SQL 生成 Prompt |
