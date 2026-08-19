"""NL2SQL 上下文规划：数据域路由、Join 路径补全和字段裁剪。"""

from __future__ import annotations

import copy
import logging
import re
import heapq
from collections import Counter
from itertools import combinations

logger = logging.getLogger(__name__)


class SchemaContextPlanner:
    """把粗召回表转换成生成 SQL 所需的最小 Schema 上下文。"""

    def __init__(self, table_schemas: dict[str, dict]):
        self.table_schemas = table_schemas
        self._short_to_full: dict[str, list[str]] = {}
        for full_name in table_schemas:
            short = full_name.rsplit(".", 1)[-1]
            self._short_to_full.setdefault(short.casefold(), []).append(full_name)
        self.graph = self._build_relation_graph()
        self.edge_costs = self._build_edge_costs()
        self.relation_columns = self._build_relation_columns()

    def resolve_table(self, name: str, source_table: str | None = None) -> str | None:
        """把短表名解析成已加载 Schema 中的全限定名。"""
        normalized = name.strip("`")
        if normalized in self.table_schemas:
            return normalized
        candidates = self._short_to_full.get(normalized.casefold(), [])
        if len(candidates) == 1:
            return candidates[0]
        if source_table and "." in source_table:
            database = source_table.rsplit(".", 1)[0].casefold()
            same_database = [
                candidate
                for candidate in candidates
                if candidate.rsplit(".", 1)[0].casefold() == database
            ]
            if len(same_database) == 1:
                return same_database[0]
        return None

    def infer_biz_line(self, query: str) -> str:
        """保守推断业务线；证据不充分时返回空字符串，避免误过滤。"""
        grouped: dict[str, list[dict]] = {}
        for schema in self.table_schemas.values():
            biz_line = str(schema.get("biz_line") or "").strip()
            if biz_line:
                grouped.setdefault(biz_line, []).append(schema)
        if len(grouped) <= 1:
            return next(iter(grouped), "")

        query_folded = query.casefold()
        query_tokens = self._tokens(query)
        scores: Counter[str] = Counter()
        for biz_line, schemas in grouped.items():
            if biz_line.casefold() in query_folded:
                scores[biz_line] += 5
            domain_tokens: Counter[str] = Counter()
            for schema in schemas:
                texts = [
                    schema.get("display_name", ""),
                    schema.get("description", ""),
                    " ".join(schema.get("tags", [])),
                ]
                domain_tokens.update(self._tokens(" ".join(texts)))
            scores[biz_line] += sum(
                min(domain_tokens[token], 3) for token in query_tokens
            )

        ranked = scores.most_common(2)
        if not ranked or ranked[0][1] < 2:
            return ""
        if len(ranked) > 1 and ranked[0][1] < ranked[1][1] + 2:
            return ""
        return ranked[0][0]

    def add_join_bridges(
        self,
        candidates: list[dict],
        max_hops: int = 2,
        max_tables: int = 8,
    ) -> tuple[list[dict], list[list[str]]]:
        """用候选表之间的最短关系路径补齐桥接表。"""
        result = list(candidates)
        present = {candidate["table_name"] for candidate in result}
        anchors = [candidate["table_name"] for candidate in candidates]
        paths: list[list[str]] = []

        for source, target in combinations(anchors, 2):
            path = self._shortest_path(source, target, max_hops=max_hops)
            if not path:
                continue
            paths.append(path)
            for table_name in path[1:-1]:
                if table_name in present:
                    continue
                if len(result) >= max_tables:
                    break
                schema = self.table_schemas.get(table_name)
                if not schema:
                    continue
                result.append(
                    {
                        "table_name": table_name,
                        "score": 0.0,
                        "source": "relation_path",
                        "pinned": True,
                        "relation_bridge": True,
                        "schema": schema,
                    }
                )
                present.add(table_name)

        # 确定性表不裁剪；其余按已有排序限制上下文规模。
        if len(result) > max_tables:
            required = [
                candidate
                for candidate in result
                if candidate.get("pinned") or candidate.get("relation_bridge")
            ]
            optional = [candidate for candidate in result if candidate not in required]
            result = required + optional[: max(0, max_tables - len(required))]
        return result, paths

    def prune_columns(
        self,
        candidates: list[dict],
        query: str,
        required_columns: set[str] | None = None,
        per_table_limit: int = 16,
        preserve_time_columns: bool = False,
    ) -> tuple[list[dict], dict]:
        """按问题选择字段，并保留键、关系、口径及查询结构所需字段。"""
        query_tokens = self._tokens(query)
        required_columns = {value.casefold() for value in required_columns or set()}
        planned: list[dict] = []
        total_before = 0
        total_after = 0

        for candidate in candidates:
            item = copy.copy(candidate)
            schema = copy.deepcopy(candidate.get("schema", {}))
            columns = schema.get("columns", [])
            total_before += len(columns)
            table_name = candidate["table_name"]
            short_name = table_name.rsplit(".", 1)[-1]
            matched_columns = candidate.get("matched_columns", [])
            matched_rank = {
                (
                    entry.get("column_name") if isinstance(entry, dict) else str(entry)
                ).casefold(): rank
                for rank, entry in enumerate(matched_columns)
            }
            mandatory = self.relation_columns.get(table_name, set()).copy()
            if preserve_time_columns:
                mandatory.update(
                    self._select_temporal_columns(columns, query_tokens, limit=3)
                )
            scored: list[tuple[float, int, dict]] = []

            for position, column in enumerate(columns):
                if column.get("is_sensitive"):
                    continue
                column_name = str(column.get("name") or "")
                folded_name = column_name.casefold()
                qualified_names = {
                    folded_name,
                    f"{short_name}.{column_name}".casefold(),
                    f"{table_name}.{column_name}".casefold(),
                }
                key_marker = str(column.get("key") or "").strip().upper()
                is_key = key_marker in {
                    "PRI",
                    "UNI",
                    "MUL",
                    "TRUE",
                    "YES",
                    "KEY",
                } or key_marker.endswith(" KEY")
                is_required = bool(qualified_names & required_columns)
                is_mandatory = is_key or is_required or folded_name in mandatory
                if is_mandatory:
                    mandatory.add(folded_name)
                    column["_context_required"] = True

                text = " ".join(
                    str(column.get(key) or "")
                    for key in (
                        "name",
                        "display_name",
                        "comment",
                        "description",
                        "business_logic",
                    )
                )
                overlap = len(query_tokens & self._tokens(text))
                score = float(overlap * 2)
                if folded_name in matched_rank:
                    score += max(1.0, 8.0 - matched_rank[folded_name] * 0.25)
                if is_mandatory:
                    score += 100.0
                if column.get("enum_values"):
                    score += 0.25
                if column.get("is_skip_index") and not is_mandatory:
                    score -= 100.0
                scored.append((score, -position, column))

            scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
            selected = [entry[2] for entry in scored[:per_table_limit]]
            selected_names = {
                str(column.get("name") or "").casefold() for column in selected
            }
            for _, _, column in scored[per_table_limit:]:
                name = str(column.get("name") or "").casefold()
                if name in mandatory and name not in selected_names:
                    selected.append(column)
                    selected_names.add(name)

            # 小表保持完整；大表按原字段顺序输出，降低模型阅读成本。
            if len(columns) <= per_table_limit:
                selected = [
                    column
                    for column in columns
                    if not column.get("is_sensitive")
                    and (
                        not column.get("is_skip_index")
                        or column.get("_context_required")
                    )
                ]
            else:
                original_order = {
                    str(column.get("name") or "").casefold(): index
                    for index, column in enumerate(columns)
                }
                selected.sort(
                    key=lambda column: original_order.get(
                        str(column.get("name") or "").casefold(), 10**9
                    )
                )

            schema["columns"] = selected
            schema["original_column_count"] = len(columns)
            schema["selected_column_names"] = [
                column.get("name", "") for column in selected
            ]
            item["schema"] = schema
            item["selected_columns"] = schema["selected_column_names"]
            planned.append(item)
            total_after += len(selected)

        stats = {
            "table_count": len(planned),
            "column_count_before": total_before,
            "column_count_after": total_after,
            "columns_pruned": max(0, total_before - total_after),
        }
        return planned, stats

    @classmethod
    def _select_temporal_columns(
        cls,
        columns: list[dict],
        query_tokens: set[str],
        limit: int,
    ) -> set[str]:
        """为时间过滤/分组保留少量最可能使用的日期时间字段。"""
        ranked: list[tuple[float, int, str]] = []
        preferred_names = {
            "event_time": 10,
            "transaction_time": 10,
            "order_time": 10,
            "payment_time": 10,
            "create_time": 9,
            "created_at": 9,
            "completed_at": 7,
            "business_date": 7,
            "trade_date": 7,
            "updated_at": 3,
            "update_time": 3,
            "deleted_at": 1,
        }
        for position, column in enumerate(columns):
            name = str(column.get("name") or "").casefold()
            data_type = str(column.get("type") or "").casefold()
            name_looks_temporal = bool(
                re.search(r"(?:^|_)(?:date|time|day|month|year)(?:_|$)|_at$", name)
            )
            type_is_temporal = any(
                marker in data_type
                for marker in ("date", "time", "timestamp", "datetime")
            )
            if not (name_looks_temporal or type_is_temporal):
                continue
            text = " ".join(
                str(column.get(key) or "")
                for key in ("name", "display_name", "comment", "description")
            )
            score = float(preferred_names.get(name, 5 if type_is_temporal else 1))
            score += 2 * len(query_tokens & cls._tokens(text))
            ranked.append((score, -position, name))
        ranked.sort(reverse=True)
        return {name for _, _, name in ranked[:limit]}

    def _build_relation_graph(self) -> dict[str, set[str]]:
        graph = {table_name: set() for table_name in self.table_schemas}
        for source, schema in self.table_schemas.items():
            for relation in schema.get("relations", []):
                target = self.resolve_table(
                    str(relation.get("target_table") or ""), source_table=source
                )
                if target and target != source:
                    graph[source].add(target)
                    graph[target].add(source)
        return graph

    def _build_relation_columns(self) -> dict[str, set[str]]:
        columns: dict[str, set[str]] = {
            table_name: set() for table_name in self.table_schemas
        }
        for source, schema in self.table_schemas.items():
            for relation in schema.get("relations", []):
                source_column = str(relation.get("column") or "").casefold()
                target_column = str(relation.get("target_column") or "").casefold()
                target = self.resolve_table(
                    str(relation.get("target_table") or ""), source_table=source
                )
                if source_column:
                    columns[source].add(source_column)
                if target and target_column:
                    columns[target].add(target_column)
        return columns

    def _build_edge_costs(self) -> dict[tuple[str, str], float]:
        """按关系基数给边设置代价，优先稳定的一对一/多对一关系。"""
        costs: dict[tuple[str, str], float] = {}
        cardinality_cost = {
            "one_to_one": 1.0,
            "many_to_one": 1.0,
            "one_to_many": 1.15,
            "unknown": 1.25,
            "many_to_many": 1.8,
        }
        for source, schema in self.table_schemas.items():
            for relation in schema.get("relations", []):
                target = self.resolve_table(
                    str(relation.get("target_table") or ""), source_table=source
                )
                if not target:
                    continue
                cost = cardinality_cost.get(
                    str(relation.get("cardinality") or "unknown"), 1.25
                )
                for edge in ((source, target), (target, source)):
                    costs[edge] = min(costs.get(edge, cost), cost)
        return costs

    def _shortest_path(
        self, source: str, target: str, max_hops: int
    ) -> list[str] | None:
        if source == target:
            return [source]
        queue = [(0.0, 0, source, [source])]
        best_cost: dict[tuple[str, int], float] = {(source, 0): 0.0}
        while queue:
            cost, hops, current, path = heapq.heappop(queue)
            if current == target:
                return path
            if hops >= max_hops:
                continue
            for neighbor in sorted(self.graph.get(current, set())):
                if neighbor in path:
                    continue
                next_path = [*path, neighbor]
                next_hops = hops + 1
                next_cost = cost + self.edge_costs.get((current, neighbor), 1.25)
                state = (neighbor, next_hops)
                if next_cost >= best_cost.get(state, float("inf")):
                    continue
                best_cost[state] = next_cost
                heapq.heappush(queue, (next_cost, next_hops, neighbor, next_path))
        return None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        if not text:
            return set()
        try:
            import jieba

            chinese_tokens = {
                token.casefold()
                for token in jieba.lcut(text)
                if len(token.strip()) >= 2
            }
        except ImportError:
            chinese_tokens = set()
        lexical_tokens = {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text)
        }
        return chinese_tokens | lexical_tokens
