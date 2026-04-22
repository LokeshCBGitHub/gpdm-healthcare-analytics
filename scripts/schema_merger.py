from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger('gpdm.schema_merger')


@dataclass
class MergePlan:
    action: str
    source_table: str
    target_table: str
    column_map: Dict[str, str] = field(default_factory=dict)
    new_columns: List[Dict[str, str]] = field(default_factory=list)
    join_candidates: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ''
    dq_checks: List[str] = field(default_factory=list)


@dataclass
class MergeReport:
    action: str
    source_table: str
    target_table: str
    rows_inserted: int = 0
    columns_added: List[str] = field(default_factory=list)
    indexes_created: List[str] = field(default_factory=list)
    joins_discovered: List[str] = field(default_factory=list)
    dq_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


class SchemaMerger:

    def __init__(self, db_path: str):
        self.db_path = db_path

    def plan_merge(self, new_table: str,
                   new_columns: List[Dict],
                   mapping_result: Any = None,
                   existing_schema: Dict = None) -> MergePlan:
        if existing_schema is None:
            existing_schema = self._get_existing_schema()

        best_match = None
        best_similarity = 0.0

        for ex_table, ex_cols in existing_schema.items():
            sim = self._table_similarity(new_columns, ex_cols, mapping_result)
            if sim > best_similarity:
                best_similarity = sim
                best_match = ex_table

        logger.info("Best match for '%s': '%s' (similarity=%.2f)",
                    new_table, best_match, best_similarity)

        if best_similarity >= 0.85:
            col_overlap = self._column_overlap(new_columns, existing_schema.get(best_match, []))
            if col_overlap >= 0.9:
                col_map = self._build_column_map(new_columns, existing_schema[best_match], mapping_result)
                return MergePlan(
                    action='append',
                    source_table=new_table,
                    target_table=best_match,
                    column_map=col_map,
                    confidence=best_similarity,
                    reasoning=f"Column overlap {col_overlap:.0%} with '{best_match}' — appending rows",
                    dq_checks=['null_check', 'type_check', 'range_check'],
                )
            else:
                new_cols = self._find_new_columns(new_columns, existing_schema[best_match])
                col_map = self._build_column_map(new_columns, existing_schema[best_match], mapping_result)
                return MergePlan(
                    action='extend',
                    source_table=new_table,
                    target_table=best_match,
                    column_map=col_map,
                    new_columns=new_cols,
                    confidence=best_similarity,
                    reasoning=f"Same entity as '{best_match}' with {len(new_cols)} new columns",
                    dq_checks=['null_check', 'type_check', 'range_check', 'key_check'],
                )
        else:
            joins = self._discover_join_candidates(new_columns, existing_schema, mapping_result)
            return MergePlan(
                action='create',
                source_table=new_table,
                target_table=new_table,
                join_candidates=joins,
                confidence=0.5,
                reasoning=f"New entity (best match '{best_match}' only {best_similarity:.0%})",
                dq_checks=['null_check', 'type_check', 'completeness_check'],
            )

    def execute_merge(self, plan: MergePlan) -> MergeReport:
        t0 = time.time()
        report = MergeReport(
            action=plan.action,
            source_table=plan.source_table,
            target_table=plan.target_table,
        )

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                if plan.action == 'append':
                    report = self._execute_append(conn, plan, report)
                elif plan.action == 'extend':
                    report = self._execute_extend(conn, plan, report)
                elif plan.action == 'create':
                    report = self._execute_create(conn, plan, report)
                conn.commit()
            except Exception as e:
                conn.rollback()
                report.errors.append(str(e))
                logger.error("Merge failed: %s", e)
            finally:
                conn.close()
        except Exception as e:
            report.errors.append(f"DB connection failed: {e}")

        report.duration_ms = round((time.time() - t0) * 1000)
        logger.info("Merge %s: %s → %s (%d rows, %dms)",
                    plan.action, plan.source_table, plan.target_table,
                    report.rows_inserted, report.duration_ms)
        return report

    def _execute_append(self, conn: sqlite3.Connection,
                        plan: MergePlan, report: MergeReport) -> MergeReport:
        if not plan.column_map:
            report.errors.append("No column mapping for append")
            return report

        src_cols = ', '.join(f'"{k}"' for k in plan.column_map.keys())
        tgt_cols = ', '.join(f'"{v}"' for v in plan.column_map.values())

        sql = (f'INSERT INTO "{plan.target_table}" ({tgt_cols}) '
               f'SELECT {src_cols} FROM "{plan.source_table}"')
        try:
            cursor = conn.execute(sql)
            report.rows_inserted = cursor.rowcount
        except sqlite3.Error as e:
            report.errors.append(f"Append failed: {e}")

        return report

    def _execute_extend(self, conn: sqlite3.Connection,
                        plan: MergePlan, report: MergeReport) -> MergeReport:
        for col_def in plan.new_columns:
            col_name = col_def.get('name', '')
            col_type = col_def.get('type', 'TEXT')
            try:
                conn.execute(f'ALTER TABLE "{plan.target_table}" ADD COLUMN "{col_name}" {col_type}')
                report.columns_added.append(col_name)
            except sqlite3.OperationalError as e:
                if 'duplicate column' not in str(e).lower():
                    report.errors.append(f"Add column {col_name}: {e}")

        if plan.column_map:
            report = self._execute_append(conn, plan, report)

        return report

    def _execute_create(self, conn: sqlite3.Connection,
                        plan: MergePlan, report: MergeReport) -> MergeReport:
        for join in plan.join_candidates:
            col = join.get('column', '')
            idx_name = f"ix_gpdm_{plan.target_table}_{col}"
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} '
                            f'ON "{plan.target_table}" ("{col}")')
                report.indexes_created.append(idx_name)
            except sqlite3.Error as e:
                report.errors.append(f"Index {idx_name}: {e}")
            report.joins_discovered.append(
                f"{plan.target_table}.{col} → {join.get('target_table', '?')}.{join.get('target_column', '?')}"
            )

        try:
            row = conn.execute(f'SELECT COUNT(*) FROM "{plan.target_table}"').fetchone()
            report.rows_inserted = row[0] if row else 0
        except sqlite3.Error:
            pass

        return report


    def _get_existing_schema(self) -> Dict[str, List[str]]:
        schema = {}
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'gpdm_%'"
                ).fetchall()
                for (table,) in tables:
                    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                    schema[table] = [c[1] for c in cols]
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to read existing schema: %s", e)
        return schema

    def _table_similarity(self, new_columns: List[Dict],
                          existing_cols: List[str],
                          mapping_result: Any = None) -> float:
        if not new_columns or not existing_cols:
            return 0.0

        new_names = {c.get('name', '').lower() for c in new_columns}
        ex_names = {c.lower() for c in existing_cols}

        overlap = len(new_names & ex_names)
        union = len(new_names | ex_names)

        jaccard = overlap / union if union > 0 else 0

        if mapping_result and hasattr(mapping_result, 'auto_approved'):
            concept_matches = sum(1 for m in mapping_result.auto_approved
                                  if m.canonical_name.lower() in ex_names)
            concept_boost = concept_matches * 0.05
            jaccard = min(1.0, jaccard + concept_boost)

        return jaccard

    def _column_overlap(self, new_columns: List[Dict],
                        existing_cols: List[str]) -> float:
        new_names = {c.get('name', '').lower() for c in new_columns}
        ex_names = {c.lower() for c in existing_cols}
        if not ex_names:
            return 0.0
        return len(new_names & ex_names) / len(ex_names)

    def _build_column_map(self, new_columns: List[Dict],
                          existing_cols: List[str],
                          mapping_result: Any = None) -> Dict[str, str]:
        col_map = {}
        ex_lower = {c.lower(): c for c in existing_cols}

        for col in new_columns:
            name = col.get('name', '')
            if name.lower() in ex_lower:
                col_map[name] = ex_lower[name.lower()]
            elif mapping_result:
                for m in getattr(mapping_result, 'mappings', []):
                    if m.source_column == name and m.canonical_name.lower() in ex_lower:
                        col_map[name] = ex_lower[m.canonical_name.lower()]
                        break

        return col_map

    def _find_new_columns(self, new_columns: List[Dict],
                          existing_cols: List[str]) -> List[Dict[str, str]]:
        ex_lower = {c.lower() for c in existing_cols}
        new_cols = []
        for col in new_columns:
            name = col.get('name', '')
            if name.lower() not in ex_lower:
                dtype = col.get('dtype', 'TEXT')
                sql_type = 'REAL' if dtype in ('numeric', 'float') else \
                           'INTEGER' if dtype == 'integer' else 'TEXT'
                new_cols.append({'name': name, 'type': sql_type})
        return new_cols

    def _discover_join_candidates(self, new_columns: List[Dict],
                                  existing_schema: Dict,
                                  mapping_result: Any = None) -> List[Dict[str, str]]:
        candidates = []
        id_concepts = {'member_id', 'claim_id', 'encounter_id', 'provider_id'}

        for col in new_columns:
            name = col.get('name', '').lower()
            is_id = any(idc in name for idc in ['_id', '_key', '_no', 'npi'])

            if mapping_result:
                for m in getattr(mapping_result, 'mappings', []):
                    if m.source_column == col.get('name', '') and m.concept in id_concepts:
                        is_id = True
                        break

            if is_id:
                for ex_table, ex_cols in existing_schema.items():
                    for ex_col in ex_cols:
                        if ex_col.lower() == name or (
                            name.replace('_', '') == ex_col.lower().replace('_', '')
                        ):
                            candidates.append({
                                'column': col.get('name', ''),
                                'target_table': ex_table,
                                'target_column': ex_col,
                            })

        return candidates
