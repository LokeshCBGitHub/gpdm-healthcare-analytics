from __future__ import annotations
import hashlib
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BASE = (os.environ.get('GPDM_BASE_DIR', '') or
         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataSource(ABC):

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> List[Dict]:
        pass

    @abstractmethod
    def execute_scalar(self, sql: str, params: tuple = ()) -> Any:
        pass

    @abstractmethod
    def stream(self, sql: str, params: tuple = (),
               batch_size: int = 10000) -> Iterator[List[Dict]]:
        pass

    @abstractmethod
    def health_check(self) -> Dict:
        pass

    @abstractmethod
    def table_stats(self, table: str) -> Dict:
        pass


class SQLiteSource(DataSource):

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA cache_size=-200000')
            self._conn.execute('PRAGMA mmap_size=1073741824')
            self._conn.execute('PRAGMA temp_store=MEMORY')
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def execute_scalar(self, sql: str, params: tuple = ()) -> Any:
        conn = self._get_conn()
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def stream(self, sql: str, params: tuple = (),
               batch_size: int = 10000) -> Iterator[List[Dict]]:
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [dict(zip(cols, row)) for row in rows]

    def health_check(self) -> Dict:
        try:
            conn = self._get_conn()
            n = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
            size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            return {'status': 'healthy', 'backend': 'sqlite', 'tables': n,
                    'size_mb': round(size_mb, 1)}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def table_stats(self, table: str) -> Dict:
        try:
            conn = self._get_conn()
            n = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            return {'table': table, 'rows': n, 'backend': 'sqlite'}
        except Exception as e:
            return {'error': str(e)}


class DatabricksSource(DataSource):

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._connector = None

    def _get_connector(self):
        if self._connector is None:
            try:
                from databricks_connector import DatabricksConnector
                self._connector = DatabricksConnector(self.config)
            except Exception as e:
                logger.warning("Databricks not available: %s", e)
                raise
        return self._connector

    def execute(self, sql: str, params: tuple = ()) -> List[Dict]:
        conn = self._get_connector()
        return conn.execute_query(sql)

    def execute_scalar(self, sql: str, params: tuple = ()) -> Any:
        rows = self.execute(sql, params)
        if rows and rows[0]:
            return list(rows[0].values())[0]
        return None

    def stream(self, sql: str, params: tuple = (),
               batch_size: int = 10000) -> Iterator[List[Dict]]:
        results = self.execute(sql, params)
        for i in range(0, len(results), batch_size):
            yield results[i:i + batch_size]

    def health_check(self) -> Dict:
        try:
            conn = self._get_connector()
            return conn.health_check()
        except Exception as e:
            return {'status': 'unavailable', 'error': str(e)}

    def table_stats(self, table: str) -> Dict:
        try:
            rows = self.execute(f"DESCRIBE DETAIL {table}")
            if rows:
                return {
                    'table': table,
                    'rows': rows[0].get('numRows', -1),
                    'size_bytes': rows[0].get('sizeInBytes', 0),
                    'partitions': rows[0].get('numPartitions', 0),
                    'backend': 'databricks',
                }
        except Exception:
            pass
        return {'table': table, 'backend': 'databricks'}


class SmartCache:

    def __init__(self, max_entries: int = 500, default_ttl: float = 300.0):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._access_counts: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, sql: str, params: tuple = ()) -> str:
        return hashlib.md5(f"{sql}|{params}".encode()).hexdigest()

    def get(self, sql: str, params: tuple = ()) -> Optional[Any]:
        key = self._key(sql, params)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['time'] < entry['ttl']:
                self._hits += 1
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                entry['ttl'] = min(
                    entry['ttl'] * 1.2,
                    3600.0 if self._access_counts[key] > 10 else self.default_ttl)
                return entry['result']
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, sql: str, result: Any, params: tuple = (),
            ttl: float = None):
        key = self._key(sql, params)
        if len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        self._cache[key] = {
            'result': result,
            'time': time.time(),
            'ttl': ttl or self.default_ttl,
        }

    def invalidate_pattern(self, pattern: str):
        to_remove = [k for k, v in self._cache.items()
                     if pattern in str(v.get('sql', ''))]
        for k in to_remove:
            del self._cache[k]

    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            'entries': len(self._cache),
            'max_entries': self.max_entries,
            'hit_rate': round(self._hits / total, 3) if total > 0 else 0,
            'hits': self._hits,
            'misses': self._misses,
        }


class QueryPlanner:

    MATERIALIZED_TABLES = {
        '_gpdm_summary_monthly', '_gpdm_summary_region',
        '_gpdm_summary_visit_type', '_gpdm_summary_provider',
        '_gpdm_summary_denial', '_gpdm_summary_member_util',
        '_gpdm_kpi_facts', '_gpdm_member_risk', '_gpdm_provider_scores',
        '_gpdm_hybrid_scores',
    }

    def __init__(self):
        self.stats = {'routed_local': 0, 'routed_remote': 0,
                      'cache_hits': 0, 'total_queries': 0}

    def plan(self, sql: str, local: DataSource,
             remote: Optional[DataSource] = None) -> Tuple[DataSource, str]:
        self.stats['total_queries'] += 1
        sql_lower = sql.lower().strip()

        for table in self.MATERIALIZED_TABLES:
            if table in sql_lower:
                self.stats['routed_local'] += 1
                return local, sql

        if 'limit' in sql_lower and self._extract_limit(sql_lower) <= 1000:
            self.stats['routed_local'] += 1
            return local, sql

        is_heavy = ('claims' in sql_lower or 'encounters' in sql_lower) and \
                   any(kw in sql_lower for kw in ['group by', 'count(', 'sum(', 'avg('])
        if is_heavy and remote is not None:
            health = remote.health_check()
            if health.get('status') in ('healthy', 'ok'):
                self.stats['routed_remote'] += 1
                optimized = self._optimize_for_databricks(sql)
                return remote, optimized

        self.stats['routed_local'] += 1
        return local, sql

    def _extract_limit(self, sql: str) -> int:
        import re
        m = re.search(r'limit\s+(\d+)', sql)
        return int(m.group(1)) if m else 999999

    def _optimize_for_databricks(self, sql: str) -> str:
        result = sql
        result = result.replace('IFNULL(', 'COALESCE(')
        result = result.replace("date('now')", 'current_date()')
        result = result.replace("datetime('now')", 'current_timestamp()')
        return result


class PartitionManager:

    def __init__(self):
        self.partition_cols = {
            'claims_4m': ['service_date', 'region', 'status'],
            'encounters': ['ADMIT_DATE', 'VISIT_TYPE'],
            'members': ['REGION', 'LOB'],
        }

    def add_partition_filter(self, sql: str, table: str,
                              filters: Dict[str, Any]) -> str:
        if table not in self.partition_cols:
            return sql

        clauses = []
        for col, val in filters.items():
            if col in self.partition_cols[table]:
                if isinstance(val, (list, tuple)):
                    quoted = ','.join(f"'{v}'" for v in val)
                    clauses.append(f"{col} IN ({quoted})")
                elif isinstance(val, dict):
                    if 'gte' in val:
                        clauses.append(f"{col} >= '{val['gte']}'")
                    if 'lte' in val:
                        clauses.append(f"{col} <= '{val['lte']}'")
                else:
                    clauses.append(f"{col} = '{val}'")

        if not clauses:
            return sql

        sql_lower = sql.lower()
        if 'where' in sql_lower:
            insert_pos = sql_lower.rfind('group by')
            if insert_pos == -1:
                insert_pos = sql_lower.rfind('order by')
            if insert_pos == -1:
                insert_pos = sql_lower.rfind('limit')
            if insert_pos == -1:
                insert_pos = len(sql)
            return sql[:insert_pos] + ' AND ' + ' AND '.join(clauses) + ' ' + sql[insert_pos:]
        else:
            insert_pos = sql_lower.find('group by')
            if insert_pos == -1:
                insert_pos = sql_lower.find('order by')
            if insert_pos == -1:
                insert_pos = sql_lower.find('limit')
            if insert_pos == -1:
                insert_pos = len(sql)
            return sql[:insert_pos] + ' WHERE ' + ' AND '.join(clauses) + ' ' + sql[insert_pos:]

    def estimate_scan_reduction(self, table: str,
                                 filters: Dict[str, Any]) -> float:
        if table not in self.partition_cols:
            return 1.0
        n_filters = sum(1 for k in filters if k in self.partition_cols[table])
        return max(0.01, 0.5 ** n_filters)


class StreamProcessor:

    def __init__(self, engine: 'ScaleEngine'):
        self.engine = engine
        self.processed_batches = 0
        self.total_rows_processed = 0

    REQUIRED_FIELDS = {
        'claims':     {'claim_id', 'member_id', 'service_date', 'billed_amount'},
        'members':    {'member_id'},
        'providers':  {'npi'},
        'encounters': {'encounter_id', 'member_id', 'service_date', 'rendering_npi', 'visit_type'},
    }

    TABLE_MAP = {
        'claims':     'claims_4m',
        'members':    'members',
        'providers':  'providers',
        'encounters': 'encounters',
    }

    KNOWN_COLUMNS = {
        'claims':     {'claim_id', 'member_id', 'provider_id', 'facility_id', 'diagnosis_code',
                       'encounter_type', 'region', 'billed_amount', 'paid_amount', 'status',
                       'service_date', 'denial_reason'},
        'members':    {'member_id', 'region', 'facility', 'pcp_npi', 'enrollment_date',
                       'last_service_date', 'gender', 'age', 'risk_score', 'chronic_conditions',
                       'plan_type', 'lob'},
        'providers':  {'npi', 'specialty', 'department', 'region', 'facility', 'provider_type',
                       'status', 'panel_size', 'accepts_new_patients'},
        'encounters': {'encounter_id', 'member_id', 'service_date', 'encounter_date',
                       'rendering_npi', 'region', 'primary_diagnosis', 'facility', 'paid_amount',
                       'billed_amount', 'visit_type', 'admit_date', 'discharge_date',
                       'length_of_stay', 'encounter_status'},
    }

    BATCH_SIZE = 5000

    def ingest(self, table: str, rows: List[Dict]) -> Dict:
        if not rows or not isinstance(rows, list):
            return {'rows_accepted': 0, 'rows_rejected': 0,
                    'validation_errors': ['No rows provided'], 'summaries_updated': False, 'models_updated': False}

        if table not in self.TABLE_MAP:
            return {'rows_accepted': 0, 'rows_rejected': len(rows),
                    'validation_errors': [f'Unknown table: {table}. Must be one of: {", ".join(self.TABLE_MAP)}'],
                    'summaries_updated': False, 'models_updated': False}

        t0 = time.time()
        db_table = self.TABLE_MAP[table]
        required = self.REQUIRED_FIELDS[table]
        known = self.KNOWN_COLUMNS[table]

        accepted_rows = []
        errors = []

        for i, row in enumerate(rows):
            normalized = {k.lower().strip(): v for k, v in row.items()}

            missing = required - set(normalized.keys())
            if missing:
                errors.append(f"Row {i}: missing required fields: {', '.join(sorted(missing))}")
                continue

            clean = {k: normalized[k] for k in normalized if k in known}
            accepted_rows.append(clean)

        if not accepted_rows:
            return {'rows_accepted': 0, 'rows_rejected': len(rows),
                    'validation_errors': errors, 'summaries_updated': False, 'models_updated': False}

        source = self.engine.local
        conn = source._get_conn() if hasattr(source, '_get_conn') else None
        inserted = 0
        duplicates = 0

        if conn:
            id_col = self._id_column(table)

            for batch_start in range(0, len(accepted_rows), self.BATCH_SIZE):
                batch = accepted_rows[batch_start:batch_start + self.BATCH_SIZE]
                cols = list(batch[0].keys())
                placeholders = ','.join(['?'] * len(cols))
                col_str = ','.join(cols)

                if id_col:
                    batch_ids = [r[id_col] for r in batch if id_col in r]
                    if batch_ids:
                        existing = set()
                        for chunk_start in range(0, len(batch_ids), 500):
                            chunk = batch_ids[chunk_start:chunk_start + 500]
                            ph = ','.join(['?'] * len(chunk))
                            found = conn.execute(
                                f"SELECT {id_col} FROM {db_table} WHERE {id_col} IN ({ph})",
                                chunk).fetchall()
                            existing.update(r[0] for r in found)
                        new_rows = [r for r in batch if r.get(id_col) not in existing]
                        duplicates += len(batch) - len(new_rows)
                        batch = new_rows

                if batch:
                    try:
                        conn.executemany(
                            f"INSERT INTO {db_table} ({col_str}) VALUES ({placeholders})",
                            [tuple(r.get(c) for c in cols) for r in batch])
                        conn.commit()
                        inserted += len(batch)
                    except Exception as e:
                        logger.warning("Batch insert failed for %s: %s", db_table, e)
                        errors.append(f"Batch insert error: {str(e)}")

        summaries_updated = False
        if table == 'claims' and inserted > 0:
            try:
                summaries_updated = self._refresh_summaries_incremental(conn, accepted_rows)
            except Exception as e:
                logger.warning("Summary refresh failed: %s", e)

        self.processed_batches += 1
        self.total_rows_processed += inserted

        result = {
            'rows_accepted': inserted,
            'rows_rejected': len(rows) - inserted,
            'duplicates_skipped': duplicates,
            'validation_errors': errors,
            'summaries_updated': summaries_updated,
            'models_updated': False,
            'time_seconds': round(time.time() - t0, 2),
            'rows_per_second': round(inserted / max(time.time() - t0, 0.001)),
        }
        return result

    def ingest_claims_batch(self, rows: List[Dict]) -> Dict:
        raw = self.ingest('claims', rows)
        return {
            'status': 'ok' if raw['rows_accepted'] > 0 else 'error',
            'rows_ingested': raw['rows_accepted'],
            'batch_number': self.processed_batches,
            'total_processed': self.total_rows_processed,
            'summary_updates': {'summaries_updated': raw['summaries_updated']},
            'time_ms': round(raw.get('time_seconds', 0) * 1000, 1),
        }

    @staticmethod
    def _id_column(table: str) -> Optional[str]:
        return {'claims': 'claim_id', 'members': 'member_id',
                'providers': 'npi', 'encounters': 'encounter_id'}.get(table)

    def _refresh_summaries_incremental(self, conn, rows: List[Dict]) -> bool:
        if not conn:
            return False

        from collections import Counter

        months_touched = set()
        for r in rows:
            month = str(r.get('service_date', ''))[:7]
            if month and len(month) == 7:
                months_touched.add(month)

        if not months_touched:
            return False

        for month in months_touched:
            try:
                conn.execute("DELETE FROM _gpdm_summary_monthly WHERE month = ?", (month,))
                conn.execute("""
                    INSERT INTO _gpdm_summary_monthly
                        (month, region, claim_status, visit_type,
                         claim_count, denied_count, total_billed, total_paid,
                         avg_paid, avg_billed, unique_members, unique_providers,
                         has_denial_reason)
                    SELECT
                        substr(service_date, 1, 7) as month,
                        COALESCE(region, 'Unknown') as region,
                        COALESCE(status, 'Unknown') as claim_status,
                        COALESCE(encounter_type, 'ALL') as visit_type,
                        COUNT(*) as claim_count,
                        SUM(CASE WHEN UPPER(status) = 'DENIED' THEN 1 ELSE 0 END) as denied_count,
                        ROUND(SUM(CAST(billed_amount AS REAL)), 2) as total_billed,
                        ROUND(SUM(CAST(paid_amount AS REAL)), 2) as total_paid,
                        ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                        ROUND(AVG(CAST(billed_amount AS REAL)), 2) as avg_billed,
                        COUNT(DISTINCT member_id) as unique_members,
                        COUNT(DISTINCT provider_id) as unique_providers,
                        SUM(CASE WHEN denial_reason IS NOT NULL AND denial_reason != '' THEN 1 ELSE 0 END) as has_denial_reason
                    FROM claims_4m
                    WHERE substr(service_date, 1, 7) = ?
                    GROUP BY month, region, claim_status, visit_type
                """, (month,))
            except Exception as e:
                logger.warning("Summary refresh for %s failed: %s", month, e)

        try:
            conn.execute("DELETE FROM _gpdm_kpi_facts")
            conn.execute("""
                INSERT INTO _gpdm_kpi_facts
                    (total_claims, total_members, total_providers, total_paid, total_billed,
                     avg_paid, total_denied, denial_rate, earliest_date, latest_date, total_months)
                SELECT
                    COUNT(*),
                    (SELECT COUNT(*) FROM members),
                    (SELECT COUNT(DISTINCT provider_id) FROM claims_4m),
                    ROUND(SUM(CAST(paid_amount AS REAL)), 2),
                    ROUND(SUM(CAST(billed_amount AS REAL)), 2),
                    ROUND(AVG(CAST(paid_amount AS REAL)), 2),
                    SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END),
                    ROUND(SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END)*100.0
                          / NULLIF(COUNT(*), 0), 2),
                    MIN(service_date),
                    MAX(service_date),
                    COUNT(DISTINCT substr(service_date, 1, 7))
                FROM claims_4m
            """)
        except Exception as e:
            logger.warning("KPI facts refresh failed: %s", e)

        conn.commit()
        return True


class ScaleEngine:

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(_BASE, 'data', 'healthcare_production.db')
        self.local = SQLiteSource(self.db_path)
        self.remote: Optional[DatabricksSource] = None
        self.cache = SmartCache(max_entries=500, default_ttl=300)
        self.planner = QueryPlanner()
        self.partitions = PartitionManager()
        self.stream_processor = StreamProcessor(self)

    def connect_databricks(self, config: Dict):
        self.remote = DatabricksSource(config)
        health = self.remote.health_check()
        logger.info("Databricks connected: %s", health)

    def query(self, sql: str, params: tuple = (),
              use_cache: bool = True,
              partition_filters: Dict = None) -> List[Dict]:
        if partition_filters:
            for table in self.partitions.partition_cols:
                if table in sql.lower():
                    sql = self.partitions.add_partition_filter(
                        sql, table, partition_filters)
                    break

        if use_cache:
            cached = self.cache.get(sql, params)
            if cached is not None:
                self.planner.stats['cache_hits'] += 1
                return cached

        source, optimized_sql = self.planner.plan(
            sql, self.local, self.remote)

        result = source.execute(optimized_sql, params)

        if use_cache and len(result) < 50000:
            self.cache.put(sql, result, params)

        return result

    def scalar(self, sql: str, params: tuple = ()) -> Any:
        rows = self.query(sql, params)
        if rows and rows[0]:
            return list(rows[0].values())[0]
        return None

    def stream(self, sql: str, params: tuple = (),
               batch_size: int = 10000) -> Iterator[List[Dict]]:
        source, optimized_sql = self.planner.plan(
            sql, self.local, self.remote)
        return source.stream(optimized_sql, params, batch_size)

    def ingest(self, rows: List[Dict]) -> Dict:
        return self.stream_processor.ingest_claims_batch(rows)

    def health(self) -> Dict:
        result = {
            'local': self.local.health_check(),
            'cache': self.cache.stats(),
            'routing': self.planner.stats,
        }
        if self.remote:
            result['remote'] = self.remote.health_check()
        return result
