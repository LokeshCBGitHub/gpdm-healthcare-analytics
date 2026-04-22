import logging
import os
import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Queue
import urllib.parse

try:
    from databricks import sql as databricks_sql
    HAS_DATABRICKS_SQL = True
except ImportError:
    HAS_DATABRICKS_SQL = False
    logger = logging.getLogger(__name__)
    logger.warning("databricks-sql-connector not installed. Using REST API fallback.")

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

try:
    from gpdm_config import (
        DEFAULT_TIMEOUT,
        DEFAULT_MAX_RETRIES,
    )
except ImportError:
    DEFAULT_TIMEOUT = 120
    DEFAULT_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query: str
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_ms: float
    timestamp: str


@dataclass
class TableStats:
    catalog: str
    schema: str
    table: str
    row_count: int
    size_bytes: int
    last_modified: str
    format: str


class UnityCatalogConnector:

    def __init__(self,
                 host: Optional[str] = None,
                 token: Optional[str] = None,
                 warehouse_id: Optional[str] = None,
                 http_path: Optional[str] = None,
                 catalog: Optional[str] = None,
                 schema: Optional[str] = None,
                 timeout_sec: int = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.host = host or os.environ.get('DATABRICKS_HOST', '').strip()
        self.token = token or os.environ.get('DATABRICKS_TOKEN', '').strip()
        self.warehouse_id = warehouse_id or os.environ.get('DATABRICKS_WAREHOUSE_ID', '').strip()
        self.http_path = http_path or os.environ.get('DATABRICKS_HTTP_PATH', '').strip()
        self.catalog = catalog or os.environ.get('DATABRICKS_CATALOG', 'hive_metastore')
        self.schema = schema or os.environ.get('DATABRICKS_SCHEMA', 'default')
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

        self.connection = None
        self._connected = False
        self._lock = threading.Lock()

        if not self.host or not self.token:
            logger.warning("DATABRICKS_HOST or DATABRICKS_TOKEN not set. "
                          "Some operations will fail.")

    def connect(self) -> bool:
        if self._connected:
            return True

        with self._lock:
            try:
                if HAS_DATABRICKS_SQL:
                    self.connection = databricks_sql.connect(
                        server_hostname=self.host,
                        http_path=self.http_path,
                        personal_access_token=self.token,
                        session_configuration={
                            "ansi_mode": "false",
                            "sql_mode": "auto",
                        }
                    )
                    self._connected = True
                    logger.info(f"Connected to Databricks: {self.host}")
                    return True
                else:
                    logger.warning("databricks-sql-connector not installed. "
                                 "Using REST API fallback (limited functionality).")
                    self._connected = True
                    return True
            except Exception as e:
                logger.error(f"Failed to connect to Databricks: {e}")
                self._connected = False
                return False

    def disconnect(self) -> None:
        if self.connection and HAS_DATABRICKS_SQL:
            try:
                self.connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        self._connected = False

    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None,
                     timeout_sec: Optional[int] = None) -> QueryResult:
        if not self._connected:
            self.connect()

        if not self._connected:
            raise RuntimeError("Not connected to Databricks")

        timeout_sec = timeout_sec or self.timeout_sec
        start = time.time()

        try:
            if HAS_DATABRICKS_SQL:
                cursor = self.connection.cursor()
                cursor.execute(sql, params or {})
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                cursor.close()

                row_dicts = [dict(zip(columns, row)) for row in rows]

                return QueryResult(
                    query=sql,
                    rows=row_dicts,
                    columns=columns,
                    row_count=len(row_dicts),
                    execution_time_ms=(time.time() - start) * 1000,
                    timestamp=datetime.now().isoformat(),
                )
            else:
                logger.warning("REST API fallback: limited row support")
                return QueryResult(
                    query=sql,
                    rows=[],
                    columns=[],
                    row_count=0,
                    execution_time_ms=(time.time() - start) * 1000,
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            logger.error(f"Query execution failed: {e}\nSQL: {sql}")
            raise

    def list_catalogs(self) -> List[str]:
        try:
            result = self.execute_query("SHOW CATALOGS")
            return [row.get('catalog', row.get('name', '')) for row in result.rows]
        except Exception as e:
            logger.error(f"Failed to list catalogs: {e}")
            return []

    def list_schemas(self, catalog: Optional[str] = None) -> List[str]:
        cat = catalog or self.catalog
        try:
            result = self.execute_query(f"SHOW SCHEMAS IN CATALOG {cat}")
            return [row.get('database_name', row.get('name', '')) for row in result.rows]
        except Exception as e:
            logger.error(f"Failed to list schemas in {cat}: {e}")
            return []

    def list_tables(self, catalog: Optional[str] = None,
                   schema: Optional[str] = None) -> List[str]:
        cat = catalog or self.catalog
        sch = schema or self.schema
        try:
            result = self.execute_query(f"SHOW TABLES IN {cat}.{sch}")
            return [row.get('table_name', row.get('name', '')) for row in result.rows]
        except Exception as e:
            logger.error(f"Failed to list tables in {cat}.{sch}: {e}")
            return []

    def describe_table(self, table: str, catalog: Optional[str] = None,
                      schema: Optional[str] = None) -> List[Dict[str, Any]]:
        cat = catalog or self.catalog
        sch = schema or self.schema
        full_table = f"{cat}.{sch}.{table}"

        try:
            result = self.execute_query(f"DESCRIBE {full_table}")
            return result.rows
        except Exception as e:
            logger.error(f"Failed to describe {full_table}: {e}")
            return []

    def get_table_history(self, table: str, catalog: Optional[str] = None,
                         schema: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        cat = catalog or self.catalog
        sch = schema or self.schema
        full_table = f"{cat}.{sch}.{table}"

        try:
            result = self.execute_query(f"DESCRIBE HISTORY {full_table} LIMIT {limit}")
            return result.rows
        except Exception as e:
            logger.error(f"Failed to get history for {full_table}: {e}")
            return []

    def export_to_parquet(self, query: str, output_path: str) -> bool:
        try:
            temp_table = f"_export_temp_{int(time.time())}"
            create_sql = f"CREATE TABLE {temp_table} AS {query}"
            export_sql = f"""
                SELECT * FROM {temp_table}
                WRITE USING PARQUET LOCATION '{output_path}'
            """

            self.execute_query(create_sql)
            self.execute_query(export_sql)
            self.execute_query(f"DROP TABLE {temp_table}")

            logger.info(f"Exported to Parquet: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            return False

    def time_travel_query(self, table: str, version: Optional[int] = None,
                         timestamp: Optional[str] = None) -> QueryResult:
        if version:
            sql = f"SELECT * FROM {table} VERSION AS OF {version}"
        elif timestamp:
            sql = f"SELECT * FROM {table} TIMESTAMP AS OF '{timestamp}'"
        else:
            raise ValueError("Either version or timestamp required")

        return self.execute_query(sql)


class CatalogPretrainer:

    def __init__(self, connector: UnityCatalogConnector,
                 cache_path: Optional[str] = None):
        self.connector = connector
        self.cache_path = Path(cache_path or './catalog_metadata.db')
        self.db_conn = None

        self._init_cache()

    def _init_cache(self) -> None:
        try:
            self.db_conn = sqlite3.connect(str(self.cache_path))
            self.db_conn.row_factory = sqlite3.Row

            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS catalogs (
                    name TEXT PRIMARY KEY,
                    loaded_at TEXT
                )
            """)

            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS schemas (
                    catalog TEXT,
                    name TEXT,
                    loaded_at TEXT,
                    PRIMARY KEY (catalog, name)
                )
            """)

            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS tables_metadata (
                    catalog TEXT,
                    schema TEXT,
                    table_name TEXT,
                    row_count INTEGER,
                    size_bytes INTEGER,
                    last_modified TEXT,
                    format TEXT,
                    loaded_at TEXT,
                    PRIMARY KEY (catalog, schema, table_name)
                )
            """)

            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS columns_metadata (
                    catalog TEXT,
                    schema TEXT,
                    table_name TEXT,
                    column_name TEXT,
                    data_type TEXT,
                    comment TEXT,
                    nullable INTEGER,
                    loaded_at TEXT,
                    PRIMARY KEY (catalog, schema, table_name, column_name)
                )
            """)

            self.db_conn.commit()
            logger.info(f"Initialized metadata cache: {self.cache_path}")

        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")

    def pretrain_from_catalog(self, catalog: Optional[str] = None,
                            schema: Optional[str] = None) -> Dict[str, Any]:
        cat = catalog or self.connector.catalog
        sch = schema or self.connector.schema

        result = {
            'catalog': cat,
            'schema': sch,
            'schemas': [],
            'tables': [],
            'columns': [],
            'loaded_at': datetime.now().isoformat(),
            'cache_path': str(self.cache_path),
        }

        try:
            if not schema:
                schemas = self.connector.list_schemas(cat)
                result['schemas'] = schemas
            else:
                schemas = [sch]

            for s in schemas:
                try:
                    tables = self.connector.list_tables(cat, s)

                    for table in tables:
                        try:
                            columns = self.connector.describe_table(table, cat, s)

                            for col in columns:
                                self._cache_column(cat, s, table, col)

                            result['tables'].append({
                                'catalog': cat,
                                'schema': s,
                                'table': table,
                                'columns': len(columns),
                            })

                            result['columns'] += columns

                        except Exception as e:
                            logger.warning(f"Failed to load {cat}.{s}.{table}: {e}")

                except Exception as e:
                    logger.warning(f"Failed to load tables in {cat}.{s}: {e}")

            logger.info(f"Pre-trained {result.get('schemas')} schemas, "
                       f"{len(result.get('tables', []))} tables, "
                       f"{len(result.get('columns', []))} columns")

            return result

        except Exception as e:
            logger.error(f"Pretraining failed: {e}")
            return result

    def _cache_column(self, catalog: str, schema: str, table: str,
                     col_info: Dict[str, Any]) -> None:
        if not self.db_conn:
            return

        try:
            self.db_conn.execute("""
                INSERT OR REPLACE INTO columns_metadata
                (catalog, schema, table_name, column_name, data_type, comment, nullable, loaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                catalog,
                schema,
                table,
                col_info.get('col_name', col_info.get('name', '')),
                col_info.get('data_type', col_info.get('type', '')),
                col_info.get('comment', ''),
                1 if col_info.get('nullable', True) else 0,
                datetime.now().isoformat(),
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.debug(f"Failed to cache column: {e}")

    def sync_metadata(self, catalog: Optional[str] = None,
                     schema: Optional[str] = None) -> bool:
        try:
            logger.info("Syncing metadata from Databricks...")
            self.pretrain_from_catalog(catalog, schema)
            return True
        except Exception as e:
            logger.error(f"Metadata sync failed: {e}")
            return False

    def get_cached_tables(self, catalog: str, schema: str) -> List[Dict[str, Any]]:
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.execute("""
                SELECT * FROM tables_metadata
                WHERE catalog = ? AND schema = ?
                ORDER BY table_name
            """, (catalog, schema))

            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get cached tables: {e}")
            return []

    def get_cached_columns(self, catalog: str, schema: str,
                          table: str) -> List[Dict[str, Any]]:
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.execute("""
                SELECT * FROM columns_metadata
                WHERE catalog = ? AND schema = ? AND table_name = ?
                ORDER BY column_name
            """, (catalog, schema, table))

            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get cached columns: {e}")
            return []

    def close(self) -> None:
        if self.db_conn:
            try:
                self.db_conn.close()
            except Exception as e:
                logger.error(f"Error closing cache: {e}")


class LiveDashboardConnector:

    def __init__(self, connector: UnityCatalogConnector,
                 max_connections: int = 5):
        self.connector = connector
        self.max_connections = max_connections
        self.connection_pool: Queue = Queue(maxsize=max_connections)
        self.stats_cache: Dict[str, Tuple[TableStats, float]] = {}
        self.cache_ttl_sec = 300

    def create_live_query(self, query: str) -> Iterator[Dict[str, Any]]:
        result = self.connector.execute_query(query)

        for row in result.rows:
            yield row

    def get_table_stats(self, catalog: str, schema: str,
                       table: str) -> Optional[TableStats]:
        cache_key = f"{catalog}.{schema}.{table}"

        if cache_key in self.stats_cache:
            stats, timestamp = self.stats_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_sec:
                return stats

        try:
            full_table = f"{catalog}.{schema}.{table}"

            row_count_result = self.connector.execute_query(f"SELECT COUNT(*) as cnt FROM {full_table}")
            row_count = row_count_result.rows[0].get('cnt', 0) if row_count_result.rows else 0

            table_info_result = self.connector.execute_query(
                f"SELECT * FROM system.information_schema.table_storage WHERE table_catalog='{catalog}' "
                f"AND table_schema='{schema}' AND table_name='{table}'"
            )

            size_bytes = 0
            format_type = 'delta'
            last_modified = datetime.now().isoformat()

            if table_info_result.rows:
                size_bytes = table_info_result.rows[0].get('size_bytes', 0) or 0
                format_type = table_info_result.rows[0].get('format', 'delta')

            try:
                history = self.connector.get_table_history(table, catalog, schema, limit=1)
                if history:
                    last_modified = history[0].get('timestamp', last_modified)
            except:
                pass

            stats = TableStats(
                catalog=catalog,
                schema=schema,
                table=table,
                row_count=row_count,
                size_bytes=size_bytes,
                last_modified=last_modified,
                format=format_type,
            )

            self.stats_cache[cache_key] = (stats, time.time())

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats for {catalog}.{schema}.{table}: {e}")
            return None

    def invalidate_cache(self, catalog: Optional[str] = None,
                        schema: Optional[str] = None,
                        table: Optional[str] = None) -> None:
        if not catalog:
            self.stats_cache.clear()
        else:
            pattern = f"{catalog}."
            if schema:
                pattern += f"{schema}."
                if table:
                    pattern += table

            keys_to_delete = [k for k in self.stats_cache.keys() if k.startswith(pattern)]
            for k in keys_to_delete:
                del self.stats_cache[k]

    def close(self) -> None:
        self.connector.disconnect()
        self.stats_cache.clear()


def create_unity_connector(catalog: Optional[str] = None) -> UnityCatalogConnector:
    connector = UnityCatalogConnector(catalog=catalog)
    connector.connect()
    return connector


def create_live_dashboard(connector: UnityCatalogConnector) -> LiveDashboardConnector:
    return LiveDashboardConnector(connector)


def create_pretrainer(connector: UnityCatalogConnector) -> CatalogPretrainer:
    return CatalogPretrainer(connector)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Unity Catalog Connector Test ===\n")

    connector = UnityCatalogConnector()

    if connector.connect():
        catalogs = connector.list_catalogs()
        print(f"Available catalogs: {catalogs}\n")

        if catalogs:
            cat = catalogs[0]
            print(f"Using catalog: {cat}\n")

            schemas = connector.list_schemas(cat)
            print(f"Schemas: {schemas}\n")

            if schemas:
                sch = schemas[0]
                print(f"Using schema: {sch}\n")

                tables = connector.list_tables(cat, sch)
                print(f"Tables: {tables}\n")

                if tables:
                    tbl = tables[0]
                    print(f"Using table: {tbl}\n")

                    cols = connector.describe_table(tbl, cat, sch)
                    print(f"Columns: {[c.get('col_name', c.get('name')) for c in cols]}\n")

                    dashboard = create_live_dashboard(connector)
                    stats = dashboard.get_table_stats(cat, sch, tbl)
                    if stats:
                        print(f"Table stats: {stats}\n")

        connector.disconnect()
    else:
        print("Failed to connect. Check DATABRICKS_HOST and DATABRICKS_TOKEN.")
