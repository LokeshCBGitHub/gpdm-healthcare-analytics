import os
import sys
import time
import json
import sqlite3
import logging
import struct
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
PARQUET_DIR = os.path.join(DATA_DIR, 'parquet')

HAS_PYARROW = False
HAS_DUCKDB = False
HAS_POLARS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    HAS_PYARROW = True
    log.info("PyArrow %s available", pa.__version__)
except ImportError:
    pass

try:
    import duckdb
    HAS_DUCKDB = True
    log.info("DuckDB available")
except ImportError:
    pass

try:
    import polars as pl
    HAS_POLARS = True
    log.info("Polars available")
except ImportError:
    pass


PARTITION_SPEC = {
    'claims': {
        'partition_cols': ['REGION', 'CLAIM_TYPE'],
        'sort_cols': ['SERVICE_DATE', 'MEMBER_ID'],
        'row_group_size': 500000,
    },
    'encounters': {
        'partition_cols': ['REGION', 'VISIT_TYPE'],
        'sort_cols': ['SERVICE_DATE', 'MEMBER_ID'],
        'row_group_size': 500000,
    },
    'members': {
        'partition_cols': ['REGION'],
        'sort_cols': ['MEMBER_ID'],
        'row_group_size': 250000,
    },
    'diagnoses': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'DIAGNOSIS_DATE'],
        'row_group_size': 500000,
    },
    'prescriptions': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'PRESCRIPTION_DATE'],
        'row_group_size': 250000,
    },
    'lab_results': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'LAB_DATE'],
        'row_group_size': 1000000,
    },
    'authorizations': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'REQUEST_DATE'],
        'row_group_size': 250000,
    },
    'quality_measures': {
        'partition_cols': ['MEASUREMENT_YEAR'],
        'sort_cols': ['MEMBER_ID', 'MEASURE_CODE'],
        'row_group_size': 500000,
    },
    'providers': {
        'partition_cols': [],
        'sort_cols': ['NPI'],
        'row_group_size': 100000,
    },
    'referrals': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'REFERRAL_DATE'],
        'row_group_size': 250000,
    },
    'appointments': {
        'partition_cols': [],
        'sort_cols': ['MEMBER_ID', 'APPOINTMENT_DATE'],
        'row_group_size': 250000,
    },
}

COMPRESSION = 'snappy'
PARQUET_VERSION = '2.6'


class ParquetDataEngine:

    def __init__(self, db_path: str, parquet_dir: str = None):
        self.db_path = db_path
        self.parquet_dir = parquet_dir or PARQUET_DIR
        self._duckdb_conn = None
        self._stats_cache: Dict[str, Dict] = {}
        self._available = HAS_PYARROW or HAS_DUCKDB or HAS_POLARS
        if not self._available:
            log.warning("No large-data library available (pyarrow/duckdb/polars). "
                        "Parquet engine will use SQLite fallback. "
                        "Install: pip install pyarrow duckdb polars")

    @property
    def engine_type(self) -> str:
        if HAS_DUCKDB:
            return 'duckdb'
        if HAS_POLARS:
            return 'polars'
        if HAS_PYARROW:
            return 'pyarrow'
        return 'sqlite_fallback'

    def export_to_parquet(self, tables: List[str] = None) -> Dict[str, Any]:
        if not HAS_PYARROW:
            log.warning("PyArrow required for Parquet export. Install: pip install pyarrow")
            return {'error': 'pyarrow not installed', 'tables': []}

        os.makedirs(self.parquet_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        report = {'tables': [], 'total_rows': 0, 'total_size_mb': 0, 'duration_s': 0}
        t0 = time.time()

        if not tables:
            tables = list(PARTITION_SPEC.keys())

        for table in tables:
            try:
                spec = PARTITION_SPEC.get(table, {
                    'partition_cols': [], 'sort_cols': [], 'row_group_size': 500000
                })

                cols_info = conn.execute("PRAGMA table_info(%s)" % table).fetchall()
                if not cols_info:
                    continue

                col_names = [c[1] for c in cols_info]
                rows = conn.execute("SELECT * FROM %s" % table).fetchall()
                if not rows:
                    continue

                col_arrays = {}
                for i, cname in enumerate(col_names):
                    values = [row[i] for row in rows]
                    col_type = cols_info[i][2].upper()
                    if col_type in ('INTEGER', 'INT'):
                        col_arrays[cname] = pa.array(
                            [int(v) if v is not None and str(v).strip() != '' else None for v in values],
                            type=pa.int64()
                        )
                    elif col_type == 'REAL':
                        col_arrays[cname] = pa.array(
                            [float(v) if v is not None and str(v).strip() != '' else None for v in values],
                            type=pa.float64()
                        )
                    else:
                        col_arrays[cname] = pa.array(
                            [str(v) if v is not None else None for v in values],
                            type=pa.string()
                        )

                arrow_table = pa.table(col_arrays)

                if spec['sort_cols']:
                    valid_sort = [c for c in spec['sort_cols'] if c in col_names]
                    if valid_sort:
                        sort_indices = pc.sort_indices(arrow_table, sort_keys=[(c, 'ascending') for c in valid_sort])
                        arrow_table = arrow_table.take(sort_indices)

                table_dir = os.path.join(self.parquet_dir, table)
                os.makedirs(table_dir, exist_ok=True)

                if spec['partition_cols']:
                    valid_parts = [c for c in spec['partition_cols'] if c in col_names]
                    if valid_parts:
                        pq.write_to_dataset(
                            arrow_table,
                            root_path=table_dir,
                            partition_cols=valid_parts,
                            compression=COMPRESSION,
                            version=PARQUET_VERSION,
                            row_group_size=spec['row_group_size'],
                        )
                    else:
                        out_path = os.path.join(table_dir, '%s.parquet' % table)
                        pq.write_table(
                            arrow_table, out_path,
                            compression=COMPRESSION,
                            version=PARQUET_VERSION,
                            row_group_size=spec['row_group_size'],
                        )
                else:
                    out_path = os.path.join(table_dir, '%s.parquet' % table)
                    pq.write_table(
                        arrow_table, out_path,
                        compression=COMPRESSION,
                        version=PARQUET_VERSION,
                        row_group_size=spec['row_group_size'],
                    )

                total_size = 0
                for root, dirs, files in os.walk(table_dir):
                    for f in files:
                        if f.endswith('.parquet'):
                            total_size += os.path.getsize(os.path.join(root, f))

                report['tables'].append({
                    'table': table,
                    'rows': len(rows),
                    'columns': len(col_names),
                    'partitions': len(spec['partition_cols']),
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'compression': COMPRESSION,
                })
                report['total_rows'] += len(rows)
                report['total_size_mb'] += round(total_size / (1024 * 1024), 2)
                log.info("  Exported %s: %d rows, %d cols, %.2f MB",
                         table, len(rows), len(col_names), total_size / (1024 * 1024))

            except Exception as e:
                log.error("  Failed to export %s: %s", table, e)
                report['tables'].append({'table': table, 'error': str(e)})

        conn.close()
        report['duration_s'] = round(time.time() - t0, 2)
        log.info("Parquet export complete: %d tables, %d rows, %.2f MB in %.1fs",
                 len(report['tables']), report['total_rows'],
                 report['total_size_mb'], report['duration_s'])
        return report

    def query_parquet(self, sql: str, tables_hint: List[str] = None) -> Tuple[List, List, Optional[str]]:
        if HAS_DUCKDB:
            return self._query_duckdb(sql, tables_hint)
        if HAS_POLARS:
            return self._query_polars(sql, tables_hint)
        if HAS_PYARROW:
            return self._query_pyarrow(sql, tables_hint)
        return self._query_sqlite_fallback(sql)

    def _query_duckdb(self, sql: str, tables_hint: List[str] = None) -> Tuple[List, List, Optional[str]]:
        try:
            if not self._duckdb_conn:
                self._duckdb_conn = duckdb.connect(':memory:')
                self._register_parquet_tables(self._duckdb_conn)

            result = self._duckdb_conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return rows, columns, None
        except Exception as e:
            return [], [], str(e)

    def _register_parquet_tables(self, conn):
        if not os.path.exists(self.parquet_dir):
            conn.execute("ATTACH '%s' AS sqlite_db (TYPE SQLITE)" % self.db_path)
            for table in PARTITION_SPEC:
                try:
                    conn.execute("CREATE VIEW %s AS SELECT * FROM sqlite_db.%s" % (table, table))
                except Exception:
                    pass
            return

        for table in PARTITION_SPEC:
            table_dir = os.path.join(self.parquet_dir, table)
            if os.path.exists(table_dir):
                parquet_files = []
                for root, dirs, files in os.walk(table_dir):
                    for f in files:
                        if f.endswith('.parquet'):
                            parquet_files.append(os.path.join(root, f))
                if parquet_files:
                    if len(parquet_files) == 1:
                        conn.execute(
                            "CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" %
                            (table, parquet_files[0])
                        )
                    else:
                        conn.execute(
                            "CREATE VIEW %s AS SELECT * FROM read_parquet('%s/**/*.parquet', hive_partitioning=1)" %
                            (table, table_dir)
                        )
                    continue

            try:
                conn.execute("ATTACH '%s' AS sqlite_db (TYPE SQLITE)" % self.db_path)
                conn.execute("CREATE VIEW %s AS SELECT * FROM sqlite_db.%s" % (table, table))
            except Exception:
                pass

    def _query_polars(self, sql: str, tables_hint: List[str] = None) -> Tuple[List, List, Optional[str]]:
        try:
            ctx = pl.SQLContext()
            for table in PARTITION_SPEC:
                table_dir = os.path.join(self.parquet_dir, table)
                pfile = os.path.join(table_dir, '%s.parquet' % table)
                if os.path.exists(pfile):
                    df = pl.scan_parquet(pfile)
                    ctx.register(table, df)
                elif os.path.exists(table_dir):
                    df = pl.scan_parquet(os.path.join(table_dir, '**/*.parquet'))
                    ctx.register(table, df)

            result = ctx.execute(sql).collect()
            columns = result.columns
            rows = result.rows()
            return list(rows), list(columns), None
        except Exception as e:
            return [], [], str(e)

    def _query_pyarrow(self, sql: str, tables_hint: List[str] = None) -> Tuple[List, List, Optional[str]]:
        return self._query_sqlite_fallback(sql)

    def _query_sqlite_fallback(self, sql: str) -> Tuple[List, List, Optional[str]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return rows, columns, None
        except Exception as e:
            return [], [], str(e)

    def get_table_stats(self, table: str) -> Dict:
        if table in self._stats_cache:
            return self._stats_cache[table]

        stats = {'table': table, 'source': 'unknown', 'rows': 0, 'columns': 0, 'size_mb': 0}

        table_dir = os.path.join(self.parquet_dir, table)
        if HAS_PYARROW and os.path.exists(table_dir):
            try:
                dataset = ds.dataset(table_dir, format='parquet')
                stats['source'] = 'parquet'
                stats['rows'] = dataset.count_rows()
                stats['columns'] = len(dataset.schema)
                stats['schema'] = {f.name: str(f.type) for f in dataset.schema}
                total_size = 0
                for frag in dataset.get_fragments():
                    md = frag.metadata
                    if md:
                        total_size += md.serialized_size
                stats['size_mb'] = round(total_size / (1024 * 1024), 2)
                stats['row_groups'] = sum(1 for _ in dataset.get_fragments())
                self._stats_cache[table] = stats
                return stats
            except Exception:
                pass

        try:
            conn = sqlite3.connect(self.db_path)
            row_count = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
            col_count = len(conn.execute("PRAGMA table_info(%s)" % table).fetchall())
            stats['source'] = 'sqlite'
            stats['rows'] = row_count
            stats['columns'] = col_count
            page_count = conn.execute("SELECT SUM(pageno) FROM dbstat WHERE name='%s'" % table).fetchone()
            if page_count and page_count[0]:
                stats['size_mb'] = round(page_count[0] * 4096 / (1024 * 1024), 2)
            conn.close()
        except Exception:
            pass

        self._stats_cache[table] = stats
        return stats

    def estimate_scale_capacity(self) -> Dict:
        current_rows = 0
        conn = sqlite3.connect(self.db_path)
        for table in PARTITION_SPEC:
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
                current_rows += cnt
            except Exception:
                pass
        conn.close()

        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)

        return {
            'engine': self.engine_type,
            'current_rows': current_rows,
            'current_db_size_mb': round(db_size_mb, 2),
            'sqlite_max_practical_gb': 10,
            'parquet_max_practical_tb': 100,
            'recommendations': self._scale_recommendations(current_rows, db_size_mb),
            'parquet_available': HAS_PYARROW,
            'duckdb_available': HAS_DUCKDB,
            'polars_available': HAS_POLARS,
            'partition_spec': {t: {'partitions': s['partition_cols'], 'row_group': s['row_group_size']}
                              for t, s in PARTITION_SPEC.items()},
        }

    def _scale_recommendations(self, total_rows: int, db_size_mb: float) -> List[str]:
        recs = []
        if db_size_mb < 100:
            recs.append("Current data fits in SQLite. No action needed.")
        if db_size_mb >= 100 and not HAS_PYARROW:
            recs.append("CRITICAL: Install pyarrow for Parquet support: pip install pyarrow")
        if db_size_mb >= 500:
            recs.append("Export to Parquet with partitioning for 5-10x query speedup")
        if db_size_mb >= 1000:
            recs.append("Use DuckDB for columnar analytics: pip install duckdb")
        if total_rows >= 100_000_000:
            recs.append("Enable row group size tuning: 500K-1M rows per group")
            recs.append("Consider Parquet with Snappy compression for 3-5x size reduction")
        if total_rows >= 1_000_000_000:
            recs.append("Switch to distributed query with DuckDB + Parquet partitions")
            recs.append("Consider Apache Spark or Trino for multi-node processing")
        if not recs:
            recs.append("System is production-ready for current data scale")
            recs.append("Parquet + DuckDB pipeline ready to activate at GB/TB scale")
        return recs

    def benchmark(self, queries: List[str] = None) -> Dict:
        if not queries:
            queries = [
                "SELECT COUNT(*) FROM CLAIMS",
                "SELECT REGION, COUNT(*) FROM CLAIMS GROUP BY REGION",
                "SELECT CLAIM_STATUS, AVG(CAST(BILLED_AMOUNT AS REAL)) FROM CLAIMS GROUP BY CLAIM_STATUS",
                "SELECT m.PLAN_TYPE, COUNT(DISTINCT c.CLAIM_ID) FROM CLAIMS c JOIN MEMBERS m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY m.PLAN_TYPE",
            ]

        results = {'sqlite': [], 'parquet': [], 'speedup': []}
        conn = sqlite3.connect(self.db_path)

        for q in queries:
            t0 = time.time()
            try:
                conn.execute(q).fetchall()
                sqlite_ms = round((time.time() - t0) * 1000, 2)
            except Exception:
                sqlite_ms = -1
            results['sqlite'].append(sqlite_ms)

            t0 = time.time()
            self.query_parquet(q)
            parquet_ms = round((time.time() - t0) * 1000, 2)
            results['parquet'].append(parquet_ms)

            if sqlite_ms > 0 and parquet_ms > 0:
                results['speedup'].append(round(sqlite_ms / parquet_ms, 2))
            else:
                results['speedup'].append(None)

        conn.close()
        results['queries'] = queries
        results['engine'] = self.engine_type
        return results

    def health_check(self) -> Dict:
        return {
            'engine': self.engine_type,
            'pyarrow': HAS_PYARROW,
            'duckdb': HAS_DUCKDB,
            'polars': HAS_POLARS,
            'parquet_dir': self.parquet_dir,
            'parquet_exists': os.path.exists(self.parquet_dir),
            'tables_configured': list(PARTITION_SPEC.keys()),
            'db_path': self.db_path,
            'db_exists': os.path.exists(self.db_path),
            'capacity': self.estimate_scale_capacity(),
        }


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    db_path = os.path.join(DATA_DIR, 'healthcare_demo.db')
    engine = ParquetDataEngine(db_path)

    print("=" * 70)
    print("PARQUET DATA ENGINE HEALTH CHECK")
    print("=" * 70)
    health = engine.health_check()
    for k, v in health.items():
        if k != 'capacity':
            print("  %-25s %s" % (k, v))

    print()
    print("SCALE CAPACITY:")
    cap = health['capacity']
    print("  Engine:             %s" % cap['engine'])
    print("  Current rows:       %s" % f"{cap['current_rows']:,}")
    print("  Current DB size:    %.2f MB" % cap['current_db_size_mb'])
    print("  SQLite max:         %d GB" % cap['sqlite_max_practical_gb'])
    print("  Parquet max:        %d TB" % cap['parquet_max_practical_tb'])
    print()
    print("  Recommendations:")
    for r in cap['recommendations']:
        print("    - %s" % r)

    print()
    print("  Partition spec:")
    for table, spec in cap['partition_spec'].items():
        parts = spec['partitions'] if spec['partitions'] else ['none']
        print("    %-20s partitions: %-25s row_group: %s" % (table, ', '.join(parts), f"{spec['row_group']:,}"))

    if HAS_PYARROW:
        print()
        print("EXPORTING TO PARQUET...")
        report = engine.export_to_parquet()
        for t in report['tables']:
            if 'error' not in t:
                print("  %-20s %8d rows  %3d cols  %.2f MB  [%s]" % (
                    t['table'], t['rows'], t['columns'], t['size_mb'], COMPRESSION))

    print()
    print("BENCHMARK:")
    bench = engine.benchmark()
    for i, q in enumerate(bench['queries']):
        sq = bench['sqlite'][i]
        pq_ms = bench['parquet'][i]
        sp = bench['speedup'][i]
        print("  Q%d: SQLite=%6.1fms  %s=%6.1fms  speedup=%sx" % (
            i + 1, sq, bench['engine'], pq_ms, sp or 'N/A'))

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
