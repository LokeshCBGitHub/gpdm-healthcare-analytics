import os
import re
import sys
import csv
import json
import time
import sqlite3
import hashlib
import logging
import logging.handlers
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from functools import lru_cache


LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT = 5
AUDIT_BACKUP_COUNT = 10
CSV_BATCH_SIZE = 5000
DB_FRESHNESS_SECONDS = 3600
MAX_QUERY_LENGTH = 2000
MAX_RESULT_ROWS = 10000
DEFAULT_RATE_LIMIT = 30
DEFAULT_BURST = 10


_LOG_INITIALIZED = False
_LOG_LOCK = threading.Lock()


def setup_logging(log_dir: str = None, level: str = "INFO",
                  enable_audit: bool = True) -> logging.Logger:
    global _LOG_INITIALIZED

    with _LOG_LOCK:
        if _LOG_INITIALIZED:
            return logging.getLogger('gpdm')

        if log_dir is None:
            script_dir = Path(__file__).parent
            log_dir = str(script_dir.parent / 'logs')

        os.makedirs(log_dir, exist_ok=True)
        try:
            os.chmod(log_dir, 0o700)
        except OSError:
            pass

        logger = logging.getLogger('gpdm')
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()

        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console.setFormatter(console_fmt)
        logger.addHandler(console)

        app_log = os.path.join(log_dir, 'gpdm_app.log')
        file_handler = logging.handlers.RotatingFileHandler(
            app_log, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d — %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        if enable_audit:
            audit_logger = logging.getLogger('gpdm.audit')
            audit_logger.setLevel(logging.INFO)
            audit_log = os.path.join(log_dir, 'gpdm_audit.log')
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_log, maxBytes=LOG_MAX_BYTES, backupCount=AUDIT_BACKUP_COUNT,
                encoding='utf-8'
            )
            audit_fmt = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S%z'
            )
            audit_handler.setFormatter(audit_fmt)
            audit_logger.addHandler(audit_handler)

        _LOG_INITIALIZED = True
        logger.info("Logging initialized (dir=%s, level=%s)", log_dir, level)
        return logger


def get_logger(name: str = 'gpdm') -> logging.Logger:
    if not _LOG_INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


def audit_log(event: str, user: str = 'system', ip: str = '127.0.0.1',
              details: Dict = None) -> None:
    audit = get_logger('gpdm.audit')
    safe_details = _sanitize_audit_details(details or {})
    audit.info("event=%s | user=%s | ip=%s | %s",
               event, user, ip,
               " | ".join(f"{k}={v}" for k, v in safe_details.items()))


def _sanitize_audit_details(details: Dict) -> Dict:
    safe = {}
    SAFE_KEYS = {
        'intent', 'engine_mode', 'tables_used', 'row_count', 'sql_hash',
        'response_time_ms', 'success', 'error_type', 'confidence',
        'question_length', 'cache_hit',
    }
    for k, v in details.items():
        if k in SAFE_KEYS:
            safe[k] = v
    return safe


class DatabasePool:

    _instance: Optional['DatabasePool'] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._db_path: Optional[str] = None
        self._local = threading.local()
        self._loaded = False
        self._table_names: List[str] = []
        self._column_whitelist: Dict[str, set] = {}
        self._load_lock = threading.Lock()
        self._logger = get_logger('gpdm.db')

    @classmethod
    def get_instance(cls) -> 'DatabasePool':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.close_all()
            cls._instance = None

    def initialize(self, cfg: Dict[str, str]) -> Dict[str, Any]:
        with self._load_lock:
            if self._loaded:
                return {'status': 'already_loaded', 'tables': self._table_names}

            t0 = time.time()
            base_dir = resolve_base_dir(cfg)
            raw_dir = cfg.get('RAW_DIR', os.path.join(base_dir, 'data', 'raw'))

            if not os.path.exists(raw_dir):
                raise FileNotFoundError(f"Data directory not found: {raw_dir}")

            db_dir = os.path.join(base_dir, 'data')
            self._db_path = os.path.join(db_dir, 'healthcare_production.db')
            try:
                os.makedirs(db_dir, exist_ok=True)
                test_path = os.path.join(db_dir, '.db_write_test')
                with open(test_path, 'w') as f:
                    f.write('ok')
                os.remove(test_path)
            except (OSError, PermissionError):
                import tempfile
                db_dir = os.path.join(tempfile.gettempdir(), 'gpdm_healthcare')
                os.makedirs(db_dir, exist_ok=True)
                self._db_path = os.path.join(db_dir, 'healthcare_production.db')
                self._logger.info("DB directory fell back to: %s", db_dir)

            if os.path.exists(self._db_path):
                db_age = time.time() - os.path.getmtime(self._db_path)
                if db_age < DB_FRESHNESS_SECONDS:
                    self._load_schema_from_db()
                    if self._table_names:
                        self._logger.info("Using existing DB (age=%.0fs, %d tables): %s",
                                          db_age, len(self._table_names), self._db_path)
                        self._loaded = True
                        elapsed = time.time() - t0
                        return {
                            'status': 'reused',
                            'tables': self._table_names,
                            'elapsed_ms': int(elapsed * 1000),
                            'db_path': self._db_path,
                        }
                    else:
                        self._logger.warning(
                            "Existing DB has no tables, rebuilding: %s", self._db_path)
                        try:
                            os.remove(self._db_path)
                        except OSError:
                            pass

            self._logger.info("Building production DB from CSVs: %s", raw_dir)
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cursor = conn.cursor()

            total_rows = 0
            errors = []

            for filename in sorted(os.listdir(raw_dir)):
                if not filename.endswith('.csv'):
                    continue

                table_name = filename.replace('.csv', '')
                if not _is_valid_identifier(table_name):
                    errors.append(f"Skipped invalid table name: {filename}")
                    continue

                try:
                    csv_path = os.path.join(raw_dir, filename)
                    rows_loaded = self._load_csv_to_table(cursor, csv_path, table_name)
                    self._table_names.append(table_name)
                    total_rows += rows_loaded
                    self._logger.info("  Loaded %s: %d rows", table_name, rows_loaded)
                except Exception as e:
                    errors.append(f"Error loading {filename}: {str(e)}")
                    self._logger.error("Failed to load %s: %s", filename, e)

            conn.commit()

            self._create_indexes(cursor)
            conn.commit()
            conn.close()

            self._load_schema_from_db()
            self._loaded = True

            elapsed = time.time() - t0
            stats = {
                'status': 'loaded',
                'tables': self._table_names,
                'total_rows': total_rows,
                'elapsed_ms': int(elapsed * 1000),
                'db_path': self._db_path,
                'errors': errors,
            }

            if errors:
                self._logger.warning("DB load completed with %d errors", len(errors))
            else:
                self._logger.info("DB loaded: %d tables, %d rows in %dms",
                                  len(self._table_names), total_rows, stats['elapsed_ms'])

            return stats

    def _load_csv_to_table(self, cursor: sqlite3.Cursor, csv_path: str,
                           table_name: str) -> int:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            if not columns:
                raise ValueError(f"No columns found in {csv_path}")

            safe_columns = []
            for col in columns:
                if _is_valid_identifier(col):
                    safe_columns.append(col)
                else:
                    self._logger.warning("Skipping invalid column: %s.%s",
                                         table_name, col)

            if not safe_columns:
                raise ValueError(f"No valid columns in {csv_path}")

            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            col_defs = ', '.join([f'"{col}" TEXT' for col in safe_columns])
            cursor.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

            placeholders = ', '.join(['?' for _ in safe_columns])
            insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'

            row_count = 0
            batch = []

            for row in reader:
                values = [row.get(col, '') for col in safe_columns]
                batch.append(values)
                row_count += 1

                if len(batch) >= CSV_BATCH_SIZE:
                    cursor.executemany(insert_sql, batch)
                    batch = []

            if batch:
                cursor.executemany(insert_sql, batch)

            return row_count

    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        INDEX_COLUMNS = {
            'claims': ['MEMBER_ID', 'ENCOUNTER_ID', 'SERVICE_DATE', 'CLAIM_STATUS', 'KP_REGION'],
            'encounters': ['ENCOUNTER_ID', 'MEMBER_ID', 'FACILITY_NAME', 'VISIT_TYPE'],
            'members': ['MEMBER_ID', 'KP_REGION'],
            'providers': ['NPI', 'SPECIALTY'],
            'diagnoses': ['DIAGNOSIS_ID', 'ICD10_CODE'],
            'prescriptions': ['PRESCRIPTION_ID', 'MEMBER_ID', 'DRUG_NAME'],
            'referrals': ['REFERRAL_ID', 'MEMBER_ID'],
        }

        for table, cols in INDEX_COLUMNS.items():
            if table not in self._table_names:
                continue
            for col in cols:
                try:
                    idx_name = f"idx_{table}_{col}"
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ("{col}")')
                except sqlite3.Error as e:
                    self._logger.debug('Index creation skipped: %s', e)

    def _load_schema_from_db(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        self._table_names = [r[0] for r in cursor.fetchall()]

        self._column_whitelist = {}
        for table in self._table_names:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = {r[1].upper() for r in cursor.fetchall()}
            self._column_whitelist[table] = cols

    def get_connection(self) -> sqlite3.Connection:
        if not self._loaded and not self._db_path:
            raise RuntimeError("DatabasePool not initialized. Call initialize() first.")

        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn

        return conn

    def execute_query(self, sql: str, timeout: int = 30) -> Tuple[bool, List[Dict], str]:
        t0 = time.time()

        sql_normalized = sql.strip().upper()
        if not sql_normalized.startswith('SELECT'):
            return False, [], "Only SELECT queries are allowed"

        BLOCKED = {'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
                    'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'}
        sql_words = set(re.findall(r'\b[A-Z]+\b', sql_normalized))
        blocked_found = sql_words & BLOCKED
        if blocked_found:
            return False, [], f"Blocked SQL keywords: {blocked_found}"

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            raw_rows = cursor.fetchall()

            if raw_rows and hasattr(raw_rows[0], 'keys'):
                rows = [dict(r) for r in raw_rows]
            else:
                rows = [{'value': r[0]} for r in raw_rows] if raw_rows else []

            elapsed = time.time() - t0
            self._logger.debug("Query OK: %.0fms, %d rows", elapsed * 1000, len(rows))

            return True, rows, ""

        except sqlite3.Error as e:
            elapsed = time.time() - t0
            self._logger.warning("SQL error (%.0fms): %s — %s",
                                 elapsed * 1000, str(e), sql[:200])
            return False, [], f"SQL Error: {str(e)}"
        except Exception as e:
            self._logger.error("Unexpected query error: %s", str(e), exc_info=True)
            return False, [], f"Query Error: {str(e)}"

    def validate_table(self, table_name: str) -> bool:
        return table_name in self._table_names

    def validate_column(self, table_name: str, column_name: str) -> bool:
        cols = self._column_whitelist.get(table_name, set())
        return column_name.upper() in cols

    def get_table_names(self) -> List[str]:
        return list(self._table_names)

    def get_column_names(self, table_name: str) -> List[str]:
        return list(self._column_whitelist.get(table_name, set()))

    def close_all(self) -> None:
        conn = getattr(self._local, 'conn', None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def health_check(self) -> Dict[str, Any]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

            table_counts = {}
            for table in self._table_names:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    table_counts[table] = cursor.fetchone()[0]
                except sqlite3.Error:
                    table_counts[table] = -1

            return {
                'status': 'healthy',
                'db_path': self._db_path,
                'tables': self._table_names,
                'table_row_counts': table_counts,
                'total_rows': sum(v for v in table_counts.values() if v > 0),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }


_VALID_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$')


def _is_valid_identifier(name: str) -> bool:
    return bool(_VALID_IDENTIFIER_RE.match(name))


def validate_question(question: str) -> Tuple[bool, str, str]:
    if not question or not isinstance(question, str):
        return False, '', 'Question must be a non-empty string'

    question = ' '.join(question.split())

    if len(question) > MAX_QUERY_LENGTH:
        return False, '', f'Question too long ({len(question)} chars, max {MAX_QUERY_LENGTH})'

    if any(ord(c) < 32 and c not in ('\n', '\t') for c in question):
        return False, '', 'Question contains invalid control characters'

    INJECTION_PATTERNS = [
        r';\s*(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC)',
        r'--\s*$',
        r'/\*.*\*/',
        r'\bUNION\s+ALL\s+SELECT\b',
        r'\bOR\s+1\s*=\s*1\b',
        r"'\s*OR\s*'",
    ]
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            get_logger('gpdm.security').warning(
                "Blocked potential SQL injection: %s", question[:100])
            return False, '', 'Query contains disallowed patterns'

    return True, question.strip(), ''


def validate_sql_output(sql: str, allowed_tables: List[str] = None) -> Tuple[bool, str]:
    if not sql or not isinstance(sql, str):
        return False, 'Empty SQL'

    sql_upper = sql.strip().upper()

    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, 'Only SELECT statements allowed'

    BLOCKED_OPS = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
                    'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
                    'ATTACH', 'DETACH']
    for op in BLOCKED_OPS:
        if re.search(rf'\b{op}\b', sql_upper):
            return False, f'Blocked operation: {op}'

    if allowed_tables:
        cte_names = set()
        if sql_upper.startswith('WITH'):
            cte_matches = re.findall(r'\bWITH\s+(\w+)\s+AS\b|\,\s*(\w+)\s+AS\b', sql, re.IGNORECASE)
            for groups in cte_matches:
                for g in groups:
                    if g:
                        cte_names.add(g.lower())

        from_match = re.findall(r'\bFROM\s+"?(\w+)"?', sql, re.IGNORECASE)
        join_match = re.findall(r'\bJOIN\s+"?(\w+)"?', sql, re.IGNORECASE)
        referenced_tables = set(from_match + join_match)

        allowed_set = {t.lower() for t in allowed_tables} | cte_names
        for table in referenced_tables:
            if table.lower() not in allowed_set:
                return False, f'Table not in whitelist: {table}'

    return True, ''


class RateLimiter:

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT, burst: int = DEFAULT_BURST):
        self.rate = requests_per_minute / 60.0
        self.burst = burst
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._logger = get_logger('gpdm.ratelimit')

    def allow(self, client_ip: str) -> bool:
        now = time.time()

        with self._lock:
            if client_ip not in self._buckets:
                self._buckets[client_ip] = {
                    'tokens': self.burst,
                    'last_time': now,
                }

            bucket = self._buckets[client_ip]

            elapsed = now - bucket['last_time']
            bucket['tokens'] = min(self.burst, bucket['tokens'] + elapsed * self.rate)
            bucket['last_time'] = now

            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            else:
                self._logger.warning("Rate limited: %s", client_ip)
                return False

    def cleanup(self, max_age: float = 3600) -> None:
        now = time.time()
        with self._lock:
            stale = [ip for ip, b in self._buckets.items()
                     if now - b['last_time'] > max_age]
            for ip in stale:
                del self._buckets[ip]


def resolve_base_dir(cfg: Dict[str, str] = None) -> str:
    env_base = os.environ.get('GPDM_BASE_DIR')
    if env_base and os.path.isdir(env_base):
        return os.path.abspath(env_base)

    if cfg:
        cfg_base = cfg.get('BASE_DIR', '')
        if cfg_base and os.path.isdir(cfg_base):
            return os.path.abspath(cfg_base)

    script_dir = Path(__file__).resolve().parent
    auto_base = script_dir.parent
    if (auto_base / 'data').exists() or (auto_base / 'semantic_catalog').exists():
        return str(auto_base)

    return str(auto_base)


def resolve_config(cfg_input: Dict[str, str] = None) -> Dict[str, str]:
    cfg = dict(cfg_input or {})
    base = resolve_base_dir(cfg)

    cfg['BASE_DIR'] = base
    cfg['DATA_DIR'] = os.path.join(base, 'data')
    cfg['RAW_DIR'] = os.path.join(base, 'data', 'raw')
    cfg['LOG_DIR'] = os.path.join(base, 'logs')
    cfg['CATALOG_DIR'] = os.path.join(base, 'semantic_catalog')
    cfg['SCRIPTS_DIR'] = os.path.join(base, 'scripts')

    logger = get_logger('gpdm.config')
    for key in ['RAW_DIR', 'CATALOG_DIR']:
        path = cfg[key]
        if not os.path.exists(path):
            logger.warning("Config path missing: %s=%s", key, path)

    return cfg


_START_TIME = time.time()
_QUERY_METRICS = {
    'total_queries': 0,
    'successful_queries': 0,
    'failed_queries': 0,
    'total_response_time_ms': 0,
    'errors': [],
}
_METRICS_LOCK = threading.Lock()


def record_query_metric(success: bool, response_time_ms: float,
                        error: str = None) -> None:
    with _METRICS_LOCK:
        _QUERY_METRICS['total_queries'] += 1
        if success:
            _QUERY_METRICS['successful_queries'] += 1
        else:
            _QUERY_METRICS['failed_queries'] += 1
            if error:
                _QUERY_METRICS['errors'].append({
                    'time': datetime.now(timezone.utc).isoformat(),
                    'error': str(error)[:200],
                })
                _QUERY_METRICS['errors'] = _QUERY_METRICS['errors'][-100:]
        _QUERY_METRICS['total_response_time_ms'] += response_time_ms


def get_health_status() -> Dict[str, Any]:
    uptime = time.time() - _START_TIME

    with _METRICS_LOCK:
        total = _QUERY_METRICS['total_queries']
        avg_ms = (_QUERY_METRICS['total_response_time_ms'] / total
                  if total > 0 else 0)
        error_rate = (_QUERY_METRICS['failed_queries'] / total * 100
                      if total > 0 else 0)

    db_health = {'status': 'not_initialized'}
    try:
        pool = DatabasePool.get_instance()
        if pool._loaded:
            db_health = pool.health_check()
    except Exception as e:
        db_health = {'status': 'error', 'error': str(e)}

    return {
        'status': 'healthy' if error_rate < 20 else 'degraded',
        'version': '2.0.0',
        'uptime_seconds': int(uptime),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'queries': {
            'total': total,
            'successful': _QUERY_METRICS['successful_queries'],
            'failed': _QUERY_METRICS['failed_queries'],
            'avg_response_ms': round(avg_ms, 1),
            'error_rate_pct': round(error_rate, 1),
        },
        'database': db_health,
        'python_version': sys.version.split()[0],
    }


def preflight_check(cfg: Dict[str, str]) -> Dict[str, Any]:
    logger = get_logger('gpdm.startup')
    logger.info("Running preflight checks...")
    results = {}

    raw_dir = cfg.get('RAW_DIR', '')
    if os.path.isdir(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        results['data_dir'] = {
            'status': 'ok' if csv_files else 'warning',
            'path': raw_dir,
            'csv_files': len(csv_files),
        }
    else:
        results['data_dir'] = {'status': 'error', 'path': raw_dir, 'error': 'not found'}

    catalog_dir = cfg.get('CATALOG_DIR', '')
    if os.path.isdir(catalog_dir):
        results['catalog'] = {'status': 'ok', 'path': catalog_dir}
    else:
        results['catalog'] = {'status': 'warning', 'path': catalog_dir, 'error': 'not found'}

    modules = {}
    for mod_name in ['sqlite3', 'json', 'csv', 'hashlib', 'logging']:
        try:
            __import__(mod_name)
            modules[mod_name] = 'available'
        except ImportError:
            modules[mod_name] = 'missing'
    results['modules'] = modules

    log_dir = cfg.get('LOG_DIR', '')
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            test_file = os.path.join(log_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('ok')
            os.remove(test_file)
            results['log_dir'] = {'status': 'ok', 'path': log_dir}
        except Exception as e:
            fallback_log = os.path.join('/tmp', 'gpdm_logs')
            try:
                os.makedirs(fallback_log, exist_ok=True)
                cfg['LOG_DIR'] = fallback_log
                results['log_dir'] = {'status': 'warning', 'path': fallback_log,
                                      'note': f'Fell back from {log_dir}: {e}'}
            except Exception:
                results['log_dir'] = {'status': 'warning', 'path': log_dir, 'error': str(e)}

    CRITICAL_CHECKS = {'data_dir'}
    has_critical = any(
        r.get('status') == 'error' for k, r in results.items()
        if isinstance(r, dict) and k in CRITICAL_CHECKS
    )
    results['overall'] = 'fail' if has_critical else 'ok'

    if has_critical:
        logger.error("Preflight FAILED: %s", json.dumps(results, indent=2))
    else:
        logger.info("Preflight OK")

    return results


_shutdown_handlers: List = []


def register_shutdown_handler(handler) -> None:
    _shutdown_handlers.append(handler)


def graceful_shutdown(signum=None, frame=None) -> None:
    logger = get_logger('gpdm')
    logger.info("Graceful shutdown initiated (signal=%s)", signum)

    try:
        DatabasePool.get_instance().close_all()
    except Exception:
        pass

    for handler in _shutdown_handlers:
        try:
            handler()
        except Exception as e:
            logger.error("Shutdown handler error: %s", e)

    logger.info("Shutdown complete")
