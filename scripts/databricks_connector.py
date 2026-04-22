import os
import re
import json
import ssl
import time
import threading
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

try:
    from production import get_logger, audit_log
except ImportError:
    import logging
    def get_logger(name='gpdm'):
        return logging.getLogger(name)
    def audit_log(event, **kwargs):
        pass


DEFAULT_CATALOG = 'healthcare_prod'
DEFAULT_SCHEMA = 'analytics'
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY = 60


class DatabricksConfig:

    def __init__(self):
        self.host: str = ''
        self.token: str = ''
        self.client_id: str = ''
        self.client_secret: str = ''
        self.warehouse_id: str = ''
        self.catalog: str = DEFAULT_CATALOG
        self.schema: str = DEFAULT_SCHEMA
        self.timeout: int = DEFAULT_TIMEOUT
        self.max_retries: int = DEFAULT_MAX_RETRIES
        self.http_path: str = ''
        self.auth_type: str = ''

    @classmethod
    def from_environment(cls) -> 'DatabricksConfig':
        cfg = cls()
        cfg.host = os.environ.get('DATABRICKS_HOST', '').strip().rstrip('/')
        cfg.token = os.environ.get('DATABRICKS_TOKEN', '').strip()
        cfg.client_id = os.environ.get('DATABRICKS_CLIENT_ID', '').strip()
        cfg.client_secret = os.environ.get('DATABRICKS_CLIENT_SECRET', '').strip()
        cfg.warehouse_id = os.environ.get('DATABRICKS_WAREHOUSE_ID', '').strip()
        cfg.catalog = os.environ.get('DATABRICKS_CATALOG', DEFAULT_CATALOG).strip()
        cfg.schema = os.environ.get('DATABRICKS_SCHEMA', DEFAULT_SCHEMA).strip()
        cfg.http_path = os.environ.get('DATABRICKS_HTTP_PATH', '').strip()
        timeout_str = os.environ.get('DATABRICKS_TIMEOUT', str(DEFAULT_TIMEOUT))
        try:
            cfg.timeout = int(timeout_str)
        except ValueError:
            cfg.timeout = DEFAULT_TIMEOUT
        cfg._detect_auth_type()
        return cfg

    @classmethod
    def from_config_file(cls, config_path: str) -> 'DatabricksConfig':
        cfg = cls()
        if not os.path.exists(config_path):
            return cfg

        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip().upper()
                value = value.strip().strip('"').strip("'")

                mapping = {
                    'DATABRICKS_HOST': 'host',
                    'DATABRICKS_TOKEN': 'token',
                    'DATABRICKS_CLIENT_ID': 'client_id',
                    'DATABRICKS_CLIENT_SECRET': 'client_secret',
                    'DATABRICKS_WAREHOUSE_ID': 'warehouse_id',
                    'DATABRICKS_CATALOG': 'catalog',
                    'DATABRICKS_SCHEMA': 'schema',
                    'DATABRICKS_HTTP_PATH': 'http_path',
                    'DATABRICKS_TIMEOUT': 'timeout',
                    'DATABRICKS_MAX_RETRIES': 'max_retries',
                    'HOST': 'host',
                    'TOKEN': 'token',
                    'CLIENT_ID': 'client_id',
                    'CLIENT_SECRET': 'client_secret',
                    'WAREHOUSE_ID': 'warehouse_id',
                    'CATALOG': 'catalog',
                    'SCHEMA': 'schema',
                    'HTTP_PATH': 'http_path',
                    'TIMEOUT': 'timeout',
                }

                attr = mapping.get(key)
                if attr:
                    if attr in ('timeout', 'max_retries'):
                        try:
                            setattr(cfg, attr, int(value))
                        except ValueError:
                            pass
                    else:
                        setattr(cfg, attr, value.rstrip('/'))

        cfg._detect_auth_type()
        return cfg

    @classmethod
    def auto_discover(cls, base_dir: str = None) -> 'DatabricksConfig':
        logger = get_logger('gpdm.databricks')

        cfg = cls()

        if base_dir is None:
            base_dir = str(Path(__file__).resolve().parent.parent)

        db_cfg_path = os.path.join(base_dir, 'paramset', 'databricks.cfg')
        if os.path.exists(db_cfg_path):
            logger.info("Loading Databricks config from: %s", db_cfg_path)
            cfg = cls.from_config_file(db_cfg_path)

        main_cfg_path = os.path.join(base_dir, 'paramset', 'gpdm_chatbot.cfg')
        if os.path.exists(main_cfg_path):
            main_cfg = cls.from_config_file(main_cfg_path)
            for attr in ('host', 'token', 'client_id', 'client_secret',
                         'warehouse_id', 'catalog', 'schema', 'http_path'):
                if not getattr(cfg, attr) and getattr(main_cfg, attr):
                    setattr(cfg, attr, getattr(main_cfg, attr))

        env_cfg = cls.from_environment()
        for attr in ('host', 'token', 'client_id', 'client_secret',
                     'warehouse_id', 'catalog', 'schema', 'http_path'):
            env_val = getattr(env_cfg, attr)
            if env_val:
                setattr(cfg, attr, env_val)

        cfg._detect_auth_type()
        return cfg

    def _detect_auth_type(self) -> None:
        if self.token:
            self.auth_type = 'pat'
        elif self.client_id and self.client_secret:
            self.auth_type = 'oauth'
        else:
            self.auth_type = ''

    def is_configured(self) -> bool:
        return bool(self.host and self.warehouse_id and self.auth_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'warehouse_id': self.warehouse_id,
            'catalog': self.catalog,
            'schema': self.schema,
            'auth_type': self.auth_type,
            'http_path': self.http_path,
            'timeout': self.timeout,
            'is_configured': self.is_configured(),
        }


class OAuthTokenManager:

    def __init__(self, host: str, client_id: str, client_secret: str):
        self._host = host.rstrip('/')
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0
        self._lock = threading.Lock()
        self._logger = get_logger('gpdm.databricks.oauth')

    def get_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token

            self._token = self._fetch_token()
            return self._token

    def reset_token(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = 0

    def _fetch_token(self) -> str:
        token_url = f"https://{self._host}/oidc/v1/token"
        data = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'scope': 'all-apis',
        }).encode('utf-8')

        import base64
        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()

        req = urllib.request.Request(
            token_url,
            data=data,
            headers={
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            method='POST'
        )

        ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                body = json.loads(resp.read().decode('utf-8'))
                token = body['access_token']
                expires_in = body.get('expires_in', 3600)
                self._expires_at = time.time() + expires_in
                self._logger.info("OAuth token acquired (expires in %ds)", expires_in)
                return token
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            self._logger.error("OAuth token error %d: %s", e.code, error_body[:200])
            raise ConnectionError(f"OAuth token request failed: HTTP {e.code}")
        except Exception as e:
            self._logger.error("OAuth token error: %s", e)
            raise ConnectionError(f"OAuth token request failed: {e}")


class CircuitBreaker:

    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD, recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        return self.state != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN


class DatabricksConnector:

    _instance: Optional['DatabricksConnector'] = None
    _init_lock = threading.Lock()

    API_PATH = '/api/2.0/sql/statements'

    def __init__(self, config: DatabricksConfig = None):
        self._config = config or DatabricksConfig()
        self._oauth_manager: Optional[OAuthTokenManager] = None
        self._circuit_breaker = CircuitBreaker(failure_threshold=CIRCUIT_BREAKER_THRESHOLD, recovery_timeout=CIRCUIT_BREAKER_RECOVERY)
        self._logger = get_logger('gpdm.databricks')
        self._ssl_context = ssl.create_default_context()
        self._connected = False
        self._stats = {
            'queries_executed': 0,
            'queries_failed': 0,
            'total_response_time_ms': 0,
            'last_query_time': None,
        }

        if self._config.auth_type == 'oauth':
            self._oauth_manager = OAuthTokenManager(
                self._config.host,
                self._config.client_id,
                self._config.client_secret
            )

    @classmethod
    def get_instance(cls, config: DatabricksConfig = None) -> 'DatabricksConnector':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._init_lock:
            cls._instance = None

    def _get_auth_header(self) -> str:
        if self._config.auth_type == 'pat':
            return f'Bearer {self._config.token}'
        elif self._config.auth_type == 'oauth' and self._oauth_manager:
            token = self._oauth_manager.get_token()
            return f'Bearer {token}'
        else:
            raise ConnectionError("No valid authentication configured")

    def test_connection(self) -> Dict[str, Any]:
        if not self._config.is_configured():
            return {
                'connected': False,
                'error': 'Databricks not configured',
                'config': self._config.to_dict(),
            }

        try:
            success, rows, error = self.execute_query("SELECT 1 AS health_check")
            if success:
                self._connected = True
                self._logger.info("Databricks connection verified: %s", self._config.host)
                return {
                    'connected': True,
                    'host': self._config.host,
                    'warehouse_id': self._config.warehouse_id,
                    'catalog': self._config.catalog,
                    'schema': self._config.schema,
                    'auth_type': self._config.auth_type,
                }
            else:
                return {'connected': False, 'error': error}
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def execute_query(self, sql: str, timeout: int = None
                      ) -> Tuple[bool, List[Dict], str]:
        t0 = time.time()
        timeout = timeout or self._config.timeout

        if not self._circuit_breaker.allow_request():
            return False, [], "Circuit breaker OPEN — Databricks temporarily unavailable"

        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            return False, [], "Only SELECT queries are allowed"

        try:
            auth_header = self._get_auth_header()
        except ConnectionError as e:
            return False, [], str(e)

        url = f"https://{self._config.host}{self.API_PATH}"
        payload = {
            'statement': sql,
            'warehouse_id': self._config.warehouse_id,
            'wait_timeout': f'{timeout}s',
            'disposition': 'INLINE',
            'format': 'JSON_ARRAY',
        }

        if self._config.catalog:
            payload['catalog'] = self._config.catalog
        if self._config.schema:
            payload['schema'] = self._config.schema

        body = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                'Authorization': auth_header,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST'
        )

        last_error = ''
        for attempt in range(1, self._config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    req, context=self._ssl_context, timeout=timeout + 10
                ) as resp:
                    response_body = json.loads(resp.read().decode('utf-8'))

                status = response_body.get('status', {}).get('state', 'UNKNOWN')

                if status == 'SUCCEEDED':
                    rows = self._parse_response(response_body)
                    elapsed = (time.time() - t0) * 1000

                    self._circuit_breaker.record_success()
                    self._stats['queries_executed'] += 1
                    self._stats['total_response_time_ms'] += elapsed
                    self._stats['last_query_time'] = datetime.now(timezone.utc).isoformat()

                    self._logger.info(
                        "Databricks query OK: %.0fms, %d rows", elapsed, len(rows)
                    )
                    audit_log(
                        event='DATABRICKS_QUERY',
                        details={
                            'success': True,
                            'row_count': len(rows),
                            'response_time_ms': round(elapsed, 1),
                            'question_length': len(sql),
                        }
                    )
                    return True, rows, ''

                elif status == 'FAILED':
                    error_msg = response_body.get('status', {}).get('error', {})
                    last_error = error_msg.get('message', 'Query execution failed')
                    self._logger.warning(
                        "Databricks query FAILED (attempt %d): %s",
                        attempt, last_error[:200]
                    )
                    break

                elif status in ('PENDING', 'RUNNING'):
                    statement_id = response_body.get('statement_id', '')
                    if statement_id:
                        success, rows, error = self._poll_statement(
                            statement_id, auth_header, timeout, t0
                        )
                        if success:
                            return True, rows, ''
                        last_error = error
                    else:
                        last_error = f"Query in state {status} with no statement_id"
                    break

                elif status == 'CANCELED':
                    last_error = 'Query was canceled'
                    break

                else:
                    last_error = f"Unexpected status: {status}"

            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='replace')[:500]
                last_error = f"HTTP {e.code}: {error_body}"

                if e.code == 429:
                    wait_time = min(2 ** attempt, 30)
                    self._logger.warning(
                        "Rate limited (attempt %d), waiting %ds", attempt, wait_time
                    )
                    time.sleep(wait_time)
                    continue
                elif e.code in (502, 503, 504):
                    wait_time = min(2 ** attempt, 15)
                    self._logger.warning(
                        "Server error %d (attempt %d), retrying in %ds",
                        e.code, attempt, wait_time
                    )
                    time.sleep(wait_time)
                    continue
                elif e.code == 401:
                    if self._oauth_manager and attempt == 1:
                        self._oauth_manager.reset_token()
                        continue
                    last_error = "Authentication failed"
                    break
                elif e.code == 403:
                    last_error = "Access denied — check permissions"
                    break
                else:
                    break

            except urllib.error.URLError as e:
                last_error = f"Connection error: {e.reason}"
                wait_time = min(2 ** attempt, 10)
                self._logger.warning(
                    "Connection error (attempt %d): %s", attempt, e.reason
                )
                time.sleep(wait_time)
                continue

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                self._logger.error("Unexpected error: %s", e, exc_info=True)
                break

        elapsed = (time.time() - t0) * 1000
        self._circuit_breaker.record_failure()
        self._stats['queries_failed'] += 1

        self._logger.error("Databricks query FAILED after %d attempts: %s",
                          self._config.max_retries, last_error[:200])
        audit_log(
            event='DATABRICKS_QUERY_FAILED',
            details={
                'success': False,
                'error_type': last_error[:50],
                'response_time_ms': round(elapsed, 1),
            }
        )
        return False, [], last_error

    def _poll_statement(self, statement_id: str, auth_header: str,
                        timeout: int, start_time: float
                        ) -> Tuple[bool, List[Dict], str]:
        poll_url = f"https://{self._config.host}{self.API_PATH}/{statement_id}"

        while time.time() - start_time < timeout:
            req = urllib.request.Request(
                poll_url,
                headers={
                    'Authorization': auth_header,
                    'Accept': 'application/json',
                },
                method='GET'
            )

            try:
                with urllib.request.urlopen(
                    req, context=self._ssl_context, timeout=30
                ) as resp:
                    body = json.loads(resp.read().decode('utf-8'))

                status = body.get('status', {}).get('state', 'UNKNOWN')

                if status == 'SUCCEEDED':
                    rows = self._parse_response(body)
                    elapsed = (time.time() - start_time) * 1000
                    self._circuit_breaker.record_success()
                    self._stats['queries_executed'] += 1
                    self._stats['total_response_time_ms'] += elapsed
                    return True, rows, ''
                elif status == 'FAILED':
                    error = body.get('status', {}).get('error', {}).get('message', 'Failed')
                    return False, [], error
                elif status == 'CANCELED':
                    return False, [], 'Query canceled'
                else:
                    time.sleep(1)

            except Exception as e:
                return False, [], f"Poll error: {str(e)}"

        return False, [], f"Query timed out after {timeout}s"

    def _parse_response(self, response: Dict) -> List[Dict]:
        result = response.get('result', response.get('manifest', {}))

        columns = []
        schema = (response.get('manifest', {}).get('schema', {})
                  .get('columns', []))
        if schema:
            columns = [col.get('name', f'col_{i}') for i, col in enumerate(schema)]

        data_array = response.get('result', {}).get('data_array', [])

        if not data_array:
            chunks = response.get('result', {}).get('external_links', [])
            if chunks:
                pass

        if not columns and data_array:
            columns = [f'col_{i}' for i in range(len(data_array[0]))]

        rows = []
        for row_data in data_array:
            row_dict = {}
            for i, col_name in enumerate(columns):
                if i < len(row_data):
                    row_dict[col_name] = row_data[i]
                else:
                    row_dict[col_name] = None
            rows.append(row_dict)

        return rows

    def get_tables(self) -> List[str]:
        sql = f"SHOW TABLES IN {self._config.catalog}.{self._config.schema}"
        success, rows, error = self.execute_query(sql)
        if success:
            return [r.get('tableName', r.get('table_name', '')) for r in rows
                    if r.get('tableName') or r.get('table_name')]
        return []

    def get_table_schema(self, table_name: str) -> List[Dict]:
        if not re.match(r'^[a-zA-Z_]\w{0,127}$', table_name):
            return []
        fqn = f"{self._config.catalog}.{self._config.schema}.{table_name}"
        sql = f"DESCRIBE TABLE {fqn}"
        success, rows, error = self.execute_query(sql)
        if success:
            return rows
        return []

    def health_check(self) -> Dict[str, Any]:
        if not self._config.is_configured():
            return {
                'status': 'not_configured',
                'config': self._config.to_dict(),
            }

        try:
            success, rows, error = self.execute_query("SELECT 1 AS check_val")
            if success:
                avg_ms = (self._stats['total_response_time_ms'] /
                          max(1, self._stats['queries_executed']))
                return {
                    'status': 'healthy',
                    'host': self._config.host,
                    'auth_type': self._config.auth_type,
                    'circuit_breaker': self._circuit_breaker.state,
                    'stats': {
                        'queries_executed': self._stats['queries_executed'],
                        'queries_failed': self._stats['queries_failed'],
                        'avg_response_ms': round(avg_ms, 1),
                    },
                }
            else:
                return {'status': 'unhealthy', 'error': error}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class DataSourceManager:

    DATABRICKS = 'databricks'
    LOCAL = 'local'

    _instance: Optional['DataSourceManager'] = None

    def __init__(self):
        self._mode: str = self.LOCAL
        self._databricks: Optional[DatabricksConnector] = None
        self._db_config: Optional[DatabricksConfig] = None
        self._logger = get_logger('gpdm.datasource')
        self._initialized = False

    @classmethod
    def get_instance(cls) -> 'DataSourceManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, base_dir: str = None, force_mode: str = None
                   ) -> Dict[str, Any]:
        if force_mode:
            self._mode = force_mode
            if force_mode == self.DATABRICKS:
                return self._init_databricks(base_dir)
            else:
                self._initialized = True
                return {'mode': self.LOCAL, 'reason': 'forced'}

        self._db_config = DatabricksConfig.auto_discover(base_dir)

        if self._db_config.is_configured():
            self._logger.info("Databricks config found: host=%s, auth=%s",
                              self._db_config.host, self._db_config.auth_type)
            result = self._init_databricks(base_dir)
            if result.get('connected'):
                return result
            else:
                self._logger.warning(
                    "Databricks connection failed, falling back to local: %s",
                    result.get('error', 'unknown')
                )

        self._mode = self.LOCAL
        self._initialized = True
        self._logger.info("Using local data source (Databricks not configured)")
        return {
            'mode': self.LOCAL,
            'reason': 'databricks_not_configured' if not self._db_config.is_configured()
                      else 'databricks_connection_failed',
        }

    def _init_databricks(self, base_dir: str = None) -> Dict[str, Any]:
        try:
            self._databricks = DatabricksConnector(self._db_config)
            conn_test = self._databricks.test_connection()

            if conn_test.get('connected'):
                self._mode = self.DATABRICKS
                self._initialized = True
                self._logger.info("Connected to Databricks: %s", self._db_config.host)
                return {
                    'mode': self.DATABRICKS,
                    'connected': True,
                    **conn_test,
                }
            else:
                return {
                    'mode': self.LOCAL,
                    'connected': False,
                    'error': conn_test.get('error', 'Connection test failed'),
                }
        except Exception as e:
            self._logger.error("Databricks init error: %s", e)
            return {
                'mode': self.LOCAL,
                'connected': False,
                'error': str(e),
            }

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_databricks(self) -> bool:
        return self._mode == self.DATABRICKS

    def execute_query(self, sql: str, local_fallback_fn=None,
                      cfg: Dict = None) -> Tuple[bool, List[Dict], str]:
        if self._mode == self.DATABRICKS and self._databricks:
            spark_sql = self._translate_to_spark(sql)

            success, rows, error = self._databricks.execute_query(spark_sql)

            if success:
                return True, rows, ''

            if local_fallback_fn:
                self._logger.warning(
                    "Databricks query failed, falling back to local: %s", error[:100]
                )
                return local_fallback_fn(sql, cfg or {})
            else:
                return False, [], error

        if local_fallback_fn:
            return local_fallback_fn(sql, cfg or {})

        return False, [], "No data source available"

    def _translate_to_spark(self, sqlite_sql: str) -> str:
        try:
            from databricks_dialect import DatabricksDialect as NewDialect
            catalog = self._db_config.catalog if self._db_config else 'healthcare'
            schema = self._db_config.schema if self._db_config else 'production'
            dialect = NewDialect(catalog=catalog, schema=schema)
            return dialect.translate(sqlite_sql)
        except ImportError:
            pass

        try:
            from graph_vector_engine import DatabricksDialect
            tables = re.findall(r'\bFROM\s+"?(\w+)"?', sqlite_sql, re.IGNORECASE)
            tables += re.findall(r'\bJOIN\s+"?(\w+)"?', sqlite_sql, re.IGNORECASE)
            tables = list(set(tables))
            if self._db_config:
                DatabricksDialect.DEFAULT_CATALOG = self._db_config.catalog
                DatabricksDialect.DEFAULT_SCHEMA = self._db_config.schema
            return DatabricksDialect.translate(
                sqlite_sql, tables=tables, use_catalog=True
            )
        except ImportError:
            sql = sqlite_sql
            sql = sql.replace('AS REAL)', 'AS DOUBLE)')
            sql = re.sub(r"SUBSTR\((\w+), 1, 7\)", r"DATE_FORMAT(\1, 'yyyy-MM')", sql)
            sql = re.sub(r"SUBSTR\((\w+), 1, 4\)", r"YEAR(\1)", sql)
            sql = re.sub(r"(\w+) LIKE '(\d{4})%'", r"YEAR(\1) = \2", sql)

            if self._db_config:
                prefix = f"{self._db_config.catalog}.{self._db_config.schema}"
                tables = re.findall(r'\bFROM\s+"?(\w+)"?', sql, re.IGNORECASE)
                tables += re.findall(r'\bJOIN\s+"?(\w+)"?', sql, re.IGNORECASE)
                for table in set(tables):
                    sql = re.sub(
                        rf'\bFROM\s+{table}\b',
                        f'FROM {prefix}.{table}',
                        sql, count=1
                    )
                    sql = re.sub(
                        rf'\bJOIN\s+{table}\b',
                        f'JOIN {prefix}.{table}',
                        sql, count=1
                    )

            return sql

    def health_check(self) -> Dict[str, Any]:
        result = {
            'mode': self._mode,
            'initialized': self._initialized,
        }

        if self._databricks:
            result['databricks'] = self._databricks.health_check()
        else:
            result['databricks'] = {'status': 'not_configured'}

        return result


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Databricks Connector — Configuration Check")
    print("=" * 60)

    base_dir = str(Path(__file__).resolve().parent.parent)
    config = DatabricksConfig.auto_discover(base_dir)

    print(f"\nConfiguration:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")

    if config.is_configured():
        print(f"\nTesting connection to {config.host}...")
        connector = DatabricksConnector(config)
        result = connector.test_connection()
        print(f"  Connected: {result.get('connected', False)}")
        if result.get('error'):
            print(f"  Error: {result['error']}")
    else:
        print("\nDatabricks NOT configured.")
        print("To configure, create paramset/databricks.cfg with:")
        print("  DATABRICKS_HOST=your-workspace.cloud.databricks.com")
        print("  DATABRICKS_TOKEN=dapi_xxxxx  (or CLIENT_ID + CLIENT_SECRET)")
        print("  DATABRICKS_WAREHOUSE_ID=abc123def456")
        print("  DATABRICKS_CATALOG=healthcare_prod")
        print("  DATABRICKS_SCHEMA=analytics")

    print(f"\nData Source Manager:")
    mgr = DataSourceManager.get_instance()
    ds_result = mgr.initialize(base_dir)
    print(f"  Mode: {ds_result['mode']}")
    if ds_result.get('reason'):
        print(f"  Reason: {ds_result['reason']}")
    if ds_result.get('connected'):
        print(f"  Connected: {ds_result['connected']}")
