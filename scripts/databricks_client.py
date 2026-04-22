from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import connection_config as _cc

_log = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None


class DatabricksError(Exception):
    pass


class _RestAdapter:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Optional[List[Any]] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if requests is None:
            raise DatabricksError("python-requests not installed")
        cfg = _cc.get_databricks_config()
        host = (cfg.get('host') or '').rstrip('/')
        token = cfg.get('token')
        warehouse = cfg.get('warehouse_id')
        if not host or not token or not warehouse:
            raise DatabricksError("databricks_direct host/token/warehouse_id missing")
        url = f"{host}/api/2.0/sql/statements"
        body: Dict[str, Any] = {
            'warehouse_id': warehouse,
            'statement': sql,
            'catalog': cfg.get('catalog') or None,
            'schema': cfg.get('schema') or None,
            'wait_timeout': '30s',
            'disposition': 'INLINE',
            'format': 'JSON_ARRAY',
        }
        if limit is not None:
            body['row_limit'] = int(limit)
        if params:
            body['parameters'] = [{'value': str(p)} for p in params]

        timeout = float(_cc.get_gateway_config().get('databricks_timeout_s', 30))
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        resp = requests.post(url, json=body, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise DatabricksError(f"dbx rest: {resp.status_code} {resp.text[:200]}")
        payload = resp.json() if resp.content else {}
        state = (payload.get('status') or {}).get('state', 'SUCCEEDED')
        stmt_id = payload.get('statement_id')
        deadline = time.time() + timeout
        while state in ('PENDING', 'RUNNING') and stmt_id and time.time() < deadline:
            time.sleep(0.5)
            pr = requests.get(f"{url}/{stmt_id}", headers=headers, timeout=timeout)
            if pr.status_code >= 400:
                raise DatabricksError(f"dbx poll: {pr.status_code} {pr.text[:200]}")
            payload = pr.json()
            state = (payload.get('status') or {}).get('state')
        if state != 'SUCCEEDED':
            err = (payload.get('status') or {}).get('error') or {}
            raise DatabricksError(f"dbx state={state}: {err.get('message', '')}")

        manifest = payload.get('manifest') or {}
        schema = (manifest.get('schema') or {}).get('columns') or []
        col_names = [c.get('name') for c in schema]
        result = payload.get('result') or {}
        data = result.get('data_array') or []
        return [dict(zip(col_names, row)) for row in data]


class _SqlConnAdapter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conn = None

    def _get_conn(self):
        try:
            from databricks import sql as dbx_sql
        except ImportError as e:
            raise DatabricksError(
                "databricks-sql-connector not installed; "
                "pip install databricks-sql-connector") from e
        cfg = _cc.get_databricks_config()
        host = cfg.get('host', '').replace('https://', '').rstrip('/')
        with self._lock:
            if self._conn is None:
                self._conn = dbx_sql.connect(
                    server_hostname=host,
                    http_path=cfg.get('http_path'),
                    access_token=cfg.get('token'),
                    catalog=cfg.get('catalog') or None,
                    schema=cfg.get('schema') or None,
                )
            return self._conn

    def execute(self, sql: str, params: Optional[List[Any]] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            if limit is not None and 'LIMIT' not in sql.upper():
                sql = sql.rstrip(' ;') + f' LIMIT {int(limit)}'
            cur.execute(sql, params or [])
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            cur.close()


class DatabricksClient:
    def __init__(self) -> None:
        self._rest = _RestAdapter()
        self._sqlconn = _SqlConnAdapter()

    def enabled(self) -> bool:
        return _cc.is_backend_enabled('databricks_direct')

    def _adapter(self):
        cfg = _cc.get_databricks_config()
        mode = (cfg.get('execution') or 'rest').lower()
        return self._sqlconn if mode == 'sqlconn' else self._rest

    def execute(self, sql: str, params: Optional[List[Any]] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.enabled():
            raise DatabricksError("databricks_direct not enabled")
        cfg = _cc.get_databricks_config()
        retry = cfg.get('retry') or {}
        max_attempts = int(retry.get('max_attempts', 2))
        base = float(retry.get('backoff_base_s', 2.0))
        last: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._adapter().execute(sql, params, limit)
            except DatabricksError as e:
                last = e
                if attempt < max_attempts:
                    time.sleep(base ** attempt)
        raise last or DatabricksError("databricks retry exhausted")

    def health(self) -> Dict[str, Any]:
        cfg = _cc.get_databricks_config()
        return {
            'enabled': self.enabled(),
            'execution': (cfg.get('execution') or 'rest'),
            'host_configured': bool(cfg.get('host')),
            'warehouse_configured': bool(cfg.get('warehouse_id')),
            'requests_available': requests is not None,
        }


CLIENT = DatabricksClient()
