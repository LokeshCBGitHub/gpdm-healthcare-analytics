from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import connection_config as _cc
import iics_client as _iics
import databricks_client as _dbx

try:
    import live_cache as _lc
except Exception:
    _lc = None

_log = logging.getLogger(__name__)


class GatewayError(Exception):
    pass


class _Breaker:
    def __init__(self, threshold: int = 3, cool_off_s: int = 60) -> None:
        self.threshold = threshold
        self.cool_off_s = cool_off_s
        self._state: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def _slot(self, name: str) -> Dict[str, float]:
        with self._lock:
            return self._state.setdefault(name, {'fails': 0, 'open_until': 0.0})

    def allow(self, name: str) -> bool:
        s = self._slot(name)
        return time.time() >= s['open_until']

    def on_success(self, name: str) -> None:
        s = self._slot(name)
        s['fails'] = 0
        s['open_until'] = 0.0

    def on_failure(self, name: str) -> None:
        s = self._slot(name)
        s['fails'] += 1
        if s['fails'] >= self.threshold:
            s['open_until'] = time.time() + self.cool_off_s
            _log.warning("circuit open for %s (cool_off=%ds)", name, self.cool_off_s)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._state.items()}


def _make_breaker() -> _Breaker:
    gw = _cc.get_gateway_config()
    cb = (gw.get('circuit_breaker') or {})
    return _Breaker(
        threshold=int(cb.get('fail_threshold', 3)),
        cool_off_s=int(cb.get('cool_off_s', 60)),
    )


BREAKER = _make_breaker()


def _sqlite_path() -> str:
    cfg = _cc.get_sqlite_config()
    p = cfg.get('path') or os.environ.get('GPDM_DB_PATH') or 'data/healthcare_demo.db'
    if not os.path.isabs(p):
        p = os.path.join(
            (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), p
        )
    return p


def _sqlite_query(sql: str, params: Optional[List[Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = _sqlite_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if limit is not None and 'LIMIT' not in sql.upper():
            sql = sql.rstrip(' ;') + f' LIMIT {int(limit)}'
        cur = conn.execute(sql, params or [])
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _sqlite_select_table(logical: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    phys = _cc.get_logical_table(logical, 'sqlite') or logical
    return _sqlite_query(f'SELECT * FROM {phys}', None, limit)


def _try_with_breaker(backend: str, fn) -> Tuple[bool, Any]:
    if not BREAKER.allow(backend):
        return False, GatewayError(f"{backend} circuit open")
    try:
        out = fn()
        BREAKER.on_success(backend)
        return True, out
    except Exception as e:
        BREAKER.on_failure(backend)
        _log.info("backend %s failed: %s", backend, e)
        return False, e


def _priority() -> List[str]:
    order = _cc.priority_order()
    return order or ['sqlite']


def query_dataset(name: str,
                   params: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   tables: Optional[List[str]] = None) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    tried: List[str] = []

    def _do() -> Dict[str, Any]:
        nonlocal last_err
        for backend in _priority():
            tried.append(backend)
            if backend == 'iics':
                ok, r = _try_with_breaker(backend,
                    lambda: _iics.CLIENT.query_dataset(name, params, limit))
                if ok:
                    return {'rows': r, 'backend': backend, 'tried': tried}
                last_err = r
            elif backend == 'databricks_direct':
                phys = _cc.get_logical_table(name, 'databricks') or name
                sql = f'SELECT * FROM {phys}'
                ok, r = _try_with_breaker(backend,
                    lambda: _dbx.CLIENT.execute(sql, None, limit))
                if ok:
                    return {'rows': r, 'backend': backend, 'tried': tried}
                last_err = r
            elif backend == 'sqlite':
                ok, r = _try_with_breaker(backend,
                    lambda: _sqlite_select_table(name, limit))
                if ok:
                    return {'rows': r, 'backend': backend, 'tried': tried}
                last_err = r
        raise GatewayError(
            f"all backends failed for dataset={name}: {last_err}")

    if _lc is not None and tables:
        return _lc.CACHE.get_or_compute(
            namespace=f'ds:{name}',
            key=(tuple(sorted((params or {}).items())), limit),
            tables=tables,
            db_path=_sqlite_path(),
            compute=_do,
            ttl=float(_cc.get_gateway_config().get('cache_ttl_s', 120.0)),
        )
    return _do()


def execute_sql(sql: str, params: Optional[List[Any]] = None,
                 limit: Optional[int] = None) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for backend in _priority():
        if backend == 'databricks_direct':
            ok, r = _try_with_breaker(backend,
                lambda: _dbx.CLIENT.execute(sql, params, limit))
            if ok:
                return {'rows': r, 'backend': backend}
            last_err = r
        elif backend == 'sqlite':
            ok, r = _try_with_breaker(backend,
                lambda: _sqlite_query(sql, params, limit))
            if ok:
                return {'rows': r, 'backend': backend}
            last_err = r
    raise GatewayError(f"no SQL backend succeeded: {last_err}")


def run_taskflow(name: str,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _cc.is_backend_enabled('iics'):
        return {'status': 'skipped', 'reason': 'iics disabled'}
    ok, r = _try_with_breaker('iics',
        lambda: _iics.CLIENT.run_taskflow(name, params))
    if ok:
        if _lc is not None:
            try:
                _lc.CACHE.invalidate_all()
            except Exception:
                pass
        return {'status': 'triggered', 'result': r}
    raise GatewayError(f"taskflow {name} failed: {r}")


def status() -> Dict[str, Any]:
    return {
        'config': _cc.status(),
        'priority': _priority(),
        'circuit': BREAKER.snapshot(),
        'iics': _iics.CLIENT.health() if _cc.is_backend_enabled('iics') else
                {'enabled': False},
        'databricks_direct': _dbx.CLIENT.health()
                if _cc.is_backend_enabled('databricks_direct') else
                {'enabled': False},
        'sqlite': {'enabled': _cc.is_backend_enabled('sqlite'),
                    'path': _sqlite_path(),
                    'exists': os.path.isfile(_sqlite_path())},
    }


def reset_breaker() -> Dict[str, Any]:
    global BREAKER
    BREAKER = _make_breaker()
    return {'status': 'reset'}
