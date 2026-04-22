from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_log = logging.getLogger(__name__)


class LiveCache:

    def __init__(self, default_ttl: float = 120.0):
        self._default_ttl = float(default_ttl)
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._version_cache: Dict[str, Tuple[float, str]] = {}
        self._version_lease = 2.0
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def data_version(self, db_path: str,
                      tables: Iterable[str]) -> str:
        key = db_path + '::' + ','.join(sorted(tables))
        now = time.time()
        with self._lock:
            cached = self._version_cache.get(key)
            if cached and (now - cached[0]) < self._version_lease:
                return cached[1]

        fp_parts: List[str] = []
        try:
            conn = sqlite3.connect(db_path)
            try:
                for t in tables:
                    try:
                        row = conn.execute(
                            f"SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM {t}"
                        ).fetchone()
                        fp_parts.append(f"{t}:{row[0]}:{row[1]}")
                    except sqlite3.Error:
                        fp_parts.append(f"{t}:missing")
            finally:
                conn.close()
        except sqlite3.Error as e:
            fp_parts.append(f"db_err:{e}")

        fp = hashlib.md5("|".join(fp_parts).encode("utf-8")).hexdigest()[:12]
        with self._lock:
            self._version_cache[key] = (now, fp)
        return fp

    def invalidate_version(self, db_path: str,
                           tables: Iterable[str]) -> None:
        key = db_path + '::' + ','.join(sorted(tables))
        with self._lock:
            self._version_cache.pop(key, None)

    def get_or_compute(
        self,
        namespace: str,
        params: Tuple,
        compute: Callable[[], Any],
        *,
        db_path: Optional[str] = None,
        tables: Optional[Iterable[str]] = None,
        ttl: Optional[float] = None,
    ) -> Any:
        ttl = self._default_ttl if ttl is None else float(ttl)
        version = ''
        if db_path and tables:
            version = self.data_version(db_path, list(tables))
        key = self._make_key(namespace, params, version)
        now = time.time()

        with self._lock:
            entry = self._store.get(key)
            if entry:
                if entry['expires'] > now:
                    self._hits += 1
                    return entry['value']
                else:
                    self._store.pop(key, None)

        value = compute()
        with self._lock:
            self._store[key] = {
                'value': value,
                'expires': time.time() + ttl,
                'namespace': namespace,
                'version': version,
            }
            self._misses += 1
            self._sweep_namespace(namespace, keep_version=version)
        return value

    def invalidate_namespace(self, namespace: str) -> int:
        with self._lock:
            victims = [k for k, v in self._store.items()
                        if v.get('namespace') == namespace]
            for k in victims:
                self._store.pop(k, None)
            self._invalidations += len(victims)
            return len(victims)

    def invalidate_all(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._version_cache.clear()
            self._invalidations += n
            return n

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                'hits': self._hits,
                'misses': self._misses,
                'invalidations': self._invalidations,
                'entries': len(self._store),
                'hit_rate': (self._hits / total) if total else 0.0,
                'namespaces': sorted(set(v.get('namespace', '?')
                                          for v in self._store.values())),
            }

    @staticmethod
    def _make_key(namespace: str, params: Tuple, version: str) -> str:
        try:
            pk = repr(params)
        except Exception:
            pk = str(params)
        return f"{namespace}|{version}|{pk}"

    def _sweep_namespace(self, namespace: str, keep_version: str) -> None:
        now = time.time()
        victims = [
            k for k, v in self._store.items()
            if v.get('namespace') == namespace
            and (v['expires'] < now or v.get('version') != keep_version)
        ]
        for k in victims:
            self._store.pop(k, None)


CACHE = LiveCache(default_ttl=120.0)


def cached(namespace: str,
            tables: Iterable[str],
            ttl: float = 120.0,
            db_path_arg: str = 'db_path'):
    tables = list(tables)

    def deco(fn):
        def wrapper(*args, **kwargs):
            dbp = kwargs.get(db_path_arg)
            if dbp is None and args:
                dbp = args[0]
            params = (args, tuple(sorted(kwargs.items())))
            return CACHE.get_or_compute(
                namespace=namespace,
                params=params,
                compute=lambda: fn(*args, **kwargs),
                db_path=dbp,
                tables=tables,
                ttl=ttl,
            )
        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper

    return deco
