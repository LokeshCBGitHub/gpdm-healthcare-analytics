import os
import sqlite3
import hashlib
import json
import time
import re
import logging
import pickle
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

try:
    from gpdm_config import (
        SCHEMA_CACHE_TTL_HOURS, QUERY_CACHE_TTL_VOLATILE,
        QUERY_CACHE_TTL_REFERENCE, CACHE_FILLER_WORDS,
    )
except ImportError:
    SCHEMA_CACHE_TTL_HOURS = 24.0
    QUERY_CACHE_TTL_VOLATILE = 1.0
    QUERY_CACHE_TTL_REFERENCE = 24.0
    CACHE_FILLER_WORDS = frozenset({'what', 'is', 'the', 'show', 'me', 'give', 'find', 'get',
        'list', 'display', 'tell', 'how', 'many', 'which', 'are', 'were', 'was', 'has',
        'have', 'does', 'do', 'a', 'an', 'of', 'for', 'in', 'by', 'to', 'from', 'with',
        'each', 'every', 'all', 'please', 'can', 'you', 'i', 'want', 'need', 'would',
        'like', 'could'})

logger = logging.getLogger('gpdm.cache')


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    sql_hits: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'semantic_hits': self.semantic_hits,
            'sql_hits': self.sql_hits,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate
        }


class CacheManager:

    def __init__(self, data_dir: str, use_redis: bool = False, redis_url: str = None,
                 db_path: str = None):
        self.data_dir = data_dir
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.cache_db = os.path.join(data_dir, 'query_cache.db')
        self.schema_cache_db = os.path.join(data_dir, 'schema_cache.db')
        self.intelligence_cache_dir = os.path.join(data_dir, 'intelligence_cache')
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._db_path = db_path

        self._schema_fingerprint = ''
        if db_path:
            try:
                from schema_persistence import compute_db_fingerprint
                self._schema_fingerprint = compute_db_fingerprint(db_path)
            except Exception as e:
                logger.warning("Could not compute schema fingerprint: %s", e)

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.intelligence_cache_dir, exist_ok=True)

        self._init_db()

        if self._schema_fingerprint:
            self._invalidate_stale_schema_results()

        logger.info(f"CacheManager initialized with data_dir={data_dir}, "
                     f"schema_fingerprint={self._schema_fingerprint[:8] if self._schema_fingerprint else 'none'}")

    def _init_db(self):
        with self._get_connection(self.cache_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    semantic_key TEXT,
                    sql_hash TEXT,
                    question TEXT,
                    sql_text TEXT,
                    result_json TEXT,
                    created_at REAL,
                    ttl_hours REAL,
                    hit_count INTEGER DEFAULT 0,
                    last_hit_at REAL,
                    is_reference INTEGER DEFAULT 0,
                    schema_fingerprint TEXT DEFAULT ''
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_semantic ON query_cache(semantic_key)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sql ON query_cache(sql_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON query_cache(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_schema_fp ON query_cache(schema_fingerprint)')
            try:
                conn.execute('ALTER TABLE query_cache ADD COLUMN schema_fingerprint TEXT DEFAULT ""')
            except Exception:
                pass
            conn.commit()

        with self._get_connection(self.schema_cache_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_cache (
                    db_path TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    schema_json TEXT,
                    cached_at REAL,
                    ttl_hours REAL DEFAULT 24.0
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_fingerprint ON schema_cache(fingerprint)')
            conn.commit()

        with self._get_connection(self.schema_cache_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS intelligence_cache (
                    key TEXT PRIMARY KEY,
                    data_json TEXT,
                    cached_at REAL,
                    schema_fingerprint TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_intel_fingerprint ON intelligence_cache(schema_fingerprint)')
            conn.commit()

    def _invalidate_stale_schema_results(self):
        try:
            with self._get_connection(self.cache_db) as conn:
                stale = conn.execute(
                    'SELECT COUNT(*) FROM query_cache WHERE schema_fingerprint != ? AND schema_fingerprint != ""',
                    (self._schema_fingerprint,)
                ).fetchone()[0]

                if stale > 0:
                    conn.execute(
                        'DELETE FROM query_cache WHERE schema_fingerprint != ? AND schema_fingerprint != ""',
                        (self._schema_fingerprint,)
                    )
                    conn.execute(
                        'DELETE FROM query_cache WHERE schema_fingerprint = "" OR schema_fingerprint IS NULL'
                    )
                    conn.commit()
                    logger.info("Invalidated %d stale cache entries (schema changed)", stale)
                else:
                    legacy = conn.execute(
                        'SELECT COUNT(*) FROM query_cache WHERE schema_fingerprint = "" OR schema_fingerprint IS NULL'
                    ).fetchone()[0]
                    if legacy > 0:
                        conn.execute(
                            'DELETE FROM query_cache WHERE schema_fingerprint = "" OR schema_fingerprint IS NULL'
                        )
                        conn.commit()
                        logger.info("Cleaned up %d legacy cache entries (no schema fingerprint)", legacy)
        except Exception as e:
            logger.warning("Could not invalidate stale cache: %s", e)

    @contextmanager
    def _get_connection(self, db_path: str):
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


    def get_schema(self, db_path: str) -> Optional[Dict]:
        with self._lock:
            try:
                with self._get_connection(self.schema_cache_db) as conn:
                    row = conn.execute(
                        'SELECT schema_json, cached_at, ttl_hours FROM schema_cache WHERE db_path = ?',
                        (db_path,)
                    ).fetchone()

                    if not row:
                        logger.debug(f"Schema cache miss for {db_path}")
                        return None

                    age_hours = (time.time() - row['cached_at']) / 3600
                    if age_hours > row['ttl_hours']:
                        logger.debug(f"Schema cache expired for {db_path}")
                        return None

                    logger.debug(f"Schema cache hit for {db_path}")
                    return json.loads(row['schema_json'])
            except Exception as e:
                logger.error(f"Error retrieving schema cache: {e}")
                return None

    def set_schema(self, db_path: str, schema_data: Dict, ttl_hours: float = None):
        if ttl_hours is None:
            ttl_hours = SCHEMA_CACHE_TTL_HOURS
        with self._lock:
            try:
                fingerprint = self._compute_schema_fingerprint(db_path)

                with self._get_connection(self.schema_cache_db) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO schema_cache
                        (db_path, fingerprint, schema_json, cached_at, ttl_hours)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (db_path, fingerprint, json.dumps(schema_data), time.time(), ttl_hours))
                    conn.commit()

                logger.info(f"Schema cached for {db_path} with fingerprint {fingerprint[:8]}")
            except Exception as e:
                logger.error(f"Error setting schema cache: {e}")

    def _compute_schema_fingerprint(self, db_path: str) -> str:
        try:
            if not os.path.exists(db_path):
                return hashlib.md5(b'nonexistent').hexdigest()

            with self._get_connection(db_path) as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()

                fingerprint_parts = []
                for table in tables:
                    table_name = table[0]
                    fingerprint_parts.append(f"TABLE:{table_name}")

                    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                    for col in columns:
                        fingerprint_parts.append(f"COL:{col[1]}:{col[2]}")

                    try:
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        fingerprint_parts.append(f"ROWS:{row_count}")
                    except Exception:
                        fingerprint_parts.append("ROWS:0")

                fingerprint_str = '|'.join(fingerprint_parts)
                return hashlib.md5(fingerprint_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing schema fingerprint for {db_path}: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()

    def is_schema_stale(self, db_path: str) -> bool:
        with self._lock:
            try:
                with self._get_connection(self.schema_cache_db) as conn:
                    row = conn.execute(
                        'SELECT fingerprint FROM schema_cache WHERE db_path = ?',
                        (db_path,)
                    ).fetchone()

                    if not row:
                        return True

                    current_fingerprint = self._compute_schema_fingerprint(db_path)
                    is_stale = row['fingerprint'] != current_fingerprint

                    if is_stale:
                        logger.info(f"Schema detected as stale for {db_path}")

                    return is_stale
            except Exception as e:
                logger.error(f"Error checking schema staleness: {e}")
                return True


    def get_query_result(self, question: str, sql: str = None) -> Optional[Dict]:
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    semantic_key = self._semantic_cache_key(question)
                    fp_clause = ''
                    fp_params = [semantic_key]
                    if self._schema_fingerprint:
                        fp_clause = ' AND schema_fingerprint = ?'
                        fp_params.append(self._schema_fingerprint)
                    row = conn.execute(
                        f'SELECT result_json, created_at, ttl_hours FROM query_cache WHERE semantic_key = ?{fp_clause} ORDER BY created_at DESC LIMIT 1',
                        fp_params
                    ).fetchone()

                    if row:
                        age_hours = (time.time() - row['created_at']) / 3600
                        if age_hours <= row['ttl_hours']:
                            self.stats.hits += 1
                            self.stats.semantic_hits += 1

                            cache_key = hashlib.md5((semantic_key + str(row['created_at'])).encode()).hexdigest()
                            conn.execute(
                                'UPDATE query_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE cache_key = ?',
                                (time.time(), cache_key)
                            )
                            conn.commit()

                            logger.debug(f"Query result hit (semantic) for question: {question[:50]}")
                            return json.loads(row['result_json'])

                    if sql:
                        sql_hash = self._sql_cache_key(sql)
                        sql_params = [sql_hash]
                        if self._schema_fingerprint:
                            sql_params.append(self._schema_fingerprint)
                        row = conn.execute(
                            f'SELECT result_json, created_at, ttl_hours FROM query_cache WHERE sql_hash = ?{fp_clause} ORDER BY created_at DESC LIMIT 1',
                            sql_params
                        ).fetchone()

                        if row:
                            age_hours = (time.time() - row['created_at']) / 3600
                            if age_hours <= row['ttl_hours']:
                                self.stats.hits += 1
                                self.stats.sql_hits += 1
                                logger.debug(f"Query result hit (SQL) for question: {question[:50]}")
                                return json.loads(row['result_json'])

                    self.stats.misses += 1
                    logger.debug(f"Query result miss for question: {question[:50]}")
                    return None
            except Exception as e:
                logger.error(f"Error retrieving query result: {e}")
                self.stats.misses += 1
                return None

    def set_query_result(self, question: str, sql: str, result: Dict,
                        ttl_hours: float = None, is_reference: bool = False):
        with self._lock:
            try:
                semantic_key = self._semantic_cache_key(question)
                sql_hash = self._sql_cache_key(sql)
                cache_key = hashlib.md5((semantic_key + sql_hash + str(time.time())).encode()).hexdigest()

                if is_reference:
                    ttl_hours = QUERY_CACHE_TTL_REFERENCE
                elif ttl_hours is None:
                    ttl_hours = QUERY_CACHE_TTL_VOLATILE

                with self._get_connection(self.cache_db) as conn:
                    conn.execute('''
                        INSERT INTO query_cache
                        (cache_key, semantic_key, sql_hash, question, sql_text, result_json, created_at, ttl_hours, is_reference, schema_fingerprint)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (cache_key, semantic_key, sql_hash, question, sql,
                          json.dumps(result), time.time(), ttl_hours, 1 if is_reference else 0,
                          self._schema_fingerprint))
                    conn.commit()

                logger.info(f"Query result cached for: {question[:50]} (TTL: {ttl_hours}h)")
            except Exception as e:
                logger.error(f"Error setting query result cache: {e}")

    def _semantic_cache_key(self, question: str) -> str:
        try:
            words = re.findall(r'[a-z0-9]+', question.lower())

            meaningful = [w for w in words if w not in CACHE_FILLER_WORDS]

            stemmed = []
            for w in meaningful:
                if len(w) > 4 and w.endswith('ies'):
                    w = w[:-3] + 'y'
                elif len(w) > 4 and w.endswith('ses') or (len(w) > 4 and w.endswith('zes')):
                    w = w[:-2]
                elif len(w) > 3 and w.endswith('s') and w[-2] not in 'su':
                    w = w[:-1]
                elif len(w) > 4 and w.endswith('ing'):
                    w = w[:-3]
                elif len(w) > 4 and w.endswith('ed') and w[-3] not in 'aeiou':
                    w = w[:-2]
                stemmed.append(w)

            key = '+'.join(sorted(stemmed))

            return hashlib.md5(key.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating semantic cache key: {e}")
            return hashlib.md5(question.lower().encode()).hexdigest()

    def _sql_cache_key(self, sql: str) -> str:
        try:
            normalized = ' '.join(sql.lower().split())
            return hashlib.md5(normalized.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating SQL cache key: {e}")
            return hashlib.md5(sql.encode()).hexdigest()


    def get_intelligence(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                with self._get_connection(self.schema_cache_db) as conn:
                    row = conn.execute(
                        'SELECT data_json FROM intelligence_cache WHERE key = ?',
                        (key,)
                    ).fetchone()

                    if row:
                        logger.debug(f"Intelligence cache hit for key: {key}")
                        return json.loads(row['data_json'])

                    logger.debug(f"Intelligence cache miss for key: {key}")
                    return None
            except Exception as e:
                logger.error(f"Error retrieving intelligence cache: {e}")
                return None

    def set_intelligence(self, key: str, data: Any, schema_fingerprint: str = None):
        with self._lock:
            try:
                with self._get_connection(self.schema_cache_db) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO intelligence_cache
                        (key, data_json, cached_at, schema_fingerprint)
                        VALUES (?, ?, ?, ?)
                    ''', (key, json.dumps(data, default=str), time.time(), schema_fingerprint))
                    conn.commit()

                logger.info(f"Intelligence cached for key: {key}")
            except Exception as e:
                logger.error(f"Error setting intelligence cache: {e}")


    def evict_expired(self):
        with self._lock:
            try:
                current_time = time.time()

                with self._get_connection(self.cache_db) as conn:
                    deleted = conn.execute('''
                        DELETE FROM query_cache
                        WHERE (current_time - created_at) / 3600.0 > ttl_hours
                    ''', {'current_time': current_time}).rowcount
                    conn.commit()
                    self.stats.evictions += deleted

                    if deleted > 0:
                        logger.info(f"Evicted {deleted} expired query cache entries")
            except Exception as e:
                logger.error(f"Error evicting expired entries: {e}")

    def clear_all(self):
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    conn.execute('DELETE FROM query_cache')
                    conn.commit()

                with self._get_connection(self.schema_cache_db) as conn:
                    conn.execute('DELETE FROM schema_cache')
                    conn.execute('DELETE FROM intelligence_cache')
                    conn.commit()

                for filename in os.listdir(self.intelligence_cache_dir):
                    filepath = os.path.join(self.intelligence_cache_dir, filename)
                    if os.path.isfile(filepath):
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            logger.warning(f"Could not remove {filepath}: {e}")

                logger.info("All caches cleared")
            except Exception as e:
                logger.error(f"Error clearing caches: {e}")

    def clear_query_cache(self):
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    conn.execute('DELETE FROM query_cache')
                    conn.commit()

                logger.info("Query result cache cleared")
            except Exception as e:
                logger.error(f"Error clearing query cache: {e}")

    def warm(self, top_queries: List[Tuple[str, str]]):
        with self._lock:
            logger.info(f"Starting cache warm with {len(top_queries)} queries")

            for i, (question, sql) in enumerate(top_queries):
                try:
                    semantic_key = self._semantic_cache_key(question)
                    sql_hash = self._sql_cache_key(sql)

                    logger.debug(f"Warmed query {i+1}/{len(top_queries)}: {question[:50]}")
                except Exception as e:
                    logger.error(f"Error warming query {i}: {e}")

            logger.info(f"Cache warm completed for {len(top_queries)} queries")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats_dict = self.stats.to_dict()

            try:
                if os.path.exists(self.cache_db):
                    stats_dict['query_cache_size_bytes'] = os.path.getsize(self.cache_db)

                if os.path.exists(self.schema_cache_db):
                    stats_dict['schema_cache_size_bytes'] = os.path.getsize(self.schema_cache_db)
            except Exception as e:
                logger.warning(f"Could not get cache database sizes: {e}")

            try:
                with self._get_connection(self.cache_db) as conn:
                    stats_dict['query_cache_entries'] = conn.execute(
                        'SELECT COUNT(*) FROM query_cache'
                    ).fetchone()[0]

                with self._get_connection(self.schema_cache_db) as conn:
                    stats_dict['schema_cache_entries'] = conn.execute(
                        'SELECT COUNT(*) FROM schema_cache'
                    ).fetchone()[0]
                    stats_dict['intelligence_cache_entries'] = conn.execute(
                        'SELECT COUNT(*) FROM intelligence_cache'
                    ).fetchone()[0]
            except Exception as e:
                logger.warning(f"Could not get cache entry counts: {e}")

            return stats_dict

    def reset_stats(self):
        with self._lock:
            self.stats = CacheStats()
            logger.info("Cache statistics reset")
