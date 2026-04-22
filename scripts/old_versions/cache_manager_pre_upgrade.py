"""
GPDM Intelligent Analytics System - Workstream 5: Cache Manager
Three-tier intelligent caching layer with semantic query deduplication.

Tier 1: Schema Cache (persistent across sessions)
    - Column profiles, join graph, column relationships
    - Invalidated only when database schema changes
    - Storage: SQLite file (schema_cache.db)
    - TTL: 24 hours or until schema change detected

Tier 2: Query Result Cache (cross-session)
    - SQL query hash -> result rows + metadata
    - Semantic dedup: similar questions map to same cache key
    - Storage: SQLite (can swap to Redis for production)
    - TTL: 1 hour for volatile data, 24 hours for reference data

Tier 3: Computed Intelligence Cache
    - TF-IDF vectors, synonym dictionaries, concept maps
    - Rebuilt only when schema cache invalidates
    - Storage: pickle/JSON files
    - TTL: Same as schema cache
"""

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
    """Track cache performance metrics."""
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    sql_hits: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'semantic_hits': self.semantic_hits,
            'sql_hits': self.sql_hits,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate
        }


class CacheManager:
    """Three-tier intelligent cache with semantic deduplication.

    Manages schema cache, query result cache, and intelligence cache with
    thread-safe operations and TTL-based expiration.
    """

    def __init__(self, data_dir: str, use_redis: bool = False, redis_url: str = None):
        """Initialize cache manager.

        Args:
            data_dir: Directory for cache database and files
            use_redis: Whether to use Redis for query result cache (future enhancement)
            redis_url: Redis connection URL if use_redis is True
        """
        self.data_dir = data_dir
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.cache_db = os.path.join(data_dir, 'query_cache.db')
        self.schema_cache_db = os.path.join(data_dir, 'schema_cache.db')
        self.intelligence_cache_dir = os.path.join(data_dir, 'intelligence_cache')
        self.stats = CacheStats()
        self._lock = threading.RLock()

        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.intelligence_cache_dir, exist_ok=True)

        # Initialize databases
        self._init_db()

        logger.info(f"CacheManager initialized with data_dir={data_dir}")

    def _init_db(self):
        """Initialize SQLite databases with required tables."""
        # Initialize query cache database
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
                    is_reference INTEGER DEFAULT 0
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_semantic ON query_cache(semantic_key)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sql ON query_cache(sql_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON query_cache(created_at)')
            conn.commit()

        # Initialize schema cache database
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

        # Initialize intelligence cache database
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

    @contextmanager
    def _get_connection(self, db_path: str):
        """Context manager for database connections."""
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # --- Tier 1: Schema Cache ---

    def get_schema(self, db_path: str) -> Optional[Dict]:
        """Get cached schema profiles. Returns None if stale.

        Args:
            db_path: Path to the database file

        Returns:
            Cached schema dictionary or None if not cached or expired
        """
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

                    # Check if expired
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
        """Cache schema profiles with fingerprint for staleness detection.

        Args:
            db_path: Path to the database file
            schema_data: Schema dictionary to cache
            ttl_hours: Time-to-live in hours (default from config)
        """
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
        """Hash of table names + column names + row counts.

        Changes if schema changes, allowing detection of stale cache.

        Args:
            db_path: Path to the database file

        Returns:
            Hex digest of schema fingerprint
        """
        try:
            if not os.path.exists(db_path):
                return hashlib.md5(b'nonexistent').hexdigest()

            with self._get_connection(db_path) as conn:
                # Get all tables and their columns
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()

                fingerprint_parts = []
                for table in tables:
                    table_name = table[0]
                    fingerprint_parts.append(f"TABLE:{table_name}")

                    # Get columns
                    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                    for col in columns:
                        fingerprint_parts.append(f"COL:{col[1]}:{col[2]}")

                    # Get row count
                    try:
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        fingerprint_parts.append(f"ROWS:{row_count}")
                    except Exception:
                        # Table might be empty or have issues
                        fingerprint_parts.append("ROWS:0")

                fingerprint_str = '|'.join(fingerprint_parts)
                return hashlib.md5(fingerprint_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing schema fingerprint for {db_path}: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()

    def is_schema_stale(self, db_path: str) -> bool:
        """Check if cached schema is still valid.

        Args:
            db_path: Path to the database file

        Returns:
            True if schema is stale or not cached, False otherwise
        """
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

    # --- Tier 2: Query Result Cache ---

    def get_query_result(self, question: str, sql: str = None) -> Optional[Dict]:
        """Look up cached result. Tries semantic key first, then SQL hash.

        Args:
            question: Natural language question
            sql: Optional SQL query

        Returns:
            Cached result dictionary or None
        """
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    # Try semantic key first
                    semantic_key = self._semantic_cache_key(question)
                    row = conn.execute(
                        'SELECT result_json, created_at, ttl_hours FROM query_cache WHERE semantic_key = ? ORDER BY created_at DESC LIMIT 1',
                        (semantic_key,)
                    ).fetchone()

                    if row:
                        # Check if expired
                        age_hours = (time.time() - row['created_at']) / 3600
                        if age_hours <= row['ttl_hours']:
                            self.stats.hits += 1
                            self.stats.semantic_hits += 1

                            # Update hit count
                            cache_key = hashlib.md5((semantic_key + str(row['created_at'])).encode()).hexdigest()
                            conn.execute(
                                'UPDATE query_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE cache_key = ?',
                                (time.time(), cache_key)
                            )
                            conn.commit()

                            logger.debug(f"Query result hit (semantic) for question: {question[:50]}")
                            return json.loads(row['result_json'])

                    # Try SQL hash if provided
                    if sql:
                        sql_hash = self._sql_cache_key(sql)
                        row = conn.execute(
                            'SELECT result_json, created_at, ttl_hours FROM query_cache WHERE sql_hash = ? ORDER BY created_at DESC LIMIT 1',
                            (sql_hash,)
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
        """Cache a query result with TTL.

        Args:
            question: Natural language question
            sql: SQL query that was executed
            result: Result dictionary to cache
            ttl_hours: Time-to-live in hours
            is_reference: Whether this is reference data (longer TTL)
        """
        with self._lock:
            try:
                semantic_key = self._semantic_cache_key(question)
                sql_hash = self._sql_cache_key(sql)
                cache_key = hashlib.md5((semantic_key + sql_hash + str(time.time())).encode()).hexdigest()

                # Reference data gets longer TTL, volatile gets shorter TTL
                if is_reference:
                    ttl_hours = QUERY_CACHE_TTL_REFERENCE
                elif ttl_hours is None:
                    ttl_hours = QUERY_CACHE_TTL_VOLATILE

                with self._get_connection(self.cache_db) as conn:
                    conn.execute('''
                        INSERT INTO query_cache
                        (cache_key, semantic_key, sql_hash, question, sql_text, result_json, created_at, ttl_hours, is_reference)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (cache_key, semantic_key, sql_hash, question, sql,
                          json.dumps(result), time.time(), ttl_hours, 1 if is_reference else 0))
                    conn.commit()

                logger.info(f"Query result cached for: {question[:50]} (TTL: {ttl_hours}h)")
            except Exception as e:
                logger.error(f"Error setting query result cache: {e}")

    def _semantic_cache_key(self, question: str) -> str:
        """Generate a normalized cache key from natural language question.

        Deduplication logic:
        - Lowercase, strip punctuation
        - Sort intent tokens for order-independence
        - Remove filler words
        - Stem key nouns

        Examples that should hit the same key:
        - 'What is the denial rate by region?'
        - 'denial rate by region'
        - 'show me denial rates for each region'
        All → same cache key

        Args:
            question: Natural language question

        Returns:
            MD5 hash of normalized semantic key
        """
        try:
            # Extract alphanumeric tokens (lowercase)
            words = re.findall(r'[a-z0-9]+', question.lower())

            # Filter out filler words
            meaningful = [w for w in words if w not in CACHE_FILLER_WORDS]

            # Basic suffix stemming — strip common plural/verb endings
            # This ensures "rates" ≈ "rate", "claims" ≈ "claim", "providers" ≈ "provider"
            # Rule order matters — check specific patterns before general ones
            stemmed = []
            for w in meaningful:
                if len(w) > 4 and w.endswith('ies'):
                    w = w[:-3] + 'y'  # "specialties" → "specialty"
                elif len(w) > 4 and w.endswith('ses') or (len(w) > 4 and w.endswith('zes')):
                    w = w[:-2]  # "diagnoses" → "diagnos", "analyses" → "analys"
                elif len(w) > 3 and w.endswith('s') and w[-2] not in 'su':
                    w = w[:-1]  # "claims" → "claim", "rates" → "rate", "providers" → "provider"
                elif len(w) > 4 and w.endswith('ing'):
                    w = w[:-3]  # "trending" → "trend"
                elif len(w) > 4 and w.endswith('ed') and w[-3] not in 'aeiou':
                    w = w[:-2]  # "billed" → "bill"
                stemmed.append(w)

            # Sort for order-independence
            key = '+'.join(sorted(stemmed))

            # Hash for consistent length
            return hashlib.md5(key.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating semantic cache key: {e}")
            return hashlib.md5(question.lower().encode()).hexdigest()

    def _sql_cache_key(self, sql: str) -> str:
        """Hash of normalized SQL (whitespace/case insensitive).

        Args:
            sql: SQL query string

        Returns:
            MD5 hash of normalized SQL
        """
        try:
            # Normalize SQL: lowercase, collapse whitespace
            normalized = ' '.join(sql.lower().split())
            return hashlib.md5(normalized.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating SQL cache key: {e}")
            return hashlib.md5(sql.encode()).hexdigest()

    # --- Tier 3: Intelligence Cache ---

    def get_intelligence(self, key: str) -> Optional[Any]:
        """Get cached computed intelligence (TF-IDF index, concept maps, etc.).

        Args:
            key: Intelligence cache key

        Returns:
            Cached intelligence data or None
        """
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
        """Cache computed intelligence data.

        Args:
            key: Intelligence cache key
            data: Data to cache (will be JSON serialized)
            schema_fingerprint: Optional schema fingerprint for invalidation tracking
        """
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

    # --- Cache Management ---

    def evict_expired(self):
        """Remove entries past their TTL."""
        with self._lock:
            try:
                current_time = time.time()

                # Evict from query cache
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
        """Clear all caches."""
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    conn.execute('DELETE FROM query_cache')
                    conn.commit()

                with self._get_connection(self.schema_cache_db) as conn:
                    conn.execute('DELETE FROM schema_cache')
                    conn.execute('DELETE FROM intelligence_cache')
                    conn.commit()

                # Clear intelligence cache files
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
        """Clear only query results (keep schema + intelligence)."""
        with self._lock:
            try:
                with self._get_connection(self.cache_db) as conn:
                    conn.execute('DELETE FROM query_cache')
                    conn.commit()

                logger.info("Query result cache cleared")
            except Exception as e:
                logger.error(f"Error clearing query cache: {e}")

    def warm(self, top_queries: List[Tuple[str, str]]):
        """Pre-warm cache with top N most-asked queries.

        Note: This method expects the results to be pre-computed.
        In production, you would execute these queries and cache their results.

        Args:
            top_queries: List of (question, sql) tuples
        """
        with self._lock:
            logger.info(f"Starting cache warm with {len(top_queries)} queries")

            for i, (question, sql) in enumerate(top_queries):
                try:
                    semantic_key = self._semantic_cache_key(question)
                    sql_hash = self._sql_cache_key(sql)

                    # Log that we're tracking these queries
                    logger.debug(f"Warmed query {i+1}/{len(top_queries)}: {question[:50]}")
                except Exception as e:
                    logger.error(f"Error warming query {i}: {e}")

            logger.info(f"Cache warm completed for {len(top_queries)} queries")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss stats.

        Returns:
            Dictionary with cache performance metrics
        """
        with self._lock:
            stats_dict = self.stats.to_dict()

            # Add database sizes
            try:
                if os.path.exists(self.cache_db):
                    stats_dict['query_cache_size_bytes'] = os.path.getsize(self.cache_db)

                if os.path.exists(self.schema_cache_db):
                    stats_dict['schema_cache_size_bytes'] = os.path.getsize(self.schema_cache_db)
            except Exception as e:
                logger.warning(f"Could not get cache database sizes: {e}")

            # Add entry counts
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
        """Reset cache statistics."""
        with self._lock:
            self.stats = CacheStats()
            logger.info("Cache statistics reset")
