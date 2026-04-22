import os
import re
import json
import time
import sqlite3
import hashlib
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from threading import Lock

try:
    from gpdm_config import (
        KPI_MIN_FREQUENCY, KPI_MIN_USERS, TRENDING_WINDOW_DAYS,
        QUERY_RETENTION_DAYS, MAX_SUGGESTIONS_PER_CATEGORY,
    )
except ImportError:
    KPI_MIN_FREQUENCY = 3
    KPI_MIN_USERS = 2
    TRENDING_WINDOW_DAYS = 7
    QUERY_RETENTION_DAYS = 90
    MAX_SUGGESTIONS_PER_CATEGORY = 6

logger = logging.getLogger('gpdm.tracker')


@dataclass
class QueryRecord:
    question: str
    normalized: str
    user: str
    timestamp: float
    intent: str
    tables: List[str]
    columns: List[str]
    success: bool
    response_time_ms: float
    feedback: Optional[str] = None
    sql: str = ''
    error: str = ''


class QueryTracker:

    SCHEMA_VERSION = 1

    FILLER_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
        'by', 'for', 'in', 'on', 'at', 'to', 'of', 'with', 'from',
        'what', 'how', 'many', 'much', 'when', 'where', 'which', 'who',
        'can', 'could', 'would', 'should', 'will', 'did', 'do', 'does',
        'please', 'show', 'tell', 'give', 'get', 'find', 'search', 'list'
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, 'query_tracker.db')
        self._db_lock = Lock()
        self._init_db()
        logger.info(f"QueryTracker initialized with db at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS query_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        normalized TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        intent TEXT NOT NULL,
                        tables_json TEXT,
                        columns_json TEXT,
                        success BOOLEAN NOT NULL,
                        response_time_ms REAL NOT NULL,
                        feedback TEXT,
                        sql TEXT,
                        error TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(question, user_id, timestamp)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS kpi_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        question TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        frequency INTEGER DEFAULT 0,
                        last_asked DATETIME,
                        auto_detected BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS suggested_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT UNIQUE NOT NULL,
                        questions_json TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_end DATETIME,
                        question_order TEXT
                    )
                ''')

                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_log(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_intent ON query_log(intent)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_user ON query_log(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_normalized ON query_log(normalized)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_feedback ON query_log(feedback)')

                conn.commit()
                logger.debug("Database schema initialized successfully")
            except sqlite3.Error as e:
                logger.error(f"Database initialization error: {e}")
                raise
            finally:
                conn.close()

    def record(self, record: QueryRecord):
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO query_log (
                        question, normalized, user_id, timestamp, intent,
                        tables_json, columns_json, success, response_time_ms,
                        feedback, sql, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.question,
                    record.normalized,
                    record.user,
                    record.timestamp,
                    record.intent,
                    json.dumps(record.tables),
                    json.dumps(record.columns),
                    record.success,
                    record.response_time_ms,
                    record.feedback,
                    record.sql,
                    record.error
                ))
                conn.commit()
                logger.debug(f"Recorded query from {record.user}: {record.question[:50]}")
            except sqlite3.IntegrityError:
                logger.debug(f"Duplicate query record skipped for {record.user}")
            except sqlite3.Error as e:
                logger.error(f"Error recording query: {e}")
            finally:
                conn.close()

    def record_feedback(self, question: str, feedback: str):
        if feedback not in ('positive', 'negative'):
            logger.warning(f"Invalid feedback value: {feedback}")
            return

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE query_log
                    SET feedback = ?
                    WHERE question = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (feedback, question))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.debug(f"Recorded {feedback} feedback for: {question[:50]}")
            except sqlite3.Error as e:
                logger.error(f"Error recording feedback: {e}")
            finally:
                conn.close()


    def get_most_asked(self, limit: int = None) -> List[Dict]:
        if limit is None:
            limit = MAX_SUGGESTIONS_PER_CATEGORY
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        question,
                        COUNT(*) as count,
                        MAX(timestamp) as last_asked,
                        intent
                    FROM query_log
                    WHERE success = 1
                    GROUP BY question, intent
                    ORDER BY count DESC
                    LIMIT ?
                ''', (limit,))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'question': row['question'],
                        'count': row['count'],
                        'last_asked': datetime.fromtimestamp(row['last_asked']).isoformat(),
                        'intent': row['intent']
                    })
                return results
            except sqlite3.Error as e:
                logger.error(f"Error fetching most asked: {e}")
                return []
            finally:
                conn.close()

    def get_trending(self, days: int = None, limit: int = 5) -> List[Dict]:
        if days is None:
            days = TRENDING_WINDOW_DAYS
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cutoff_time = time.time() - (days * 86400)

                cursor.execute('''
                    SELECT
                        question,
                        COUNT(CASE WHEN timestamp >= ? THEN 1 END) as recent_count,
                        COUNT(CASE WHEN timestamp < ? THEN 1 END) as older_count
                    FROM query_log
                    WHERE success = 1
                    GROUP BY question
                    HAVING recent_count >= 2 AND older_count > 0
                    ORDER BY (CAST(recent_count AS FLOAT) / (older_count + recent_count)) DESC
                    LIMIT ?
                ''', (cutoff_time, cutoff_time, limit))

                results = []
                for row in cursor.fetchall():
                    if row['older_count'] > 0:
                        trend_pct = ((row['recent_count'] - row['older_count']) / row['older_count']) * 100
                        results.append({
                            'question': row['question'],
                            'trend_pct': round(trend_pct, 1),
                            'recent_count': row['recent_count']
                        })
                return results
            except sqlite3.Error as e:
                logger.error(f"Error fetching trending: {e}")
                return []
            finally:
                conn.close()

    def get_peer_suggestions(self, current_question: str, limit: int = 3) -> List[str]:
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT DISTINCT user_id,
                        CAST(timestamp / 3600 AS INTEGER) as session_hour
                    FROM query_log
                    WHERE question = ?
                ''', (current_question,))

                sessions = cursor.fetchall()
                if not sessions:
                    return []

                session_conditions = [
                    (row['user_id'], row['session_hour']) for row in sessions
                ]

                co_occurrence = Counter()
                for user_id, session_hour in session_conditions:
                    cursor.execute('''
                        SELECT question
                        FROM query_log
                        WHERE user_id = ?
                        AND CAST(timestamp / 3600 AS INTEGER) = ?
                        AND question != ?
                        AND success = 1
                    ''', (user_id, session_hour, current_question))

                    for row in cursor.fetchall():
                        co_occurrence[row['question']] += 1

                return [q for q, _ in co_occurrence.most_common(limit)]
            except sqlite3.Error as e:
                logger.error(f"Error fetching peer suggestions: {e}")
                return []
            finally:
                conn.close()

    def get_suggested_questions(self) -> Dict[str, List[Dict]]:
        return {
            'frequently_asked': self.get_most_asked(limit=10),
            'trending_this_week': self.get_trending(days=7, limit=5),
            'you_might_also_ask': [],
            'auto_detected_kpis': self.detect_kpis()
        }


    def detect_kpis(self) -> List[Dict]:
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT
                        question,
                        intent,
                        COUNT(*) as frequency,
                        COUNT(DISTINCT user_id) as user_count,
                        SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END) as negative_count
                    FROM query_log
                    WHERE success = 1
                    AND intent IN ('rate', 'trend')
                    GROUP BY question, intent
                    HAVING frequency >= ?
                    AND user_count >= ?
                    AND (negative_count = 0 OR negative_count IS NULL)
                    ORDER BY frequency DESC
                ''', (KPI_MIN_FREQUENCY, KPI_MIN_USERS))

                results = []
                for row in cursor.fetchall():
                    kpi_name = self._generate_kpi_name(row['question'])
                    results.append({
                        'name': kpi_name,
                        'question': row['question'],
                        'frequency': row['frequency'],
                        'intent': row['intent']
                    })

                return results
            except sqlite3.Error as e:
                logger.error(f"Error detecting KPIs: {e}")
                return []
            finally:
                conn.close()

    def register_kpi(self, name: str, question: str, intent: str):
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO kpi_registry
                    (name, question, intent, last_asked, auto_detected)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, 0)
                ''', (name, question, intent))
                conn.commit()
                logger.info(f"Registered KPI: {name}")
            except sqlite3.Error as e:
                logger.error(f"Error registering KPI: {e}")
            finally:
                conn.close()


    def get_usage_stats(self) -> Dict:
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                now = time.time()
                today_start = now - 86400
                week_start = now - (7 * 86400)

                cursor.execute('SELECT COUNT(*) as total FROM query_log')
                total = cursor.fetchone()['total'] or 0

                cursor.execute('''
                    SELECT
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                        AVG(response_time_ms) as avg_response_time
                    FROM query_log
                ''')
                stats = cursor.fetchone()
                success_rate = (stats['success_count'] / total * 100) if total > 0 else 0
                avg_time = stats['avg_response_time'] or 0

                cursor.execute('SELECT COUNT(DISTINCT user_id) as users FROM query_log')
                unique_users = cursor.fetchone()['users'] or 0

                cursor.execute(
                    'SELECT COUNT(*) as count FROM query_log WHERE timestamp >= ?',
                    (today_start,)
                )
                queries_today = cursor.fetchone()['count'] or 0

                cursor.execute(
                    'SELECT COUNT(*) as count FROM query_log WHERE timestamp >= ?',
                    (week_start,)
                )
                queries_week = cursor.fetchone()['count'] or 0

                cursor.execute('''
                    SELECT intent, COUNT(*) as count
                    FROM query_log
                    GROUP BY intent
                    ORDER BY count DESC
                ''')
                top_intents = {row['intent']: row['count'] for row in cursor.fetchall()}

                cursor.execute('''
                    SELECT tables_json, COUNT(*) as count
                    FROM query_log
                    WHERE tables_json IS NOT NULL
                    GROUP BY tables_json
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                top_tables = {}
                for row in cursor.fetchall():
                    try:
                        tables = json.loads(row['tables_json'])
                        for table in tables:
                            top_tables[table] = top_tables.get(table, 0) + row['count']
                    except json.JSONDecodeError:
                        pass

                cursor.execute('''
                    SELECT
                        feedback,
                        COUNT(*) as count
                    FROM query_log
                    GROUP BY feedback
                ''')
                feedback_summary = {'positive': 0, 'negative': 0, 'none': 0}
                for row in cursor.fetchall():
                    key = row['feedback'] or 'none'
                    feedback_summary[key] = row['count']

                return {
                    'total_queries': total,
                    'success_rate': round(success_rate, 1),
                    'avg_response_time_ms': round(avg_time, 2),
                    'unique_users': unique_users,
                    'queries_today': queries_today,
                    'queries_this_week': queries_week,
                    'top_intents': top_intents,
                    'top_tables': dict(sorted(top_tables.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:10]),
                    'feedback_summary': feedback_summary
                }
            except sqlite3.Error as e:
                logger.error(f"Error fetching usage stats: {e}")
                return {}
            finally:
                conn.close()

    def get_failure_patterns(self, limit: int = 10) -> List[Dict]:
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        question,
                        error,
                        COUNT(*) as count,
                        MAX(timestamp) as last_seen
                    FROM query_log
                    WHERE success = 0 AND error != ''
                    GROUP BY question, error
                    ORDER BY count DESC
                    LIMIT ?
                ''', (limit,))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'question': row['question'],
                        'error': row['error'],
                        'count': row['count'],
                        'last_seen': datetime.fromtimestamp(row['last_seen']).isoformat()
                    })
                return results
            except sqlite3.Error as e:
                logger.error(f"Error fetching failure patterns: {e}")
                return []
            finally:
                conn.close()


    @staticmethod
    def normalize_question(question: str) -> str:
        tokens = question.lower().split()

        filtered = [
            t for t in tokens
            if t not in QueryTracker.FILLER_WORDS
            and not re.match(r'^\d+$', t)
        ]

        sorted_tokens = sorted(filtered)

        return ' '.join(sorted_tokens)


    def prune_old(self, days: int = None):
        if days is None:
            days = QUERY_RETENTION_DAYS
        with self._db_lock:
            conn = self._get_connection()
            try:
                cutoff_time = time.time() - (days * 86400)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM query_log WHERE timestamp < ?', (cutoff_time,))
                conn.commit()
                logger.info(f"Pruned {cursor.rowcount} old query records (older than {days} days)")
            except sqlite3.Error as e:
                logger.error(f"Error pruning old records: {e}")
            finally:
                conn.close()

    def export_golden_queries(self) -> List[Dict]:
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT
                        question,
                        intent,
                        tables_json,
                        columns_json,
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(CASE WHEN feedback = 'positive' THEN 1 END) as positive_count
                    FROM query_log
                    WHERE success = 1
                    AND feedback = 'positive'
                    GROUP BY question, intent
                    HAVING positive_count >= 2
                    ORDER BY positive_count DESC
                ''')

                results = []
                for row in cursor.fetchall():
                    try:
                        tables = json.loads(row['tables_json'] or '[]')
                        columns = json.loads(row['columns_json'] or '[]')
                    except json.JSONDecodeError:
                        tables = []
                        columns = []

                    results.append({
                        'question': row['question'],
                        'intent': row['intent'],
                        'tables': tables,
                        'columns': columns,
                        'response_time_ms': round(row['avg_response_time'], 2),
                        'feedback_count': row['positive_count']
                    })

                return results
            except sqlite3.Error as e:
                logger.error(f"Error exporting golden queries: {e}")
                return []
            finally:
                conn.close()


    @staticmethod
    def _generate_kpi_name(question: str) -> str:
        words = question.lower().split()
        key_words = [
            w for w in words[:5]
            if len(w) > 3 and w not in QueryTracker.FILLER_WORDS
        ]

        name = '_'.join(key_words)[:50]
        return name or 'kpi_' + hashlib.md5(question.encode()).hexdigest()[:8]
