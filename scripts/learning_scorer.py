import os
import sqlite3
import json
import logging
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryScore:
    query_id: str
    timestamp: datetime
    question: str
    intent: str
    sql: str
    sql_accuracy: float
    intent_match: float
    table_selection: float
    column_resolution: float
    join_correctness: float
    response_time_score: float
    narrative_quality: float
    latency_ms: float
    query_type: str
    tables_involved: List[str]
    complexity_level: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        scores = [
            (self.sql_accuracy, 0.25),
            (self.intent_match, 0.15),
            (self.table_selection, 0.15),
            (self.column_resolution, 0.12),
            (self.join_correctness, 0.12),
            (self.response_time_score, 0.10),
            (self.narrative_quality, 0.11),
        ]
        return sum(score * weight for score, weight in scores)


@dataclass
class PerformanceMetrics:
    category_key: str
    category_type: str
    sample_count: int
    avg_overall: float
    avg_sql_accuracy: float
    avg_intent_match: float
    avg_table_selection: float
    avg_column_resolution: float
    avg_join_correctness: float
    avg_response_time: float
    avg_narrative_quality: float
    success_rate: float
    avg_latency_ms: float
    stddev_overall: Optional[float] = None
    trend: str = "stable"
    last_updated: datetime = field(default_factory=datetime.now)


class LearningScorer:

    ACCURACY_THRESHOLD = 0.80
    POOR_PERFORMANCE_THRESHOLD = 0.70
    GAP_THRESHOLD = 5
    ACCURACY_DROP_PERCENT = 0.10
    MIN_SAMPLES_FOR_STATS = 10
    STALENESS_HOURS = 24
    RESPONSE_TIME_TARGET_MS = 2000

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "learning_scores.db")
        elif os.path.isdir(db_path):
            db_path = os.path.join(db_path, "learning_scores.db")
        elif not db_path.endswith('.db'):
            db_path = os.path.join(db_path, "learning_scores.db")

        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        logger.info(f"LearningScorer initialized with database: {self.db_path}")

    def _initialize_database(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_scores (
                    query_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    sql_accuracy REAL NOT NULL,
                    intent_match REAL NOT NULL,
                    table_selection REAL NOT NULL,
                    column_resolution REAL NOT NULL,
                    join_correctness REAL NOT NULL,
                    response_time_score REAL NOT NULL,
                    narrative_quality REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    query_type TEXT NOT NULL,
                    tables_involved TEXT NOT NULL,
                    complexity_level TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    overall_score REAL NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_key TEXT NOT NULL,
                    category_type TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    avg_overall REAL NOT NULL,
                    avg_sql_accuracy REAL NOT NULL,
                    avg_intent_match REAL NOT NULL,
                    avg_table_selection REAL NOT NULL,
                    avg_column_resolution REAL NOT NULL,
                    avg_join_correctness REAL NOT NULL,
                    avg_response_time REAL NOT NULL,
                    avg_narrative_quality REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    stddev_overall REAL,
                    trend TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    UNIQUE(category_key, category_type)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gap_type TEXT NOT NULL,
                    gap_key TEXT NOT NULL,
                    severity_score REAL NOT NULL,
                    description TEXT NOT NULL,
                    first_detected TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    samples_count INTEGER NOT NULL,
                    failure_rate REAL NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_type TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    affected_categories TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correction_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    wrong_sql TEXT NOT NULL,
                    correct_sql TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    feedback_at TEXT NOT NULL,
                    applied_fix INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blind_spots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    column_name TEXT,
                    query_pattern TEXT NOT NULL,
                    priority_score REAL NOT NULL,
                    identified_at TEXT NOT NULL
                )
            """)

            self.connection.commit()
            logger.info("Database schema initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def score_query(
        self,
        question: str,
        intent: str,
        sql: str,
        sql_accuracy: float,
        intent_match: float,
        table_selection: float,
        column_resolution: float,
        join_correctness: float,
        latency_ms: float,
        narrative_quality: float,
        result: Any,
        query_type: str = "unknown",
        tables_involved: List[str] = None,
        complexity_level: str = "simple",
        details: Dict[str, Any] = None,
    ) -> QueryScore:
        query_id = f"q_{datetime.now().timestamp()}_{hash(sql) % 10000}"

        response_time_score = max(0, min(1, 1 - (latency_ms / self.RESPONSE_TIME_TARGET_MS)))

        scores = {
            'sql_accuracy': max(0, min(1, sql_accuracy)),
            'intent_match': max(0, min(1, intent_match)),
            'table_selection': max(0, min(1, table_selection)),
            'column_resolution': max(0, min(1, column_resolution)),
            'join_correctness': max(0, min(1, join_correctness)),
            'response_time_score': response_time_score,
            'narrative_quality': max(0, min(1, narrative_quality)),
        }

        tables_involved = tables_involved or []
        details = details or {}
        success = sql_accuracy >= self.ACCURACY_THRESHOLD

        query_score = QueryScore(
            query_id=query_id,
            timestamp=datetime.now(),
            question=question,
            intent=intent,
            sql=sql,
            latency_ms=latency_ms,
            query_type=query_type,
            tables_involved=tables_involved,
            complexity_level=complexity_level,
            success=success,
            details=details,
            **scores
        )

        self._save_query_score(query_score)

        self._update_category_metrics(query_score)

        self._detect_gaps(query_score)

        self._check_retrain_triggers()

        logger.info(
            f"Scored query: {query_type} | Overall: {query_score.overall_score:.3f} | "
            f"Success: {success} | Latency: {latency_ms}ms"
        )

        return query_score

    def _save_query_score(self, score: QueryScore):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO query_scores VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                score.query_id,
                score.timestamp.isoformat(),
                score.question,
                score.intent,
                score.sql,
                score.sql_accuracy,
                score.intent_match,
                score.table_selection,
                score.column_resolution,
                score.join_correctness,
                score.response_time_score,
                score.narrative_quality,
                score.latency_ms,
                score.query_type,
                json.dumps(score.tables_involved),
                score.complexity_level,
                int(score.success),
                json.dumps(score.details),
                score.overall_score,
            ))
            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving query score: {e}")
            self.connection.rollback()

    def _update_category_metrics(self, score: QueryScore):
        categories = [
            ("query_type", score.query_type),
            ("complexity", score.complexity_level),
            *[("table", table) for table in score.tables_involved],
        ]

        time_category = self._get_time_category()
        categories.append(("time_period", time_category))

        for cat_type, cat_key in categories:
            self._update_category(cat_type, cat_key, score)

    def _update_category(self, category_type: str, category_key: str, score: QueryScore):
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT sql_accuracy, intent_match, table_selection, column_resolution,
                       join_correctness, response_time_score, narrative_quality,
                       latency_ms, success, overall_score
                FROM query_scores
                WHERE (? = 'query_type' AND query_type = ?)
                   OR (? = 'complexity' AND complexity_level = ?)
                   OR (? = 'table' AND tables_involved LIKE ?)
                   OR (? = 'time_period' AND datetime(timestamp) >= datetime('now', ?))
            """, (
                category_type, category_key,
                category_type, category_key,
                category_type, f'%"{category_key}"%',
                category_type, f'-{self._get_time_offset(category_key)} hours',
            ))

            rows = cursor.fetchall()
            if not rows:
                return

            metrics = {
                'sql_accuracy': [],
                'intent_match': [],
                'table_selection': [],
                'column_resolution': [],
                'join_correctness': [],
                'response_time_score': [],
                'narrative_quality': [],
                'latency_ms': [],
                'success': [],
                'overall': [],
            }

            for row in rows:
                metrics['sql_accuracy'].append(row[0])
                metrics['intent_match'].append(row[1])
                metrics['table_selection'].append(row[2])
                metrics['column_resolution'].append(row[3])
                metrics['join_correctness'].append(row[4])
                metrics['response_time_score'].append(row[5])
                metrics['narrative_quality'].append(row[6])
                metrics['latency_ms'].append(row[7])
                metrics['success'].append(row[8])
                metrics['overall'].append(row[9])

            avg_overall = mean(metrics['overall']) if metrics['overall'] else 0
            avg_sql_accuracy = mean(metrics['sql_accuracy']) if metrics['sql_accuracy'] else 0
            avg_intent_match = mean(metrics['intent_match']) if metrics['intent_match'] else 0
            avg_table_selection = mean(metrics['table_selection']) if metrics['table_selection'] else 0
            avg_column_resolution = mean(metrics['column_resolution']) if metrics['column_resolution'] else 0
            avg_join_correctness = mean(metrics['join_correctness']) if metrics['join_correctness'] else 0
            avg_response_time = mean(metrics['response_time_score']) if metrics['response_time_score'] else 0
            avg_narrative_quality = mean(metrics['narrative_quality']) if metrics['narrative_quality'] else 0
            avg_latency_ms = mean(metrics['latency_ms']) if metrics['latency_ms'] else 0
            success_rate = sum(metrics['success']) / len(metrics['success']) if metrics['success'] else 0

            trend = self._calculate_trend(category_type, category_key)

            stddev_overall = None
            if len(metrics['overall']) >= self.MIN_SAMPLES_FOR_STATS:
                try:
                    stddev_overall = stdev(metrics['overall'])
                except:
                    pass

            cursor.execute("""
                INSERT OR REPLACE INTO category_performance VALUES (
                    (SELECT id FROM category_performance
                     WHERE category_key = ? AND category_type = ?),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                category_key, category_type,
                category_key,
                category_type,
                len(rows),
                avg_overall,
                avg_sql_accuracy,
                avg_intent_match,
                avg_table_selection,
                avg_column_resolution,
                avg_join_correctness,
                avg_response_time,
                avg_narrative_quality,
                success_rate,
                avg_latency_ms,
                stddev_overall,
                trend,
                datetime.now().isoformat(),
            ))

            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating category metrics: {e}")
            self.connection.rollback()

    def _get_time_category(self) -> str:
        now = datetime.now()
        if now.hour < 6:
            return "night"
        elif now.hour < 12:
            return "morning"
        elif now.hour < 18:
            return "afternoon"
        else:
            return "evening"

    def _get_time_offset(self, time_category: str) -> int:
        offsets = {
            "hour": 1,
            "day": 24,
            "week": 168,
            "month": 720,
            "night": 6,
            "morning": 6,
            "afternoon": 6,
            "evening": 6,
        }
        return offsets.get(time_category, 24)

    def _calculate_trend(self, category_type: str, category_key: str) -> str:
        try:
            cursor = self.connection.cursor()

            query = """
                SELECT overall_score, timestamp
                FROM query_scores
                WHERE (? = 'query_type' AND query_type = ?)
                   OR (? = 'complexity' AND complexity_level = ?)
                   OR (? = 'table' AND tables_involved LIKE ?)
                ORDER BY timestamp DESC
                LIMIT 50
            """

            cursor.execute(query, (
                category_type, category_key,
                category_type, category_key,
                category_type, f'%"{category_key}"%',
            ))

            rows = cursor.fetchall()
            if len(rows) < 10:
                return "stable"

            recent = [r[0] for r in rows[:len(rows)//2]]
            older = [r[0] for r in rows[len(rows)//2:]]

            recent_avg = mean(recent)
            older_avg = mean(older)
            change_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

            if change_pct > 0.05:
                return "improving"
            elif change_pct < -0.05:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "stable"

    def _detect_gaps(self, score: QueryScore):
        try:
            cursor = self.connection.cursor()

            if score.sql_accuracy < self.POOR_PERFORMANCE_THRESHOLD:
                for table in score.tables_involved:
                    gap_key = f"sql_accuracy_{table}"
                    self._record_gap(
                        gap_type="sql_accuracy_failure",
                        gap_key=gap_key,
                        severity_score=1 - score.sql_accuracy,
                        description=f"Low SQL accuracy on table: {table}",
                    )

            if score.intent_match < self.POOR_PERFORMANCE_THRESHOLD:
                gap_key = f"intent_match_{score.query_type}"
                self._record_gap(
                    gap_type="intent_mismatch",
                    gap_key=gap_key,
                    severity_score=1 - score.intent_match,
                    description=f"Intent matching failure for {score.query_type} queries",
                )

            if score.column_resolution < self.POOR_PERFORMANCE_THRESHOLD:
                gap_key = f"column_resolution_complex"
                self._record_gap(
                    gap_type="column_resolution",
                    gap_key=gap_key,
                    severity_score=1 - score.column_resolution,
                    description=f"Column resolution issues in {score.complexity_level} queries",
                )

            if score.join_correctness < self.POOR_PERFORMANCE_THRESHOLD and len(score.tables_involved) > 1:
                tables_str = "_".join(sorted(score.tables_involved))
                gap_key = f"join_failure_{tables_str}"
                self._record_gap(
                    gap_type="join_failure",
                    gap_key=gap_key,
                    severity_score=1 - score.join_correctness,
                    description=f"JOIN failures across tables: {', '.join(score.tables_involved)}",
                )

            if score.response_time_score < 0.5:
                gap_key = f"latency_{score.query_type}"
                self._record_gap(
                    gap_type="performance_latency",
                    gap_key=gap_key,
                    severity_score=1 - score.response_time_score,
                    description=f"High latency for {score.query_type} queries ({score.latency_ms}ms)",
                )

            if score.complexity_level == "complex" and score.overall_score < self.ACCURACY_THRESHOLD:
                gap_key = "complex_queries_weakness"
                self._record_gap(
                    gap_type="complexity_gap",
                    gap_key=gap_key,
                    severity_score=1 - score.overall_score,
                    description="Weak performance on complex multi-table queries",
                )

        except Exception as e:
            logger.error(f"Error detecting gaps: {e}")

    def _record_gap(
        self,
        gap_type: str,
        gap_key: str,
        severity_score: float,
        description: str,
    ):
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                "SELECT id, samples_count FROM detected_gaps WHERE gap_type = ? AND gap_key = ?",
                (gap_type, gap_key)
            )
            existing = cursor.fetchone()

            now = datetime.now().isoformat()

            if existing:
                gap_id, sample_count = existing
                cursor.execute(
                    "UPDATE detected_gaps SET last_seen = ?, samples_count = ?, failure_rate = ? "
                    "WHERE id = ?",
                    (now, sample_count + 1, min(1.0, severity_score * 1.1), gap_id)
                )
            else:
                cursor.execute("""
                    INSERT INTO detected_gaps
                    (gap_type, gap_key, severity_score, description, first_detected, last_seen, samples_count, failure_rate)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """, (gap_type, gap_key, severity_score, description, now, now, severity_score))

            self.connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording gap: {e}")
            self.connection.rollback()

    def _check_retrain_triggers(self):
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT category_key, category_type, avg_overall, trend
                FROM category_performance
                WHERE trend = 'declining' AND avg_overall < ?
            """, (self.ACCURACY_THRESHOLD,))

            declining_categories = cursor.fetchall()
            if declining_categories:
                affected = ", ".join([f"{cat[0]}" for cat in declining_categories])
                self._record_trigger(
                    trigger_type="accuracy_drop",
                    trigger_reason="Performance declining in multiple categories",
                    severity="high",
                    affected_categories=affected,
                    recommendation="Retrain on identified weak areas",
                )

            cursor.execute("SELECT COUNT(*) FROM detected_gaps WHERE severity_score > 0.3")
            gap_count = cursor.fetchone()[0]

            if gap_count >= self.GAP_THRESHOLD:
                cursor.execute("""
                    SELECT gap_key, severity_score FROM detected_gaps
                    ORDER BY severity_score DESC LIMIT 5
                """)
                top_gaps = cursor.fetchall()
                affected = ", ".join([g[0] for g in top_gaps])
                self._record_trigger(
                    trigger_type="gap_threshold",
                    trigger_reason=f"Detected {gap_count} significant knowledge gaps",
                    severity="critical",
                    affected_categories=affected,
                    recommendation="Prioritize retraining on top knowledge gaps",
                )

            cursor.execute("""
                SELECT COUNT(*) FROM query_scores
                WHERE datetime(timestamp) >= datetime('now', '-24 hours')
            """)
            recent_queries = cursor.fetchone()[0]

            if recent_queries == 0:
                self._record_trigger(
                    trigger_type="staleness",
                    trigger_reason="No queries processed in last 24 hours",
                    severity="low",
                    affected_categories="general",
                    recommendation="Monitor system activity and trigger retraining when active",
                )

            cursor.execute("""
                SELECT AVG(overall_score) FROM query_scores
                WHERE datetime(timestamp) >= datetime('now', '-7 days')
            """)
            result = cursor.fetchone()
            recent_avg = result[0] if result[0] else 1.0

            cursor.execute("""
                SELECT AVG(overall_score) FROM query_scores
                WHERE datetime(timestamp) >= datetime('now', '-30 days')
                AND datetime(timestamp) < datetime('now', '-7 days')
            """)
            result = cursor.fetchone()
            older_avg = result[0] if result[0] else 1.0

            if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.ACCURACY_DROP_PERCENT:
                self._record_trigger(
                    trigger_type="confidence_decline",
                    trigger_reason="Confidence scores declining over time",
                    severity="medium",
                    affected_categories="general",
                    recommendation="Review recent queries and retrain on patterns",
                )

        except Exception as e:
            logger.error(f"Error checking retrain triggers: {e}")

    def _record_trigger(
        self,
        trigger_type: str,
        trigger_reason: str,
        severity: str,
        affected_categories: str,
        recommendation: str,
    ):
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT id FROM retrain_triggers
                WHERE trigger_type = ? AND datetime(triggered_at) >= datetime('now', '-1 hour')
                AND acknowledged = 0
            """, (trigger_type,))

            if cursor.fetchone():
                return

            cursor.execute("""
                INSERT INTO retrain_triggers
                (trigger_type, trigger_reason, severity, triggered_at, affected_categories, recommendation)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (trigger_type, trigger_reason, severity, datetime.now().isoformat(), affected_categories, recommendation))

            self.connection.commit()
            logger.warning(f"Retrain trigger recorded: {trigger_type} ({severity})")
        except sqlite3.Error as e:
            logger.error(f"Error recording trigger: {e}")
            self.connection.rollback()

    def record_outcome(
        self,
        question: str,
        sql: str,
        success: bool,
        latency: float,
        details: Dict[str, Any] = None,
    ):
        details = details or {}
        success_score = 0.9 if success else 0.3

        self.score_query(
            question=question,
            intent=details.get("intent", "unknown"),
            sql=sql,
            sql_accuracy=success_score,
            intent_match=details.get("intent_match", 0.5),
            table_selection=details.get("table_selection", 0.5),
            column_resolution=details.get("column_resolution", 0.5),
            join_correctness=details.get("join_correctness", 0.5),
            latency_ms=latency,
            narrative_quality=details.get("narrative_quality", 0.5),
            result=details.get("result"),
            query_type=details.get("query_type", "unknown"),
            tables_involved=details.get("tables_involved", []),
            complexity_level=details.get("complexity", "simple"),
            details=details,
        )

    def learn_from_correction(
        self,
        question: str,
        wrong_sql: str,
        correct_sql: str,
        reason: str,
    ):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO correction_feedback
                (question, wrong_sql, correct_sql, reason, feedback_at)
                VALUES (?, ?, ?, ?, ?)
            """, (question, wrong_sql, correct_sql, reason, datetime.now().isoformat()))

            self.connection.commit()
            logger.info(f"Correction recorded for question: {question[:50]}...")
        except sqlite3.Error as e:
            logger.error(f"Error recording correction: {e}")
            self.connection.rollback()

    def should_retrain(self) -> Tuple[bool, Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()

            reasons = {
                "accuracy_drop": False,
                "gap_threshold": False,
                "pattern_shift": False,
                "staleness": False,
                "confidence_decline": False,
                "details": [],
            }

            cursor.execute("""
                SELECT trigger_type, severity, affected_categories, recommendation
                FROM retrain_triggers
                WHERE acknowledged = 0
                ORDER BY triggered_at DESC
            """)

            triggers = cursor.fetchall()
            should_retrain = len(triggers) > 0

            for trigger in triggers:
                trigger_type = trigger[0]
                if trigger_type == "accuracy_drop":
                    reasons["accuracy_drop"] = True
                elif trigger_type == "gap_threshold":
                    reasons["gap_threshold"] = True
                elif trigger_type == "pattern_shift":
                    reasons["pattern_shift"] = True
                elif trigger_type == "staleness":
                    reasons["staleness"] = True
                elif trigger_type == "confidence_decline":
                    reasons["confidence_decline"] = True

                reasons["details"].append({
                    "type": trigger_type,
                    "severity": trigger[1],
                    "categories": trigger[2],
                    "recommendation": trigger[3],
                })

            return should_retrain, reasons

        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return False, {"error": str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        try:
            cursor = self.connection.cursor()

            cursor.execute("SELECT AVG(overall_score), COUNT(*) FROM query_scores")
            result = cursor.fetchone()
            overall_avg = result[0] or 0.5
            total_queries = result[1] or 0

            cursor.execute("SELECT SUM(success), COUNT(*) FROM query_scores")
            result = cursor.fetchone()
            successful = result[0] or 0
            total = result[1] or 1
            success_rate = successful / total if total > 0 else 0

            cursor.execute("SELECT COUNT(*) FROM detected_gaps WHERE severity_score > 0.3")
            critical_gaps = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM retrain_triggers WHERE acknowledged = 0")
            pending_triggers = cursor.fetchone()[0]

            grade = self._calculate_grade(overall_avg, success_rate, critical_gaps, pending_triggers)

            cursor.execute("""
                SELECT category_key, avg_overall, trend
                FROM category_performance
                WHERE category_type != 'time_period'
                ORDER BY avg_overall ASC
                LIMIT 5
            """)

            weak_areas = [
                {"category": row[0], "score": row[1], "trend": row[2]}
                for row in cursor.fetchall()
            ]

            cursor.execute("""
                SELECT category_key, avg_overall, trend
                FROM category_performance
                WHERE category_type != 'time_period'
                ORDER BY avg_overall DESC
                LIMIT 5
            """)

            strong_areas = [
                {"category": row[0], "score": row[1], "trend": row[2]}
                for row in cursor.fetchall()
            ]

            return {
                "grade": grade,
                "overall_score": overall_avg,
                "success_rate": success_rate,
                "total_queries_processed": total_queries,
                "critical_gaps": critical_gaps,
                "pending_retrain_triggers": pending_triggers,
                "weak_areas": weak_areas,
                "strong_areas": strong_areas,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}

    def _calculate_grade(
        self,
        overall_avg: float,
        success_rate: float,
        critical_gaps: int,
        pending_triggers: int,
    ) -> str:
        score = 0

        score += overall_avg * 40

        score += success_rate * 30

        gap_score = max(0, (1 - (critical_gaps / 10)) * 20)
        score += gap_score

        trigger_score = max(0, (1 - (pending_triggers / 5)) * 10)
        score += trigger_score

        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def get_weak_areas(self, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT category_key, category_type, avg_overall, sample_count, trend,
                       avg_sql_accuracy, avg_intent_match, avg_table_selection,
                       avg_column_resolution, avg_join_correctness
                FROM category_performance
                WHERE sample_count >= ?
                ORDER BY avg_overall ASC
                LIMIT ?
            """, (self.MIN_SAMPLES_FOR_STATS, limit))

            weak_areas = []
            for row in cursor.fetchall():
                weak_areas.append({
                    "category": row[0],
                    "type": row[1],
                    "overall_score": row[2],
                    "sample_count": row[3],
                    "trend": row[4],
                    "breakdown": {
                        "sql_accuracy": row[5],
                        "intent_match": row[6],
                        "table_selection": row[7],
                        "column_resolution": row[8],
                        "join_correctness": row[9],
                    }
                })

            return weak_areas

        except Exception as e:
            logger.error(f"Error getting weak areas: {e}")
            return []

    def get_strong_areas(self, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT category_key, category_type, avg_overall, sample_count, trend,
                       avg_sql_accuracy, avg_intent_match, avg_table_selection,
                       avg_column_resolution, avg_join_correctness
                FROM category_performance
                WHERE sample_count >= ?
                ORDER BY avg_overall DESC
                LIMIT ?
            """, (self.MIN_SAMPLES_FOR_STATS, limit))

            strong_areas = []
            for row in cursor.fetchall():
                strong_areas.append({
                    "category": row[0],
                    "type": row[1],
                    "overall_score": row[2],
                    "sample_count": row[3],
                    "trend": row[4],
                    "breakdown": {
                        "sql_accuracy": row[5],
                        "intent_match": row[6],
                        "table_selection": row[7],
                        "column_resolution": row[8],
                        "join_correctness": row[9],
                    }
                })

            return strong_areas

        except Exception as e:
            logger.error(f"Error getting strong areas: {e}")
            return []

    def get_improvement_plan(self) -> Dict[str, Any]:
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT gap_type, gap_key, severity_score, description, samples_count
                FROM detected_gaps
                WHERE severity_score > 0.3
                ORDER BY severity_score DESC
                LIMIT 10
            """)

            gaps = []
            for row in cursor.fetchall():
                gaps.append({
                    "type": row[0],
                    "key": row[1],
                    "severity": row[2],
                    "description": row[3],
                    "occurrences": row[4],
                    "priority": "critical" if row[2] > 0.6 else "high" if row[2] > 0.4 else "medium",
                })

            cursor.execute("""
                SELECT question, wrong_sql, correct_sql, reason
                FROM correction_feedback
                WHERE applied_fix = 0
                ORDER BY feedback_at DESC
                LIMIT 5
            """)

            corrections = []
            for row in cursor.fetchall():
                corrections.append({
                    "question": row[0],
                    "issue": row[3],
                    "examples_count": 1,
                })

            weak_categories = self.get_weak_areas(limit=3)

            return {
                "priority_gaps": gaps,
                "unapplied_corrections": corrections,
                "weak_categories_to_retrain": weak_categories,
                "estimated_effort": self._estimate_retraining_effort(gaps),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating improvement plan: {e}")
            return {"error": str(e)}

    def _estimate_retraining_effort(self, gaps: List[Dict]) -> str:
        if not gaps:
            return "minimal"

        total_severity = sum(g["severity"] for g in gaps)
        gap_count = len(gaps)

        if total_severity > 5 or gap_count > 8:
            return "extensive"
        elif total_severity > 3 or gap_count > 5:
            return "substantial"
        elif total_severity > 1 or gap_count > 2:
            return "moderate"
        else:
            return "minimal"

    def get_learning_report(self) -> Dict[str, Any]:
        health = self.get_system_health()
        weak = self.get_weak_areas(limit=5)
        strong = self.get_strong_areas(limit=5)
        should_retrain, reasons = self.should_retrain()
        improvement_plan = self.get_improvement_plan()

        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT date(timestamp) as day, AVG(overall_score), COUNT(*) as count
                FROM query_scores
                WHERE datetime(timestamp) >= datetime('now', '-30 days')
                GROUP BY date(timestamp)
                ORDER BY day ASC
            """)

            historical_scores = [
                {"date": row[0], "score": row[1], "queries": row[2]}
                for row in cursor.fetchall()
            ]

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            historical_scores = []

        return {
            "report_generated": datetime.now().isoformat(),
            "system_health": health,
            "weak_areas": weak,
            "strong_areas": strong,
            "should_retrain": should_retrain,
            "retrain_reasons": reasons,
            "improvement_plan": improvement_plan,
            "historical_performance": historical_scores,
        }

    def acknowledge_trigger(self, trigger_id: int):
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE retrain_triggers SET acknowledged = 1 WHERE id = ?", (trigger_id,))
            self.connection.commit()
            logger.info(f"Trigger {trigger_id} acknowledged")
        except sqlite3.Error as e:
            logger.error(f"Error acknowledging trigger: {e}")
            self.connection.rollback()

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    scorer = LearningScorer()

    score1 = scorer.score_query(
        question="How many claims were submitted in the last month?",
        intent="count",
        sql="SELECT COUNT(*) FROM claims WHERE submission_date >= DATE('now', '-30 days')",
        sql_accuracy=0.95,
        intent_match=0.98,
        table_selection=0.95,
        column_resolution=0.92,
        join_correctness=1.0,
        latency_ms=250,
        narrative_quality=0.9,
        result={"count": 1234},
        query_type="aggregation",
        tables_involved=["claims"],
        complexity_level="simple",
    )

    score2 = scorer.score_query(
        question="What's the relationship between member demographics and claim amounts?",
        intent="correlation",
        sql="SELECT m.*, c.amount FROM members m JOIN claims c ON m.id = c.member_id",
        sql_accuracy=0.4,
        intent_match=0.6,
        table_selection=0.8,
        column_resolution=0.5,
        join_correctness=0.7,
        latency_ms=3500,
        narrative_quality=0.3,
        result=None,
        query_type="join",
        tables_involved=["members", "claims"],
        complexity_level="complex",
    )

    scorer.learn_from_correction(
        question="Average claim amount per provider",
        wrong_sql="SELECT provider_id, AVG(amount) FROM claims GROUP BY provider_id",
        correct_sql="SELECT p.name, AVG(c.amount) FROM providers p JOIN claims c ON p.id = c.provider_id GROUP BY p.id, p.name",
        reason="Missing JOIN to providers table for proper grouping"
    )

    health = scorer.get_system_health()
    print(f"\nSystem Health: {health['grade']}")
    print(f"Overall Score: {health['overall_score']:.3f}")
    print(f"Success Rate: {health['success_rate']:.1%}")

    should_retrain, reasons = scorer.should_retrain()
    print(f"\nShould Retrain: {should_retrain}")
    if reasons["details"]:
        print("Reasons:")
        for reason in reasons["details"]:
            print(f"  - {reason['type']}: {reason['recommendation']}")

    plan = scorer.get_improvement_plan()
    print(f"\nPriority Gaps ({len(plan['priority_gaps'])} total):")
    for gap in plan["priority_gaps"][:3]:
        print(f"  - {gap['key']}: {gap['description']} (severity: {gap['severity']:.2f})")

    report = scorer.get_learning_report()
    print(f"\nSystem Grade: {report['system_health']['grade']}")
    print(f"Weak Areas: {len(report['weak_areas'])}")
    print(f"Strong Areas: {len(report['strong_areas'])}")

    scorer.close()
