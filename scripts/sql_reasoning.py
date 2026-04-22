import json
import logging
import sqlite3
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Set, Tuple, Any
from enum import Enum
import numpy as np
from pathlib import Path

logger = logging.getLogger('gpdm.sql_reasoning')


class AggregationFunc(Enum):
    SUM = 'SUM'
    AVG = 'AVG'
    COUNT = 'COUNT'
    MIN = 'MIN'
    MAX = 'MAX'
    STDDEV = 'STDDEV'


class JoinType(Enum):
    INNER = 'INNER JOIN'
    LEFT = 'LEFT JOIN'
    RIGHT = 'RIGHT JOIN'
    FULL = 'FULL OUTER JOIN'


@dataclass
class ColumnProfile:
    table: str
    name: str
    data_type: str
    sample_values: List[Any] = field(default_factory=list)
    distinct_count: int = 0
    null_pct: float = 0.0
    is_numeric: bool = False
    is_date: bool = False
    is_categorical: bool = False
    is_id: bool = False
    is_text: bool = False
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    semantic_tags: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repaired_sql: Optional[str] = None
    semantic_issues: List[str] = field(default_factory=list)


@dataclass
class SQLResult:
    sql: str
    tables_used: List[str]
    columns_used: List[str]
    confidence: float
    reasoning: List[str]
    approach: str
    joins_applied: List[str] = field(default_factory=list)
    derived_metrics: List[str] = field(default_factory=list)
    healthcare_intent: Optional[str] = None
    is_derived_metric: bool = False


@dataclass
class SelectNode:
    columns: List[Tuple[str, Optional[str]]]
    distinct: bool = False


@dataclass
class FromNode:
    table: str
    alias: Optional[str] = None


@dataclass
class JoinNode:
    join_type: JoinType
    table: str
    alias: Optional[str]
    on_condition: str


@dataclass
class WhereNode:
    conditions: List[Tuple[str, str]]


@dataclass
class GroupByNode:
    columns: List[str]


@dataclass
class HavingNode:
    conditions: List[str]


@dataclass
class OrderByNode:
    columns: List[Tuple[str, str]]


@dataclass
class LimitNode:
    count: int


class SQLQueryAST:

    def __init__(self):
        self.select: Optional[SelectNode] = None
        self.from_: Optional[FromNode] = None
        self.joins: List[JoinNode] = []
        self.where: Optional[WhereNode] = None
        self.group_by: Optional[GroupByNode] = None
        self.having: Optional[HavingNode] = None
        self.order_by: Optional[OrderByNode] = None
        self.limit: Optional[LimitNode] = None

    def render(self) -> str:
        parts = []

        if self.select:
            select_list = []
            for expr, alias in self.select.columns:
                if alias:
                    select_list.append(f"{expr} AS {alias}")
                else:
                    select_list.append(expr)
            distinct = 'DISTINCT ' if self.select.distinct else ''
            parts.append(f"SELECT {distinct}{', '.join(select_list)}")

        if self.from_:
            from_str = self.from_.table
            if self.from_.alias:
                from_str += f" {self.from_.alias}"
            parts.append(f"FROM {from_str}")

        for join in self.joins:
            join_str = f"{join.join_type.value} {join.table}"
            if join.alias:
                join_str += f" {join.alias}"
            join_str += f" ON {join.on_condition}"
            parts.append(join_str)

        if self.where and self.where.conditions:
            where_expr = ' AND '.join([f"{col} {op} {val}" for col, op, val in self.where.conditions])
            parts.append(f"WHERE {where_expr}")

        if self.group_by:
            parts.append(f"GROUP BY {', '.join(self.group_by.columns)}")

        if self.having and self.having.conditions:
            having_expr = ' AND '.join(self.having.conditions)
            parts.append(f"HAVING {having_expr}")

        if self.order_by:
            order_list = [f"{col} {direction}" for col, direction in self.order_by.columns]
            parts.append(f"ORDER BY {', '.join(order_list)}")

        if self.limit:
            parts.append(f"LIMIT {self.limit.count}")

        return '\n'.join(parts)


class SQLReasoningEngine:

    def __init__(self, schema_learner=None, db_path: str = '', embedder=None):
        self.schema_learner = schema_learner
        self.db_path = db_path
        self._embedder = embedder
        self._healthcare_metrics = self._build_healthcare_metrics()
        self._derived_metric_builders = self._build_derived_metric_builders()
        self._pattern_db = self._init_pattern_db()
        self._column_aliases = self._build_column_aliases()

        if embedder:
            logger.info(f"SQLReasoningEngine initialized with shared embedder (vocab={embedder.size})")
        else:
            logger.info(f"SQLReasoningEngine initialized (embedder will be built on first use)")


    @property
    def pattern_count(self) -> int:
        try:
            if not self.db_path or not self._pattern_db:
                return 0
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM query_patterns WHERE success = 1")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.debug(f"Failed to count patterns: {e}")
            return 0

    @property
    def template_count(self) -> int:
        try:
            if not self.db_path or not self._pattern_db:
                return 0
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM query_patterns")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.debug(f"Failed to get template count: {e}")
            return 0

    def generate(self, understanding: Optional[Dict] = None, question: Optional[str] = None) -> SQLResult:
        reasoning = []

        if not understanding and not question:
            reasoning.append("No input (understanding or question) provided")
            return SQLResult(
                sql='',
                tables_used=[],
                columns_used=[],
                confidence=0.0,
                reasoning=reasoning,
                approach='fallback'
            )

        corrected = self._correct_question(question) if question else question

        if understanding and question:
            understanding = self._correct_understanding_from_question(understanding, question)

        constructed = None
        if understanding:
            if question and 'question' not in understanding:
                understanding['question'] = question
            constructed = self._generate_from_understanding(understanding)
            if constructed.confidence >= 0.70:
                validation = self.validate_sql(constructed.sql)
                if validation.valid:
                    reasoning.append(f"Constructed from understanding (conf={constructed.confidence:.2f})")
                    constructed.reasoning = reasoning + constructed.reasoning
                    return constructed

        semantic_result = None
        required_tables = understanding.get('target_tables', []) if understanding else []
        if corrected and self.db_path:
            semantic_result = self._semantic_pattern_match(corrected)
            if semantic_result.confidence >= 0.70:
                sem_validation = self.validate_sql(semantic_result.sql)
                if sem_validation.valid:
                    if required_tables:
                        sql_lower = semantic_result.sql.lower()
                        tables_match = any(t.lower() in sql_lower for t in required_tables)
                        if not tables_match:
                            reasoning.append(
                                f"Semantic match rejected — SQL doesn't use required tables "
                                f"{required_tables}, got: {semantic_result.sql[:80]}"
                            )
                            semantic_result = None
                    if semantic_result:
                        if constructed and constructed.confidence >= 0.60:
                            if getattr(constructed, 'is_derived_metric', False):
                                reasoning.append("Derived metric construction preferred over pattern")
                                constructed.reasoning = reasoning + constructed.reasoning
                                return constructed
                        reasoning.append(f"Semantic match (conf={semantic_result.confidence:.2f})")
                        semantic_result.reasoning = reasoning + semantic_result.reasoning
                        return semantic_result
                else:
                    reasoning.append(f"Semantic match rejected — invalid SQL: {sem_validation.errors}")
                    semantic_result = None

        if constructed and constructed.confidence >= 0.50:
            validation = self.validate_sql(constructed.sql)
            if validation.valid:
                reasoning.append(f"Constructed from understanding (lower conf={constructed.confidence:.2f})")
                constructed.reasoning = reasoning + constructed.reasoning
                return constructed

        if semantic_result and semantic_result.confidence >= 0.50:
            sem_val = self.validate_sql(semantic_result.sql)
            if sem_val.valid:
                if required_tables:
                    sql_lower = semantic_result.sql.lower()
                    if not any(t.lower() in sql_lower for t in required_tables):
                        reasoning.append(f"Low-conf semantic match rejected — wrong tables")
                        semantic_result = None
                if semantic_result:
                    reasoning.append(f"Semantic match (lower conf={semantic_result.confidence:.2f})")
                    semantic_result.reasoning = reasoning + semantic_result.reasoning
                    return semantic_result

        if question:
            question_result = self._generate_from_question(question)
            if question_result.confidence > 0.40:
                question_result.reasoning = reasoning + question_result.reasoning
                return question_result
            reasoning.extend(question_result.reasoning)

        reasoning.append("Falling back to executive summary")
        fallback = self._fallback_generation()
        fallback.reasoning = reasoning
        return fallback

    def learn(self, question: str, sql: str, success: bool = True) -> None:
        if not self.db_path or not self._pattern_db:
            logger.debug("Pattern storage not available")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            q_hash = hashlib.sha256(question.encode()).hexdigest()
            pattern_id = hashlib.sha256(f"{question}{sql}".encode()).hexdigest()
            tables = ', '.join(self._extract_tables_from_sql(sql))
            healthcare_intent = self._infer_healthcare_intent(sql)

            conn.execute('''
                INSERT OR IGNORE INTO query_patterns
                (pattern_id, question_hash, question, sql, success, created_at,
                 tables_used, healthcare_intent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pattern_id, q_hash, question, sql, success,
                  datetime.now().isoformat(), tables, healthcare_intent))

            conn.commit()
            conn.close()
            logger.info(f"Learned pattern: {question[:50]}")
        except Exception as e:
            logger.warning(f"Failed to learn pattern: {e}")

    def validate_sql(self, sql: str) -> ValidationResult:
        errors = []
        warnings = []
        semantic_issues = []

        if not sql:
            return ValidationResult(valid=False, errors=['Empty SQL'])

        try:
            validate_db = self.db_path if self.db_path else ':memory:'
            conn = sqlite3.connect(validate_db)
            conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            test_sql = f"SELECT * FROM ({sql}) LIMIT 1"
            conn.execute(test_sql)
            conn.close()
        except sqlite3.Error as e:
            try:
                conn.close()
            except:
                pass
            errors.append(f"SQL syntax error: {str(e)}")

        if self.schema_learner and hasattr(self.schema_learner, 'tables'):
            schema_errors, schema_warnings = self._validate_against_schema(sql)
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)

        semantic_issues = self._check_semantic_issues(sql)

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            semantic_issues=semantic_issues
        )


    def _init_pattern_db(self) -> bool:
        if not self.db_path:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    question_hash TEXT UNIQUE,
                    question TEXT,
                    sql TEXT,
                    success BOOLEAN,
                    created_at TEXT,
                    tables_used TEXT,
                    healthcare_intent TEXT
                )
            ''')
            conn.commit()
            conn.close()
            logger.debug("Pattern database initialized")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize pattern DB: {e}")
            return False

    def _build_column_aliases(self) -> Dict[str, str]:
        return {
            'cost': ['paid_amount', 'claim_amount', 'allowed_amount', 'payment'],
            'member': ['member_id', 'patient_id', 'subscriber_id', 'enrollee_id'],
            'date': ['service_date', 'claim_date', 'admit_date', 'discharge_date'],
            'provider': ['provider_id', 'npi', 'facility_id'],
            'diagnosis': ['diagnosis_code', 'icd_code', 'primary_diagnosis'],
            'status': ['claim_status', 'visit_status', 'admission_status'],
        }

    def _build_healthcare_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {
            'pmpm': {
                'name': 'PMPM',
                'description': 'Per Member Per Month cost',
                'formula': 'SUM(cost) / COUNT(DISTINCT member)',
                'requires': ['cost', 'member', 'date']
            },
            'readmission_rate': {
                'name': 'Readmission Rate',
                'description': '30-day readmission rate',
                'formula': 'Self-join on encounters within 30 days',
                'requires': ['admission_date', 'discharge_date', 'encounter_id']
            },
            'denial_rate': {
                'name': 'Denial Rate',
                'description': 'Claims denied / total claims',
                'formula': 'SUM(CASE WHEN status IN (DENIED) THEN 1 ELSE 0 END) / COUNT(*)',
                'requires': ['claim_status']
            },
            'alos': {
                'name': 'ALOS',
                'description': 'Average Length of Stay',
                'formula': 'AVG(julianday(discharge) - julianday(admit))',
                'requires': ['admit_date', 'discharge_date']
            },
            'yield_rate': {
                'name': 'Yield Rate',
                'description': 'Paid / Billed ratio',
                'formula': 'SUM(paid) / SUM(billed)',
                'requires': ['paid_amount', 'billed_amount']
            },
            'er_per_1000': {
                'name': 'ER per 1000',
                'description': 'Emergency room visits per 1000 members',
                'formula': 'COUNT(ER visits) * 1000 / COUNT(DISTINCT members)',
                'requires': ['visit_type', 'member_id']
            },
        }

    def _build_derived_metric_builders(self) -> Dict[str, callable]:
        return {
            'pmpm': self._build_pmpm_sql,
            'readmission_rate': self._build_readmission_sql,
            'denial_rate': self._build_denial_rate_sql,
            'alos': self._build_alos_sql,
            'yield_rate': self._build_yield_rate_sql,
            'er_per_1000': self._build_er_per_1000_sql,
            'noshow_rate': self._build_noshow_rate_sql,
        }


    def _correct_understanding_from_question(self, understanding: Dict, question: str) -> Dict:
        q = question.lower()
        q_clean = re.sub(r'[^a-z0-9\s]', ' ', q)
        words = q_clean.split()

        EXPLICIT_TABLE_NAMES = {
            'referral': 'referrals', 'referrals': 'referrals',
            'appointment': 'appointments', 'appointments': 'appointments',
            'encounter': 'encounters', 'encounters': 'encounters',
            'claim': 'claims', 'claims': 'claims',
            'member': 'members', 'members': 'members',
            'provider': 'providers', 'providers': 'providers',
            'prescription': 'prescriptions', 'prescriptions': 'prescriptions',
            'diagnosis': 'diagnoses', 'diagnoses': 'diagnoses',
        }
        explicit_tables = []
        for w in words:
            if w in EXPLICIT_TABLE_NAMES:
                t = EXPLICIT_TABLE_NAMES[w]
                if t not in explicit_tables:
                    explicit_tables.append(t)

        COLUMN_TABLE_MAP = {
            'paid amount': ('claims', 'PAID_AMOUNT', 'METRIC'),
            'billed amount': ('claims', 'BILLED_AMOUNT', 'METRIC'),
            'allowed amount': ('claims', 'ALLOWED_AMOUNT', 'METRIC'),
            'risk score': ('members', 'RISK_SCORE', 'METRIC'),
            'length of stay': ('encounters', 'LENGTH_OF_STAY', 'METRIC'),
            'days supply': ('prescriptions', 'DAYS_SUPPLY', 'METRIC'),
            'panel size': ('providers', 'PANEL_SIZE', 'METRIC'),
            'drug class': ('prescriptions', 'DRUG_CLASS', 'DIMENSION'),
            'rvu': ('cpt_codes', 'RVU', 'METRIC'),
            'cpt': ('cpt_codes', 'CPT_CODE', 'DIMENSION'),
            'los': ('encounters', 'LENGTH_OF_STAY', 'METRIC'),
            'copay': ('claims', 'COPAY', 'METRIC'),
            'coinsurance': ('claims', 'COINSURANCE', 'METRIC'),
        }
        column_override_table = None
        column_override_col = None
        column_override_type = None
        for keyword, (tbl, col, ctype) in COLUMN_TABLE_MAP.items():
            if keyword in q:
                column_override_table = tbl
                column_override_col = col
                column_override_type = ctype
                break

        GEO_FILTERS = {
            'california': {'column': 'KP_REGION', 'values': ['NCAL', 'SCAL'], 'op': 'IN'},
            'northern california': {'column': 'KP_REGION', 'values': ['NCAL'], 'op': '='},
            'southern california': {'column': 'KP_REGION', 'values': ['SCAL'], 'op': '='},
            'hawaii': {'column': 'KP_REGION', 'values': ['HI'], 'op': '='},
            'colorado': {'column': 'KP_REGION', 'values': ['CO'], 'op': '='},
            'georgia': {'column': 'KP_REGION', 'values': ['GA'], 'op': '='},
            'northwest': {'column': 'KP_REGION', 'values': ['NW'], 'op': '='},
            'mid-atlantic': {'column': 'KP_REGION', 'values': ['MAS', 'MID'], 'op': 'IN'},
        }
        geo_filter = None
        for geo_name, geo_info in GEO_FILTERS.items():
            if geo_name in q:
                geo_filter = geo_info
                break

        COMBO_OVERRIDES = [
            (['denied', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'DENIED'}]),
            (['pending', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'PENDING'}]),
            (['expired', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'EXPIRED'}]),
            (['cancelled', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'CANCELLED'}]),
            (['approved', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'APPROVED'}]),
            (['completed', 'referral'], 'referrals', [{'column': 'STATUS', 'op': '=', 'value': 'COMPLETED'}]),
            (['cancelled', 'appointment'], 'appointments', [{'column': 'STATUS', 'op': '=', 'value': 'CANCELLED'}]),
            (['denied', 'claim'], 'claims', [{'column': 'CLAIM_STATUS', 'op': '=', 'value': 'DENIED'}]),
        ]
        combo_table = None
        combo_filters = None
        for keywords_needed, tbl, filters in COMBO_OVERRIDES:
            if all(any(k in w for w in words) for k in keywords_needed):
                combo_table = tbl
                combo_filters = filters
                break

        corrected = dict(understanding)

        if combo_table:
            corrected['target_tables'] = [combo_table]
            existing_filters = corrected.get('filter_conditions', [])
            corrected['filter_conditions'] = existing_filters + combo_filters
            corrected['entities'] = [e for e in corrected.get('entities', [])
                                     if e.get('resolved_to') != 'denial' or combo_table == 'claims']

        elif column_override_table:
            current_tables = corrected.get('target_tables', [])
            if column_override_table not in current_tables:
                corrected['target_tables'] = [column_override_table]
            if column_override_type == 'METRIC':
                existing_metrics = corrected.get('metric_columns', [])
                if not any(column_override_col.upper() in str(m).upper() for m in existing_metrics):
                    agg = 'AVG'
                    if any(kw in q for kw in ['how many', 'number of', 'count']):
                        agg = 'COUNT'
                    elif any(kw in q for kw in ['total', 'overall', 'sum']):
                        agg = 'SUM'
                    elif any(kw in q for kw in ['maximum', 'highest', 'largest']):
                        agg = 'MAX'
                    elif any(kw in q for kw in ['minimum', 'lowest', 'smallest']):
                        agg = 'MIN'
                    elif any(kw in q for kw in ['average', 'mean', 'avg']):
                        agg = 'AVG'
                    corrected['metric_columns'] = [(column_override_col, agg)] + [
                        m for m in existing_metrics
                        if not (isinstance(m, str) and m.upper() == column_override_col.upper())
                    ]
            elif column_override_type == 'DIMENSION':
                existing_dims = corrected.get('dimension_columns', [])
                if not any(column_override_col.upper() in str(d).upper() for d in existing_dims):
                    corrected['dimension_columns'] = [column_override_col] + existing_dims

        elif explicit_tables:
            current_tables = corrected.get('target_tables', [])
            if not current_tables:
                corrected['target_tables'] = explicit_tables

        if geo_filter:
            existing_filters = corrected.get('filter_conditions', [])
            existing_filters = [f for f in existing_filters
                                if not any(geo_word in str(f).lower()
                                           for geo_word in ['california', 'hawaii', 'colorado',
                                                            'georgia', 'northwest', 'atlantic'])]
            has_geo = any('KP_REGION' in str(f) for f in existing_filters)
            if not has_geo:
                if geo_filter['op'] == '=':
                    existing_filters.append({
                        'column': geo_filter['column'],
                        'op': '=',
                        'value': geo_filter['values'][0]
                    })
                else:
                    existing_filters.append({
                        'column': geo_filter['column'],
                        'op': 'IN',
                        'value': geo_filter['values']
                    })
                corrected['filter_conditions'] = existing_filters

            if 'cost' in q or 'paid' in q or 'billed' in q or 'expense' in q:
                current_tables = corrected.get('target_tables', [])
                if 'claims' not in current_tables:
                    corrected['target_tables'] = ['claims'] + current_tables

        STATUS_FILTER_MAP = {
            'denied': {'claims': ('CLAIM_STATUS', 'DENIED'), 'referrals': ('STATUS', 'DENIED'), 'appointments': ('STATUS', 'DENIED')},
            'cancelled': {'claims': ('CLAIM_STATUS', 'CANCELLED'), 'referrals': ('STATUS', 'CANCELLED'), 'appointments': ('STATUS', 'CANCELLED')},
            'pending': {'referrals': ('STATUS', 'PENDING'), 'appointments': ('STATUS', 'PENDING'), 'claims': ('CLAIM_STATUS', 'PENDING')},
            'approved': {'referrals': ('STATUS', 'APPROVED'), 'claims': ('CLAIM_STATUS', 'APPROVED')},
            'completed': {'referrals': ('STATUS', 'COMPLETED'), 'claims': ('CLAIM_STATUS', 'COMPLETED')},
            'paid': {'claims': ('CLAIM_STATUS', 'PAID')},
            'rejected': {'claims': ('CLAIM_STATUS', 'REJECTED'), 'referrals': ('STATUS', 'REJECTED')},
        }
        for status_word, table_filters in STATUS_FILTER_MAP.items():
            if status_word in q:
                current_tables = corrected.get('target_tables', [])
                for table in current_tables:
                    if table in table_filters:
                        col, val = table_filters[table]
                        existing_filters = corrected.get('filter_conditions', [])
                        if not any(f.get('column') == col for f in existing_filters):
                            existing_filters.append({
                                'column': col,
                                'op': '=',
                                'value': val
                            })
                            corrected['filter_conditions'] = existing_filters
                        break

        count_keywords = ['how many', 'total number', 'number of', 'count of', 'count all']
        is_count_query = any(kw in q for kw in count_keywords)
        if is_count_query:
            corrected['intent'] = 'count'
            corrected['sql_approach'] = 'SELECT COUNT(*)'
            corrected['metric_columns'] = []
            corrected['dimension_columns'] = []
            if explicit_tables:
                corrected['target_tables'] = [explicit_tables[0]]

        max_keywords = ['maximum', 'most expensive', 'largest', 'biggest', 'highest', 'max']
        min_keywords = ['minimum', 'least expensive', 'smallest', 'cheapest', 'lowest', 'min']
        is_max_query = any(kw in q for kw in max_keywords)
        is_min_query = any(kw in q for kw in min_keywords)
        if is_max_query or is_min_query:
            agg_func = 'MAX' if is_max_query else 'MIN'
            existing_metrics = corrected.get('metric_columns', [])
            if existing_metrics:
                fixed_metrics = []
                for m in existing_metrics:
                    if isinstance(m, (list, tuple)) and len(m) >= 2:
                        fixed_metrics.append((m[0], agg_func))
                    elif isinstance(m, str):
                        fixed_metrics.append((m, agg_func))
                    else:
                        fixed_metrics.append(m)
                corrected['metric_columns'] = fixed_metrics
            else:
                if 'paid' in q or 'cost' in q or 'amount' in q:
                    corrected['metric_columns'] = [('PAID_AMOUNT', agg_func)]
                    if 'claims' not in corrected.get('target_tables', []):
                        corrected['target_tables'] = ['claims'] + corrected.get('target_tables', [])
                elif 'billed' in q:
                    corrected['metric_columns'] = [('BILLED_AMOUNT', agg_func)]
                    if 'claims' not in corrected.get('target_tables', []):
                        corrected['target_tables'] = ['claims'] + corrected.get('target_tables', [])
                elif 'risk' in q or 'score' in q:
                    corrected['metric_columns'] = [('RISK_SCORE', agg_func)]
            if 'single' in q:
                if explicit_tables:
                    corrected['target_tables'] = [explicit_tables[0]]
                elif 'claim' in q:
                    corrected['target_tables'] = ['claims']

        if 'pmpm' in q:
            current_tables = corrected.get('target_tables', [])
            if 'claims' in current_tables and current_tables[0] != 'claims':
                current_tables.remove('claims')
                corrected['target_tables'] = ['claims'] + current_tables
            elif 'claims' not in current_tables:
                corrected['target_tables'] = ['claims'] + current_tables

        PLAN_TYPE_FIXES = {
            'medicare': 'Medicare Advantage',
            'medicaid': 'Medicaid',
            'hmo': 'HMO', 'ppo': 'PPO', 'epo': 'EPO', 'hdhp': 'HDHP',
        }
        existing_filters = corrected.get('filter_conditions', [])
        for i, f in enumerate(existing_filters):
            col = str(f.get('column', '')).upper()
            val = str(f.get('value', '')).lower()
            if 'PLAN_TYPE' in col or 'plan_type' in str(f.get('column', '')):
                if val in PLAN_TYPE_FIXES:
                    existing_filters[i] = dict(f, value=PLAN_TYPE_FIXES[val])
            vals = f.get('values', [])
            if vals and ('PLAN_TYPE' in col or 'plan_type' in str(f.get('column', ''))):
                existing_filters[i]['values'] = [PLAN_TYPE_FIXES.get(v.lower(), v) for v in vals]
        corrected['filter_conditions'] = existing_filters

        is_percentage_query = any(kw in q for kw in ['percentage', 'percent', '%', 'pct', 'rate'])
        if is_percentage_query:
            corrected['intent'] = 'percentage'
            corrected['sql_approach'] = 'CASE WHEN aggregation with 100.0 * numerator / denominator'
            corrected['is_rate_query'] = True

        if 'most common' in q or 'most frequent' in q or 'top diagnoses' in q or 'top reasons' in q:
            corrected['intent'] = 'top_n'
            corrected['sql_approach'] = 'GROUP BY dimension, COUNT(*), ORDER BY COUNT(*) DESC, LIMIT 10'
            if 'diagnosis' in q or 'diagnoses' in q:
                corrected['target_tables'] = ['diagnoses']
                corrected['dimension_columns'] = ['ICD10_DESCRIPTION']
            elif 'visit' in q:
                corrected['target_tables'] = ['encounters']
                corrected['dimension_columns'] = ['VISIT_TYPE']
            elif 'facility' in q:
                corrected['target_tables'] = ['encounters'] if 'encounters' not in corrected.get('target_tables', []) else corrected.get('target_tables', [])
                corrected['dimension_columns'] = ['FACILITY']
            elif 'department' in q:
                corrected['target_tables'] = ['encounters'] if 'encounters' not in corrected.get('target_tables', []) else corrected.get('target_tables', [])
                corrected['dimension_columns'] = ['DEPARTMENT']

        if ('prescription' in q or 'medication' in q or 'drug' in q or 'pharmacy' in q):
            if 'cost' in q or 'expense' in q or 'spend' in q or 'total' in q:
                corrected['target_tables'] = ['prescriptions']
                corrected['metric_columns'] = [('COST', 'SUM')]

        if ('cost' in q or 'expensive' in q or 'high cost' in q) and not column_override_table:
            dims = corrected.get('dimension_columns', [])
            tables = corrected.get('target_tables', [])
            if not dims and 'claims' in tables:
                if 'cause' in q or 'causing' in q or 'driver' in q or 'why' in q:
                    corrected['dimension_columns'] = ['FACILITY']
                    corrected['intent'] = 'breakdown'
                    corrected['sql_approach'] = 'GROUP BY dimension, aggregate metrics, ORDER BY DESC'
                    if not corrected.get('metric_columns'):
                        corrected['metric_columns'] = [('PAID_AMOUNT', 'SUM'), ('BILLED_AMOUNT', 'SUM')]

        if 'category' in q and column_override_table == 'cpt_codes':
            existing_dims = corrected.get('dimension_columns', [])
            if not any('CATEGORY' in str(d).upper() for d in existing_dims):
                corrected['dimension_columns'] = ['CATEGORY'] + existing_dims
            corrected['intent'] = 'breakdown'
            corrected['sql_approach'] = 'GROUP BY dimension, aggregate metrics'

        if ('cost per encounter' in q or 'cost per visit' in q) or ('per encounter' in q and 'cost' in q) or ('per visit' in q and 'cost' in q):
            corrected['target_tables'] = ['claims']
            corrected['metric_columns'] = [('PAID_AMOUNT', 'SUM')]
            corrected['dimension_columns'] = []
            corrected['sql_approach'] = 'SUM(PAID_AMOUNT) / COUNT(DISTINCT ENCOUNTER_ID)'
            corrected['intent'] = 'cost_per_encounter'

        if 'average panel size' in q and 'per provider' in q:
            corrected['target_tables'] = ['providers']
            corrected['metric_columns'] = [('PANEL_SIZE', 'AVG')]
            corrected['dimension_columns'] = []
            corrected['intent'] = 'average'
            corrected['sql_approach'] = 'SELECT AVG(PANEL_SIZE) FROM providers'

        return corrected

    @staticmethod
    def _correct_question(question: str) -> str:
        FIXES = {
            'clams': 'claims', 'calims': 'claims', 'cliams': 'claims',
            'deniel': 'denial', 'deniels': 'denials',
            'membrs': 'members', 'memebers': 'members',
            'presciption': 'prescription', 'presciptions': 'prescriptions',
            'perscription': 'prescription',
            'averge': 'average', 'avrage': 'average',
            'encountr': 'encounter', 'encountrs': 'encounters',
            'provders': 'providers', 'providrs': 'providers',
            'referall': 'referral', 'referals': 'referrals',
            'diagnoseis': 'diagnoses', 'dignoses': 'diagnoses',
            'appoitment': 'appointment', 'appoitments': 'appointments',
            'utilzation': 'utilization', 'performace': 'performance',
            'specilty': 'specialty', 'facilty': 'facility',
        }
        words = question.lower().split()
        return ' '.join(FIXES.get(w.strip('.,?!;:'), w) for w in words)

    _TABLE_ENTITY_MAP = {
        'claims': 'claims', 'claim': 'claims',
        'encounters': 'encounters', 'encounter': 'encounters',
        'members': 'members', 'member': 'members',
        'diagnoses': 'diagnoses', 'diagnosis': 'diagnoses',
        'providers': 'providers', 'provider': 'providers',
        'prescriptions': 'prescriptions', 'prescription': 'prescriptions',
        'appointments': 'appointments', 'appointment': 'appointments',
        'referrals': 'referrals', 'referral': 'referrals',
        'cpt_codes': 'cpt_codes',
        'patient': 'members', 'patients': 'members', 'enrolled': 'members',
        'enrollment': 'members', 'demographics': 'members', 'population': 'members',
        'race': 'members', 'gender': 'members', 'language': 'members',
        'city': 'members', 'state': 'members', 'risk': 'members',
        'visit': 'encounters', 'visits': 'encounters', 'admission': 'encounters',
        'admissions': 'encounters', 'inpatient': 'encounters', 'outpatient': 'encounters',
        'telehealth': 'encounters', 'emergency': 'encounters', 'er': 'encounters',
        'disposition': 'encounters', 'discharged': 'encounters', 'ama': 'encounters',
        'readmission': 'encounters', 'readmissions': 'encounters',
        'expired': 'encounters', 'admitted': 'encounters', 'transferred': 'encounters',
        'complaint': 'encounters', 'department': 'encounters',
        'los': 'encounters', 'lengthofstay': 'encounters',
        'medication': 'prescriptions', 'medications': 'prescriptions',
        'pharmacy': 'prescriptions', 'drug': 'prescriptions', 'drugs': 'prescriptions',
        'rx': 'prescriptions', 'metformin': 'prescriptions', 'statin': 'prescriptions',
        'refill': 'prescriptions', 'refills': 'prescriptions',
        'specialist': 'providers', 'cardiologist': 'providers', 'cardiologists': 'providers',
        'surgeon': 'providers', 'surgeons': 'providers', 'pediatrician': 'providers',
        'psychiatrist': 'providers', 'oncologist': 'providers', 'md': 'providers',
        'mds': 'providers', 'nps': 'providers', 'pas': 'providers', 'npi': 'providers',
        'panel': 'providers',
        'noshow': 'appointments', 'noshows': 'appointments', 'scheduled': 'appointments',
        'pcp': 'appointments', 'cancelled': 'appointments',
        'duration': 'appointments', 'booked': 'appointments',
        'severity': 'diagnoses', 'chronic': 'diagnoses', 'hcc': 'diagnoses',
        'rvu': 'cpt_codes', 'cpt': 'cpt_codes', 'procedure': 'cpt_codes',
        'procedures': 'cpt_codes',
        'denial': 'claims', 'denials': 'claims', 'denied': 'claims',
        'paid': 'claims', 'billed': 'claims', 'copay': 'claims',
        'copays': 'claims', 'coinsurance': 'claims', 'adjudicate': 'claims',
        'adjudication': 'claims', 'yield': 'claims',
    }

    def _detect_question_tables(self, question: str) -> List[str]:
        q_lower = question.lower()
        q_clean = re.sub(r'[^a-z0-9\s]', ' ', q_lower)
        words = q_clean.split()

        detected = set()
        for w in words:
            if w in self._TABLE_ENTITY_MAP:
                detected.add(self._TABLE_ENTITY_MAP[w])

        for i in range(len(words) - 1):
            bigram = words[i] + words[i+1]
            if bigram in self._TABLE_ENTITY_MAP:
                detected.add(self._TABLE_ENTITY_MAP[bigram])

        phrase_map = {
            'length of stay': 'encounters',
            'no show': 'appointments',
            'no-show': 'appointments',
            'risk score': 'members',
            'chronic condition': 'diagnoses',
            'panel size': 'providers',
            'days supply': 'prescriptions',
            'medication class': 'prescriptions',
            'medication name': 'prescriptions',
            'medication cost': 'prescriptions',
            'prescription cost': 'prescriptions',
            'drug type': 'prescriptions',
            'drug class': 'prescriptions',
            'appointment type': 'appointments',
            'appointment status': 'appointments',
            'referral completion': 'referrals',
            'referral status': 'referrals',
            'visit type': 'encounters',
            'encounter status': 'encounters',
            'chief complaint': 'encounters',
            'claim status': 'claims',
            'claim type': 'claims',
            'denial rate': 'claims',
            'denial reason': 'claims',
            'member responsibility': 'claims',
            'plan type': 'claims',
            'cpt code': 'cpt_codes',
            'accepting new patients': 'providers',
        }
        for phrase, table in phrase_map.items():
            if phrase in q_lower:
                detected.add(table)

        return list(detected)

    def _sql_uses_any_table(self, sql_text: str, required_tables: List[str]) -> bool:
        sql_lower = sql_text.lower()
        return any(t in sql_lower for t in required_tables)

    def _semantic_pattern_match(self, question: str) -> SQLResult:
        reasoning = []

        if not self.db_path:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='semantic_match')

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT question, sql, healthcare_intent FROM query_patterns WHERE success = 1"
            )
            patterns = cursor.fetchall()
            conn.close()

            if not patterns:
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.0, reasoning=reasoning, approach='semantic_match')

            embedder = getattr(self, '_embedder', None)
            if embedder is None:
                embedder = self._build_lightweight_embedder(patterns)
                self._embedder = embedder

            if embedder is None:
                reasoning.append("No embedder available, falling back to token matching")
                return self._token_pattern_match(question)

            q_vec = embedder.encode(question)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0:
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.0, reasoning=reasoning, approach='semantic_match')

            if not hasattr(self, '_pattern_cache') or self._pattern_cache_size != len(patterns):
                self._pattern_cache = []
                for stored_q, stored_sql, intent in patterns:
                    p_vec = embedder.encode(stored_q)
                    self._pattern_cache.append((stored_q, stored_sql, intent, p_vec))
                self._pattern_cache_size = len(patterns)

            required_tables = self._detect_question_tables(question)

            best_match = None
            best_sim = 0.0
            best_constrained = None
            best_constrained_sim = 0.0

            for stored_q, stored_sql, intent, p_vec in self._pattern_cache:
                p_norm = float(np.linalg.norm(p_vec))
                if p_norm == 0:
                    continue

                sim = float(np.dot(q_vec, p_vec)) / (q_norm * p_norm)

                if sim > best_sim:
                    best_sim = sim
                    best_match = (stored_sql, intent, stored_q, sim)

                if required_tables and sim > best_constrained_sim:
                    if self._sql_uses_any_table(stored_sql, required_tables):
                        best_constrained_sim = sim
                        best_constrained = (stored_sql, intent, stored_q, sim)

            chosen = None
            chosen_sim = 0.0
            if required_tables and best_constrained and best_constrained_sim > 0.55:
                chosen = best_constrained
                chosen_sim = best_constrained_sim
                reasoning.append(f"Semantic+table-constrained to {required_tables}")
            elif best_match and best_sim > 0.65:
                chosen = best_match
                chosen_sim = best_sim

            if chosen and chosen_sim > 0.50:
                sql, intent, matched_q, sim = chosen
                confidence = min(0.96, sim)
                reasoning.append(f"Semantic match '{matched_q[:50]}' sim={sim:.3f}")
                return SQLResult(
                    sql=sql,
                    tables_used=self._extract_tables_from_sql(sql),
                    columns_used=self._extract_columns_from_sql(sql),
                    confidence=confidence,
                    reasoning=reasoning,
                    approach='semantic_match',
                    healthcare_intent=intent
                )

        except Exception as e:
            reasoning.append(f"Semantic matching failed: {e}")
            logger.debug(f"Semantic matching error: {e}")

        return SQLResult(sql='', tables_used=[], columns_used=[],
                       confidence=0.0, reasoning=reasoning, approach='semantic_match')

    def _build_lightweight_embedder(self, patterns):
        try:
            from deep_understanding import _SkipGramEmbedder, HEALTHCARE_CONCEPTS

            if not self.schema_learner:
                logger.warning("Cannot build embedder without schema_learner")
                return None

            embedder = _SkipGramEmbedder(dim=384, epochs=8)
            embedder.train(self.schema_learner, HEALTHCARE_CONCEPTS)

            if embedder._trained:
                logger.info(f"Built lightweight embedder: vocab={embedder.size}, dim={embedder.dim}")
                return embedder
            else:
                logger.warning("Lightweight embedder training failed")
                return None
        except ImportError:
            logger.warning("deep_understanding not available for embedder")
            return None
        except Exception as e:
            logger.warning(f"Failed to build lightweight embedder: {e}")
            return None

    def _token_pattern_match(self, question: str) -> SQLResult:
        reasoning = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT question, sql, healthcare_intent FROM query_patterns WHERE success = 1"
            )
            patterns = cursor.fetchall()
            conn.close()

            STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                         'did', 'have', 'has', 'had', 'be', 'been', 'being', 'will',
                         'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'it', 'its', 'we', 'our', 'us', 'me', 'my', 'i', 'you',
                         'your', 'they', 'their', 'them', 'this', 'that', 'these',
                         'what', 'which', 'who', 'how', 'when', 'where', 'show',
                         'give', 'tell', 'get', 'all', 'some', 'any', 'many', 'much'}

            def tokenize(text):
                return [w for w in re.split(r'[^a-z0-9]+', text.lower())
                        if w and w not in STOPWORDS and len(w) > 1]

            required_tables = self._detect_question_tables(question)
            q_tokens = set(tokenize(question))
            if not q_tokens:
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.0, reasoning=reasoning, approach='token_match')

            best = None
            best_score = 0.0
            for stored_q, stored_sql, intent in patterns:
                p_tokens = set(tokenize(stored_q))
                if not p_tokens:
                    continue
                overlap = len(q_tokens & p_tokens)
                union = len(q_tokens | p_tokens)
                jaccard = overlap / union if union else 0
                if required_tables and not self._sql_uses_any_table(stored_sql, required_tables):
                    jaccard *= 0.3
                if jaccard > best_score:
                    best_score = jaccard
                    best = (stored_sql, intent, stored_q)

            if best and best_score > 0.25:
                sql, intent, matched_q = best
                confidence = min(0.90, 0.40 + best_score)
                reasoning.append(f"Token match '{matched_q[:50]}' jaccard={best_score:.3f}")
                return SQLResult(
                    sql=sql, tables_used=self._extract_tables_from_sql(sql),
                    columns_used=self._extract_columns_from_sql(sql),
                    confidence=confidence, reasoning=reasoning,
                    approach='token_match', healthcare_intent=intent)
        except Exception as e:
            reasoning.append(f"Token matching failed: {e}")

        return SQLResult(sql='', tables_used=[], columns_used=[],
                       confidence=0.0, reasoning=reasoning, approach='token_match')

    def _generate_from_understanding(self, understanding: Dict) -> SQLResult:
        reasoning = []

        try:
            intent = understanding.get('intent', 'aggregate')
            reasoning.append(f"Intent: {intent}")

            target_tables = understanding.get('target_tables', [])
            metric_columns = understanding.get('metric_columns', [])
            dimension_columns = understanding.get('dimension_columns', [])
            filter_conditions = understanding.get('filter_conditions', [])
            sql_approach = understanding.get('sql_approach', '')
            is_rate_query = understanding.get('is_rate_query', False)

            if not target_tables:
                reasoning.append("No target tables identified")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            if intent == 'cost_per_encounter' and target_tables and 'claims' in target_tables:
                reasoning.append("Building cost per encounter query")
                sql = "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT ENCOUNTER_ID), 0), 2) as cost_per_encounter FROM claims WHERE ENCOUNTER_ID IS NOT NULL"

                validation = self.validate_sql(sql)
                return SQLResult(
                    sql=sql,
                    tables_used=['claims'],
                    columns_used=['PAID_AMOUNT', 'ENCOUNTER_ID'],
                    confidence=0.90 if validation.valid else 0.50,
                    reasoning=reasoning + ["Generated cost per encounter query"],
                    approach='template_direct',
                    healthcare_intent='cost_per_encounter'
                )

            if intent == 'top_n' and dimension_columns:
                reasoning.append("Building top N query with GROUP BY")
                if not target_tables:
                    target_tables = ['diagnoses']

                table = target_tables[0]
                dim_col = dimension_columns[0]

                sql = f"SELECT {dim_col}, COUNT(*) as cnt FROM {table} GROUP BY {dim_col} ORDER BY cnt DESC LIMIT 10"

                validation = self.validate_sql(sql)
                return SQLResult(
                    sql=sql,
                    tables_used=target_tables,
                    columns_used=[dim_col],
                    confidence=0.85 if validation.valid else 0.50,
                    reasoning=reasoning + ["Generated top N query"],
                    approach='template_direct',
                    healthcare_intent='top_n'
                )

            if is_rate_query and target_tables and 'claims' in target_tables:
                reasoning.append("Building percentage/rate query with CASE WHEN")
                status_filter = None
                status_col = 'CLAIM_STATUS'
                for f in filter_conditions:
                    if f.get('column', '').upper() in ['CLAIM_STATUS', 'STATUS']:
                        status_filter = f
                        status_col = f.get('column', 'CLAIM_STATUS')
                        break

                if status_filter:
                    status_value = status_filter.get('value', '')
                    filter_conditions = [f for f in filter_conditions if f.get('column', '').upper() not in ['CLAIM_STATUS', 'STATUS']]
                    understanding['filter_conditions'] = filter_conditions

                    sql = f"SELECT ROUND(100.0 * SUM(CASE WHEN {status_col}='{status_value}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as percentage FROM {target_tables[0]}"

                    if filter_conditions:
                        where_parts = []
                        for f in filter_conditions:
                            col = f.get('column', '')
                            op = f.get('op', '=')
                            val = f.get('value', '')
                            if op == '=':
                                where_parts.append(f"{col}='{val}'")
                            elif op in ['IN', 'in']:
                                vals = f.get('values', [val])
                                vals_str = ','.join(f"'{v}'" for v in vals)
                                where_parts.append(f"{col} IN ({vals_str})")
                        if where_parts:
                            sql += " WHERE " + " AND ".join(where_parts)

                    validation = self.validate_sql(sql)
                    return SQLResult(
                        sql=sql,
                        tables_used=target_tables,
                        columns_used=[status_col],
                        confidence=0.85 if validation.valid else 0.50,
                        reasoning=reasoning + ["Generated percentage query with CASE WHEN"],
                        approach='template_direct',
                        healthcare_intent='percentage'
                    )

            for metric_name, builder in self._derived_metric_builders.items():
                if self._is_derived_metric_intent(understanding, metric_name):
                    result = builder(understanding)
                    if result and result.confidence > 0.5 and result.sql:
                        dm_validation = self.validate_sql(result.sql)
                        if dm_validation.valid:
                            reasoning.append(f"Derived metric: {metric_name}")
                            result.reasoning = reasoning
                            result.is_derived_metric = True
                            return result
                        else:
                            reasoning.append(f"Derived metric {metric_name} SQL invalid: {dm_validation.errors}")

            ast = self._build_ast_from_understanding(understanding)
            sql = ast.render()

            if not sql:
                reasoning.append("Could not build query AST")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.2, reasoning=reasoning, approach='template_direct')

            validation = self.validate_sql(sql)
            confidence = 0.75 if validation.valid else 0.20

            reasoning.append(f"Generated query with {len(target_tables)} tables")

            return SQLResult(
                sql=sql,
                tables_used=target_tables,
                columns_used=self._extract_columns_from_understanding(understanding),
                confidence=confidence,
                reasoning=reasoning,
                approach='template_direct',
                healthcare_intent=understanding.get('intent')
            )

        except Exception as e:
            reasoning.append(f"Generation from understanding failed: {e}")
            logger.debug(f"Understanding generation error: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _generate_from_question(self, question: str) -> SQLResult:
        reasoning = []

        try:
            keywords = self._extract_keywords(question)
            reasoning.append(f"Keywords: {keywords}")

            tables = self._identify_tables_from_keywords(keywords)
            if not tables:
                reasoning.append("Could not identify tables")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.0, reasoning=reasoning, approach='fallback')

            if 'top' in question.lower() or 'highest' in question.lower():
                metric_col = self._find_metric_column(tables[0])
                if metric_col:
                    ast = SQLQueryAST()
                    ast.select = SelectNode(columns=[(f'{tables[0]}.*', None)])
                    ast.from_ = FromNode(table=tables[0])
                    ast.order_by = OrderByNode(columns=[(metric_col, 'DESC')])
                    ast.limit = LimitNode(count=10)
                    sql = ast.render()
                    reasoning.append("Generated TOP N query")
                    return SQLResult(sql=sql, tables_used=tables, columns_used=[],
                                   confidence=0.5, reasoning=reasoning, approach='fallback')

            ast = SQLQueryAST()
            ast.select = SelectNode(columns=[('COUNT(*)', 'total')])
            ast.from_ = FromNode(table=tables[0])
            sql = ast.render()
            reasoning.append("Generated COUNT(*) fallback")
            return SQLResult(sql=sql, tables_used=tables, columns_used=[],
                           confidence=0.3, reasoning=reasoning, approach='fallback')

        except Exception as e:
            reasoning.append(f"Question-based generation failed: {e}")
            logger.debug(f"Question generation error: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='fallback')

    def _fallback_generation(self) -> SQLResult:
        reasoning = ['Fallback: generating executive summary']

        executive_sql = (
            "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims "
            "UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims "
            "UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' "
            "THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims "
            "UNION ALL SELECT 'Unique Members', COUNT(DISTINCT MEMBER_ID) FROM members "
            "UNION ALL SELECT 'Total Encounters', COUNT(*) FROM encounters"
        )

        return SQLResult(
            sql=executive_sql,
            tables_used=['claims', 'members', 'encounters'],
            columns_used=['PAID_AMOUNT', 'CLAIM_STATUS', 'MEMBER_ID'],
            confidence=0.40,
            reasoning=reasoning,
            approach='fallback_executive'
        )


    def _build_pmpm_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building PMPM metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            cost_col = self._resolve_column(tables[0], 'cost')
            member_col = self._resolve_column(tables[0], 'member')
            date_col = self._resolve_column(tables[0], 'date')

            if not all([cost_col, member_col, date_col]):
                reasoning.append("Could not resolve PMPM columns")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            question = understanding.get('question', '').lower()
            dimension_cols = understanding.get('dimension_columns', [])
            intent = understanding.get('intent', '')

            group_by_col = None
            if intent in ('comparison', 'breakdown') or ' vs ' in question or ' by ' in question:
                if 'plan' in question or 'hmo' in question or 'ppo' in question or 'medicare' in question:
                    group_by_col = f'{tables[0]}.PLAN_TYPE' if self._column_exists(tables[0], 'PLAN_TYPE') else None
                elif 'region' in question:
                    group_by_col = f'{tables[0]}.KP_REGION' if self._column_exists(tables[0], 'KP_REGION') else None
                elif dimension_cols:
                    dim_col = dimension_cols[0] if isinstance(dimension_cols[0], str) else dimension_cols[0]
                    if self._column_exists(tables[0], dim_col):
                        group_by_col = f'{tables[0]}.{dim_col}'

            ast = SQLQueryAST()
            select_cols = []
            if group_by_col:
                select_cols.append((group_by_col, group_by_col.split('.')[-1]))
            select_cols.append(
                (f'ROUND(SUM(CAST({cost_col} AS REAL)) / NULLIF(COUNT(DISTINCT {member_col} || \'-\' || strftime(\'%Y-%m\', {date_col})), 0), 2)',
                 'pmpm')
            )
            select_cols.append(('COUNT(*)', 'total_claims'))
            ast.select = SelectNode(columns=select_cols)
            ast.from_ = FromNode(table=tables[0])

            conditions = understanding.get('filter_conditions', [])
            if ' vs ' in question or ' versus ' in question:
                plan_values = []
                for plan in ['HMO', 'PPO', 'EPO', 'HDHP', 'Medicaid', 'Medicare Advantage']:
                    if plan.lower() in question:
                        plan_values.append(plan)
                if plan_values and len(plan_values) >= 2:
                    conditions = [c for c in conditions
                                  if 'plan_type' not in str(c.get('column', '')).lower()]
                    conditions.append({
                        'column': 'PLAN_TYPE', 'op': 'IN', 'value': plan_values
                    })

            if conditions:
                where = self._build_where_from_conditions(conditions, tables[0], available_tables=tables)
                if where:
                    ast.where = where

            if group_by_col:
                ast.group_by = GroupByNode(columns=[group_by_col])

            sql = ast.render()
            reasoning.append("Built PMPM query")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[cost_col, member_col, date_col],
                confidence=0.9,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['pmpm']
            )
        except Exception as e:
            reasoning.append(f"PMPM build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_readmission_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building readmission rate metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            table = tables[0]
            admit_col = self._resolve_column(table, 'date', prefer='admit')
            discharge_col = self._resolve_column(table, 'date', prefer='discharge')
            member_col = self._resolve_column(table, 'member')
            encounter_col = 'encounter_id'

            if not all([admit_col, discharge_col, member_col]):
                reasoning.append("Could not resolve readmission columns")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            admit_name = admit_col.split('.')[-1]
            discharge_name = discharge_col.split('.')[-1]
            member_name = member_col.split('.')[-1]

            sql = f"""
SELECT
    ROUND(100.0 * COUNT(CASE WHEN e2.{encounter_col} IS NOT NULL THEN 1 END) / COUNT(*), 2) as readmission_pct,
    COUNT(*) as total_admissions,
    COUNT(CASE WHEN e2.{encounter_col} IS NOT NULL THEN 1 END) as readmitted
FROM {table} e1
LEFT JOIN {table} e2 ON
    e1.{member_name} = e2.{member_name}
    AND julianday(e2.{admit_name}) - julianday(e1.{discharge_name}) BETWEEN 0 AND 30
    AND e1.{encounter_col} != e2.{encounter_col}
"""
            reasoning.append("Built 30-day readmission query with self-join")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[admit_col, discharge_col, member_col],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['readmission_rate']
            )
        except Exception as e:
            reasoning.append(f"Readmission build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_denial_rate_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building denial rate metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            table = tables[0]
            status_col = self._resolve_column(table, 'status')

            if not status_col:
                reasoning.append("Could not resolve status column")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            ast = SQLQueryAST()
            ast.select = SelectNode(columns=[
                ('ROUND(100.0 * SUM(CASE WHEN ' + status_col + " IN ('DENIED','REJECTED') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
                 'denial_rate_pct'),
                ('COUNT(*)', 'total_claims'),
                ("SUM(CASE WHEN " + status_col + " IN ('DENIED','REJECTED') THEN 1 ELSE 0 END)", 'denied_claims')
            ])
            ast.from_ = FromNode(table=table)

            conditions = understanding.get('filter_conditions', [])
            if conditions:
                ast.where = self._build_where_from_conditions(conditions, table)

            sql = ast.render()
            reasoning.append("Built denial rate query")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[status_col],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['denial_rate']
            )
        except Exception as e:
            reasoning.append(f"Denial rate build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_alos_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building ALOS metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            table = tables[0]

            has_los_col = self._column_exists(table, 'LENGTH_OF_STAY')

            ast = SQLQueryAST()
            if has_los_col:
                los_expr = f'{table}.LENGTH_OF_STAY'
                ast.select = SelectNode(columns=[
                    ('COUNT(*)', 'admission_count'),
                    (f'ROUND(AVG(CAST({los_expr} AS REAL)), 2)', 'avg_los'),
                    (f'MIN(CAST({los_expr} AS INTEGER))', 'min_los'),
                    (f'MAX(CAST({los_expr} AS INTEGER))', 'max_los')
                ])
                reasoning.append("Using direct LENGTH_OF_STAY column")
            else:
                admit_col = self._resolve_column(table, 'date', prefer='admit')
                discharge_col = self._resolve_column(table, 'date', prefer='discharge')
                if not all([admit_col, discharge_col]) or admit_col == discharge_col:
                    reasoning.append("Could not resolve distinct date columns for LOS")
                    return SQLResult(sql='', tables_used=[], columns_used=[],
                                   confidence=0.3, reasoning=reasoning, approach='template_direct')
                ast.select = SelectNode(columns=[
                    ('COUNT(*)', 'admission_count'),
                    (f'ROUND(AVG(CAST((julianday({discharge_col}) - julianday({admit_col})) AS REAL)), 2)', 'avg_los'),
                    (f'MIN(CAST((julianday({discharge_col}) - julianday({admit_col})) AS INTEGER))', 'min_los'),
                    (f'MAX(CAST((julianday({discharge_col}) - julianday({admit_col})) AS INTEGER))', 'max_los')
                ])
            ast.from_ = FromNode(table=table)

            conditions = understanding.get('filter_conditions', [])
            if conditions:
                ast.where = self._build_where_from_conditions(conditions, table)

            sql = ast.render()
            reasoning.append("Built ALOS query")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[admit_col, discharge_col],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['alos']
            )
        except Exception as e:
            reasoning.append(f"ALOS build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_yield_rate_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building yield rate metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            table = tables[0]
            paid_col = self._resolve_column(table, 'paid')
            billed_col = self._resolve_column(table, 'billed')

            if not all([paid_col, billed_col]):
                reasoning.append("Could not resolve paid/billed columns")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            ast = SQLQueryAST()
            ast.select = SelectNode(columns=[
                (f'ROUND(100.0 * SUM({paid_col}) / NULLIF(SUM({billed_col}), 0), 2)', 'yield_rate_pct'),
                (f'SUM({paid_col})', 'total_paid'),
                (f'SUM({billed_col})', 'total_billed')
            ])
            ast.from_ = FromNode(table=table)

            conditions = understanding.get('filter_conditions', [])
            if conditions:
                ast.where = self._build_where_from_conditions(conditions, table)

            sql = ast.render()
            reasoning.append("Built yield rate query")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[paid_col, billed_col],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['yield_rate']
            )
        except Exception as e:
            reasoning.append(f"Yield rate build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_er_per_1000_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building ER per 1000 metric']
        tables = understanding.get('target_tables', [])
        if not tables:
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

        try:
            table = tables[0]
            visit_type_col = self._resolve_column(table, 'visit_type')
            member_col = self._resolve_column(table, 'member')

            if not all([visit_type_col, member_col]):
                reasoning.append("Could not resolve ER columns")
                return SQLResult(sql='', tables_used=[], columns_used=[],
                               confidence=0.3, reasoning=reasoning, approach='template_direct')

            ast = SQLQueryAST()
            ast.select = SelectNode(columns=[
                (f"ROUND(1000.0 * COUNT(CASE WHEN {visit_type_col} = 'ER' THEN 1 END) / NULLIF(COUNT(DISTINCT {member_col}), 0), 2)",
                 'er_per_1000'),
                (f"COUNT(CASE WHEN {visit_type_col} = 'ER' THEN 1 END)", 'er_visits'),
                (f'COUNT(DISTINCT {member_col})', 'distinct_members')
            ])
            ast.from_ = FromNode(table=table)

            conditions = understanding.get('filter_conditions', [])
            if conditions:
                where = self._build_where_from_conditions(conditions, table, available_tables=[table])
                if where:
                    ast.where = where

            sql = ast.render()
            reasoning.append("Built ER per 1000 query")

            return SQLResult(
                sql=sql,
                tables_used=tables,
                columns_used=[visit_type_col, member_col],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['er_per_1000']
            )
        except Exception as e:
            reasoning.append(f"ER per 1000 build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')

    def _build_noshow_rate_sql(self, understanding: Dict) -> SQLResult:
        reasoning = ['Building no-show rate metric']

        try:
            table = 'appointments'
            dimensions = understanding.get('dimension_columns', [])

            select_cols = []
            group_cols = []
            for dim in dimensions:
                if self._column_exists(table, dim):
                    qualified = f'{table}.{dim}'
                    select_cols.append((qualified, dim))
                    group_cols.append(qualified)

            select_cols.extend([
                ('COUNT(*)', 'total_appointments'),
                (f"SUM(CASE WHEN {table}.STATUS = 'NO_SHOW' THEN 1 ELSE 0 END)", 'no_shows'),
                (f"ROUND(SUM(CASE WHEN {table}.STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)", 'noshow_rate_pct'),
            ])

            ast = SQLQueryAST()
            ast.select = SelectNode(columns=select_cols)
            ast.from_ = FromNode(table=table)

            if group_cols:
                ast.group_by = GroupByNode(columns=group_cols)
                ast.order_by = OrderByNode(columns=[
                    (f"ROUND(SUM(CASE WHEN {table}.STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)", 'DESC')
                ])

            conditions = understanding.get('filter_conditions', [])
            if conditions:
                where = self._build_where_from_conditions(conditions, table, available_tables=[table])
                if where:
                    ast.where = where

            sql = ast.render()
            reasoning.append("Built no-show rate query")

            return SQLResult(
                sql=sql,
                tables_used=[table],
                columns_used=[f'{table}.STATUS'],
                confidence=0.85,
                reasoning=reasoning,
                approach='template_direct',
                is_derived_metric=True,
                derived_metrics=['noshow_rate']
            )
        except Exception as e:
            reasoning.append(f"No-show rate build failed: {e}")
            return SQLResult(sql='', tables_used=[], columns_used=[],
                           confidence=0.0, reasoning=reasoning, approach='template_direct')


    def _build_ast_from_understanding(self, understanding: Dict) -> SQLQueryAST:
        ast = SQLQueryAST()

        intent = understanding.get('intent', 'aggregate')
        target_tables = understanding.get('target_tables', [])
        metric_columns = understanding.get('metric_columns', [])
        dimension_columns = understanding.get('dimension_columns', [])
        filter_conditions = understanding.get('filter_conditions', [])

        if not target_tables:
            return ast

        table = target_tables[0]
        used_tables = set(target_tables)

        def resolve_to_table(col_name: str) -> str:
            for t in target_tables:
                if self._column_exists(t, col_name):
                    return f'{t}.{col_name}'
            found = self._find_column_table(col_name)
            if found:
                ref_table = found.split('.')[0]
                used_tables.add(ref_table)
                return found
            return f'{table}.{col_name}'

        select_cols = []
        for dim in dimension_columns:
            qualified = resolve_to_table(dim)
            select_cols.append((qualified, dim))

        for metric_entry in metric_columns:
            if isinstance(metric_entry, (list, tuple)) and len(metric_entry) >= 2:
                metric_col, agg_func = metric_entry[0], metric_entry[1]
            elif isinstance(metric_entry, str):
                metric_col = metric_entry
                agg_func = 'SUM'
            else:
                continue
            qualified = resolve_to_table(metric_col)
            select_cols.append((f'{agg_func}(CAST({qualified} AS REAL))', f'{agg_func.lower()}_{metric_col.lower()}'))

        if not select_cols:
            select_cols = [('COUNT(*)', 'total')]

        ast.select = SelectNode(columns=select_cols, distinct=intent == 'distinct')

        ast.from_ = FromNode(table=table)

        joined_tables = {table}
        for join_table in list(used_tables):
            if join_table in joined_tables:
                continue
            join_condition = None
            for src_table in list(joined_tables):
                join_condition = self._find_join_condition(src_table, join_table)
                if join_condition:
                    break
            if join_condition:
                ast.joins.append(JoinNode(
                    join_type=JoinType.LEFT,
                    table=join_table,
                    alias=None,
                    on_condition=join_condition
                ))
                joined_tables.add(join_table)

        if filter_conditions:
            ast.where = self._build_where_from_conditions(
                filter_conditions, table, available_tables=list(joined_tables)
            )

        if dimension_columns and (metric_columns or len(dimension_columns) > 0):
            group_cols = [resolve_to_table(d) for d in dimension_columns]
            ast.group_by = GroupByNode(columns=group_cols)

        if metric_columns:
            metric_col, agg_func = metric_columns[0]
            qualified = resolve_to_table(metric_col)
            ast.order_by = OrderByNode(columns=[(f'{agg_func}(CAST({qualified} AS REAL))', 'DESC')])
        elif dimension_columns:
            order_cols = [(resolve_to_table(dimension_columns[0]), 'ASC')]
            ast.order_by = OrderByNode(columns=order_cols)

        ast.limit = LimitNode(count=1000)

        return ast

    def _build_where_from_conditions(self, conditions: List[Dict], table: str,
                                     available_tables: Optional[List[str]] = None) -> WhereNode:
        if available_tables is None:
            available_tables = [table]
        available_set = set(t.lower() for t in available_tables)

        where_conditions = []

        for condition in conditions:
            column = condition.get('column', '')
            op = condition.get('op', '=')
            value = condition.get('value', '')

            qualified_col = self._qualify_column(table, column)

            if '.' in qualified_col:
                ref_table = qualified_col.split('.')[0].lower()
                if ref_table not in available_set:
                    logger.debug(f"Skipping WHERE condition: {qualified_col} — table {ref_table} not in FROM/JOIN")
                    continue

            if not op or op == '=':
                op = condition.get('operator', op or '=')
            if not value and 'values' in condition:
                vals = condition['values']
                if isinstance(vals, list):
                    if len(vals) == 1:
                        value = vals[0]
                        op = '='
                    else:
                        value = vals
                        op = 'IN'

            if isinstance(value, list):
                formatted_vals = ', '.join(f"'{v}'" for v in value)
                value = f"({formatted_vals})"
                op = 'IN'
            elif isinstance(value, str) and not value.startswith(('(', "'")):
                value = f"'{value}'"
            elif isinstance(value, (int, float)):
                value = str(value)

            where_conditions.append((qualified_col, op, value))

        if not where_conditions:
            return None
        return WhereNode(conditions=where_conditions)


    def _qualify_column(self, table: str, column: str) -> str:
        if '.' in column:
            parts = column.split('.', 1)
            tbl, col = parts[0], parts[1]
            if self._column_exists(tbl, col):
                return column
            resolved = self._find_column_table(col)
            if resolved:
                return resolved
            return column

        if self._column_exists(table, column):
            return f'{table}.{column}'

        resolved = self._find_column_table(column)
        if resolved:
            return resolved

        return f'{table}.{column}'

    def _column_exists(self, table: str, column: str) -> bool:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return True
        columns = self.schema_learner.tables.get(table, [])
        return any(c.name.upper() == column.upper() for c in columns)

    def _find_column_table(self, column: str) -> Optional[str]:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return None
        for tbl_name, cols in self.schema_learner.tables.items():
            if any(c.name.upper() == column.upper() for c in cols):
                return f'{tbl_name}.{column}'
        return None

    def _resolve_column(self, table: str, semantic_type: str, prefer: str = '') -> Optional[str]:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return None

        columns = self.schema_learner.tables.get(table, [])
        if not columns:
            return None

        aliases = self._column_aliases.get(semantic_type, [])
        for alias in aliases:
            for col in columns:
                if col.name.lower() == alias.lower():
                    return f'{table}.{col.name}'

        for col in columns:
            if any(tag.lower() == semantic_type.lower() for tag in (col.semantic_tags or [])):
                return f'{table}.{col.name}'

        if semantic_type == 'date':
            for col in columns:
                if col.is_date:
                    if prefer and prefer.lower() in col.name.lower():
                        return f'{table}.{col.name}'
            for col in columns:
                if col.is_date:
                    return f'{table}.{col.name}'

        if semantic_type == 'cost':
            for col in columns:
                if col.is_numeric and ('cost' in col.name.lower() or 'paid' in col.name.lower()):
                    return f'{table}.{col.name}'

        return None

    def _find_join_condition(self, table1: str, table2: str) -> Optional[str]:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return None

        cols1 = {c.name for c in self.schema_learner.tables.get(table1, [])}
        cols2 = {c.name for c in self.schema_learner.tables.get(table2, [])}
        common = cols1 & cols2

        if 'MEMBER_ID' in common:
            return f'{table1}.MEMBER_ID = {table2}.MEMBER_ID'

        id_cols = sorted([c for c in common if c.upper().endswith('_ID')])
        if id_cols:
            return f'{table1}.{id_cols[0]} = {table2}.{id_cols[0]}'

        if 'providers' in (table1, table2):
            npi_map = {
                'claims': ('BILLING_NPI', 'NPI'),
                'encounters': ('SUPERVISING_NPI', 'NPI'),
                'prescriptions': ('PRESCRIBING_NPI', 'NPI'),
                'referrals': ('REFERRED_TO_NPI', 'NPI'),
                'diagnoses': ('DIAGNOSING_NPI', 'NPI'),
                'appointments': ('PROVIDER_NPI', 'NPI'),
                'members': ('PCP_NPI', 'NPI'),
            }
            other = table2 if table1 == 'providers' else table1
            if other in npi_map:
                other_col, prov_col = npi_map[other]
                if table1 == 'providers':
                    return f'{table1}.{prov_col} = {table2}.{other_col}'
                else:
                    return f'{table1}.{other_col} = {table2}.{prov_col}'

        if 'CPT_CODE' in common:
            return f'{table1}.CPT_CODE = {table2}.CPT_CODE'

        if hasattr(self.schema_learner, 'join_graph'):
            join_graph = self.schema_learner.join_graph
            if table1 in join_graph and table2 in join_graph[table1]:
                condition = join_graph[table1][table2]
                if '=' in condition:
                    return condition
                if condition.upper() not in ('PLAN_TYPE', 'KP_REGION', 'FACILITY', 'COPAY',
                                              'ICD10_DESCRIPTION', 'RENDERING_NPI', 'MRN'):
                    return f'{table1}.{condition} = {table2}.{condition}'

        return None

    def _identify_tables_from_keywords(self, keywords: List[str]) -> List[str]:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return []

        schema_tables = set(self.schema_learner.tables.keys())
        found_tables = []

        for keyword in keywords:
            for schema_table in schema_tables:
                if keyword.lower() in schema_table.lower() or schema_table.lower() in keyword.lower():
                    if schema_table not in found_tables:
                        found_tables.append(schema_table)
                    break

        return found_tables[:1] if found_tables else list(schema_tables)[:1] if schema_tables else []

    def _find_metric_column(self, table: str) -> Optional[str]:
        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return None

        columns = self.schema_learner.tables.get(table, [])
        for col in columns:
            if col.is_numeric and ('cost' in col.name.lower() or 'amount' in col.name.lower()):
                return f'{table}.{col.name}'

        for col in columns:
            if col.is_numeric:
                return f'{table}.{col.name}'

        return None

    def _is_derived_metric_intent(self, understanding: Dict, metric_name: str) -> bool:
        intent = understanding.get('intent', '').lower()
        entities = understanding.get('entities', [])
        entity_texts = [e.get('text', '').lower() for e in entities]
        question = understanding.get('question', '').lower()
        sql_approach = understanding.get('sql_approach', '').lower()

        metric_keywords = {
            'pmpm': ['pmpm', 'per member', 'per capita'],
            'readmission_rate': ['readmission', 'returns', '30-day'],
            'denial_rate': ['denial', 'denied', 'rejection'],
            'alos': ['length of stay', 'los', 'alos'],
            'yield_rate': ['yield', 'paid/billed', 'payment rate'],
            'er_per_1000': ['er rate', 'emergency', 'per 1000'],
            'noshow_rate': ['no-show', 'noshow', 'no show', 'missed appointment'],
        }

        keywords = metric_keywords.get(metric_name, [])
        search_texts = [intent, question, sql_approach] + entity_texts
        for keyword in keywords:
            if any(keyword in text for text in search_texts):
                return True

        return False


    def _validate_against_schema(self, sql: str) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        if not self.schema_learner or not hasattr(self.schema_learner, 'tables'):
            return errors, warnings

        schema_tables = set(self.schema_learner.tables.keys())
        tables = self._extract_tables_from_sql(sql)

        for table in tables:
            if table not in schema_tables:
                errors.append(f"Table '{table}' not in schema")

        return errors, warnings

    def _check_semantic_issues(self, sql: str) -> List[str]:
        issues = []

        upper_sql = sql.upper()

        has_agg = any(agg in upper_sql for agg in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN('])
        has_group = 'GROUP BY' in upper_sql
        has_select_star = 'SELECT *' in upper_sql or 'SELECT ' in upper_sql and '*' in upper_sql.split('FROM')[0]

        if has_agg and not has_group and not 'WHERE' in upper_sql:
            issues.append("Aggregation without GROUP BY (likely intended)")

        return issues


    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        tables = []
        upper_sql = sql.upper()
        keywords = ['FROM', 'JOIN']

        for keyword in keywords:
            pattern = rf'{keyword}\s+(\w+)'
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        columns = []

        match = re.search(r'SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM', sql, re.IGNORECASE)
        if match:
            select_part = match.group(1)
            cols = re.findall(r'(\w+\.\w+|\w+(?=\s+(?:AS|,)))', select_part)
            columns.extend(cols)

        match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|;|$)', sql, re.IGNORECASE)
        if match:
            where_part = match.group(1)
            cols = re.findall(r'(\w+\.\w+|\w+)\s*[=<>]', where_part)
            columns.extend(cols)

        return list(set(columns))

    def _extract_columns_from_understanding(self, understanding: Dict) -> List[str]:
        columns = []

        columns.extend(understanding.get('target_columns', []))

        for col, _ in understanding.get('metric_columns', []):
            columns.append(col)

        columns.extend(understanding.get('dimension_columns', []))

        for condition in understanding.get('filter_conditions', []):
            columns.append(condition.get('column', ''))

        return [c for c in columns if c]

    def _extract_keywords(self, question: str) -> List[str]:
        words = question.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'by', 'for', 'is', 'are', 'was', 'were', 'be', 'been'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _infer_healthcare_intent(self, sql: str) -> Optional[str]:
        upper_sql = sql.upper()

        intents = {
            'readmission_rate': ['READMISSION', 'READMIT', 'JULIAN'],
            'denial_rate': ['DENIAL', 'DENIED', 'REJECTED'],
            'pmpm': ['PMPM', 'PER MEMBER PER MONTH'],
            'alos': ['LENGTH', 'STAY', 'JULIAN', 'DISCHARGE', 'ADMIT'],
            'yield_rate': ['YIELD', 'PAID', 'BILLED'],
            'er_per_1000': ['ER', 'EMERGENCY', 'PER 1000'],
            'risk_stratification': ['RISK', 'STRATIF'],
            'cost_breakdown': ['COST', 'BREAKDOWN'],
        }

        for intent, keywords in intents.items():
            if any(keyword in upper_sql for keyword in keywords):
                return intent

        return None


    def _low_confidence_result(self, reason: str, approach: str) -> SQLResult:
        return SQLResult(
            sql='',
            tables_used=[],
            columns_used=[],
            confidence=0.0,
            reasoning=[reason],
            approach=approach
        )
