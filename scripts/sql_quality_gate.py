import re
import sqlite3
import logging
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

logger = logging.getLogger('gpdm.sql_quality_gate')


@dataclass
class ValidationResult:
    passed: bool
    corrected_sql: str
    violations: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    blocked: bool = False


@dataclass
class TableProfile:
    name: str
    row_count: int
    key_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]


TABLE_ROW_COUNTS = {
    'claims': 60000,
    'members': 25000,
    'encounters': 50000,
    'diagnoses': 20000,
    'prescriptions': 12000,
    'appointments': 10000,
    'referrals': 5000,
    'providers': 3000,
    'cpt_codes': 81,
}

REFERENCE_VALUES = {
    ('claims', 'PAID_AMOUNT', 'MAX'): 210600.41,
    ('claims', 'PAID_AMOUNT', 'SUM'): 133923772.42,
    ('claims', 'PAID_AMOUNT', 'AVG'): 2232.06,
    ('members', 'RISK_SCORE', 'AVG'): 0.84,
    ('members', 'RISK_SCORE', 'MIN'): 0.10,
    ('members', 'RISK_SCORE', 'MAX'): 5.0,
    ('prescriptions', 'DAYS_SUPPLY', 'AVG'): 40.18,
    ('prescriptions', 'COST', 'SUM'): 402433.0,
    ('encounters', 'LENGTH_OF_STAY', 'AVG'): 3.86,
    ('cpt_codes', 'RVU', 'AVG'): 13.1,
}

AVG_ROWS_PER_MEMBER = {
    'claims': 2.4,
    'encounters': 2.3,
    'prescriptions': 1.3,
    'diagnoses': 1.5,
    'referrals': 1.1,
    'appointments': 1.2,
}

TABLE_EXCLUSIVE_COLUMNS = {
    'claims': {'CLAIM_ID', 'CLAIM_STATUS', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT',
               'PAID_AMOUNT', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE', 'ICD10_CODE',
               'ICD10_DESCRIPTION', 'CPT_CODE', 'PLAN_TYPE', 'KP_REGION',
               'BILLING_NPI', 'RENDERING_NPI'},
    'members': {'RISK_SCORE', 'AGE', 'GENDER', 'RACE', 'LANGUAGE',
                'CHRONIC_CONDITIONS', 'CITY', 'STATE', 'ZIP_CODE'},
    'encounters': {'ENCOUNTER_TYPE', 'DEPARTMENT', 'LENGTH_OF_STAY',
                   'CHIEF_COMPLAINT', 'DISPOSITION', 'ACUITY_LEVEL'},
    'prescriptions': {'DRUG_NAME', 'DRUG_CLASS', 'DAYS_SUPPLY', 'REFILLS',
                      'COST', 'PRESCRIBING_NPI'},
    'appointments': {'APPOINTMENT_TYPE', 'DURATION_MINUTES', 'STATUS',
                     'DEPARTMENT'},
    'referrals': {'REFERRAL_TYPE', 'URGENCY', 'STATUS',
                  'REFERRING_NPI', 'REFERRED_TO_NPI'},
    'diagnoses': {'ICD10_CODE', 'SEVERITY', 'HCC_FLAG', 'CHRONIC_FLAG'},
    'providers': {'SPECIALTY', 'PANEL_SIZE', 'DEGREE', 'NPI'},
    'cpt_codes': {'CPT_CODE', 'DESCRIPTION', 'CATEGORY', 'RVU', 'BASE_COST'},
}

QUESTION_TABLE_SEMANTICS = {
    'referral': 'referrals', 'referrals': 'referrals',
    'encounter': 'encounters', 'encounters': 'encounters', 'visit': 'encounters',
    'prescription': 'prescriptions', 'prescriptions': 'prescriptions',
    'medication': 'prescriptions', 'drug': 'prescriptions',
    'appointment': 'appointments', 'appointments': 'appointments',
    'member': 'members', 'members': 'members', 'patient': 'members',
    'provider': 'providers', 'providers': 'providers', 'doctor': 'providers',
    'diagnosis': 'diagnoses', 'diagnoses': 'diagnoses',
    'claim': 'claims', 'claims': 'claims',
    'rvu': 'cpt_codes', 'cpt': 'cpt_codes', 'procedure': 'cpt_codes',
    'risk score': 'members', 'risk_score': 'members',
    'days supply': 'prescriptions', 'days_supply': 'prescriptions',
    'length of stay': 'encounters', 'los': 'encounters',
    'copay': 'claims', 'coinsurance': 'claims', 'deductible': 'claims',
    'panel size': 'providers', 'panel_size': 'providers',
    'specialty': 'providers',
    'billed amount': 'claims', 'billed': 'claims',
    'paid amount': 'claims', 'paid': 'claims',
    'allowed amount': 'claims',
    'cost': 'prescriptions',
    'denial': 'claims', 'denied': 'claims',
}

AGG_INTENT_MAP = {
    'maximum': 'MAX', 'max': 'MAX', 'highest': 'MAX', 'largest': 'MAX',
    'biggest': 'MAX', 'most expensive': 'MAX', 'peak': 'MAX', 'top': 'MAX',
    'minimum': 'MIN', 'min': 'MIN', 'lowest': 'MIN', 'smallest': 'MIN',
    'cheapest': 'MIN', 'least': 'MIN', 'bottom': 'MIN',
    'average': 'AVG', 'avg': 'AVG', 'mean': 'AVG', 'typical': 'AVG',
    'total': 'SUM', 'sum': 'SUM', 'overall': 'SUM', 'combined': 'SUM',
    'count': 'COUNT', 'how many': 'COUNT', 'number of': 'COUNT',
    'total number': 'COUNT',
}

GROUP_BY_SEMANTICS = {
    'status': {
        'claims': 'CLAIM_STATUS',
        'referrals': 'STATUS',
        'appointments': 'STATUS',
        'encounters': 'ENCOUNTER_STATUS',
    },
    'region': {
        'claims': 'KP_REGION',
        'encounters': 'KP_REGION',
    },
    'plan type': {
        'claims': 'PLAN_TYPE',
    },
    'plan': {
        'claims': 'PLAN_TYPE',
    },
    'department': {
        'encounters': 'DEPARTMENT',
        'appointments': 'DEPARTMENT',
    },
    'gender': {
        'members': 'GENDER',
    },
    'drug class': {
        'prescriptions': 'DRUG_CLASS',
    },
    'specialty': {
        'providers': 'SPECIALTY',
    },
    'category': {
        'cpt_codes': 'CATEGORY',
    },
}


class SQLQualityGate:

    def __init__(self, db_path: str, schema_learner=None):
        self.db_path = db_path
        self.schema_learner = schema_learner
        self._correction_log = []
        self._learned_corrections = {}
        self._table_columns = {}

        self._load_schema()

        self._load_learned_corrections()

        logger.info("SQLQualityGate initialized with %d table profiles, %d learned corrections",
                     len(self._table_columns), len(self._learned_corrections))

    def _load_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cur.fetchall():
                if table_name.startswith('sqlite_') or table_name.startswith('gpdm_'):
                    continue
                cur.execute(f"PRAGMA table_info({table_name})")
                cols = set()
                for row in cur.fetchall():
                    cols.add(row[1].upper())
                self._table_columns[table_name.lower()] = cols
            conn.close()
        except Exception as e:
            logger.warning("SQLQualityGate schema load error: %s", e)

    def _load_learned_corrections(self):
        store_path = os.path.join(os.path.dirname(self.db_path), 'quality_gate_corrections.json')
        try:
            if os.path.exists(store_path):
                with open(store_path, 'r') as f:
                    self._learned_corrections = json.load(f)
        except Exception:
            self._learned_corrections = {}

    def _save_learned_correction(self, question_hash: str, correction: Dict):
        self._learned_corrections[question_hash] = correction
        store_path = os.path.join(os.path.dirname(self.db_path), 'quality_gate_corrections.json')
        try:
            with open(store_path, 'w') as f:
                json.dump(self._learned_corrections, f, indent=2)
        except Exception as e:
            logger.debug("Could not save correction: %s", e)


    def validate(self, question: str, sql: str,
                 understanding: Optional[Dict] = None,
                 execution_result: Optional[Tuple] = None) -> ValidationResult:
        if not sql or sql.startswith('--'):
            return ValidationResult(passed=True, corrected_sql=sql)

        q_lower = question.lower().strip()
        q_words = set(re.findall(r'\b\w+\b', q_lower))
        sql_upper = sql.upper()
        violations = []
        corrections = []
        corrected_sql = sql
        confidence_adj = 0.0

        alignment = self._check_question_sql_alignment(q_lower, q_words, sql_upper, sql)
        if alignment and alignment.get('corrected_sql'):
            return ValidationResult(
                passed=False,
                corrected_sql=alignment['corrected_sql'],
                violations=alignment.get('violations', []),
                corrections=alignment.get('corrections', []),
                confidence_adjustment=-0.3,
            )

        table_result = self._check_table_correctness(q_lower, q_words, sql_upper, sql, understanding)
        if table_result:
            violations.extend(table_result['violations'])
            if table_result.get('corrected_sql'):
                corrected_sql = table_result['corrected_sql']
                corrections.extend(table_result.get('corrections', []))
                confidence_adj -= 0.2

        agg_result = self._check_aggregation_correctness(q_lower, q_words, sql_upper, corrected_sql, understanding)
        if agg_result:
            violations.extend(agg_result['violations'])
            if agg_result.get('corrected_sql'):
                corrected_sql = agg_result['corrected_sql']
                corrections.extend(agg_result.get('corrections', []))
                confidence_adj -= 0.15

        join_result = self._check_join_integrity(q_lower, q_words, sql_upper, corrected_sql, understanding)
        if join_result:
            violations.extend(join_result['violations'])
            if join_result.get('corrected_sql'):
                corrected_sql = join_result['corrected_sql']
                corrections.extend(join_result.get('corrections', []))
                confidence_adj -= 0.25

        where_result = self._check_where_validity(q_lower, q_words, sql_upper, corrected_sql, understanding)
        if where_result:
            violations.extend(where_result['violations'])
            if where_result.get('corrected_sql'):
                corrected_sql = where_result['corrected_sql']
                corrections.extend(where_result.get('corrections', []))
                confidence_adj -= 0.15

        groupby_result = self._check_group_by_intent(q_lower, q_words, sql_upper, corrected_sql, understanding)
        if groupby_result:
            violations.extend(groupby_result['violations'])
            if groupby_result.get('corrected_sql'):
                corrected_sql = groupby_result['corrected_sql']
                corrections.extend(groupby_result.get('corrections', []))
                confidence_adj -= 0.20

        if execution_result:
            sanity_result = self._check_data_sanity(q_lower, sql_upper, corrected_sql, execution_result)
            if sanity_result:
                violations.extend(sanity_result['violations'])
                if sanity_result.get('corrected_sql'):
                    corrected_sql = sanity_result['corrected_sql']
                    corrections.extend(sanity_result.get('corrections', []))
                    confidence_adj -= 0.30

        if corrections:
            import hashlib
            q_hash = hashlib.md5(q_lower.encode()).hexdigest()[:12]
            self._save_learned_correction(q_hash, {
                'question': q_lower,
                'original_sql': sql,
                'corrected_sql': corrected_sql,
                'violations': violations,
                'corrections': corrections,
            })

        passed = len(violations) == 0
        return ValidationResult(
            passed=passed,
            corrected_sql=corrected_sql,
            violations=violations,
            corrections=corrections,
            confidence_adjustment=confidence_adj,
        )

    def validate_post_execution(self, question: str, sql: str,
                                rows: List, columns: List,
                                understanding: Optional[Dict] = None) -> ValidationResult:
        return self.validate(
            question, sql, understanding,
            execution_result=(rows, columns, None)
        )


    def _check_question_sql_alignment(self, q_lower: str, q_words: Set[str],
                                       sql_upper: str, sql: str) -> Optional[Dict]:
        sql_tables = self._extract_tables_from_sql(sql_upper)

        most_common_match = re.search(r'most\s+common\s+(\w+)', q_lower)
        if most_common_match:
            entity = most_common_match.group(1).lower()
            mc_map = {
                'diagnos': ('diagnoses', 'ICD10_DESCRIPTION'),
                'diagnosis': ('diagnoses', 'ICD10_DESCRIPTION'),
                'encounter': ('encounters', 'VISIT_TYPE'),
                'procedure': ('cpt_codes', 'DESCRIPTION'),
                'drug': ('prescriptions', 'DRUG_NAME'),
                'medication': ('prescriptions', 'DRUG_NAME'),
                'complaint': ('encounters', 'CHIEF_COMPLAINT'),
                'specialty': ('providers', 'SPECIALTY'),
            }
            for key, (table, col) in mc_map.items():
                if entity.startswith(key):
                    expected_sql = (
                        f"SELECT {col}, COUNT(*) as cnt FROM {table} "
                        f"GROUP BY {col} ORDER BY cnt DESC LIMIT 10"
                    )
                    if any(t.startswith('gpdm_') for t in sql_tables) or table not in sql_tables:
                        return {
                            'violations': [f"ALIGNMENT: 'most common {entity}' should query {table}.{col}"],
                            'corrections': [f"Replaced with GROUP BY {col} from {table}"],
                            'corrected_sql': expected_sql,
                        }
                    break

        by_match = re.search(r'(?:average|avg|mean|total|sum|count)\s+(.+?)\s+by\s+(\w+(?:\s+\w+)?)', q_lower)
        if by_match:
            metric_phrase = by_match.group(1).strip()
            group_phrase = by_match.group(2).strip()
            needs_rebuild = False
            if 'GROUP BY' not in sql_upper:
                needs_rebuild = True
            else:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute(sql)
                    test_rows = cur.fetchall()
                    conn.close()
                    if not test_rows:
                        needs_rebuild = True
                    elif len(test_rows) <= 1:
                        needs_rebuild = True
                    elif test_rows[0][0] is None or str(test_rows[0][0]).strip() == '':
                        needs_rebuild = True
                except Exception:
                    needs_rebuild = True

            if needs_rebuild:
                corrected = self._build_group_by_query(metric_phrase, group_phrase, q_lower, q_words)
                if corrected:
                    return {
                        'violations': [f"ALIGNMENT: '{q_lower}' GROUP BY query is invalid or missing"],
                        'corrections': [f"Built GROUP BY query for '{metric_phrase}' by '{group_phrase}'"],
                        'corrected_sql': corrected,
                    }

        pct_match = re.search(r'(?:percentage|percent|rate|proportion|ratio)\s+(?:of\s+)?(\w+)', q_lower)
        if pct_match:
            pct_entity = pct_match.group(1).lower()
            status_map = {
                'denied': ('claims', 'CLAIM_STATUS', 'DENIED'),
                'denial': ('claims', 'CLAIM_STATUS', 'DENIED'),
                'cancelled': ('appointments', 'STATUS', 'CANCELLED'),
                'approved': ('referrals', 'STATUS', 'APPROVED'),
                'pending': ('claims', 'CLAIM_STATUS', 'PENDING'),
                'no': ('appointments', 'STATUS', 'NO_SHOW'),
            }
            for key, (table, col, val) in status_map.items():
                if key in q_lower:
                    rate_by_match = re.search(r'(?:rate|percentage)\s+by\s+(\w+(?:\s+\w+)?)', q_lower)
                    if rate_by_match:
                        group_phrase = rate_by_match.group(1).strip()
                        group_col_map_local = {
                            'plan type': 'PLAN_TYPE', 'plan': 'PLAN_TYPE',
                            'region': 'KP_REGION', 'department': 'DEPARTMENT',
                            'type': 'CLAIM_TYPE', 'gender': 'GENDER',
                            'race': 'RACE', 'specialty': 'SPECIALTY',
                            'facility': 'FACILITY', 'status': 'CLAIM_STATUS',
                        }
                        group_col = group_col_map_local.get(group_phrase)
                        if group_col:
                            CROSS_TABLE_COLS = {
                                'RACE': 'members', 'GENDER': 'members', 'KP_REGION': 'members',
                                'SPECIALTY': 'providers', 'DEPARTMENT': 'encounters',
                            }
                            group_table = CROSS_TABLE_COLS.get(group_col, table)
                            if group_table != table and group_table == 'members':
                                corrected = (
                                    f"SELECT m.{group_col}, ROUND(100.0 * SUM(CASE WHEN {table}.{col} = '{val}' THEN 1 ELSE 0 END) "
                                    f"/ COUNT(*), 2) AS pct, COUNT(*) as total FROM {table} "
                                    f"JOIN members m ON {table}.MEMBER_ID = m.MEMBER_ID "
                                    f"GROUP BY m.{group_col} ORDER BY pct DESC"
                                )
                            elif group_table != table and group_table == 'providers':
                                corrected = (
                                    f"SELECT p.{group_col}, ROUND(100.0 * SUM(CASE WHEN {table}.{col} = '{val}' THEN 1 ELSE 0 END) "
                                    f"/ COUNT(*), 2) AS pct, COUNT(*) as total FROM {table} "
                                    f"JOIN providers p ON {table}.RENDERING_NPI = p.NPI "
                                    f"GROUP BY p.{group_col} ORDER BY pct DESC"
                                )
                            else:
                                corrected = (
                                    f"SELECT {group_col}, ROUND(100.0 * SUM(CASE WHEN {col} = '{val}' THEN 1 ELSE 0 END) "
                                    f"/ COUNT(*), 2) AS pct, COUNT(*) as total FROM {table} "
                                    f"GROUP BY {group_col} ORDER BY pct DESC"
                                )
                            return {
                                'violations': [f"ALIGNMENT: rate by {group_phrase} needs GROUP BY"],
                                'corrections': [f"Built GROUP BY {group_col} percentage query"],
                                'corrected_sql': corrected,
                            }

                    corrected = (
                        f"SELECT ROUND(100.0 * SUM(CASE WHEN {col} = '{val}' THEN 1 ELSE 0 END) "
                        f"/ COUNT(*), 2) AS pct FROM {table}"
                    )
                    needs_override = 'CASE' not in sql_upper
                    if not needs_override:
                        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_upper, re.DOTALL)
                        if select_match:
                            first_col = select_match.group(1).split(',')[0].strip()
                            if 'CASE' not in first_col and 'ROUND' not in first_col:
                                needs_override = True
                    if needs_override:
                        return {
                            'violations': [f"ALIGNMENT: percentage/rate query needs pct as first column"],
                            'corrections': [f"Built clean percentage query with CASE WHEN {col}='{val}'"],
                            'corrected_sql': corrected,
                        }
                    break

        col_ref_pattern = re.findall(r'(\w+)\.(\w+)', sql_upper)
        for alias_or_table, col_name in col_ref_pattern:
            if col_name in ('REAL', 'INTEGER', 'TEXT', 'AS', 'DESC', 'ASC'):
                continue
            resolved_table = None
            if alias_or_table.lower() in self._table_columns:
                resolved_table = alias_or_table.lower()
            else:
                alias_pattern = re.search(
                    rf'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?{re.escape(alias_or_table)}\b',
                    sql_upper
                )
                if alias_pattern:
                    resolved_table = alias_pattern.group(1).lower()

            if resolved_table and resolved_table in self._table_columns:
                if col_name not in self._table_columns[resolved_table]:
                    return self._fix_missing_column(q_lower, q_words, sql, resolved_table, col_name)

        avg_per_match = re.search(r'(?:average|avg|mean)\s+(.+?)\s+per\s+(\w+)', q_lower)
        if avg_per_match:
            metric_phrase = avg_per_match.group(1).strip()
            per_entity = avg_per_match.group(2).strip()
            if 'GROUP BY' in sql_upper and 'FIRST_NAME' in sql_upper:
                col_map = {
                    'panel size': ('providers', 'PANEL_SIZE'),
                    'risk score': ('members', 'RISK_SCORE'),
                    'days supply': ('prescriptions', 'DAYS_SUPPLY'),
                    'length of stay': ('encounters', 'LENGTH_OF_STAY'),
                    'paid amount': ('claims', 'PAID_AMOUNT'),
                    'billed amount': ('claims', 'BILLED_AMOUNT'),
                    'copay': ('claims', 'COPAY'),
                    'cost': ('prescriptions', 'COST'),
                }
                for phrase, (table, col) in col_map.items():
                    if phrase in metric_phrase:
                        corrected = f"SELECT ROUND(AVG(CAST({col} AS REAL)), 2) AS avg_{col.lower()} FROM {table}"
                        return {
                            'violations': [f"ALIGNMENT: 'average {metric_phrase} per {per_entity}' should be simple AVG"],
                            'corrections': [f"Replaced GROUP BY query with AVG({col}) FROM {table}"],
                            'corrected_sql': corrected,
                        }

        if any(kw in q_lower for kw in ['highest cost', 'most expensive', 'highest paid']):
            if 'CLAIM_ID' in sql_upper and 'MAX(' not in sql_upper:
                corrected = "SELECT MAX(CAST(PAID_AMOUNT AS REAL)) AS max_paid_amount FROM claims"
                return {
                    'violations': ["ALIGNMENT: 'highest cost' should return MAX value, not row details"],
                    'corrections': ["Replaced detail query with MAX(PAID_AMOUNT)"],
                    'corrected_sql': corrected,
                }

        if 'cost per encounter' in q_lower or 'cost of encounter' in q_lower:
            if 'COST' in sql_upper and 'encounters' in sql_tables:
                corrected = (
                    "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) / "
                    "COUNT(DISTINCT ENCOUNTER_ID), 2) AS avg_cost_per_encounter FROM claims "
                    "WHERE ENCOUNTER_ID IS NOT NULL"
                )
                return {
                    'violations': ["ALIGNMENT: encounters has no COST column; derive from claims"],
                    'corrections': ["Computed cost per encounter from claims.PAID_AMOUNT / ENCOUNTER_ID"],
                    'corrected_sql': corrected,
                }

        COLUMN_ALIGNMENT = {
            'paid amount': 'PAID_AMOUNT',
            'billed amount': 'BILLED_AMOUNT',
            'allowed amount': 'ALLOWED_AMOUNT',
            'risk score': 'RISK_SCORE',
            'length of stay': 'LENGTH_OF_STAY',
            'panel size': 'PANEL_SIZE',
            'days supply': 'DAYS_SUPPLY',
        }
        for phrase, expected_col in COLUMN_ALIGNMENT.items():
            if phrase in q_lower and expected_col not in sql_upper:
                table_map = {
                    'PAID_AMOUNT': 'claims', 'BILLED_AMOUNT': 'claims',
                    'ALLOWED_AMOUNT': 'claims', 'RISK_SCORE': 'members',
                    'LENGTH_OF_STAY': 'encounters', 'PANEL_SIZE': 'providers',
                    'DAYS_SUPPLY': 'prescriptions',
                }
                table = table_map.get(expected_col, 'claims')
                agg = 'AVG'
                if any(kw in q_words for kw in ['total', 'sum']):
                    agg = 'SUM'
                elif any(kw in q_words for kw in ['maximum', 'highest', 'max']):
                    agg = 'MAX'
                elif any(kw in q_words for kw in ['minimum', 'lowest', 'min']):
                    agg = 'MIN'
                corrected = f"SELECT ROUND({agg}(CAST({expected_col} AS REAL)), 2) AS {agg.lower()}_{expected_col.lower()} FROM {table}"
                return {
                    'violations': [f"COLUMN_MISMATCH: Question says '{phrase}' but SQL doesn't use {expected_col}"],
                    'corrections': [f"Replaced with {agg}({expected_col}) FROM {table}"],
                    'corrected_sql': corrected,
                }

        FILTER_ALIGNMENT = {
            'denied': ('claims', 'CLAIM_STATUS', 'DENIED'),
            'cancelled': ('appointments', 'STATUS', 'CANCELLED'),
            'pending': ('claims', 'CLAIM_STATUS', 'PENDING'),
            'approved': ('referrals', 'STATUS', 'APPROVED'),
            'paid': ('claims', 'CLAIM_STATUS', 'PAID'),
        }
        for keyword, (table, col, val) in FILTER_ALIGNMENT.items():
            if keyword in q_words:
                if any(kw in q_lower for kw in ['percentage', 'percent', 'rate', 'ratio']):
                    continue
                if f"'{val}'" not in sql_upper and f"= '{val}'" not in sql_upper:
                    is_count = any(kw in q_lower for kw in ['how many', 'count', 'number of'])
                    if is_count:
                        corrected = f"SELECT COUNT(*) AS total FROM {table} WHERE {col} = '{val}'"
                        return {
                            'violations': [f"FILTER_MISSING: Question mentions '{keyword}' but SQL has no WHERE {col}='{val}'"],
                            'corrections': [f"Added WHERE {col} = '{val}' filter"],
                            'corrected_sql': corrected,
                        }

        DEMOGRAPHIC_FILTERS = {
            r'\bfemale\b': ('members', 'GENDER', 'F'),
            r'\bmale\b(?!\s+(?:and|or)\s)': ('members', 'GENDER', 'M'),
            r'\bactive\s+providers?\b': ('providers', 'STATUS', 'ACTIVE'),
            r'\binactive\s+providers?\b': ('providers', 'STATUS', 'INACTIVE'),
        }
        for pattern_re, (table, col, val) in DEMOGRAPHIC_FILTERS.items():
            if re.search(pattern_re, q_lower):
                if f"'{val}'" not in sql_upper:
                    is_count = any(kw in q_lower for kw in ['how many', 'count', 'number of', 'total'])
                    if is_count:
                        corrected = f"SELECT COUNT(*) AS total FROM {table} WHERE {col} = '{val}'"
                        return {
                            'violations': [f"DEMOGRAPHIC_FILTER: Question implies {col}='{val}' but SQL has no filter"],
                            'corrections': [f"Added WHERE {col} = '{val}'"],
                            'corrected_sql': corrected,
                        }
                    else:
                        if 'WHERE' in sql_upper:
                            corrected = re.sub(r'WHERE\s+', f"WHERE {col} = '{val}' AND ", sql, count=1, flags=re.IGNORECASE)
                        else:
                            for kw in ['GROUP BY', 'ORDER BY', 'LIMIT']:
                                if kw in sql_upper:
                                    corrected = re.sub(kw, f"WHERE {col} = '{val}' {kw}", sql, count=1, flags=re.IGNORECASE)
                                    break
                            else:
                                corrected = sql.rstrip(';') + f" WHERE {col} = '{val}'"
                        return {
                            'violations': [f"DEMOGRAPHIC_FILTER: Missing {col}='{val}' filter"],
                            'corrections': [f"Added WHERE {col} = '{val}'"],
                            'corrected_sql': corrected,
                        }

        DISTRIBUTION_PATTERNS = {
            r'claim\s*type[s]?\s+distribution': ('claims', 'CLAIM_TYPE'),
            r'claim[s]?\s+status\s+distribution': ('claims', 'CLAIM_STATUS'),
            r'claim[s]?\s+status\s+(?:distribution\s+)?(?:across|by)\s+(?:all\s+)?region': None,
            r'facility\s+utilization': ('encounters', 'VISIT_TYPE'),
            r'encounter\s+type[s]?\s+distribution': ('encounters', 'VISIT_TYPE'),
        }
        for pat_re, mapping in DISTRIBUTION_PATTERNS.items():
            if re.search(pat_re, q_lower):
                if 'GROUP BY' not in sql_upper:
                    if mapping is None:
                        corrected = (
                            "SELECT m.KP_REGION, c.CLAIM_STATUS, COUNT(*) as cnt "
                            "FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                            "GROUP BY m.KP_REGION, c.CLAIM_STATUS "
                            "ORDER BY m.KP_REGION, cnt DESC"
                        )
                    else:
                        table, col = mapping
                        corrected = (
                            f"SELECT {col}, COUNT(*) as cnt FROM {table} "
                            f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY cnt DESC"
                        )
                    return {
                        'violations': [f"DISTRIBUTION: '{pat_re}' requires GROUP BY but SQL has none"],
                        'corrections': [f"Built GROUP BY query for distribution"],
                        'corrected_sql': corrected,
                    }

        PER_METRICS = [
            (r'(?:average|avg)\s+encounters?\s+per\s+member',
             "SELECT ROUND(CAST(COUNT(*) AS REAL) / COUNT(DISTINCT MEMBER_ID), 2) AS avg_encounters_per_member FROM encounters"),
            (r'claims?\s+per\s+(?:rendering\s+)?provider',
             "SELECT p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME AS provider_name, COUNT(c.CLAIM_ID) as claim_count FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY p.NPI, provider_name ORDER BY claim_count DESC LIMIT 20"),
            (r'(?:average|avg)\s+claims?\s+per\s+member',
             "SELECT ROUND(CAST(COUNT(*) AS REAL) / COUNT(DISTINCT MEMBER_ID), 2) AS avg_claims_per_member FROM claims"),
        ]
        for pat_re, corrected_sql in PER_METRICS:
            if re.search(pat_re, q_lower):
                if 'DISTINCT' not in sql_upper or 'COUNT' not in sql_upper or ('GROUP BY' not in sql_upper and '/' not in sql):
                    return {
                        'violations': [f"DERIVED_METRIC: '{pat_re}' needs cross-table aggregation"],
                        'corrections': [f"Built derived metric query"],
                        'corrected_sql': corrected_sql,
                    }

        sum_match = re.search(r'(?:sum|total)\s+(?:of\s+)?(?:all\s+)?(.+?)(?:\s+cost[s]?|\s+amount[s]?)', q_lower)
        if sum_match and 'SUM' not in sql_upper:
            entity = sum_match.group(1).strip().lower()
            SUM_MAP = {
                'prescription': ('prescriptions', 'COST'),
                'drug': ('prescriptions', 'COST'),
                'claim': ('claims', 'PAID_AMOUNT'),
                'paid': ('claims', 'PAID_AMOUNT'),
                'billed': ('claims', 'BILLED_AMOUNT'),
            }
            for key, (table, col) in SUM_MAP.items():
                if key in entity:
                    corrected = f"SELECT ROUND(SUM(CAST({col} AS REAL)), 2) AS total FROM {table}"
                    return {
                        'violations': [f"SUM_MISSING: 'sum of {entity}' should use SUM({col}) not COUNT"],
                        'corrections': [f"Replaced with SUM({col}) FROM {table}"],
                        'corrected_sql': corrected,
                    }

        TEMPORAL_PATTERNS = [
            (r'(?:by|per)\s+quarter', 'quarter'),
            (r'over\s+time', 'month'),
            (r'(?:by|per)\s+month', 'month'),
            (r'trend\s+by\s+month', 'month'),
        ]
        for pat_re, granularity in TEMPORAL_PATTERNS:
            if re.search(pat_re, q_lower) and 'GROUP BY' not in sql_upper:
                TEMPORAL_TABLE_MAP = {
                    'claim': ('claims', 'SERVICE_DATE'),
                    'encounter': ('encounters', 'SERVICE_DATE'),
                    'prescription': ('prescriptions', 'PRESCRIPTION_DATE'),
                    'diagnosis': ('diagnoses', 'DIAGNOSIS_DATE'),
                    'referral': ('referrals', 'REFERRAL_DATE'),
                    'appointment': ('appointments', 'APPOINTMENT_DATE'),
                    'fill': ('prescriptions', 'PRESCRIPTION_DATE'),
                }
                table = None
                date_col = None
                for key, (t, dc) in TEMPORAL_TABLE_MAP.items():
                    if key in q_lower:
                        table, date_col = t, dc
                        break
                if table and date_col:
                    if granularity == 'quarter':
                        time_expr = f"strftime('%Y', {date_col}) || '-Q' || ((CAST(strftime('%m', {date_col}) AS INTEGER) - 1) / 3 + 1)"
                    else:
                        time_expr = f"strftime('%Y-%m', {date_col})"
                    corrected = (
                        f"SELECT {time_expr} AS period, COUNT(*) AS cnt "
                        f"FROM {table} WHERE {date_col} IS NOT NULL AND {date_col} != '' "
                        f"GROUP BY period ORDER BY period"
                    )
                    return {
                        'violations': [f"TEMPORAL: Question needs {granularity} grouping but SQL has no GROUP BY"],
                        'corrections': [f"Added temporal grouping by {granularity}"],
                        'corrected_sql': corrected,
                    }

        if re.search(r'what\s+tables?\s+(?:exist|are\s+there|do\s+we\s+have|in\s+the)', q_lower):
            corrected = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            return {
                'violations': ['META_QUERY: Question asks about schema not data'],
                'corrections': ['Replaced with sqlite_master query'],
                'corrected_sql': corrected,
            }

        vs_match = re.search(r'(?:average|avg|mean|total|sum)\s+(\w+)\s+vs\s+(?:average|avg|mean|total|sum)\s+(\w+)', q_lower)
        if not vs_match:
            vs_match = re.search(r'(\w+)\s+vs\s+(\w+)\s+(?:amount|cost|value)', q_lower)
        if vs_match:
            col1_word = vs_match.group(1).lower()
            col2_word = vs_match.group(2).lower()
            COL_MAP = {
                'copay': 'COPAY', 'paid': 'PAID_AMOUNT', 'billed': 'BILLED_AMOUNT',
                'allowed': 'ALLOWED_AMOUNT', 'deductible': 'DEDUCTIBLE',
                'coinsurance': 'COINSURANCE', 'cost': 'COST',
            }
            c1 = COL_MAP.get(col1_word)
            c2 = COL_MAP.get(col2_word)
            if c1 and c2:
                corrected = (
                    f"SELECT ROUND(AVG(CAST({c1} AS REAL)), 2) AS avg_{col1_word}, "
                    f"ROUND(AVG(CAST({c2} AS REAL)), 2) AS avg_{col2_word} FROM claims"
                )
                return {
                    'violations': [f"COMPARISON: Need both {c1} and {c2} in result"],
                    'corrections': [f"Built comparison query with both columns"],
                    'corrected_sql': corrected,
                }

        if re.search(r'(?:member|patient)s?\s+with\s+chronic', q_lower):
            is_count = any(kw in q_lower for kw in ['how many', 'count', 'number', 'total'])
            if is_count and 'CHRONIC_CONDITIONS' not in sql_upper:
                corrected = "SELECT COUNT(*) AS total FROM members WHERE CHRONIC_CONDITIONS > 0"
                return {
                    'violations': ['CHRONIC_FILTER: Missing CHRONIC_CONDITIONS > 0 filter'],
                    'corrections': ['Added WHERE CHRONIC_CONDITIONS > 0'],
                    'corrected_sql': corrected,
                }

        region_match = re.search(r'(?:member|patient)s?\s+in\s+(california|ncal|scal|colorado|georgia|hawaii|northwest|mid.atlantic)', q_lower)
        if region_match:
            region_name = region_match.group(1).lower()
            REGION_MAP = {
                'california': "KP_REGION IN ('NCAL', 'SCAL')",
                'ncal': "KP_REGION = 'NCAL'",
                'scal': "KP_REGION = 'SCAL'",
                'colorado': "KP_REGION = 'CO'",
                'georgia': "KP_REGION = 'GA'",
                'hawaii': "KP_REGION = 'HI'",
                'northwest': "KP_REGION = 'NW'",
                'mid-atlantic': "KP_REGION = 'MAS'",
                'mid atlantic': "KP_REGION = 'MAS'",
            }
            region_filter = REGION_MAP.get(region_name)
            if region_filter and region_filter.split("'")[1] not in sql_upper:
                is_count = any(kw in q_lower for kw in ['how many', 'count', 'number', 'total'])
                if is_count:
                    corrected = f"SELECT COUNT(*) AS total FROM members WHERE {region_filter}"
                    return {
                        'violations': [f"REGION_FILTER: Missing region filter for {region_name}"],
                        'corrections': [f"Added WHERE {region_filter}"],
                        'corrected_sql': corrected,
                    }

        if re.search(r'top\s+specialt(?:y|ies)\s+by', q_lower) and 'GROUP BY' not in sql_upper:
            corrected = (
                "SELECT SPECIALTY, COUNT(*) as cnt FROM providers "
                "GROUP BY SPECIALTY ORDER BY cnt DESC LIMIT 10"
            )
            return {
                'violations': ['GROUP_BY_MISSING: Top specialties requires GROUP BY SPECIALTY'],
                'corrections': ['Built GROUP BY SPECIALTY query'],
                'corrected_sql': corrected,
            }

        if re.search(r'(?:average|avg)\s+copay\b', q_lower):
            if 'GROUP BY' in sql_upper and 'by' not in q_lower:
                corrected = "SELECT ROUND(AVG(CAST(COPAY AS REAL)), 2) AS avg_copay FROM claims"
                return {
                    'violations': ['UNNECESSARY_GROUP_BY: Simple average should not have GROUP BY'],
                    'corrections': ['Simplified to AVG(COPAY) FROM claims'],
                    'corrected_sql': corrected,
                }

        return None

    def _build_group_by_query(self, metric_phrase: str, group_phrase: str,
                               q_lower: str, q_words: Set[str]) -> Optional[str]:
        group_col_map = {
            'region': ('KP_REGION', ['claims', 'encounters']),
            'status': ('CLAIM_STATUS', ['claims']),
            'plan type': ('PLAN_TYPE', ['claims']),
            'plan': ('PLAN_TYPE', ['claims']),
            'department': ('DEPARTMENT', ['encounters', 'appointments']),
            'gender': ('GENDER', ['members']),
            'drug class': ('DRUG_CLASS', ['prescriptions']),
            'specialty': ('SPECIALTY', ['providers']),
            'category': ('CATEGORY', ['cpt_codes']),
            'type': ('CLAIM_TYPE', ['claims']),
        }

        metric_col_map = {
            'billed amount': ('BILLED_AMOUNT', 'claims'),
            'paid amount': ('PAID_AMOUNT', 'claims'),
            'allowed amount': ('ALLOWED_AMOUNT', 'claims'),
            'cost': ('COST', 'prescriptions'),
            'risk score': ('RISK_SCORE', 'members'),
            'length of stay': ('LENGTH_OF_STAY', 'encounters'),
            'panel size': ('PANEL_SIZE', 'providers'),
            'days supply': ('DAYS_SUPPLY', 'prescriptions'),
            'rvu': ('RVU', 'cpt_codes'),
        }

        group_col = None
        group_tables = None
        for phrase, (col, tables) in group_col_map.items():
            if phrase in group_phrase:
                group_col = col
                group_tables = tables
                break

        if not group_col:
            return None

        agg = 'AVG'
        if any(kw in q_words for kw in ['total', 'sum']):
            agg = 'SUM'
        elif any(kw in q_words for kw in ['count', 'number']):
            agg = 'COUNT'

        metric_col = None
        metric_table = None
        for phrase, (col, table) in metric_col_map.items():
            if phrase in metric_phrase:
                metric_col = col
                metric_table = table
                break

        if not metric_col:
            table = group_tables[0]
            return f"SELECT {group_col}, COUNT(*) as cnt FROM {table} GROUP BY {group_col} ORDER BY cnt DESC"

        table = metric_table
        if group_tables and metric_table in group_tables:
            return (
                f"SELECT {group_col}, ROUND({agg}(CAST({metric_col} AS REAL)), 2) as {agg.lower()}_{metric_col.lower()}, "
                f"COUNT(*) as cnt FROM {table} GROUP BY {group_col} ORDER BY {agg.lower()}_{metric_col.lower()} DESC"
            )
        elif group_tables:
            g_table = group_tables[0]
            return (
                f"SELECT {g_table}.{group_col}, ROUND({agg}(CAST({table}.{metric_col} AS REAL)), 2) as {agg.lower()}_{metric_col.lower()}, "
                f"COUNT(*) as cnt FROM {table} JOIN {g_table} ON {table}.MEMBER_ID = {g_table}.MEMBER_ID "
                f"GROUP BY {g_table}.{group_col} ORDER BY {agg.lower()}_{metric_col.lower()} DESC"
            )

        return None

    def _fix_missing_column(self, q_lower: str, q_words: Set[str],
                             sql: str, table: str, missing_col: str) -> Optional[Dict]:
        fix_map = {
            ('encounters', 'COST'): {
                'corrected_sql': (
                    "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) / "
                    "COUNT(DISTINCT ENCOUNTER_ID), 2) AS avg_cost_per_encounter FROM claims "
                    "WHERE ENCOUNTER_ID IS NOT NULL"
                ),
                'msg': "encounters has no COST column; computed from claims",
            },
            ('encounters', 'REGION'): {
                'corrected_sql': None,
                'msg': "Use KP_REGION instead of REGION",
            },
        }

        key = (table, missing_col)
        if key in fix_map:
            fix = fix_map[key]
            if fix['corrected_sql']:
                return {
                    'violations': [f"MISSING_COLUMN: {table}.{missing_col} does not exist"],
                    'corrections': [fix['msg']],
                    'corrected_sql': fix['corrected_sql'],
                }

        table_cols = self._table_columns.get(table, set())
        for real_col in table_cols:
            if missing_col in real_col or real_col in missing_col:
                corrected = re.sub(
                    rf'\b{re.escape(missing_col)}\b',
                    real_col,
                    sql,
                    flags=re.IGNORECASE
                )
                return {
                    'violations': [f"MISSING_COLUMN: {table}.{missing_col} → {table}.{real_col}"],
                    'corrections': [f"Replaced {missing_col} with {real_col}"],
                    'corrected_sql': corrected,
                }

        return None


    def _check_table_correctness(self, q_lower: str, q_words: Set[str],
                                  sql_upper: str, sql: str,
                                  understanding: Optional[Dict]) -> Optional[Dict]:
        violations = []
        corrections = []

        sql_tables = self._extract_tables_from_sql(sql_upper)

        required_table = None
        for phrase, table in QUESTION_TABLE_SEMANTICS.items():
            if len(phrase.split()) > 1:
                if phrase in q_lower:
                    required_table = table
                    break
            else:
                if phrase in q_words:
                    required_table = table
                    break

        if required_table and required_table not in sql_tables:
            violations.append(
                f"TABLE_MISMATCH: Question asks about '{required_table}' but SQL "
                f"queries {sql_tables}"
            )
            corrected = self._rebuild_sql_with_correct_table(sql, required_table, q_lower, q_words, understanding)
            if corrected:
                corrections.append(f"Redirected query to {required_table} table")
                return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}
            return {'violations': violations}

        if understanding:
            target_tables = understanding.get('target_tables', [])
            if target_tables and len(target_tables) == 1:
                expected = target_tables[0].lower()
                if expected in TABLE_ROW_COUNTS and expected not in sql_tables:
                    violations.append(
                        f"UNDERSTANDING_TABLE_MISMATCH: Understanding targets '{expected}' "
                        f"but SQL uses {sql_tables}"
                    )

        return {'violations': violations} if violations else None


    def _check_aggregation_correctness(self, q_lower: str, q_words: Set[str],
                                        sql_upper: str, sql: str,
                                        understanding: Optional[Dict]) -> Optional[Dict]:
        violations = []
        corrections = []

        expected_agg = None
        for phrase, agg in AGG_INTENT_MAP.items():
            if len(phrase.split()) > 1:
                if phrase in q_lower:
                    expected_agg = agg
                    break
            else:
                if phrase in q_words:
                    expected_agg = agg
                    break

        if not expected_agg:
            return None

        actual_aggs = set()
        for agg in ['SUM', 'AVG', 'COUNT', 'MAX', 'MIN']:
            if f'{agg}(' in sql_upper:
                actual_aggs.add(agg)

        if expected_agg == 'COUNT' and 'COUNT' in actual_aggs:
            return None

        if expected_agg and expected_agg not in actual_aggs:
            wrong_agg = actual_aggs - {'COUNT'}
            if wrong_agg:
                wrong = list(wrong_agg)[0]
                violations.append(
                    f"AGGREGATION_MISMATCH: Question implies {expected_agg} but SQL uses {wrong}"
                )
                corrected = self._fix_aggregation(sql, wrong, expected_agg)
                if corrected:
                    corrections.append(f"Changed {wrong}() to {expected_agg}()")
                    return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

        return {'violations': violations} if violations else None


    def _check_join_integrity(self, q_lower: str, q_words: Set[str],
                               sql_upper: str, sql: str,
                               understanding: Optional[Dict]) -> Optional[Dict]:
        violations = []
        corrections = []

        has_join = ' JOIN ' in sql_upper
        if not has_join:
            return None

        sql_tables = self._extract_tables_from_sql(sql_upper)

        primary_table = None
        for phrase, table in QUESTION_TABLE_SEMANTICS.items():
            if len(phrase.split()) > 1:
                if phrase in q_lower:
                    primary_table = table
                    break
            else:
                if phrase in q_words:
                    primary_table = table
                    break

        if not primary_table and understanding:
            target_tables = understanding.get('target_tables', [])
            if target_tables:
                primary_table = target_tables[0].lower()

        if not primary_table:
            return None

        has_count_star = 'COUNT(*)' in sql_upper or 'COUNT(DISTINCT' in sql_upper
        is_count_question = any(kw in q_lower for kw in ['how many', 'count of', 'count all', 'total number', 'number of'])

        if is_count_question and has_count_star and len(sql_tables) > 1:
            violations.append(
                f"JOIN_INFLATION_COUNT: '{primary_table}' count query has JOIN to "
                f"{sql_tables - {primary_table}}, which inflates COUNT(*)"
            )
            corrected = f"SELECT COUNT(*) AS total FROM {primary_table}"
            corrections.append(f"Removed JOIN — count queries should use single table")
            return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

        has_sum = 'SUM(' in sql_upper
        has_avg = 'AVG(' in sql_upper
        if (has_sum or has_avg) and len(sql_tables) > 1:
            agg_match = re.findall(r'(?:SUM|AVG)\s*\(\s*(?:CAST\s*\()?\s*(\w+)\.(\w+)', sql_upper)
            if agg_match:
                agg_table, agg_col = agg_match[0][0].lower(), agg_match[0][1]
                other_tables = sql_tables - {agg_table}

                needs_other_tables = False
                for other_t in other_tables:
                    if f'{other_t.upper()}.' in sql_upper.split('FROM')[0]:
                        needs_other_tables = True
                    where_part = sql_upper.split('WHERE')[1] if 'WHERE' in sql_upper else ''
                    group_part = sql_upper.split('GROUP BY')[1] if 'GROUP BY' in sql_upper else ''
                    if f'{other_t.upper()}.' in where_part or f'{other_t.upper()}.' in group_part:
                        needs_other_tables = True

                if not needs_other_tables and other_tables:
                    multiplier = 1.0
                    for ot in other_tables:
                        multiplier *= AVG_ROWS_PER_MEMBER.get(ot, 1.0)
                    if multiplier > 1.1:
                        violations.append(
                            f"JOIN_INFLATION_AGG: {agg_table}.{agg_col} aggregation JOINs "
                            f"{other_tables} unnecessarily — inflates result by ~{multiplier:.1f}x"
                        )
                        corrected = self._rebuild_single_table_agg(sql, agg_table, agg_col, q_lower, q_words)
                        if corrected:
                            corrections.append(f"Removed unnecessary JOIN — aggregation uses only {agg_table}")
                            return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

        join_patterns = re.findall(r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql_upper)
        for t1, c1, t2, c2 in join_patterns:
            bad_join_cols = {'PLAN_TYPE', 'KP_REGION', 'STATUS', 'GENDER', 'RACE',
                            'CATEGORY', 'SPECIALTY', 'CLAIM_STATUS', 'DEPARTMENT'}
            if c1.upper() in bad_join_cols or c2.upper() in bad_join_cols:
                violations.append(
                    f"BAD_JOIN_KEY: JOIN on {t1}.{c1} = {t2}.{c2} — this is a categorical "
                    f"column, not a key. Causes cartesian product."
                )
                corrected = sql.replace(
                    f"{t1}.{c1} = {t2}.{c2}",
                    f"{t1}.MEMBER_ID = {t2}.MEMBER_ID"
                )
                t1_cols = self._table_columns.get(t1.lower(), set())
                t2_cols = self._table_columns.get(t2.lower(), set())
                if 'MEMBER_ID' in t1_cols and 'MEMBER_ID' in t2_cols:
                    corrections.append(f"Fixed JOIN key: {c1} → MEMBER_ID")
                    return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

        return {'violations': violations} if violations else None


    def _check_where_validity(self, q_lower: str, q_words: Set[str],
                               sql_upper: str, sql: str,
                               understanding: Optional[Dict]) -> Optional[Dict]:
        violations = []
        corrections = []

        if 'WHERE' not in sql_upper:
            return None

        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
                                sql_upper, re.DOTALL)
        if not where_match:
            return None
        where_clause = where_match.group(1).strip()

        global_keywords = {'maximum', 'minimum', 'total', 'overall', 'all', 'entire',
                          'average', 'highest', 'lowest'}
        is_global = bool(q_words & global_keywords)

        filter_keywords = {'denied', 'approved', 'pending', 'cancelled', 'california',
                          'hmo', 'ppo', 'medicare', 'medicaid', 'inpatient', 'outpatient',
                          'emergency', 'male', 'female', 'chronic', 'above', 'below',
                          'over', 'under', 'between', 'where', 'for', 'with', 'only'}
        has_filter_intent = bool(q_words & filter_keywords)

        if is_global and not has_filter_intent:
            if 'ICD10' in where_clause or 'DESCRIPTION' in where_clause:
                violations.append(
                    f"OVER_FILTERED: Global query has WHERE on diagnosis: {where_clause[:80]}"
                )
                corrected = re.sub(
                    r'\s+WHERE\s+.+?(?=\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
                    '', sql, flags=re.DOTALL | re.IGNORECASE
                ).strip()
                corrections.append("Removed restrictive WHERE clause from global query")
                return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

            string_values = re.findall(r"=\s*'([^']+)'", where_clause)
            for val in string_values:
                val_lower = val.lower()
                if not any(word in q_lower for word in val_lower.split()):
                    violations.append(
                        f"PHANTOM_FILTER: WHERE clause has '{val}' but question doesn't mention it"
                    )
                    corrected = re.sub(
                        r'\s+WHERE\s+.+?(?=\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
                        '', sql, flags=re.DOTALL | re.IGNORECASE
                    ).strip()
                    corrections.append(f"Removed phantom WHERE filter '{val}'")
                    return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

        return {'violations': violations} if violations else None


    def _check_group_by_intent(self, q_lower: str, q_words: Set[str],
                               sql_upper: str, sql: str,
                               understanding: Optional[Dict]) -> Optional[Dict]:
        violations = []
        corrections = []

        by_pattern = re.search(r'\bby\s+(\w+(?:\s+\w+)?)\b', q_lower)
        if not by_pattern:
            return None

        group_by_phrase = by_pattern.group(1).strip().lower()

        has_group_by = 'GROUP BY' in sql_upper

        if not has_group_by:
            violations.append(
                f"GROUP_BY_MISSING: Question asks to group 'by {group_by_phrase}' "
                f"but SQL has no GROUP BY clause"
            )

            sql_tables = self._extract_tables_from_sql(sql_upper)
            target_table = None

            matched_col = None
            for phrase_key, table_map in GROUP_BY_SEMANTICS.items():
                if phrase_key == group_by_phrase or phrase_key in group_by_phrase:
                    for possible_table in sql_tables:
                        if possible_table in table_map:
                            target_table = possible_table
                            matched_col = table_map[possible_table]
                            break

            if target_table and matched_col:
                corrected = self._add_group_by_to_sql(
                    sql, target_table, matched_col, q_lower, q_words
                )
                if corrected:
                    corrections.append(
                        f"Added GROUP BY {matched_col} for 'by {group_by_phrase}' intent"
                    )
                    return {'violations': violations, 'corrections': corrections, 'corrected_sql': corrected}

            return {'violations': violations}

        return None


    def _check_data_sanity(self, q_lower: str, sql_upper: str, sql: str,
                           execution_result: Tuple) -> Optional[Dict]:
        rows, columns, error = execution_result
        if error or not rows:
            return None

        violations = []

        if 'COUNT(*)' in sql_upper or 'COUNT(' in sql_upper:
            try:
                count_val = None
                for i, col in enumerate(columns):
                    if 'count' in str(col).lower() or 'total' in str(col).lower():
                        count_val = rows[0][i]
                        break
                if count_val is None and len(rows) > 0 and len(rows[0]) > 0:
                    count_val = rows[0][0]

                if count_val and isinstance(count_val, (int, float)):
                    sql_tables = self._extract_tables_from_sql(sql_upper)
                    for table in sql_tables:
                        expected = TABLE_ROW_COUNTS.get(table)
                        if expected and count_val > expected * 2:
                            violations.append(
                                f"INFLATED_COUNT: Got {count_val:,.0f} but {table} only has "
                                f"{expected:,} rows — JOIN inflation detected"
                            )
            except (IndexError, TypeError, ValueError):
                pass

        for (ref_table, ref_col, ref_agg), ref_val in REFERENCE_VALUES.items():
            if ref_agg + '(' in sql_upper and ref_col in sql_upper:
                try:
                    actual_val = float(rows[0][0])
                    if abs(actual_val - ref_val) / max(abs(ref_val), 0.001) > 0.10:
                        if ref_agg in ('SUM', 'COUNT'):
                            if actual_val > ref_val * 1.5:
                                violations.append(
                                    f"INFLATED_VALUE: {ref_agg}({ref_col}) = {actual_val:,.2f} "
                                    f"but reference is {ref_val:,.2f} — possible JOIN inflation"
                                )
                except (IndexError, TypeError, ValueError):
                    pass

        return {'violations': violations} if violations else None


    def _extract_tables_from_sql(self, sql_upper: str) -> Set[str]:
        tables = set()
        from_match = re.findall(r'FROM\s+(\w+)', sql_upper)
        for t in from_match:
            tables.add(t.lower())
        join_match = re.findall(r'JOIN\s+(\w+)', sql_upper)
        for t in join_match:
            tables.add(t.lower())
        tables -= {'as', 'on', 'where', 'select', 'and', 'or', 'not',
                   'cast', 'real', 'case', 'when', 'then', 'else', 'end'}
        return tables

    def _rebuild_sql_with_correct_table(self, sql: str, correct_table: str,
                                         q_lower: str, q_words: Set[str],
                                         understanding: Optional[Dict]) -> Optional[str]:
        sql_upper = sql.upper()

        is_count = any(kw in q_lower for kw in ['how many', 'count', 'number of', 'total number'])
        has_max = any(kw in q_words for kw in ['maximum', 'highest', 'largest', 'biggest'])
        has_min = any(kw in q_words for kw in ['minimum', 'lowest', 'smallest'])
        has_avg = any(kw in q_words for kw in ['average', 'avg', 'mean'])
        has_sum = any(kw in q_words for kw in ['total', 'sum', 'overall'])

        if is_count:
            return f"SELECT COUNT(*) AS total FROM {correct_table}"

        correct_agg = None
        if has_min:
            correct_agg = 'MIN'
        elif has_max:
            correct_agg = 'MAX'
        elif has_avg:
            correct_agg = 'AVG'
        elif has_sum:
            correct_agg = 'SUM'
        else:
            correct_agg = 'AVG'

        target_col = self._find_column_for_aggregation(correct_table, q_lower, q_words, understanding)

        if target_col:
            return (f"SELECT {correct_agg}(CAST({correct_table}.{target_col} AS REAL)) "
                    f"AS {correct_agg.lower()}_{target_col.lower()} FROM {correct_table}")

        return None

    def _find_column_for_aggregation(self, table: str, q_lower: str,
                                      q_words: Set[str],
                                      understanding: Optional[Dict]) -> Optional[str]:
        table_cols = self._table_columns.get(table, set())
        if not table_cols:
            return None

        if understanding:
            metrics = understanding.get('metric_columns', [])
            if metrics:
                if isinstance(metrics[0], (list, tuple)):
                    col = metrics[0][0]
                else:
                    col = metrics[0]

                col_upper = col.upper()
                if col_upper in table_cols:
                    return col

        semantic_map = {
            'claims': {
                'paid': 'PAID_AMOUNT',
                'billed': 'BILLED_AMOUNT',
                'allowed': 'ALLOWED_AMOUNT',
                'risk': 'RISK_SCORE',
                'cost': 'PAID_AMOUNT',
            },
            'members': {
                'risk': 'RISK_SCORE',
                'score': 'RISK_SCORE',
                'age': 'AGE',
            },
            'encounters': {
                'stay': 'LENGTH_OF_STAY',
                'los': 'LENGTH_OF_STAY',
            },
            'prescriptions': {
                'cost': 'COST',
                'days': 'DAYS_SUPPLY',
                'supply': 'DAYS_SUPPLY',
            },
            'cpt_codes': {
                'rvu': 'RVU',
                'cost': 'BASE_COST',
            },
        }

        q_concepts = q_lower.split()
        table_map = semantic_map.get(table, {})

        for concept, col in table_map.items():
            if concept in q_lower and col in table_cols:
                return col


        return None

    def _fix_aggregation(self, sql: str, wrong_agg: str, correct_agg: str) -> Optional[str]:
        pattern = re.compile(
            rf'{wrong_agg}\s*\(\s*((?:CAST\s*\()?[^)]+\)?\s*(?:AS\s+\w+\s*\))?)',
            re.IGNORECASE
        )
        corrected = pattern.sub(f'{correct_agg}(\\1', sql, count=1)

        corrected = re.sub(
            rf'\b{wrong_agg.lower()}_(\w+)',
            f'{correct_agg.lower()}_\\1',
            corrected,
            flags=re.IGNORECASE
        )

        if corrected != sql:
            return corrected
        return None

    def _rebuild_single_table_agg(self, sql: str, table: str, col: str,
                                    q_lower: str, q_words: Set[str]) -> Optional[str]:
        sql_upper = sql.upper()

        agg = 'SUM'
        if 'AVG(' in sql_upper:
            agg = 'AVG'
        elif 'MAX(' in sql_upper:
            agg = 'MAX'
        elif 'MIN(' in sql_upper:
            agg = 'MIN'
        elif 'COUNT(' in sql_upper:
            agg = 'COUNT'

        group_match = re.search(r'GROUP\s+BY\s+(\w+)\.(\w+)', sql_upper)
        group_col = None
        if group_match:
            group_table = group_match.group(1).lower()
            group_column = group_match.group(2)
            table_cols = self._table_columns.get(table, set())
            if group_column.upper() in table_cols:
                group_col = f"{table}.{group_column}"

        where_match = re.search(
            r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
            sql, flags=re.DOTALL | re.IGNORECASE
        )
        where_clause = ''
        if where_match:
            where_raw = where_match.group(1)
            conditions = re.split(r'\s+AND\s+', where_raw, flags=re.IGNORECASE)
            valid_conds = []
            for cond in conditions:
                cond_upper = cond.upper()
                if f'{table.upper()}.' in cond_upper or '.' not in cond:
                    valid_conds.append(cond.strip())
            if valid_conds:
                where_clause = ' WHERE ' + ' AND '.join(valid_conds)

        if group_col:
            return (f"SELECT {group_col}, {agg}(CAST({table}.{col} AS REAL)) "
                    f"AS {agg.lower()}_{col.lower()} FROM {table}"
                    f"{where_clause} GROUP BY {group_col} "
                    f"ORDER BY {agg.lower()}_{col.lower()} DESC LIMIT 100")
        else:
            return (f"SELECT {agg}(CAST({table}.{col} AS REAL)) "
                    f"AS {agg.lower()}_{col.lower()} FROM {table}{where_clause}")

    def _add_group_by_to_sql(self, sql: str, table: str, group_col: str,
                              q_lower: str, q_words: Set[str]) -> Optional[str]:
        sql_upper = sql.upper()

        agg = 'COUNT'
        if 'SUM(' in sql_upper:
            agg = 'SUM'
        elif 'AVG(' in sql_upper:
            agg = 'AVG'
        elif 'MAX(' in sql_upper:
            agg = 'MAX'
        elif 'MIN(' in sql_upper:
            agg = 'MIN'

        agg_match = re.search(
            rf'{agg}\s*\(\s*(?:CAST\s*\()?(?:\w+\.)?(\w+)',
            sql_upper
        )
        agg_col = None
        if agg_match:
            agg_col = agg_match.group(1)

        if not agg_col and agg != 'COUNT':
            return None

        where_clause = ''
        where_match = re.search(
            r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
            sql, flags=re.DOTALL | re.IGNORECASE
        )
        if where_match:
            where_raw = where_match.group(1)
            where_clause = f' WHERE {where_raw}'

        if agg == 'COUNT':
            return (f"SELECT {table}.{group_col}, COUNT(*) AS total "
                    f"FROM {table}{where_clause} "
                    f"GROUP BY {table}.{group_col} "
                    f"ORDER BY total DESC LIMIT 100")
        else:
            return (f"SELECT {table}.{group_col}, {agg}(CAST({table}.{agg_col} AS REAL)) "
                    f"AS {agg.lower()}_{agg_col.lower()} FROM {table}{where_clause} "
                    f"GROUP BY {table}.{group_col} "
                    f"ORDER BY {agg.lower()}_{agg_col.lower()} DESC LIMIT 100")


    def train_from_ground_truth(self, ground_truth: List[Dict]):
        for entry in ground_truth:
            q = entry['question'].lower()
            sql = entry['sql']
            import hashlib
            q_hash = hashlib.md5(q.encode()).hexdigest()[:12]
            self._learned_corrections[q_hash] = {
                'question': q,
                'correct_sql': sql,
                'expected_value': entry.get('expected_value'),
                'source': 'ground_truth',
            }

        store_path = os.path.join(os.path.dirname(self.db_path), 'quality_gate_corrections.json')
        try:
            with open(store_path, 'w') as f:
                json.dump(self._learned_corrections, f, indent=2)
            logger.info("Trained quality gate with %d ground truth entries", len(ground_truth))
        except Exception as e:
            logger.warning("Could not save ground truth: %s", e)

    def get_correction_stats(self) -> Dict:
        return {
            'total_learned_corrections': len(self._learned_corrections),
            'correction_sources': dict(
                (s, sum(1 for v in self._learned_corrections.values()
                        if v.get('source') == s))
                for s in set(v.get('source', 'auto') for v in self._learned_corrections.values())
            ),
        }
