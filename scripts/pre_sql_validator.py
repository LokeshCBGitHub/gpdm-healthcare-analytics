import re
import logging
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.pre_sql_validator')


@dataclass
class ValidationResult:
    valid: bool = True
    corrected_sql: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0


SUPERLATIVE_TO_AGG = {
    'highest': 'MAX', 'maximum': 'MAX', 'max': 'MAX', 'biggest': 'MAX',
    'largest': 'MAX', 'greatest': 'MAX', 'most expensive': 'MAX',
    'top': 'MAX',
    'lowest': 'MIN', 'minimum': 'MIN', 'min': 'MIN', 'smallest': 'MIN',
    'least': 'MIN', 'cheapest': 'MIN', 'fewest': 'MIN',
}

AVERAGE_WORDS = {'average', 'avg', 'mean', 'typical'}
SUM_WORDS = {'total', 'sum', 'overall', 'combined', 'aggregate', 'cumulative'}
COUNT_WORDS = {'count', 'number', 'how many', 'total number', 'quantity'}

NUMERIC_COLUMNS = {
    'PAID_AMOUNT', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'COPAY', 'COINSURANCE',
    'DEDUCTIBLE', 'MEMBER_RESPONSIBILITY', 'COST', 'RVU', 'RISK_SCORE',
    'PANEL_SIZE', 'LENGTH_OF_STAY', 'DAYS_SUPPLY', 'REFILLS_USED',
    'QUANTITY', 'TOTAL_COST',
}

PHRASE_TO_COLUMN = {
    'paid amount': 'PAID_AMOUNT', 'cost': 'PAID_AMOUNT', 'spend': 'PAID_AMOUNT',
    'spending': 'PAID_AMOUNT', 'expense': 'PAID_AMOUNT', 'payment': 'PAID_AMOUNT',
    'billed amount': 'BILLED_AMOUNT', 'billed': 'BILLED_AMOUNT',
    'charge': 'BILLED_AMOUNT', 'charges': 'BILLED_AMOUNT',
    'allowed amount': 'ALLOWED_AMOUNT', 'allowed': 'ALLOWED_AMOUNT',
    'copay': 'COPAY', 'copayment': 'COPAY',
    'coinsurance': 'COINSURANCE',
    'deductible': 'DEDUCTIBLE',
    'member responsibility': 'MEMBER_RESPONSIBILITY',
    'risk score': 'RISK_SCORE', 'risk': 'RISK_SCORE',
    'panel size': 'PANEL_SIZE',
    'length of stay': 'LENGTH_OF_STAY', 'los': 'LENGTH_OF_STAY',
    'stay': 'LENGTH_OF_STAY', 'bed days': 'LENGTH_OF_STAY',
    'days supply': 'DAYS_SUPPLY', 'day supply': 'DAYS_SUPPLY',
    'supply': 'DAYS_SUPPLY',
    'rvu': 'RVU', 'rvus': 'RVU', 'relative value': 'RVU',
    'refills': 'REFILLS_USED', 'refill': 'REFILLS_USED',
    'drug cost': 'COST', 'medication cost': 'COST', 'prescription cost': 'COST',
}

GROUPABLE_ENTITIES = {
    'provider', 'providers', 'doctor', 'doctors', 'physician', 'physicians',
    'specialist', 'specialists', 'region', 'regions', 'facility', 'facilities',
    'department', 'departments', 'specialty', 'specialties', 'payer', 'payers',
    'plan', 'plans', 'diagnosis', 'diagnoses', 'medication', 'medications',
    'member', 'members', 'patient', 'patients', 'category', 'categories',
    'type', 'types', 'status', 'class', 'group',
}

GROUP_BY_PHRASES = {
    'by region': 'KP_REGION', 'per region': 'KP_REGION', 'by area': 'KP_REGION',
    'by plan type': 'PLAN_TYPE', 'per plan': 'PLAN_TYPE', 'by plan': 'PLAN_TYPE',
    'by payer': 'PLAN_TYPE', 'per payer': 'PLAN_TYPE',
    'by specialty': 'SPECIALTY', 'per specialty': 'SPECIALTY',
    'by department': 'DEPARTMENT', 'per department': 'DEPARTMENT',
    'by facility': 'FACILITY', 'per facility': 'FACILITY',
    'by visit type': 'VISIT_TYPE', 'per visit type': 'VISIT_TYPE',
    'by gender': 'GENDER', 'per gender': 'GENDER',
    'by race': 'RACE', 'per race': 'RACE',
    'by category': 'CATEGORY', 'per category': 'CATEGORY',
    'by status': 'STATUS', 'per status': 'STATUS',
    'by claim status': 'CLAIM_STATUS', 'by claim type': 'CLAIM_TYPE',
    'by medication': 'MEDICATION_NAME', 'by drug': 'MEDICATION_NAME',
    'by medication class': 'MEDICATION_CLASS', 'by drug class': 'MEDICATION_CLASS',
    'by diagnosis': 'ICD10_DESCRIPTION', 'by condition': 'ICD10_DESCRIPTION',
    'by provider': 'RENDERING_NPI', 'per provider': 'RENDERING_NPI',
    'by member': 'MEMBER_ID', 'per member': 'MEMBER_ID',
    'by severity': 'SEVERITY', 'by urgency': 'URGENCY',
    'by month': None,
    'by year': None,
    'by quarter': None,
    'over time': None,
}

TABLE_ENTITY_WORDS = {
    'claims': ['claim', 'claims', 'billing', 'reimbursement'],
    'members': ['member', 'members', 'patient', 'patients', 'enrollee', 'enrollees', 'beneficiary'],
    'encounters': ['encounter', 'encounters', 'visit', 'visits', 'admission', 'admissions'],
    'providers': ['provider', 'providers', 'doctor', 'doctors', 'physician', 'physicians', 'npi'],
    'prescriptions': ['prescription', 'prescriptions', 'medication', 'medications', 'drug', 'drugs', 'rx', 'pharmacy'],
    'diagnoses': ['diagnosis', 'diagnoses', 'condition', 'conditions', 'icd', 'disease'],
    'referrals': ['referral', 'referrals', 'referred'],
    'appointments': ['appointment', 'appointments', 'scheduled', 'no-show', 'noshow'],
    'cpt_codes': ['cpt', 'procedure', 'rvu', 'rvus', 'procedure code'],
}


class PreSQLValidator:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._table_columns: Dict[str, set] = {}
        self._build_column_registry()
        self._correction_log: List[Dict] = []

    def _build_column_registry(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_gpdm%' AND name NOT LIKE 'query_patterns%'")
            for (table_name,) in cursor.fetchall():
                cursor.execute(f"PRAGMA table_info({table_name})")
                self._table_columns[table_name] = {row[1] for row in cursor.fetchall()}
            conn.close()
            logger.info("PreSQLValidator: loaded %d tables", len(self._table_columns))
        except Exception as e:
            logger.error("PreSQLValidator: failed to load columns: %s", e)


    def validate_and_correct(self, question: str, sql: str) -> ValidationResult:
        if not sql or sql.startswith('--'):
            return ValidationResult(valid=True)

        result = ValidationResult(valid=True, corrected_sql=sql)
        q = question.lower().strip()
        current_sql = sql

        checks = [
            self._check_aggregation_semantics,
            self._check_group_by_completeness,
            self._check_table_routing,
            self._check_column_existence,
            self._check_superlative_correctness,
            self._check_filter_completeness,
            self._check_join_safety,
        ]

        for check in checks:
            check_result = check(q, current_sql)
            if check_result:
                corrected_sql, violation, correction = check_result
                if corrected_sql != current_sql:
                    result.violations.append(violation)
                    result.corrections.append(correction)
                    result.confidence_adjustment -= 0.05
                    current_sql = corrected_sql
                    result.valid = False

        result.corrected_sql = current_sql

        if result.corrections:
            self._correction_log.append({
                'question': question,
                'original_sql': sql,
                'corrected_sql': current_sql,
                'violations': result.violations,
                'corrections': result.corrections,
            })
            logger.info("PreSQLValidator corrected %d issues: %s",
                        len(result.corrections), result.violations)

        return result


    def _check_aggregation_semantics(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        asked_agg = None
        target_col = None

        for word, agg in SUPERLATIVE_TO_AGG.items():
            if word in q:
                words_after = q[q.index(word) + len(word):]
                has_groupable = any(ge in words_after.split() for ge in GROUPABLE_ENTITIES)
                has_by_pattern = bool(re.search(r'\bby\s+\w+', words_after))
                has_top_n = bool(re.search(r'top\s+\d+', q))

                if not has_groupable and not has_by_pattern and not has_top_n:
                    asked_agg = agg
                    break

        if not asked_agg:
            for avg_word in AVERAGE_WORDS:
                if avg_word in q.split() or avg_word in q:
                    asked_agg = 'AVG'
                    break

        if not asked_agg:
            if any(w in q for w in ['total cost', 'total paid', 'total billed',
                                     'total spend', 'total amount', 'sum of']):
                asked_agg = 'SUM'

        if asked_agg:
            for phrase, col in sorted(PHRASE_TO_COLUMN.items(), key=lambda x: -len(x[0])):
                if phrase in q:
                    target_col = col
                    break

        if asked_agg and target_col:
            has_correct_agg = asked_agg in sql_upper
            has_target_col = target_col in sql_upper

            if not has_correct_agg and has_target_col:
                return self._fix_aggregation(sql, asked_agg, target_col, q)

            elif has_correct_agg and not has_target_col:
                return self._fix_column(sql, asked_agg, target_col, q)

            elif not has_correct_agg and not has_target_col:
                return self._rebuild_aggregation(sql, asked_agg, target_col, q)

        return None

    def _fix_aggregation(self, sql: str, correct_agg: str, target_col: str,
                         q: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        wrong_aggs = []
        for agg in ['SUM', 'AVG', 'MAX', 'MIN', 'COUNT']:
            if agg != correct_agg and f'{agg}(' in sql_upper:
                pattern = rf'{agg}\s*\(\s*(?:CAST\s*\()?\s*{target_col}'
                if re.search(pattern, sql_upper):
                    wrong_aggs.append(agg)

        if wrong_aggs:
            new_sql = sql
            for wrong_agg in wrong_aggs:
                pattern = rf'({wrong_agg})\s*(\(\s*(?:CAST\s*\()?\s*{target_col})'
                new_sql = re.sub(pattern, f'{correct_agg}\\2', new_sql, flags=re.IGNORECASE)
            return (new_sql,
                    f'Wrong aggregation: {wrong_aggs[0]}({target_col}) should be {correct_agg}({target_col})',
                    f'Fixed: {wrong_aggs[0]} → {correct_agg}')

        if correct_agg in ('MAX', 'MIN') and 'ORDER BY' in sql_upper and 'LIMIT' in sql_upper:
            from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            table = from_match.group(1) if from_match else 'claims'
            where_match = re.search(r'(WHERE\s+.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE)
            where_clause = where_match.group(1).strip() if where_match else ''
            new_sql = f"SELECT {correct_agg}(CAST({target_col} AS REAL)) AS {correct_agg.lower()}_{target_col.lower()} FROM {table}"
            if where_clause:
                new_sql += f" {where_clause}"
            return (new_sql,
                    f'ORDER BY + LIMIT used instead of {correct_agg}',
                    f'Rewrote to {correct_agg}({target_col})')

        return None

    def _fix_column(self, sql: str, correct_agg: str, target_col: str,
                    q: str) -> Optional[Tuple[str, str, str]]:
        pattern = rf'{correct_agg}\s*\(\s*(?:CAST\s*\()?\s*(\w+)'
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            wrong_col = match.group(1)
            if wrong_col != target_col and wrong_col in ('ENCOUNTER_ID', 'CLAIM_ID',
                'MEMBER_ID', 'RX_ID', 'REFERRAL_ID', 'NPI', 'APPOINTMENT_ID'):
                new_sql = sql.replace(wrong_col, target_col)
                return (new_sql,
                        f'Wrong column: {correct_agg}({wrong_col}) should be {correct_agg}({target_col})',
                        f'Fixed column: {wrong_col} → {target_col}')
        return None

    def _rebuild_aggregation(self, sql: str, correct_agg: str, target_col: str,
                             q: str) -> Optional[Tuple[str, str, str]]:
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if not from_match:
            return None

        table = from_match.group(1)

        correct_table = None
        for t, cols in self._table_columns.items():
            if target_col in cols:
                correct_table = t
                break

        if not correct_table:
            return None

        where_match = re.search(r'(WHERE\s+.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE)
        where_clause = where_match.group(1).strip() if where_match else ''

        group_match = re.search(r'(GROUP BY\s+\S+)', sql, re.IGNORECASE)
        group_clause = group_match.group(1) if group_match else ''

        if group_clause:
            group_col_match = re.search(r'GROUP BY\s+(\S+)', group_clause, re.IGNORECASE)
            group_col = group_col_match.group(1) if group_col_match else ''
            new_sql = f"SELECT {group_col}, ROUND({correct_agg}(CAST({target_col} AS REAL)), 2) AS {correct_agg.lower()}_{target_col.lower()} FROM {correct_table}"
        else:
            new_sql = f"SELECT ROUND({correct_agg}(CAST({target_col} AS REAL)), 2) AS {correct_agg.lower()}_{target_col.lower()} FROM {correct_table}"

        if where_clause:
            new_sql += f" {where_clause}"
        if group_clause:
            new_sql += f" {group_clause}"

        return (new_sql,
                f'Rebuilt: needed {correct_agg}({target_col}) FROM {correct_table}',
                f'Complete rebuild of aggregation query')


    def _check_group_by_completeness(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        for phrase, column in GROUP_BY_PHRASES.items():
            if phrase in q and column:
                if 'GROUP BY' not in sql_upper:
                    return self._add_group_by(sql, column, q)
                elif column not in sql_upper:
                    return self._fix_group_by_column(sql, column, q)
        return None

    def _add_group_by(self, sql: str, group_col: str, q: str) -> Optional[Tuple[str, str, str]]:
        target_table = None
        for table, cols in self._table_columns.items():
            if group_col in cols:
                target_table = table
                break

        if not target_table:
            return None

        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        current_table = from_match.group(1) if from_match else None

        if current_table and current_table in self._table_columns:
            if group_col not in self._table_columns[current_table]:
                agg_match = re.search(r'((?:SUM|AVG|MAX|MIN|COUNT)\s*\([^)]+\))', sql, re.IGNORECASE)
                agg_expr = agg_match.group(1) if agg_match else 'COUNT(*)'

                agg_col_match = re.search(r'(?:SUM|AVG|MAX|MIN)\s*\(\s*(?:CAST\s*\()?\s*(\w+)', agg_expr, re.IGNORECASE)
                if agg_col_match:
                    agg_col = agg_col_match.group(1)
                    if agg_col in self._table_columns.get(target_table, set()):
                        new_sql = f"SELECT {group_col}, ROUND({agg_expr}, 2) FROM {target_table} GROUP BY {group_col}"
                        return (new_sql,
                                f'Missing GROUP BY {group_col}',
                                f'Added GROUP BY {group_col}, switched to {target_table}')

        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_part = select_match.group(1)
            if group_col.upper() not in sql.upper():
                new_select = f"{group_col}, {select_part}"
                sql = sql.replace(select_part, new_select, 1)

        if 'ORDER BY' in sql.upper():
            sql = re.sub(r'(ORDER BY)', f'GROUP BY {group_col} \\1', sql, count=1, flags=re.IGNORECASE)
        elif 'LIMIT' in sql.upper():
            sql = re.sub(r'(LIMIT)', f'GROUP BY {group_col} \\1', sql, count=1, flags=re.IGNORECASE)
        else:
            sql += f' GROUP BY {group_col}'

        return (sql,
                f'Missing GROUP BY {group_col}',
                f'Added GROUP BY {group_col}')

    def _fix_group_by_column(self, sql: str, correct_col: str, q: str) -> Optional[Tuple[str, str, str]]:
        group_match = re.search(r'GROUP BY\s+(\w+)', sql, re.IGNORECASE)
        if group_match:
            wrong_col = group_match.group(1)
            if wrong_col.upper() != correct_col.upper():
                new_sql = sql.replace(wrong_col, correct_col)
                return (new_sql,
                        f'Wrong GROUP BY: {wrong_col} should be {correct_col}',
                        f'Fixed GROUP BY column')
        return None


    def _check_table_routing(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        question_table = None
        question_confidence = 0

        for table, words in TABLE_ENTITY_WORDS.items():
            for word in words:
                if word in q:
                    conf = len(word)
                    if conf > question_confidence:
                        question_table = table
                        question_confidence = conf

        if not question_table:
            return None

        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            sql_table = from_match.group(1).lower()
            if sql_table != question_table:
                if question_table in self._table_columns:
                    sql_table_words = TABLE_ENTITY_WORDS.get(sql_table, [])
                    if not any(w in q for w in sql_table_words):
                        return self._fix_table_routing(sql, sql_table, question_table, q)
        return None

    def _fix_table_routing(self, sql: str, wrong_table: str, correct_table: str,
                           q: str) -> Optional[Tuple[str, str, str]]:
        correct_cols = self._table_columns.get(correct_table, set())

        new_sql = re.sub(rf'\b{wrong_table}\b', correct_table, sql, flags=re.IGNORECASE)

        col_pattern = re.findall(r'(?:SELECT|WHERE|GROUP BY|ORDER BY)\s+.*?(?:FROM|WHERE|GROUP|ORDER|LIMIT|$)',
                                 new_sql, re.IGNORECASE | re.DOTALL)

        return (new_sql,
                f'Wrong table: querying {wrong_table} instead of {correct_table}',
                f'Swapped table to {correct_table}')


    def _check_column_existence(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if not from_match:
            return None

        table = from_match.group(1).lower()
        if table not in self._table_columns:
            return None

        valid_cols = self._table_columns[table]

        sql_cols = set()
        for match in re.finditer(r'\b([A-Z][A-Z_0-9]+)\b', sql):
            word = match.group(1)
            if word not in ('SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER',
                            'ASC', 'DESC', 'LIMIT', 'AND', 'OR', 'NOT', 'NULL',
                            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'IN',
                            'LIKE', 'BETWEEN', 'JOIN', 'ON', 'LEFT', 'RIGHT',
                            'INNER', 'OUTER', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
                            'ROUND', 'CAST', 'REAL', 'INTEGER', 'TEXT', 'DISTINCT',
                            'HAVING', 'UNION', 'EXCEPT', 'INTERSECT', 'EXISTS',
                            'DENIED', 'PAID', 'PENDING', 'APPROVED', 'HMO', 'PPO',
                            'INPATIENT', 'OUTPATIENT', 'EMERGENCY', 'TELEHEALTH',
                            'MALE', 'FEMALE', 'FILLED', 'CANCELLED', 'COMPLETED'):
                sql_cols.add(word)

        missing_cols = sql_cols - valid_cols
        if missing_cols:
            all_valid = set()
            for t, cols in self._table_columns.items():
                all_valid |= cols
            truly_missing = missing_cols - all_valid
            if truly_missing:
                logger.warning("Columns not found in any table: %s", truly_missing)

        return None


    def _check_superlative_correctness(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        if 'GROUP BY' in sql_upper:
            return None

        is_max_query = any(w in q for w in ['highest', 'maximum', 'biggest', 'largest', 'greatest', 'most expensive'])
        is_min_query = any(w in q for w in ['lowest', 'minimum', 'smallest', 'least', 'cheapest'])

        if not is_max_query and not is_min_query:
            return None

        q_words = set(q.split())
        if q_words & GROUPABLE_ENTITIES:
            has_by = bool(re.search(r'\bby\s+\w+', q))
            has_which_what = bool(re.search(r'\b(which|what|who)\b', q))
            if has_by or has_which_what:
                return None
            for sup in ['highest', 'lowest', 'maximum', 'minimum', 'biggest', 'smallest']:
                if sup in q:
                    after_sup = q[q.index(sup) + len(sup):].strip().split()
                    if after_sup and after_sup[-1] in GROUPABLE_ENTITIES:
                        return None

        target_agg = 'MAX' if is_max_query else 'MIN'

        if target_agg not in sql_upper and 'ORDER BY' in sql_upper and 'LIMIT' in sql_upper:
            target_col = None
            for phrase, col in sorted(PHRASE_TO_COLUMN.items(), key=lambda x: -len(x[0])):
                if phrase in q:
                    target_col = col
                    break

            if not target_col:
                order_match = re.search(r'ORDER\s+BY\s+(\w+)', sql, re.IGNORECASE)
                if order_match:
                    target_col = order_match.group(1)

            if target_col:
                from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                table = from_match.group(1) if from_match else 'claims'
                where_match = re.search(r'(WHERE\s+.+?)(?:ORDER BY|LIMIT|$)', sql, re.IGNORECASE)
                where_part = f" {where_match.group(1).strip()}" if where_match else ''
                new_sql = f"SELECT {target_agg}(CAST({target_col} AS REAL)) AS {target_agg.lower()}_{target_col.lower()} FROM {table}{where_part}"
                return (new_sql,
                        f'ORDER BY+LIMIT used instead of {target_agg}({target_col})',
                        f'Rewrote to proper {target_agg}() — ORDER BY+LIMIT is NOT equivalent')

        if target_agg not in sql_upper:
            target_col = None
            for phrase, col in sorted(PHRASE_TO_COLUMN.items(), key=lambda x: -len(x[0])):
                if phrase in q:
                    target_col = col
                    break

            if target_col:
                from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                table = from_match.group(1) if from_match else 'claims'
                for t, cols in self._table_columns.items():
                    if target_col in cols:
                        table = t
                        break
                where_match = re.search(r'(WHERE\s+.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE)
                where_part = f" {where_match.group(1).strip()}" if where_match else ''
                new_sql = f"SELECT {target_agg}(CAST({target_col} AS REAL)) AS {target_agg.lower()}_{target_col.lower()} FROM {table}{where_part}"
                return (new_sql,
                        f'Missing {target_agg}({target_col}) for superlative query',
                        f'Generated correct {target_agg}() query')

        return None


    def _check_filter_completeness(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()
        filters_needed = []

        if 'denied' in q and 'referral' not in q and 'DENIED' not in sql_upper:
            filters_needed.append(("CLAIM_STATUS = 'DENIED'", 'claims'))
        if 'pending' in q and 'PENDING' not in sql_upper:
            filters_needed.append(("CLAIM_STATUS = 'PENDING'", 'claims'))
        if 'inpatient' in q and 'INPATIENT' not in sql_upper:
            filters_needed.append(("VISIT_TYPE = 'INPATIENT'", 'encounters'))
        if 'outpatient' in q and 'OUTPATIENT' not in sql_upper:
            filters_needed.append(("VISIT_TYPE = 'OUTPATIENT'", 'encounters'))
        if 'emergency' in q and ('ER' not in sql_upper and 'EMERGENCY' not in sql_upper):
            filters_needed.append(("VISIT_TYPE = 'EMERGENCY'", 'encounters'))
        if 'telehealth' in q and 'TELEHEALTH' not in sql_upper:
            filters_needed.append(("VISIT_TYPE = 'TELEHEALTH'", 'encounters'))
        if 'female' in q and 'F' not in sql_upper and 'FEMALE' not in sql_upper:
            filters_needed.append(("GENDER = 'F'", 'members'))
        if 'male' in q and not 'female' in q and "'M'" not in sql_upper and 'MALE' not in sql_upper:
            filters_needed.append(("GENDER = 'M'", 'members'))
        if 'chronic' in q and 'CHRONIC' not in sql_upper and 'IS_CHRONIC' not in sql_upper:
            filters_needed.append(("IS_CHRONIC = 'Y'", 'diagnoses'))
        if 'filled' in q and 'prescription' in q and 'FILLED' not in sql_upper:
            filters_needed.append(("STATUS = 'FILLED'", 'prescriptions'))
        if 'hmo' in q and 'HMO' not in sql_upper:
            filters_needed.append(("PLAN_TYPE = 'HMO'", 'claims'))

        if any(w in q for w in ['rate', 'percentage', 'percent', 'ratio', 'proportion']):
            filters_needed = []

        if any(w in q for w in ['by status', 'breakdown', 'distribution', 'by type']):
            filters_needed = []

        if not filters_needed:
            return None

        filter_clause, expected_table = filters_needed[0]

        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match and from_match.group(1).lower() == expected_table:
            if 'WHERE' in sql_upper:
                new_sql = re.sub(r'(WHERE\s+)', f'WHERE {filter_clause} AND ', sql, count=1, flags=re.IGNORECASE)
            else:
                new_sql = re.sub(r'(FROM\s+\w+)', f'\\1 WHERE {filter_clause}', sql, count=1, flags=re.IGNORECASE)
            return (new_sql,
                    f'Missing filter: {filter_clause}',
                    f'Added WHERE {filter_clause}')

        return None


    def _check_join_safety(self, q: str, sql: str) -> Optional[Tuple[str, str, str]]:
        sql_upper = sql.upper()

        if 'COUNT' not in sql_upper or 'JOIN' not in sql_upper:
            return None

        simple_count_patterns = [
            r'how many (\w+)',
            r'count (?:of |all )?(\w+)',
            r'total number of (\w+)',
            r'number of (\w+)',
        ]

        for pattern in simple_count_patterns:
            match = re.search(pattern, q)
            if match:
                entity = match.group(1)
                for table, words in TABLE_ENTITY_WORDS.items():
                    if entity in words or entity.rstrip('s') in words:
                        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                        if from_match and from_match.group(1).lower() == table:
                            new_sql = f"SELECT COUNT(*) FROM {table}"
                            where_match = re.search(r'(WHERE\s+.+?)(?:GROUP BY|ORDER BY|JOIN|LIMIT|$)',
                                                    sql, re.IGNORECASE)
                            if where_match:
                                where_text = where_match.group(1).strip()
                                where_text = re.sub(r'\w+\.', '', where_text)
                                new_sql += f" {where_text}"
                            if sql != new_sql:
                                return (new_sql,
                                        'Unnecessary JOIN inflating COUNT',
                                        f'Simplified to COUNT(*) FROM {table}')
                        break

        return None


    def get_correction_patterns(self) -> List[Dict]:
        return list(self._correction_log)

    def get_stats(self) -> Dict[str, int]:
        if not self._correction_log:
            return {'total_corrections': 0}

        violation_types = {}
        for entry in self._correction_log:
            for v in entry['violations']:
                vtype = v.split(':')[0].strip()
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

        return {
            'total_corrections': len(self._correction_log),
            'violation_types': violation_types,
        }
