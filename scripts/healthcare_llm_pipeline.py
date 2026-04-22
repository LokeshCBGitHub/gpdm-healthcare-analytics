import os
import re
import json
import time
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger('gpdm.healthcare_llm')


class HealthcareDomainKnowledge:

    CLINICAL_CONCEPTS = {
        'readmission': 'Patient returning to hospital within 30 days of discharge',
        'length of stay': 'Number of days from admission to discharge (encounters.LENGTH_OF_STAY)',
        'acuity': 'Severity/complexity of patient condition',
        'comorbidity': 'Multiple co-existing conditions (diagnoses.IS_CHRONIC)',
        'HCC': 'Hierarchical Condition Category — CMS risk adjustment model (diagnoses.HCC_CODE)',
        'HEDIS': 'Healthcare Effectiveness Data and Information Set — quality measures',
        'PCP': 'Primary Care Physician (members.PCP_NPI → providers.NPI)',
        'NPI': 'National Provider Identifier — unique 10-digit provider ID',
        'ICD-10': 'International Classification of Diseases, 10th revision (diagnoses.ICD10_CODE)',
        'CPT': 'Current Procedural Terminology — procedure codes (claims.CPT_CODE)',
        'NDC': 'National Drug Code — medication identifier (prescriptions.NDC_CODE)',
        'formulary': 'Approved medication list for the health plan',
        'prior authorization': 'Pre-approval required before service delivery',
        'adjudication': 'Claims processing and payment determination',
        'capitation': 'Fixed per-member payment to providers',
        'fee-for-service': 'Payment per individual service rendered',
    }

    FINANCIAL_METRICS = {
        'PMPM': 'Per Member Per Month cost = SUM(PAID_AMOUNT) / COUNT(DISTINCT MEMBER_ID)',
        'MLR': 'Medical Loss Ratio = medical costs / premium revenue',
        'denial rate': 'Percentage of claims denied = denied / total claims',
        'clean claim rate': 'Claims processed without manual intervention',
        'days in AR': 'Average days from service to payment',
        'cost per encounter': 'Total cost divided by number of encounters',
        'PMPY': 'Per Member Per Year cost',
        'allowed amount': 'Maximum amount the plan will pay (claims.ALLOWED_AMOUNT)',
        'member responsibility': 'Copay + coinsurance + deductible (claims.MEMBER_RESPONSIBILITY)',
    }

    KP_CONTEXT = {
        'regions': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'HI', 'GA', 'WA', 'MID'],
        'plan_types': ['HMO', 'PPO', 'POS', 'Medicare Advantage', 'Medicaid'],
        'visit_types': ['INPATIENT', 'OUTPATIENT', 'EMERGENCY', 'TELEHEALTH', 'OBSERVATION', 'URGENT'],
        'claim_statuses': ['APPROVED', 'DENIED', 'PENDING', 'PARTIALLY_APPROVED'],
    }

    HEALTHCARE_PATTERNS = {
        'readmission rate': {'intent': 'rate', 'tables': ['encounters'], 'concept': 'readmission'},
        'no-show rate': {'intent': 'rate', 'tables': ['appointments'], 'filter': ('STATUS', '=', 'NO_SHOW')},
        'denial rate': {'intent': 'rate', 'tables': ['claims'], 'filter': ('CLAIM_STATUS', '=', 'DENIED')},
        'PMPM': {'intent': 'aggregate', 'sub_intent': 'per_unit', 'tables': ['claims', 'members']},
        'risk score': {'intent': 'aggregate', 'tables': ['members'], 'column': 'RISK_SCORE'},
        'chronic conditions': {'intent': 'count', 'tables': ['diagnoses'], 'filter': ('IS_CHRONIC', '=', 'Y')},
        'care gaps': {'intent': 'list', 'tables': ['diagnoses', 'members'], 'concept': 'preventive_care'},
        'star rating': {'intent': 'summary', 'tables': ['members', 'claims'], 'concept': 'quality_metrics'},
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self._schema_summary = None
        self._sample_values = {}

    def build_schema_context(self, schema_graph) -> str:
        if self._schema_summary:
            return self._schema_summary

        parts = []
        parts.append("═══ DATABASE SCHEMA (Kaiser Permanente Medicare Advantage) ═══")
        parts.append("")

        for table_name, concept in schema_graph.tables.items():
            cols = schema_graph.columns.get(table_name, {})
            col_details = []
            for col_name, sem_type in cols.items():
                type_hint = ''
                if sem_type.is_money:
                    type_hint = ' [MONEY]'
                elif sem_type.is_date:
                    type_hint = ' [DATE]'
                elif sem_type.is_identifier:
                    type_hint = ' [ID]'
                elif sem_type.groupable:
                    type_hint = ' [GROUPABLE]'
                elif sem_type.aggregatable:
                    type_hint = ' [NUMERIC]'
                col_details.append(f"    {col_name}{type_hint}")

            parts.append(f"TABLE: {table_name} — {concept.description}")
            parts.append(f"  Primary Key: {concept.primary_key}")
            if concept.primary_date:
                parts.append(f"  Date Column: {concept.primary_date}")
            parts.append("  Columns:")
            parts.extend(col_details)
            parts.append("")

        parts.append("═══ JOIN RELATIONSHIPS ═══")
        for jp in schema_graph.join_paths:
            parts.append(f"  {jp.from_table}.{jp.from_col} → {jp.to_table}.{jp.to_col}")

        parts.append("")
        parts.append("═══ IMPORTANT DOMAIN RULES ═══")
        parts.append("  - For money: always CAST(column AS REAL) and ROUND(..., 2)")
        parts.append("  - For counts with JOINs: use COUNT(DISTINCT id_column)")
        parts.append("  - For PMPM: SUM(PAID_AMOUNT) / COUNT(DISTINCT MEMBER_ID)")
        parts.append("  - For denial rate: 100.0 * SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) / COUNT(*)")
        parts.append("  - For trends: SUBSTR(date_col, 1, 7) for monthly, SUBSTR(date_col, 1, 4) for yearly")
        parts.append("  - For 'which X' (single answer): LIMIT 1. For 'top N': LIMIT N.")
        parts.append("  - SQLite syntax: no ILIKE (use LOWER()), no LIMIT..OFFSET without ORDER BY")
        parts.append("  - Use table prefixes when JOINing (e.g., claims.MEMBER_ID)")

        self._schema_summary = '\n'.join(parts)
        return self._schema_summary

    def get_domain_hints(self, question: str) -> Dict[str, Any]:
        q_lower = question.lower()
        hints = {'concepts': [], 'metrics': [], 'patterns': []}

        for concept, desc in self.CLINICAL_CONCEPTS.items():
            if concept.lower() in q_lower:
                hints['concepts'].append((concept, desc))

        for metric, formula in self.FINANCIAL_METRICS.items():
            if metric.lower() in q_lower:
                hints['metrics'].append((metric, formula))

        for pattern, spec in self.HEALTHCARE_PATTERNS.items():
            if pattern.lower() in q_lower:
                hints['patterns'].append((pattern, spec))

        return hints

    def enrich_prompt_with_domain(self, question: str) -> str:
        hints = self.get_domain_hints(question)
        if not any(hints.values()):
            return ''

        parts = ["\n═══ DOMAIN CONTEXT FOR THIS QUESTION ═══"]
        for concept, desc in hints['concepts']:
            parts.append(f"  {concept}: {desc}")
        for metric, formula in hints['metrics']:
            parts.append(f"  {metric}: {formula}")
        for pattern, spec in hints['patterns']:
            parts.append(f"  Pattern '{pattern}': intent={spec.get('intent')}, tables={spec.get('tables')}")

        return '\n'.join(parts)

    def load_sample_values(self, db_path: str):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            samples = {
                'KP_REGION': "SELECT DISTINCT KP_REGION FROM claims LIMIT 10",
                'PLAN_TYPE': "SELECT DISTINCT PLAN_TYPE FROM claims LIMIT 10",
                'VISIT_TYPE': "SELECT DISTINCT VISIT_TYPE FROM encounters LIMIT 10",
                'CLAIM_STATUS': "SELECT DISTINCT CLAIM_STATUS FROM claims LIMIT 10",
                'SPECIALTY': "SELECT DISTINCT SPECIALTY FROM providers LIMIT 20",
                'MEDICATION_CLASS': "SELECT DISTINCT MEDICATION_CLASS FROM prescriptions LIMIT 15",
                'CLAIM_TYPE': "SELECT DISTINCT CLAIM_TYPE FROM claims LIMIT 10",
            }
            for col, sql in samples.items():
                try:
                    rows = cur.execute(sql).fetchall()
                    self._sample_values[col] = [r[0] for r in rows if r[0]]
                except:
                    pass

            conn.close()
        except Exception as e:
            logger.warning("Failed to load sample values: %s", e)

    def get_value_context(self) -> str:
        if not self._sample_values:
            return ''
        parts = ["\n═══ VALID COLUMN VALUES ═══"]
        for col, vals in self._sample_values.items():
            parts.append(f"  {col}: {', '.join(str(v) for v in vals[:10])}")
        return '\n'.join(parts)


class LLMIntentEngine:

    SYSTEM_PROMPT = """You are a healthcare analytics NLU engine for Kaiser Permanente Medicare Advantage.
Your job is to parse natural language questions into structured intent for SQL generation.

You MUST return ONLY valid JSON — no explanation, no markdown, no extra text.

INTENT TYPES (pick exactly one):
- count: "how many X" → COUNT query
- aggregate: "total/average/sum of X" → aggregate function query
- rank: "top N X" or "which X has most Y" → GROUP BY + ORDER BY + LIMIT
- trend: "over time" / "per month" / "monthly" → date-grouped time series
- rate: "percentage" / "denial rate" / "what % of" → ratio/percentage calculation
- compare: "X vs Y" / "compare A to B" → side-by-side comparison
- list: "show me" / "list" / "which members" → SELECT with details
- summary: "overview" / "summary" / "everything about" → multi-metric summary
- correlate: "relationship between" / "correlation" → statistical relationship

RULES:
- Tables MUST be real table names from the schema
- Columns MUST be real column names from the schema
- For "breakdown" or "distribution" → intent=aggregate with group_by
- For "per month" / "monthly" / "by month" → intent=trend
- For "most" / "top" / "highest" / "which X" → intent=rank
- For follow-up questions, infer missing context from conversation history
- filter values must match actual database values (e.g., 'DENIED' not 'denied')
"""

    def __init__(self, model_loader, schema_graph, domain: HealthcareDomainKnowledge):
        self.model = model_loader
        self.graph = schema_graph
        self.domain = domain
        self._schema_context = domain.build_schema_context(schema_graph)
        self._value_context = domain.get_value_context()
        self._parse_count = 0
        self._success_count = 0

    def parse(self, question: str, conversation_history: List[Dict] = None) -> Optional[Dict]:
        self._parse_count += 1

        if not self.model or not self.model.is_loaded:
            return None

        prompt = self._build_prompt(question, conversation_history)

        try:
            t0 = time.time()
            response = self.model.generate(prompt, max_tokens=600, temperature=0.05)
            elapsed = time.time() - t0
            logger.info("LLM NLU: %.2fs for '%s'", elapsed, question[:60])

            intent_data = self._extract_json(response)
            if not intent_data:
                logger.warning("LLM NLU returned invalid JSON: %s", response[:200])
                return None

            validated = self._validate_intent(intent_data, question)
            if validated:
                self._success_count += 1
                return validated

            return None

        except Exception as e:
            logger.error("LLM NLU failed: %s", e)
            return None

    def _build_prompt(self, question: str, history: List[Dict] = None) -> str:
        parts = [self.SYSTEM_PROMPT]
        parts.append("")
        parts.append(self._schema_context)
        if self._value_context:
            parts.append(self._value_context)

        domain_hints = self.domain.enrich_prompt_with_domain(question)
        if domain_hints:
            parts.append(domain_hints)

        if history:
            parts.append("\n═══ CONVERSATION HISTORY (for follow-up context) ═══")
            for turn in history[-3:]:
                parts.append(f"  Q: {turn.get('question', '')}")
                parts.append(f"  → intent={turn.get('intent', '')}, tables={turn.get('tables', [])}")

        parts.append(self._get_few_shot_examples())

        parts.append(f'\nQuestion: "{question}"')
        parts.append("\nReturn ONLY the JSON object:")

        return '\n'.join(parts)

    def _get_few_shot_examples(self) -> str:
        return """
═══ EXAMPLES ═══
Q: "how many denied claims do we have?"
{"intent": "count", "tables": ["claims"], "filters": [{"column": "CLAIM_STATUS", "operator": "=", "value": "DENIED", "table": "claims"}], "group_by": [], "agg_function": "COUNT", "agg_column": "", "agg_table": "", "order_by": "", "limit": null, "temporal": false}

Q: "what is the average paid amount per claim?"
{"intent": "aggregate", "tables": ["claims"], "filters": [], "group_by": [], "agg_function": "AVG", "agg_column": "PAID_AMOUNT", "agg_table": "claims", "order_by": "", "limit": null, "temporal": false}

Q: "top 5 specialties by claim count"
{"intent": "rank", "tables": ["claims", "providers"], "filters": [], "group_by": [["providers", "SPECIALTY"]], "agg_function": "COUNT", "agg_column": "", "agg_table": "", "order_by": "desc", "limit": 5, "temporal": false}

Q: "claims per month in 2024"
{"intent": "trend", "tables": ["claims"], "filters": [{"column": "SERVICE_DATE", "operator": "LIKE", "value": "2024%", "table": "claims"}], "group_by": [], "agg_function": "COUNT", "agg_column": "", "agg_table": "", "order_by": "", "limit": null, "temporal": true, "time_granularity": "month"}

Q: "denial rate by region"
{"intent": "rate", "tables": ["claims"], "filters": [{"column": "CLAIM_STATUS", "operator": "=", "value": "DENIED", "table": "claims"}], "group_by": [["claims", "KP_REGION"]], "agg_function": "COUNT", "agg_column": "", "agg_table": "", "order_by": "", "limit": null, "temporal": false}

Q: "top medications by cost"
{"intent": "rank", "tables": ["prescriptions"], "filters": [], "group_by": [["prescriptions", "MEDICATION_NAME"]], "agg_function": "SUM", "agg_column": "COST", "agg_table": "prescriptions", "order_by": "desc", "limit": 10, "temporal": false}
"""

    def _extract_json(self, response: str) -> Optional[Dict]:
        if not response:
            return None

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        depth = 0
        start = None
        for i, ch in enumerate(response):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(response[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    start = None

        return None

    def _validate_intent(self, data: Dict, question: str) -> Optional[Dict]:
        valid_intents = {'count', 'aggregate', 'rank', 'trend', 'rate',
                         'compare', 'list', 'summary', 'correlate', 'exists'}

        intent = data.get('intent', '')
        if intent not in valid_intents:
            logger.warning("LLM returned invalid intent: %s", intent)
            return None

        tables = data.get('tables', [])
        valid_tables = [t for t in tables if t in self.graph.tables]
        if not valid_tables and tables:
            logger.warning("LLM returned no valid tables: %s", tables)
            return None
        data['tables'] = valid_tables or ['claims']

        validated_gb = []
        for gb in data.get('group_by', []):
            if isinstance(gb, list) and len(gb) == 2:
                tbl, col = gb
                if col in self.graph.columns.get(tbl, {}):
                    validated_gb.append(gb)
                else:
                    for t in self.graph.columns:
                        if col in self.graph.columns[t]:
                            validated_gb.append([t, col])
                            break
            elif isinstance(gb, str):
                for t in self.graph.columns:
                    if gb in self.graph.columns[t]:
                        validated_gb.append([t, gb])
                        break
        data['group_by'] = validated_gb

        agg_col = data.get('agg_column', '')
        agg_table = data.get('agg_table', '')
        if agg_col:
            if agg_table and agg_col not in self.graph.columns.get(agg_table, {}):
                for t in data['tables']:
                    if agg_col in self.graph.columns.get(t, {}):
                        data['agg_table'] = t
                        break
            elif not agg_table:
                for t in data['tables']:
                    if agg_col in self.graph.columns.get(t, {}):
                        data['agg_table'] = t
                        break

        validated_filters = []
        for f in data.get('filters', []):
            if isinstance(f, dict) and 'column' in f:
                col = f['column']
                tbl = f.get('table', '')
                if tbl and col in self.graph.columns.get(tbl, {}):
                    validated_filters.append(f)
                else:
                    for t in self.graph.columns:
                        if col in self.graph.columns[t]:
                            f['table'] = t
                            validated_filters.append(f)
                            break
        data['filters'] = validated_filters

        return data

    @property
    def stats(self) -> Dict:
        return {
            'total': self._parse_count,
            'success': self._success_count,
            'rate': self._success_count / max(self._parse_count, 1),
        }


class LLMSQLEngine:

    SQL_SYSTEM_PROMPT = """You are a healthcare analytics SQL expert for Kaiser Permanente.
Generate a single, correct SQLite SQL query for the given question.

CRITICAL RULES:
1. Use ONLY tables and columns from the provided schema
2. SQLite syntax: SUBSTR not SUBSTRING, || for concat, no ILIKE
3. For money columns: ROUND(SUM(CAST(col AS REAL)), 2)
4. For counts with JOINs: COUNT(DISTINCT id_column) to avoid inflation
5. Always use table prefixes in JOINed queries (e.g., claims.MEMBER_ID)
6. For "which X has THE most Y" → LIMIT 1. For "top N" → LIMIT N. Default LIMIT 10.
7. For trends: GROUP BY SUBSTR(date_col, 1, 7) ORDER BY period
8. For rates: 100.0 * SUM(CASE WHEN condition THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
9. Never use SELECT * in production queries
10. Filter values must be UPPERCASE for status columns (DENIED, APPROVED, etc.)

Return ONLY the SQL query — no explanation, no markdown.
"""

    def __init__(self, model_loader, schema_graph, domain: HealthcareDomainKnowledge,
                 db_path: str):
        self.model = model_loader
        self.graph = schema_graph
        self.domain = domain
        self.db_path = db_path
        self._schema_context = domain.build_schema_context(schema_graph)
        self._gen_count = 0
        self._valid_count = 0

    def generate_sql(self, question: str, intent_data: Dict = None) -> Optional[str]:
        self._gen_count += 1

        if not self.model or not self.model.is_loaded:
            return None

        prompt = self._build_sql_prompt(question, intent_data)

        try:
            t0 = time.time()
            response = self.model.generate(prompt, max_tokens=400, temperature=0.05)
            elapsed = time.time() - t0
            logger.info("LLM SQL: %.2fs for '%s'", elapsed, question[:60])

            sql = self._extract_sql(response)
            if not sql:
                logger.warning("LLM SQL returned no valid SQL: %s", response[:200])
                return None

            validated_sql = self._validate_sql(sql)
            if validated_sql:
                self._valid_count += 1
                return validated_sql

            return None

        except Exception as e:
            logger.error("LLM SQL generation failed: %s", e)
            return None

    def _build_sql_prompt(self, question: str, intent_data: Dict = None) -> str:
        parts = [self.SQL_SYSTEM_PROMPT]
        parts.append("")
        parts.append(self._schema_context)

        domain_hints = self.domain.enrich_prompt_with_domain(question)
        if domain_hints:
            parts.append(domain_hints)

        if intent_data:
            parts.append(f"\nPARSED INTENT (use as guidance):")
            parts.append(f"  Intent: {intent_data.get('intent', 'unknown')}")
            if intent_data.get('tables'):
                parts.append(f"  Tables: {intent_data['tables']}")
            if intent_data.get('agg_column'):
                parts.append(f"  Aggregation: {intent_data.get('agg_function', 'SUM')}({intent_data['agg_column']})")
            if intent_data.get('group_by'):
                parts.append(f"  Group by: {intent_data['group_by']}")
            if intent_data.get('filters'):
                parts.append(f"  Filters: {intent_data['filters']}")
            if intent_data.get('limit'):
                parts.append(f"  Limit: {intent_data['limit']}")

        parts.append(self._get_sql_examples())

        parts.append(f'\nQuestion: "{question}"')
        parts.append("\nSQL:")

        return '\n'.join(parts)

    def _get_sql_examples(self) -> str:
        return """
═══ SQL EXAMPLES ═══
Q: "how many members?"
SQL: SELECT COUNT(DISTINCT MEMBER_ID) AS total FROM members

Q: "total paid amount"
SQL: SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)), 2) AS total_paid FROM claims

Q: "denial rate by region"
SQL: SELECT KP_REGION, COUNT(*) AS total, SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count, ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denial_rate FROM claims GROUP BY KP_REGION ORDER BY denial_rate DESC

Q: "claims per month"
SQL: SELECT SUBSTR(SERVICE_DATE, 1, 7) AS period, COUNT(*) AS cnt FROM claims WHERE SERVICE_DATE IS NOT NULL AND SERVICE_DATE != '' GROUP BY SUBSTR(SERVICE_DATE, 1, 7) ORDER BY period

Q: "top 5 medications by cost"
SQL: SELECT MEDICATION_NAME, ROUND(SUM(CAST(COST AS REAL)), 2) AS total_cost FROM prescriptions GROUP BY MEDICATION_NAME ORDER BY total_cost DESC LIMIT 5

Q: "which providers have the most claims?"
SQL: SELECT providers.NPI, providers.SPECIALTY, COUNT(DISTINCT claims.CLAIM_ID) AS claim_count FROM claims INNER JOIN providers ON claims.RENDERING_NPI = providers.NPI GROUP BY providers.NPI, providers.SPECIALTY ORDER BY claim_count DESC LIMIT 10
"""

    def _extract_sql(self, response: str) -> Optional[str]:
        if not response:
            return None

        response = response.strip()

        response = re.sub(r'```sql\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = response.strip()

        match = re.search(r'(SELECT\s+.+?)(?:;|\Z)', response, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            sql = ' '.join(sql.split())
            return sql

        return None

    def _validate_sql(self, sql: str) -> Optional[str]:
        sql_upper = sql.upper()

        table_pattern = r'\b(?:FROM|JOIN)\s+(\w+)\b'
        referenced_tables = re.findall(table_pattern, sql, re.IGNORECASE)
        for t in referenced_tables:
            t_lower = t.lower()
            if t_lower not in self.graph.tables and t_lower not in ('_dummy', 'dual'):
                logger.warning("LLM SQL references unknown table: %s", t)
                return None

        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            test_sql = f"SELECT * FROM ({sql}) AS _test LIMIT 0"
            cur.execute(test_sql)
            conn.close()
            return sql
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            logger.warning("LLM SQL validation failed: %s | SQL: %s", error_msg, sql[:200])

            fixed_sql = self._try_auto_fix(sql, error_msg)
            if fixed_sql:
                return fixed_sql

            return None
        except Exception as e:
            logger.warning("LLM SQL validation error: %s", e)
            return None

    def _try_auto_fix(self, sql: str, error: str) -> Optional[str]:
        match = re.search(r'no such column:\s*(\w+\.)?(\w+)', error)
        if match:
            bad_col = match.group(2)
            fixed = re.sub(rf'\b\w*\.?{bad_col}\b', 'NULL', sql, count=1)
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute(f"SELECT * FROM ({fixed}) AS _test LIMIT 0")
                conn.close()
                return fixed
            except:
                pass

        match = re.search(r'ambiguous column name:\s*(\w+)', error)
        if match:
            col = match.group(1)
            for t in self.graph.columns:
                if col in self.graph.columns[t]:
                    fixed = re.sub(rf'\b{col}\b(?!\.)', f'{t}.{col}', sql)
                    try:
                        conn = sqlite3.connect(self.db_path)
                        conn.execute(f"SELECT * FROM ({fixed}) AS _test LIMIT 0")
                        conn.close()
                        return fixed
                    except:
                        pass

        return None

    @property
    def stats(self) -> Dict:
        return {
            'total': self._gen_count,
            'valid': self._valid_count,
            'rate': self._valid_count / max(self._gen_count, 1),
        }


class LLMNarrativeEngine:

    NARRATIVE_SYSTEM_PROMPT = """You are a senior healthcare analyst at Kaiser Permanente, writing executive-level summaries.

Given a SQL query, its results, and the original question, write a concise, insightful narrative.

STYLE:
- Start with the direct answer to the question (1 sentence)
- Add 1-2 sentences of context or implications
- If relevant, add a brief recommendation or next step
- Use healthcare terminology appropriately
- Be concise — no more than 4-5 sentences total
- Format numbers with commas (e.g., 25,000 not 25000)
- For money, use $ with 2 decimal places
- For percentages, use 1 decimal place

DO NOT:
- Say "based on the data" or "according to the query"
- Repeat the SQL query
- Use bullet points or markdown
- Be vague — give specific numbers and context
"""

    def __init__(self, model_loader, domain: HealthcareDomainKnowledge):
        self.model = model_loader
        self.domain = domain

    def generate_narrative(self, question: str, sql: str, rows: List,
                           columns: List[str] = None) -> Optional[str]:
        if not self.model or not self.model.is_loaded:
            return None

        prompt = self._build_narrative_prompt(question, sql, rows, columns)

        try:
            t0 = time.time()
            response = self.model.generate(prompt, max_tokens=300, temperature=0.3)
            elapsed = time.time() - t0
            logger.info("LLM narrative: %.2fs", elapsed)

            if response and len(response) > 20:
                narrative = response.strip()
                narrative = re.sub(r'^(?:Answer|Response|Narrative|Summary):\s*', '', narrative)
                return narrative

            return None

        except Exception as e:
            logger.error("LLM narrative failed: %s", e)
            return None

    def _build_narrative_prompt(self, question: str, sql: str, rows: List,
                                 columns: List[str] = None) -> str:
        parts = [self.NARRATIVE_SYSTEM_PROMPT]

        parts.append(f'\nQuestion: "{question}"')

        if rows and len(rows) <= 20:
            parts.append(f"\nResults ({len(rows)} rows):")
            if columns:
                parts.append(f"  Columns: {', '.join(columns)}")
            for row in rows[:15]:
                parts.append(f"  {row}")
        elif rows:
            parts.append(f"\nResults: {len(rows)} rows (showing first 10)")
            if columns:
                parts.append(f"  Columns: {', '.join(columns)}")
            for row in rows[:10]:
                parts.append(f"  {row}")

        domain_hints = self.domain.enrich_prompt_with_domain(question)
        if domain_hints:
            parts.append(domain_hints)

        parts.append("\nWrite a concise narrative (3-5 sentences):")

        return '\n'.join(parts)


class ConversationManager:

    FOLLOWUP_PATTERNS = [
        r'^(?:and|but|also|what about|how about|now|ok)\b',
        r'^(?:show|break|split|group)\s+(?:it|that|this|them)\b',
        r'^by\s+\w+',
        r'^for\s+\w+\s+(?:only|specifically)',
        r'^(?:same|those|these)\b',
        r'^(?:and|or)\s+by\b',
        r'^what\s+(?:if|about)\b',
        r'^(?:can you|could you)\s+(?:also|break|split)',
    ]

    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, intent_data: Dict, sql: str, rows: List):
        self.history.append({
            'question': question,
            'intent': intent_data.get('intent', ''),
            'tables': intent_data.get('tables', []),
            'filters': intent_data.get('filters', []),
            'group_by': intent_data.get('group_by', []),
            'agg_column': intent_data.get('agg_column', ''),
            'sql': sql,
            'row_count': len(rows) if rows else 0,
            'timestamp': time.time(),
        })
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def is_followup(self, question: str) -> bool:
        if not self.history:
            return False
        q_lower = question.lower().strip()
        return any(re.search(p, q_lower) for p in self.FOLLOWUP_PATTERNS)

    def get_history_for_prompt(self) -> List[Dict]:
        return self.history[-3:] if self.history else []

    def enrich_followup(self, intent_data: Dict) -> Dict:
        if not self.history:
            return intent_data

        last = self.history[-1]

        if not intent_data.get('tables') and last.get('tables'):
            intent_data['tables'] = list(last['tables'])

        if not intent_data.get('filters') and last.get('filters'):
            intent_data['filters'] = list(last['filters'])

        if not intent_data.get('agg_column') and last.get('agg_column'):
            intent_data['agg_column'] = last['agg_column']

        return intent_data


class HealthcareLLMPipeline:

    def __init__(self, db_path: str, schema_graph=None, model_loader=None,
                 config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        self._init_start = time.time()

        if schema_graph:
            self.graph = schema_graph
        else:
            from schema_graph import SemanticSchemaGraph
            self.graph = SemanticSchemaGraph(db_path)

        self.domain = HealthcareDomainKnowledge(db_path)
        self.domain.load_sample_values(db_path)

        if model_loader:
            self.model = model_loader
        else:
            self.model = self._init_model()

        self.nlu_engine = LLMIntentEngine(self.model, self.graph, self.domain)
        self.sql_engine = LLMSQLEngine(self.model, self.graph, self.domain, db_path)
        self.narrative_engine = LLMNarrativeEngine(self.model, self.domain)

        self.conversation = ConversationManager()

        self._rule_parser = None
        self._sql_constructor = None
        self._init_fallbacks()

        elapsed = time.time() - self._init_start
        llm_status = f"model={self.model.model_name}" if self.model and self.model.is_loaded else "no model"
        logger.info("HealthcareLLMPipeline initialized in %.1fs (%s)", elapsed, llm_status)

    def _init_model(self):
        from local_llm_engine import LocalModelLoader, LLMConfig

        llm_config = LLMConfig(self.config)
        if not llm_config.enabled:
            logger.debug("External LLM not configured — using built-in intelligence engines")
            return None

        loader = LocalModelLoader(
            model_dir=llm_config.model_dir,
            allow_download=llm_config.allow_download,
        )

        available = loader.detect_available()
        if not available:
            logger.warning("No local models found — LLM features disabled")
            return None

        backends = loader.detect_backends()
        if not any(backends.values()):
            logger.warning("No inference backend — LLM features disabled")
            return None

        success = loader.load(preferred_model=llm_config.preferred_model)
        if success:
            logger.info("LLM loaded: %s via %s", loader.model_name, loader._backend)
            return loader
        else:
            logger.warning("Failed to load LLM model — using fallbacks")
            return None

    def _init_fallbacks(self):
        try:
            from intent_parser import IntentParser
            self._rule_parser = IntentParser(self.graph)
        except Exception as e:
            logger.warning("Rule-based parser unavailable: %s", e)

        try:
            from sql_constructor import SQLConstructor
            self._sql_constructor = SQLConstructor(self.graph, self.db_path)
        except Exception as e:
            logger.warning("SQL constructor unavailable: %s", e)

    @property
    def llm_available(self) -> bool:
        return self.model is not None and self.model.is_loaded

    def process(self, question: str, session_id: str = '') -> Dict[str, Any]:
        t_start = time.time()
        result = {
            'sql': '', 'rows': [], 'columns': [], 'narrative': '',
            'source': '', 'confidence': 0.0, 'intent': {},
            'error': None, 'timing': {},
        }

        t_nlu = time.time()
        intent_data = None
        nlu_source = 'none'

        history = self.conversation.get_history_for_prompt() if self.conversation.is_followup(question) else None

        if self.llm_available:
            intent_data = self.nlu_engine.parse(question, conversation_history=history)
            if intent_data:
                nlu_source = 'llm'
                if history:
                    intent_data = self.conversation.enrich_followup(intent_data)

        if not intent_data and self._rule_parser:
            parsed = self._rule_parser.parse(question)
            intent_data = self._parsed_intent_to_dict(parsed)
            nlu_source = 'rules'

        if not intent_data:
            result['error'] = 'Failed to parse question'
            return result

        result['intent'] = intent_data
        result['timing']['nlu_ms'] = (time.time() - t_nlu) * 1000

        t_sql = time.time()
        sql = None
        sql_source = 'none'

        if self.llm_available:
            sql = self.sql_engine.generate_sql(question, intent_data)
            if sql:
                sql_source = 'llm'

        if not sql and self._sql_constructor and self._rule_parser:
            try:
                if nlu_source == 'llm':
                    parsed = self._dict_to_parsed_intent(intent_data, question)
                else:
                    parsed = self._rule_parser.parse(question)

                construct_result = self._sql_constructor.construct(parsed)
                sql = construct_result.get('sql', '')
                sql_source = 'schema_graph'
            except Exception as e:
                logger.warning("SQL constructor fallback failed: %s", e)

        if not sql:
            result['error'] = 'Failed to generate SQL'
            result['source'] = f'nlu_{nlu_source}'
            return result

        result['sql'] = sql
        result['timing']['sql_ms'] = (time.time() - t_sql) * 1000

        t_exec = time.time()
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()

            result['rows'] = rows
            result['columns'] = columns
        except Exception as e:
            result['error'] = str(e)
            result['source'] = f'nlu_{nlu_source}+sql_{sql_source}'
            logger.warning("SQL execution failed: %s | SQL: %s", e, sql[:200])

            if sql_source == 'llm' and self._sql_constructor and self._rule_parser:
                try:
                    parsed = self._rule_parser.parse(question)
                    construct_result = self._sql_constructor.construct(parsed)
                    fallback_sql = construct_result.get('sql', '')
                    if fallback_sql:
                        conn = sqlite3.connect(self.db_path)
                        cur = conn.cursor()
                        cur.execute(fallback_sql)
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        conn.close()
                        result['sql'] = fallback_sql
                        result['rows'] = rows
                        result['columns'] = columns
                        result['error'] = None
                        sql_source = 'schema_graph_fallback'
                except Exception as fb_e:
                    logger.warning("Schema graph fallback also failed: %s", fb_e)

            if result['error']:
                return result

        result['timing']['exec_ms'] = (time.time() - t_exec) * 1000

        t_narr = time.time()
        narrative = None

        if self.llm_available and result['rows']:
            narrative = self.narrative_engine.generate_narrative(
                question, sql, result['rows'], result['columns']
            )
            if narrative:
                result['narrative'] = narrative

        if not narrative and result['rows']:
            result['narrative'] = self._template_narrative(
                question, result['rows'], result['columns'], intent_data
            )

        result['timing']['narrative_ms'] = (time.time() - t_narr) * 1000

        if self.conversation and result['rows']:
            self.conversation.add_turn(question, intent_data, sql, result['rows'])

        result['source'] = f'nlu_{nlu_source}+sql_{sql_source}'
        result['confidence'] = 0.9 if sql_source == 'llm' else 0.85
        result['timing']['total_ms'] = (time.time() - t_start) * 1000

        return result

    def _parsed_intent_to_dict(self, parsed) -> Dict:
        return {
            'intent': parsed.intent,
            'sub_intent': getattr(parsed, 'sub_intent', ''),
            'tables': parsed.tables,
            'agg_function': parsed.agg_function or '',
            'agg_column': parsed.agg_column or '',
            'agg_table': parsed.agg_table or '',
            'group_by': [[t, c] for t, c in parsed.group_by],
            'filters': [
                {'column': f.column, 'operator': f.operator,
                 'value': f.value, 'table': f.table_hint}
                for f in parsed.filters
            ],
            'order_by': parsed.order_by or '',
            'limit': parsed.limit,
            'temporal': parsed.temporal,
            'time_granularity': getattr(parsed, 'time_granularity', ''),
            'comparison': getattr(parsed, 'comparison', False),
            'compare_values': getattr(parsed, 'compare_values', []),
            'columns': [(t, c) for t, c in parsed.columns],
        }

    def _dict_to_parsed_intent(self, data: Dict, question: str):
        from intent_parser import ParsedIntent, ParsedFilter, normalize_typos

        normalized = normalize_typos(question.lower().strip())
        intent = ParsedIntent(
            original_question=question,
            normalized_question=normalized,
            intent=data.get('intent', 'count'),
        )
        intent.sub_intent = data.get('sub_intent', '')
        intent.tables = data.get('tables', [])
        intent.agg_function = data.get('agg_function', '')
        intent.agg_column = data.get('agg_column', '')
        intent.agg_table = data.get('agg_table', '')
        intent.order_by = data.get('order_by', '')
        intent.limit = data.get('limit')
        intent.temporal = data.get('temporal', False)
        intent.time_granularity = data.get('time_granularity', '')
        intent.comparison = data.get('comparison', False)
        intent.compare_values = data.get('compare_values', [])

        for gb in data.get('group_by', []):
            if isinstance(gb, (list, tuple)) and len(gb) == 2:
                intent.group_by.append((gb[0], gb[1]))

        for f in data.get('filters', []):
            if isinstance(f, dict):
                intent.filters.append(ParsedFilter(
                    column=f.get('column', ''),
                    operator=f.get('operator', '='),
                    value=f.get('value', ''),
                    table_hint=f.get('table', ''),
                    confidence=0.85,
                ))

        import re as _re
        words = _re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', normalized)
        col_matches = self.graph.find_columns_for_words(
            words, intent.tables or None, raw_question=normalized
        )
        intent.columns = [(t, c) for t, c, st, conf in col_matches[:10]]
        intent.confidence = 0.85

        return intent

    def _template_narrative(self, question: str, rows: List, columns: List[str],
                             intent_data: Dict) -> str:
        intent = intent_data.get('intent', '')
        row_count = len(rows)

        if row_count == 0:
            return "No results found for this query."

        if row_count == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            if isinstance(val, (int, float)):
                formatted = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
                return f"The result is {formatted}."
            return f"The result is {val}."

        if row_count == 1:
            parts = []
            for i, col in enumerate(columns):
                val = rows[0][i]
                if isinstance(val, float):
                    parts.append(f"{col}: {val:,.2f}")
                elif isinstance(val, int):
                    parts.append(f"{col}: {val:,}")
                else:
                    parts.append(f"{col}: {val}")
            return "; ".join(parts) + "."

        if intent in ('rank', 'trend'):
            first_row = rows[0]
            return (f"Found {row_count} results. "
                    f"Top result: {first_row[0]} with {first_row[-1] if len(first_row) > 1 else 'N/A'}.")

        return f"Query returned {row_count} rows across {len(columns)} columns."

    def stats(self) -> Dict[str, Any]:
        return {
            'llm_available': self.llm_available,
            'model': self.model.model_name if self.model else 'none',
            'nlu': self.nlu_engine.stats if self.nlu_engine else {},
            'sql': self.sql_engine.stats if self.sql_engine else {},
            'conversation_turns': len(self.conversation.history),
        }


def create_healthcare_llm_pipeline(db_path: str, schema_graph=None,
                                     config: Dict = None) -> HealthcareLLMPipeline:
    return HealthcareLLMPipeline(
        db_path=db_path,
        schema_graph=schema_graph,
        config=config,
    )


def diagnose_healthcare_llm() -> Dict[str, Any]:
    from local_llm_engine import LocalModelLoader

    report = {
        'backends': {},
        'models': [],
        'status': 'not_ready',
        'recommendation': '',
    }

    loader = LocalModelLoader()
    report['backends'] = loader.detect_backends()
    available = loader.detect_available()
    report['models'] = [
        {'name': s.name, 'path': p, 'params': s.param_count}
        for s, p in available
    ]

    if available and any(report['backends'].values()):
        report['status'] = 'ready'
        report['recommendation'] = f"Ready — best model: {available[0][0].name}"
    elif not any(report['backends'].values()):
        report['status'] = 'no_backend'
        report['recommendation'] = (
            "Install an inference backend:\n"
            "  pip install llama-cpp-python --break-system-packages"
        )
    else:
        report['status'] = 'no_model'
        report['recommendation'] = (
            "Download a GGUF model to ~/models/:\n"
            "  Llama 3 8B (best): ~4.5GB\n"
            "  Phi-3 Mini (fast): ~2.3GB"
        )

    return report


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    print("=" * 80)
    print("  HEALTHCARE LLM PIPELINE — Diagnostic & Test")
    print("=" * 80)

    report = diagnose_healthcare_llm()
    print(f"\nStatus: {report['status']}")
    print(f"Backends: {report['backends']}")
    print(f"Models: {report['models']}")
    print(f"Recommendation: {report['recommendation']}")

    if report['status'] == 'ready':
        print("\n" + "=" * 80)
        print("  Running pipeline test...")
        print("=" * 80)

        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare_production.db')
        if os.path.exists(db_path):
            pipeline = create_healthcare_llm_pipeline(
                db_path=db_path,
                config={'LLM_ENABLED': 'true'},
            )

            test_questions = [
                "how many members do we have?",
                "what is the denial rate?",
                "top 5 medications by cost",
                "claims per month",
            ]

            for q in test_questions:
                print(f"\nQ: {q}")
                result = pipeline.process(q)
                print(f"  Source: {result['source']}")
                print(f"  SQL: {result['sql'][:150]}")
                print(f"  Rows: {len(result['rows'])}")
                print(f"  Narrative: {(result['narrative'] or '')[:150]}")
                if result['error']:
                    print(f"  ERROR: {result['error']}")
                print(f"  Timing: {result['timing']}")
    else:
        print("\nNo LLM available — testing with rule-based fallbacks...")
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare_production.db')
        if os.path.exists(db_path):
            pipeline = create_healthcare_llm_pipeline(db_path=db_path)
            result = pipeline.process("how many claims?")
            print(f"  Source: {result['source']}")
            print(f"  SQL: {result['sql'][:150]}")
            print(f"  Rows: {len(result['rows'])}")
            print(f"  Narrative: {result['narrative'][:150] if result['narrative'] else 'N/A'}")

    print("\n✓ Done")
