import os
import re
import time
import json
import math
import sqlite3
import hashlib
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from tokenizer_utils import tokenize_query

logger = logging.getLogger('gpdm.advanced')


class SQLSelfHealer:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.max_retries = 2
        self._schema_cache = {}
        self._load_schema()

    def _load_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            for t in tables:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
                self._schema_cache[t.upper()] = {c[1].upper(): c[2] for c in cols}
            conn.close()
        except Exception:
            pass

    def heal(self, sql: str, error: Optional[str], rows: List,
             columns: List, question: str, regenerate_fn=None) -> Dict:
        actions = []
        probe_results = {}

        if error and sql:
            fixed_sql = self._fix_sql_error(sql, error)
            if fixed_sql and fixed_sql != sql:
                actions.append(f"Fixed SQL error: {error[:60]}")
                new_rows, new_cols, new_err = self._execute(fixed_sql)
                if not new_err:
                    return {
                        'healed': True, 'sql': fixed_sql, 'rows': new_rows,
                        'columns': new_cols, 'error': None,
                        'heal_actions': actions, 'probe_results': {}
                    }

            if regenerate_fn and self.max_retries > 0:
                actions.append("Attempting regeneration with rephrased question")
                alt_question = self._rephrase(question)
                new_result = regenerate_fn(alt_question)
                if new_result and new_result.get('sql'):
                    nr, nc, ne = self._execute(new_result['sql'])
                    if not ne and nr:
                        actions.append(f"Regenerated successfully with: '{alt_question}'")
                        return {
                            'healed': True, 'sql': new_result['sql'],
                            'rows': nr, 'columns': nc, 'error': None,
                            'heal_actions': actions, 'probe_results': {}
                        }

        if rows and sql and not error:
            probes = self._generate_probes(sql, columns, rows, question)
            for probe_name, probe_sql in probes.items():
                try:
                    pr, pc, pe = self._execute(probe_sql)
                    if not pe and pr:
                        probe_results[probe_name] = {
                            'value': pr[0][0] if pr else None,
                            'passed': True
                        }
                        if probe_name == 'row_count_check':
                            actual = len(rows)
                            expected = pr[0][0]
                            _sql_upper = (sql or '').upper()
                            _is_aggregate = any(agg in _sql_upper for agg in
                                               ['COUNT(', 'SUM(', 'AVG(', 'ROUND(', 'GROUP BY'])
                            if actual < expected * 0.5 and not (_is_aggregate and actual <= 30):
                                actions.append(
                                    f"Data Completeness Note: Result contains {actual:,} records "
                                    f"out of {expected:,} total in the source table — "
                                    f"verify that claim status filters and date ranges are inclusive"
                                )
                                probe_results[probe_name]['warning'] = True
                        elif probe_name == 'null_check':
                            null_pct = pr[0][0]
                            if null_pct and null_pct > 50:
                                actions.append(
                                    f"Warning: {null_pct:.0f}% NULL values in result"
                                )
                                probe_results[probe_name]['warning'] = True
                except Exception:
                    probe_results[probe_name] = {'passed': False}

        return {
            'healed': bool(actions),
            'sql': sql, 'rows': rows, 'columns': columns,
            'error': error, 'heal_actions': actions,
            'probe_results': probe_results
        }

    def _fix_sql_error(self, sql: str, error: str) -> Optional[str]:
        err = error.lower()
        fixed = sql

        m = re.search(r'no such column:\s*(\S+)', err)
        if m:
            bad_col = m.group(1)
            for table, cols in self._schema_cache.items():
                if bad_col.upper() in cols:
                    fixed = re.sub(
                        rf'\b{re.escape(bad_col)}\b',
                        f'{table}.{bad_col}', fixed, count=1
                    )
                    return fixed

        m = re.search(r'no such table:\s*(\S+)', err)
        if m:
            bad_table = m.group(1).upper()
            for real_table in self._schema_cache:
                if bad_table in real_table or real_table in bad_table:
                    fixed = re.sub(
                        rf'\b{re.escape(m.group(1))}\b',
                        real_table.lower(), fixed
                    )
                    return fixed

        m = re.search(r'ambiguous column name:\s*(\S+)', err)
        if m:
            col = m.group(1)
            for table, cols in self._schema_cache.items():
                if col.upper() in cols:
                    fixed = re.sub(
                        rf'(?<![.\w]){re.escape(col)}(?![.\w])',
                        f'{table.lower()}.{col}', fixed
                    )
                    return fixed

        if 'not an aggregate' in err or 'must appear in the GROUP BY' in err:
            select_cols = re.findall(r'SELECT\s+(.+?)\s+FROM', sql, re.I | re.S)
            group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER|\s+HAVING|\s+LIMIT|$)',
                                    sql, re.I | re.S)
            if select_cols and group_match:
                return None

        return None

    def _generate_probes(self, sql: str, columns: List, rows: List,
                         question: str) -> Dict[str, str]:
        probes = {}

        table_match = re.search(r'FROM\s+(\w+)', sql, re.I)
        if not table_match:
            return probes

        main_table = table_match.group(1)

        probes['row_count_check'] = f"SELECT COUNT(*) FROM {main_table}"

        num_cols = [c for i, c in enumerate(columns)
                    if rows and isinstance(rows[0][i] if i < len(rows[0]) else None, (int, float))]
        if num_cols:
            col = num_cols[0]
            probes['null_check'] = (
                f"SELECT ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) "
                f"/ COUNT(*), 1) FROM {main_table}"
            )

        if any(kw in sql.upper() for kw in ['AVG(', 'SUM(', 'COUNT(']):
            agg_match = re.search(r'(AVG|SUM|COUNT)\(([^)]+)\)', sql, re.I)
            if agg_match:
                func, col = agg_match.group(1), agg_match.group(2)
                probes['range_check'] = (
                    f"SELECT MIN({col}), MAX({col}), COUNT(DISTINCT {col}) "
                    f"FROM {main_table}"
                )

        return probes

    def _rephrase(self, question: str) -> str:
        q = question.lower().strip()
        if 'show' not in q and 'list' not in q and 'what' not in q:
            return f"Show me {question}"
        if 'by' not in q:
            return question + " grouped by category"
        return f"Calculate {question}"

    def _execute(self, sql: str) -> Tuple[List, List, Optional[str]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(sql)
            cols = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return rows, cols, None
        except Exception as e:
            return [], [], str(e)


class QueryDecomposer:

    COMPLEX_PATTERNS = [
        r'\b(compared?\s+to|versus|vs\.?|relative\s+to)\b',
        r'\b(and\s+also|as\s+well\s+as|along\s+with|together\s+with)\b',
        r'\b(before\s+and\s+after|year\s+over\s+year|month\s+over\s+month)\b',
        r'\b(correlation|relationship\s+between)\b',
        r'\b(what\s+percentage\s+of.+that\s+also)\b',
        r'\b(among\s+those\s+who|of\s+the\s+ones\s+that|subset\s+where)\b',
        r'\b(higher|lower|more|less|greater|fewer)\s+than\s+(the\s+)?(average|median|mean)\b',
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path

    def needs_decomposition(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in self.COMPLEX_PATTERNS)

    def decompose(self, question: str) -> List[Dict]:
        q = question.lower()
        steps = []

        cmp = re.search(r'(.+?)\s+(?:compared?\s+to|versus|vs\.?)\s+(.+)', q)
        if cmp:
            steps.append({
                'step': 1, 'type': 'compute',
                'question': cmp.group(1).strip(),
                'label': 'Group A'
            })
            steps.append({
                'step': 2, 'type': 'compute',
                'question': cmp.group(2).strip(),
                'label': 'Group B'
            })
            steps.append({
                'step': 3, 'type': 'compose',
                'action': 'compare',
                'label': 'Comparison'
            })
            return steps

        subset = re.search(
            r'(?:among|of)\s+(?:those|the\s+ones|patients|members|providers)\s+'
            r'(?:who|that|where)\s+(.+?)[,]\s*(.+)', q
        )
        if subset:
            steps.append({
                'step': 1, 'type': 'filter',
                'question': subset.group(1).strip(),
                'label': 'Filter criteria'
            })
            steps.append({
                'step': 2, 'type': 'compute',
                'question': subset.group(2).strip(),
                'depends_on': 1,
                'label': 'Analysis on filtered set'
            })
            return steps

        avg_cmp = re.search(
            r'(.+?)\s+(higher|lower|more|less|greater|fewer)\s+than\s+'
            r'(?:the\s+)?(?:average|mean|median)\s*(.*)', q
        )
        if avg_cmp:
            steps.append({
                'step': 1, 'type': 'compute',
                'question': f"average {avg_cmp.group(1).strip()} {avg_cmp.group(3).strip()}",
                'label': 'Compute average'
            })
            direction = 'above' if avg_cmp.group(2) in ('higher', 'more', 'greater') else 'below'
            steps.append({
                'step': 2, 'type': 'filter',
                'question': f"{avg_cmp.group(1).strip()} {direction} average",
                'depends_on': 1,
                'label': f'Filter {direction} average'
            })
            return steps

        return []

    def compose_results(self, step_results: List[Dict]) -> Dict:
        if not step_results:
            return {}

        compare_step = next((s for s in step_results if s.get('type') == 'compose'
                             and s.get('action') == 'compare'), None)
        if compare_step and len(step_results) >= 2:
            a = step_results[0]
            b = step_results[1]
            return self._build_comparison(a, b)

        return step_results[-1] if step_results else {}

    def _build_comparison(self, a: Dict, b: Dict) -> Dict:
        a_rows = a.get('rows', [])
        b_rows = b.get('rows', [])
        a_cols = a.get('columns', [])
        b_cols = b.get('columns', [])

        if len(a_rows) == 1 and len(b_rows) == 1:
            return {
                'columns': ['Metric', a.get('label', 'A'), b.get('label', 'B'), 'Difference'],
                'rows': [[
                    a_cols[0] if a_cols else 'Value',
                    a_rows[0][0],
                    b_rows[0][0],
                    round(a_rows[0][0] - b_rows[0][0], 2) if isinstance(a_rows[0][0], (int, float)) and isinstance(b_rows[0][0], (int, float)) else 'N/A'
                ]],
            }

        combined_cols = ['Source'] + (a_cols or b_cols)
        combined_rows = []
        for r in a_rows[:20]:
            combined_rows.append([a.get('label', 'A')] + list(r))
        for r in b_rows[:20]:
            combined_rows.append([b.get('label', 'B')] + list(r))
        return {'columns': combined_cols, 'rows': combined_rows}


class ConfidenceScorer:

    def __init__(self, schema_cache: Dict = None):
        self.schema = schema_cache or {}

    def score(self, question: str, sql: str, rows: List, columns: List,
              error: str, intent: Dict, strategy: Dict,
              neural_context: Dict = None, nlp_enrichment: Dict = None,
              sql_result: Dict = None) -> Dict:
        neural_context = neural_context or {}
        nlp_enrichment = nlp_enrichment or {}
        sql_result = sql_result or {}
        scores = {}

        scores['execution'] = 0.0 if error else 1.0

        scores['plausibility'] = self._plausibility_score(rows, columns, question)

        scores['schema_alignment'] = self._schema_alignment(sql)

        scores['semantic_match'] = self._semantic_match(question, sql, columns)

        scores['strategy'] = strategy.get('confidence', 0.5)

        scores['neural_corroboration'] = self._neural_corroboration(
            neural_context, sql_result, intent
        )

        scores['entity_coverage'] = self._entity_coverage(
            nlp_enrichment, sql, sql_result
        )

        weights = {
            'execution': 0.25,
            'plausibility': 0.18,
            'schema_alignment': 0.17,
            'semantic_match': 0.17,
            'strategy': 0.08,
            'neural_corroboration': 0.08,
            'entity_coverage': 0.07,
        }
        overall = sum(scores[k] * weights[k] for k in weights)
        scores['overall'] = round(overall, 3)
        scores['grade'] = (
            'A' if overall >= 0.85 else
            'B' if overall >= 0.70 else
            'C' if overall >= 0.50 else
            'D' if overall >= 0.30 else 'F'
        )
        scores['uncertain'] = overall < 0.60
        if scores['uncertain']:
            scores['uncertainty_reason'] = self._explain_uncertainty(scores)

        return scores

    def _plausibility_score(self, rows: List, columns: List, question: str) -> float:
        if not rows:
            return 0.3
        score = 0.5
        if len(rows) > 0:
            score += 0.2
        if len(columns) >= 2:
            score += 0.1
        null_count = sum(1 for r in rows for v in r if v is None)
        total = max(len(rows) * max(len(columns), 1), 1)
        if null_count / total < 0.1:
            score += 0.2
        return min(score, 1.0)

    def _schema_alignment(self, sql: str) -> float:
        if not sql or not self.schema:
            return 0.5
        tables_in_sql = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.I)
        tables = [t for pair in tables_in_sql for t in pair if t]
        if not tables:
            return 0.5
        valid = sum(1 for t in tables if t.upper() in self.schema)
        return valid / len(tables) if tables else 0.5

    def _semantic_match(self, question: str, sql: str, columns: List) -> float:
        q = question.lower()
        score = 0.5

        if 'count' in q and 'COUNT' in sql.upper():
            score += 0.2
        if ('average' in q or 'avg' in q) and 'AVG' in sql.upper():
            score += 0.2
        if ('total' in q or 'sum' in q) and 'SUM' in sql.upper():
            score += 0.2
        if 'rate' in q and ('CASE' in sql.upper() or '%' in q):
            score += 0.2
        if 'by' in q and 'GROUP BY' in sql.upper():
            score += 0.1

        q_words = tokenize_query(q, min_length=3)
        col_words = set(w.lower() for c in (columns or []) for w in c.replace('_', ' ').split())
        overlap = q_words & col_words
        if overlap:
            score += min(len(overlap) * 0.05, 0.2)

        return min(score, 1.0)

    def _neural_corroboration(self, neural_context: Dict,
                               sql_result: Dict, intent: Dict) -> float:
        score = 0.5

        schema_matches = neural_context.get('schema_matches', [])
        if schema_matches:
            tables_used = set(t.upper() for t in sql_result.get('tables_used', []))
            neural_tables = set()
            for sm in schema_matches:
                node_id = sm.get('node_id', '')
                if '.' in node_id:
                    neural_tables.add(node_id.split('.')[0].upper())
                else:
                    neural_tables.add(node_id.upper())
            if tables_used and neural_tables:
                overlap = tables_used & neural_tables
                score = len(overlap) / max(len(tables_used), 1)
                score = min(score + 0.3, 1.0)

        recalled = neural_context.get('recalled_patterns', [])
        if recalled:
            good_recalls = [s for _, s in recalled if s > 0.5]
            if good_recalls:
                score = min(score + 0.2, 1.0)

        if sql_result.get('intent_fusion'):
            fusion = sql_result['intent_fusion']
            if fusion.get('tfidf') == fusion.get('hopfield'):
                score = min(score + 0.15, 1.0)

        return score

    def _entity_coverage(self, nlp_enrichment: Dict,
                          sql: str, sql_result: Dict) -> float:
        entities = nlp_enrichment.get('entities', [])
        if not entities:
            return 0.7

        sql_upper = (sql or '').upper()
        covered = 0
        for entity_text, entity_type in entities:
            if entity_text.upper() in sql_upper:
                covered += 1

        coverage = covered / len(entities) if entities else 0.7
        return min(coverage + 0.3, 1.0)

    def _explain_uncertainty(self, scores: Dict) -> str:
        reasons = []
        if scores.get('execution', 1) == 0:
            reasons.append("SQL execution failed")
        if scores.get('plausibility', 1) < 0.5:
            reasons.append("results look implausible")
        if scores.get('schema_alignment', 1) < 0.5:
            reasons.append("referenced tables may not match the question")
        if scores.get('semantic_match', 1) < 0.5:
            reasons.append("SQL operations don't clearly match the question's intent")
        if scores.get('neural_corroboration', 1) < 0.4:
            reasons.append("neural understanding disagrees with the SQL approach")
        if scores.get('entity_coverage', 1) < 0.4:
            reasons.append("some mentioned entities weren't captured in the query")
        return '; '.join(reasons) if reasons else "multiple weak signals"


class NarrativeEngine:


    def __init__(self, domain_config=None):
        self.domain_config = domain_config
        if domain_config:
            self._rate_context = domain_config.rate_context or {}
            self._recommendations = domain_config.recommendations or {}
        else:
            self._rate_context = {}
            self._recommendations = {}

    @staticmethod
    def _detect_entity_from_name(col_name: str) -> str:
        cn = col_name.lower()
        if 'npi' in cn or 'provider' in cn or 'physician' in cn or 'doctor' in cn:
            return 'provider'
        if 'facility' in cn or 'hospital' in cn or 'clinic' in cn or 'location' in cn or 'site' in cn:
            return 'facility'
        if 'plan' in cn and 'type' in cn:
            return 'plan'
        if 'region' in cn or 'state' in cn or 'county' in cn or 'zip' in cn:
            return 'region'
        if 'month' in cn or 'quarter' in cn or 'year' in cn or 'date' in cn or 'period' in cn:
            return 'period'
        if 'member' in cn or 'patient' in cn:
            return 'member'
        if 'diagnosis' in cn or 'dx' in cn or 'icd' in cn:
            return 'diagnosis'
        if 'procedure' in cn or 'cpt' in cn:
            return 'procedure'
        return 'category'

    def generate(self, question: str, sql: str, rows: List, columns: List,
                 intent: Dict, clinical_context: Dict = None,
                 confidence: Dict = None, anomalies: List = None,
                 heal_actions: List = None, decompose_steps: List = None) -> str:
        if not rows:
            return self._no_results_narrative(question, columns)

        q = question.lower()
        col_lower = [c.lower() for c in columns] if columns else []
        entity_type = self._detect_entity(col_lower)
        rate_domain = self._detect_rate_domain(q, col_lower)
        parts = []

        direct = self._direct_answer(question, rows, columns, entity_type)
        if direct:
            parts.append(direct)

        asks_which = any(w in q for w in ('which', 'who'))
        asks_extreme = any(w in q for w in ('highest', 'lowest', 'most', 'least', 'best', 'worst'))
        asks_driver = any(w in q for w in ('driving', 'driver', 'cause', 'causing', 'why'))

        if direct and len(rows) == 1:
            pass
        elif asks_which and asks_extreme and len(rows) >= 2:
            parts.append(self._narrate_comparison(rows, columns, question, entity_type))
        elif asks_driver and len(rows) >= 2:
            parts.append(self._narrate_breakdown(rows, columns, question, entity_type))
        elif 'rate' in q or 'percentage' in q or 'pct' in q or rate_domain:
            parts.append(self._narrate_rate(rows, columns, question, entity_type, rate_domain))
        elif 'top' in q or 'highest' in q or 'best' in q or 'worst' in q or 'lowest' in q:
            parts.append(self._narrate_ranking(rows, columns, question, entity_type))
        elif 'trend' in q or 'over time' in q or 'monthly' in q or 'quarterly' in q:
            parts.append(self._narrate_trend(rows, columns, question, entity_type))
        elif 'compare' in q or 'vs' in q or 'versus' in q or 'by' in q:
            parts.append(self._narrate_comparison(rows, columns, question, entity_type))
        elif len(rows) == 1:
            parts.append(self._narrate_single_value(rows, columns, question, entity_type))
        elif len(columns) >= 2:
            parts.append(self._narrate_breakdown(rows, columns, question, entity_type))
        else:
            parts.append(f"Analysis returned {len(rows)} records across the requested dimension.")

        if asks_driver and len(rows) >= 2 and not any(
            isinstance(rows[0][i], (int, float)) for i in range(len(columns))
        ):
            from collections import Counter
            cat_summaries = []
            id_keywords = {'id', '_id', 'npi', 'member_id', 'encounter_id', 'diagnosis_id'}
            for ci, cn in enumerate(columns):
                if any(kw in cn.lower() for kw in id_keywords):
                    continue
                vals = [str(r[ci]) for r in rows if ci < len(r) and r[ci] is not None]
                unique_count = len(set(vals))
                if 2 <= unique_count <= min(20, len(vals) * 0.8):
                    top3 = Counter(vals).most_common(3)
                    label = cn.replace('_', ' ').title()
                    top_str = ', '.join(f"{v} ({c})" for v, c in top3)
                    cat_summaries.append(f"{label}: {top_str}")
            if cat_summaries:
                parts[0] = (
                    f"Found {len(rows)} related records. "
                    + "Key patterns: " + "; ".join(cat_summaries[:4]) + "."
                )
            elif len(rows) > 0:
                parts[0] = f"Found {len(rows)} records related to the query — drill into specific categories or time periods for more insight."

        if anomalies:
            alerts = [a for a in anomalies if a.get('severity') in ('high', 'medium')]
            if alerts:
                parts.append(self._narrate_anomalies(alerts, entity_type, rate_domain))

        if confidence and confidence.get('grade'):
            grade = confidence['grade']
            if grade in ('D', 'F'):
                parts.append(
                    "Data Confidence Advisory: This result has a low confidence score. "
                    "The query may not fully capture the intended metric — consider refining "
                    "the question or validating against a known report."
                )
            elif grade == 'C':
                parts.append(
                    "Note: Moderate confidence — results are directionally accurate but "
                    "may benefit from additional filtering or validation."
                )

        if heal_actions:
            parts.append(self._narrate_heal_actions(heal_actions))

        if clinical_context and isinstance(clinical_context, dict):
            ctx = self._narrate_clinical_context(clinical_context)
            if ctx:
                parts.append(ctx)

        recs = self._generate_recommendations(q, rows, columns, anomalies, rate_domain, entity_type)
        if recs:
            parts.append(recs)

        return " ".join(parts)


    def _detect_entity(self, col_lower: List[str]) -> str:
        if not col_lower:
            return 'item'
        first_col = col_lower[0]
        if self.domain_config:
            et = self.domain_config.find_entity_type(first_col)
            if et != 'unknown':
                return et
        result = self._detect_entity_from_name(first_col)
        return result if result != 'category' else 'item'

    def _direct_answer(self, question: str, rows: List, columns: List,
                        entity_type: str) -> str:
        if not rows or not columns:
            return ''

        q = question.lower()
        col_lower = [c.lower() for c in columns]

        try:
            asks_which = any(w in q for w in ('which', 'who', 'what'))
            asks_extreme = any(w in q for w in ('highest', 'lowest', 'most',
                                                 'least', 'best', 'worst', 'top'))
            asks_how_many = any(w in q for w in ('how many', 'count', 'total number'))
            asks_average = any(w in q for w in ('average', 'avg', 'mean'))
            asks_what_is = q.startswith(('what is', 'what are', "what's"))

            metric_idx = -1
            metric_val = None
            _priority_kw = ('rate', 'pct', 'percent', 'avg', 'average', 'ratio')
            for i, c in enumerate(columns):
                if any(kw in c.lower() for kw in _priority_kw):
                    if i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                        metric_idx = i
                        metric_val = rows[0][i]
                        break
            if metric_val is None:
                for i, c in enumerate(columns):
                    if any(kw in c.lower() for kw in ('amount', 'cost', 'paid', 'billed')):
                        if i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                            metric_idx = i
                            metric_val = rows[0][i]
                            break
            if metric_val is None:
                for i in range(len(columns) - 1, -1, -1):
                    if i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                        if columns[i].lower() not in ('record_count', 'row_count', 'total_count'):
                            metric_idx = i
                            metric_val = rows[0][i]
                            break
            if metric_val is None:
                for i in range(len(columns) - 1, -1, -1):
                    if i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                        metric_idx = i
                        metric_val = rows[0][i]
                        break

            dim_idx = 0
            for i, c in enumerate(columns):
                if i < len(rows[0]) and not isinstance(rows[0][i], (int, float)):
                    dim_idx = i
                    break

            if metric_val is None:
                return ''

            is_rate = any(kw in col_lower[metric_idx] for kw in
                         ('rate', 'pct', 'percent', 'ratio', 'avg'))
            is_currency = any(kw in col_lower[metric_idx] for kw in
                            ('amount', 'cost', 'paid', 'billed', 'charge', 'revenue'))

            if is_rate and 0 <= metric_val <= 100:
                val_str = f"{metric_val:.1f}%"
            elif is_currency:
                val_str = f"${metric_val:,.2f}"
            elif metric_val == int(metric_val) and metric_val > 0:
                val_str = f"{int(metric_val):,}"
            else:
                val_str = f"{metric_val:,.2f}"

            metric_label = self._humanize_column(columns[metric_idx])

            if len(rows) == 1 and not (asks_which and asks_extreme):
                if asks_how_many:
                    return f"There are {val_str} total."
                elif asks_average:
                    return f"The {metric_label.lower()} is {val_str}."
                elif asks_what_is:
                    return f"The {metric_label.lower()} is {val_str}."
                else:
                    return f"Result: {val_str} ({metric_label.lower()})."

            if asks_extreme and len(rows) >= 1:
                top_entity = rows[0][dim_idx] if dim_idx < len(rows[0]) else '?'
                top_val = rows[0][metric_idx] if metric_idx < len(rows[0]) else '?'
                if is_rate and isinstance(top_val, (int, float)):
                    top_str = f"{top_val:.1f}%"
                elif is_currency and isinstance(top_val, (int, float)):
                    top_str = f"${top_val:,.2f}"
                elif isinstance(top_val, (int, float)) and top_val == int(top_val):
                    top_str = f"{int(top_val):,}"
                elif isinstance(top_val, (int, float)):
                    top_str = f"{top_val:,.2f}"
                else:
                    top_str = str(top_val)
                return f"{top_entity} has the {'highest' if 'highest' in q or 'most' in q or 'top' in q else 'lowest'} {metric_label.lower()} at {top_str}."

            if len(rows) >= 2 and metric_idx >= 0:
                all_vals = [r[metric_idx] for r in rows
                           if metric_idx < len(r) and isinstance(r[metric_idx], (int, float))]
                if all_vals:
                    lo, hi = min(all_vals), max(all_vals)
                    avg = sum(all_vals) / len(all_vals)
                    if is_rate:
                        return f"Across {len(rows)} groups, {metric_label.lower()} ranges from {lo:.1f}% to {hi:.1f}% (average {avg:.1f}%)."
                    elif is_currency:
                        return f"Across {len(rows)} groups, {metric_label.lower()} ranges from ${lo:,.2f} to ${hi:,.2f} (average ${avg:,.2f})."
                    else:
                        return f"Across {len(rows)} groups, {metric_label.lower()} ranges from {self._fmt(lo)} to {self._fmt(hi)}."

        except Exception:
            pass
        return ''

    def _detect_rate_domain(self, question: str, col_lower: List[str]) -> Optional[str]:
        combined = question + ' ' + ' '.join(col_lower)
        if any(w in combined for w in ('denial', 'denied', 'deny', 'rejection')):
            return 'denial'
        if any(w in combined for w in ('readmission', 'readmit', 'bounce-back')):
            return 'readmission'
        if any(w in combined for w in ('utilization', 'occupancy', 'capacity')):
            return 'utilization'
        if any(w in combined for w in ('no_show', 'no show', 'missed appointment')):
            return 'no_show'
        if any(w in combined for w in ('er_visit', 'emergency', 'ed visit')):
            return 'er_visit'
        return None

    def _entity_label(self, entity_type: str, value) -> str:
        val_str = str(value)

        if len(val_str) > 2 and val_str[:2].isalpha() and any(c.isdigit() for c in val_str[2:]):
            prefix = ''.join(c for c in val_str if c.isalpha())[:3].upper()
            return f"{prefix} {val_str}"

        if val_str.isdigit() and len(val_str) >= 5:
            entity_label = entity_type.replace('_', ' ').title()
            return f"{entity_label} #{val_str}"

        if entity_type and entity_type not in ('item', 'category'):
            prefix = entity_type.replace('_', ' ').title()
            return f"{prefix} {val_str}"

        return val_str


    def _no_results_narrative(self, question: str, columns: List) -> str:
        parts = ["The query returned no matching records."]
        parts.append(
            "Consider broadening your date range, adjusting filters, or verifying "
            "that the requested data elements exist in the current dataset."
        )
        return " ".join(parts)


    def _narrate_single_value(self, rows, columns, question, entity_type):
        col_lower = [c.lower() for c in columns] if columns else []
        rate_domain = self._detect_rate_domain(question.lower(), col_lower)

        if len(columns) > 2 and len(rows[0]) > 2:
            parts = []
            rate_val = None
            rate_col_label = None
            for i, col in enumerate(columns):
                v = rows[0][i]
                label = self._humanize_column(col)
                if isinstance(v, (int, float)):
                    is_rate_col = any(kw in col.lower() for kw in ('rate', 'pct', 'percent', 'ratio', 'avg'))
                    if is_rate_col and 0 <= v <= 100:
                        parts.append(f"{label}: {v:.1f}%")
                        rate_val = v
                        rate_col_label = label
                    elif v == int(v) and v > 100:
                        parts.append(f"{label}: {int(v):,}")
                    elif isinstance(v, float):
                        parts.append(f"{label}: {v:,.2f}")
                    else:
                        parts.append(f"{label}: {self._fmt(v)}")
                elif v is not None:
                    parts.append(f"{label}: {v}")

            base = " | ".join(parts[:8]) + "."

            if rate_domain and rate_val is not None:
                ctx = self._rate_context.get(rate_domain, {})
                benchmark = ctx.get('benchmark', '')
                if rate_val <= ctx.get('good', 0):
                    base += f" The {rate_col_label.lower()} is within a healthy range ({benchmark})."
                elif rate_val <= ctx.get('acceptable', 0):
                    base += f" The {rate_col_label.lower()} falls within acceptable limits ({benchmark})."
                elif rate_val <= ctx.get('elevated', 0):
                    base += f" The {rate_col_label.lower()} is elevated compared to industry norms ({benchmark}). Warrants monitoring."
                else:
                    base += f" The {rate_col_label.lower()} is significantly above benchmark thresholds ({benchmark}). Immediate review recommended."
            return base

        val = rows[0][-1] if len(rows[0]) > 1 else rows[0][0]
        col_label = self._humanize_column(columns[-1] if len(columns) > 1 else columns[0]) if columns else 'result'
        formatted = self._fmt(val)

        base = f"The {col_label} is {formatted}."

        if rate_domain and isinstance(val, (int, float)) and 0 <= val <= 100:
            ctx = self._rate_context.get(rate_domain, {})
            benchmark = ctx.get('benchmark', '')
            if val <= ctx.get('good', 0):
                base += f" This is within a healthy range ({benchmark})."
            elif val <= ctx.get('acceptable', 0):
                base += f" This falls within acceptable limits ({benchmark})."
            elif val <= ctx.get('elevated', 0):
                base += f" This is elevated compared to industry norms ({benchmark}). Warrants monitoring."
            else:
                base += f" This is significantly above benchmark thresholds ({benchmark}). Immediate review recommended."
        elif isinstance(val, (int, float)) and val > 100 and rate_domain:
            base = f"The {col_label} is {int(val):,}."
        return base

    def _narrate_rate(self, rows, columns, question, entity_type, rate_domain):
        q = question.lower()
        if len(rows) == 1 and len(columns) >= 1:
            return self._narrate_single_value(rows, columns, question, entity_type)

        if len(rows) > 1 and len(columns) >= 2:
            metric_idx = self._find_metric_column(rows, columns)
            vals = [r[metric_idx] for r in rows if isinstance(r[metric_idx], (int, float))]

            if not vals:
                return f"Rate analysis returned {len(rows)} results across {self._entity_plural(entity_type)}."

            mean_val = sum(vals) / len(vals)
            min_val, max_val = min(vals), max(vals)
            spread = max_val - min_val

            metric_col_name = columns[metric_idx].lower() if metric_idx < len(columns) else ''
            _rate_indicators = ('rate', 'pct', 'percent', 'percentage', 'ratio', 'proportion')
            _col_looks_like_rate = any(ri in metric_col_name for ri in _rate_indicators)
            is_actual_rate = max_val <= 100 and min_val >= 0 and _col_looks_like_rate
            unit = '%' if is_actual_rate else ''

            sorted_rows = sorted(rows, key=lambda r: r[metric_idx] if isinstance(r[metric_idx], (int, float)) else 0, reverse=True)
            top_label = self._entity_label(entity_type, sorted_rows[0][0])
            bot_label = self._entity_label(entity_type, sorted_rows[-1][0])

            domain_label = self._domain_label(rate_domain)

            if is_actual_rate:
                parts = [
                    f"Across {len(rows)} {self._entity_plural(entity_type)}, "
                    f"{domain_label} range from {self._fmt(min_val)}% to {self._fmt(max_val)}%, "
                    f"with a network average of {mean_val:.1f}%."
                ]

                if spread > 20:
                    parts.append(
                        f"There is a {spread:.1f} percentage-point spread between the highest "
                        f"({top_label} at {self._fmt(max_val)}%) and lowest "
                        f"({bot_label} at {self._fmt(min_val)}%) — "
                        f"this level of variation indicates significant inconsistency across the network "
                        f"and warrants a focused review."
                    )
                elif spread > 10:
                    parts.append(
                        f"Moderate variance observed ({spread:.1f} pp spread). "
                        f"Top: {top_label} at {self._fmt(max_val)}%. "
                        f"Bottom: {bot_label} at {self._fmt(min_val)}%."
                    )

                if rate_domain:
                    ctx = self._rate_context.get(rate_domain, {})
                    benchmark = ctx.get('benchmark', '')
                    if benchmark and mean_val > ctx.get('elevated', float('inf')):
                        parts.append(
                            f"The network average of {mean_val:.1f}% exceeds industry benchmarks ({benchmark}). "
                            f"This suggests a systemic issue rather than isolated outliers."
                        )
            else:
                col_label = self._humanize_column(columns[metric_idx])
                parts = [
                    f"Across {len(rows)} {self._entity_plural(entity_type)}, "
                    f"{col_label} ranges from {self._fmt(min_val)} to {self._fmt(max_val)}, "
                    f"with an average of {mean_val:,.1f}."
                ]
                if sorted_rows:
                    parts.append(
                        f"Highest: {top_label} ({self._fmt(max_val)}). "
                        f"Lowest: {bot_label} ({self._fmt(min_val)})."
                    )

            return " ".join(parts)

        return f"Rate analysis returned {len(rows)} results."

    def _narrate_ranking(self, rows, columns, question, entity_type):
        if len(rows) >= 2:
            metric_idx = self._find_metric_column(rows, columns)
            top_label = self._entity_label(entity_type, rows[0][0])
            top_val = self._fmt(rows[0][metric_idx] if metric_idx < len(rows[0]) else rows[0][-1])
            metric_name = self._humanize_column(columns[metric_idx]) if metric_idx < len(columns) else 'value'
            return (
                f"Among {len(rows)} {self._entity_plural(entity_type)}, "
                f"{top_label} ranks highest with a {metric_name} of {top_val}. "
                f"The second-highest is {self._entity_label(entity_type, rows[1][0])} "
                f"at {self._fmt(rows[1][metric_idx] if metric_idx < len(rows[1]) else rows[1][-1])}."
            )
        return f"Ranking shows {len(rows)} {self._entity_plural(entity_type)}."

    def _narrate_trend(self, rows, columns, question, entity_type):
        if len(rows) >= 3 and len(columns) >= 2:
            metric_idx = self._find_metric_column(rows, columns)
            vals = [r[metric_idx] for r in rows if isinstance(r[metric_idx], (int, float))]
            if vals:
                first_half = vals[:len(vals)//2]
                second_half = vals[len(vals)//2:]
                avg1 = sum(first_half) / len(first_half) if first_half else 0
                avg2 = sum(second_half) / len(second_half) if second_half else 0
                direction = "upward" if avg2 > avg1 else "downward" if avg2 < avg1 else "flat"
                pct = abs(avg2 - avg1) / max(avg1, 1) * 100
                metric_name = self._humanize_column(columns[metric_idx]) if metric_idx < len(columns) else 'metric'

                parts = [
                    f"Over the {len(rows)} periods analyzed, {metric_name} shows "
                    f"an {direction} trajectory with a {pct:.1f}% shift "
                    f"(from an average of {self._fmt(avg1)} to {self._fmt(avg2)})."
                ]
                if direction == 'upward' and pct > 15:
                    parts.append("This acceleration warrants investigation — is this driven by volume changes, acuity shifts, or process breakdowns?")
                elif direction == 'downward' and pct > 15:
                    parts.append("This improvement trend is encouraging — identify the interventions or operational changes that may be driving it.")
                return " ".join(parts)
        return f"Trend data spans {len(rows)} reporting periods."

    def _narrate_comparison(self, rows, columns, question, entity_type):
        if len(rows) >= 2 and len(columns) >= 2:
            metric_idx = self._find_metric_column(rows, columns)
            metric_name = self._humanize_column(columns[metric_idx]) if metric_idx < len(columns) else 'metric'

            sorted_rows = sorted(
                [r for r in rows if isinstance(r[metric_idx], (int, float))],
                key=lambda r: r[metric_idx], reverse=True
            )
            if not sorted_rows:
                return f"Comparison returned {len(rows)} {self._entity_plural(entity_type)}."

            top_row = sorted_rows[0]
            bot_row = sorted_rows[-1]
            top_val = top_row[metric_idx]
            bot_val = bot_row[metric_idx]
            gap = abs(top_val - bot_val)

            q = question.lower()
            asks_which = any(w in q for w in ('which', 'who', 'what'))
            asks_highest = any(w in q for w in ('highest', 'most', 'best', 'top', 'greatest', 'maximum'))
            asks_lowest = any(w in q for w in ('lowest', 'least', 'worst', 'bottom', 'minimum', 'fewest'))

            if asks_which and (asks_highest or asks_lowest):
                target = bot_row if asks_lowest else top_row
                target_val = bot_val if asks_lowest else top_val
                rank_word = "lowest" if asks_lowest else "highest"
                parts = [
                    f"{self._entity_label(entity_type, target[0])} has the {rank_word} "
                    f"{metric_name} at {self._fmt(target_val)}."
                ]
                if len(sorted_rows) >= 3:
                    avg_val = sum(r[metric_idx] for r in sorted_rows) / len(sorted_rows)
                    parts.append(
                        f"Across all {len(sorted_rows)} {self._entity_plural(entity_type)}, "
                        f"the average is {self._fmt(avg_val)} "
                        f"(range: {self._fmt(bot_val)} to {self._fmt(top_val)})."
                    )
                if len(sorted_rows) >= 2:
                    runner = sorted_rows[1] if not asks_lowest else sorted_rows[-2]
                    runner_val = runner[metric_idx]
                    parts.append(
                        f"Next closest: {self._entity_label(entity_type, runner[0])} at {self._fmt(runner_val)}."
                    )
                return " ".join(parts)

            return (
                f"Comparison across {len(rows)} {self._entity_plural(entity_type)} reveals "
                f"a {self._fmt(gap)} gap in {metric_name}: "
                f"{self._entity_label(entity_type, top_row[0])} leads at {self._fmt(top_val)}, "
                f"while {self._entity_label(entity_type, bot_row[0])} trails at {self._fmt(bot_val)}."
            )
        return f"Comparison returned {len(rows)} {self._entity_plural(entity_type)}."

    def _narrate_breakdown(self, rows, columns, question, entity_type):
        q = question.lower()
        if len(rows) >= 2:
            metric_idx = self._find_metric_column(rows, columns)
            vals = [r[metric_idx] for r in rows if isinstance(r[metric_idx], (int, float))]
            if not vals:
                return f"Results span {len(rows)} {self._entity_plural(entity_type)}."

            metric_name = self._humanize_column(columns[metric_idx]) if metric_idx < len(columns) else 'volume'

            sorted_rows = sorted(
                [r for r in rows if isinstance(r[metric_idx], (int, float))],
                key=lambda r: r[metric_idx], reverse=True
            )
            total = sum(r[metric_idx] for r in sorted_rows)

            asks_driver = any(w in q for w in ('driving', 'driver', 'cause', 'causing', 'why', 'contributing'))
            if asks_driver and sorted_rows:
                parts = []
                top3 = sorted_rows[:min(3, len(sorted_rows))]
                top3_total = sum(r[metric_idx] for r in top3)
                top3_pct = (top3_total / total * 100) if total > 0 else 0

                parts.append(
                    f"Top {len(top3)} contributors account for {top3_pct:.0f}% of total {metric_name}:"
                )
                for i, r in enumerate(top3):
                    val = r[metric_idx]
                    pct = (val / total * 100) if total > 0 else 0
                    label = self._entity_label(entity_type, r[0])
                    parts.append(f"({i+1}) {label}: {self._fmt(val)} ({pct:.1f}% of total)")

                if len(sorted_rows) > 3:
                    remainder = total - top3_total
                    parts.append(
                        f"Remaining {len(sorted_rows) - 3} categories account for "
                        f"{self._fmt(remainder)} ({(remainder/total*100):.0f}%)."
                    )
                return " ".join(parts)

            top_label = self._entity_label(entity_type, sorted_rows[0][0])
            top_val = sorted_rows[0][metric_idx]
            top_pct = (top_val / total * 100) if total else 0

            parts = [
                f"Distribution across {len(rows)} {self._entity_plural(entity_type)} "
                f"shows a total {metric_name} of {self._fmt(total)}. "
                f"{top_label} represents the largest share at {self._fmt(top_val)} "
                f"({top_pct:.1f}% of total)."
            ]

            if len(sorted_rows) >= 2:
                second = sorted_rows[1]
                second_pct = (second[metric_idx] / total * 100) if total else 0
                parts.append(
                    f"Followed by {self._entity_label(entity_type, second[0])} "
                    f"at {self._fmt(second[metric_idx])} ({second_pct:.1f}%)."
                )

            return " ".join(parts)

        return f"Results span {len(rows)} {self._entity_plural(entity_type)}."


    def _narrate_anomalies(self, alerts: List[Dict], entity_type: str, rate_domain: Optional[str]) -> str:
        domain_label = self._domain_label(rate_domain) if rate_domain else 'metric'
        rate_alerts = [a for a in alerts if any(kw in a.get('column','').lower() for kw in ('rate', 'pct', 'percent', 'avg', 'ratio'))]
        best_alerts = rate_alerts if rate_alerts else alerts
        best_alerts = [a for a in best_alerts
                       if not (abs(a.get('value', 0)) < 0.01 and abs(a.get('mean', 0)) < 0.01)]
        best_alerts = [a for a in best_alerts
                       if a.get('label', '').lower() not in ('unknown', 'none', '', 'null')]
        if not best_alerts:
            return ""

        parts = [f"Outlier Alert ({len(best_alerts)} flagged):"]
        for a in best_alerts[:3]:
            label = a.get('label', 'Unknown')
            entity_lbl = self._entity_label(entity_type, label)
            val = a.get('value', 0)
            mean = a.get('mean', 0)
            z = a.get('z_score', 0)
            col = a.get('column', '').lower()
            is_rate = any(kw in col for kw in ('rate', 'pct', 'percent', 'ratio', 'avg'))
            unit = '%' if is_rate else ''
            deviation_pct = abs(val - mean) / max(abs(mean), 0.01) * 100

            if z > 4:
                severity_text = "extreme statistical outlier"
            elif z > 3:
                severity_text = "significant outlier"
            else:
                severity_text = "notable deviation"

            direction = "above" if val > mean else "below"
            article = 'an' if severity_text[0] in 'aeiou' else 'a'

            parts.append(
                f"{entity_lbl} shows {article} {severity_text} — "
                f"{domain_label} of {self._fmt(val)}{unit} vs. the network average of {self._fmt(mean)}{unit} "
                f"({deviation_pct:.0f}% {direction} average, {z:.1f} standard deviations). "
                f"This warrants a focused chart audit or claims review."
            )
        return " ".join(parts)


    def _narrate_heal_actions(self, heal_actions: List[str]) -> str:
        cleaned = []
        for action in heal_actions[:2]:
            if 'missing JOIN' in action or 'full table' in action:
                cleaned.append(
                    "Data Completeness Note: The result set may be a subset of the full population. "
                    "Verify that appropriate claim status filters and date ranges are applied."
                )
            elif 'Fixed SQL' in action:
                cleaned.append("Query was automatically corrected for a schema mismatch and re-executed.")
            else:
                cleaned.append(action)
        return " ".join(cleaned)


    def _narrate_clinical_context(self, clinical_context: Dict) -> str:
        items = []
        skip_keys = {'sql', 'query', 'confidence', 'intent', 'explanation',
                     'strategy', 'latency', 'source', 'cache', 'engine',
                     'semantic_intent', 'semantic_confidence', 'intent_backend',
                     'backend', 'model', 'nlp_tier', 'processing_time',
                     'semantic intent', 'semantic confidence', 'intent backend'}
        label_map = {
            'category': 'Clinical Category',
            'risk_level': 'Risk Level',
            'care_setting': 'Care Setting',
            'quality_measure': 'Quality Measure',
            'regulatory_flag': 'Regulatory Flag',
            'cms_measure': 'CMS Quality Measure',
            'hedis_measure': 'HEDIS Measure',
        }
        for k, v in clinical_context.items():
            if k.lower() in skip_keys:
                continue
            if v and not isinstance(v, (dict, list)):
                val_str = str(v)
                if (len(val_str) > 120 or 'SELECT' in val_str.upper()
                        or 'FROM' in val_str.upper() or '<b>' in val_str
                        or '<br>' in val_str or val_str.startswith('{')):
                    continue
                label = label_map.get(k, k.replace('_', ' ').title())
                items.append(f"{label}: {v}")
        if items:
            return "Clinical Context — " + "; ".join(items[:4]) + "."
        return ""


    def _generate_recommendations(self, question: str, rows: List, columns: List,
                                   anomalies: List, rate_domain: Optional[str],
                                   entity_type: str) -> str:
        q = question.lower()
        rec_pool = []

        if rate_domain == 'denial':
            vals = self._extract_numeric_values(rows, columns)
            mean_val = sum(vals) / len(vals) if vals else 0
            if mean_val > 15 or any(v > 25 for v in vals):
                rec_pool.extend(self._recommendations.get('denial_high', [])[:2])
            if vals and (max(vals) - min(vals)) > 15:
                rec_pool.extend(self._recommendations.get('denial_variance', [])[:2])
        elif rate_domain == 'readmission':
            rec_pool.extend(self._recommendations.get('readmission_high', [])[:2])

        if anomalies:
            high_sev = [a for a in anomalies if a.get('severity') == 'high']
            if high_sev:
                if rate_domain == 'denial' or 'denial' in q:
                    rec_pool.extend(self._recommendations.get('denial_high', [])[:1])
                else:
                    rec_pool.extend(self._recommendations.get('cost_outlier', [])[:1])

        if anomalies and any(a.get('type') == 'concentration' for a in anomalies):
            rec_pool.extend(self._recommendations.get('volume_concentration', [])[:1])

        if len(rows) > 200:
            rec_pool.extend(self._recommendations.get('data_quality', [])[:1])

        if not rec_pool:
            general_recs = self._recommendations.get('general', [])
            col_str = ' '.join(c.lower() for c in (columns or []))
            filtered = []
            for r in general_recs:
                if 'time period' in r.lower() and any(t in col_str for t in ['month', 'year', 'quarter', 'period', 'date']):
                    continue
                if 'region' in r.lower() and any(t in col_str for t in ['region', 'facility', 'location']):
                    continue
                filtered.append(r)
            rec_pool.extend(filtered[:2])

        seen = set()
        unique = []
        for r in rec_pool:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        unique = unique[:3]

        return "Recommended Actions: " + " | ".join(f"({i+1}) {r}" for i, r in enumerate(unique))


    def _find_metric_column(self, rows, columns) -> int:
        col_lower = [c.lower() for c in columns] if columns else []

        def _has_variance(col_idx):
            vals = [r[col_idx] for r in rows if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if not vals:
                return False
            return any(abs(v) > 0.001 for v in vals)

        for i, c in enumerate(col_lower):
            if any(kw in c for kw in ('rate', 'pct', 'percent', 'avg', 'average', 'ratio', 'score')):
                if rows and isinstance(rows[0][i], (int, float)) and _has_variance(i):
                    return i
        for i in range(len(columns) - 1, 0, -1):
            if rows and i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                if _has_variance(i):
                    return i
        for i in range(len(columns) - 1, 0, -1):
            if rows and i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                return i
        return min(1, len(columns) - 1) if len(columns) > 1 else 0

    def _extract_numeric_values(self, rows, columns) -> List[float]:
        idx = self._find_metric_column(rows, columns)
        return [r[idx] for r in rows if idx < len(r) and isinstance(r[idx], (int, float))]

    def _humanize_column(self, col_name: str) -> str:
        replacements = {
            'pct': 'rate', 'avg': 'average', 'cnt': 'count', 'amt': 'amount',
            'los': 'length of stay', 'npi': 'NPI', 'er': 'emergency',
            'dx': 'diagnosis', 'cpt': 'procedure', 'drg': 'DRG',
        }
        name = col_name.replace('_', ' ').strip()
        for abbr, full in replacements.items():
            name = re.sub(rf'\b{abbr}\b', full, name, flags=re.IGNORECASE)
        name = re.sub(r'\b(\w+)\s+\1\b', r'\1', name, flags=re.IGNORECASE)
        return name

    def _entity_plural(self, entity_type: str) -> str:
        plurals = {
            'provider': 'providers', 'facility': 'facilities', 'plan': 'plan types',
            'region': 'regions', 'diagnosis': 'diagnoses', 'procedure': 'procedures',
            'member': 'members', 'period': 'periods', 'item': 'categories',
        }
        return plurals.get(entity_type, 'categories')

    def _domain_label(self, rate_domain: Optional[str]) -> str:
        if not rate_domain:
            return 'rates'
        label = rate_domain.replace('_', '-') + ' rates'
        return label

    def _fmt(self, val):
        if isinstance(val, float):
            if abs(val) >= 1e6:
                return f"${val/1e6:,.1f}M"
            if abs(val) >= 1e3:
                return f"{val:,.0f}"
            return f"{val:,.2f}"
        if isinstance(val, int):
            if abs(val) >= 1e6:
                return f"${val/1e6:,.1f}M"
            return f"{val:,}"
        return str(val)


class SemanticCache:

    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.92):
        self.cache = {}
        self.max_size = max_size
        self.threshold = similarity_threshold
        self._word_idf = {}
        self._total_docs = 0

    def get(self, question: str) -> Optional[Dict]:
        q_norm = self._normalize(question)
        q_hash = hashlib.md5(q_norm.encode()).hexdigest()

        if q_hash in self.cache:
            entry = self.cache[q_hash]
            result = entry['result']
            if result.get('error'):
                logger.info("Cache entry rejected (contains error): '%s'", question[:40])
                return None
            entry['hits'] += 1
            return result

        q_words = set(q_norm.split())
        best_score = 0
        best_entry = None

        for h, entry in self.cache.items():
            c_words = set(entry['normalized'].split())
            score = self._weighted_jaccard(q_words, c_words)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry:
            result = best_entry['result'].copy()
            if result.get('error'):
                logger.info("Cache entry rejected (contains error): '%s'", question[:40])
                return None
            best_entry['hits'] += 1
            logger.info("Cache hit (%.2f similarity): '%s'", best_score, question[:40])
            result['cache_hit'] = True
            result['cache_similarity'] = round(best_score, 3)
            return result

        return None

    def put(self, question: str, result: Dict):
        q_norm = self._normalize(question)
        q_hash = hashlib.md5(q_norm.encode()).hexdigest()

        self._total_docs += 1
        for w in set(q_norm.split()):
            self._word_idf[w] = self._word_idf.get(w, 0) + 1

        self.cache[q_hash] = {
            'question': question,
            'normalized': q_norm,
            'result': result,
            'timestamp': time.time(),
            'hits': 0,
        }

        if len(self.cache) > self.max_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1]['hits'], x[1]['timestamp'])
            )
            for h, _ in sorted_entries[:len(self.cache) - self.max_size]:
                del self.cache[h]

    def _normalize(self, q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        stopwords = {'show', 'me', 'the', 'a', 'an', 'of', 'in', 'for',
                     'and', 'or', 'please', 'can', 'you', 'give', 'list',
                     'display', 'tell'}
        words = [w for w in q.split() if w not in stopwords]
        return ' '.join(words)

    def _weighted_jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        intersection = a & b
        union = a | b
        if not union:
            return 0.0

        def idf(w):
            df = self._word_idf.get(w, 1)
            return math.log(max(self._total_docs, 1) / df + 1)

        inter_weight = sum(idf(w) for w in intersection)
        union_weight = sum(idf(w) for w in union)
        return inter_weight / union_weight if union_weight else 0.0

    @property
    def stats(self):
        return {
            'size': len(self.cache),
            'total_docs': self._total_docs,
            'vocab_size': len(self._word_idf),
            'total_hits': sum(e['hits'] for e in self.cache.values()),
        }


class AnomalyDetector:

    def _label_column(self, col_name: str) -> str:
        cn = col_name.lower()
        label = cn.replace('_', ' ')
        abbrevs = {
            'avg': 'average', 'cnt': 'count', 'pct': 'percentage',
            'amt': 'amount', 'qty': 'quantity', 'num': 'number',
            'los': 'length of stay', 'dt': 'date', 'ts': 'timestamp',
        }
        words = label.split()
        expanded = [abbrevs.get(w, w) for w in words]
        return ' '.join(expanded)

    def _label_entity(self, row, col_idx: int) -> str:
        if len(row) == 0 or col_idx == 0:
            return "Record"
        val = str(row[0])
        if len(val) > 2 and val[:3].isalpha() and val[3:].replace('-', '').isdigit():
            prefix = val[:3].upper()
            return f"{prefix} {val}"
        if val.isdigit() and len(val) == 10:
            return f"Provider (NPI: {val})"
        if val.isdigit() and len(val) > 5:
            return f"ID {val}"
        return str(row[0])

    def detect(self, rows: List, columns: List, question: str = '') -> List[Dict]:
        if not rows or not columns:
            return []

        findings = []
        q = question.lower()

        for col_idx, col_name in enumerate(columns):
            values = []
            for r in rows:
                if col_idx < len(r) and isinstance(r[col_idx], (int, float)):
                    values.append((r, r[col_idx]))

            if len(values) < 3:
                continue

            nums = [v[1] for v in values]
            mean = sum(nums) / len(nums)
            variance = sum((x - mean) ** 2 for x in nums) / len(nums)
            std = math.sqrt(variance) if variance > 0 else 0

            if std == 0:
                continue

            if abs(mean) < 0.001 and max(abs(v) for v in nums) < 0.01:
                continue

            metric_label = self._label_column(col_name)

            for row, val in values:
                z = abs(val - mean) / std if std > 0 else 0
                if z > 2.5:
                    entity = self._label_entity(row, col_idx)
                    if entity.lower() in ('unknown', 'none', '', 'record', 'null'):
                        continue
                    deviation_pct = abs(val - mean) / max(abs(mean), 0.01) * 100

                    if z > 4:
                        severity_word = "extreme"
                        risk_note = "Requires immediate operational review."
                    elif z > 3:
                        severity_word = "significant"
                        risk_note = "Recommend targeted chart audit or claims review."
                    else:
                        severity_word = "elevated"
                        risk_note = "Flag for monitoring in next review cycle."

                    direction = "above" if val > mean else "below"

                    cn = col_name.lower()
                    is_rate = any(kw in cn for kw in ('rate', 'pct', 'percent', 'ratio', 'avg', 'average'))
                    unit_suffix = '%' if is_rate else ''
                    val_display = f"{self._fmt(val)}{unit_suffix}"
                    mean_display = f"{self._fmt(mean)}{unit_suffix}"

                    findings.append({
                        'type': 'outlier',
                        'severity': 'high' if z > 3 else 'medium',
                        'column': col_name,
                        'value': val,
                        'label': str(row[0]) if len(row) > 0 and col_idx > 0 else 'Record',
                        'z_score': round(z, 2),
                        'mean': round(mean, 2),
                        'message': (
                            f"{entity}: {metric_label} of {val_display} is "
                            f"{severity_word} — {deviation_pct:.0f}% {direction} the network average "
                            f"of {mean_display} ({z:.1f}σ deviation). {risk_note}"
                        ),
                    })

            if nums and max(nums) > 0:
                max_val = max(nums)
                total = sum(nums)
                if total > 0:
                    concentration = max_val / total
                    if concentration > 0.5 and len(nums) > 3:
                        top_row = max(values, key=lambda x: x[1])
                        entity = self._label_entity(top_row[0], col_idx)
                        findings.append({
                            'type': 'concentration',
                            'severity': 'medium',
                            'column': col_name,
                            'message': (
                                f"Volume Concentration Risk: {entity} accounts for "
                                f"{concentration*100:.0f}% of total {metric_label}. "
                                f"This level of concentration creates operational "
                                f"dependency risk and may indicate network adequacy gaps."
                            ),
                        })

            first_col = columns[0].lower() if columns else ''
            is_time_series = any(kw in first_col for kw in (
                'month', 'quarter', 'year', 'date', 'period', 'week', 'day',
                'time', 'yr', 'qtr', 'mo',
            )) or any(kw in q for kw in ('trend', 'over time', 'monthly', 'quarterly', 'yearly'))

            if is_time_series and len(nums) >= 5:
                for i in range(1, len(nums)):
                    if nums[i-1] != 0:
                        pct_change = abs(nums[i] - nums[i-1]) / abs(nums[i-1]) * 100
                        if pct_change > 50:
                            period = rows[i][0] if len(rows[i]) > 0 and col_idx > 0 else f"Period {i+1}"
                            direction = "increase" if nums[i] > nums[i-1] else "decrease"
                            findings.append({
                                'type': 'sudden_change',
                                'severity': 'high' if pct_change > 100 else 'medium',
                                'column': col_name,
                                'message': (
                                    f"Period-over-Period Alert at {period}: {metric_label} "
                                    f"showed a {pct_change:.0f}% {direction} "
                                    f"({self._fmt(nums[i-1])} → {self._fmt(nums[i])}). "
                                    f"Investigate whether this reflects a coding change, "
                                    f"payer policy update, or true utilization shift."
                                ),
                            })

        findings.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x.get('severity', 'low'), 3))
        return findings[:5]

    def _fmt(self, val):
        if isinstance(val, float):
            if abs(val) >= 1e6:
                return f"{val/1e6:,.1f}M"
            if abs(val) >= 1e3:
                return f"{val:,.0f}"
            return f"{val:,.2f}"
        return f"{val:,}" if isinstance(val, int) else str(val)
