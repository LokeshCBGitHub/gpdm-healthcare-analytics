import os
import re
import json
import time
import math
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

logger = logging.getLogger('gpdm.deep_intel')


def _load_benchmarks_from_config(config_path: Optional[str] = None) -> Tuple[Dict, Dict]:
    benchmarks = {}
    keyword_map = {}

    if not config_path:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'domain_config.json'
        )

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            benchmarks = config.get('benchmarks', {})
            keyword_map = config.get('benchmark_keyword_map', {})
            for _key, bench in benchmarks.items():
                if 'percentiles' in bench:
                    bench['percentiles'] = {
                        int(k): v for k, v in bench['percentiles'].items()
                    }
            logger.info("Loaded %d benchmarks and %d keyword mappings from config",
                        len(benchmarks), len(keyword_map))
        except Exception as e:
            logger.warning("Failed to load benchmark config: %s", e)

    return benchmarks, keyword_map


_BENCHMARKS, _KEYWORD_MAP = _load_benchmarks_from_config()


class BenchmarkEngine:

    def __init__(self, domain_config=None):
        if domain_config and domain_config.benchmarks:
            self._benchmarks = domain_config.benchmarks
            self._keyword_map = domain_config.benchmark_keyword_map
        else:
            self._benchmarks = _BENCHMARKS
            self._keyword_map = _KEYWORD_MAP

    def benchmark(self, question: str, rows: List, columns: List,
                  intent: Dict = None) -> Dict:
        q = question.lower()
        result = {
            'has_benchmark': False,
            'benchmark_name': None,
            'benchmark_value': None,
            'your_value': None,
            'comparison': None,
            'percentile': None,
            'source': None,
            'message': None
        }

        if not self._benchmarks or not self._keyword_map:
            return result

        matched_key = None
        q_words = set(re.findall(r'\b\w+\b', q))
        for keyword, bkey in self._keyword_map.items():
            kw_words = keyword.split()
            if len(kw_words) == 1:
                if keyword in q_words:
                    matched_key = bkey
                    break
            else:
                pattern = r'\b' + r'\s+'.join(re.escape(w) for w in kw_words) + r'\b'
                if re.search(pattern, q):
                    matched_key = bkey
                    break

        if not matched_key or matched_key not in self._benchmarks:
            return result

        bench = self._benchmarks[matched_key]

        your_value = self._extract_primary_value(rows, columns, bench)
        if your_value is None:
            return result

        result['has_benchmark'] = True
        result['benchmark_name'] = matched_key.replace('_', ' ').title()
        result['benchmark_value'] = bench['benchmark']
        result['your_value'] = round(your_value, 2)
        result['unit'] = bench.get('unit', '')
        result['source'] = bench.get('source', '')

        pcts = bench.get('percentiles', {})
        if pcts:
            if your_value <= pcts.get(25, float('inf')):
                result['percentile'] = 'Top 25%' if bench['direction'] == 'lower_is_better' else 'Bottom 25%'
            elif your_value <= pcts.get(50, float('inf')):
                result['percentile'] = 'Top 50%' if bench['direction'] == 'lower_is_better' else '25th-50th percentile'
            elif your_value <= pcts.get(75, float('inf')):
                result['percentile'] = '50th-75th percentile'
            else:
                result['percentile'] = 'Bottom 25%' if bench['direction'] == 'lower_is_better' else 'Top 25%'

        diff = your_value - bench['benchmark']
        pct_diff = abs(diff) / bench['benchmark'] * 100 if bench['benchmark'] else 0
        direction = bench.get('direction', 'neutral')

        if direction == 'lower_is_better':
            if diff < 0:
                result['comparison'] = 'better'
                result['message'] = (
                    f"Your {matched_key.replace('_',' ')} of {your_value:.1f}{bench['unit']} is "
                    f"{pct_diff:.0f}% better than the industry benchmark of "
                    f"{bench['benchmark']}{bench['unit']} ({bench.get('source','')})."
                )
            else:
                result['comparison'] = 'worse'
                result['message'] = (
                    f"Your {matched_key.replace('_',' ')} of {your_value:.1f}{bench['unit']} is "
                    f"{pct_diff:.0f}% above the industry benchmark of "
                    f"{bench['benchmark']}{bench['unit']} ({bench.get('source','')}) — improvement opportunity."
                )
        elif direction == 'higher_is_better':
            if diff > 0:
                result['comparison'] = 'better'
                result['message'] = (
                    f"Your {matched_key.replace('_',' ')} of {your_value:.1f}{bench['unit']} "
                    f"exceeds the industry benchmark of {bench['benchmark']}{bench['unit']} "
                    f"by {pct_diff:.0f}% ({bench.get('source','')})."
                )
            else:
                result['comparison'] = 'worse'
                result['message'] = (
                    f"Your {matched_key.replace('_',' ')} of {your_value:.1f}{bench['unit']} is "
                    f"{pct_diff:.0f}% below the industry benchmark of "
                    f"{bench['benchmark']}{bench['unit']} ({bench.get('source','')}) — needs attention."
                )
        else:
            result['comparison'] = 'neutral'
            result['message'] = (
                f"Your {matched_key.replace('_',' ')} is {your_value:.1f}{bench['unit']}. "
                f"Industry median: {bench['benchmark']}{bench['unit']} ({bench.get('source','')})."
            )

        if result['percentile']:
            result['message'] += f" Ranking: {result['percentile']}."

        return result

    def _extract_primary_value(self, rows, columns, bench=None):
        if not rows:
            return None

        cols_lower = [(i, c.lower()) for i, c in enumerate(columns or [])]

        def _avg_col(idx):
            vals = [r[idx] for r in rows if idx < len(r) and isinstance(r[idx], (int, float))]
            return sum(vals) / len(vals) if vals else None

        def _is_plausible_for_benchmark(val, benchmark_info):
            if not benchmark_info:
                return True
            unit = benchmark_info.get('unit', '')
            if unit == '%' and (val < 0 or val > 100):
                return False
            if unit == 'per 1000' and (val < 0 or val > 2000):
                return False
            if unit == 'days' and (val < 0 or val > 365):
                return False
            return True

        candidates = []

        priority_keywords = ['rate', 'pct', 'percent', 'avg', 'average', 'ratio', 'score']
        for keyword in priority_keywords:
            for i, cl in cols_lower:
                if keyword in cl and rows and i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                    val = _avg_col(i)
                    if val is not None:
                        candidates.append((val, i, cl))

        for i in range(len(columns or []) - 1, -1, -1):
            if rows and i < len(rows[0]) and isinstance(rows[0][i], (int, float)):
                val = _avg_col(i)
                if val is not None and not any(c[1] == i for c in candidates):
                    candidates.append((val, i, cols_lower[i][1] if i < len(cols_lower) else ''))

        if len(rows) == 1 and not candidates:
            for i, v in enumerate(rows[0]):
                if isinstance(v, (int, float)):
                    cl = cols_lower[i][1] if i < len(cols_lower) else ''
                    candidates.append((float(v), i, cl))

        if bench and bench.get('unit') == '%':
            rate_cols = ['rate', 'pct', 'percent', 'percentage', 'ratio']
            for val, idx, colname in candidates:
                if any(rc in colname for rc in rate_cols) and _is_plausible_for_benchmark(val, bench):
                    return val
            return None

        for val, idx, colname in candidates:
            if _is_plausible_for_benchmark(val, bench):
                return val

        return None


class MultiReasoningEngine:

    def __init__(self, domain_config=None):
        self.domain_config = domain_config

    def _classify_domains(self, question: str, columns: List[str]) -> set:
        if self.domain_config:
            return self.domain_config.classify_domain_from_columns(columns or [])

        combined = question.lower() + ' ' + ' '.join(
            c.lower().replace('_', ' ') for c in (columns or [])
        )
        domains = set()

        if any(w in combined for w in ['paid', 'billed', 'amount', 'cost', 'revenue',
                                        'charge', 'claim', 'payment', 'expense']):
            domains.add('financial')

        if any(w in combined for w in ['diagnosis', 'icd', 'cpt', 'encounter',
                                        'readmit', 'readmission', 'patient', 'member',
                                        'provider', 'prescription', 'procedure']):
            domains.add('clinical')

        if any(w in combined for w in ['appointment', 'schedule', 'utiliz', 'capacity',
                                        'wait', 'throughput', 'staffing', 'volume']):
            domains.add('operational')

        return domains if domains else {'general'}

    def reason(self, question: str, rows: List, columns: List,
               intent: Dict, sql: str, anomalies: List = None) -> Dict:
        q = question.lower()
        reasoning = {
            'strategies_applied': [],
            'insights': [],
            'risk_flags': [],
            'opportunities': [],
            'recommended_actions': [],
            'reasoning_depth': 'standard',
        }

        stat = self._statistical_reasoning(rows, columns, q)
        reasoning['strategies_applied'].append('statistical')
        reasoning['insights'].extend(stat.get('insights', []))

        domains = self._classify_domains(q, columns)

        if 'clinical' in domains:
            clinical = self._clinical_reasoning(rows, columns, q)
            reasoning['strategies_applied'].append('clinical')
            reasoning['insights'].extend(clinical.get('insights', []))
            reasoning['risk_flags'].extend(clinical.get('risk_flags', []))
            reasoning['reasoning_depth'] = 'deep'

        if 'operational' in domains:
            ops = self._operational_reasoning(rows, columns, q)
            reasoning['strategies_applied'].append('operational')
            reasoning['insights'].extend(ops.get('insights', []))
            reasoning['opportunities'].extend(ops.get('opportunities', []))
            reasoning['reasoning_depth'] = 'deep'

        if 'financial' in domains:
            fin = self._financial_reasoning(rows, columns, q)
            reasoning['strategies_applied'].append('financial')
            reasoning['insights'].extend(fin.get('insights', []))
            reasoning['recommended_actions'].extend(fin.get('actions', []))

        if anomalies:
            reasoning['strategies_applied'].append('anomaly_investigation')
            for a in anomalies[:3]:
                reasoning['risk_flags'].append(
                    f"Anomaly in {a.get('column','')}: {a.get('message','')}"
                )

        return reasoning

    def _statistical_reasoning(self, rows, columns, question):
        insights = []
        if not rows or not columns:
            return {'insights': []}

        for col_idx, col_name in enumerate(columns):
            vals = [r[col_idx] for r in rows
                    if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if len(vals) < 2:
                continue

            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(variance) if variance > 0 else 0
            cv = (std / mean * 100) if mean != 0 else 0
            total = sum(vals)

            col_label = col_name.replace('_', ' ')

            if cv > 50:
                insights.append(
                    f"High variability in {col_label} (CV={cv:.0f}%) — "
                    f"results are spread widely from {min(vals):,.0f} to {max(vals):,.0f}"
                )
            elif cv < 10 and len(vals) > 3:
                insights.append(
                    f"{col_label} is remarkably consistent across categories "
                    f"(CV={cv:.0f}%, range {min(vals):,.0f}–{max(vals):,.0f})"
                )

            if len(vals) >= 5:
                sorted_vals = sorted(vals, reverse=True)
                top_20_count = max(1, int(len(sorted_vals) * 0.2))
                top_20_sum = sum(sorted_vals[:top_20_count])
                if total > 0 and top_20_sum / total > 0.6:
                    insights.append(
                        f"Pareto effect: top {top_20_count} of {len(vals)} categories "
                        f"account for {top_20_sum/total*100:.0f}% of total {col_label}"
                    )

        return {'insights': insights[:3]}

    def _clinical_reasoning(self, rows, columns, question):
        insights = []
        risk_flags = []

        RATE_KEYWORDS = {'rate', 'pct', 'percent', 'ratio', 'percentage'}

        for col_idx, col_name in enumerate(columns or []):
            cn = col_name.lower()
            vals = [r[col_idx] for r in (rows or [])
                    if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if not vals:
                continue

            avg_val = sum(vals) / len(vals)
            is_rate_col = any(kw in cn for kw in RATE_KEYWORDS)
            is_plausible_rate = is_rate_col and 0 <= avg_val <= 100

            if ('readmi' in cn or 'readmi' in question) and is_plausible_rate:
                if avg_val > 15:
                    risk_flags.append(
                        f"Readmission rate of {avg_val:.1f}% exceeds CMS target of 15.5% — "
                        f"review discharge planning and post-acute care coordination"
                    )
            if ('denial' in cn or 'denied' in cn or 'denial' in question) and is_plausible_rate:
                if avg_val > 10:
                    risk_flags.append(
                        f"Denial rate {avg_val:.1f}% is above industry average of 8.2% — "
                        f"investigate top denial reason codes and prior auth workflows"
                    )
            if ('los' in cn or 'length' in cn or ('length' in question and 'stay' in question)):
                if 0 < avg_val < 365:
                    if avg_val > 5:
                        insights.append(
                            f"Average LOS of {avg_val:.1f} days — consider case management "
                            f"review for conditions with LOS > 7 days"
                        )

        return {'insights': insights[:2], 'risk_flags': risk_flags[:2]}

    def _operational_reasoning(self, rows, columns, question):
        insights = []
        opportunities = []

        for col_idx, col_name in enumerate(columns or []):
            vals = [r[col_idx] for r in (rows or [])
                    if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if not vals:
                continue

            avg_val = sum(vals) / len(vals)
            cn = col_name.lower()

            if 'utiliz' in cn or 'utiliz' in question:
                if avg_val < 65:
                    opportunities.append(
                        f"Utilization at {avg_val:.0f}% indicates unused capacity — "
                        f"opportunity to increase patient volume or consolidate resources"
                    )
                elif avg_val > 90:
                    insights.append(
                        f"Utilization at {avg_val:.0f}% — near capacity, risk of "
                        f"bottlenecks and burnout"
                    )

            if 'cost' in cn or 'expense' in cn:
                if len(vals) >= 3:
                    sorted_vals = sorted(vals, reverse=True)
                    top_pct = sorted_vals[0] / sum(vals) * 100 if sum(vals) > 0 else 0
                    if top_pct > 30:
                        opportunities.append(
                            f"Top cost category represents {top_pct:.0f}% of total — "
                            f"focused cost reduction here has highest ROI"
                        )

        return {'insights': insights[:2], 'opportunities': opportunities[:2]}

    def _financial_reasoning(self, rows, columns, question):
        insights = []
        actions = []

        for col_idx, col_name in enumerate(columns or []):
            vals = [r[col_idx] for r in (rows or [])
                    if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if not vals:
                continue

            cn = col_name.lower()
            total = sum(vals)
            avg_val = sum(vals) / len(vals) if vals else 0

            if 'paid' in cn or 'amount' in cn or 'revenue' in cn:
                if len(vals) >= 3:
                    max_val = max(vals)
                    if total > 0 and max_val / total > 0.4:
                        actions.append(
                            "Revenue is heavily concentrated — diversify "
                            "across regions/plan types to reduce risk"
                        )
                    if min(vals) < avg_val * 0.5:
                        min_row = rows[vals.index(min(vals))]
                        label = min_row[0] if min_row else 'Bottom segment'
                        insights.append(
                            f"{label} is performing at less than half the average — "
                            f"investigate for growth potential or cost issues"
                        )

        return {'insights': insights[:2], 'actions': actions[:2]}


class PrecursorInsightEngine:

    def __init__(self, db_path: str, domain_config=None):
        self.db_path = db_path
        self.domain_config = domain_config
        self._schema = {}
        self._load_schema()

    def _load_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            for t in tables:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
                self._schema[t.lower()] = [
                    {'name': c[1], 'type': c[2]} for c in cols
                ]
            conn.close()
        except Exception as e:
            logger.debug("Schema load failed: %s", e)

    def generate_deep_dives(self, question: str, primary_rows: List,
                            primary_columns: List) -> List[Dict]:
        q = question.lower()
        q_words = set(re.findall(r'[a-z]+', q))
        dives = []

        target_table = self._find_question_table(q_words)
        if not target_table:
            return dives

        cols = self._schema.get(target_table, [])
        col_names = [c['name'] for c in cols]
        col_types = {c['name'].lower(): c['type'].upper() for c in cols}

        categorical_cols = [c['name'] for c in cols
                           if self._is_categorical_col(c, target_table)]
        numeric_cols = [c['name'] for c in cols
                       if self._is_numeric_col(c)]
        date_cols = [c['name'] for c in cols
                    if self._is_date_col(c)]
        id_cols = [c['name'] for c in cols
                  if c['name'].lower().endswith('_id')]

        for cat_col in categorical_cols[:3]:
            agg_parts = []
            agg_parts.append("COUNT(*) as count")
            for nc in numeric_cols[:2]:
                agg_parts.append(f"ROUND(AVG({nc}), 2) as avg_{nc.lower()}")

            sql = (f"SELECT {cat_col}, {', '.join(agg_parts)} "
                   f"FROM {target_table} "
                   f"GROUP BY {cat_col} ORDER BY count DESC LIMIT 10")
            dives.append({
                'label': f'Breakdown by {cat_col.replace("_", " ").title()}',
                'sql': sql,
                'type': 'breakdown',
            })

        if date_cols and numeric_cols:
            date_col = date_cols[0]
            metric_col = numeric_cols[0]
            sql = (f"SELECT strftime('%Y-%m', {date_col}) as month, "
                   f"COUNT(*) as volume, ROUND(AVG({metric_col}), 2) as avg_{metric_col.lower()} "
                   f"FROM {target_table} "
                   f"WHERE {date_col} IS NOT NULL "
                   f"GROUP BY month ORDER BY month")
            dives.append({
                'label': f'{metric_col.replace("_", " ").title()} Trend Over Time',
                'sql': sql,
                'type': 'trend',
            })

        results = []
        for dive in dives[:4]:
            try:
                result = self._execute_dive(dive)
                if result and result.get('rows'):
                    results.append(result)
            except Exception as e:
                logger.debug("Deep dive failed: %s", e)

        return results[:4]

    def _find_question_table(self, q_words: set) -> Optional[str]:
        best_table = None
        best_score = 0
        for tbl_name in self._schema:
            tbl_lower = tbl_name.lower()
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 2 else tbl_lower
            score = 0
            if tbl_stem in q_words or tbl_lower in q_words:
                score += 10
            tbl_words = set(tbl_lower.replace('_', ' ').split())
            score += len(tbl_words & q_words) * 3
            for col in self._schema[tbl_name]:
                col_words = set(col['name'].lower().replace('_', ' ').split())
                col_words -= {'id', 'key', 'code'}
                score += len(col_words & q_words)
            if score > best_score:
                best_score = score
                best_table = tbl_name
        return best_table if best_score >= 1 else (
            list(self._schema.keys())[0] if self._schema else None
        )

    def _is_categorical_col(self, col_info: Dict, table: str) -> bool:
        name = col_info['name'].lower()
        ctype = col_info['type'].upper()
        if name.endswith('_id'):
            return False
        if ctype in ('TEXT', 'VARCHAR', 'CHAR', 'NVARCHAR'):
            return True
        if self.domain_config:
            et = self.domain_config.find_entity_type(col_info['name'])
            if et != 'unknown':
                return True
        return False

    def _is_numeric_col(self, col_info: Dict) -> bool:
        name = col_info['name'].lower()
        ctype = col_info['type'].upper()
        if name.endswith('_id'):
            return False
        return any(t in ctype for t in ['INT', 'REAL', 'FLOAT', 'DECIMAL', 'NUMERIC', 'DOUBLE'])

    def _is_date_col(self, col_info: Dict) -> bool:
        name = col_info['name'].lower()
        ctype = col_info['type'].upper()
        if any(d in ctype for d in ['DATE', 'TIME', 'TIMESTAMP']):
            return True
        if any(d in name for d in ['date', '_dt', '_ts', '_time']):
            return True
        return False

    def _execute_dive(self, template: Dict) -> Optional[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(template['sql'])
            cols = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            return {
                'label': template['label'],
                'type': template.get('type', 'breakdown'),
                'columns': cols,
                'rows': rows[:20],
                'row_count': len(rows),
                'sql': template['sql'],
            }
        except Exception as e:
            logger.debug("Dive query failed: %s — %s", template['label'], e)
            return None


class DataGapDetector:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema = {}
        self._row_counts = {}
        self._null_profiles = {}
        self._load_profile()

    def _load_profile(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            for t in tables:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
                self._schema[t.lower()] = {c[1].lower(): c[2] for c in cols}

                rc = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                self._row_counts[t.lower()] = rc[0] if rc else 0

                null_info = {}
                for c in cols:
                    col_name = c[1]
                    try:
                        null_cnt = conn.execute(
                            f"SELECT COUNT(*) FROM {t} WHERE {col_name} IS NULL"
                        ).fetchone()[0]
                        total = self._row_counts[t.lower()]
                        null_info[col_name.lower()] = round(null_cnt / max(total, 1) * 100, 1)
                    except:
                        pass
                self._null_profiles[t.lower()] = null_info

            conn.close()
        except Exception as e:
            logger.debug("Profile failed: %s", e)

    def detect(self, question: str, sql: str, error: str,
               rows: List, columns: List) -> Dict:
        issues = []
        severity = 'none'
        q = question.lower()

        if error:
            human_msg = self._humanize_error(error, sql)
            issues.append({'type': 'error', 'message': human_msg, 'severity': 'critical'})
            severity = 'critical'

        elif not rows and sql:
            issues.append({
                'type': 'no_results',
                'message': self._explain_no_results(question, sql),
                'severity': 'warning'
            })
            severity = 'warning'

        elif rows:
            null_issues = self._check_nulls(rows, columns)
            issues.extend(null_issues)

            row_issue = self._check_row_count(rows, sql)
            if row_issue:
                issues.append(row_issue)

            if issues:
                severity = max(
                    (i.get('severity', 'info') for i in issues),
                    key=lambda s: {'info': 0, 'warning': 1, 'critical': 2}.get(s, 0)
                )

        missing = self._check_missing_entities(question)
        if missing:
            issues.extend(missing)
            if not severity or severity == 'none':
                severity = 'info'

        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'severity': severity,
            'data_quality_score': self._quality_score(rows, columns),
        }

    def _humanize_error(self, error: str, sql: str) -> str:
        err = error.lower()

        if 'no such table' in err:
            m = re.search(r'no such table:\s*(\S+)', err)
            table = m.group(1) if m else 'unknown'
            available = ', '.join(sorted(self._schema.keys()))
            return (
                f"The table '{table}' doesn't exist in this database. "
                f"Available tables: {available}. "
                f"Try rephrasing your question using one of these data sources."
            )

        if 'no such column' in err:
            m = re.search(r'no such column:\s*(\S+)', err)
            col = m.group(1) if m else 'unknown'
            return (
                f"The column '{col}' was not found. This data point may not be "
                f"available in the current dataset, or it might be named differently."
            )

        if 'ambiguous column' in err:
            return (
                "The query references a column that exists in multiple tables. "
                "Try being more specific about which data source you mean."
            )

        if 'syntax error' in err:
            return (
                "The query couldn't be processed correctly. "
                "Try rephrasing your question in simpler terms."
            )

        return f"Query issue: {error[:200]}. Try rephrasing your question."

    def _explain_no_results(self, question: str, sql: str) -> str:
        tables_in_sql = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.I)
        tables = [t.lower() for pair in tables_in_sql for t in pair if t]

        empty_tables = [t for t in tables if self._row_counts.get(t, 0) == 0]
        if empty_tables:
            return (
                f"No results found because table(s) {', '.join(empty_tables)} "
                f"contain no data. The data may not be loaded for this period."
            )

        if 'WHERE' in sql.upper():
            return (
                "No data matches your specific criteria. Try broadening your "
                "search — for example, remove date filters or use a wider category."
            )

        return (
            "The query returned no results. The requested data may not exist "
            "in the current dataset. Try a different question or check data availability."
        )

    def _check_nulls(self, rows, columns):
        issues = []
        if not rows or not columns:
            return issues

        for col_idx, col_name in enumerate(columns):
            null_count = sum(1 for r in rows if col_idx < len(r) and r[col_idx] is None)
            null_pct = null_count / len(rows) * 100
            if null_pct > 20:
                issues.append({
                    'type': 'null_data',
                    'message': (
                        f"{col_name.replace('_',' ')} has {null_pct:.0f}% missing values "
                        f"— results may be incomplete"
                    ),
                    'severity': 'warning' if null_pct > 50 else 'info'
                })
        return issues[:2]

    def _check_row_count(self, rows, sql):
        if len(rows) == 1 and 'GROUP BY' in sql.upper():
            return {
                'type': 'single_group',
                'message': "Only one group found — the grouping column may have limited variety",
                'severity': 'info'
            }
        return None

    def _check_missing_entities(self, question):
        issues = []
        q = question.lower()
        q_words = set(re.findall(r'[a-z]+', q))

        known_stems = set()
        for tbl in self._schema:
            tbl_lower = tbl.lower()
            known_stems.add(tbl_lower)
            if tbl_lower.endswith('s') and len(tbl_lower) > 2:
                known_stems.add(tbl_lower[:-1])
            for col_name in [c.lower() for c in (self._schema.get(tbl, {}) or {})]:
                known_stems.update(col_name.replace('_', ' ').split())

        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'my', 'me', 'i', 'we', 'our', 'us', 'you', 'your', 'they', 'their',
            'it', 'its', 'he', 'she', 'him', 'her', 'by', 'for', 'in', 'of',
            'to', 'and', 'or', 'not', 'with', 'from', 'this', 'that', 'these',
            'those', 'at', 'on', 'as', 'but', 'if', 'so', 'than', 'then',
            'about', 'into', 'through', 'between', 'after', 'before',
            'how', 'what', 'which', 'who', 'where', 'when', 'why',
            'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could',
            'will', 'would', 'should', 'may', 'might', 'shall',
            'show', 'give', 'get', 'tell', 'list', 'find', 'see',
            'look', 'display', 'describe', 'explain', 'want', 'need',
            'like', 'know', 'think', 'make', 'take', 'put', 'let',
            'compare', 'compared', 'comparing', 'comparison',
            'change', 'changed', 'changes', 'changing',
            'differ', 'different', 'difference', 'differs',
            'increase', 'increased', 'increasing', 'decrease', 'decreased',
            'decreasing', 'affect', 'affected', 'affecting', 'impact',
            'improve', 'improved', 'improving', 'reduce', 'reduced',
            'analyze', 'analyse', 'analyzed', 'analysis', 'identify',
            'determine', 'evaluate', 'assess', 'calculate', 'compute',
            'total', 'average', 'count', 'number', 'rate', 'per',
            'sum', 'mean', 'median', 'minimum', 'maximum', 'percent',
            'percentage', 'proportion', 'ratio', 'amount', 'value',
            'volume', 'frequency', 'distribution', 'breakdown',
            'highest', 'lowest', 'most', 'least', 'best', 'worst',
            'top', 'bottom', 'first', 'last', 'largest', 'smallest',
            'over', 'time', 'trend', 'period', 'month', 'months',
            'year', 'years', 'quarter', 'quarterly', 'week', 'weekly',
            'daily', 'annual', 'annually', 'monthly', 'yearly',
            'recent', 'recently', 'current', 'currently', 'last',
            'previous', 'prior', 'since', 'during', 'today',
            'across', 'within', 'among', 'per', 'each', 'every',
            'all', 'any', 'some', 'many', 'few', 'much', 'more',
            'less', 'other', 'same', 'group', 'grouped', 'category',
            'high', 'low', 'expensive', 'costly', 'cheap', 'dollar',
            'dollars', 'cost', 'costs', 'price', 'priced', 'spend',
            'spending', 'paid', 'billed', 'charged',
            'patient', 'patients', 'member', 'members', 'provider',
            'providers', 'clinical', 'medical', 'health', 'healthcare',
            'care', 'treatment', 'service', 'services',
            'also', 'just', 'only', 'very', 'really', 'actually',
            'specifically', 'particular', 'particular', 'overall',
            'general', 'generally', 'typically', 'usually', 'often',
            'sometimes', 'always', 'never', 'still', 'yet', 'already',
            'even', 'though', 'however', 'therefore', 'please',
            'data', 'information', 'details', 'results', 'records',
            'report', 'summary', 'overview', 'view', 'table',
            'metric', 'metrics', 'measure', 'measures', 'measurement',
            'indicator', 'indicators', 'benchmark', 'benchmarks',
            'performance', 'utilization', 'outcome', 'outcomes',
            'efficiency', 'effectiveness', 'quality', 'score', 'scores',
            'scoring', 'rating', 'ratings', 'ranking', 'rankings',
            'index', 'factor', 'factors', 'variable', 'variables',
            'dimension', 'dimensions', 'segment', 'segments',
            'female', 'females', 'male', 'males', 'gender', 'genders',
            'age', 'ages', 'elderly', 'pediatric', 'adult', 'adults',
            'population', 'demographic', 'demographics', 'cohort', 'cohorts',
            'ethnicity', 'race', 'language', 'region', 'regions',
            'admission', 'admissions', 'readmission', 'readmissions',
            'discharge', 'discharged', 'inpatient', 'outpatient',
            'emergency', 'chronic', 'acute', 'diagnosis', 'diagnoses',
            'procedure', 'procedures', 'prescription', 'prescriptions',
            'referral', 'referrals', 'appointment', 'appointments',
            'encounter', 'encounters', 'claim', 'claims',
            'denial', 'denials', 'denied', 'approved', 'approval',
            'specialty', 'specialties', 'condition', 'conditions',
            'comorbidity', 'comorbidities', 'medication', 'medications',
            'preventive', 'screening', 'compliance', 'adherence',
            'stratification', 'anomaly', 'anomalies', 'outlier', 'outliers',
            'driving', 'driven', 'driver', 'drivers',
        }

        candidate_words = q_words - stop_words - known_stems
        for cw in list(candidate_words):
            for stem in known_stems:
                if cw in stem or stem in cw:
                    candidate_words.discard(cw)
                    break

        for cw in sorted(candidate_words):
            if len(cw) > 5:
                issues.append({
                    'type': 'missing_data',
                    'message': (
                        f"'{cw}' is not recognized in the current dataset. "
                        f"Available data sources: {', '.join(sorted(self._schema.keys()))}."
                    ),
                    'severity': 'info'
                })

        return issues[:2]

    def _quality_score(self, rows, columns):
        if not rows:
            return 0
        total_cells = len(rows) * max(len(columns), 1)
        null_cells = sum(1 for r in rows for v in r if v is None)
        return round((1 - null_cells / max(total_cells, 1)) * 100)


class RootCauseAnalyzer:

    def __init__(self, db_path: str):
        self.db_path = db_path

    def investigate(self, anomalies: List[Dict], original_sql: str,
                    original_question: str) -> List[Dict]:
        investigations = []

        for anomaly in anomalies[:2]:
            col = anomaly.get('column', '')
            label = anomaly.get('label', '')
            atype = anomaly.get('type', '')

            drill_queries = self._generate_drill_queries(
                col, label, atype, original_sql
            )

            for dq in drill_queries[:2]:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.execute(dq['sql'])
                    cols = [d[0] for d in cursor.description] if cursor.description else []
                    rows = cursor.fetchall()
                    conn.close()

                    if rows:
                        investigations.append({
                            'anomaly': anomaly.get('message', ''),
                            'investigation': dq['label'],
                            'columns': cols,
                            'rows': rows[:10],
                            'finding': self._summarize_finding(rows, cols, dq['label']),
                        })
                except Exception as e:
                    logger.debug("Root cause query failed: %s", e)

        return investigations

    def _generate_drill_queries(self, column, label, anomaly_type,
                                original_sql) -> List[Dict]:
        queries = []

        table_match = re.search(r'FROM\s+(\w+)', original_sql, re.I)
        if not table_match:
            return queries

        main_table = table_match.group(1)

        if label and label != 'Row':
            queries.append({
                'label': f'Details for {label}',
                'sql': f"SELECT * FROM {main_table} WHERE {main_table}.ROWID IN "
                       f"(SELECT ROWID FROM {main_table} LIMIT 20)"
            })

        if anomaly_type == 'concentration' and column:
            queries.append({
                'label': f'{column} distribution',
                'sql': f"SELECT {column}, COUNT(*) as frequency FROM {main_table} "
                       f"GROUP BY {column} ORDER BY frequency DESC LIMIT 10"
            })

        return queries

    def _summarize_finding(self, rows, cols, label):
        if not rows:
            return "No additional data found."
        return f"Found {len(rows)} related records for '{label}'."
