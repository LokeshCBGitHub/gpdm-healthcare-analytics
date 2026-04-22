from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DQ_DIR = DATA_DIR / "dq"


ROLE_PATTERNS = {
    'id': {
        'names': r'^(claim_id|encounter_id|diagnosis_id|referral_id|rx_id|prescription_id)$',
        'description': 'Primary identifier — should be unique per row',
    },
    'foreign_key': {
        'names': r'(member_id|provider_id|facility_id|rendering_npi|billing_npi|pcp_npi|^npi$)',
        'description': 'Foreign key reference — uniqueness not expected',
    },
    'date': {
        'names': r'(date|_dt$|_datetime$|timestamp|_time$|_at$|^dos$)',
        'description': 'Date or timestamp field',
    },
    'amount': {
        'names': r'(amount|_amt$|cost|charge|payment|copay|coinsurance|deductible|fee|price|total_paid|total_billed|avg_paid|avg_billed)',
        'description': 'Financial amount',
    },
    'code': {
        'names': r'(^icd10|^cpt_code|^ndc$|^hcc|^drg|diagnosis_code|procedure_code|drug_code|^cpt$)',
        'description': 'Healthcare code (ICD-10, CPT, NDC, etc.)',
    },
    'category': {
        'names': r'(status|type|category|gender|sex|race|region|facility|department|specialty|plan|lob|visit_type|encounter_type|claim_type|provider_type)',
        'description': 'Categorical / enumerated value',
    },
    'description': {
        'names': r'(description|_desc$|_name$|_text$|notes|comments|reason|narrative)',
        'description': 'Free text description',
    },
    'flag': {
        'names': r'(^is_|^has_|_flag$|_yn$|accepts_|chronic)',
        'description': 'Boolean flag',
    },
    'score': {
        'names': r'(score|rating|rank|percentile|risk_score|priority)',
        'description': 'Numeric score or rating',
    },
}

CODE_VALIDATORS = {
    'npi':  (r'^\d{10}$', 'Must be exactly 10 digits'),
    'icd10': (r'^[A-Z]\d{2}\.?\d{0,4}$', 'Letter + 2 digits + optional decimal digits'),
    'cpt':  (r'^\d{5}$', 'Must be exactly 5 digits'),
    'ndc':  (r'^\d{11}$', 'Must be exactly 11 digits'),
    'hcc':  (r'^\d{1,3}$', 'Must be 1-3 digits'),
    'drg':  (r'^\d{1,4}$', 'Must be 1-4 digits'),
}

WEIGHT_PROFILES = {
    'id':          {'completeness': 0.30, 'uniqueness': 0.40, 'validity': 0.20, 'consistency': 0.10},
    'foreign_key': {'completeness': 0.50, 'uniqueness': 0.05, 'validity': 0.25, 'consistency': 0.20},
    'date':        {'completeness': 0.35, 'uniqueness': 0.05, 'validity': 0.40, 'consistency': 0.20},
    'amount':      {'completeness': 0.40, 'uniqueness': 0.05, 'validity': 0.30, 'consistency': 0.25},
    'code':        {'completeness': 0.30, 'uniqueness': 0.05, 'validity': 0.45, 'consistency': 0.20},
    'category':    {'completeness': 0.40, 'uniqueness': 0.05, 'validity': 0.30, 'consistency': 0.25},
    'description': {'completeness': 0.50, 'uniqueness': 0.05, 'validity': 0.10, 'consistency': 0.35},
    'flag':        {'completeness': 0.40, 'uniqueness': 0.05, 'validity': 0.35, 'consistency': 0.20},
    'score':       {'completeness': 0.40, 'uniqueness': 0.10, 'validity': 0.30, 'consistency': 0.20},
    'default':     {'completeness': 0.40, 'uniqueness': 0.15, 'validity': 0.25, 'consistency': 0.20},
}

DATE_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}$',
    r'^\d{2}/\d{2}/\d{4}$',
    r'^\d{4}/\d{2}/\d{2}$',
    r'^\d{4}-\d{2}-\d{2}[T ]\d{2}',
]


class DQEngine:

    SKIP_PREFIXES = ('_gpdm_', '_dq_', '_data_', '_schema_', '_audit_',
                     '_table_', '_referential_', 'dim_date', 'sqlite_')

    def __init__(self, db_path: str, sample_size: int = 50000):
        self.db_path = db_path
        self.sample_size = sample_size
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


    @staticmethod
    def detect_role(col_name: str) -> str:
        lower = col_name.lower().strip()
        for role, spec in ROLE_PATTERNS.items():
            if re.search(spec['names'], lower):
                return role
        return 'default'

    @staticmethod
    def detect_code_type(col_name: str) -> Optional[str]:
        lower = col_name.lower().strip()
        if re.search(r'(desc|name|text|narrative|reason|_id$|^id$)', lower):
            return None
        for code_type in CODE_VALIDATORS:
            if code_type in lower:
                return code_type
        if lower == 'npi':
            return 'npi'
        if lower.endswith('_npi') and 'rendering' in lower:
            return 'npi'
        return None


    def check_completeness(self, table: str, col: str, total_rows: int) -> float:
        if total_rows == 0:
            return 100.0
        conn = self._get_conn()
        null_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE [{col}] IS NULL OR TRIM(CAST([{col}] AS TEXT)) = ''"
        ).fetchone()[0]
        return round((1 - null_count / total_rows) * 100, 2)


    def check_uniqueness(self, table: str, col: str, role: str, total_rows: int) -> float:
        if role in ('category', 'flag', 'description', 'amount', 'score', 'foreign_key'):
            return 100.0

        conn = self._get_conn()
        row = conn.execute(
            f"SELECT COUNT(DISTINCT [{col}]) as dist, "
            f"COUNT(*) as non_null "
            f"FROM {table} WHERE [{col}] IS NOT NULL AND TRIM(CAST([{col}] AS TEXT)) != ''"
        ).fetchone()

        distinct = row[0] or 0
        non_null = row[1] or 1
        return round((distinct / non_null) * 100, 2)


    def check_validity(self, table: str, col: str, role: str, total_rows: int) -> float:
        conn = self._get_conn()

        code_type = self.detect_code_type(col)

        if code_type and code_type in CODE_VALIDATORS:
            pattern, _ = CODE_VALIDATORS[code_type]
            sample = conn.execute(
                f"SELECT [{col}] FROM {table} "
                f"WHERE [{col}] IS NOT NULL AND TRIM(CAST([{col}] AS TEXT)) != '' "
                f"LIMIT {self.sample_size}"
            ).fetchall()
            if not sample:
                return 100.0
            valid = sum(1 for r in sample if re.match(pattern, str(r[0]).strip()))
            return round((valid / len(sample)) * 100, 2)

        if role == 'date':
            sample = conn.execute(
                f"SELECT [{col}] FROM {table} "
                f"WHERE [{col}] IS NOT NULL AND TRIM(CAST([{col}] AS TEXT)) != '' "
                f"LIMIT {self.sample_size}"
            ).fetchall()
            if not sample:
                return 100.0
            valid = 0
            for r in sample:
                val = str(r[0]).strip()
                if any(re.match(p, val) for p in DATE_PATTERNS):
                    valid += 1
            return round((valid / len(sample)) * 100, 2)

        if role == 'amount':
            sample = conn.execute(
                f"SELECT [{col}] FROM {table} "
                f"WHERE [{col}] IS NOT NULL AND TRIM(CAST([{col}] AS TEXT)) != '' "
                f"LIMIT {self.sample_size}"
            ).fetchall()
            if not sample:
                return 100.0
            valid = 0
            for r in sample:
                try:
                    val = float(str(r[0]).replace(',', ''))
                    if val >= 0:
                        valid += 1
                except (ValueError, TypeError):
                    pass
            return round((valid / len(sample)) * 100, 2)

        return 100.0


    def check_consistency(self, table: str, col: str, all_cols: List[str],
                          total_rows: int) -> float:
        conn = self._get_conn()
        lower = col.lower()

        if 'paid' in lower and ('amount' in lower or lower.endswith('_amt')):
            billed_col = None
            for c in all_cols:
                cl = c.lower()
                if 'billed' in cl and ('amount' in cl or cl.endswith('_amt')):
                    billed_col = c
                    break
            if billed_col:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE CAST([{col}] AS REAL) > CAST([{billed_col}] AS REAL) * 1.05 "
                    f"AND [{col}] IS NOT NULL AND [{billed_col}] IS NOT NULL "
                    f"AND CAST([{col}] AS REAL) > 0"
                ).fetchone()
                violations = row[0]
                checked = conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE [{col}] IS NOT NULL AND [{billed_col}] IS NOT NULL "
                    f"AND CAST([{col}] AS REAL) > 0"
                ).fetchone()[0]
                if checked > 0:
                    return round((1 - violations / checked) * 100, 2)

        if 'discharge' in lower and ('date' in lower or lower.endswith('_dt')):
            admit_col = None
            for c in all_cols:
                cl = c.lower()
                if 'admit' in cl and ('date' in cl or cl.endswith('_dt')):
                    admit_col = c
                    break
            if admit_col:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE [{col}] < [{admit_col}] "
                    f"AND [{col}] IS NOT NULL AND [{col}] != '' "
                    f"AND [{admit_col}] IS NOT NULL AND [{admit_col}] != ''"
                ).fetchone()
                violations = row[0]
                checked = conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE [{col}] IS NOT NULL AND [{col}] != '' "
                    f"AND [{admit_col}] IS NOT NULL AND [{admit_col}] != ''"
                ).fetchone()[0]
                if checked > 0:
                    return round((1 - violations / checked) * 100, 2)

        return 100.0


    def check_table(self, table: str) -> Dict[str, Any]:
        conn = self._get_conn()
        t0 = time.time()

        total_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols_info]

        column_checks = {}
        column_scores = {}

        for col in col_names:
            role = self.detect_role(col)
            code_type = self.detect_code_type(col) if role == 'code' else None

            completeness = self.check_completeness(table, col, total_rows)
            uniqueness = self.check_uniqueness(table, col, role, total_rows)
            validity = self.check_validity(table, col, role, total_rows)
            consistency = self.check_consistency(table, col, col_names, total_rows)

            weights = WEIGHT_PROFILES.get(role, WEIGHT_PROFILES['default'])
            col_score = (
                completeness * weights['completeness'] +
                uniqueness   * weights['uniqueness'] +
                validity     * weights['validity'] +
                consistency  * weights['consistency']
            )

            issues = []
            if completeness < 90:
                issues.append(f"Low completeness: {completeness}% populated")
            if role == 'id' and uniqueness < 99:
                issues.append(f"Duplicate values: only {uniqueness}% unique")
            if validity < 95:
                issues.append(f"Format issues: {validity}% valid")
            if consistency < 95:
                issues.append(f"Cross-field issues: {consistency}% consistent")

            column_checks[col] = {
                'role': role,
                'code_type': code_type,
                'completeness': completeness,
                'uniqueness': uniqueness,
                'validity': validity,
                'consistency': consistency,
                'column_score': round(col_score, 2),
                'weights_used': role,
                'issues': issues,
            }
            column_scores[col] = round(col_score, 2)

        if column_scores:
            weighted_sum = 0
            weight_total = 0
            for col, score in column_scores.items():
                w = 1.5 if column_checks[col]['role'] == 'id' else 1.0
                weighted_sum += score * w
                weight_total += w
            dq_score = round(weighted_sum / weight_total, 2)
        else:
            dq_score = 0

        return {
            'table_name': table,
            'total_rows': total_rows,
            'total_columns': len(col_names),
            'checked_at': datetime.now().isoformat(),
            'column_checks': column_checks,
            'column_scores': column_scores,
            'dq_score': dq_score,
            'dq_grade': self._grade(dq_score),
            'elapsed_ms': round((time.time() - t0) * 1000, 1),
        }


    def run(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        conn = self._get_conn()
        t0 = time.time()

        if tables:
            all_tables = tables
        else:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            all_tables = [
                r[0] for r in rows
                if not any(r[0].lower().startswith(p) for p in self.SKIP_PREFIXES)
            ]

        log.info("DQ scan: %d tables", len(all_tables))

        reports = {}
        for table in all_tables:
            try:
                report = self.check_table(table)
                reports[table] = report
                log.info("  %-25s score=%.1f (%s)  %d cols  %s rows  %dms",
                         table, report['dq_score'], report['dq_grade'],
                         report['total_columns'], f"{report['total_rows']:,}",
                         report['elapsed_ms'])
            except Exception as e:
                log.warning("  %-25s ERROR: %s", table, e)
                reports[table] = {'table_name': table, 'error': str(e)}

        scores = [r['dq_score'] for r in reports.values() if 'dq_score' in r]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0

        grades = {}
        for r in reports.values():
            g = r.get('dq_grade', 'N/A')
            grades[g] = grades.get(g, 0) + 1

        issue_counts: Dict[str, int] = {}
        for r in reports.values():
            for col_data in r.get('column_checks', {}).values():
                for issue in col_data.get('issues', []):
                    issue_type = issue.split(':')[0]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_tables': len(reports),
            'average_dq_score': avg_score,
            'overall_grade': self._grade(avg_score),
            'grade_distribution': grades,
            'table_scores': [
                {
                    'table_name': r['table_name'],
                    'dq_score': r.get('dq_score', 0),
                    'grade': r.get('dq_grade', 'N/A'),
                    'total_rows': r.get('total_rows', 0),
                    'total_columns': r.get('total_columns', 0),
                }
                for r in reports.values() if 'dq_score' in r
            ],
            'issues_by_type': issue_counts,
            'elapsed_seconds': round(time.time() - t0, 1),
        }

        return {
            'summary': summary,
            'tables': reports,
        }


    def save(self, result: Dict[str, Any], output_dir: Optional[str] = None):
        out = Path(output_dir) if output_dir else DQ_DIR
        out.mkdir(parents=True, exist_ok=True)

        for table_name, report in result.get('tables', {}).items():
            path = out / f"{table_name}_dq.json"
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        summary_path = out / "dq_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(result['summary'], f, indent=2, default=str)

        log.info("DQ reports saved to %s", out)


    @staticmethod
    def _grade(score: float) -> str:
        if score >= 95: return 'A'
        if score >= 90: return 'B'
        if score >= 80: return 'C'
        if score >= 70: return 'D'
        return 'F'


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    ap = argparse.ArgumentParser(description='GPDM Data Quality Engine')
    ap.add_argument('--db', default=str(DATA_DIR / 'healthcare_production.db'),
                    help='Database path')
    ap.add_argument('--tables', nargs='*', help='Specific tables to check (default: all)')
    ap.add_argument('--output', default=None, help='Output directory (default: data/dq/)')
    ap.add_argument('--json', action='store_true', help='Print results as JSON')
    args = ap.parse_args()

    engine = DQEngine(args.db)
    try:
        result = engine.run(tables=args.tables)
        engine.save(result, output_dir=args.output)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            s = result['summary']
            print(f"\nDQ Summary: {s['average_dq_score']}% ({s['overall_grade']})")
            print(f"Tables: {s['total_tables']}  |  Time: {s['elapsed_seconds']}s")
            for ts in sorted(s['table_scores'], key=lambda x: x['dq_score']):
                print(f"  {ts['table_name']:25s}  {ts['dq_score']:5.1f}%  ({ts['grade']})")
    finally:
        engine.close()
