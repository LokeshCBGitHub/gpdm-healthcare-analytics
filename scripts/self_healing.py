import sqlite3
import json
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger('kp.self_healing')
logger.setLevel(logging.DEBUG)


class SelfHealingEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ground_truth = {}
        self._fix_log = []
        self._rules = self._build_validation_rules()
        self._compute_ground_truth()

    def _compute_ground_truth(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM members")
            self._ground_truth['total_members'] = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM claims")
            self._ground_truth['total_claims'] = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM encounters")
            self._ground_truth['total_encounters'] = c.fetchone()[0]

            c.execute("SELECT SUM(CAST(PAID_AMOUNT AS REAL)), SUM(CAST(BILLED_AMOUNT AS REAL)), SUM(CAST(ALLOWED_AMOUNT AS REAL)) FROM claims")
            paid, billed, allowed = c.fetchone()
            self._ground_truth['total_paid'] = paid or 0
            self._ground_truth['total_billed'] = billed or 0
            self._ground_truth['total_allowed'] = allowed or 0

            c.execute("SELECT AVG(CAST(RISK_SCORE AS REAL)), MIN(CAST(RISK_SCORE AS REAL)), MAX(CAST(RISK_SCORE AS REAL)) FROM members")
            avg_r, min_r, max_r = c.fetchone()
            self._ground_truth['avg_risk'] = avg_r or 0
            self._ground_truth['risk_range'] = (min_r or 0, max_r or 5)

            c.execute("SELECT COUNT(DISTINCT KP_REGION) FROM members")
            self._ground_truth['region_count'] = c.fetchone()[0]

            c.execute("SELECT COUNT(DISTINCT PLAN_TYPE) FROM members")
            self._ground_truth['plan_type_count'] = c.fetchone()[0]

            c.execute("SELECT CLAIM_STATUS, COUNT(*) FROM claims GROUP BY CLAIM_STATUS")
            self._ground_truth['claim_status_dist'] = dict(c.fetchall())

            conn.close()
            logger.info(f"Ground truth computed: {len(self._ground_truth)} metrics")
        except Exception as e:
            logger.error(f"Ground truth computation failed: {e}")

    def _build_validation_rules(self) -> List[Dict]:
        return [
            {'type': 'percentage', 'check': lambda v: 0 <= v <= 100,
             'fix': lambda v: max(0, min(100, v)),
             'msg': 'Percentage out of range [0, 100]'},

            {'type': 'pmpm', 'check': lambda v: 0 < v < 50000,
             'fix': lambda v: abs(v) if v < 0 else v,
             'msg': 'PMPM value unreasonable'},

            {'type': 'count', 'check': lambda v: v >= 0 and isinstance(v, (int, float)),
             'fix': lambda v: max(0, int(v)),
             'msg': 'Count should be non-negative integer'},

            {'type': 'rate_per_1000', 'check': lambda v: 0 <= v <= 10000,
             'fix': lambda v: max(0, min(10000, v)),
             'msg': 'Rate per 1000 out of range'},

            {'type': 'risk_score', 'check': lambda v: 0 <= v <= 10,
             'fix': lambda v: max(0, min(10, v)),
             'msg': 'Risk score out of range [0, 10]'},

            {'type': 'star_rating', 'check': lambda v: 0 <= v <= 5,
             'fix': lambda v: max(0, min(5, v)),
             'msg': 'Star rating out of range [0, 5]'},
        ]


    def validate_value(self, value: Any, value_type: str, context: str = "") -> Tuple[Any, bool, str]:
        if value is None:
            return 0, True, f"NULL value corrected to 0 [{context}]"

        if isinstance(value, float):
            if math.isnan(value):
                return 0, True, f"NaN corrected to 0 [{context}]"
            if math.isinf(value):
                return 0, True, f"Inf corrected to 0 [{context}]"

        for rule in self._rules:
            if rule['type'] == value_type:
                if not rule['check'](value):
                    fixed = rule['fix'](value)
                    msg = f"{rule['msg']}: {value} -> {fixed} [{context}]"
                    logger.warning(f"Self-heal: {msg}")
                    self._fix_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'context': context,
                        'original': value,
                        'fixed': fixed,
                        'rule': rule['type'],
                        'message': msg
                    })
                    return fixed, True, msg

        return value, False, ""

    def validate_dashboard(self, dashboard: Dict, dashboard_type: str) -> Dict:
        issues_found = 0
        issues_fixed = 0

        def _walk_and_fix(obj, path=""):
            nonlocal issues_found, issues_fixed

            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}"

                    if isinstance(v, dict) and 'value' in v:
                        fmt = v.get('format', '')
                        val = v['value']

                        if isinstance(val, (int, float)):
                            vtype = 'percentage' if fmt == 'percent' else \
                                    'pmpm' if fmt == 'currency' else \
                                    'count' if fmt == 'number' else None

                            if vtype:
                                fixed_val, was_fixed, msg = self.validate_value(val, vtype, new_path)
                                if was_fixed:
                                    v['value'] = fixed_val
                                    v['_self_healed'] = True
                                    v['_heal_note'] = msg
                                    issues_found += 1
                                    issues_fixed += 1

                    elif isinstance(v, (int, float)) and isinstance(v, float):
                        if math.isnan(v) or math.isinf(v):
                            obj[k] = 0
                            issues_found += 1
                            issues_fixed += 1

                    _walk_and_fix(v, new_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _walk_and_fix(item, f"{path}[{i}]")

        _walk_and_fix(dashboard)

        if issues_found > 0:
            logger.info(f"Self-healing [{dashboard_type}]: {issues_found} issues found, {issues_fixed} fixed")

        return dashboard


    def check_cross_dashboard_consistency(self, dashboards: Dict[str, Dict]) -> List[Dict]:
        issues = []

        member_counts = {}
        for name, db in dashboards.items():
            self._extract_member_count(db, name, member_counts)

        if len(set(member_counts.values())) > 1:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Total members inconsistent across dashboards',
                'details': member_counts,
                'fix': f'Expected: {self._ground_truth.get("total_members", "unknown")}'
            })

        return issues

    def _extract_member_count(self, obj, dashboard_name, results):
        if isinstance(obj, dict):
            if 'total_members' in obj:
                val = obj['total_members']
                if isinstance(val, dict):
                    val = val.get('value', val)
                results[dashboard_name] = val
            for v in obj.values():
                self._extract_member_count(v, dashboard_name, results)


    def validate_query_result(self, question: str, result: Dict) -> Dict:
        fixes_applied = []

        if not result.get('data') and not result.get('answer'):
            result['_self_heal_warning'] = 'Query returned no data'
            return result

        data = result.get('data', [])
        if isinstance(data, list):
            for i, row in enumerate(data):
                if isinstance(row, dict):
                    for col, val in row.items():
                        if isinstance(val, float):
                            if math.isnan(val) or math.isinf(val):
                                row[col] = 0
                                fixes_applied.append(f"Fixed NaN/Inf in {col} row {i}")

        question_lower = question.lower()

        if 'how many members' in question_lower or 'total members' in question_lower:
            gt = self._ground_truth.get('total_members')
            if gt and result.get('data'):
                reported = self._extract_first_numeric(result['data'])
                if reported and abs(reported - gt) / gt > 0.05:
                    result['_self_heal_warning'] = (
                        f"Reported {reported} members but ground truth is {gt} "
                        f"(diff={abs(reported-gt)/gt*100:.1f}%)"
                    )

        if fixes_applied:
            result['_self_heal_fixes'] = fixes_applied

        return result

    def _extract_first_numeric(self, data):
        if isinstance(data, list) and data:
            row = data[0]
            if isinstance(row, dict):
                for v in row.values():
                    if isinstance(v, (int, float)):
                        return v
            elif isinstance(row, (list, tuple)):
                for v in row:
                    if isinstance(v, (int, float)):
                        return v
        return None


    def detect_suspicious_zeros(self, dashboard: Dict, dashboard_type: str) -> List[Dict]:
        suspicious = []
        never_zero_patterns = [
            'total_members', 'pmpm_revenue', 'pmpm_cost', 'total_claims',
            'avg_risk', 'clean_claims_rate', 'retention_rate',
            'encounters', 'per_1000', 'pharmacy_pmpm'
        ]

        def _check(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}"
                    if isinstance(v, dict) and 'value' in v:
                        val = v['value']
                        if val == 0 or val == 0.0:
                            for pattern in never_zero_patterns:
                                if pattern in k.lower():
                                    suspicious.append({
                                        'path': new_path,
                                        'metric': k,
                                        'value': val,
                                        'severity': 'HIGH',
                                        'message': f'{k} is zero — this should never happen for a real dataset'
                                    })
                    _check(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")

        _check(dashboard)

        if suspicious:
            logger.warning(f"Suspicious zeros in {dashboard_type}: {len(suspicious)} found")

        return suspicious


    def run_full_health_check(self, dashboard_engine) -> Dict:
        report = {
            'timestamp': datetime.now().isoformat(),
            'dashboards': {},
            'cross_consistency': [],
            'suspicious_zeros': [],
            'fixes_applied': [],
            'grade': 'A',
            'total_issues': 0
        }

        dashboards = {}
        dashboard_methods = {
            'financial': dashboard_engine.get_financial_performance,
            'stars': dashboard_engine.get_stars_performance,
            'member_experience': dashboard_engine.get_member_experience,
            'rada': dashboard_engine.get_risk_adjustment_coding,
            'membership': dashboard_engine.get_membership_market_share,
            'utilization': dashboard_engine.get_service_utilization,
            'executive_summary': dashboard_engine.get_executive_summary,
        }

        for name, method in dashboard_methods.items():
            try:
                db = method()
                dashboards[name] = db

                validated = self.validate_dashboard(db, name)

                zeros = self.detect_suspicious_zeros(db, name)

                report['dashboards'][name] = {
                    'status': 'OK' if not zeros else 'WARNING',
                    'suspicious_zeros': len(zeros),
                    'details': zeros
                }
                report['suspicious_zeros'].extend(zeros)

            except Exception as e:
                report['dashboards'][name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                report['total_issues'] += 1

        report['cross_consistency'] = self.check_cross_dashboard_consistency(dashboards)
        report['total_issues'] += len(report['cross_consistency'])
        report['total_issues'] += len(report['suspicious_zeros'])

        report['fixes_applied'] = self._fix_log[-50:]

        total = report['total_issues']
        if total == 0:
            report['grade'] = 'A'
        elif total <= 2:
            report['grade'] = 'B'
        elif total <= 5:
            report['grade'] = 'C'
        elif total <= 10:
            report['grade'] = 'D'
        else:
            report['grade'] = 'F'

        logger.info(f"Health check complete: grade={report['grade']}, issues={total}")
        return report

    def get_fix_log(self) -> List[Dict]:
        return self._fix_log

    def get_ground_truth(self) -> Dict:
        return self._ground_truth
