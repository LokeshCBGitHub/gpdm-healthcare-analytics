from __future__ import annotations
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SmartKPI:

    def __init__(self, name: str, query: str, unit: str = '',
                 direction: str = 'lower_is_better'):
        self.name = name
        self.query = query
        self.unit = unit
        self.direction = direction
        self.history: List[Tuple[str, float]] = []
        self.baseline: Optional[Dict] = None
        self.current_value: Optional[float] = None
        self.current_date: Optional[str] = None

    def compute(self, conn: sqlite3.Connection) -> Dict:
        try:
            row = conn.execute(self.query).fetchone()
            self.current_value = float(row[0]) if row and row[0] else None
            self.current_date = time.strftime('%Y-%m-%d')
        except Exception as e:
            return {'name': self.name, 'error': str(e)}

        if self.current_value is None:
            return {'name': self.name, 'value': None}

        result = {
            'name': self.name,
            'value': self.current_value,
            'unit': self.unit,
            'date': self.current_date,
        }

        if self.baseline:
            result['anomaly'] = self._detect_anomaly()

        if len(self.history) >= 3:
            result['trend'] = self._analyze_trend()
            result['forecast'] = self._forecast(periods=3)

        if len(self.history) >= 5:
            result['confidence_interval'] = self._confidence_interval()

        result['explanation'] = self._explain(result)

        return result

    def learn_history(self, conn: sqlite3.Connection, history_query: str):
        try:
            rows = conn.execute(history_query).fetchall()
            self.history = [(str(r[0]), float(r[1])) for r in rows if r[1] is not None]
            if len(self.history) >= 3:
                values = np.array([v for _, v in self.history])
                self.baseline = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'p5': float(np.percentile(values, 5)),
                    'p95': float(np.percentile(values, 95)),
                    'n': len(values),
                }
        except Exception as e:
            logger.debug("History load failed for %s: %s", self.name, e)

    def _detect_anomaly(self) -> Dict:
        bl = self.baseline
        if not bl or bl['std'] < 1e-10:
            return {'is_anomaly': False, 'reason': 'insufficient_baseline'}

        z = (self.current_value - bl['mean']) / bl['std']
        is_anomaly = abs(z) > 2.0
        is_alarm = abs(z) > 3.0

        if self.direction == 'lower_is_better':
            is_bad = self.current_value > bl['p95']
        else:
            is_bad = self.current_value < bl['p5']

        return {
            'is_anomaly': is_anomaly,
            'is_alarm': is_alarm,
            'z_score': round(z, 2),
            'is_bad_direction': is_bad,
            'severity': 'CRITICAL' if is_alarm and is_bad else (
                'WARNING' if is_anomaly and is_bad else (
                'INFO' if is_anomaly else 'NORMAL')),
        }

    def _analyze_trend(self) -> Dict:
        values = np.array([v for _, v in self.history])
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        if values[0] != 0:
            pct_change = (values[-1] - values[0]) / abs(values[0]) * 100
        else:
            pct_change = 0

        direction = 'increasing' if slope > 0 else 'decreasing'
        if self.direction == 'lower_is_better':
            trend_quality = 'improving' if slope < 0 else 'worsening'
        else:
            trend_quality = 'improving' if slope > 0 else 'worsening'

        return {
            'direction': direction,
            'quality': trend_quality,
            'slope': round(float(slope), 4),
            'pct_change': round(pct_change, 1),
            'n_periods': len(values),
        }

    def _forecast(self, periods: int = 3) -> Dict:
        values = np.array([v for _, v in self.history])
        x = np.arange(len(values))

        if len(values) < 3:
            return {'status': 'insufficient_data'}

        slope, intercept = np.polyfit(x, values, 1)
        residuals = values - (slope * x + intercept)
        se = float(np.std(residuals))

        forecasts = []
        for i in range(1, periods + 1):
            next_x = len(values) + i - 1
            point = slope * next_x + intercept
            margin = 1.96 * se * math.sqrt(1 + 1/len(values) + (next_x - x.mean())**2 / np.sum((x - x.mean())**2))
            forecasts.append({
                'period': i,
                'point': round(float(point), 2),
                'lower': round(float(point - margin), 2),
                'upper': round(float(point + margin), 2),
            })

        return {'forecasts': forecasts, 'model': 'linear'}

    def _confidence_interval(self, alpha: float = 0.05) -> Dict:
        values = np.array([v for _, v in self.history])
        n = len(values)
        se = float(np.std(values, ddof=1) / math.sqrt(n))
        t_crit = 2.0 if n > 30 else 2.5 if n > 10 else 3.0
        return {
            'lower': round(float(np.mean(values) - t_crit * se), 2),
            'upper': round(float(np.mean(values) + t_crit * se), 2),
            'se': round(se, 4),
            'confidence': round(1 - alpha, 2),
        }

    def _explain(self, result: Dict) -> str:
        parts = []

        if self.unit == '%':
            parts.append(f"{self.name} is currently {self.current_value:.1f}%")
        elif self.unit == '$':
            parts.append(f"{self.name} is currently ${self.current_value:,.0f}")
        else:
            parts.append(f"{self.name} is currently {self.current_value:,.1f} {self.unit}".strip())

        anomaly = result.get('anomaly', {})
        if anomaly.get('severity') == 'CRITICAL':
            parts.append(f"This is CRITICAL — outside the normal range ({anomaly['z_score']:.1f} standard deviations from your data's normal range).")
        elif anomaly.get('severity') == 'WARNING':
            parts.append(f"This is elevated (outside the normal range by {anomaly['z_score']:.1f} standard deviations).")

        trend = result.get('trend', {})
        if trend:
            parts.append(f"The trend is {trend['quality']} ({trend['direction']} {abs(trend['pct_change']):.1f}% over {trend['n_periods']} periods).")

        forecast = result.get('forecast', {})
        if forecast.get('forecasts'):
            next_f = forecast['forecasts'][0]
            parts.append(f"Next period forecast: {next_f['point']:.1f} (range {next_f['lower']:.1f}–{next_f['upper']:.1f}).")

        return ' '.join(parts)


class RootCauseAnalyzer:

    DRILL_DIMENSIONS = [
        ('region', 'region'),
        ('encounter_type', 'encounter_type'),
        ('denial_reason', 'denial_reason'),
        ('month', "SUBSTR(service_date, 1, 7)"),
        ('provider_id', 'provider_id'),
        ('facility_id', 'facility_id'),
    ]

    def __init__(self):
        pass

    def analyze(self, conn: sqlite3.Connection,
                kpi_name: str,
                metric_sql: str,
                group_by_col: str = None,
                comparison: str = 'recent_vs_prior') -> Dict:
        findings = []

        if comparison == 'recent_vs_prior':
            recent_filter = "service_date >= date('now', '-90 days')"
            prior_filter = "service_date >= date('now', '-180 days') AND service_date < date('now', '-90 days')"
        else:
            recent_filter = "1=1"
            prior_filter = "1=1"

        for dim_name, dim_col in self.DRILL_DIMENSIONS:
            try:
                drill_sql = f"""
                    SELECT
                        {dim_col} as dimension_value,
                        SUM(CASE WHEN {recent_filter} THEN 1 ELSE 0 END) as recent_count,
                        SUM(CASE WHEN {prior_filter} THEN 1 ELSE 0 END) as prior_count,
                        SUM(CASE WHEN status='DENIED' AND ({recent_filter}) THEN 1 ELSE 0 END) as recent_denied,
                        SUM(CASE WHEN status='DENIED' AND ({prior_filter}) THEN 1 ELSE 0 END) as prior_denied,
                        AVG(CASE WHEN {recent_filter} THEN paid_amount END) as recent_avg_paid,
                        AVG(CASE WHEN {prior_filter} THEN paid_amount END) as prior_avg_paid
                    FROM claims_4m
                    WHERE {dim_col} IS NOT NULL
                      AND service_date >= date('now', '-180 days')
                    GROUP BY {dim_col}
                    HAVING recent_count > 100 AND prior_count > 100
                    ORDER BY recent_count DESC
                    LIMIT 20
                """
                rows = conn.execute(drill_sql).fetchall()
                cols = [d[0] for d in conn.execute(drill_sql).description] if rows else []

                for row in rows:
                    r = dict(zip(cols, row))
                    recent_denial_rate = (r['recent_denied'] / max(r['recent_count'], 1)) * 100
                    prior_denial_rate = (r['prior_denied'] / max(r['prior_count'], 1)) * 100
                    rate_change = recent_denial_rate - prior_denial_rate

                    recent_cost = float(r.get('recent_avg_paid', 0) or 0)
                    prior_cost = float(r.get('prior_avg_paid', 0) or 0)
                    cost_change = recent_cost - prior_cost

                    if abs(rate_change) > 2.0 or abs(cost_change) > 100:
                        findings.append({
                            'dimension': dim_name,
                            'value': r['dimension_value'],
                            'denial_rate_change': round(rate_change, 1),
                            'cost_change': round(cost_change, 0),
                            'recent_volume': r['recent_count'],
                            'prior_volume': r['prior_count'],
                            'impact': abs(rate_change) * r['recent_count'],
                        })

            except Exception as e:
                logger.debug("Drill %s failed: %s", dim_name, e)

        findings.sort(key=lambda x: x.get('impact', 0), reverse=True)

        narrative = self._generate_narrative(findings[:5])

        return {
            'kpi': kpi_name,
            'findings': findings[:10],
            'top_driver': findings[0] if findings else None,
            'narrative': narrative,
            'dimensions_analyzed': len(self.DRILL_DIMENSIONS),
        }

    def _generate_narrative(self, top_findings: List[Dict]) -> str:
        if not top_findings:
            return "No significant dimensional drivers found for this KPI change."

        parts = ["Breakdown of what's driving this change:"]
        for i, f in enumerate(top_findings[:3], 1):
            if f['denial_rate_change'] != 0:
                dir_word = "increased" if f['denial_rate_change'] > 0 else "decreased"
                parts.append(
                    f"{i}. {f['dimension']}='{f['value']}': denial rate {dir_word} by "
                    f"{abs(f['denial_rate_change']):.1f}pp "
                    f"({f['recent_volume']:,} recent claims)")
            if f['cost_change'] != 0:
                dir_word = "increased" if f['cost_change'] > 0 else "decreased"
                parts.append(
                    f"   Cost {dir_word} by ${abs(f['cost_change']):,.0f} per claim")

        return ' '.join(parts)


class KPICorrelationEngine:

    def __init__(self):
        self.correlations: Dict[str, Dict] = {}

    def compute_correlations(self, kpi_histories: Dict[str, List[float]],
                              max_lag: int = 6) -> Dict:
        names = list(kpi_histories.keys())
        results = {}

        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                a = np.array(kpi_histories[name_a], dtype=float)
                b = np.array(kpi_histories[name_b], dtype=float)
                min_len = min(len(a), len(b))
                if min_len < 5:
                    continue

                a = a[:min_len]
                b = b[:min_len]

                a_norm = (a - a.mean()) / (a.std() + 1e-10)
                b_norm = (b - b.mean()) / (b.std() + 1e-10)

                best_corr = 0.0
                best_lag = 0
                for lag in range(-max_lag, max_lag + 1):
                    if lag >= 0:
                        corr = float(np.mean(a_norm[lag:] * b_norm[:min_len-lag]))
                    else:
                        corr = float(np.mean(a_norm[:min_len+lag] * b_norm[-lag:]))
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                pair_key = f"{name_a}→{name_b}"
                if abs(best_corr) > 0.3:
                    leading = name_a if best_lag > 0 else (name_b if best_lag < 0 else 'simultaneous')
                    results[pair_key] = {
                        'correlation': round(best_corr, 3),
                        'lag': best_lag,
                        'leading_indicator': leading,
                        'lag_periods': abs(best_lag),
                        'relationship': (
                            f"{leading} leads by {abs(best_lag)} periods "
                            f"({'positive' if best_corr > 0 else 'inverse'} correlation)"
                        ),
                    }

        self.correlations = results
        return results


class SmartKPIEngine:

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.kpis: Dict[str, SmartKPI] = {}
        self._root_cause_analyzer = RootCauseAnalyzer()
        self.correlator = KPICorrelationEngine()
        self._define_kpis()

    def _define_kpis(self):
        self.kpis['denial_rate'] = SmartKPI(
            'Denial Rate', "SELECT denial_rate FROM _gpdm_kpi_facts",
            unit='%', direction='lower_is_better')

        self.kpis['avg_paid'] = SmartKPI(
            'Average Paid Amount',
            "SELECT avg_paid FROM _gpdm_kpi_facts",
            unit='$', direction='lower_is_better')

        self.kpis['total_members'] = SmartKPI(
            'Total Members',
            "SELECT total_members FROM _gpdm_kpi_facts",
            unit='', direction='higher_is_better')

        self.kpis['high_risk_members'] = SmartKPI(
            'High Risk Members',
            "SELECT COUNT(*) FROM _gpdm_hybrid_scores WHERE priority_score > 0.08",
            unit='count', direction='lower_is_better')

        self.kpis['actionable_interventions'] = SmartKPI(
            'Actionable Interventions',
            "SELECT COUNT(*) FROM _gpdm_hybrid_scores WHERE recommended_action != 'MONITOR'",
            unit='count', direction='higher_is_better')

    def compute_all(self, conn: sqlite3.Connection = None) -> Dict:
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        for name, kpi in self.kpis.items():
            if name == 'denial_rate':
                kpi.learn_history(conn, """
                    SELECT month,
                           ROUND(SUM(denied_count)*100.0/NULLIF(SUM(claim_count),0), 2)
                    FROM _gpdm_summary_monthly
                    GROUP BY month ORDER BY month
                """)
            elif name == 'avg_paid':
                kpi.learn_history(conn, """
                    SELECT month,
                           ROUND(SUM(total_paid)*1.0/NULLIF(SUM(claim_count),0), 2)
                    FROM _gpdm_summary_monthly
                    GROUP BY month ORDER BY month
                """)
            elif name == 'total_members':
                kpi.learn_history(conn, """
                    SELECT month, SUM(unique_members)
                    FROM _gpdm_summary_monthly
                    GROUP BY month ORDER BY month
                """)

        results = {}
        for name, kpi in self.kpis.items():
            results[name] = kpi.compute(conn)

        histories = {}
        for name, kpi in self.kpis.items():
            if kpi.history:
                histories[name] = [v for _, v in kpi.history]
        if len(histories) >= 2:
            results['_correlations'] = self.correlator.compute_correlations(histories)

        anomalies = [r for r in results.values()
                     if isinstance(r, dict) and r.get('anomaly', {}).get('is_anomaly')]
        results['_anomaly_summary'] = {
            'count': len(anomalies),
            'critical': sum(1 for a in anomalies
                           if a.get('anomaly', {}).get('severity') == 'CRITICAL'),
            'kpis': [a['name'] for a in anomalies if 'name' in a],
        }

        return results

    def root_cause_analysis(self, kpi_name: str,
                            conn: sqlite3.Connection = None) -> Dict:
        if conn is None:
            conn = sqlite3.connect(self.db_path)
        return self._root_cause_analyzer.analyze(conn, kpi_name, '')

    def cross_kpi_correlations(self, conn: sqlite3.Connection = None) -> Dict:
        if conn is None:
            conn = sqlite3.connect(self.db_path)
        for name, kpi in self.kpis.items():
            if not kpi.history:
                if name == 'denial_rate':
                    kpi.learn_history(conn, """
                        SELECT month,
                               ROUND(SUM(denied_count)*100.0/NULLIF(SUM(claim_count),0), 2)
                        FROM _gpdm_summary_monthly
                        GROUP BY month ORDER BY month
                    """)
                elif name == 'avg_paid':
                    kpi.learn_history(conn, """
                        SELECT month,
                               ROUND(SUM(total_paid)*1.0/NULLIF(SUM(claim_count),0), 2)
                        FROM _gpdm_summary_monthly
                        GROUP BY month ORDER BY month
                    """)
                elif name == 'total_members':
                    kpi.learn_history(conn, """
                        SELECT month, SUM(unique_members)
                        FROM _gpdm_summary_monthly
                        GROUP BY month ORDER BY month
                    """)
        histories = {}
        for name, kpi in self.kpis.items():
            if kpi.history:
                histories[name] = [v for _, v in kpi.history]
        if len(histories) >= 2:
            return self.correlator.compute_correlations(histories)
        return {}

    def generate_report(self, conn: sqlite3.Connection = None) -> str:
        return self.explain_all(conn)

    def explain_all(self, conn: sqlite3.Connection = None) -> str:
        results = self.compute_all(conn)
        explanations = []
        for name, data in results.items():
            if isinstance(data, dict) and 'explanation' in data:
                explanations.append(data['explanation'])

        summary = results.get('_anomaly_summary', {})
        if summary.get('count', 0) > 0:
            explanations.append(
                f"\nUNUSUAL ACTIVITY DETECTED: {summary['count']} KPIs are outside "
                f"your data's normal range ({summary.get('critical', 0)} critical). "
                f"Affected: {', '.join(summary.get('kpis', []))}")

        corrs = results.get('_correlations', {})
        if corrs:
            explanations.append("\nEARLY WARNING SIGNALS:")
            for pair, info in corrs.items():
                explanations.append(f"  {info['relationship']}")

        return '\n'.join(explanations)
