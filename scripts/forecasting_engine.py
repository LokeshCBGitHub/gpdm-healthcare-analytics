import sqlite3
import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger('gpdm.forecasting')


class ForecastEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _query(self, sql: str) -> List[Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(sql)
            results = c.fetchall()
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Forecast query error: {e}")
            return []


    @staticmethod
    def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        n = len(x)
        if n < 2:
            return 0, y[0] if y else 0, 0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return 0, sum_y / n, 0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return slope, intercept, max(0, r_squared)

    @staticmethod
    def moving_average(data: List[float], window: int = 3) -> List[float]:
        if len(data) < window:
            return data
        result = []
        for i in range(len(data) - window + 1):
            result.append(sum(data[i:i + window]) / window)
        return result

    @staticmethod
    def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
        if not data:
            return []
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[-1])
        return result

    @staticmethod
    def forecast_next_n(data: List[float], n: int = 6, method: str = 'linear') -> Dict[str, Any]:
        if len(data) < 3:
            return {'forecast': [data[-1]] * n if data else [0] * n, 'confidence': 'low', 'trend': 'insufficient_data'}

        x = list(range(len(data)))
        y = data

        if method == 'linear':
            slope, intercept, r2 = ForecastEngine.linear_regression(x, y)

            forecasts = []
            for i in range(n):
                future_x = len(data) + i
                forecasts.append(slope * future_x + intercept)

            residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
            std_dev = math.sqrt(sum(r ** 2 for r in residuals) / max(len(residuals) - 2, 1))

            upper = [f + 1.96 * std_dev for f in forecasts]
            lower = [f - 1.96 * std_dev for f in forecasts]

            pct_change = (forecasts[-1] - data[-1]) / abs(data[-1]) * 100 if data[-1] != 0 else 0
            if pct_change > 5:
                trend = 'increasing'
            elif pct_change < -5:
                trend = 'decreasing'
            else:
                trend = 'stable'

            return {
                'forecast': [round(f, 2) for f in forecasts],
                'upper_ci': [round(f, 2) for f in upper],
                'lower_ci': [round(f, 2) for f in lower],
                'trend': trend,
                'pct_change': round(pct_change, 1),
                'r_squared': round(r2, 3),
                'confidence': 'high' if r2 > 0.7 else 'moderate' if r2 > 0.4 else 'low',
                'method': 'linear_regression'
            }

        elif method == 'exponential':
            smoothed = ForecastEngine.exponential_smoothing(data, alpha=0.3)
            last_val = smoothed[-1]
            if len(smoothed) >= 3:
                recent_trend = (smoothed[-1] - smoothed[-3]) / 2
            else:
                recent_trend = 0

            forecasts = [last_val + recent_trend * (i + 1) for i in range(n)]
            std_dev = math.sqrt(sum((d - s) ** 2 for d, s in zip(data, smoothed)) / len(data))

            return {
                'forecast': [round(f, 2) for f in forecasts],
                'upper_ci': [round(f + 1.96 * std_dev, 2) for f in forecasts],
                'lower_ci': [round(f - 1.96 * std_dev, 2) for f in forecasts],
                'trend': 'increasing' if recent_trend > 0 else 'decreasing' if recent_trend < 0 else 'stable',
                'confidence': 'moderate',
                'method': 'exponential_smoothing'
            }

        return {'forecast': [data[-1]] * n, 'confidence': 'low', 'method': 'fallback'}


    def forecast_revenue(self, periods: int = 6) -> Dict[str, Any]:
        monthly = self._query('''
            SELECT substr(SERVICE_DATE, 1, 7) as month,
                   SUM(CAST(BILLED_AMOUNT AS REAL)) as billed,
                   SUM(CAST(PAID_AMOUNT AS REAL)) as paid,
                   COUNT(*) as volume,
                   SUM(CAST(ALLOWED_AMOUNT AS REAL)) as allowed
            FROM claims
            WHERE SERVICE_DATE != ''
            GROUP BY month ORDER BY month
        ''')

        if not monthly:
            return {'error': 'No monthly data available'}

        months = [r[0] for r in monthly]
        billed = [r[1] for r in monthly]
        paid = [r[2] for r in monthly]
        volume = [r[3] for r in monthly]

        billed_forecast = self.forecast_next_n(billed, periods)
        paid_forecast = self.forecast_next_n(paid, periods)
        volume_forecast = self.forecast_next_n([float(v) for v in volume], periods)

        allowed = [r[4] if len(r) > 4 and r[4] else b for r, b in zip(monthly, billed)]
        yield_rates = [p / a * 100 if a else 0 for p, a in zip(paid, allowed)]
        yield_forecast = self.forecast_next_n(yield_rates, periods)

        return {
            'title': 'Revenue Forecast',
            'historical_months': months,
            'billed': {
                'historical': [round(b, 2) for b in billed],
                **billed_forecast,
                'interpretation': self._interpret_trend(billed_forecast, 'revenue', 'billed amounts')
            },
            'paid': {
                'historical': [round(p, 2) for p in paid],
                **paid_forecast,
                'interpretation': self._interpret_trend(paid_forecast, 'revenue', 'realized revenue')
            },
            'volume': {
                'historical': volume,
                **volume_forecast,
                'interpretation': self._interpret_trend(volume_forecast, 'volume', 'claims volume')
            },
            'yield_rate': {
                'historical': [round(y, 1) for y in yield_rates],
                **yield_forecast,
                'interpretation': self._interpret_trend(yield_forecast, 'efficiency', 'collection yield')
            },
            'recovery_assessment': self._assess_recovery(paid_forecast, billed_forecast)
        }

    def forecast_denials(self, periods: int = 6) -> Dict[str, Any]:
        monthly = self._query('''
            SELECT substr(SERVICE_DATE, 1, 7) as month,
                   COUNT(*) as total,
                   SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied,
                   SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN CAST(BILLED_AMOUNT AS REAL) ELSE 0 END) as denied_amount
            FROM claims WHERE SERVICE_DATE != ''
            GROUP BY month ORDER BY month
        ''')

        if not monthly:
            return {'error': 'No data'}

        months = [r[0] for r in monthly]
        denial_rates = [r[2] / r[1] * 100 if r[1] else 0 for r in monthly]
        denied_amounts = [r[3] for r in monthly]

        rate_forecast = self.forecast_next_n(denial_rates, periods)
        amount_forecast = self.forecast_next_n(denied_amounts, periods)

        return {
            'title': 'Denial Trend Forecast',
            'historical_months': months,
            'denial_rate': {
                'historical': [round(r, 1) for r in denial_rates],
                **rate_forecast,
                'interpretation': self._interpret_trend(rate_forecast, 'denial_rate', 'denial rate')
            },
            'denied_amount': {
                'historical': [round(a, 2) for a in denied_amounts],
                **amount_forecast,
                'interpretation': self._interpret_trend(amount_forecast, 'cost', 'denied revenue')
            },
            'by_reason': self._forecast_denial_by_reason(periods)
        }

    def forecast_pmpm(self, periods: int = 6) -> Dict[str, Any]:
        monthly = self._query('''
            SELECT substr(c.SERVICE_DATE, 1, 7) as month,
                   SUM(CAST(c.PAID_AMOUNT AS REAL)) as paid,
                   COUNT(DISTINCT c.MEMBER_ID) as members
            FROM claims c WHERE c.SERVICE_DATE != ''
            GROUP BY month ORDER BY month
        ''')

        if not monthly:
            return {'error': 'No data'}

        months = [r[0] for r in monthly]
        pmpm = [r[1] / r[2] if r[2] and r[2] > 0 else 0 for r in monthly]

        forecast = self.forecast_next_n(pmpm, periods)

        return {
            'title': 'PMPM Cost Forecast',
            'historical_months': months,
            'pmpm': {
                'historical': [round(p, 2) for p in pmpm],
                **forecast,
                'benchmark': '$350-500 commercial, $800-1200 MA',
                'interpretation': self._interpret_trend(forecast, 'cost', 'PMPM cost')
            }
        }

    def forecast_utilization(self, periods: int = 6) -> Dict[str, Any]:
        monthly = self._query('''
            SELECT substr(SERVICE_DATE, 1, 7) as month,
                   VISIT_TYPE,
                   COUNT(*) as encounters
            FROM encounters WHERE SERVICE_DATE != ''
            GROUP BY month, VISIT_TYPE ORDER BY month
        ''')

        if not monthly:
            return {'error': 'No data'}

        by_type = defaultdict(lambda: {'months': [], 'counts': []})
        for r in monthly:
            by_type[r[1]]['months'].append(r[0])
            by_type[r[1]]['counts'].append(r[2])

        forecasts = {}
        for vtype, data in by_type.items():
            forecast = self.forecast_next_n([float(c) for c in data['counts']], periods)
            forecasts[vtype] = {
                'historical': data['counts'],
                **forecast,
                'interpretation': self._interpret_trend(forecast, 'volume', f'{vtype} visits')
            }

        return {
            'title': 'Utilization Forecast by Visit Type',
            'by_visit_type': forecasts
        }

    def get_comprehensive_forecast(self, periods: int = 6) -> Dict[str, Any]:
        return {
            'title': 'GPDM Comprehensive Financial & Operational Forecast',
            'forecast_horizon': f'{periods} periods ahead',
            'generated_at': datetime.now().isoformat(),
            'revenue': self.forecast_revenue(periods),
            'denials': self.forecast_denials(periods),
            'pmpm': self.forecast_pmpm(periods),
            'utilization': self.forecast_utilization(periods),
            'executive_outlook': self._generate_executive_outlook(periods)
        }


    def _forecast_denial_by_reason(self, periods: int) -> Dict[str, Any]:
        reasons = self._query('''
            SELECT substr(SERVICE_DATE, 1, 7) as month, DENIAL_REASON, COUNT(*)
            FROM claims WHERE DENIAL_REASON != '' AND SERVICE_DATE != ''
            GROUP BY month, DENIAL_REASON ORDER BY month
        ''')

        by_reason = defaultdict(lambda: defaultdict(int))
        all_months = set()
        for r in reasons:
            by_reason[r[1]][r[0]] = r[2]
            all_months.add(r[0])

        sorted_months = sorted(all_months)
        result = {}
        for reason, month_data in by_reason.items():
            counts = [month_data.get(m, 0) for m in sorted_months]
            forecast = self.forecast_next_n([float(c) for c in counts], periods)
            result[reason] = {
                'historical': counts,
                **forecast
            }

        return result

    def _interpret_trend(self, forecast: Dict, metric_type: str, label: str) -> str:
        trend = forecast.get('trend', 'unknown')
        pct = forecast.get('pct_change', 0)
        confidence = forecast.get('confidence', 'low')

        if metric_type == 'revenue':
            if trend == 'increasing':
                return f"{label} is projected to increase by {abs(pct):.1f}% — positive sign for loss recovery. Confidence: {confidence}."
            elif trend == 'decreasing':
                return f"WARNING: {label} is projected to decline by {abs(pct):.1f}% — intensifies the loss slump. Immediate action required. Confidence: {confidence}."
            else:
                return f"{label} is projected to remain flat — insufficient organic growth to drive recovery. Active intervention needed. Confidence: {confidence}."

        elif metric_type == 'denial_rate':
            if trend == 'decreasing':
                return f"{label} is trending downward by {abs(pct):.1f}% — denial management efforts showing results. Confidence: {confidence}."
            elif trend == 'increasing':
                return f"WARNING: {label} is trending upward by {abs(pct):.1f}% — compounding revenue losses. Escalate denial management. Confidence: {confidence}."
            else:
                return f"{label} is stable — not improving. Current denial prevention measures are insufficient. Confidence: {confidence}."

        elif metric_type == 'cost':
            if trend == 'increasing':
                return f"{label} is projected to increase by {abs(pct):.1f}% — puts additional pressure on margins. Confidence: {confidence}."
            elif trend == 'decreasing':
                return f"{label} is projected to decrease by {abs(pct):.1f}% — favorable for margin recovery. Confidence: {confidence}."
            else:
                return f"{label} is projected to remain stable. Confidence: {confidence}."

        elif metric_type == 'volume':
            if trend == 'increasing':
                return f"{label} is projected to grow by {abs(pct):.1f}% — ensure capacity can absorb growth. Confidence: {confidence}."
            elif trend == 'decreasing':
                return f"{label} is projected to decline by {abs(pct):.1f}% — volume-driven revenue at risk. Confidence: {confidence}."
            else:
                return f"{label} is projected to remain stable. Confidence: {confidence}."

        return f"{label} trend: {trend} ({pct:+.1f}%). Confidence: {confidence}."

    def _assess_recovery(self, paid_forecast: Dict, billed_forecast: Dict) -> Dict[str, Any]:
        paid_trend = paid_forecast.get('trend', 'unknown')
        paid_pct = paid_forecast.get('pct_change', 0)
        billed_pct = billed_forecast.get('pct_change', 0)

        if paid_trend == 'increasing' and paid_pct > 3:
            status = 'recovering'
            message = f'Revenue is trending upward ({paid_pct:+.1f}%). If sustained, the loss slump recovery is on track.'
        elif paid_trend == 'stable':
            status = 'stalled'
            message = 'Revenue is flat — organic growth alone will not drive recovery. Active revenue cycle optimization and denial management needed.'
        else:
            status = 'declining'
            message = f'Revenue is declining ({paid_pct:+.1f}%). The loss slump is deepening. Urgent action required across denial management, payer contracts, and cost control.'

        return {
            'recovery_status': status,
            'message': message,
            'paid_trajectory': paid_pct,
            'billed_trajectory': billed_pct,
            'gap_direction': 'narrowing' if (billed_pct - paid_pct) < 0 else 'widening'
        }

    def _generate_executive_outlook(self, periods: int) -> Dict[str, Any]:
        revenue = self.forecast_revenue(periods)
        denials = self.forecast_denials(periods)
        pmpm = self.forecast_pmpm(periods)

        recovery = revenue.get('recovery_assessment', {})
        denial_trend = denials.get('denial_rate', {}).get('trend', 'unknown')
        pmpm_trend = pmpm.get('pmpm', {}).get('trend', 'unknown')

        score = 5
        if recovery.get('recovery_status') == 'recovering':
            score += 2
        elif recovery.get('recovery_status') == 'declining':
            score -= 2

        if denial_trend == 'decreasing':
            score += 1
        elif denial_trend == 'increasing':
            score -= 1

        if pmpm_trend == 'decreasing':
            score += 1
        elif pmpm_trend == 'increasing':
            score -= 0.5

        score = max(1, min(10, score))

        return {
            'outlook_score': round(score, 1),
            'outlook_label': 'Favorable' if score >= 7 else 'Cautiously Optimistic' if score >= 5 else 'Concerning' if score >= 3 else 'Critical',
            'revenue_status': recovery.get('recovery_status', 'unknown'),
            'denial_direction': denial_trend,
            'cost_direction': pmpm_trend,
            'key_risks': self._identify_key_risks(revenue, denials, pmpm),
            'key_opportunities': self._identify_opportunities(revenue, denials, pmpm)
        }

    def _identify_key_risks(self, revenue, denials, pmpm) -> List[str]:
        risks = []
        if revenue.get('paid', {}).get('trend') == 'decreasing':
            risks.append('Revenue declining — loss slump deepening without intervention')
        if denials.get('denial_rate', {}).get('trend') == 'increasing':
            risks.append('Denial rate increasing — accelerating revenue loss')
        if pmpm.get('pmpm', {}).get('trend') == 'increasing':
            risks.append('PMPM costs rising — margin compression')
        if revenue.get('yield_rate', {}).get('trend') == 'decreasing':
            risks.append('Collection yield declining — worsening payer reimbursement')
        if not risks:
            risks.append('No critical risks identified in current trajectory')
        return risks

    def _identify_opportunities(self, revenue, denials, pmpm) -> List[str]:
        opps = []
        if denials.get('denial_rate', {}).get('trend') == 'decreasing':
            opps.append('Denial rates improving — continue denial management acceleration')
        if revenue.get('volume', {}).get('trend') == 'increasing':
            opps.append('Growing claim volume — ensure capacity and capture rate')

        opps.extend([
            'HCC risk score optimization for MA members — each 0.1 point increase = $100-300 PMPM',
            'Top 3 denial reasons represent 60-70% of denied revenue — targeted prevention yields fastest ROI',
            'Shift ER visits to lower-cost settings (telehealth/urgent care) — 20-40% cost reduction per visit'
        ])
        return opps


_forecast_engine = None

def get_forecast_engine(db_path: str) -> ForecastEngine:
    global _forecast_engine
    if _forecast_engine is None or _forecast_engine.db_path != db_path:
        _forecast_engine = ForecastEngine(db_path)
    return _forecast_engine
