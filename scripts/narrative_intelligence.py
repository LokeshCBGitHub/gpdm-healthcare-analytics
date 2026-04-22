import logging
import sqlite3
import re
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None


logger = logging.getLogger('gpdm.narrative_intelligence')


class Intent(Enum):
    TREND = 'trend'
    COMPARISON = 'comparison'
    BREAKDOWN = 'breakdown'
    ANOMALY = 'anomaly'
    EXECUTIVE = 'executive'
    PERFORMANCE = 'performance'
    OPPORTUNITY = 'opportunity'
    RISK = 'risk'
    UNKNOWN = 'unknown'


class Persona(Enum):
    EXECUTIVE = 'executive'
    CLINICAL = 'clinical'
    OPERATIONAL = 'operational'
    FINANCIAL = 'financial'
    AUTO = 'auto'


class Severity(Enum):
    NORMAL = 'normal'
    WARNING = 'warning'
    CRITICAL = 'critical'
    POSITIVE = 'positive'


HEALTHCARE_BENCHMARKS = {
    'denial_rate': {'industry_avg': 0.08, 'good': 0.05, 'concerning': 0.15},
    'readmission_rate': {'industry_avg': 0.16, 'good': 0.12, 'concerning': 0.20},
    'pmpm_commercial': {'industry_avg': 450, 'good': 350, 'concerning': 600},
    'pmpm_medicare': {'industry_avg': 950, 'good': 800, 'concerning': 1200},
    'er_per_1000': {'industry_avg': 450, 'good': 350, 'concerning': 550},
    'yield_rate': {'industry_avg': 0.90, 'good': 0.95, 'concerning': 0.80},
    'alos': {'industry_avg': 4.5, 'good': 3.5, 'concerning': 6.0},
    'ar_days': {'industry_avg': 35, 'good': 25, 'concerning': 50},
}


@dataclass
class StatisticalProfile:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    outliers: List[Tuple[int, float]] = field(default_factory=list)
    outlier_indices: List[int] = field(default_factory=list)

    trend_slope: float = 0.0
    trend_direction: str = 'flat'
    trend_r_squared: float = 0.0
    trend_p_value_approx: float = 1.0

    gini_coefficient: float = 0.0
    top10_pct_share: float = 0.0

    segment_stats: Dict[str, Dict] = field(default_factory=dict)

    numeric_columns: List[str] = field(default_factory=list)


@dataclass
class NarrativeResult:
    summary: str
    narrative: str
    insights: List[str]
    recommendations: List[str]
    benchmark_comparison: Dict[str, Any]
    data_quality_notes: List[str]
    severity: str
    persona_used: str
    statistical_profile: Optional[StatisticalProfile] = None
    intent_detected: str = 'unknown'


class NarrativeIntelligence:

    def __init__(self, db_path: str = ''):
        self.db_path = db_path
        self.default_persona = Persona.AUTO
        logger.info(f'NarrativeIntelligence initialized with db_path={db_path}')

    def set_persona(self, persona: str) -> None:
        try:
            self.default_persona = Persona[persona.upper()]
            logger.info(f'Default persona set to {self.default_persona.value}')
        except KeyError:
            logger.warning(f'Invalid persona "{persona}", keeping {self.default_persona.value}')

    def generate(
        self,
        question: str,
        sql: str,
        rows: List[Dict],
        columns: List[str],
        intent: str = 'unknown',
        persona: str = 'auto',
        context: Optional[Dict[str, Any]] = None
    ) -> NarrativeResult:
        try:
            context = context or {}

            analysis = self._analyze_data(rows, columns, context)

            detected_intent = self._detect_intent(question, analysis, intent)

            target_persona = self._resolve_persona(persona)

            benchmark_cmp = self._compare_benchmarks(analysis, context)

            insights = self._extract_insights(analysis, benchmark_cmp, context, detected_intent)

            recommendations = self._generate_recommendations(
                insights, analysis, benchmark_cmp, detected_intent, context
            )

            severity = self._assess_severity(analysis, benchmark_cmp, insights)

            summary, narrative = self._compose_narrative(
                question, analysis, insights, recommendations,
                detected_intent, target_persona, severity, context
            )

            quality_notes = self._assess_data_quality(rows, columns, analysis)

            result = NarrativeResult(
                summary=summary,
                narrative=narrative,
                insights=insights,
                recommendations=recommendations,
                benchmark_comparison=benchmark_cmp,
                data_quality_notes=quality_notes,
                severity=severity,
                persona_used=target_persona.value,
                statistical_profile=analysis,
                intent_detected=detected_intent.value
            )

            logger.info(f'Generated narrative for intent={detected_intent.value}, '
                       f'persona={target_persona.value}, severity={severity}')
            return result

        except Exception as e:
            logger.exception(f'Error generating narrative: {e}')
            return self._degraded_result(question, str(e))

    def _analyze_data(self, rows: List[Dict], columns: List[str],
                     context: Dict) -> StatisticalProfile:
        profile = StatisticalProfile()

        if not rows or not columns:
            return profile

        profile.count = len(rows)

        numeric_data: Dict[str, List[float]] = defaultdict(list)
        col_index = {col: i for i, col in enumerate(columns)} if columns else {}
        for row in rows:
            for col in columns:
                if isinstance(row, dict):
                    val = row.get(col)
                elif isinstance(row, (list, tuple)):
                    idx = col_index.get(col, -1)
                    val = row[idx] if 0 <= idx < len(row) else None
                else:
                    val = None
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numeric_data[col].append(float(val))

        if not numeric_data:
            return profile

        profile.numeric_columns = list(numeric_data.keys())

        primary_col = context.get('primary_metric', profile.numeric_columns[0])
        if primary_col not in numeric_data:
            primary_col = profile.numeric_columns[0]

        values = numeric_data[primary_col]
        profile = self._compute_full_statistics(values, profile)

        if context.get('segment_by'):
            segment_col = context['segment_by']
            segments: Dict[str, List[float]] = defaultdict(list)
            for i, row in enumerate(rows):
                if i < len(values):
                    if isinstance(row, dict):
                        seg_key = row.get(segment_col, 'unknown')
                    elif isinstance(row, (list, tuple)):
                        seg_idx = col_index.get(segment_col, -1)
                        seg_key = row[seg_idx] if 0 <= seg_idx < len(row) else 'unknown'
                    else:
                        seg_key = 'unknown'
                    segments[seg_key].append(values[i])

            for seg_key, seg_values in segments.items():
                if len(seg_values) >= 2:
                    profile.segment_stats[str(seg_key)] = {
                        'count': len(seg_values),
                        'mean': sum(seg_values) / len(seg_values),
                        'min': min(seg_values),
                        'max': max(seg_values),
                    }

        logger.debug(f'Statistical analysis complete: {profile.count} records, '
                    f'{len(numeric_data)} numeric columns')
        return profile

    def _compute_full_statistics(self, values: List[float],
                                profile: StatisticalProfile) -> StatisticalProfile:
        if not values:
            return profile

        try:
            if np:
                arr = np.array(values)
                profile.count = len(arr)
                profile.mean = float(np.mean(arr))
                profile.median = float(np.median(arr))
                profile.std = float(np.std(arr))
                profile.min = float(np.min(arr))
                profile.max = float(np.max(arr))
                profile.p10 = float(np.percentile(arr, 10))
                profile.p25 = float(np.percentile(arr, 25))
                profile.p50 = float(np.percentile(arr, 50))
                profile.p75 = float(np.percentile(arr, 75))
                profile.p90 = float(np.percentile(arr, 90))
                profile.skew = float(np.mean(((arr - profile.mean) / max(profile.std, 1e-10)) ** 3))
                profile.kurtosis = float(np.mean(((arr - profile.mean) / max(profile.std, 1e-10)) ** 4) - 3)
            else:
                profile.count = len(values)
                sorted_vals = sorted(values)
                profile.mean = sum(values) / len(values)
                profile.median = sorted_vals[len(sorted_vals) // 2]
                profile.min = min(values)
                profile.max = max(values)
                profile.std = math.sqrt(sum((x - profile.mean) ** 2 for x in values) / max(len(values), 1))
                profile.p25 = sorted_vals[int(len(sorted_vals) * 0.25)]
                profile.p75 = sorted_vals[int(len(sorted_vals) * 0.75)]
                profile.p50 = profile.median
                profile.p10 = sorted_vals[int(len(sorted_vals) * 0.10)]
                profile.p90 = sorted_vals[int(len(sorted_vals) * 0.90)]

            iqr = profile.p75 - profile.p25
            if iqr > 0:
                lower_bound = profile.p25 - 1.5 * iqr
                upper_bound = profile.p75 + 1.5 * iqr
                for i, v in enumerate(values):
                    if v < lower_bound or v > upper_bound:
                        profile.outliers.append((i, v))
                        profile.outlier_indices.append(i)

            if len(values) > 2:
                slope, r_squared, p_value = self._linear_regression(values)
                profile.trend_slope = slope
                profile.trend_r_squared = r_squared
                profile.trend_p_value_approx = p_value

                if slope > 0.001 * max(abs(profile.mean), 1):
                    profile.trend_direction = 'increasing'
                elif slope < -0.001 * max(abs(profile.mean), 1):
                    profile.trend_direction = 'decreasing'
                else:
                    profile.trend_direction = 'flat'

            profile.gini_coefficient = self._compute_gini(values)
            profile.top10_pct_share = self._compute_top_n_share(values, 0.1)

        except Exception as e:
            logger.warning(f'Statistical computation error: {e}')

        return profile

    def _linear_regression(self, values: List[float]) -> Tuple[float, float, float]:
        try:
            n = len(values)
            if n < 2:
                return 0.0, 0.0, 1.0

            x = list(range(n))
            mean_x = sum(x) / n
            mean_y = sum(values) / n

            ss_xy = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
            ss_xx = sum((x[i] - mean_x) ** 2 for i in range(n))
            ss_yy = sum((values[i] - mean_y) ** 2 for i in range(n))

            if ss_xx == 0:
                return 0.0, 0.0, 1.0

            slope = ss_xy / ss_xx
            intercept = mean_y - slope * mean_x

            predictions = [intercept + slope * xi for xi in x]
            ss_res = sum((values[i] - predictions[i]) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / max(ss_yy, 1e-10)) if ss_yy > 0 else 0.0

            if n > 2 and ss_xx > 0:
                s_err = math.sqrt(ss_res / max(n - 2, 1))
                t_stat = abs(slope / max(s_err / math.sqrt(ss_xx), 1e-10))
                p_value = 0.05 if t_stat > 2 else 0.5
            else:
                p_value = 1.0

            return slope, r_squared, p_value

        except Exception as e:
            logger.warning(f'Linear regression error: {e}')
            return 0.0, 0.0, 1.0

    def _compute_gini(self, values: List[float]) -> float:
        try:
            sorted_vals = sorted([v for v in values if v > 0])
            if not sorted_vals:
                return 0.0

            n = len(sorted_vals)
            cumsum = sum(sorted_vals)
            gini = (2 * sum((i + 1) * sorted_vals[i] for i in range(n))) / (n * cumsum) - (n + 1) / n
            return max(0.0, min(1.0, gini))
        except:
            return 0.0

    def _compute_top_n_share(self, values: List[float], top_pct: float) -> float:
        try:
            sorted_vals = sorted([v for v in values if v > 0], reverse=True)
            if not sorted_vals:
                return 0.0

            n = max(1, int(len(sorted_vals) * top_pct))
            top_sum = sum(sorted_vals[:n])
            total_sum = sum(sorted_vals)
            return (top_sum / total_sum * 100) if total_sum > 0 else 0.0
        except:
            return 0.0

    def _detect_intent(self, question: str, analysis: StatisticalProfile,
                      explicit_intent: str) -> Intent:
        if explicit_intent and explicit_intent != 'unknown':
            try:
                return Intent[explicit_intent.upper()]
            except KeyError:
                pass

        q_lower = question.lower()

        if any(w in q_lower for w in ['trend', 'over time', 'change', 'growth', 'decline', 'trajectory']):
            return Intent.TREND
        elif any(w in q_lower for w in ['compare', 'vs', 'difference', 'between', 'versus', 'relative']):
            return Intent.COMPARISON
        elif any(w in q_lower for w in ['breakdown', 'distribution', 'segment', 'category', 'by']):
            return Intent.BREAKDOWN
        elif any(w in q_lower for w in ['anomaly', 'outlier', 'unusual', 'unexpected', 'spike', 'surge', 'jump']):
            return Intent.ANOMALY
        elif any(w in q_lower for w in ['opportunity', 'improve', 'optimize', 'growth', 'potential']):
            return Intent.OPPORTUNITY
        elif any(w in q_lower for w in ['risk', 'issue', 'problem', 'concern', 'alert', 'threat']):
            return Intent.RISK
        elif len(analysis.outliers) > analysis.count * 0.05:
            return Intent.ANOMALY
        elif abs(analysis.trend_slope) > 0.05 * max(abs(analysis.mean), 1):
            return Intent.TREND

        return Intent.UNKNOWN

    def _resolve_persona(self, persona: str) -> Persona:
        if persona == 'auto' or persona == '':
            return self.default_persona
        try:
            return Persona[persona.upper()]
        except KeyError:
            return self.default_persona

    def _compare_benchmarks(self, analysis: StatisticalProfile,
                           context: Dict) -> Dict[str, Any]:
        comparison = {}

        metric_type = context.get('metric_type', '')
        metric_label = context.get('metric_label', metric_type)

        if metric_type in HEALTHCARE_BENCHMARKS:
            bench = HEALTHCARE_BENCHMARKS[metric_type]
            value = analysis.mean

            status = 'acceptable'
            if metric_type in ['yield_rate']:
                if value >= bench['good']:
                    status = 'good'
                elif value <= bench['concerning']:
                    status = 'concerning'
            else:
                if value <= bench['good']:
                    status = 'good'
                elif value >= bench['concerning']:
                    status = 'concerning'

            variance_pct = ((value - bench['industry_avg']) / max(bench['industry_avg'], 1e-10) * 100)

            deviation_sigma = (value - bench['industry_avg']) / max(bench['industry_avg'], 1e-10)

            comparison = {
                'metric': metric_type,
                'metric_label': metric_label,
                'value': value,
                'industry_average': bench['industry_avg'],
                'good_threshold': bench['good'],
                'concerning_threshold': bench['concerning'],
                'variance_pct': variance_pct,
                'deviation_sigma': deviation_sigma,
                'status': status,
            }

        return comparison

    def _extract_insights(self, analysis: StatisticalProfile,
                         benchmark_cmp: Dict, context: Dict,
                         intent: Intent) -> List[str]:
        insights = []

        try:
            if analysis.count > 0:
                if analysis.mean > 0:
                    insights.append(
                        f'Central tendency: mean of {self._format_number(analysis.mean)} '
                        f'(median {self._format_number(analysis.median)}) across {analysis.count} records.'
                    )

            if abs(analysis.skew) > 0.5:
                if analysis.skew > 0.5:
                    insights.append(
                        f'Distribution shows right skew (skew={analysis.skew:.2f})—concentration '
                        f'of low values with high outliers.'
                    )
                else:
                    insights.append(
                        f'Distribution shows left skew (skew={analysis.skew:.2f})—concentration '
                        f'of high values with low outliers.'
                    )
            else:
                insights.append('Distribution is relatively symmetric.')

            if analysis.mean > 0:
                cv = (analysis.std / analysis.mean) * 100
                if cv > 50:
                    insights.append(
                        f'High variability (CV={cv:.1f}%)—substantial fluctuation with '
                        f'range from {self._format_number(analysis.min)} to {self._format_number(analysis.max)}.'
                    )
                elif cv < 15:
                    insights.append(
                        f'Low variability (CV={cv:.1f}%)—values are tightly clustered '
                        f'within a narrow range.'
                    )
                else:
                    insights.append(
                        f'Moderate variability (CV={cv:.1f}%)—values range from '
                        f'{self._format_number(analysis.min)} to {self._format_number(analysis.max)}.'
                    )

            if analysis.trend_direction == 'increasing':
                rate_str = self._format_number(analysis.trend_slope)
                insights.append(
                    f'Trend: increasing at {rate_str} per period (R²={analysis.trend_r_squared:.3f}), '
                    f'suggesting accelerating growth or deterioration.'
                )
            elif analysis.trend_direction == 'decreasing':
                rate_str = self._format_number(abs(analysis.trend_slope))
                insights.append(
                    f'Trend: decreasing at {rate_str} per period (R²={analysis.trend_r_squared:.3f}), '
                    f'showing sustained decline or improvement.'
                )

            if analysis.outliers:
                outlier_pct = (len(analysis.outliers) / max(analysis.count, 1)) * 100
                insights.append(
                    f'Outliers: {len(analysis.outliers)} extreme values ({outlier_pct:.1f}% of data) '
                    f'detected beyond IQR bounds, ranging from {self._format_number(min(o[1] for o in analysis.outliers))} '
                    f'to {self._format_number(max(o[1] for o in analysis.outliers))}.'
                )

            if analysis.gini_coefficient > 0.3:
                insights.append(
                    f'Concentration: Gini coefficient of {analysis.gini_coefficient:.3f} indicates unequal '
                    f'distribution—top 10% accounts for {analysis.top10_pct_share:.1f}% of total (potential Pareto effect).'
                )

            if benchmark_cmp:
                metric = benchmark_cmp.get('metric_label', benchmark_cmp.get('metric', 'metric'))
                value = benchmark_cmp.get('value', 0)
                variance = benchmark_cmp.get('variance_pct', 0)
                status = benchmark_cmp.get('status', '')

                if status == 'concerning':
                    if variance > 0:
                        insights.append(
                            f'Benchmark gap: {metric} of {self._format_number(value)} is '
                            f'{abs(variance):.1f}% above industry average of '
                            f'{self._format_number(benchmark_cmp["industry_average"])}, '
                            f'placing performance in concerning range.'
                        )
                    else:
                        insights.append(
                            f'Benchmark gap: {metric} of {self._format_number(value)} is '
                            f'{abs(variance):.1f}% below industry average of '
                            f'{self._format_number(benchmark_cmp["industry_average"])}, '
                            f'indicating concerning underperformance.'
                        )
                elif status == 'good':
                    insights.append(
                        f'Benchmark excellence: {metric} of {self._format_number(value)} '
                        f'exceeds industry average of {self._format_number(benchmark_cmp["industry_average"])} '
                        f'by {variance:.1f}%, demonstrating strong performance.'
                    )

            if analysis.segment_stats and len(analysis.segment_stats) > 1:
                segments = list(analysis.segment_stats.values())
                means = [s['mean'] for s in segments]
                max_diff = max(means) - min(means)
                insights.append(
                    f'Segmentation: Performance varies significantly across groups—'
                    f'mean range spans {self._format_number(min(means))} to {self._format_number(max(means))} '
                    f'({self._format_number(max_diff)} differential).'
                )

        except Exception as e:
            logger.warning(f'Insight extraction error: {e}')

        return insights[:8]

    def _generate_recommendations(self, insights: List[str],
                                 analysis: StatisticalProfile,
                                 benchmark_cmp: Dict,
                                 intent: Intent,
                                 context: Dict) -> List[str]:
        recommendations = []

        try:
            severity_hints = []

            if analysis.mean > 0 and (analysis.std / analysis.mean) > 0.3:
                severity_hints.append('HIGH')
                recommendations.append(
                    '[HIGH] Investigate root causes of high variability. '
                    'Identify and standardize processes with outlier performance to reduce fluctuation.'
                )

            if len(analysis.outliers) > analysis.count * 0.02:
                severity_hints.append('MEDIUM')
                recommendations.append(
                    '[MEDIUM] Review identified outlier records. Determine if they represent '
                    'data errors, special circumstances, or genuine anomalies requiring intervention.'
                )

            if analysis.trend_direction == 'decreasing' and intent in [Intent.RISK, Intent.PERFORMANCE]:
                severity_hints.append('HIGH')
                recommendations.append(
                    '[HIGH] Address declining trend immediately. Implement corrective measures '
                    'and establish weekly monitoring to track reversal progress.'
                )
            elif analysis.trend_direction == 'increasing' and intent in [Intent.RISK, Intent.ANOMALY]:
                severity_hints.append('HIGH')
                recommendations.append(
                    '[HIGH] Escalate accelerating trend. Root cause analysis required within '
                    f'{int(analysis.count/4)} business days.'
                )

            if benchmark_cmp.get('status') == 'concerning':
                variance = abs(benchmark_cmp.get('variance_pct', 0))
                severity_hints.append('HIGH' if variance > 50 else 'MEDIUM')
                metric = benchmark_cmp.get('metric_label', 'metric')
                recommendations.append(
                    f"[{'HIGH' if variance > 50 else 'MEDIUM'}] Benchmark gap closure: "
                    f'Current {metric} lags industry by {variance:.1f}%. '
                    f'Benchmark top performers in this space and develop targeted improvement plan.'
                )

            if analysis.gini_coefficient > 0.4:
                recommendations.append(
                    '[MEDIUM] Address concentration: Top performers are outsized. '
                    'Investigate what drives their success and replicate best practices across lower performers.'
                )

            if analysis.count < 5:
                recommendations.append(
                    '[LOW] Expand sample size. Current dataset ({}) is too small for robust '
                    'conclusions—aim for 30+ records.'.format(analysis.count)
                )
            elif analysis.count < 30:
                recommendations.append(
                    '[LOW] Increase dataset. With {} records, consider broader analysis for confirmation.'.format(analysis.count)
                )

            if analysis.segment_stats and len(analysis.segment_stats) > 1:
                worst_segment = min(analysis.segment_stats.items(),
                                  key=lambda x: x[1].get('mean', 0))
                recommendations.append(
                    f"[MEDIUM] Target segment '{worst_segment[0]}' for improvement—"
                    f"performing {self._format_number(worst_segment[1]['mean'])} vs best practice."
                )

        except Exception as e:
            logger.warning(f'Recommendation generation error: {e}')

        return recommendations[:4]

    def _assess_severity(self, analysis: StatisticalProfile,
                        benchmark_cmp: Dict, insights: List[str]) -> str:
        try:
            score = 0

            if analysis.outliers:
                outlier_pct = len(analysis.outliers) / max(analysis.count, 1)
                if outlier_pct > 0.1:
                    score += 2
                elif outlier_pct > 0.05:
                    score += 1

            if analysis.trend_direction == 'decreasing' and analysis.trend_r_squared > 0.5:
                score += 2

            if benchmark_cmp:
                variance = abs(benchmark_cmp.get('variance_pct', 0))
                if variance > 50:
                    score += 2
                elif variance > 20:
                    score += 1

            if analysis.mean > 0 and (analysis.std / analysis.mean) > 0.5:
                score += 1

            if score >= 4:
                return Severity.CRITICAL.value
            elif score >= 2:
                return Severity.WARNING.value
            elif benchmark_cmp.get('status') == 'good':
                return Severity.POSITIVE.value
            else:
                return Severity.NORMAL.value

        except Exception as e:
            logger.warning(f'Severity assessment error: {e}')
            return Severity.NORMAL.value

    def _compose_narrative(self, question: str, analysis: StatisticalProfile,
                          insights: List[str], recommendations: List[str],
                          intent: Intent, persona: Persona,
                          severity: str, context: Dict) -> Tuple[str, str]:
        try:
            summary = self._build_summary(severity, analysis, insights, persona)

            opening = self._build_opening(persona, analysis, severity)
            findings = self._build_findings(insights, persona, severity, analysis)
            implications = self._build_implications(severity, persona, intent, analysis)
            actions = self._build_actions(recommendations, persona)

            narrative = ' '.join(filter(None, [opening, findings, implications, actions]))

            return summary, narrative

        except Exception as e:
            logger.warning(f'Narrative composition error: {e}')
            return 'Analysis complete.', 'Unable to generate detailed narrative.'

    def _build_summary(self, severity: str, analysis: StatisticalProfile,
                      insights: List[str], persona: Persona) -> str:
        summary_parts = []

        if severity == 'critical':
            summary_parts.append('CRITICAL:')
        elif severity == 'warning':
            summary_parts.append('Alert:')
        elif severity == 'positive':
            summary_parts.append('Strong performance:')

        if insights:
            key_insight = insights[0]
            if severity == 'critical' or severity == 'warning':
                summary_parts.append(key_insight.split('.')[0])
            else:
                summary_parts.append(key_insight[:100])

        summary = ' '.join(summary_parts)
        return summary[:200]

    def _build_opening(self, persona: Persona, analysis: StatisticalProfile,
                      severity: str) -> str:
        opening = ''

        if persona == Persona.EXECUTIVE:
            opening = (
                f'Analysis of {analysis.count} records reveals significant '
                f'strategic implications requiring leadership attention. '
            )
            if severity in ['critical', 'warning']:
                opening += 'Immediate action is needed to mitigate risk and preserve margin. '

        elif persona == Persona.CLINICAL:
            opening = (
                f'Clinical analysis across {analysis.count} cases identifies important '
                f'patterns affecting patient care quality and safety. '
            )
            if severity in ['critical', 'warning']:
                opening += 'Intervention protocols should be prioritized. '

        elif persona == Persona.OPERATIONAL:
            opening = (
                f'Operational assessment of {analysis.count} records demonstrates '
                f'efficiency and throughput patterns requiring management attention. '
            )
            if severity in ['critical', 'warning']:
                opening += 'Process optimization is urgent. '

        elif persona == Persona.FINANCIAL:
            opening = (
                f'Financial analysis of {analysis.count} transactions reveals revenue '
                f'and cost dynamics with material P&L implications. '
            )
            if severity in ['critical', 'warning']:
                opening += 'Revenue leakage or cost containment opportunities exist. '

        else:
            opening = f'Analysis of {analysis.count} data points identified significant patterns. '

        return opening

    def _build_findings(self, insights: List[str], persona: Persona,
                       severity: str, analysis: StatisticalProfile) -> str:
        if not insights:
            return ''

        relevant = insights[:3]

        if persona == Persona.EXECUTIVE:
            findings_text = (
                f'Key findings include: {relevant[0] if relevant else "analysis complete"}. '
            )
            if len(relevant) > 1:
                findings_text += relevant[1] + '. '

        elif persona == Persona.CLINICAL:
            findings_text = (
                f'Clinical findings show: {relevant[0] if relevant else "analysis complete"}. '
            )
            if len(relevant) > 1:
                findings_text += 'Additionally, ' + relevant[1].lower() + '. '

        elif persona == Persona.OPERATIONAL:
            findings_text = (
                f'Operational analysis reveals: {relevant[0] if relevant else "analysis complete"}. '
            )
            if len(relevant) > 1:
                findings_text += 'Further, ' + relevant[1].lower() + '. '

        elif persona == Persona.FINANCIAL:
            findings_text = (
                f'Financial metrics demonstrate: {relevant[0] if relevant else "analysis complete"}. '
            )
            if len(relevant) > 1:
                findings_text += relevant[1] + '. '

        else:
            findings_text = ' '.join(relevant) + '. '

        return findings_text

    def _build_implications(self, severity: str, persona: Persona,
                           intent: Intent, analysis: StatisticalProfile) -> str:
        if severity == 'critical':
            if persona == Persona.EXECUTIVE:
                return (
                    'This represents material business risk requiring board-level visibility and '
                    'executive action. Quantify financial impact and develop mitigation strategy immediately. '
                )
            elif persona == Persona.CLINICAL:
                return (
                    'This threatens patient safety or care quality. Escalate to medical leadership '
                    'and implement immediate care pathway adjustments. '
                )
            elif persona == Persona.FINANCIAL:
                return (
                    'This creates significant revenue leakage or margin pressure. '
                    'Financial impact exceeds threshold for executive escalation. '
                )
            else:
                return 'Urgent remediation required. Establish task force to address root cause. '

        elif severity == 'warning':
            if persona == Persona.EXECUTIVE:
                return (
                    'This warrants management attention and targeted improvement initiatives. '
                    'Develop 90-day action plan with accountability. '
                )
            elif persona == Persona.CLINICAL:
                return (
                    'This requires clinical protocol review and evidence-based intervention design. '
                    'Engage care management and quality teams. '
                )
            else:
                return 'Address via departmental action plan. Monitor weekly for improvement. '

        elif severity == 'positive':
            return (
                'Strong performance indicates effective practices and operational excellence. '
                'Document processes and use as model for improvement across organization. '
            )

        else:
            return 'Continue monitoring trends. No immediate action required at this time. '

    def _build_actions(self, recommendations: List[str], persona: Persona) -> str:
        if not recommendations:
            return ''

        actions_text = 'Recommended actions: '

        if persona == Persona.EXECUTIVE:
            actions_text += 'Allocate resources and establish executive sponsorship. '
        elif persona == Persona.CLINICAL:
            actions_text += 'Coordinate with clinical governance. '
        elif persona == Persona.FINANCIAL:
            actions_text += 'Implement financial controls and monitoring. '

        actions_text += ' '.join(rec[:80] for rec in recommendations[:2])

        return actions_text

    def _assess_data_quality(self, rows: List[Dict], columns: List[str],
                            analysis: StatisticalProfile) -> List[str]:
        notes = []

        try:
            if rows:
                col_index = {col: i for i, col in enumerate(columns)} if columns else {}
                null_counts = defaultdict(int)
                for row in rows:
                    for col in columns:
                        if isinstance(row, dict):
                            val = row.get(col)
                        elif isinstance(row, (list, tuple)):
                            idx = col_index.get(col, -1)
                            val = row[idx] if 0 <= idx < len(row) else None
                        else:
                            val = None
                        if val is None:
                            null_counts[col] += 1

                for col, count in null_counts.items():
                    pct = (count / len(rows)) * 100
                    if pct > 20:
                        notes.append(f'WARNING: {col} has {pct:.1f}% missing values—may bias analysis.')
                    elif pct > 5:
                        notes.append(f'Note: {col} has {pct:.1f}% missing values.')

            if analysis.count < 5:
                notes.append('CAUTION: Very small sample (n={})—conclusions may lack statistical power.'.format(analysis.count))
            elif analysis.count < 30:
                notes.append('Note: Modest sample size (n={})—wider confidence intervals apply.'.format(analysis.count))

            if analysis.min == analysis.max:
                notes.append('WARNING: All values identical—no variation detected. Check data source.')

            if analysis.mean != 0 and analysis.max / max(abs(analysis.mean), 1e-10) > 1000:
                notes.append('CAUTION: Extreme range detected (max >> min)—verify data integrity.')

            if analysis.count > 10:
                sorted_vals = sorted([v for v in [analysis.mean] if v])
                if analysis.std == 0:
                    notes.append('No variance detected—all data points are identical.')

        except Exception as e:
            logger.warning(f'Data quality assessment error: {e}')

        return notes

    def _format_number(self, value: Union[int, float], is_pct: bool = False) -> str:
        if isinstance(value, int):
            return f'{value:,}'

        if abs(value) >= 1e6:
            return f'${value/1e6:.1f}M' if not is_pct else f'{value:.1f}M'
        elif abs(value) >= 1e3:
            return f'${value/1e3:.1f}K' if not is_pct else f'{value:.1f}K'
        elif is_pct:
            return f'{value:.1f}%'
        elif abs(value) < 1:
            return f'{value:.3f}'
        else:
            return f'{value:.2f}'

    def _degraded_result(self, question: str, error: str) -> NarrativeResult:
        return NarrativeResult(
            summary='Analysis unable to complete',
            narrative=(
                f'The narrative generation process encountered an error: {error}. '
                'Please verify the query syntax, data availability, and try again.'
            ),
            insights=['Unable to extract insights due to processing error'],
            recommendations=['Verify query syntax and data structure'],
            benchmark_comparison={},
            data_quality_notes=[f'Processing error: {error}'],
            severity=Severity.NORMAL.value,
            persona_used=self.default_persona.value,
            statistical_profile=None,
            intent_detected=Intent.UNKNOWN.value
        )
