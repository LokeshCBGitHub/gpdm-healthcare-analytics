import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


@dataclass
class InsightResult:
    title: str
    description: str
    value: Any
    confidence: float
    data_quality: str
    visualization_hint: str


class InsightEngine:

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    def generate_insights(
        self,
        question: str,
        sql: str,
        rows: List[Dict[str, Any]],
        columns: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not rows or not columns:
            return {
                'statistical_summary': {},
                'patterns': [],
                'anomalies': [],
                'correlations': [],
                'distributions': {},
                'probabilistic': {},
                'discrete_analysis': {},
                'time_series': {},
                'narrative': 'No data available for analysis.',
                'recommendations': [],
                'executive_summary': 'No data to analyze.',
                'quality_score': 0.0,
            }

        insights = {}

        data = self._rows_to_columns(rows, columns)

        insights['statistical_summary'] = self.statistical_summary(data, columns)

        insights['patterns'] = self.detect_patterns(data, columns)

        insights['anomalies'] = self.detect_anomalies(data, columns)

        insights['correlations'] = self.correlation_analysis(data, columns)

        insights['distributions'] = self.distribution_analysis(data, columns)

        insights['probabilistic'] = self.bayesian_inference(
            data, columns,
            prior=metadata.get('prior') if metadata else None
        )

        mc_insights = self.monte_carlo_simulation(data, columns, n_sims=5000)
        insights['monte_carlo'] = mc_insights

        insights['markov'] = self.markov_chain_analysis(data, columns)

        insights['discrete_analysis'] = {
            'pareto': self.pareto_analysis(data, columns),
            'entropy': self.entropy_analysis(data, columns),
        }

        insights['time_series'] = self.detect_time_series(data, columns)

        insights['narrative'] = self.generate_narrative(
            insights, question, len(rows), columns
        )
        insights['recommendations'] = self.generate_recommendations(insights)
        insights['executive_summary'] = self.generate_executive_summary(insights)

        insights['quality_score'] = self._calculate_quality_score(
            insights, len(rows), len(columns)
        )

        return insights

    @staticmethod
    def _rows_to_columns(rows: List[Dict], columns: List[str]) -> Dict[str, List[Any]]:
        data = {col: [] for col in columns}
        for row in rows:
            for col in columns:
                data[col].append(row.get(col))
        return data

    def statistical_summary(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        summary = {}

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]

            if not values:
                summary[col] = {
                    'type': 'categorical',
                    'unique_count': len(set(data.get(col, []))),
                    'most_common': self._most_common(data.get(col, []), 1),
                }
                continue

            n = len(values)
            sorted_vals = sorted(values)

            mean = sum(values) / n if n > 0 else 0
            median = self._median(sorted_vals)
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            variance = sum((x - mean) ** 2 for x in values) / n if n > 0 else 0
            std = math.sqrt(variance)

            q1 = self._quantile(sorted_vals, 0.25)
            q3 = self._quantile(sorted_vals, 0.75)
            iqr = q3 - q1

            skewness = self._skewness(values, mean, std)
            kurtosis = self._kurtosis(values, mean, std)

            cv = (std / mean * 100) if mean != 0 else 0

            summary[col] = {
                'type': 'numeric',
                'count': n,
                'mean': round(mean, 4),
                'median': round(median, 4),
                'std': round(std, 4),
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'q1': round(q1, 4),
                'q3': round(q3, 4),
                'iqr': round(iqr, 4),
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'cv': round(cv, 4),
            }

        return summary

    def detect_patterns(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> List[Dict[str, Any]]:
        patterns = []

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
            if len(values) < 4:
                continue

            trend = self._detect_trend(values)
            if trend:
                patterns.append({
                    'column': col,
                    'pattern_type': 'trend',
                    'direction': trend['direction'],
                    'strength': round(trend['strength'], 4),
                    'description': f"Strong {trend['direction']} trend (r² = {trend['strength']:.4f})"
                })

            autocorr = self._autocorrelation(values, lag=min(12, len(values) // 4))
            if autocorr and autocorr > 0.5:
                patterns.append({
                    'column': col,
                    'pattern_type': 'seasonality',
                    'lag': min(12, len(values) // 4),
                    'autocorr': round(autocorr, 4),
                    'description': f"Seasonal pattern detected (lag={min(12, len(values) // 4)}, autocorr={autocorr:.4f})"
                })

            for lag in [2, 3, 4, 5, 6, 12]:
                if lag >= len(values):
                    break
                ac = self._autocorrelation(values, lag=lag)
                if ac and ac > 0.6:
                    patterns.append({
                        'column': col,
                        'pattern_type': 'cycle',
                        'period': lag,
                        'strength': round(ac, 4),
                        'description': f"Cyclical pattern with period ~{lag}"
                    })
                    break

        return patterns

    def detect_anomalies(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> List[Dict[str, Any]]:
        anomalies = []

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
            if len(values) < 3:
                continue

            z_anomalies = self._zscore_anomalies(values, threshold=3.0)
            if z_anomalies:
                anomalies.extend([
                    {
                        'column': col,
                        'method': 'z_score',
                        'index': idx,
                        'value': val,
                        'z_score': round(z, 4),
                        'severity': 'high' if abs(z) > 3 else 'medium'
                    }
                    for idx, val, z in z_anomalies[:3]
                ])

            iqr_anomalies = self._iqr_anomalies(values)
            if iqr_anomalies:
                anomalies.extend([
                    {
                        'column': col,
                        'method': 'iqr',
                        'index': idx,
                        'value': val,
                        'severity': 'high' if val > iqr_anomalies[1] else 'medium'
                    }
                    for idx, val in iqr_anomalies[0][:3]
                ])

            grubbs_outlier = self._grubbs_test(values)
            if grubbs_outlier:
                anomalies.append({
                    'column': col,
                    'method': 'grubbs',
                    'value': grubbs_outlier,
                    'severity': 'high',
                    'description': f"Grubbs test identified outlier: {grubbs_outlier}"
                })

        return anomalies

    def correlation_analysis(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> List[Dict[str, Any]]:
        correlations = []

        numeric_cols = [
            col for col in columns
            if any(isinstance(v, (int, float)) for v in data.get(col, []))
        ]

        if len(numeric_cols) < 2:
            return correlations

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                vals1 = [v for v in data.get(col1, []) if isinstance(v, (int, float))]
                vals2 = [v for v in data.get(col2, []) if isinstance(v, (int, float))]

                if len(vals1) != len(vals2) or len(vals1) < 3:
                    continue

                pearson = self._pearson_correlation(vals1, vals2)
                spearman = self._spearman_correlation(vals1, vals2)

                if abs(pearson) > 0.5 or abs(spearman) > 0.5:
                    correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'pearson': round(pearson, 4),
                        'spearman': round(spearman, 4),
                        'strength': self._correlation_strength(abs(pearson)),
                        'description': f"{col1} and {col2}: r={pearson:.4f}"
                    })

        return sorted(correlations, key=lambda x: abs(x['pearson']), reverse=True)[:10]

    def distribution_analysis(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        distributions = {}

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
            if len(values) < 3:
                continue

            shapiro_p = self._shapiro_wilk_approx(values)
            is_normal = shapiro_p > 0.05

            skewness = self._skewness(values, sum(values)/len(values),
                                     math.sqrt(sum((x - sum(values)/len(values))**2 for x in values) / len(values)))
            kurtosis = self._kurtosis(values, sum(values)/len(values),
                                     math.sqrt(sum((x - sum(values)/len(values))**2 for x in values) / len(values)))

            distributions[col] = {
                'is_normal': is_normal,
                'normality_p_value': round(shapiro_p, 4),
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'distribution_type': self._infer_distribution(values),
                'description': 'Normal distribution' if is_normal else 'Non-normal distribution'
            }

        return distributions

    def bayesian_inference(
        self,
        data: Dict[str, List[Any]],
        columns: List[str],
        prior: Optional[Dict] = None
    ) -> Dict[str, Any]:
        result = {}

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
            if len(values) < 2:
                continue

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)

            prior_mean = prior.get('mean', 0) if prior else 0
            prior_variance = prior.get('variance', 1) if prior else 1

            posterior_variance = 1 / (1/prior_variance + len(values)/variance) if variance > 0 else prior_variance
            posterior_mean = posterior_variance * (prior_mean/prior_variance + sum(values)/variance)

            ci_lower = posterior_mean - 1.96 * math.sqrt(posterior_variance)
            ci_upper = posterior_mean + 1.96 * math.sqrt(posterior_variance)

            result[col] = {
                'posterior_mean': round(posterior_mean, 4),
                'posterior_variance': round(posterior_variance, 4),
                'map_estimate': round(posterior_mean, 4),
                'credible_interval_95': [round(ci_lower, 4), round(ci_upper, 4)],
                'description': f"MAP: {posterior_mean:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
            }

        return result

    def monte_carlo_simulation(
        self,
        data: Dict[str, List[Any]],
        columns: List[str],
        n_sims: int = 5000
    ) -> Dict[str, Any]:
        result = {}

        for col in columns:
            values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
            if len(values) < 2:
                continue

            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

            import random
            simulated_means = []
            for _ in range(min(n_sims, 1000)):
                sample = [random.gauss(mean, std) for _ in range(len(values))]
                simulated_means.append(sum(sample) / len(sample))

            simulated_means.sort()
            ci_lower = simulated_means[int(0.025 * len(simulated_means))]
            ci_upper = simulated_means[int(0.975 * len(simulated_means))]

            result[col] = {
                'simulated_mean': round(sum(simulated_means) / len(simulated_means), 4),
                'confidence_interval_95': [round(ci_lower, 4), round(ci_upper, 4)],
                'value_at_risk_95': round(min(values), 4),
                'description': f"Monte Carlo CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
            }

        return result

    def markov_chain_analysis(
        self,
        data: Dict[str, List[Any]],
        columns: List[str],
        states: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        result = {}

        for col in columns:
            values = data.get(col, [])
            if len(values) < 2:
                continue

            unique_vals = set(v for v in values if v is not None)
            if len(unique_vals) > 20 or any(isinstance(v, (int, float)) for v in unique_vals):
                continue

            transitions = defaultdict(lambda: defaultdict(int))
            for i in range(len(values) - 1):
                if values[i] is not None and values[i+1] is not None:
                    transitions[values[i]][values[i+1]] += 1

            if not transitions:
                continue

            trans_probs = {}
            for state, counts in transitions.items():
                total = sum(counts.values())
                trans_probs[state] = {
                    next_state: count / total
                    for next_state, count in counts.items()
                }

            steady_state = self._compute_steady_state(trans_probs)

            result[col] = {
                'transition_matrix': trans_probs,
                'steady_state': {k: round(v, 4) for k, v in steady_state.items()},
                'unique_states': len(unique_vals),
                'description': f"Markov chain with {len(unique_vals)} states"
            }

        return result

    def pareto_analysis(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> Dict[str, Any]:
        result = {}

        for col in columns:
            values = data.get(col, [])
            if not values:
                continue

            if any(isinstance(v, (int, float)) for v in values):
                numeric_vals = [v for v in values if isinstance(v, (int, float))]
                numeric_vals.sort(reverse=True)
                cumsum = 0
                total = sum(abs(v) for v in numeric_vals)

                vital_few_count = 0
                for val in numeric_vals:
                    cumsum += abs(val)
                    vital_few_count += 1
                    if cumsum >= 0.8 * total:
                        break

                result[col] = {
                    'pareto_80_20': round(vital_few_count / len(numeric_vals) * 100, 2),
                    'vital_few_count': vital_few_count,
                    'total_count': len(numeric_vals),
                    'description': f"{vital_few_count} values ({vital_few_count/len(numeric_vals)*100:.1f}%) generate 80% of total"
                }
            else:
                counter = Counter(values)
                sorted_items = counter.most_common()
                cumsum = 0
                total = sum(count for _, count in sorted_items)

                vital_few = []
                for item, count in sorted_items:
                    cumsum += count
                    vital_few.append(item)
                    if cumsum >= 0.8 * total:
                        break

                result[col] = {
                    'vital_few': vital_few,
                    'vital_few_count': len(vital_few),
                    'total_categories': len(counter),
                    'description': f"{len(vital_few)} categories generate 80% of observations"
                }

        return result

    def entropy_analysis(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> Dict[str, Any]:
        result = {}

        for col in columns:
            values = [v for v in data.get(col, []) if v is not None]
            if not values:
                continue

            counter = Counter(values)
            total = len(values)
            entropy = 0
            for count in counter.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            max_entropy = math.log2(len(counter)) if len(counter) > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            diversity = 'high' if entropy > math.log2(10) else ('medium' if entropy > math.log2(2) else 'low')

            result[col] = {
                'shannon_entropy': round(entropy, 4),
                'normalized_entropy': round(normalized_entropy, 4),
                'unique_values': len(counter),
                'diversity': diversity,
                'description': f"Shannon entropy: {entropy:.4f}, diversity: {diversity}"
            }

        return result

    def detect_time_series(
        self,
        data: Dict[str, List[Any]],
        columns: List[str]
    ) -> Dict[str, Any]:
        result = {}

        date_cols = []
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                date_cols.append(col)

        if not date_cols:
            return result

        for date_col in date_cols:
            for col in columns:
                if col == date_col or 'date' in col.lower() or 'time' in col.lower():
                    continue

                values = [v for v in data.get(col, []) if isinstance(v, (int, float))]
                if len(values) < 4:
                    continue

                trend = self._detect_trend(values)
                result[f"{date_col}__{col}"] = {
                    'has_trend': trend is not None and trend['strength'] > 0.3,
                    'trend_direction': trend['direction'] if trend else 'none',
                    'suitable_for_forecast': len(values) >= 12,
                    'recommended_methods': self._recommend_forecast_methods(values),
                    'description': f"Time series analysis for {col} over {date_col}"
                }

        return result

    def generate_narrative(
        self,
        insights: Dict[str, Any],
        question: str,
        row_count: int,
        columns: List[str]
    ) -> str:
        paragraphs = []

        paragraphs.append(
            f"Analysis of {row_count} records across {len(columns)} columns in response to: '{question}'"
        )

        stats = insights.get('statistical_summary', {})
        if stats:
            numeric_summary = []
            for col, summary in list(stats.items())[:3]:
                if summary.get('type') == 'numeric':
                    numeric_summary.append(
                        f"{col}: mean={summary.get('mean', 0):.2f}, "
                        f"std={summary.get('std', 0):.2f}, range=[{summary.get('min', 0)}, {summary.get('max', 0)}]"
                    )
            if numeric_summary:
                paragraphs.append(
                    f"Key statistics: {'; '.join(numeric_summary)}"
                )

        patterns = insights.get('patterns', [])
        if patterns:
            pattern_summary = [p.get('description', '') for p in patterns[:2]]
            if pattern_summary:
                paragraphs.append(
                    f"Detected patterns: {'; '.join(pattern_summary)}"
                )

        anomalies = insights.get('anomalies', [])
        if anomalies:
            paragraphs.append(
                f"Found {len(anomalies)} anomalies using multiple detection methods (Z-score, IQR, Grubbs)."
            )

        corrs = insights.get('correlations', [])
        if corrs:
            top_corr = corrs[0]
            paragraphs.append(
                f"Strong correlation found: {top_corr.get('column1')} and {top_corr.get('column2')} "
                f"(r={top_corr.get('pearson', 0):.4f})"
            )

        quality = insights.get('quality_score', 0)
        if quality < 0.5:
            paragraphs.append("Note: Data quality is limited; interpret with caution.")

        return " ".join(paragraphs)

    def generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        recommendations = []

        if insights.get('anomalies'):
            recommendations.append(
                "Investigate identified anomalies for data quality issues or genuine outliers."
            )

        patterns = insights.get('patterns', [])
        trend_patterns = [p for p in patterns if p.get('pattern_type') == 'trend']
        if trend_patterns:
            direction = trend_patterns[0].get('direction', 'unknown')
            recommendations.append(
                f"Prepare for {direction} trend; plan resources accordingly."
            )

        seasonal_patterns = [p for p in patterns if p.get('pattern_type') == 'seasonality']
        if seasonal_patterns:
            recommendations.append(
                "Implement seasonal adjustment strategies to improve forecasting accuracy."
            )

        corrs = insights.get('correlations', [])
        if corrs and abs(corrs[0].get('pearson', 0)) > 0.8:
            recommendations.append(
                f"Consider multicollinearity when building predictive models."
            )

        dists = insights.get('distributions', {})
        non_normal = [col for col, dist in dists.items() if not dist.get('is_normal')]
        if non_normal:
            recommendations.append(
                "Use non-parametric statistical tests due to non-normal distributions."
            )

        discrete = insights.get('discrete_analysis', {})
        entropy_analysis = discrete.get('entropy', {})
        high_entropy = [col for col, e in entropy_analysis.items() if e.get('diversity') == 'high']
        if high_entropy:
            recommendations.append(
                f"High-entropy columns ({', '.join(high_entropy[:2])}) have diverse values; "
                "consider dimensionality reduction."
            )

        return recommendations[:5]

    def generate_executive_summary(self, insights: Dict[str, Any]) -> str:
        summary_parts = []

        stats = insights.get('statistical_summary', {})
        numeric_cols = len([s for s in stats.values() if s.get('type') == 'numeric'])
        summary_parts.append(
            f"Analysis identified patterns in {numeric_cols} numeric dimensions."
        )

        anomalies = insights.get('anomalies', [])
        patterns = insights.get('patterns', [])
        corrs = insights.get('correlations', [])

        findings = []
        if anomalies:
            findings.append(f"{len(anomalies)} anomalies")
        if patterns:
            findings.append(f"{len(patterns)} patterns")
        if corrs:
            findings.append(f"{len(corrs)} correlations")

        if findings:
            summary_parts.append(f"Key findings include {', '.join(findings)}.")

        quality = insights.get('quality_score', 0)
        if quality > 0.7:
            summary_parts.append("Data quality is sufficient for actionable decision-making.")
        elif quality > 0.4:
            summary_parts.append("Additional data validation is recommended before action.")
        else:
            summary_parts.append("Further investigation needed to ensure data reliability.")

        return " ".join(summary_parts[:3])


    @staticmethod
    def _median(sorted_vals: List[float]) -> float:
        n = len(sorted_vals)
        if n == 0:
            return 0
        return (sorted_vals[n // 2] if n % 2 == 1 else
                (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2)

    @staticmethod
    def _quantile(sorted_vals: List[float], q: float) -> float:
        if not sorted_vals:
            return 0
        idx = q * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

    @staticmethod
    def _skewness(values: List[float], mean: float, std: float) -> float:
        if std == 0 or len(values) < 3:
            return 0
        n = len(values)
        m3 = sum((x - mean) ** 3 for x in values) / n
        return m3 / (std ** 3)

    @staticmethod
    def _kurtosis(values: List[float], mean: float, std: float) -> float:
        if std == 0 or len(values) < 4:
            return 0
        n = len(values)
        m4 = sum((x - mean) ** 4 for x in values) / n
        return (m4 / (std ** 4)) - 3

    @staticmethod
    def _most_common(values: List, n: int = 1) -> List:
        counter = Counter(v for v in values if v is not None)
        return [v for v, _ in counter.most_common(n)]

    @staticmethod
    def _detect_trend(values: List[float]) -> Optional[Dict[str, Any]]:
        if len(values) < 2:
            return None

        n = len(values)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        numerator = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        slope = numerator / denominator
        r_squared = (numerator ** 2) / (denominator * sum((v - y_mean) ** 2 for v in values))

        return {
            'direction': 'upward' if slope > 0 else 'downward',
            'slope': slope,
            'strength': r_squared
        }

    @staticmethod
    def _autocorrelation(values: List[float], lag: int = 1) -> float:
        if len(values) <= lag:
            return 0

        mean = sum(values) / len(values)
        c0 = sum((v - mean) ** 2 for v in values) / len(values)

        if c0 == 0:
            return 0

        c_lag = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(len(values) - lag)) / len(values)
        return c_lag / c0

    @staticmethod
    def _zscore_anomalies(values: List[float], threshold: float = 3.0) -> List[Tuple[int, float, float]]:
        if len(values) < 2:
            return []

        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

        if std == 0:
            return []

        anomalies = []
        for i, val in enumerate(values):
            z = abs((val - mean) / std)
            if z >= threshold:
                anomalies.append((i, val, z))

        return sorted(anomalies, key=lambda x: x[2], reverse=True)

    @staticmethod
    def _iqr_anomalies(values: List[float]) -> Tuple[List[Tuple[int, float]], float]:
        if len(values) < 4:
            return [], 0

        sorted_vals = sorted(values)
        q1 = InsightEngine._quantile(sorted_vals, 0.25)
        q3 = InsightEngine._quantile(sorted_vals, 0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = [(i, v) for i, v in enumerate(values) if v < lower_bound or v > upper_bound]
        return anomalies, iqr

    @staticmethod
    def _grubbs_test(values: List[float]) -> Optional[float]:
        if len(values) < 3:
            return None

        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

        if std == 0:
            return None

        max_deviation = max(abs(v - mean) for v in values)
        outlier_idx = max(range(len(values)), key=lambda i: abs(values[i] - mean))

        return values[outlier_idx] if max_deviation > 2 * std else None

    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return 0

        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denom_x = sum((xi - x_mean) ** 2 for xi in x)
        denom_y = sum((yi - y_mean) ** 2 for yi in y)

        denominator = math.sqrt(denom_x * denom_y)
        return numerator / denominator if denominator > 0 else 0

    @staticmethod
    def _spearman_correlation(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(x) != len(y):
            return 0

        x_ranks = InsightEngine._rank(x)
        y_ranks = InsightEngine._rank(y)

        return InsightEngine._pearson_correlation(x_ranks, y_ranks)

    @staticmethod
    def _rank(values: List[float]) -> List[float]:
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank
        return ranks

    @staticmethod
    def _correlation_strength(r: float) -> str:
        abs_r = abs(r)
        if abs_r > 0.9:
            return 'very_strong'
        elif abs_r > 0.7:
            return 'strong'
        elif abs_r > 0.5:
            return 'moderate'
        elif abs_r > 0.3:
            return 'weak'
        else:
            return 'very_weak'

    @staticmethod
    def _shapiro_wilk_approx(values: List[float]) -> float:
        if len(values) < 3:
            return 1.0

        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

        if std == 0:
            return 1.0

        skewness = sum((x - mean) ** 3 for x in values) / (len(values) * (std ** 3))
        kurtosis = sum((x - mean) ** 4 for x in values) / (len(values) * (std ** 4)) - 3

        normalcy = 1.0 / (1.0 + abs(skewness) + abs(kurtosis) / 3.0)
        return normalcy

    @staticmethod
    def _infer_distribution(values: List[float]) -> str:
        if len(values) < 5:
            return 'unknown'

        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

        skewness = InsightEngine._skewness(values, mean, std)
        min_val = min(values)
        max_val = max(values)

        if min_val >= 0 and skewness > 1:
            return 'exponential'
        elif max_val - min_val < 3 * std:
            return 'uniform'
        elif abs(skewness) < 0.5:
            return 'normal'
        else:
            return 'other'

    @staticmethod
    def _compute_steady_state(trans_probs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not trans_probs:
            return {}

        states = list(trans_probs.keys())
        state_dist = {s: 1.0 / len(states) for s in states}

        for _ in range(3):
            new_dist = {}
            for state in states:
                new_dist[state] = 0
                for prev_state in states:
                    if prev_state in trans_probs and state in trans_probs[prev_state]:
                        new_dist[state] += state_dist[prev_state] * trans_probs[prev_state][state]
            state_dist = new_dist

        return state_dist

    @staticmethod
    def _recommend_forecast_methods(values: List[float]) -> List[str]:
        methods = []

        if len(values) >= 4:
            methods.append('linear_regression')

        if len(values) >= 12:
            methods.append('seasonal_decomposition')
            methods.append('holt_winters')

        if len(values) >= 8:
            methods.append('exponential_smoothing')

        trend = InsightEngine._detect_trend(values)
        if trend and trend['strength'] > 0.5:
            methods.append('arima')

        return methods

    @staticmethod
    def _calculate_quality_score(insights: Dict[str, Any], row_count: int, col_count: int) -> float:
        score = 0.5

        if row_count >= 1000:
            score += 0.2
        elif row_count >= 100:
            score += 0.1

        if col_count >= 10:
            score += 0.1
        elif col_count >= 5:
            score += 0.05

        anomalies = insights.get('anomalies', [])
        if len(anomalies) == 0:
            score += 0.1
        elif len(anomalies) > row_count * 0.1:
            score -= 0.1

        dists = insights.get('distributions', {})
        normal_dist = len([d for d in dists.values() if d.get('is_normal')])
        if normal_dist > len(dists) * 0.5:
            score += 0.05

        return min(1.0, max(0.0, score))
