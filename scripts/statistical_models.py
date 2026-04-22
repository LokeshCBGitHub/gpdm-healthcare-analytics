import math
import re
import random
import hashlib
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

logger = logging.getLogger('gpdm.stats')
random.seed(42)


class MonteCarloSimulator:

    def simulate(self, values: List[float], n_simulations: int = 10000,
                 forecast_periods: int = 4, label: str = 'metric') -> Dict:
        if not values or len(values) < 2:
            return {'applicable': False}

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else mean * 0.1

        trend = 0
        if len(values) >= 3:
            diffs = [values[i] - values[i-1] for i in range(1, len(values))]
            trend = sum(diffs) / len(diffs)

        scenarios = []
        for _ in range(n_simulations):
            path = [values[-1]]
            for t in range(forecast_periods):
                noise = random.gauss(0, std * 0.5)
                next_val = path[-1] + trend + noise
                path.append(max(0, next_val))
            scenarios.append(path[-1])

        scenarios.sort()

        p5  = scenarios[int(n_simulations * 0.05)]
        p25 = scenarios[int(n_simulations * 0.25)]
        p50 = scenarios[int(n_simulations * 0.50)]
        p75 = scenarios[int(n_simulations * 0.75)]
        p95 = scenarios[int(n_simulations * 0.95)]

        prob_decline = sum(1 for s in scenarios if s < values[-1]) / n_simulations

        return {
            'applicable': True,
            'type': 'monte_carlo',
            'label': f'Monte Carlo Forecast ({n_simulations:,} simulations)',
            'current_value': round(values[-1], 2),
            'projected_median': round(p50, 2),
            'confidence_interval_90': [round(p5, 2), round(p95, 2)],
            'confidence_interval_50': [round(p25, 2), round(p75, 2)],
            'prob_decline': round(prob_decline * 100, 1),
            'prob_increase': round((1 - prob_decline) * 100, 1),
            'trend_per_period': round(trend, 2),
            'volatility': round(std, 2),
            'message': (
                f"Based on {n_simulations:,} Monte Carlo simulations: "
                f"{label} is projected at {p50:,.1f} (median), "
                f"with 90% CI [{p5:,.1f} – {p95:,.1f}]. "
                f"Probability of decline: {prob_decline*100:.0f}%."
            ),
        }


class BayesianEstimator:

    def estimate(self, successes: int, trials: int,
                 prior_alpha: float = 1, prior_beta: float = 1,
                 label: str = 'rate') -> Dict:
        if trials <= 0:
            return {'applicable': False}

        post_alpha = prior_alpha + successes
        post_beta = prior_beta + (trials - successes)

        post_mean = post_alpha / (post_alpha + post_beta)
        post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if (post_alpha > 1 and post_beta > 1) else post_mean
        post_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))
        post_std = math.sqrt(post_var)

        ci_low = max(0, post_mean - 1.96 * post_std)
        ci_high = min(1, post_mean + 1.96 * post_std)

        freq_rate = successes / trials

        return {
            'applicable': True,
            'type': 'bayesian',
            'label': f'Bayesian Estimate: {label}',
            'frequentist_rate': round(freq_rate * 100, 2),
            'bayesian_rate': round(post_mean * 100, 2),
            'credible_interval_95': [round(ci_low * 100, 2), round(ci_high * 100, 2)],
            'posterior_mode': round(post_mode * 100, 2),
            'prior': f'Beta({prior_alpha},{prior_beta})',
            'posterior': f'Beta({post_alpha:.0f},{post_beta:.0f})',
            'uncertainty': round(post_std * 100, 3),
            'message': (
                f"Bayesian {label}: {post_mean*100:.1f}% "
                f"(95% credible interval: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]). "
                f"Frequentist: {freq_rate*100:.1f}%. "
                f"Posterior uncertainty: ±{post_std*100:.2f}%."
            ),
        }


class TimeSeriesForecaster:

    def forecast(self, values: List[float], periods_ahead: int = 4,
                 labels: List[str] = None, label: str = 'metric') -> Dict:
        if not values or len(values) < 3:
            return {'applicable': False}

        n = len(values)
        alpha = 0.3
        beta = 0.1

        level = values[0]
        trend = (values[-1] - values[0]) / (n - 1)

        fitted = []
        for t in range(n):
            if t == 0:
                fitted.append(level)
                continue
            prev_level = level
            level = alpha * values[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            fitted.append(level + trend)

        forecasts = []
        for h in range(1, periods_ahead + 1):
            forecasts.append(round(level + h * trend, 2))

        residuals = [values[i] - fitted[i] for i in range(n)]
        resid_std = math.sqrt(sum(r**2 for r in residuals) / max(n - 2, 1))

        forecast_ci = []
        for h in range(1, periods_ahead + 1):
            margin = 1.96 * resid_std * math.sqrt(h)
            forecast_ci.append([
                round(level + h * trend - margin, 2),
                round(level + h * trend + margin, 2)
            ])

        if abs(trend) < resid_std * 0.1:
            trend_desc = 'stable (no significant trend)'
        elif trend > 0:
            trend_desc = f'upward (+{trend:.2f} per period)'
        else:
            trend_desc = f'downward ({trend:.2f} per period)'

        seasonal = False
        if n >= 6:
            diffs = [values[i] - values[i-1] for i in range(1, n)]
            sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
            if sign_changes > n * 0.4:
                seasonal = True

        return {
            'applicable': True,
            'type': 'time_series',
            'label': f'Time-Series Forecast: {label}',
            'historical': [round(v, 2) for v in values],
            'fitted': [round(v, 2) for v in fitted],
            'forecast': forecasts,
            'forecast_ci_95': forecast_ci,
            'trend': round(trend, 3),
            'trend_description': trend_desc,
            'seasonal_detected': seasonal,
            'method': 'Double Exponential Smoothing (Holt)',
            'parameters': {'alpha': alpha, 'beta': beta},
            'residual_std': round(resid_std, 2),
            'message': (
                f"Forecast for {label}: next {periods_ahead} periods projected at "
                f"{', '.join(str(f) for f in forecasts)}. "
                f"Trend is {trend_desc}. "
                f"{'Seasonality detected. ' if seasonal else ''}"
                f"Method: Holt's exponential smoothing."
            ),
        }


class MarkovChainAnalyzer:

    def analyze(self, states: List[str], label: str = 'state') -> Dict:
        if not states or len(states) < 2:
            return {'applicable': False}

        transitions = defaultdict(lambda: defaultdict(int))
        state_counts = Counter(states)
        unique_states = sorted(state_counts.keys())

        for i in range(len(states) - 1):
            transitions[states[i]][states[i+1]] += 1

        trans_matrix = {}
        for from_state in unique_states:
            total = sum(transitions[from_state].values())
            if total > 0:
                trans_matrix[from_state] = {
                    to_state: round(transitions[from_state][to_state] / total, 3)
                    for to_state in unique_states
                }
            else:
                trans_matrix[from_state] = {s: 0 for s in unique_states}

        n = len(unique_states)
        state_idx = {s: i for i, s in enumerate(unique_states)}
        pi = [1.0 / n] * n

        for _ in range(100):
            new_pi = [0.0] * n
            for j in range(n):
                for i in range(n):
                    from_s = unique_states[i]
                    to_s = unique_states[j]
                    new_pi[j] += pi[i] * trans_matrix.get(from_s, {}).get(to_s, 0)
            total = sum(new_pi) or 1
            pi = [p / total for p in new_pi]

        steady_state = {unique_states[i]: round(pi[i] * 100, 1) for i in range(n)}

        absorbing = []
        for s in unique_states:
            self_prob = trans_matrix.get(s, {}).get(s, 0)
            if self_prob > 0.9:
                absorbing.append(s)

        return {
            'applicable': True,
            'type': 'markov_chain',
            'label': f'Markov Chain Analysis: {label}',
            'states': unique_states,
            'transition_matrix': trans_matrix,
            'steady_state': steady_state,
            'absorbing_states': absorbing,
            'state_distribution': {k: v for k, v in state_counts.most_common()},
            'message': (
                f"Markov transition analysis across {len(unique_states)} states. "
                f"Steady-state distribution: "
                + ", ".join(f"{s}: {p}%" for s, p in steady_state.items()) + ". "
                + (f"Absorbing states (hard to leave): {', '.join(absorbing)}. " if absorbing else "")
            ),
        }


class BootstrapEstimator:

    def estimate(self, values: List[float], n_bootstrap: int = 5000,
                 statistic: str = 'mean', label: str = 'metric') -> Dict:
        if not values or len(values) < 2:
            return {'applicable': False}

        stat_fn = {
            'mean': lambda v: sum(v) / len(v),
            'median': lambda v: sorted(v)[len(v) // 2],
            'std': lambda v: math.sqrt(sum((x - sum(v)/len(v))**2 for x in v) / len(v)),
        }.get(statistic, lambda v: sum(v) / len(v))

        observed = stat_fn(values)

        boot_stats = []
        n = len(values)
        for _ in range(n_bootstrap):
            sample = [values[random.randint(0, n-1)] for _ in range(n)]
            boot_stats.append(stat_fn(sample))

        boot_stats.sort()
        ci_low = boot_stats[int(n_bootstrap * 0.025)]
        ci_high = boot_stats[int(n_bootstrap * 0.975)]
        se = math.sqrt(sum((b - observed)**2 for b in boot_stats) / n_bootstrap)

        return {
            'applicable': True,
            'type': 'bootstrap',
            'label': f'Bootstrap CI: {label}',
            'statistic': statistic,
            'observed': round(observed, 2),
            'ci_95': [round(ci_low, 2), round(ci_high, 2)],
            'standard_error': round(se, 3),
            'n_bootstrap': n_bootstrap,
            'message': (
                f"Bootstrap {statistic} of {label}: {observed:,.2f} "
                f"(95% CI: [{ci_low:,.2f}, {ci_high:,.2f}], "
                f"SE: {se:,.3f}, {n_bootstrap:,} resamples)."
            ),
        }


class RegressionAnalyzer:

    def fit(self, x_values: List[float], y_values: List[float],
            x_label: str = 'x', y_label: str = 'y') -> Dict:
        n = len(x_values)
        if n < 3 or n != len(y_values):
            return {'applicable': False}

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        ss_xy = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        ss_xx = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        ss_yy = sum((y_values[i] - y_mean) ** 2 for i in range(n))

        if ss_xx == 0 or ss_yy == 0:
            return {'applicable': False}

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        r = ss_xy / math.sqrt(ss_xx * ss_yy)
        r_squared = r ** 2

        predicted = [intercept + slope * x for x in x_values]
        residuals = [y_values[i] - predicted[i] for i in range(n)]
        rse = math.sqrt(sum(r**2 for r in residuals) / max(n - 2, 1))

        se_slope = rse / math.sqrt(ss_xx) if ss_xx > 0 else 0
        t_stat = slope / se_slope if se_slope > 0 else 0

        if r_squared > 0.7:
            strength = 'strong'
        elif r_squared > 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'

        direction = 'positive' if slope > 0 else 'negative'

        return {
            'applicable': True,
            'type': 'regression',
            'label': f'Regression: {y_label} vs {x_label}',
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'r_squared': round(r_squared, 4),
            'correlation': round(r, 4),
            'residual_se': round(rse, 4),
            't_statistic': round(t_stat, 2),
            'significant': abs(t_stat) > 2.0,
            'strength': strength,
            'direction': direction,
            'equation': f'{y_label} = {intercept:.2f} + {slope:.4f} × {x_label}',
            'message': (
                f"Regression: {y_label} = {intercept:.2f} + {slope:.4f} × {x_label}. "
                f"R² = {r_squared:.3f} ({strength} {direction} relationship). "
                f"{'Statistically significant (|t|>2).' if abs(t_stat) > 2 else 'Not statistically significant.'} "
                f"For each unit increase in {x_label}, {y_label} changes by {slope:+.2f}."
            ),
        }


class RiskScorer:

    def score_rows(self, rows: List, columns: List) -> Dict:
        if not rows or not columns:
            return {'applicable': False}

        col_map = {c.lower().replace(' ', '_'): i for i, c in enumerate(columns)}

        risk_factors = {}
        for col_name, col_idx in col_map.items():
            if col_name.endswith('_id') or col_name == 'id':
                continue
            vals = [row[col_idx] for row in rows[:50]
                    if col_idx < len(row) and isinstance(row[col_idx], (int, float))]
            if len(vals) < 3:
                continue
            sorted_vals = sorted(vals)
            p90_idx = int(len(sorted_vals) * 0.9)
            threshold = sorted_vals[p90_idx] if sorted_vals[p90_idx] > 0 else 1
            risk_factors[col_name] = {
                'weight': 1.0,
                'high_threshold': threshold,
                'col_idx': col_idx,
            }

        if not risk_factors:
            return {'applicable': False}

        total_w = sum(f['weight'] for f in risk_factors.values())
        for f in risk_factors.values():
            f['weight'] /= total_w

        scored_rows = []
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for row in rows[:100]:
            total_score = 0
            total_weight = 0
            factors_used = []

            for factor_name, config in risk_factors.items():
                col_idx = config['col_idx']

                if col_idx is not None and col_idx < len(row):
                    val = row[col_idx]
                    if isinstance(val, (int, float)):
                        threshold = config['high_threshold']
                        normalized = min(val / threshold, 2.0) if threshold > 0 else 0
                        total_score += normalized * config['weight']
                        total_weight += config['weight']
                        factors_used.append(factor_name)

            if total_weight > 0:
                final_score = total_score / total_weight * 100
                risk_level = (
                    'critical' if final_score > 150 else
                    'high' if final_score > 100 else
                    'medium' if final_score > 50 else 'low'
                )
                risk_distribution[risk_level] += 1
                scored_rows.append({
                    'label': str(row[0]) if row else 'Row',
                    'score': round(final_score, 1),
                    'level': risk_level,
                    'factors': factors_used,
                })

        if not scored_rows:
            return {'applicable': False}

        avg_score = sum(r['score'] for r in scored_rows) / len(scored_rows)
        high_risk = [r for r in scored_rows if r['level'] in ('high', 'critical')]

        return {
            'applicable': True,
            'type': 'risk_score',
            'label': 'Predictive Risk Scores',
            'avg_score': round(avg_score, 1),
            'distribution': risk_distribution,
            'high_risk_count': len(high_risk),
            'top_risks': sorted(scored_rows, key=lambda x: -x['score'])[:5],
            'factors_used': list(set(f for r in scored_rows for f in r['factors'])),
            'message': (
                f"Risk analysis: avg score {avg_score:.0f}/100. "
                f"Distribution: {risk_distribution['critical']} critical, "
                f"{risk_distribution['high']} high, {risk_distribution['medium']} medium, "
                f"{risk_distribution['low']} low risk. "
                f"{len(high_risk)} entities flagged as elevated risk."
            ),
        }


class DistributionFitter:

    def fit(self, values: List[float], label: str = 'metric') -> Dict:
        if not values or len(values) < 5:
            return {'applicable': False}

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean)**2 for x in values) / n
        std = math.sqrt(variance) if variance > 0 else 0
        sorted_vals = sorted(values)

        if std > 0:
            skew = sum((x - mean)**3 for x in values) / (n * std**3)
        else:
            skew = 0

        if std > 0:
            kurt = sum((x - mean)**4 for x in values) / (n * std**4) - 3
        else:
            kurt = 0

        fits = {}

        jb_stat = n / 6 * (skew**2 + (kurt**2) / 4)
        fits['normal'] = {
            'score': max(0, 1 - jb_stat / 20),
            'params': {'mean': round(mean, 2), 'std': round(std, 2)},
        }

        if min(values) > 0:
            log_vals = [math.log(v) for v in values]
            log_mean = sum(log_vals) / n
            log_var = sum((x - log_mean)**2 for x in log_vals) / n
            log_std = math.sqrt(log_var) if log_var > 0 else 0
            log_skew = sum((x - log_mean)**3 for x in log_vals) / (n * max(log_std**3, 0.001))
            fits['log_normal'] = {
                'score': max(0, 1 - abs(log_skew) / 3),
                'params': {'log_mean': round(log_mean, 2), 'log_std': round(log_std, 2)},
            }

        if min(values) >= 0 and skew > 1:
            lam = 1 / mean if mean > 0 else 1
            fits['exponential'] = {
                'score': max(0, min(1, skew / 2 - 0.2)),
                'params': {'lambda': round(lam, 4), 'mean': round(mean, 2)},
            }

        expected_std = (max(values) - min(values)) / math.sqrt(12)
        uniform_ratio = std / expected_std if expected_std > 0 else 0
        fits['uniform'] = {
            'score': max(0, 1 - abs(1 - uniform_ratio)),
            'params': {'min': round(min(values), 2), 'max': round(max(values), 2)},
        }

        best_name = max(fits, key=lambda k: fits[k]['score'])
        best = fits[best_name]

        return {
            'applicable': True,
            'type': 'distribution',
            'label': f'Distribution Analysis: {label}',
            'best_fit': best_name.replace('_', ' ').title(),
            'fit_score': round(best['score'], 3),
            'parameters': best['params'],
            'statistics': {
                'mean': round(mean, 2),
                'median': round(sorted_vals[n // 2], 2),
                'std': round(std, 2),
                'skewness': round(skew, 3),
                'kurtosis': round(kurt, 3),
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'iqr': round(sorted_vals[int(n*0.75)] - sorted_vals[int(n*0.25)], 2),
            },
            'all_fits': {k: round(v['score'], 3) for k, v in fits.items()},
            'message': (
                f"Distribution of {label}: best fit is {best_name.replace('_',' ').title()} "
                f"(score: {best['score']:.2f}). "
                f"Mean: {mean:,.2f}, Std: {std:,.2f}, "
                f"Skew: {skew:+.2f}, Kurtosis: {kurt:+.2f}. "
                f"{'Right-skewed (long tail on high end). ' if skew > 0.5 else ''}"
                f"{'Left-skewed. ' if skew < -0.5 else ''}"
                f"{'Heavy-tailed (extreme values likely). ' if kurt > 2 else ''}"
            ),
        }


class WhatIfEngine:

    _SENSITIVITY_CHANGES = [
        {'change': -0.05, 'label_template': 'Reduce {metric} by 5%'},
        {'change': -0.10, 'label_template': 'Reduce {metric} by 10%'},
        {'change': +0.10, 'label_template': 'Increase {metric} by 10%'},
        {'change': +0.25, 'label_template': 'Stretch goal: {metric} +25%'},
    ]

    def analyze(self, question: str, rows: List, columns: List) -> Dict:
        if not rows or not columns:
            return {'applicable': False}

        metric_name = None
        baseline = self._get_baseline(rows, columns)
        if baseline is None:
            return {'applicable': False}

        for col in reversed(columns):
            cn = col.lower()
            if not cn.endswith('_id') and cn != 'id':
                metric_name = col.replace('_', ' ').title()
                break
        if not metric_name:
            metric_name = 'metric'

        scenarios = []
        for tmpl in self._SENSITIVITY_CHANGES:
            new_val = baseline * (1 + tmpl['change'])
            impact = new_val - baseline
            scenarios.append({
                'scenario': tmpl['label_template'].format(metric=metric_name),
                'baseline': round(baseline, 2),
                'projected': round(new_val, 2),
                'impact': round(impact, 2),
                'impact_pct': round(tmpl['change'] * 100, 1),
            })

        if not scenarios:
            return {'applicable': False}

        return {
            'applicable': True,
            'type': 'what_if',
            'label': 'What-If Scenario Analysis',
            'scenarios': scenarios,
            'message': (
                f"What-if analysis: {len(scenarios)} scenarios modeled. "
                + " | ".join(
                    f"{s['scenario']}: {s['projected']:,.1f} ({s['impact_pct']:+.0f}%)"
                    for s in scenarios
                )
            ),
        }

    def _get_baseline(self, rows, columns):
        if not rows:
            return None
        for col_idx in range(len(columns or []) - 1, -1, -1):
            vals = [r[col_idx] for r in rows if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if vals:
                return sum(vals) / len(vals)
        return None


class StatisticalAnalyzer:

    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.bayesian = BayesianEstimator()
        self.forecaster = TimeSeriesForecaster()
        self.markov = MarkovChainAnalyzer()
        self.bootstrap = BootstrapEstimator()
        self.regression = RegressionAnalyzer()
        self.risk_scorer = RiskScorer()
        self.dist_fitter = DistributionFitter()
        self.what_if = WhatIfEngine()

    def analyze(self, question: str, rows: List, columns: List,
                intent: Dict = None) -> List[Dict]:
        if not rows or not columns:
            return []

        results = []
        q = question.lower()

        num_cols = []
        for col_idx, col_name in enumerate(columns):
            vals = [r[col_idx] for r in rows
                    if col_idx < len(r) and isinstance(r[col_idx], (int, float))]
            if len(vals) >= 2:
                num_cols.append((col_idx, col_name, vals))

        if not num_cols:
            return []

        def _col_relevance(col_tuple):
            idx, name, vals = col_tuple
            cn = name.lower()
            mean_v = sum(vals) / len(vals) if vals else 0
            variance = sum((x - mean_v) ** 2 for x in vals) / len(vals) if vals else 0
            if variance < 1e-10:
                return -1

            score = 0
            q_words = set(re.findall(r'\b\w+\b', q))
            col_words = set(cn.replace('_', ' ').split())
            if col_words & q_words:
                score += 10
            agg_signals = {'rate', 'pct', 'avg', 'average', 'total', 'sum', 'mean'}
            if col_words & agg_signals:
                score += 5
            if cn in ('pct', 'count', 'cnt'):
                score -= 3
            score += idx * 0.1
            return score

        viable_cols = [(idx, name, vals) for idx, name, vals in num_cols
                       if _col_relevance((idx, name, vals)) >= 0]
        if not viable_cols:
            return []

        viable_cols.sort(key=lambda c: _col_relevance(c), reverse=True)
        primary_idx, primary_name, primary_vals = viable_cols[0]
        labels = [str(r[0]) for r in rows] if rows else []

        is_time_series = any(kw in q for kw in ['trend', 'time', 'month', 'quarter',
                                                   'year', 'over', 'period', 'growth'])
        is_rate_query = any(kw in q for kw in ['rate', 'percentage', 'ratio', 'proportion'])
        is_breakdown = any(kw in q for kw in ['by', 'per', 'breakdown', 'distribution',
                                                'compare', 'type', 'category', 'group'])
        is_forecast = any(kw in q for kw in ['forecast', 'predict', 'project', 'future',
                                               'next', 'growth', 'decline'])
        has_enough_data = len(primary_vals) >= 5

        if has_enough_data and is_breakdown:
            dist = self.dist_fitter.fit(primary_vals, primary_name)
            if dist.get('applicable'):
                results.append(dist)

        if has_enough_data:
            boot = self.bootstrap.estimate(primary_vals, label=primary_name)
            if boot.get('applicable'):
                results.append(boot)

        if is_forecast or (is_time_series and len(primary_vals) >= 4):
            mc = self.monte_carlo.simulate(primary_vals, label=primary_name)
            if mc.get('applicable'):
                results.append(mc)

        if len(primary_vals) >= 4 and is_time_series:
            ts = self.forecaster.forecast(primary_vals, labels=labels, label=primary_name)
            if ts.get('applicable'):
                results.append(ts)

        if is_rate_query:
            mean_primary = sum(primary_vals) / len(primary_vals) if primary_vals else 0
            is_plausible_rate = 0 <= mean_primary <= 100
            if len(num_cols) >= 2 and is_plausible_rate and mean_primary > 0:
                count_vals = num_cols[0][2]
                total_n = sum(count_vals)
                if total_n > 0:
                    successes = int(total_n * mean_primary / 100)
                    bay = self.bayesian.estimate(successes, int(total_n), label=primary_name)
                    if bay.get('applicable'):
                        results.append(bay)

        if is_time_series and len(rows) >= 20:
            cat_col_idx = None
            for i, c in enumerate(columns):
                cn = c.lower()
                if any(kw in cn for kw in ['status', 'state']):
                    cat_vals = [str(r[i]) for r in rows if i < len(r) and r[i] is not None]
                    if len(cat_vals) >= 20 and len(set(cat_vals)) >= 2:
                        cat_col_idx = i
                        break

            if cat_col_idx is not None:
                cat_vals = [str(r[cat_col_idx]) for r in rows if cat_col_idx < len(r)]
                mk = self.markov.analyze(cat_vals, columns[cat_col_idx])
                if mk.get('applicable'):
                    ss = mk.get('steady_state', {})
                    ss_vals = [v for v in ss.values() if v > 0]
                    if ss_vals and max(ss_vals) > 5:
                        results.append(mk)

        if len(num_cols) >= 2 and len(rows) >= 5:
            x_idx, x_name, x_vals = num_cols[0]
            y_idx, y_name, y_vals = num_cols[-1]
            if x_idx != y_idx:
                reg = self.regression.fit(x_vals[:len(y_vals)], y_vals[:len(x_vals)],
                                          x_name, y_name)
                r2 = reg.get('r_squared', 0)
                if reg.get('applicable') and 0.1 < r2 < 0.999:
                    results.append(reg)

        if len(num_cols) >= 3 and len(rows) >= 10:
            risk = self.risk_scorer.score_rows(rows, columns)
            if risk.get('applicable'):
                results.append(risk)

        if is_forecast or 'what if' in q or 'scenario' in q:
            whatif = self.what_if.analyze(question, rows, columns)
            if whatif.get('applicable'):
                results.append(whatif)

        return results[:4]
