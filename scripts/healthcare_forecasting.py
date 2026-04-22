import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger('gpdm.forecasting')


MODEL_EXPLANATIONS = {
    'holt_winters': {
        'name': 'Exponential Smoothing (Holt-Winters)',
        'plain_english': (
            'This model looks at your historical data and gives more weight to '
            'recent trends while still considering older patterns. Think of it like '
            'a weather forecast — it knows that last month matters more than last year, '
            'but seasonal patterns (like flu season spikes) still count. '
            'It automatically detects repeating cycles in your data.'
        ),
        'strengths': 'Best for data with clear seasonal patterns (e.g., claims spike every winter)',
        'hipaa_note': 'Runs entirely on your server. No patient data leaves your network.',
    },
    'linear_trend': {
        'name': 'Trend Analysis with Seasonal Adjustment',
        'plain_english': (
            'This model draws the best-fit line through your data to find the overall '
            'direction — is it going up, down, or flat? Then it layers in monthly '
            'patterns (like higher ER visits in summer). Think of it as asking: '
            '"If current trends continue, where will we be in 6 months?"'
        ),
        'strengths': 'Best for identifying long-term growth or decline trends',
        'hipaa_note': 'Runs entirely on your server. No patient data leaves your network.',
    },
    'moving_average': {
        'name': 'Smoothed Moving Average',
        'plain_english': (
            'This model takes the average of recent months to smooth out noise and '
            'random fluctuations. It answers the question: "Ignoring the bumps, '
            'what is the steady-state level?" It is the most conservative forecast '
            'and is useful when trends are unclear.'
        ),
        'strengths': 'Best for stable metrics where you want a conservative baseline',
        'hipaa_note': 'Runs entirely on your server. No patient data leaves your network.',
    },
}


FORECAST_METRICS = {
    'monthly_claim_volume': {
        'label': 'Monthly Claim Volume',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   COUNT(*) as volume
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL
            GROUP BY month ORDER BY month
        """,
        'unit': 'claims',
        'business_context': 'Forecasting claim volume helps plan staffing, '
            'budget allocation, and identify unexpected surges early.',
    },
    'monthly_revenue': {
        'label': 'Monthly Revenue',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   SUM(BILLED_AMOUNT) as revenue
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL AND BILLED_AMOUNT > 0
            GROUP BY month ORDER BY month
        """,
        'unit': 'dollars',
        'business_context': 'Revenue forecasting supports financial planning, '
            'capitation rate negotiations, and cash flow management.',
    },
    'monthly_cost': {
        'label': 'Monthly Cost / Paid Amount',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   SUM(PAID_AMOUNT) as cost
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL AND PAID_AMOUNT > 0
            GROUP BY month ORDER BY month
        """,
        'unit': 'dollars',
        'business_context': 'Cost forecasting is critical for medical loss ratio '
            'tracking, reserve setting, and identifying cost trend acceleration.',
    },
    'monthly_member_growth': {
        'label': 'Member Enrollment Trend',
        'sql': """
            SELECT strftime('%Y-%m', ENROLLMENT_DATE) as month,
                   COUNT(*) as new_members
            FROM members
            WHERE ENROLLMENT_DATE IS NOT NULL
            GROUP BY month ORDER BY month
        """,
        'unit': 'members',
        'business_context': 'Enrollment forecasting drives network adequacy planning, '
            'provider contracting, and premium pricing decisions.',
    },
    'monthly_utilization': {
        'label': 'Monthly Utilization (Encounters)',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   COUNT(*) as encounters
            FROM encounters
            WHERE SERVICE_DATE IS NOT NULL
            GROUP BY month ORDER BY month
        """,
        'unit': 'encounters',
        'business_context': 'Utilization forecasting helps with capacity planning, '
            'appointment scheduling, and identifying over/under-utilization.',
    },
    'denial_rate_trend': {
        'label': 'Claim Denial Rate Trend',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) as denial_rate
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL
            GROUP BY month ORDER BY month
        """,
        'unit': 'percent',
        'business_context': 'Denial rate trends signal revenue cycle health. '
            'Rising denials may indicate coding issues, payer policy changes, or auth gaps.',
    },
    'avg_cost_per_claim': {
        'label': 'Average Cost Per Claim',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   ROUND(AVG(PAID_AMOUNT), 2) as avg_cost
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL AND PAID_AMOUNT > 0
            GROUP BY month ORDER BY month
        """,
        'unit': 'dollars',
        'business_context': 'Cost-per-claim trends reveal unit cost inflation, '
            'case mix shifts, or changes in care intensity.',
    },
    'er_visit_trend': {
        'label': 'ER Visit Volume',
        'sql': """
            SELECT strftime('%Y-%m', SERVICE_DATE) as month,
                   COUNT(*) as er_visits
            FROM encounters
            WHERE SERVICE_DATE IS NOT NULL
                  AND UPPER(VISIT_TYPE) LIKE '%EMERGENCY%'
            GROUP BY month ORDER BY month
        """,
        'unit': 'visits',
        'business_context': 'ER volume forecasting supports staffing, diversion '
            'program planning, and identifying population health intervention needs.',
    },
}


class HealthcareForecaster:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache = {}
        logger.info('HealthcareForecaster initialized — %d metrics available', len(FORECAST_METRICS))

    def _query_data(self, sql: str) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(sql).fetchall()
            return rows
        except Exception as e:
            logger.warning('Forecast query failed: %s', e)
            return []
        finally:
            conn.close()

    def _holt_winters(self, values: np.ndarray, periods: int = 6) -> Tuple[np.ndarray, dict]:
        n = len(values)
        if n < 4:
            return np.full(periods, values.mean()), {'alpha': 0.3, 'beta': 0.1}

        alpha, beta = 0.3, 0.1
        level = values[0]
        trend = (values[min(3, n-1)] - values[0]) / min(3, n-1) if n > 1 else 0

        for i in range(1, n):
            new_level = alpha * values[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = new_level, new_trend

        forecast = np.array([level + (i + 1) * trend for i in range(periods)])

        if n >= 12:
            seasonal = np.zeros(12)
            for i in range(n):
                seasonal[i % 12] += values[i]
            counts = np.array([max(1, n // 12 + (1 if i < n % 12 else 0)) for i in range(12)])
            seasonal = seasonal / counts
            seasonal = seasonal - seasonal.mean()
            last_month = n % 12
            for i in range(periods):
                forecast[i] += seasonal[(last_month + i) % 12]

        return np.maximum(forecast, 0), {'alpha': alpha, 'beta': beta}

    def _linear_trend(self, values: np.ndarray, periods: int = 6) -> Tuple[np.ndarray, dict]:
        n = len(values)
        if n < 3:
            return np.full(periods, values.mean()), {'slope': 0, 'r_squared': 0}

        x = np.arange(n)
        slope, intercept = np.polyfit(x, values, 1)
        r_squared = 1 - np.sum((values - (slope * x + intercept)) ** 2) / np.sum((values - values.mean()) ** 2)

        forecast_x = np.arange(n, n + periods)
        forecast = slope * forecast_x + intercept

        if n >= 12:
            residuals = values - (slope * x + intercept)
            seasonal = np.zeros(12)
            counts = np.zeros(12)
            for i in range(n):
                seasonal[i % 12] += residuals[i]
                counts[i % 12] += 1
            seasonal = np.where(counts > 0, seasonal / counts, 0)
            last_month = n % 12
            for i in range(periods):
                forecast[i] += seasonal[(last_month + i) % 12]

        return np.maximum(forecast, 0), {'slope': round(slope, 2), 'r_squared': round(max(0, r_squared), 3)}

    def _moving_average(self, values: np.ndarray, periods: int = 6, window: int = 3) -> Tuple[np.ndarray, dict]:
        n = len(values)
        if n < 2:
            return np.full(periods, values.mean()), {'window': window}

        w = min(window, n)
        avg = values[-w:].mean()
        if n >= 4:
            recent_trend = (values[-1] - values[-min(4, n)]) / min(3, n - 1)
        else:
            recent_trend = 0

        forecast = np.array([avg + recent_trend * (i + 1) * 0.5 for i in range(periods)])
        return np.maximum(forecast, 0), {'window': w, 'mild_trend': round(recent_trend, 2)}

    def forecast_metric(self, metric_key: str, periods: int = 6) -> Optional[Dict[str, Any]]:
        if metric_key not in FORECAST_METRICS:
            return None

        metric = FORECAST_METRICS[metric_key]
        rows = self._query_data(metric['sql'])
        if len(rows) < 3:
            return {
                'metric': metric_key,
                'label': metric['label'],
                'error': 'Not enough historical data (need at least 3 months)',
            }

        months = [r[0] for r in rows]
        values = np.array([float(r[1] or 0) for r in rows])

        last_month = months[-1]
        try:
            last_dt = datetime.strptime(last_month, '%Y-%m')
        except ValueError:
            last_dt = datetime.now()
        future_months = []
        for i in range(1, periods + 1):
            dt = last_dt + timedelta(days=32 * i)
            future_months.append(dt.strftime('%Y-%m'))

        multi_result = None
        try:
            from forecast_runner import ForecastRunner
            runner = ForecastRunner(forecast_horizon=periods)
            multi_result = runner.run(
                dates=months,
                values=values.tolist(),
                metric_name=metric_key
            )
        except Exception as _e:
            import logging
            logging.getLogger('gpdm.forecast').warning("Multi-model runner failed, using fallback: %s", _e)

        if multi_result and multi_result.best_model:
            def _native(v):
                if hasattr(v, 'item'):
                    return v.item()
                return v

            def _native_list(lst):
                return [round(float(x), 2) for x in lst] if lst else []

            def _native_dict(d):
                if not isinstance(d, dict):
                    return d
                return {k: (_native(v) if not isinstance(v, (dict, list)) else
                           (_native_dict(v) if isinstance(v, dict) else _native_list(v)))
                        for k, v in d.items()}

            best_model_name = multi_result.best_model
            best_result = None
            all_models_info = {}

            for mr in multi_result.all_results:
                model_key = mr.model_name.lower().replace(' ', '_').replace('-', '_')
                fc_vals = mr.forecast_values if mr.forecast_values else []
                all_models_info[model_key] = {
                    'values': _native_list(fc_vals),
                    'explanation': {
                        'name': mr.model_name,
                        'plain_english': (
                            f"This model achieved {mr.mape:.1f}% MAPE (mean absolute percentage error) "
                            f"on the held-out test set. RMSE: {mr.rmse:.1f}, MAE: {mr.mae:.1f}. "
                            + (' '.join(mr.insights[:2]) if mr.insights else '')
                        ),
                        'strengths': ', '.join(mr.insights[:2]) if mr.insights else 'General purpose forecasting',
                        'hipaa_note': 'All computation performed on-premise, no PHI transmitted.',
                    },
                    'params': {
                        'mape': round(mr.mape, 2),
                        'rmse': round(mr.rmse, 2),
                        'mae': round(mr.mae, 2),
                    },
                }
                if mr.model_name == best_model_name:
                    best_result = mr

            consensus = multi_result.consensus_forecast or []
            _band = multi_result.confidence_band or ([], [])
            ci_low_vals = _band[0] if _band[0] else []
            ci_high_vals = _band[1] if _band[1] else []

            best_key = best_model_name.lower().replace(' ', '_').replace('-', '_')
            if best_key not in all_models_info:
                best_key = list(all_models_info.keys())[0] if all_models_info else 'holt_winters'

            chosen_vals = np.array(all_models_info.get(best_key, {}).get('values', consensus))
            current_avg = float(values[-3:].mean()) if len(values) >= 3 else float(values.mean())
            forecast_avg = float(np.mean(chosen_vals)) if len(chosen_vals) > 0 else current_avg
            pct_change = ((forecast_avg - current_avg) / current_avg * 100) if current_avg else 0
            direction = 'increase' if pct_change > 2 else 'decrease' if pct_change < -2 else 'remain stable'

            data_chars = multi_result.data_characteristics or {}
            trend_dir = data_chars.get('trend_direction', 'stable')
            trend_str = data_chars.get('trend_strength', 0)
            seasonality = data_chars.get('seasonality_detected', False)
            cv = data_chars.get('coefficient_of_variation', 0)

            unit = metric['unit']
            if unit == 'dollars':
                fmt_current = f"${current_avg:,.0f}"
                fmt_forecast = f"${forecast_avg:,.0f}"
            elif unit == 'percent':
                fmt_current = f"{current_avg:.1f}%"
                fmt_forecast = f"{forecast_avg:.1f}%"
            else:
                fmt_current = f"{current_avg:,.0f}"
                fmt_forecast = f"{forecast_avg:,.0f}"

            best_mape = best_result.mape if best_result else 0
            models_tested = len(multi_result.all_results)
            models_succeeded = sum(1 for m in multi_result.all_results if m.mape < 100)

            narrative = (
                f"Based on your historical data ({len(values)} months), {metric['label']} "
                f"is projected to {direction} over the next {periods} months. "
                f"The current average is {fmt_current} per month, and the forecast average "
                f"is {fmt_forecast} ({pct_change:+.1f}% change)."
                f"\n\nWe evaluated {models_tested} forecasting models and {models_succeeded} produced valid "
                f"predictions. The {best_model_name} model was selected because it achieved the lowest "
                f"prediction error ({best_mape:.1f}% MAPE) on the held-out test data."
            )

            if trend_str > 0.5:
                narrative += f"\n\nYour data shows a {trend_dir} trend (R²={trend_str:.2f}), "
                if trend_dir == 'increasing':
                    narrative += "meaning the metric has been consistently rising over time."
                elif trend_dir == 'decreasing':
                    narrative += "meaning the metric has been consistently declining."
                else:
                    narrative += "meaning the metric has been relatively flat."
            if seasonality:
                narrative += " Seasonal patterns were detected in your data — the model accounts for cyclical variations."
            if cv > 0.3:
                narrative += f" Note: your data has high variability (CV={cv:.2f}), which increases forecast uncertainty."

            agg_insights = multi_result.aggregated_insights or []
            if agg_insights:
                narrative += "\n\nKey insights from model analysis: " + "; ".join(agg_insights[:3]) + "."

            narrative += f"\n\nBusiness context: {metric['business_context']}"

            return {
                'metric': metric_key,
                'label': metric['label'],
                'unit': unit,
                'historical_months': months,
                'historical_values': [round(v, 2) for v in values.tolist()],
                'historical_dates': months,
                'forecast_months': future_months,
                'forecasts': all_models_info,
                'recommended_model': best_key,
                'confidence_interval': {
                    'low': _native_list(ci_low_vals),
                    'high': _native_list(ci_high_vals),
                },
                'summary': {
                    'current_avg': round(float(current_avg), 2),
                    'forecast_avg': round(float(forecast_avg), 2),
                    'pct_change': round(float(pct_change), 1),
                    'direction': direction,
                    'data_points': int(len(values)),
                    'models_evaluated': int(models_tested),
                    'models_succeeded': int(models_succeeded),
                    'best_mape': round(float(best_mape), 2),
                    'trend': str(trend_dir),
                    'trend_strength': round(float(trend_str), 3),
                    'seasonality': bool(seasonality),
                },
                'narrative': narrative,
                'consensus_forecast': _native_list(consensus),
                'data_characteristics': _native_dict(data_chars),
                'business_context': metric['business_context'],
                'hipaa_compliance': (
                    'All forecasting models run locally on your server. '
                    'No patient-level data is transmitted externally. '
                    f'{models_tested} models were evaluated using train/test split. '
                    'This approach is fully HIPAA-compliant as no PHI leaves your network.'
                ),
            }

        hw_forecast, hw_params = self._holt_winters(values, periods)
        lt_forecast, lt_params = self._linear_trend(values, periods)
        ma_forecast, ma_params = self._moving_average(values, periods)

        best_model = 'holt_winters'
        best_error = float('inf')
        if len(values) > 6:
            train = values[:-3]
            actual = values[-3:]
            for name, forecaster in [('holt_winters', self._holt_winters),
                                      ('linear_trend', self._linear_trend),
                                      ('moving_average', self._moving_average)]:
                pred, _ = forecaster(train, 3)
                err = np.mean(np.abs(pred - actual))
                if err < best_error:
                    best_error = err
                    best_model = name

        forecasts = {
            'holt_winters': hw_forecast,
            'linear_trend': lt_forecast,
            'moving_average': ma_forecast,
        }
        chosen = forecasts[best_model]
        current_avg = values[-3:].mean() if len(values) >= 3 else values.mean()
        forecast_avg = chosen.mean()
        pct_change = ((forecast_avg - current_avg) / current_avg * 100) if current_avg else 0
        direction = 'increase' if pct_change > 2 else 'decrease' if pct_change < -2 else 'remain stable'

        residual_std = values.std() * 0.3 if len(values) > 3 else values.std()
        ci_low = np.maximum(chosen - 1.96 * residual_std, 0)
        ci_high = chosen + 1.96 * residual_std

        unit = metric['unit']
        if unit == 'dollars':
            fmt_current = f"${current_avg:,.0f}"
            fmt_forecast = f"${forecast_avg:,.0f}"
        elif unit == 'percent':
            fmt_current = f"{current_avg:.1f}%"
            fmt_forecast = f"{forecast_avg:.1f}%"
        else:
            fmt_current = f"{current_avg:,.0f}"
            fmt_forecast = f"{forecast_avg:,.0f}"

        narrative = (
            f"Based on your historical data ({len(values)} months), {metric['label']} "
            f"is projected to {direction} over the next {periods} months. "
            f"The current average is {fmt_current} per month, and the forecast average "
            f"is {fmt_forecast} ({pct_change:+.1f}% change). "
            f"\n\nWe used the {MODEL_EXPLANATIONS[best_model]['name']} model because it had "
            f"the lowest prediction error on your recent data. "
            f"{MODEL_EXPLANATIONS[best_model]['plain_english']}"
            f"\n\nBusiness context: {metric['business_context']}"
        )

        return {
            'metric': metric_key,
            'label': metric['label'],
            'unit': unit,
            'historical_months': months,
            'historical_values': [round(v, 2) for v in values.tolist()],
            'historical_dates': months,
            'forecast_months': future_months,
            'forecasts': {
                'holt_winters': {
                    'values': [round(v, 2) for v in hw_forecast.tolist()],
                    'explanation': MODEL_EXPLANATIONS['holt_winters'],
                    'params': hw_params,
                },
                'linear_trend': {
                    'values': [round(v, 2) for v in lt_forecast.tolist()],
                    'explanation': MODEL_EXPLANATIONS['linear_trend'],
                    'params': lt_params,
                },
                'moving_average': {
                    'values': [round(v, 2) for v in ma_forecast.tolist()],
                    'explanation': MODEL_EXPLANATIONS['moving_average'],
                    'params': ma_params,
                },
            },
            'recommended_model': best_model,
            'confidence_interval': {
                'low': [round(v, 2) for v in ci_low.tolist()],
                'high': [round(v, 2) for v in ci_high.tolist()],
            },
            'summary': {
                'current_avg': round(current_avg, 2),
                'forecast_avg': round(forecast_avg, 2),
                'pct_change': round(pct_change, 1),
                'direction': direction,
                'data_points': len(values),
            },
            'narrative': narrative,
            'business_context': metric['business_context'],
            'hipaa_compliance': (
                'All forecasting models run locally on your server. '
                'No patient-level data is transmitted externally. '
                'Only aggregate statistics are used for predictions. '
                'This approach is fully HIPAA-compliant as no PHI leaves your network.'
            ),
        }

    def forecast_all(self, periods: int = 6) -> Dict[str, Any]:
        results = {}
        for key in FORECAST_METRICS:
            try:
                results[key] = self.forecast_metric(key, periods)
            except Exception as e:
                logger.warning('Forecast failed for %s: %s', key, e)
                results[key] = {'metric': key, 'label': FORECAST_METRICS[key]['label'], 'error': str(e)}

        return {
            'forecasts': results,
            'models_used': list(MODEL_EXPLANATIONS.keys()),
            'model_explanations': MODEL_EXPLANATIONS,
            'hipaa_statement': (
                'All forecasting is performed on-premise using open-source statistical models. '
                'No protected health information (PHI) is transmitted to any external service. '
                'Models used: Holt-Winters Exponential Smoothing, Linear Regression with '
                'Seasonal Adjustment, and Smoothed Moving Average. All are standard statistical '
                'methods that run entirely within your infrastructure.'
            ),
            'available_metrics': {k: v['label'] for k, v in FORECAST_METRICS.items()},
        }
