import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

logger = logging.getLogger('gpdm.forecast')

try:
    from gpdm_config import (
        FORECAST_HORIZON, FORECAST_TEST_RATIO, FORECAST_MIN_DATAPOINTS,
        HW_ALPHA, HW_BETA, HW_GAMMA, HW_MIN_SEASON,
        ES_ALPHA_MIN, ES_ALPHA_MAX, ES_ALPHA_STEPS,
        ARIMA_MAX_ORDER,
        SEASONALITY_AUTOCORR_THRESHOLD, SEASONALITY_LAG_OPTIONS,
        MC_SIMULATIONS, CONSENSUS_FAILED_WEIGHT, MAX_INSIGHTS,
    )
except ImportError:
    FORECAST_HORIZON = 6
    FORECAST_TEST_RATIO = 0.20
    FORECAST_MIN_DATAPOINTS = 6
    HW_ALPHA, HW_BETA, HW_GAMMA = 0.15, 0.08, 0.05
    HW_MIN_SEASON = 4
    ES_ALPHA_MIN, ES_ALPHA_MAX, ES_ALPHA_STEPS = 0.05, 0.95, 19
    ARIMA_MAX_ORDER = 3
    SEASONALITY_AUTOCORR_THRESHOLD = 0.25
    SEASONALITY_LAG_OPTIONS = [4, 6, 12]
    MC_SIMULATIONS = 1000
    CONSENSUS_FAILED_WEIGHT = 0.05
    MAX_INSIGHTS = 8

_DATE_FORMATS = ['%Y-%m-%d', '%Y-%m', '%Y/%m/%d', '%m/%d/%Y', '%Y%m%d', '%Y']

def _parse_date(date_str: str) -> datetime:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


@dataclass
class ModelResult:
    model_name: str
    predictions: List[float]
    actuals: List[float]
    mape: float
    rmse: float
    mae: float
    trend_direction: str
    trend_strength: float
    seasonality_detected: bool
    insights: List[str]
    forecast_values: List[float]
    forecast_dates: List[str]
    fit_time_ms: float
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ForecastResult:
    best_model: str
    best_mape: float
    all_results: List[ModelResult]
    aggregated_insights: List[str]
    consensus_forecast: List[float]
    consensus_trend: str
    confidence_band: Tuple[List[float], List[float]]
    data_characteristics: Dict
    metric_name: str = "value"
    forecast_dates: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'best_model': self.best_model,
            'best_mape': round(self.best_mape, 2),
            'all_results': [r.to_dict() for r in self.all_results],
            'aggregated_insights': self.aggregated_insights,
            'consensus_forecast': [round(v, 2) for v in self.consensus_forecast],
            'consensus_trend': self.consensus_trend,
            'confidence_band': (
                [round(v, 2) for v in self.confidence_band[0]],
                [round(v, 2) for v in self.confidence_band[1]]
            ),
            'data_characteristics': self.data_characteristics,
            'metric_name': self.metric_name,
            'forecast_dates': self.forecast_dates,
            'timestamp': self.timestamp
        }


class ForecastRunner:

    def __init__(self, forecast_horizon: int = FORECAST_HORIZON):
        self.forecast_horizon = forecast_horizon
        logger.info(f"Initialized ForecastRunner with horizon={forecast_horizon}")

    def run(self, dates: List[str], values: List[float],
            metric_name: str = 'value') -> ForecastResult:
        logger.info(f"Starting forecast run for {metric_name} with {len(values)} data points")

        if len(dates) != len(values):
            raise ValueError("dates and values must have same length")
        if len(values) < FORECAST_MIN_DATAPOINTS:
            raise ValueError(f"At least {FORECAST_MIN_DATAPOINTS} data points required")
        if any(np.isnan(v) or np.isinf(v) for v in values):
            raise ValueError("Values contain NaN or infinity")

        data_chars = self._detect_data_characteristics(values)
        logger.info(f"Data characteristics: {data_chars}")

        train_values, test_values = self._train_test_split(values, test_ratio=FORECAST_TEST_RATIO)

        all_results = []
        model_functions = [
            ('Linear Regression', self._model_linear_regression),
            ('Holt-Winters', self._model_holt_winters),
            ('ARIMA-like', self._model_arima_simple),
            ('Exponential Smoothing', self._model_exponential_smoothing),
            ('Prophet-style', self._model_prophet_style),
            ('Bayesian Bootstrap', self._model_bayesian_bootstrap),
            ('Monte Carlo', self._model_monte_carlo),
        ]

        for model_name, model_func in model_functions:
            try:
                start_time = time.time()
                result = model_func(train_values, test_values, values, dates)
                result.fit_time_ms = (time.time() - start_time) * 1000
                result.model_name = model_name
                all_results.append(result)
                logger.info(f"{model_name}: MAPE={result.mape:.2f}%, RMSE={result.rmse:.2f}")
            except Exception as e:
                logger.warning(f"{model_name} failed: {str(e)}")
                failed_result = ModelResult(
                    model_name=model_name,
                    predictions=[],
                    actuals=test_values,
                    mape=999.0,
                    rmse=999.0,
                    mae=999.0,
                    trend_direction='unknown',
                    trend_strength=0.0,
                    seasonality_detected=False,
                    insights=[],
                    forecast_values=[],
                    forecast_dates=[],
                    fit_time_ms=0,
                    success=False,
                    error_message=str(e)
                )
                all_results.append(failed_result)

        successful_results = [r for r in all_results if r.success]
        if not successful_results:
            raise RuntimeError("All models failed")

        best_result = min(successful_results, key=lambda r: r.mape)
        logger.info(f"Best model selected: {best_result.model_name} (MAPE={best_result.mape:.2f}%)")

        aggregated_insights = self._aggregate_insights(successful_results)

        consensus_forecast, lower_band, upper_band = self._consensus_forecast(successful_results)

        consensus_trend, _ = self._detect_trend(values)

        try:
            last_date = _parse_date(dates[-1])
        except:
            last_date = datetime.now()

        forecast_dates = [
            (last_date + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        result = ForecastResult(
            best_model=best_result.model_name,
            best_mape=best_result.mape,
            all_results=all_results,
            aggregated_insights=aggregated_insights,
            consensus_forecast=consensus_forecast,
            consensus_trend=consensus_trend,
            confidence_band=(lower_band, upper_band),
            data_characteristics=data_chars,
            metric_name=metric_name,
            forecast_dates=forecast_dates
        )

        logger.info(f"Forecast complete. Consensus: {consensus_forecast}")
        return result

    def _detect_data_characteristics(self, values: List[float]) -> Dict:
        values = np.array(values, dtype=float)

        trend_direction, trend_strength = self._detect_trend(values)

        seasonality = self._detect_seasonality(values)

        mean_val = np.mean(values)
        if mean_val != 0:
            cv = np.std(values) / np.abs(mean_val)
        else:
            cv = 0.0

        if len(values) > 1:
            pct_changes = np.diff(values) / np.abs(np.array(values[:-1]) + 1e-10)
            avg_growth = np.mean(pct_changes) * 100
        else:
            avg_growth = 0.0

        return {
            'data_points': len(values),
            'mean': round(float(np.mean(values)), 2),
            'std': round(float(np.std(values)), 2),
            'min': round(float(np.min(values)), 2),
            'max': round(float(np.max(values)), 2),
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 3),
            'seasonality_detected': seasonality,
            'coefficient_of_variation': round(cv, 3),
            'avg_growth_pct': round(avg_growth, 2),
        }

    def _train_test_split(self, values: List[float],
                         test_ratio: float = FORECAST_TEST_RATIO) -> Tuple[List[float], List[float]]:
        split_idx = max(int(len(values) * (1 - test_ratio)), 5)
        train = values[:split_idx]
        test = values[split_idx:]
        return train, test

    def _evaluate(self, actuals: List[float], predictions: List[float]) -> Dict:
        actuals = np.array(actuals, dtype=float)
        predictions = np.array(predictions, dtype=float)

        mask = actuals != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = 0.0

        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

        mae = np.mean(np.abs(actuals - predictions))

        return {'mape': float(mape), 'rmse': float(rmse), 'mae': float(mae)}

    def _detect_trend(self, values: List[float]) -> Tuple[str, float]:
        values = np.array(values, dtype=float)
        x = np.arange(len(values))

        n = len(values)
        m = (n * np.sum(x * values) - np.sum(x) * np.sum(values)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)

        mean_val = np.mean(values)
        ss_tot = np.sum((values - mean_val) ** 2)
        fitted = m * x + (mean_val - m * np.mean(x))
        ss_res = np.sum((values - fitted) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        trend_strength = np.clip(r_squared, 0, 1)

        if abs(m) < 1e-10:
            direction = 'stable'
        elif m > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'

        return direction, float(trend_strength)

    def _detect_seasonality(self, values: List[float]) -> bool:
        if len(values) < 8:
            return False

        values = np.array(values, dtype=float)
        mean_val = np.mean(values)
        c0 = np.sum((values - mean_val) ** 2) / len(values)

        lag = min(4, len(values) // 2)
        if c0 < 1e-10:
            return False

        c_lag = np.sum((values[:-lag] - mean_val) * (values[lag:] - mean_val)) / len(values)
        acf = c_lag / c0

        return acf > SEASONALITY_AUTOCORR_THRESHOLD

    def _aggregate_insights(self, results: List[ModelResult]) -> List[str]:
        insights_dict = {}

        for result in results:
            for insight in result.insights:
                if insight not in insights_dict:
                    insights_dict[insight] = 0
                insights_dict[insight] += 1

        sorted_insights = sorted(insights_dict.items(), key=lambda x: x[1], reverse=True)
        insights = [insight for insight, count in sorted_insights[:MAX_INSIGHTS]]

        return insights

    def _consensus_forecast(self, results: List[ModelResult]) -> Tuple[List[float], List[float], List[float]]:
        if not results:
            return [], [], []

        weights = []
        forecasts = []

        for result in results:
            if result.mape > 0 and result.mape < 100:
                weight = 1.0 / result.mape
            else:
                weight = CONSENSUS_FAILED_WEIGHT
            weights.append(weight)
            forecasts.append(result.forecast_values)

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        forecasts = np.array(forecasts, dtype=float)
        consensus = np.average(forecasts, axis=0, weights=weights)

        lower = np.min(forecasts, axis=0)
        upper = np.max(forecasts, axis=0)

        return list(consensus), list(lower), list(upper)


    def _model_linear_regression(self, train_values: List[float],
                                test_values: List[float],
                                all_values: List[float],
                                dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        x_train = np.arange(len(train))
        n = len(train)
        m = (n * np.sum(x_train * train) - np.sum(x_train) * np.sum(train)) / (n * np.sum(x_train**2) - np.sum(x_train)**2 + 1e-10)
        b = (np.sum(train) - m * np.sum(x_train)) / n

        x_test = np.arange(len(train), len(train) + len(test))
        predictions = m * x_test + b

        metrics = self._evaluate(test, predictions)

        trend_dir, trend_str = self._detect_trend(train)

        x_future = np.arange(len(all_vals), len(all_vals) + self.forecast_horizon)
        forecast = m * x_future + b

        growth_rate = m / (np.mean(train) + 1e-10) * 100
        insights = [
            f"Linear trend: {trend_dir} at {abs(growth_rate):.2f}% per period (LR, MAPE: {metrics['mape']:.1f}%)"
        ]

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        return ModelResult(
            model_name='Linear Regression',
            predictions=list(predictions),
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=False,
            insights=insights,
            forecast_values=list(forecast),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_holt_winters(self, train_values: List[float],
                           test_values: List[float],
                           all_values: List[float],
                           dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        alpha = HW_ALPHA
        beta = HW_BETA
        gamma = HW_GAMMA
        season_length = max(HW_MIN_SEASON, len(train) // 4)

        if len(train) < 2 * season_length:
            season_length = max(2, len(train) // 3)

        level = np.mean(train[:season_length])
        trend = (np.mean(train[season_length:2*season_length]) - np.mean(train[:season_length])) / season_length
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            seasonal[i] = np.mean(train[i::season_length]) - level

        predictions = []
        for i, val in enumerate(train[season_length:], start=season_length):
            season_idx = i % season_length
            pred = level + trend + seasonal[season_idx]
            predictions.append(pred)

            error = val - pred
            level = alpha * (val - seasonal[season_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - (level + trend)) + (1 - beta) * trend
            seasonal[season_idx] = gamma * (val - level) + (1 - gamma) * seasonal[season_idx]

        test_preds = []
        for i in range(len(test)):
            season_idx = (len(train) + season_length + i) % season_length
            pred = level + trend + seasonal[season_idx]
            test_preds.append(pred)
            if i < len(test):
                error = test[i] - pred
                level = alpha * (test[i] - seasonal[season_idx]) + (1 - alpha) * (level + trend)
                trend = beta * (level - (level + trend)) + (1 - beta) * trend
                seasonal[season_idx] = gamma * (test[i] - level) + (1 - gamma) * seasonal[season_idx]

        metrics = self._evaluate(test, test_preds)

        trend_dir, trend_str = self._detect_trend(train)
        seasonality = self._detect_seasonality(train)

        forecast = []
        for i in range(self.forecast_horizon):
            season_idx = (len(all_vals) + season_length + i) % season_length
            pred = level + (i + 1) * trend + seasonal[season_idx]
            forecast.append(pred)

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        insights = [
            f"Holt-Winters decomposition: Level={level:.2f}, Trend={trend:.2f}, Season={season_length} periods (MAPE: {metrics['mape']:.1f}%)"
        ]
        if seasonality:
            seasonal_strength = np.std(seasonal) / (np.abs(level) + 1e-10)
            insights.append(f"Seasonality strength: {seasonal_strength:.1%} (HW)")

        return ModelResult(
            model_name='Holt-Winters',
            predictions=test_preds,
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=seasonality,
            insights=insights,
            forecast_values=list(np.clip(forecast, 0, None)),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_arima_simple(self, train_values: List[float],
                           test_values: List[float],
                           all_values: List[float],
                           dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        best_p = 1
        best_rmse = float('inf')

        for p in [1, 2, 3]:
            if len(train) < p + 2:
                continue

            try:
                preds = []
                for i in range(p, len(train)):
                    pred = np.sum([train[i-j] for j in range(1, p+1)]) / p
                    preds.append(pred)

                rmse = np.sqrt(np.mean((train[p:] - preds) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_p = p
            except:
                pass

        ar_coeffs = np.zeros(best_p)
        for i in range(best_p):
            if len(train) > i:
                ar_coeffs[i] = np.mean(train[i:]) - np.mean(train)
        ar_coeffs = ar_coeffs / (np.sum(np.abs(ar_coeffs)) + 1e-10)

        test_preds = []
        history = list(train[-best_p:])
        for _ in range(len(test)):
            pred = np.sum([history[-j] * ar_coeffs[j-1] for j in range(1, min(best_p+1, len(history)+1))])
            test_preds.append(pred)
            history.append(pred)

        metrics = self._evaluate(test, test_preds)
        trend_dir, trend_str = self._detect_trend(train)

        forecast = []
        history = list(all_vals[-best_p:])
        for _ in range(self.forecast_horizon):
            pred = np.sum([history[-j] * ar_coeffs[j-1] for j in range(1, min(best_p+1, len(history)+1))])
            forecast.append(pred)
            history.append(pred)

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        insights = [
            f"AR({best_p}) model: autoregressive coefficients selected (MAPE: {metrics['mape']:.1f}%)"
        ]

        return ModelResult(
            model_name='ARIMA-like',
            predictions=test_preds,
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=False,
            insights=insights,
            forecast_values=list(np.clip(forecast, 0, None)),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_exponential_smoothing(self, train_values: List[float],
                                    test_values: List[float],
                                    all_values: List[float],
                                    dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        best_alpha = 0.3
        best_rmse = float('inf')

        for alpha in np.linspace(ES_ALPHA_MIN, ES_ALPHA_MAX, ES_ALPHA_STEPS):
            forecast_vals = []
            level = train[0]
            for val in train[1:]:
                forecast_vals.append(level)
                level = alpha * val + (1 - alpha) * level

            rmse = np.sqrt(np.mean((train[1:] - forecast_vals) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        level = train[0]
        for val in train:
            level = best_alpha * val + (1 - best_alpha) * level

        test_preds = []
        for val in test:
            test_preds.append(level)
            level = best_alpha * val + (1 - best_alpha) * level

        metrics = self._evaluate(test, test_preds)
        trend_dir, trend_str = self._detect_trend(train)

        forecast = []
        for _ in range(self.forecast_horizon):
            forecast.append(level)

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        insights = [
            f"Exponential smoothing: alpha={best_alpha:.2f} (SES, MAPE: {metrics['mape']:.1f}%)"
        ]

        return ModelResult(
            model_name='Exponential Smoothing',
            predictions=test_preds,
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=False,
            insights=insights,
            forecast_values=list(np.clip(forecast, 0, None)),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_prophet_style(self, train_values: List[float],
                            test_values: List[float],
                            all_values: List[float],
                            dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        x = np.arange(len(train))
        n = len(train)
        m = (n * np.sum(x * train) - np.sum(x) * np.sum(train)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)
        b = (np.sum(train) - m * np.sum(x)) / n
        trend = m * x + b

        season_length = max(HW_MIN_SEASON, len(train) // 4)
        if season_length > len(train) // 2:
            season_length = max(2, len(train) // 3)

        detrended = train - trend
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            season_indices = np.arange(i, len(detrended), season_length)
            if len(season_indices) > 0:
                seasonal[i] = np.mean(detrended[season_indices])

        test_preds = []
        for i in range(len(test)):
            trend_val = m * (len(train) + i) + b
            season_idx = (len(train) + i) % season_length
            pred = trend_val + seasonal[season_idx]
            test_preds.append(pred)

        metrics = self._evaluate(test, test_preds)
        trend_dir, trend_str = self._detect_trend(train)
        seasonality = self._detect_seasonality(train)

        forecast = []
        for i in range(self.forecast_horizon):
            trend_val = m * (len(all_vals) + i) + b
            season_idx = (len(all_vals) + i) % season_length
            pred = trend_val + seasonal[season_idx]
            forecast.append(pred)

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        insights = [
            f"Prophet-style decomposition: trend + seasonal({season_length}p) (PS, MAPE: {metrics['mape']:.1f}%)"
        ]

        return ModelResult(
            model_name='Prophet-style',
            predictions=test_preds,
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=seasonality,
            insights=insights,
            forecast_values=list(np.clip(forecast, 0, None)),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_bayesian_bootstrap(self, train_values: List[float],
                                 test_values: List[float],
                                 all_values: List[float],
                                 dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        growth_rates = np.diff(train) / (np.abs(train[:-1]) + 1e-10)
        growth_rates = growth_rates[~np.isnan(growth_rates)]
        growth_rates = growth_rates[~np.isinf(growth_rates)]

        if len(growth_rates) == 0:
            growth_rates = np.array([0.0])

        mean_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates)

        np.random.seed(42)
        test_preds = []
        current = train[-1]

        for _ in range(len(test)):
            sampled_growth = np.random.normal(mean_growth, std_growth + 1e-10)
            pred = current * (1 + sampled_growth)
            pred = max(0, pred)
            test_preds.append(pred)
            current = test[len(test_preds) - 1] if len(test_preds) <= len(test) else pred

        metrics = self._evaluate(test, test_preds)
        trend_dir, trend_str = self._detect_trend(train)

        forecast = []
        current = all_vals[-1]
        for _ in range(self.forecast_horizon):
            sampled_growth = np.random.normal(mean_growth, std_growth + 1e-10)
            pred = current * (1 + sampled_growth)
            pred = max(0, pred)
            forecast.append(pred)
            current = pred

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        insights = [
            f"Bayesian bootstrap: mean growth={mean_growth:.3f}, std={std_growth:.3f} (BB, MAPE: {metrics['mape']:.1f}%)"
        ]

        return ModelResult(
            model_name='Bayesian Bootstrap',
            predictions=test_preds,
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=False,
            insights=insights,
            forecast_values=list(np.clip(forecast, 0, None)),
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )

    def _model_monte_carlo(self, train_values: List[float],
                          test_values: List[float],
                          all_values: List[float],
                          dates: List[str]) -> ModelResult:
        train = np.array(train_values, dtype=float)
        test = np.array(test_values, dtype=float)
        all_vals = np.array(all_values, dtype=float)

        changes = np.diff(train)
        mean_change = np.mean(changes)
        std_change = np.std(changes)

        np.random.seed(42)
        n_simulations = 1000
        test_predictions = np.zeros((n_simulations, len(test)))

        for sim in range(n_simulations):
            current = train[-1]
            for t in range(len(test)):
                change = np.random.normal(mean_change, std_change + 1e-10)
                current = max(0, current + change)
                test_predictions[sim, t] = current

        test_preds = np.median(test_predictions, axis=0)

        metrics = self._evaluate(test, test_preds)
        trend_dir, trend_str = self._detect_trend(train)

        forecast_predictions = np.zeros((n_simulations, self.forecast_horizon))
        for sim in range(n_simulations):
            current = all_vals[-1]
            for t in range(self.forecast_horizon):
                change = np.random.normal(mean_change, std_change + 1e-10)
                current = max(0, current + change)
                forecast_predictions[sim, t] = current

        forecast = list(np.median(forecast_predictions, axis=0))

        forecast_dates = [
            (_parse_date(dates[-1]) + timedelta(days=30*(i+1))).strftime('%Y-%m-%d')
            for i in range(self.forecast_horizon)
        ]

        percentile_5 = np.percentile(forecast_predictions, 5, axis=0)
        percentile_95 = np.percentile(forecast_predictions, 95, axis=0)
        ci_width = np.mean(percentile_95 - percentile_5)

        insights = [
            f"Monte Carlo 1000 paths: mean change={mean_change:.2f}, std={std_change:.2f} (MC, MAPE: {metrics['mape']:.1f}%)",
            f"90% confidence interval width: {ci_width:.2f}"
        ]

        return ModelResult(
            model_name='Monte Carlo',
            predictions=list(test_preds),
            actuals=list(test),
            mape=metrics['mape'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            trend_direction=trend_dir,
            trend_strength=trend_str,
            seasonality_detected=False,
            insights=insights,
            forecast_values=forecast,
            forecast_dates=forecast_dates,
            fit_time_ms=0
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dates = [
        '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01',
        '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01',
        '2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01', '2024-03-01',
        '2024-04-01', '2024-05-01', '2024-06-01'
    ]

    values = [
        100, 105, 110, 108, 115, 120, 125, 123, 130, 135,
        138, 145, 148, 152, 155, 158, 162, 165
    ]

    runner = ForecastRunner(forecast_horizon=6)
    result = runner.run(dates, values, metric_name='Revenue')

    print("\n" + "="*60)
    print(f"Best Model: {result.best_model} (MAPE: {result.best_mape:.2f}%)")
    print("="*60)
    print("\nAggregated Insights:")
    for i, insight in enumerate(result.aggregated_insights, 1):
        print(f"  {i}. {insight}")

    print("\nConsensus Forecast (next 6 months):")
    for date, value in zip(result.forecast_dates, result.consensus_forecast):
        print(f"  {date}: {value:.2f}")

    print("\nConfidence Bands:")
    print(f"  Lower: {[f'{v:.2f}' for v in result.confidence_band[0][:3]]}")
    print(f"  Upper: {[f'{v:.2f}' for v in result.confidence_band[1][:3]]}")

    print("\nData Characteristics:")
    for key, value in result.data_characteristics.items():
        print(f"  {key}: {value}")
