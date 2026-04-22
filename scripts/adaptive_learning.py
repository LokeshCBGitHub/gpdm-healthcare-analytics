from __future__ import annotations
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BASE = (os.environ.get('GPDM_BASE_DIR', '') or
         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OnlineGBM:

    def __init__(self, task='classification', learning_rate=0.1,
                 max_depth=4, min_samples_leaf=20, subsample=0.8):
        self.task = task
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_samples_leaf
        self.subsample = subsample
        self.trees: list = []
        self.init_pred = 0.0
        self.n_seen = 0
        self.window_metrics: List[Dict] = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _predict_raw(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    n_trees: int = 20, eval_frac: float = 0.1) -> Dict:
        n = X.shape[0]
        if n == 0:
            return {'status': 'no_data'}

        if not self.trees:
            if self.task == 'classification':
                pos_rate = np.clip(y.mean(), 1e-7, 1 - 1e-7)
                self.init_pred = np.log(pos_rate / (1 - pos_rate))
            else:
                self.init_pred = float(np.mean(y))

        n_eval = max(10, int(n * eval_frac))
        idx = np.random.permutation(n)
        eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        pre_score = self._evaluate(X_eval, y_eval)

        F_train = self._predict_raw(X_train)
        from ml_pretrain import DecisionStump

        for _ in range(n_trees):
            if self.task == 'classification':
                p = self._sigmoid(F_train)
                residuals = y_train - p
            else:
                residuals = y_train - F_train

            n_sub = int(len(train_idx) * self.subsample)
            sub = np.random.choice(len(train_idx), n_sub, replace=False)

            tree = DecisionStump()
            tree.fit(X_train[sub], residuals[sub],
                     max_depth=self.max_depth,
                     min_samples_leaf=self.min_leaf)
            self.trees.append(tree)
            F_train += self.lr * tree.predict(X_train)

        post_score = self._evaluate(X_eval, y_eval)
        self.n_seen += n

        metric = {
            'batch_size': n,
            'n_trees_total': len(self.trees),
            'n_trees_added': n_trees,
            'total_seen': self.n_seen,
            'pre_score': pre_score,
            'post_score': post_score,
            'improvement': post_score.get('primary', 0) - pre_score.get('primary', 0),
            'timestamp': time.time(),
        }
        self.window_metrics.append(metric)
        return metric

    def _evaluate(self, X, y) -> Dict:
        if len(X) == 0:
            return {'primary': 0}
        raw = self._predict_raw(X)
        if self.task == 'classification':
            p = self._sigmoid(raw)
            acc = float(np.mean((p >= 0.5) == y))
            pos = p[y == 1]
            neg = p[y == 0]
            if len(pos) > 0 and len(neg) > 0:
                n_s = min(500, len(pos), len(neg))
                auc = float(np.mean(
                    np.random.choice(pos, n_s) > np.random.choice(neg, n_s)))
            else:
                auc = 0.5
            return {'primary': auc, 'accuracy': acc, 'auc': auc}
        else:
            mse = float(np.mean((y - raw) ** 2))
            mae = float(np.mean(np.abs(y - raw)))
            return {'primary': -mae, 'mse': mse, 'mae': mae}

    def predict_proba(self, X):
        return self._sigmoid(self._predict_raw(X))

    def predict(self, X):
        if self.task == 'classification':
            return (self.predict_proba(X) >= 0.5).astype(int)
        return self._predict_raw(X)

    def get_trend(self, window: int = 5) -> Dict:
        if len(self.window_metrics) < 2:
            return {'trend': 'insufficient_data'}
        recent = self.window_metrics[-window:]
        scores = [m['post_score'].get('primary', 0) for m in recent]
        if len(scores) < 2:
            return {'trend': 'insufficient_data'}
        slope = np.polyfit(range(len(scores)), scores, 1)[0]
        return {
            'trend': 'improving' if slope > 0.001 else ('degrading' if slope < -0.001 else 'stable'),
            'slope': float(slope),
            'recent_scores': scores,
            'n_batches': len(self.window_metrics),
        }


FEEDBACK_TABLE = '_gpdm_feedback_log'
OUTCOME_TABLE = '_gpdm_outcome_tracking'


def ensure_feedback_tables(conn: sqlite3.Connection):
    conn.execute(f'''CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        member_id TEXT,
        model_name TEXT,
        prediction REAL,
        actual_outcome REAL,
        feedback_type TEXT,       -- 'outcome', 'user_correction', 'clinician_override'
        feedback_source TEXT,     -- 'system', 'care_manager', 'claims_data'
        notes TEXT
    )''')
    conn.execute(f'''CREATE TABLE IF NOT EXISTS {OUTCOME_TABLE} (
        member_id TEXT,
        intervention_type TEXT,
        intervention_date TEXT,
        outcome_type TEXT,        -- 'readmitted_30d', 'churned_90d', 'cost_delta'
        outcome_value REAL,
        measured_date TEXT,
        PRIMARY KEY (member_id, intervention_type, outcome_type)
    )''')
    conn.commit()


class FeedbackLoop:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pending_feedback: List[Dict] = []

    def log_prediction(self, member_id: str, model_name: str,
                       prediction: float, intervention: str = None):
        self.pending_feedback.append({
            'member_id': member_id,
            'model_name': model_name,
            'prediction': prediction,
            'intervention': intervention,
            'timestamp': time.time(),
        })

    def log_outcome(self, member_id: str, outcome_type: str,
                    outcome_value: float, intervention_type: str = None):
        conn = sqlite3.connect(self.db_path)
        ensure_feedback_tables(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO {OUTCOME_TABLE} "
            "VALUES (?, ?, date('now'), ?, ?, date('now'))",
            (member_id, intervention_type or '', outcome_type,
             outcome_value, ))
        conn.commit()
        conn.close()

    def log_batch_outcomes(self, outcomes: List[Dict]):
        conn = sqlite3.connect(self.db_path)
        ensure_feedback_tables(conn)
        for o in outcomes:
            conn.execute(
                f"INSERT OR REPLACE INTO {OUTCOME_TABLE} "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (o['member_id'], o.get('intervention_type', ''),
                 o.get('intervention_date', ''),
                 o['outcome_type'], o['outcome_value'],
                 o.get('measured_date', time.strftime('%Y-%m-%d'))))
        conn.commit()
        conn.close()
        logger.info("Logged %d outcomes", len(outcomes))

    def collect_training_pairs(self, model_name: str,
                               outcome_type: str) -> Tuple[List[str], np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        ensure_feedback_tables(conn)
        rows = conn.execute(f"""
            SELECT o.member_id, o.outcome_value
            FROM {OUTCOME_TABLE} o
            WHERE o.outcome_type = ?
            ORDER BY o.measured_date DESC
        """, (outcome_type,)).fetchall()
        conn.close()

        if not rows:
            return [], np.array([])

        member_ids = [r[0] for r in rows]
        outcomes = np.array([float(r[1]) for r in rows])
        return member_ids, outcomes

    def get_feedback_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        ensure_feedback_tables(conn)
        stats = {}
        for row in conn.execute(
            f"SELECT outcome_type, COUNT(*), AVG(outcome_value) "
            f"FROM {OUTCOME_TABLE} GROUP BY outcome_type"):
            stats[row[0]] = {'count': row[1], 'avg_outcome': row[2]}
        conn.close()
        return stats


class DriftDetector:

    def __init__(self, sensitivity: float = 50.0, min_samples: int = 100):
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.detectors: Dict[str, _PageHinkley] = {}

    def update(self, metric_name: str, value: float) -> Dict:
        if metric_name not in self.detectors:
            self.detectors[metric_name] = _PageHinkley(self.sensitivity)
        det = self.detectors[metric_name]
        det.add(value)
        return {
            'metric': metric_name,
            'drift_detected': det.drift_detected,
            'n_seen': det.n,
            'cumulative_sum': det.sum_,
            'running_mean': det.mean_ if det.n > 0 else None,
        }

    def check_feature_drift(self, reference: np.ndarray,
                             current: np.ndarray,
                             feature_names: List[str] = None) -> Dict:
        if reference.shape[1] != current.shape[1]:
            return {'error': 'dimension_mismatch'}

        n_features = reference.shape[1]
        psi_scores = {}
        drifted_features = []

        for j in range(n_features):
            fname = feature_names[j] if feature_names and j < len(feature_names) else f'f{j}'
            psi = _compute_psi(reference[:, j], current[:, j])
            psi_scores[fname] = round(psi, 4)
            if psi > 0.2:
                drifted_features.append(fname)

        return {
            'psi_scores': psi_scores,
            'drifted_features': drifted_features,
            'n_drifted': len(drifted_features),
            'max_psi': max(psi_scores.values()) if psi_scores else 0,
            'recommendation': (
                'RETRAIN_REQUIRED' if len(drifted_features) > n_features * 0.3
                else 'MONITOR' if drifted_features
                else 'STABLE'
            ),
        }

    def check_prediction_drift(self, old_preds: np.ndarray,
                                new_preds: np.ndarray) -> Dict:
        psi = _compute_psi(old_preds, new_preds)
        ks_stat = _ks_statistic(old_preds, new_preds)
        return {
            'psi': round(psi, 4),
            'ks_statistic': round(ks_stat, 4),
            'drift': psi > 0.2 or ks_stat > 0.1,
            'severity': 'HIGH' if psi > 0.5 else ('MEDIUM' if psi > 0.2 else 'LOW'),
        }

    def get_all_status(self) -> Dict[str, Dict]:
        return {
            name: {
                'drift_detected': det.drift_detected,
                'n_seen': det.n,
                'mean': det.mean_ if det.n > 0 else None,
            }
            for name, det in self.detectors.items()
        }


class _PageHinkley:
    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha
        self.n = 0
        self.sum_ = 0.0
        self.mean_ = 0.0
        self.max_sum = 0.0
        self.drift_detected = False

    def add(self, x: float):
        self.n += 1
        self.mean_ += (x - self.mean_) / self.n
        self.sum_ += x - self.mean_ - self.alpha
        self.max_sum = max(self.max_sum, self.sum_)
        if self.max_sum - self.sum_ > self.threshold:
            self.drift_detected = True


def _compute_psi(expected: np.ndarray, actual: np.ndarray,
                 n_bins: int = 10) -> float:
    eps = 1e-4
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float) + eps
    act_counts = np.histogram(actual, bins=breakpoints)[0].astype(float) + eps

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), combined, side='right') / len(a)
    cdf_b = np.searchsorted(np.sort(b), combined, side='right') / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


REGISTRY_TABLE = '_gpdm_model_registry'


class ModelRegistry:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(f'''CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
            model_name TEXT,
            version INTEGER,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
            n_training_rows INTEGER,
            metric_name TEXT,
            metric_value REAL,
            is_active INTEGER DEFAULT 0,
            config_hash TEXT,
            notes TEXT,
            PRIMARY KEY (model_name, version)
        )''')
        conn.commit()
        conn.close()

    def register(self, model_name: str, metrics: Dict[str, float],
                 n_rows: int, config: Dict = None, notes: str = '') -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            f"SELECT COALESCE(MAX(version), 0) FROM {REGISTRY_TABLE} "
            f"WHERE model_name = ?", (model_name,))
        next_version = cur.fetchone()[0] + 1
        config_hash = hashlib.md5(
            json.dumps(config or {}, sort_keys=True).encode()).hexdigest()[:12]

        for metric_name, metric_value in metrics.items():
            conn.execute(
                f"INSERT OR REPLACE INTO {REGISTRY_TABLE} "
                "(model_name, version, n_training_rows, metric_name, "
                "metric_value, config_hash, notes) VALUES (?,?,?,?,?,?,?)",
                (model_name, next_version, n_rows, metric_name,
                 metric_value, config_hash, notes))

        conn.commit()
        conn.close()
        logger.info("Registered %s v%d: %s", model_name, next_version, metrics)
        return next_version

    def promote(self, model_name: str, version: int):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"UPDATE {REGISTRY_TABLE} SET is_active = 0 "
            f"WHERE model_name = ?", (model_name,))
        conn.execute(
            f"UPDATE {REGISTRY_TABLE} SET is_active = 1 "
            f"WHERE model_name = ? AND version = ?", (model_name, version))
        conn.commit()
        conn.close()

    def compare_versions(self, model_name: str,
                         v1: int, v2: int) -> Dict:
        conn = sqlite3.connect(self.db_path)
        def _get(v):
            rows = conn.execute(
                f"SELECT metric_name, metric_value FROM {REGISTRY_TABLE} "
                f"WHERE model_name = ? AND version = ?",
                (model_name, v)).fetchall()
            return {r[0]: r[1] for r in rows}
        m1, m2 = _get(v1), _get(v2)
        conn.close()
        comparison = {}
        for k in set(list(m1.keys()) + list(m2.keys())):
            v1_val = m1.get(k, 0)
            v2_val = m2.get(k, 0)
            comparison[k] = {
                f'v{v1}': v1_val, f'v{v2}': v2_val,
                'delta': v2_val - v1_val,
                'improved': v2_val > v1_val,
            }
        return comparison

    def get_active_version(self, model_name: str) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            f"SELECT version FROM {REGISTRY_TABLE} "
            f"WHERE model_name = ? AND is_active = 1 LIMIT 1",
            (model_name,)).fetchone()
        conn.close()
        return row[0] if row else None

    def get_history(self, model_name: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT version, trained_at, n_training_rows, metric_name, "
            f"metric_value, is_active FROM {REGISTRY_TABLE} "
            f"WHERE model_name = ? ORDER BY version DESC",
            (model_name,)).fetchall()
        conn.close()
        by_version = defaultdict(dict)
        for version, trained, n_rows, mname, mval, active in rows:
            by_version[version].update({
                'version': version, 'trained_at': trained,
                'n_rows': n_rows, 'is_active': bool(active),
                mname: mval,
            })
        return list(by_version.values())


class BenchmarkLearner:

    def __init__(self):
        self.baselines: Dict[str, Dict] = {}

    def learn_baselines(self, conn: sqlite3.Connection) -> Dict:
        baselines = {}

        try:
            rows = conn.execute("""
                SELECT month,
                       SUM(denied_count)*100.0/NULLIF(SUM(claim_count),0) as denial_rate
                FROM _gpdm_summary_monthly
                GROUP BY month
                HAVING denial_rate IS NOT NULL
                ORDER BY month
            """).fetchall()
            if rows:
                vals = np.array([float(r[1]) for r in rows])
                baselines['denial_rate'] = self._compute_baseline(vals, 'denial_rate')
        except Exception:
            pass

        try:
            rows = conn.execute("""
                SELECT month, total_paid FROM _gpdm_summary_monthly
                WHERE total_paid IS NOT NULL
            """).fetchall()
            if rows:
                vals = np.array([float(r[1]) for r in rows])
                baselines['monthly_paid'] = self._compute_baseline(vals, 'monthly_paid')
        except Exception:
            pass

        try:
            rows = conn.execute("""
                SELECT month, claim_count FROM _gpdm_summary_monthly
                WHERE claim_count IS NOT NULL
            """).fetchall()
            if rows:
                vals = np.array([float(r[1]) for r in rows])
                baselines['monthly_volume'] = self._compute_baseline(vals, 'monthly_volume')
        except Exception:
            pass

        try:
            rows = conn.execute("""
                SELECT region, denial_rate, avg_paid
                FROM _gpdm_summary_region
            """).fetchall()
            for r in rows:
                region = r[0]
                baselines[f'denial_rate_{region}'] = {
                    'value': float(r[1] or 0),
                    'type': 'point_estimate',
                }
                baselines[f'avg_paid_{region}'] = {
                    'value': float(r[2] or 0),
                    'type': 'point_estimate',
                }
        except Exception:
            pass

        try:
            rows = conn.execute("""
                SELECT churn_risk, readmission_risk FROM _gpdm_member_risk
                LIMIT 50000
            """).fetchall()
            if rows:
                churn = np.array([float(r[0]) for r in rows])
                readmit = np.array([float(r[1]) for r in rows])
                baselines['churn_risk'] = self._compute_baseline(churn, 'churn_risk')
                baselines['readmission_risk'] = self._compute_baseline(readmit, 'readmission_risk')
        except Exception:
            pass

        self.baselines = baselines
        logger.info("Learned %d KPI baselines from historical data", len(baselines))
        return baselines

    def _compute_baseline(self, values: np.ndarray, name: str) -> Dict:
        return {
            'name': name,
            'type': 'distribution',
            'n': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'p5': float(np.percentile(values, 5)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95)),
            'alert_low': float(np.percentile(values, 5)),
            'alert_high': float(np.percentile(values, 95)),
            'trend_slope': float(np.polyfit(range(len(values)), values, 1)[0])
            if len(values) >= 3 else 0.0,
        }

    def evaluate_current(self, kpi_name: str, current_value: float) -> Dict:
        bl = self.baselines.get(kpi_name)
        if not bl or bl.get('type') != 'distribution':
            return {'status': 'no_baseline', 'kpi': kpi_name}

        z_score = (current_value - bl['mean']) / (bl['std'] + 1e-10)
        percentile_rank = float(np.mean(
            np.array([bl['p5'], bl['p25'], bl['p50'], bl['p75'], bl['p95']])
            <= current_value) * 100)

        if current_value < bl['alert_low']:
            status = 'BELOW_NORMAL'
        elif current_value > bl['alert_high']:
            status = 'ABOVE_NORMAL'
        elif abs(z_score) > 1.5:
            status = 'WARNING'
        else:
            status = 'NORMAL'

        return {
            'kpi': kpi_name,
            'current': current_value,
            'baseline_mean': bl['mean'],
            'z_score': round(z_score, 2),
            'percentile_rank': round(percentile_rank, 1),
            'status': status,
            'trend': 'increasing' if bl['trend_slope'] > 0 else 'decreasing',
            'alert_range': [bl['alert_low'], bl['alert_high']],
        }


class AdaptiveLearningEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models: Dict[str, OnlineGBM] = {}
        self.feedback = FeedbackLoop(db_path)
        self.drift = DriftDetector()
        self.registry = ModelRegistry(db_path)
        self.benchmarks = BenchmarkLearner()
        self._reference_features: Optional[np.ndarray] = None

    def initialize(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        self.benchmarks.learn_baselines(conn)
        ensure_feedback_tables(conn)
        conn.close()
        logger.info("Adaptive engine initialized with %d baselines",
                    len(self.benchmarks.baselines))

    def on_new_data(self, features: np.ndarray, labels: Dict[str, np.ndarray],
                    feature_names: List[str] = None) -> Dict:
        results = {}

        if self._reference_features is not None:
            drift_result = self.drift.check_feature_drift(
                self._reference_features, features, feature_names)
            results['feature_drift'] = drift_result
            if drift_result['recommendation'] == 'RETRAIN_REQUIRED':
                logger.warning("Feature drift detected! %d features shifted.",
                              drift_result['n_drifted'])
        self._reference_features = features

        for model_name, y in labels.items():
            if model_name not in self.models:
                task = 'classification' if set(np.unique(y)).issubset({0., 1.}) else 'regression'
                self.models[model_name] = OnlineGBM(task=task)

            model = self.models[model_name]
            metric = model.partial_fit(features, y, n_trees=15)
            results[f'model_{model_name}'] = metric

            trend = model.get_trend()
            results[f'trend_{model_name}'] = trend

            post = metric.get('post_score')
            if post and isinstance(post, dict):
                self.registry.register(
                    model_name,
                    {k: v for k, v in post.items()
                     if isinstance(v, (int, float))},
                    n_rows=metric.get('batch_size', 0),
                    notes=f'online_update_batch_{model.n_seen}')

            old_preds = model.predict_proba(features) if model.task == 'classification' \
                        else model.predict(features)
            drift_status = self.drift.update(
                f'{model_name}_mean_pred', float(np.mean(old_preds)))
            if drift_status['drift_detected']:
                logger.warning("Prediction drift in %s!", model_name)
                results[f'drift_{model_name}'] = drift_status

        conn = sqlite3.connect(self.db_path)
        self.benchmarks.learn_baselines(conn)
        conn.close()
        results['benchmarks_updated'] = len(self.benchmarks.baselines)

        return results

    def get_system_health(self) -> Dict:
        return {
            'models': {
                name: {
                    'n_trees': len(m.trees),
                    'n_seen': m.n_seen,
                    'trend': m.get_trend(),
                    'active_version': self.registry.get_active_version(name),
                }
                for name, m in self.models.items()
            },
            'drift': self.drift.get_all_status(),
            'feedback': self.feedback.get_feedback_stats(),
            'benchmarks': {k: v.get('mean', v.get('value'))
                          for k, v in self.benchmarks.baselines.items()},
            'n_baselines': len(self.benchmarks.baselines),
        }
