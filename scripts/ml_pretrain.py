from __future__ import annotations
import json, math, os, pickle, sqlite3, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

HAS_SKLEARN = False
HAS_TORCH = False
HAS_SCIPY = False

try:
    from sklearn.ensemble import (GradientBoostingClassifier as _SkGBC,
                                  GradientBoostingRegressor as _SkGBR,
                                  IsolationForest as _SkIF)
    from sklearn.preprocessing import LabelEncoder as _SkLE
    from sklearn.metrics import roc_auc_score as _sk_auc
    import joblib as _joblib
    HAS_SKLEARN = True
except ImportError:
    pass

try:
    import torch as _torch
    import torch.nn as _tnn
    HAS_TORCH = _torch.cuda.is_available() or True
except ImportError:
    pass

try:
    from scipy import stats as _sp_stats
    HAS_SCIPY = True
except ImportError:
    pass

_BACKEND = 'sklearn' if HAS_SKLEARN else 'numpy'

_BASE = os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_BASE, 'data', 'models')
DB_PATH = os.path.join(_BASE, 'data', 'healthcare_production.db')
MANIFEST_FILE = os.path.join(MODEL_DIR, 'manifest.json')


class DecisionStump:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value',
                 'left_child', 'right_child', 'depth')

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.value = None
        self.left_child = None
        self.right_child = None
        self.depth = 0

    def fit(self, X: np.ndarray, residuals: np.ndarray, max_depth: int = 4,
            min_samples_leaf: int = 20, subsample_cols: float = 0.8):
        n, d = X.shape
        self.depth = 0

        n_cols = max(1, int(d * subsample_cols))
        col_idx = np.random.choice(d, n_cols, replace=False)

        self._build(X, residuals, col_idx, max_depth, min_samples_leaf, depth=0)

    def _build(self, X, residuals, col_idx, max_depth, min_samples_leaf, depth):
        n = len(residuals)
        self.value = np.mean(residuals) if n > 0 else 0.0

        if depth >= max_depth or n < 2 * min_samples_leaf:
            return

        best_gain = 0.0
        best_feat = None
        best_thresh = None
        total_var = np.var(residuals) * n

        for feat in col_idx:
            vals = X[:, feat]
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            if len(uniq) > 20:
                candidates = np.percentile(vals, np.linspace(10, 90, 10))
            else:
                candidates = (uniq[:-1] + uniq[1:]) / 2.0

            for thresh in candidates:
                left_mask = vals <= thresh
                n_left = np.sum(left_mask)
                n_right = n - n_left

                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                left_var = np.var(residuals[left_mask]) * n_left if n_left > 0 else 0
                right_var = np.var(residuals[~left_mask]) * n_right if n_right > 0 else 0
                gain = total_var - left_var - right_var

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            return

        self.feature = best_feat
        self.threshold = best_thresh

        mask = X[:, best_feat] <= best_thresh
        self.left_child = DecisionStump()
        self.left_child._build(X[mask], residuals[mask], col_idx, max_depth, min_samples_leaf, depth + 1)

        self.right_child = DecisionStump()
        self.right_child._build(X[~mask], residuals[~mask], col_idx, max_depth, min_samples_leaf, depth + 1)

    def predict_one(self, x: np.ndarray) -> float:
        if self.feature is None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left_child.predict_one(x) if self.left_child else self.value
        else:
            return self.right_child.predict_one(x) if self.right_child else self.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        result = np.empty(n)
        self._predict_batch(X, np.arange(n), result)
        return result

    def _predict_batch(self, X, indices, result):
        if len(indices) == 0:
            return
        if self.feature is None:
            result[indices] = self.value
            return
        mask = X[indices, self.feature] <= self.threshold
        left_idx = indices[mask]
        right_idx = indices[~mask]
        if self.left_child:
            self.left_child._predict_batch(X, left_idx, result)
        else:
            result[left_idx] = self.value
        if self.right_child:
            self.right_child._predict_batch(X, right_idx, result)
        else:
            result[right_idx] = self.value


class GradientBoostedModel:

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 4, min_samples_leaf: int = 20,
                 subsample: float = 0.8, subsample_cols: float = 0.8,
                 task: str = 'classification'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.subsample_cols = subsample_cols
        self.task = task
        self.trees: List[DecisionStump] = []
        self.init_prediction = 0.0
        self.feature_importances_ = None
        self.feature_names_ = None
        self.train_losses_ = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None,
            verbose: bool = True):
        n, d = X.shape
        self.feature_names_ = feature_names
        self.feature_importances_ = np.zeros(d)

        if self.task == 'classification':
            pos_rate = np.mean(y)
            self.init_prediction = np.log(pos_rate / (1 - pos_rate + 1e-10))
        else:
            self.init_prediction = np.mean(y)

        F = np.full(n, self.init_prediction, dtype=np.float64)
        self.trees = []
        self.train_losses_ = []

        for i in range(self.n_estimators):
            if self.task == 'classification':
                p = self._sigmoid(F)
                residuals = y - p
            else:
                residuals = y - F

            if self.subsample < 1.0:
                idx = np.random.choice(n, int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)

            tree = DecisionStump()
            tree.fit(X[idx], residuals[idx], max_depth=self.max_depth,
                     min_samples_leaf=self.min_samples_leaf,
                     subsample_cols=self.subsample_cols)
            self.trees.append(tree)

            update = tree.predict(X)
            F += self.learning_rate * update

            self._accumulate_importance(tree)

            if verbose and (i + 1) % 25 == 0:
                if self.task == 'classification':
                    loss = -np.mean(y * np.log(self._sigmoid(F) + 1e-10) +
                                    (1 - y) * np.log(1 - self._sigmoid(F) + 1e-10))
                else:
                    loss = np.mean((y - F) ** 2)
                self.train_losses_.append(loss)
                print(f'    Tree {i+1}/{self.n_estimators} — loss: {loss:.4f}')

        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total

    def _accumulate_importance(self, tree: DecisionStump):
        if tree.feature is not None:
            self.feature_importances_[tree.feature] += 1
            if tree.left_child:
                self._accumulate_importance(tree.left_child)
            if tree.right_child:
                self._accumulate_importance(tree.right_child)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.init_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_raw(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            return (self.predict_proba(X) >= 0.5).astype(int)
        return self.predict_raw(X)

    def add_trees(self, X: np.ndarray, y: np.ndarray, n_new_trees: int = 20):
        n = X.shape[0]
        F = self.predict_raw(X)

        for i in range(n_new_trees):
            if self.task == 'classification':
                p = self._sigmoid(F)
                residuals = y - p
            else:
                residuals = y - F

            idx = np.random.choice(n, int(n * self.subsample), replace=False)
            tree = DecisionStump()
            tree.fit(X[idx], residuals[idx], max_depth=self.max_depth,
                     min_samples_leaf=self.min_samples_leaf,
                     subsample_cols=self.subsample_cols)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
            self._accumulate_importance(tree)

        self.n_estimators = len(self.trees)

    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        if self.feature_importances_ is None:
            return []
        idx = np.argsort(self.feature_importances_)[::-1][:top_n]
        return [
            {'feature': self.feature_names_[i] if self.feature_names_ else f'f{i}',
             'importance': round(float(self.feature_importances_[i]), 4)}
            for i in idx if self.feature_importances_[i] > 0
        ]


class IsolationTree:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'size', 'depth')

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.size = 0

    def fit(self, X: np.ndarray, max_depth: int):
        self.size = X.shape[0]
        if X.shape[0] <= 1 or max_depth <= 0:
            return
        d = X.shape[1]
        feat = np.random.randint(d)
        col = X[:, feat]
        mn, mx = col.min(), col.max()
        if mn == mx:
            return
        thresh = np.random.uniform(mn, mx)
        self.feature = feat
        self.threshold = thresh

        mask = col < thresh
        self.left = IsolationTree()
        self.left.fit(X[mask], max_depth - 1)
        self.right = IsolationTree()
        self.right.fit(X[~mask], max_depth - 1)

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.feature is None:
            return depth + _c(self.size)
        if x[self.feature] < self.threshold:
            return self.left.path_length(x, depth + 1) if self.left else depth
        return self.right.path_length(x, depth + 1) if self.right else depth


def _c(n):
    if n <= 1:
        return 0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees: List[IsolationTree] = []

    def fit(self, X: np.ndarray, verbose: bool = True):
        n = X.shape[0]
        self.trees = []
        max_depth = int(np.ceil(np.log2(self.max_samples)))

        for i in range(self.n_estimators):
            idx = np.random.choice(n, min(self.max_samples, n), replace=False)
            tree = IsolationTree()
            tree.fit(X[idx], max_depth)
            self.trees.append(tree)

            if verbose and (i + 1) % 25 == 0:
                print(f'    iForest tree {i+1}/{self.n_estimators}')

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        avg_path = np.zeros(X.shape[0])
        for tree in self.trees:
            for j in range(X.shape[0]):
                avg_path[j] += tree.path_length(X[j])
        avg_path /= len(self.trees)
        c_n = _c(self.max_samples)
        return 2.0 ** (-avg_path / (c_n + 1e-10))

    def predict(self, X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
        scores = self.anomaly_scores(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)


class SklearnGBM:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=20, subsample=0.8, task='classification', **kw):
        self.task = task
        self.n_estimators = n_estimators
        self.feature_names_ = None
        self.feature_importances_ = None
        if task == 'classification':
            self._model = _SkGBC(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                subsample=subsample, validation_fraction=0.1,
                n_iter_no_change=10, random_state=42)
        else:
            self._model = _SkGBR(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                subsample=subsample, validation_fraction=0.1,
                n_iter_no_change=10, random_state=42)

    def fit(self, X, y, feature_names=None, verbose=True):
        self.feature_names_ = feature_names
        if verbose:
            print(f'    sklearn {self.task} GBM: fitting {X.shape[0]} samples, {X.shape[1]} features ...')
        self._model.fit(X, y)
        self.n_estimators = self._model.n_estimators_
        self.feature_importances_ = self._model.feature_importances_
        if verbose:
            print(f'    Done — {self.n_estimators} trees (early stopping may reduce)')

    def predict_proba(self, X):
        if self.task == 'classification':
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def predict(self, X):
        return self._model.predict(X)

    def predict_raw(self, X):
        if self.task == 'classification':
            return self._model.decision_function(X)
        return self._model.predict(X)

    def add_trees(self, X, y, n_new_trees=20):
        self._model.set_params(n_estimators=self._model.n_estimators_ + n_new_trees,
                               warm_start=True)
        self._model.fit(X, y)
        self.n_estimators = self._model.n_estimators_
        self.feature_importances_ = self._model.feature_importances_

    def get_feature_importance(self, top_n=10):
        if self.feature_importances_ is None:
            return []
        idx = np.argsort(self.feature_importances_)[::-1][:top_n]
        return [
            {'feature': self.feature_names_[i] if self.feature_names_ else f'f{i}',
             'importance': round(float(self.feature_importances_[i]), 4)}
            for i in idx if self.feature_importances_[i] > 0
        ]


class SklearnIsolationForest:

    def __init__(self, n_estimators=100, max_samples=256, **kw):
        self.n_estimators = n_estimators
        self._model = _SkIF(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=0.05, random_state=42, n_jobs=-1)

    def fit(self, X, verbose=True):
        if verbose:
            print(f'    sklearn IsolationForest: fitting {X.shape[0]} samples ...')
        self._model.fit(X)
        if verbose:
            print(f'    Done')

    def anomaly_scores(self, X):
        raw = -self._model.score_samples(X)
        mn, mx = raw.min(), raw.max()
        if mx - mn < 1e-10:
            return np.zeros(len(X))
        return (raw - mn) / (mx - mn)

    def predict(self, X, contamination=0.05):
        scores = self.anomaly_scores(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)


def create_gbm(task='classification', **kwargs):
    if HAS_SKLEARN:
        return SklearnGBM(task=task, **kwargs)
    return GradientBoostedModel(task=task, **kwargs)


def create_isolation_forest(**kwargs):
    if HAS_SKLEARN:
        return SklearnIsolationForest(**kwargs)
    return IsolationForest(**kwargs)


class FeatureEncoder:

    def __init__(self):
        self.encodings = {}
        self.feature_names = []

    def fit_transform_claims(self, df: pd.DataFrame, members_df: pd.DataFrame = None,
                              providers_df: pd.DataFrame = None) -> Tuple[np.ndarray, List[str]]:
        features = []
        names = []

        df['billed_amount'] = pd.to_numeric(df['billed_amount'], errors='coerce').fillna(0)
        df['paid_amount'] = pd.to_numeric(df['paid_amount'], errors='coerce').fillna(0)

        features.append(df['billed_amount'].values.reshape(-1, 1))
        names.append('billed_amount')
        features.append(df['paid_amount'].values.reshape(-1, 1))
        names.append('paid_amount')

        ratio = df['paid_amount'] / (df['billed_amount'] + 1)
        features.append(ratio.values.reshape(-1, 1))
        names.append('paid_billed_ratio')

        features.append(np.log1p(df['billed_amount'].values).reshape(-1, 1))
        names.append('log_billed')
        features.append(np.log1p(df['paid_amount'].values).reshape(-1, 1))
        names.append('log_paid')

        for col in ['encounter_type', 'region']:
            enc = self._encode_col(df[col], col)
            features.append(enc.reshape(-1, 1))
            names.append(f'{col}_enc')

        diag_cat = df['diagnosis_code'].str[:3].fillna('UNK')
        enc = self._encode_col(diag_cat, 'diag_cat')
        features.append(enc.reshape(-1, 1))
        names.append('diag_category')

        dates = pd.to_datetime(df['service_date'], errors='coerce')
        features.append(dates.dt.month.fillna(1).values.reshape(-1, 1).astype(float))
        names.append('service_month')
        features.append(dates.dt.dayofweek.fillna(0).values.reshape(-1, 1).astype(float))
        names.append('service_dow')
        features.append(dates.dt.year.fillna(2024).values.reshape(-1, 1).astype(float))
        names.append('service_year')

        fac_enc = self._encode_col(df['facility_id'], 'facility_id')
        features.append(fac_enc.reshape(-1, 1))
        names.append('facility_enc')

        prov_enc = self._encode_col(df['provider_id'], 'provider_id')
        features.append(prov_enc.reshape(-1, 1))
        names.append('provider_enc')

        if members_df is not None and len(members_df) > 0:
            mem_lookup = members_df.set_index('MEMBER_ID')

            ages = df['member_id'].map(mem_lookup.get('AGE', pd.Series(dtype=float))).fillna(50).astype(float)
            features.append(ages.values.reshape(-1, 1))
            names.append('member_age')

            risk = df['member_id'].map(mem_lookup.get('RISK_SCORE', pd.Series(dtype=float))).fillna(1.0).astype(float)
            features.append(risk.values.reshape(-1, 1))
            names.append('member_risk_score')

            gender = df['member_id'].map(mem_lookup.get('GENDER', pd.Series(dtype=str))).fillna('U')
            gender_enc = self._encode_col(gender, 'gender')
            features.append(gender_enc.reshape(-1, 1))
            names.append('member_gender')

            plan = df['member_id'].map(mem_lookup.get('PLAN_TYPE', pd.Series(dtype=str))).fillna('UNK')
            plan_enc = self._encode_col(plan, 'plan_type')
            features.append(plan_enc.reshape(-1, 1))
            names.append('member_plan_type')

            chronic = df['member_id'].map(mem_lookup.get('CHRONIC_CONDITIONS', pd.Series(dtype=float))).fillna(0).astype(float)
            features.append(chronic.values.reshape(-1, 1))
            names.append('chronic_conditions')

            lob = df['member_id'].map(mem_lookup.get('LOB', pd.Series(dtype=str))).fillna('UNK')
            lob_enc = self._encode_col(lob, 'lob')
            features.append(lob_enc.reshape(-1, 1))
            names.append('member_lob')

        if providers_df is not None and len(providers_df) > 0:
            prov_lookup = providers_df.set_index('NPI')
            spec = df['provider_id'].map(prov_lookup.get('SPECIALTY', pd.Series(dtype=str))).fillna('UNK')
            spec_enc = self._encode_col(spec, 'specialty')
            features.append(spec_enc.reshape(-1, 1))
            names.append('provider_specialty')

            panel = df['provider_id'].map(prov_lookup.get('PANEL_SIZE', pd.Series(dtype=float))).fillna(100).astype(float)
            features.append(panel.values.reshape(-1, 1))
            names.append('provider_panel_size')

        X = np.hstack(features).astype(np.float64)
        self.feature_names = names
        return X, names

    def _encode_col(self, series, name: str) -> np.ndarray:
        if name not in self.encodings:
            uniq = series.dropna().unique()
            self.encodings[name] = {v: i for i, v in enumerate(sorted(str(u) for u in uniq))}
        mapping = self.encodings[name]
        return series.map(lambda x: mapping.get(str(x), -1)).values.astype(float)

    def transform_claims(self, df: pd.DataFrame, members_df=None, providers_df=None):
        return self.fit_transform_claims(df, members_df, providers_df)


def build_member_features(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print('  Building member-level features...')

    members = pd.read_sql('SELECT * FROM members', conn)

    member_util = pd.read_sql('SELECT * FROM _gpdm_summary_member_util', conn)

    df = members.merge(member_util, left_on='MEMBER_ID', right_on='member_id', how='left')
    df['total_claims'] = df['total_claims'].fillna(0)
    df['total_paid'] = df['total_paid_y'].fillna(0) if 'total_paid_y' in df.columns else df.get('total_paid', pd.Series(0, index=df.index)).fillna(0)
    df['denied_claims'] = df['denied_claims'].fillna(0)

    features = []
    names = []

    features.append(pd.to_numeric(df['AGE'], errors='coerce').fillna(50).values.reshape(-1, 1))
    names.append('age')

    features.append(pd.to_numeric(df['RISK_SCORE'], errors='coerce').fillna(1.0).values.reshape(-1, 1))
    names.append('risk_score')

    features.append(pd.to_numeric(df['CHRONIC_CONDITIONS'], errors='coerce').fillna(0).values.reshape(-1, 1))
    names.append('chronic_conditions')

    features.append(df['total_claims'].values.reshape(-1, 1).astype(float))
    names.append('total_claims')

    tp = df['total_paid'].values.astype(float)
    features.append(tp.reshape(-1, 1))
    names.append('total_paid')

    avg_cost = np.where(df['total_claims'] > 0, tp / df['total_claims'].values, 0)
    features.append(avg_cost.reshape(-1, 1))
    names.append('avg_cost_per_claim')

    den_ratio = np.where(df['total_claims'] > 0,
                         df['denied_claims'].values / df['total_claims'].values, 0)
    features.append(den_ratio.reshape(-1, 1))
    names.append('denial_ratio')

    visit_div = df['visit_types'].fillna(0).astype(float).values
    features.append(visit_div.reshape(-1, 1))
    names.append('visit_type_diversity')

    region_map = {r: i for i, r in enumerate(sorted(df['REGION'].dropna().unique()))}
    features.append(df['REGION'].map(region_map).fillna(0).values.reshape(-1, 1).astype(float))
    names.append('region')

    gender_map = {'F': 0, 'M': 1, 'U': 2}
    features.append(df['GENDER'].map(gender_map).fillna(2).values.reshape(-1, 1).astype(float))
    names.append('gender')

    plan_map = {p: i for i, p in enumerate(sorted(df['PLAN_TYPE'].dropna().unique()))}
    features.append(df['PLAN_TYPE'].map(plan_map).fillna(0).values.reshape(-1, 1).astype(float))
    names.append('plan_type')

    lob_map = {l: i for i, l in enumerate(sorted(df['LOB'].dropna().unique()))}
    features.append(df['LOB'].map(lob_map).fillna(0).values.reshape(-1, 1).astype(float))
    names.append('lob')

    enroll = pd.to_datetime(df['ENROLLMENT_DATE'], errors='coerce')
    last_svc = pd.to_datetime(df['LAST_SERVICE_DATE'], errors='coerce')
    tenure = (last_svc - enroll).dt.days.fillna(365).values.astype(float)
    features.append(tenure.reshape(-1, 1))
    names.append('tenure_days')

    last_claim = pd.to_datetime(df.get('last_claim', pd.Series(dtype=str)), errors='coerce')
    now = pd.Timestamp.now()
    recency = (now - last_claim).dt.days.fillna(999).values.astype(float)
    features.append(recency.reshape(-1, 1))
    names.append('days_since_last_claim')

    X = np.hstack(features).astype(np.float64)

    churn_score = (
        (df['total_claims'] == 0).astype(float) * 0.5 +
        (recency > 180).astype(float) * 0.3 +
        (den_ratio > 0.3).astype(float) * 0.2
    )
    y_churn = (churn_score > 0.4).astype(float).values

    multi_visit = (df['total_claims'] > df['total_claims'].median()).astype(float)
    high_cost = (tp > np.percentile(tp[tp > 0], 75)).astype(float) if np.sum(tp > 0) > 0 else np.zeros_like(tp)
    high_chronic = (pd.to_numeric(df['CHRONIC_CONDITIONS'], errors='coerce').fillna(0) > 2).astype(float)
    y_readmit = ((multi_visit + high_cost + high_chronic) >= 2).astype(float).values

    print(f'  Member features: {X.shape[0]} members, {X.shape[1]} features')
    print(f'  Churn positive rate: {np.mean(y_churn)*100:.1f}%')
    print(f'  Readmission positive rate: {np.mean(y_readmit)*100:.1f}%')

    return X, y_churn, y_readmit, names


class ProviderScorer:

    def __init__(self):
        self.provider_scores = {}
        self.percentiles = {}

    def fit(self, conn: sqlite3.Connection):
        print('  Training provider efficiency scorer...')

        df = pd.read_sql('''
            SELECT npi, specialty, claim_count, total_paid, avg_paid,
                   unique_patients, denied_count
            FROM _gpdm_summary_provider
        ''', conn)

        df['denial_rate'] = df['denied_count'] / (df['claim_count'] + 1)
        df['cost_efficiency'] = df['avg_paid']
        df['patient_load'] = df['unique_patients']
        df['volume'] = df['claim_count']

        for col in ['denial_rate', 'cost_efficiency']:
            self.percentiles[col] = {
                'mean': df[col].mean(), 'std': df[col].std() + 1e-10
            }
            df[f'{col}_score'] = 100 - (df[col].rank(pct=True) * 100)

        for col in ['patient_load', 'volume']:
            self.percentiles[col] = {
                'mean': df[col].mean(), 'std': df[col].std() + 1e-10
            }
            df[f'{col}_score'] = df[col].rank(pct=True) * 100

        df['composite_score'] = (
            df['denial_rate_score'] * 0.30 +
            df['cost_efficiency_score'] * 0.30 +
            df['patient_load_score'] * 0.20 +
            df['volume_score'] * 0.20
        ).round(2)

        for _, row in df.iterrows():
            self.provider_scores[row['npi']] = {
                'npi': row['npi'],
                'specialty': row['specialty'],
                'composite_score': row['composite_score'],
                'denial_rate_score': round(row['denial_rate_score'], 1),
                'cost_efficiency_score': round(row['cost_efficiency_score'], 1),
                'volume_score': round(row['volume_score'], 1),
                'denial_rate': round(row['denial_rate'], 4),
                'avg_paid': round(row['avg_paid'], 2),
                'claim_count': int(row['claim_count']),
            }

        print(f'  Scored {len(self.provider_scores)} providers')
        scores = [v['composite_score'] for v in self.provider_scores.values()]
        print(f'  Score range: {min(scores):.1f} - {max(scores):.1f}, mean: {np.mean(scores):.1f}')

    def get_top_providers(self, n: int = 20) -> List[Dict]:
        return sorted(self.provider_scores.values(),
                      key=lambda x: x['composite_score'], reverse=True)[:n]

    def get_bottom_providers(self, n: int = 20) -> List[Dict]:
        return sorted(self.provider_scores.values(),
                      key=lambda x: x['composite_score'])[:n]

    def score_provider(self, npi: str) -> Optional[Dict]:
        return self.provider_scores.get(npi)


class MLModelStore:

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.encoder = None
        self.manifest = {}

    def train_all(self, db_path: str = DB_PATH, sample_size: int = 500000):
        os.makedirs(self.model_dir, exist_ok=True)
        t_start = time.time()

        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA cache_size=-200000')
        conn.execute('PRAGMA mmap_size=1073741824')
        conn.execute('PRAGMA temp_store=MEMORY')

        print('=' * 60)
        print('GPDM ML Pre-Training Pipeline')
        print(f'  Backend: {_BACKEND.upper()}' +
              (' (sklearn + joblib)' if HAS_SKLEARN else ' (pure numpy fallback)'))
        if HAS_TORCH:
            print(f'  PyTorch: available (device={_torch.device("cuda" if _torch.cuda.is_available() else "cpu")})')
        if HAS_SCIPY:
            print(f'  SciPy: available (advanced statistics)')
        print('  HIPAA: all computation is LOCAL ONLY — zero network calls')
        print('=' * 60)

        print('\n[1/6] Loading reference data...')
        members = pd.read_sql('SELECT * FROM members', conn)
        providers = pd.read_sql('SELECT * FROM providers', conn)
        total_claims = conn.execute('SELECT COUNT(*) FROM claims_4m').fetchone()[0]
        print(f'  {len(members)} members, {len(providers)} providers, {total_claims} claims')

        print(f'\n[2/6] Loading claims sample ({sample_size} rows)...')
        t0 = time.time()
        claims = pd.read_sql(f'''
            SELECT * FROM claims_4m
            ORDER BY RANDOM()
            LIMIT {sample_size}
        ''', conn)
        print(f'  Loaded {len(claims)} claims in {time.time()-t0:.1f}s')

        print('\n[3/6] Feature engineering...')
        self.encoder = FeatureEncoder()
        X, feature_names = self.encoder.fit_transform_claims(claims, members, providers)
        print(f'  Base claim features: {X.shape[1]} features x {X.shape[0]} samples')

        try:
            from hybrid_intelligence import EmbeddingEnrichedFeatures
            enricher = EmbeddingEnrichedFeatures(n_components=16)
            X, feature_names = enricher.enrich_claim_features(claims, X, feature_names)
            print(f'  Enriched features: {X.shape[1]} features (added embedding dims)')
        except Exception as e:
            print(f'  Embedding enrichment skipped: {e}')

        print('\n[4/6] Training models...')
        print('\n  >> Denial Predictor (GBM classification)')
        y_denial = (claims['status'].str.upper() == 'DENIED').astype(float).values
        denial_model = create_gbm(
            n_estimators=150 if HAS_SKLEARN else 80,
            learning_rate=0.08 if HAS_SKLEARN else 0.12,
            max_depth=5 if HAS_SKLEARN else 4,
            min_samples_leaf=50 if HAS_SKLEARN else 100,
            subsample=0.8 if HAS_SKLEARN else 0.6,
            task='classification'
        )
        denial_model.fit(X, y_denial, feature_names=feature_names)
        denial_preds = denial_model.predict_proba(X)
        from_model_acc = np.mean((denial_preds >= 0.5) == y_denial)
        print(f'    Accuracy: {from_model_acc*100:.1f}% | AUC proxy: {self._auc_proxy(y_denial, denial_preds):.3f}')
        print(f'    Top features: {denial_model.get_feature_importance(5)}')
        self.models['denial_predictor'] = denial_model

        print('\n  >> Cost Predictor (GBM regression)')
        y_cost = claims['paid_amount'].astype(float).values
        cost_feature_mask = [i for i, n in enumerate(feature_names) if 'paid' not in n.lower()]
        X_cost = X[:, cost_feature_mask]
        cost_names = [feature_names[i] for i in cost_feature_mask]

        cost_model = create_gbm(
            n_estimators=120 if HAS_SKLEARN else 60,
            learning_rate=0.1 if HAS_SKLEARN else 0.15,
            max_depth=5 if HAS_SKLEARN else 4,
            min_samples_leaf=50 if HAS_SKLEARN else 100,
            subsample=0.8 if HAS_SKLEARN else 0.6,
            task='regression'
        )
        cost_model.fit(X_cost, y_cost, feature_names=cost_names)
        cost_preds = cost_model.predict(X_cost)
        mae = np.mean(np.abs(y_cost - cost_preds))
        r2 = 1 - np.sum((y_cost - cost_preds)**2) / (np.sum((y_cost - np.mean(y_cost))**2) + 1e-10)
        print(f'    MAE: ${mae:,.0f} | R²: {r2:.3f}')
        print(f'    Top features: {cost_model.get_feature_importance(5)}')
        self.models['cost_predictor'] = cost_model
        self.models['cost_feature_mask'] = cost_feature_mask

        print('\n  >> Anomaly Detector (Isolation Forest)')
        numeric_cols = [i for i, n in enumerate(feature_names)
                       if n in ('billed_amount', 'paid_amount', 'paid_billed_ratio',
                               'log_billed', 'log_paid', 'member_age', 'member_risk_score',
                               'chronic_conditions')]
        X_anom = X[:, numeric_cols] if numeric_cols else X[:, :5]

        anom_model = create_isolation_forest(
            n_estimators=100 if HAS_SKLEARN else 50, max_samples=256)
        anom_model.fit(X_anom)
        anom_scores = anom_model.anomaly_scores(X_anom[:1000])
        n_anomalies = np.sum(anom_scores > np.percentile(anom_scores, 95))
        print(f'    Anomaly rate (top 5%): {n_anomalies/10:.1f}%')
        self.models['anomaly_detector'] = anom_model
        self.models['anomaly_cols'] = numeric_cols

        print('\n  >> Member-Level Models (Churn + Readmission)')
        X_mem, y_churn, y_readmit, mem_names = build_member_features(conn)

        print('\n  >> Churn Risk Predictor (GBM classification)')
        churn_model = create_gbm(
            n_estimators=100 if HAS_SKLEARN else 60,
            learning_rate=0.1 if HAS_SKLEARN else 0.15,
            max_depth=5 if HAS_SKLEARN else 4,
            min_samples_leaf=100 if HAS_SKLEARN else 200,
            subsample=0.7 if HAS_SKLEARN else 0.5,
            task='classification'
        )
        churn_model.fit(X_mem, y_churn, feature_names=mem_names)
        churn_preds = churn_model.predict_proba(X_mem)
        churn_acc = np.mean((churn_preds >= 0.5) == y_churn)
        print(f'    Accuracy: {churn_acc*100:.1f}% | AUC proxy: {self._auc_proxy(y_churn, churn_preds):.3f}')
        print(f'    Top features: {churn_model.get_feature_importance(5)}')
        self.models['churn_predictor'] = churn_model

        print('\n  >> Readmission Risk Predictor (GBM classification)')
        readmit_model = create_gbm(
            n_estimators=100 if HAS_SKLEARN else 60,
            learning_rate=0.1 if HAS_SKLEARN else 0.15,
            max_depth=5 if HAS_SKLEARN else 4,
            min_samples_leaf=100 if HAS_SKLEARN else 200,
            subsample=0.7 if HAS_SKLEARN else 0.5,
            task='classification'
        )
        readmit_model.fit(X_mem, y_readmit, feature_names=mem_names)
        readmit_preds = readmit_model.predict_proba(X_mem)
        readmit_acc = np.mean((readmit_preds >= 0.5) == y_readmit)
        print(f'    Accuracy: {readmit_acc*100:.1f}% | AUC proxy: {self._auc_proxy(y_readmit, readmit_preds):.3f}')
        print(f'    Top features: {readmit_model.get_feature_importance(5)}')
        self.models['readmission_predictor'] = readmit_model

        self.models['member_feature_names'] = mem_names

        print('\n[5/6] Training Provider Scorer...')
        provider_scorer = ProviderScorer()
        provider_scorer.fit(conn)
        self.models['provider_scorer'] = provider_scorer

        print('\n  Storing member risk scores in DB...')
        churn_all = churn_model.predict_proba(X_mem)
        readmit_all = readmit_model.predict_proba(X_mem)

        conn.execute('DROP TABLE IF EXISTS _gpdm_member_risk')
        conn.execute('''CREATE TABLE _gpdm_member_risk (
            member_id TEXT PRIMARY KEY,
            churn_risk REAL,
            readmission_risk REAL,
            risk_tier TEXT
        )''')
        members_list = pd.read_sql('SELECT MEMBER_ID FROM members', conn)['MEMBER_ID'].values
        risk_data = []
        for i, mid in enumerate(members_list[:len(churn_all)]):
            cr = float(churn_all[i])
            rr = float(readmit_all[i])
            tier = 'HIGH' if max(cr, rr) > 0.7 else ('MEDIUM' if max(cr, rr) > 0.4 else 'LOW')
            risk_data.append((mid, cr, rr, tier))

        conn.executemany('INSERT INTO _gpdm_member_risk VALUES (?,?,?,?)', risk_data)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_member_risk_tier ON _gpdm_member_risk(risk_tier)')
        conn.commit()

        high_risk = sum(1 for _, cr, rr, _ in risk_data if max(cr, rr) > 0.7)
        print(f'  Stored risk scores for {len(risk_data)} members ({high_risk} high-risk)')

        conn.execute('DROP TABLE IF EXISTS _gpdm_provider_scores')
        conn.execute('''CREATE TABLE _gpdm_provider_scores (
            npi TEXT PRIMARY KEY,
            specialty TEXT,
            composite_score REAL,
            denial_rate_score REAL,
            cost_efficiency_score REAL,
            volume_score REAL,
            claim_count INTEGER
        )''')
        for p in provider_scorer.provider_scores.values():
            conn.execute('INSERT INTO _gpdm_provider_scores VALUES (?,?,?,?,?,?,?)',
                        (p['npi'], p['specialty'], p['composite_score'],
                         p['denial_rate_score'], p['cost_efficiency_score'],
                         p['volume_score'], p['claim_count']))
        conn.commit()
        print(f'  Stored scores for {len(provider_scorer.provider_scores)} providers')

        print('\n[6/6] Saving models to disk...')
        self.save_all()

        total_time = time.time() - t_start
        self.manifest = {
            'version': 3,
            'backend': _BACKEND,
            'has_sklearn': HAS_SKLEARN,
            'has_torch': HAS_TORCH,
            'has_scipy': HAS_SCIPY,
            'hipaa_compliant': True,
            'data_locality': 'all_local_no_network',
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': round(total_time, 1),
            'total_claims_used': len(claims),
            'total_members_scored': len(risk_data),
            'total_providers_scored': len(provider_scorer.provider_scores),
            'models': {
                'denial_predictor': {
                    'type': f'{_BACKEND}_GBM_classifier',
                    'trees': denial_model.n_estimators,
                    'accuracy': round(from_model_acc * 100, 1),
                    'top_features': denial_model.get_feature_importance(5),
                },
                'cost_predictor': {
                    'type': f'{_BACKEND}_GBM_regressor',
                    'trees': cost_model.n_estimators,
                    'mae': round(mae, 0), 'r2': round(r2, 3),
                    'top_features': cost_model.get_feature_importance(5),
                },
                'anomaly_detector': {
                    'type': f'{_BACKEND}_IsolationForest',
                    'trees': anom_model.n_estimators,
                },
                'churn_predictor': {
                    'type': f'{_BACKEND}_GBM_classifier',
                    'trees': churn_model.n_estimators,
                    'accuracy': round(churn_acc * 100, 1),
                    'top_features': churn_model.get_feature_importance(5),
                },
                'readmission_predictor': {
                    'type': f'{_BACKEND}_GBM_classifier',
                    'trees': readmit_model.n_estimators,
                    'accuracy': round(readmit_acc * 100, 1),
                    'top_features': readmit_model.get_feature_importance(5),
                },
                'provider_scorer': {
                    'type': 'statistical_composite',
                    'providers_scored': len(provider_scorer.provider_scores),
                },
            }
        }

        print('\n--- Extended ML Stack ---')

        neural_status = {}
        try:
            import neural_engine as _ne
            ns = _ne.init_neural_stack(db_path=db_path, warmup=True)
            neural_status = {
                'embedder_ready': ns.get('embedder_ready', False),
                'embedder_type': ns.get('embedder_type', ''),
                'vocab_size': ns.get('vocab_size', 0),
                'schema_docs': ns.get('schema_docs', 0),
                'init_time': ns.get('init_time', 0),
            }
            print(f'  Neural engine: OK ({neural_status["vocab_size"]} vocab, '
                  f'{neural_status["schema_docs"]} schema docs)')
        except Exception as e:
            print(f'  Neural engine: skipped ({e})')

        causal_status = {}
        try:
            import causal_inference as _ci
            print('  Running DoubleML followup effect (15K sample)...')
            followup = _ci.estimate_followup_effect(db_path, max_rows=15000)
            if 'error' not in followup:
                causal_status['followup'] = {
                    'ate': followup['ate'],
                    'ci95': list(followup['ate_ci95']),
                    'n': followup['n'],
                    'n_treated': followup['n_treated'],
                }
                print(f'    Followup ATE: {followup["ate"]*100:+.2f}pp '
                      f'(N={followup["n"]}, treated={followup["n_treated"]})')
            else:
                print(f'    Followup: {followup.get("error", "?")}')
        except Exception as e:
            print(f'  Causal inference: skipped ({e})')

        torch_status = {}
        if HAS_TORCH:
            try:
                import patient_transformer as _pt
                print('  Training PatientTransformer (BEHRT)...')
                pt_result = _pt.train(db_path=db_path)
                torch_status['patient_transformer'] = pt_result
                print(f'    PatientTransformer: {pt_result}')
            except Exception as e:
                print(f'  PatientTransformer: skipped ({e})')

            try:
                from neural_engine import train_risk_net
                print('  Training ClinicalRiskNet...')
                rn_result = train_risk_net(db_path=db_path, epochs=15)
                torch_status['risk_net'] = rn_result
                print(f'    ClinicalRiskNet: {rn_result}')
            except Exception as e:
                print(f'  ClinicalRiskNet: skipped ({e})')

        hybrid_status = {}
        try:
            from hybrid_intelligence import HybridIntelligence
            print('\n--- Hybrid Intelligence Pipeline ---')
            hybrid = HybridIntelligence(db_path=db_path)
            hybrid_result = hybrid.run_full_pipeline(max_causal_sample=15000)
            hybrid_status = {
                'members_scored': hybrid_result.get('scoring', {}).get('total_members', 0),
                'interventions_assigned': hybrid_result.get('optimization', {}).get('total_assigned', 0),
                'population_ate': hybrid_result.get('scoring', {}).get('population_ate'),
                'ensemble_weights': hybrid_result.get('ensemble', {}).get('weights', {}),
                'time_seconds': hybrid_result.get('total_time_seconds', 0),
            }
        except Exception as e:
            print(f'  Hybrid intelligence: skipped ({e})')

        adaptive_status = {}
        try:
            from adaptive_learning import AdaptiveLearningEngine
            print('\n--- Adaptive Learning Initialization ---')
            ale = AdaptiveLearningEngine(db_path)
            ale.initialize()
            adaptive_status = {
                'baselines_learned': len(ale.benchmarks.baselines),
                'baseline_kpis': list(ale.benchmarks.baselines.keys()),
                'feedback_initialized': True,
            }
            print(f'  Learned {len(ale.benchmarks.baselines)} data-driven baselines')
        except Exception as e:
            print(f'  Adaptive learning: skipped ({e})')

        smart_kpi_status = {}
        try:
            from smart_kpi_engine import SmartKPIEngine
            print('\n--- Smart KPI Engine ---')
            ske = SmartKPIEngine(db_path)
            kpi_results = ske.compute_all(conn)
            anomalies = kpi_results.get('_anomaly_summary', {})
            corrs = kpi_results.get('_correlations', {})
            smart_kpi_status = {
                'kpis_computed': sum(1 for v in kpi_results.values()
                                     if isinstance(v, dict) and 'value' in v),
                'anomalies_found': anomalies.get('count', 0),
                'correlations_found': len(corrs),
                'nl_report_length': len(ske.explain_all(conn)),
            }
            print(f'  {smart_kpi_status["kpis_computed"]} KPIs computed, '
                  f'{smart_kpi_status["anomalies_found"]} anomalies, '
                  f'{smart_kpi_status["correlations_found"]} correlations')
        except Exception as e:
            print(f'  Smart KPI engine: skipped ({e})')

        self.manifest['extended_ml'] = {
            'neural_engine': neural_status,
            'causal_inference': causal_status,
            'torch_models': torch_status,
            'hybrid_intelligence': hybrid_status,
            'adaptive_learning': adaptive_status,
            'smart_kpi': smart_kpi_status,
        }

        with open(MANIFEST_FILE, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        conn.close()
        print(f'\n{"="*60}')
        print(f'Pre-training complete in {total_time:.1f}s')
        print(f'Models saved to: {self.model_dir}')
        print(f'Extended stack: neural={bool(neural_status)}, causal={bool(causal_status)}, '
              f'torch={bool(torch_status)}, adaptive={bool(adaptive_status)}, '
              f'smart_kpi={bool(smart_kpi_status)}')
        print(f'{"="*60}')

        return self.manifest

    def save_all(self):
        os.makedirs(self.model_dir, exist_ok=True)
        _save = _joblib.dump if HAS_SKLEARN else lambda obj, path: pickle.dump(obj, open(path, 'wb'), protocol=4)
        _ext = '.joblib' if HAS_SKLEARN else '.pkl'

        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f'{name}{_ext}')
            if HAS_SKLEARN:
                _joblib.dump(model, path, compress=3)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(model, f, protocol=4)
            size_kb = os.path.getsize(path) / 1024
            print(f'    Saved {name}: {size_kb:.0f} KB ({_ext})')

        if self.encoder:
            path = os.path.join(self.model_dir, f'feature_encoder{_ext}')
            if HAS_SKLEARN:
                _joblib.dump(self.encoder, path, compress=3)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(self.encoder, f, protocol=4)
            print(f'    Saved feature_encoder: {os.path.getsize(path)/1024:.0f} KB')

    def load_all(self) -> bool:
        if not os.path.exists(self.model_dir):
            return False

        manifest_path = MANIFEST_FILE
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)

        loaded = 0
        for fname in os.listdir(self.model_dir):
            if not (fname.endswith('.pkl') or fname.endswith('.joblib')):
                continue
            name = fname.rsplit('.', 1)[0]
            path = os.path.join(self.model_dir, fname)
            try:
                if fname.endswith('.joblib') and HAS_SKLEARN:
                    obj = _joblib.load(path)
                else:
                    with open(path, 'rb') as f:
                        obj = pickle.load(f)
                if name == 'feature_encoder':
                    self.encoder = obj
                else:
                    self.models[name] = obj
                loaded += 1
            except Exception as e:
                pass

        return loaded > 0

    def is_trained(self) -> bool:
        return os.path.exists(MANIFEST_FILE)

    def _auc_proxy(self, y_true, y_scores) -> float:
        if HAS_SKLEARN:
            try:
                return float(_sk_auc(y_true, y_scores))
            except Exception:
                pass
        pos = y_scores[y_true == 1]
        neg = y_scores[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n_sample = min(5000, len(pos), len(neg))
        pos_s = np.random.choice(pos, n_sample, replace=True)
        neg_s = np.random.choice(neg, n_sample, replace=True)
        return float(np.mean(pos_s > neg_s) + 0.5 * np.mean(pos_s == neg_s))


    def predict_denial_risk(self, claim_features: np.ndarray) -> np.ndarray:
        model = self.models.get('denial_predictor')
        if model is None:
            return np.array([])
        return model.predict_proba(claim_features)

    def predict_cost(self, claim_features: np.ndarray) -> np.ndarray:
        model = self.models.get('cost_predictor')
        mask = self.models.get('cost_feature_mask')
        if model is None:
            return np.array([])
        X = claim_features[:, mask] if mask else claim_features
        return model.predict(X)

    def get_member_churn_risk(self, member_features: np.ndarray) -> np.ndarray:
        model = self.models.get('churn_predictor')
        if model is None:
            return np.array([])
        return model.predict_proba(member_features)

    def get_readmission_risk(self, member_features: np.ndarray) -> np.ndarray:
        model = self.models.get('readmission_predictor')
        if model is None:
            return np.array([])
        return model.predict_proba(member_features)

    def get_anomaly_scores(self, claim_features: np.ndarray) -> np.ndarray:
        model = self.models.get('anomaly_detector')
        cols = self.models.get('anomaly_cols')
        if model is None:
            return np.array([])
        X = claim_features[:, cols] if cols else claim_features
        return model.anomaly_scores(X)

    def incremental_train(self, db_path: str, n_new_trees: int = 20):
        print(f'Incremental training with {n_new_trees} new trees...')
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')

        claims = pd.read_sql('''
            SELECT * FROM claims_4m
            WHERE service_date >= date('now', '-30 days')
            LIMIT 100000
        ''', conn)

        if len(claims) == 0:
            print('  No new data to train on')
            conn.close()
            return

        members = pd.read_sql('SELECT * FROM members', conn)
        providers = pd.read_sql('SELECT * FROM providers', conn)

        if self.encoder is None:
            print('  No encoder loaded — cannot do incremental training')
            conn.close()
            return

        X, _ = self.encoder.transform_claims(claims, members, providers)

        denial_model = self.models.get('denial_predictor')
        if denial_model:
            y_denial = (claims['status'].str.upper() == 'DENIED').astype(float).values
            denial_model.add_trees(X, y_denial, n_new_trees)
            print(f'  Denial model: {denial_model.n_estimators} trees now')

        cost_model = self.models.get('cost_predictor')
        cost_mask = self.models.get('cost_feature_mask')
        if cost_model and cost_mask:
            y_cost = claims['paid_amount'].astype(float).values
            cost_model.add_trees(X[:, cost_mask], y_cost, n_new_trees)
            print(f'  Cost model: {cost_model.n_estimators} trees now')

        self.save_all()
        conn.close()
        print('  Incremental training complete, models saved')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='GPDM ML Pre-Training')
    ap.add_argument('--action', choices=['train', 'status', 'incremental'], default='train')
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--sample', type=int, default=500000,
                    help='Number of claims to sample for training')
    args = ap.parse_args()

    store = MLModelStore()

    if args.action == 'status':
        if store.is_trained():
            with open(MANIFEST_FILE) as f:
                m = json.load(f)
            print(json.dumps(m, indent=2))
        else:
            print('No pre-trained models found. Run with --action train')

    elif args.action == 'train':
        manifest = store.train_all(db_path=args.db, sample_size=args.sample)

    elif args.action == 'incremental':
        if store.load_all():
            store.incremental_train(args.db)
        else:
            print('No existing models to update. Run --action train first.')
