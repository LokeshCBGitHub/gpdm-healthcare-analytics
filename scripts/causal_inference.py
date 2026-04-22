from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalResult:
    method: str
    outcome: str
    treatment: str
    n: int
    n_treated: int
    n_control: int
    ate: float
    ate_ci95: Tuple[float, float]
    ate_se: float
    att: Optional[float] = None
    cate_quantiles: Optional[Dict[str, float]] = None
    nuisance_metrics: Dict[str, float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ate_ci95"] = list(self.ate_ci95)
        return d


class _NumpyGBMAdapter:
    def __init__(self, model):
        self._m = model

    def fit(self, X, y):
        self._m.fit(X, y.astype(float), verbose=False)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        p1 = self._m.predict_proba(X) if hasattr(self._m, 'predict_proba') \
             else self._m.predict(X).astype(float)
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        return np.column_stack([1.0 - p1, p1])


class _NumpyKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n = X.shape[0]
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[:n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current],
                                         indices[current + fold_size:]])
            yield train_idx, test_idx
            current += fold_size


def _make_outcome_model(binary: bool):
    try:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if binary:
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=7)
    except ImportError:
        from ml_pretrain import create_gbm
        return _NumpyGBMAdapter(create_gbm(
            task='classification' if binary else 'regression',
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=50, subsample=0.7))


def _make_propensity_model():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=11)
    except ImportError:
        from ml_pretrain import create_gbm
        return _NumpyGBMAdapter(create_gbm(
            task='classification', n_estimators=80, max_depth=3,
            learning_rate=0.1, min_samples_leaf=50, subsample=0.7))


def _predict_mean(model, X: np.ndarray, binary: bool) -> np.ndarray:
    if binary and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    return model.predict(X).astype(float)


def _s_learner(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
               binary_outcome: bool) -> Tuple[np.ndarray, float]:
    XT = np.hstack([X, T.reshape(-1, 1).astype(float)])
    m = _make_outcome_model(binary_outcome).fit(XT, Y)
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    X0 = np.hstack([X, np.zeros((X.shape[0], 1))])
    cate = _predict_mean(m, X1, binary_outcome) - _predict_mean(m, X0, binary_outcome)
    ate = float(np.mean(cate))
    return cate, ate


def _t_learner(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
               binary_outcome: bool) -> Tuple[np.ndarray, float, Dict[str, float]]:
    m1 = _make_outcome_model(binary_outcome).fit(X[T == 1], Y[T == 1])
    m0 = _make_outcome_model(binary_outcome).fit(X[T == 0], Y[T == 0])
    mu1 = _predict_mean(m1, X, binary_outcome)
    mu0 = _predict_mean(m0, X, binary_outcome)
    cate = mu1 - mu0
    metrics = {
        "mu1_mean": float(mu1.mean()),
        "mu0_mean": float(mu0.mean()),
    }
    return cate, float(cate.mean()), metrics


def _x_learner(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
               binary_outcome: bool) -> np.ndarray:
    m1 = _make_outcome_model(binary_outcome).fit(X[T == 1], Y[T == 1])
    m0 = _make_outcome_model(binary_outcome).fit(X[T == 0], Y[T == 0])
    mu1 = _predict_mean(m1, X, binary_outcome)
    mu0 = _predict_mean(m0, X, binary_outcome)

    D1 = Y[T == 1] - mu0[T == 1]
    D0 = mu1[T == 0] - Y[T == 0]

    g1 = _make_outcome_model(binary=False)
    g0 = _make_outcome_model(binary=False)
    g1.fit(X[T == 1], D1)
    g0.fit(X[T == 0], D0)
    tau1 = g1.predict(X)
    tau0 = g0.predict(X)

    e = _propensity_score(X, T)
    return e * tau0 + (1.0 - e) * tau1


def _propensity_score(X: np.ndarray, T: np.ndarray,
                      clip: Tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    p = _make_propensity_model().fit(X, T)
    proba = p.predict_proba(X)
    e = proba[:, 1] if proba.ndim == 2 else proba
    return np.clip(e, clip[0], clip[1])


def _double_ml_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                   binary_outcome: bool, n_folds: int = 5) -> Tuple[float, float, Dict[str, float]]:
    try:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=23)
    except ImportError:
        kf = _NumpyKFold(n_splits=n_folds, shuffle=True, random_state=23)
    n = X.shape[0]

    mu1_oof = np.zeros(n); mu0_oof = np.zeros(n); e_oof = np.zeros(n)
    for fold, (tr, te) in enumerate(kf.split(X)):
        m1 = _make_outcome_model(binary_outcome).fit(
            X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
        m0 = _make_outcome_model(binary_outcome).fit(
            X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
        mu1_oof[te] = _predict_mean(m1, X[te], binary_outcome)
        mu0_oof[te] = _predict_mean(m0, X[te], binary_outcome)
        p = _make_propensity_model().fit(X[tr], T[tr])
        proba = p.predict_proba(X[te])
        e_raw = proba[:, 1] if proba.ndim == 2 else proba
        e_oof[te] = np.clip(e_raw, 0.02, 0.98)

    psi = (mu1_oof - mu0_oof
           + T * (Y - mu1_oof) / e_oof
           - (1 - T) * (Y - mu0_oof) / (1.0 - e_oof))
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / math.sqrt(n))
    metrics = {
        "mean_propensity_treated": float(e_oof[T == 1].mean()) if (T == 1).any() else float("nan"),
        "mean_propensity_control": float(e_oof[T == 0].mean()) if (T == 0).any() else float("nan"),
        "outcome_residual_std_treated":
            float((Y[T == 1] - mu1_oof[T == 1]).std()) if (T == 1).any() else float("nan"),
        "outcome_residual_std_control":
            float((Y[T == 0] - mu0_oof[T == 0]).std()) if (T == 0).any() else float("nan"),
    }
    return ate, se, metrics


def estimate_ate(
    rows: List[Dict[str, Any]],
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    method: str = "doubleml",
    binary_outcome: Optional[bool] = None,
) -> CausalResult:
    if not rows:
        raise ValueError("no rows supplied")

    X = np.array([[float(r.get(c, 0) or 0) for c in covariates] for r in rows],
                 dtype=np.float64)
    T = np.array([int(bool(r.get(treatment, 0))) for r in rows], dtype=np.int64)
    Y = np.array([float(r.get(outcome, 0) or 0) for r in rows], dtype=np.float64)

    if binary_outcome is None:
        binary_outcome = bool(set(np.unique(Y)).issubset({0.0, 1.0}))

    n = len(rows); n_t = int(T.sum()); n_c = n - n_t
    if n_t < 10 or n_c < 10:
        raise ValueError(f"too-small arms n_treated={n_t} n_control={n_c}")

    method = method.lower()
    if method == "slearner":
        cate, ate = _s_learner(X, T, Y, binary_outcome)
        se = float(np.std(cate, ddof=1) / math.sqrt(n))
        nuisance = {}
    elif method == "tlearner":
        cate, ate, nuisance = _t_learner(X, T, Y, binary_outcome)
        se = float(np.std(cate, ddof=1) / math.sqrt(n))
    elif method == "xlearner":
        cate = _x_learner(X, T, Y, binary_outcome)
        ate = float(cate.mean())
        se = float(np.std(cate, ddof=1) / math.sqrt(n))
        nuisance = {}
    elif method in ("doubleml", "aipw", "dml"):
        ate, se, nuisance = _double_ml_ate(X, T, Y, binary_outcome)
        cate, _, _ = _t_learner(X, T, Y, binary_outcome)
    else:
        raise ValueError(f"unknown method {method}")

    ci95 = (ate - 1.96 * se, ate + 1.96 * se)
    cate_q = {
        "p10": float(np.quantile(cate, 0.10)),
        "p25": float(np.quantile(cate, 0.25)),
        "p50": float(np.quantile(cate, 0.50)),
        "p75": float(np.quantile(cate, 0.75)),
        "p90": float(np.quantile(cate, 0.90)),
    }
    return CausalResult(
        method=method,
        outcome=outcome,
        treatment=treatment,
        n=n, n_treated=n_t, n_control=n_c,
        ate=ate, ate_ci95=ci95, ate_se=se,
        att=float(cate[T == 1].mean()) if (T == 1).any() else None,
        cate_quantiles=cate_q,
        nuisance_metrics=nuisance,
        notes=f"binary_outcome={binary_outcome}",
    )


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=15.0)
    c.row_factory = sqlite3.Row
    return c


def _stratified_sample(rows: List[Dict], treatment_key: str,
                       max_rows: int) -> List[Dict]:
    if len(rows) <= max_rows:
        return rows
    treated = [r for r in rows if r[treatment_key]]
    control = [r for r in rows if not r[treatment_key]]
    ratio = len(treated) / max(len(rows), 1)
    n_t = max(50, int(max_rows * ratio))
    n_c = max_rows - n_t
    rng = np.random.RandomState(42)
    t_idx = rng.choice(len(treated), min(n_t, len(treated)), replace=False)
    c_idx = rng.choice(len(control), min(n_c, len(control)), replace=False)
    return [treated[i] for i in t_idx] + [control[i] for i in c_idx]


def estimate_followup_effect(db_path: str,
                              max_rows: int = 50000) -> Dict[str, Any]:
    _member_limit = max(max_rows * 3, 30000)
    with _conn(db_path) as c:
        sampled_members = [r[0] for r in c.execute(
            "SELECT DISTINCT MEMBER_ID FROM encounters "
            "WHERE VISIT_TYPE='INPATIENT' ORDER BY RANDOM() LIMIT ?",
            (_member_limit,))]
        member_set = set(sampled_members)

        ip = list(c.execute("""
            SELECT e.ENCOUNTER_ID, e.MEMBER_ID, e.ADMIT_DATE, e.DISCHARGE_DATE,
                   COALESCE(e.LENGTH_OF_STAY,0) LOS,
                   CAST(COALESCE(
                       (JULIANDAY('now') - JULIANDAY(m.DATE_OF_BIRTH)) / 365.25,
                       40
                   ) AS INTEGER) AS member_age,
                   COALESCE(m.CHRONIC_CONDITIONS,0) chr
            FROM encounters e JOIN members m ON m.MEMBER_ID=e.MEMBER_ID
            WHERE e.VISIT_TYPE='INPATIENT' AND e.DISCHARGE_DATE IS NOT NULL
            ORDER BY e.MEMBER_ID, e.DISCHARGE_DATE
        """))
        ip = [r for r in ip if r["MEMBER_ID"] in member_set]

        by_m_enc: Dict[str, List[Tuple[str, str]]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, "
            "CASE WHEN ADMIT_DATE IS NOT NULL AND ADMIT_DATE != '' THEN ADMIT_DATE ELSE SERVICE_DATE END AS d, "
            "VISIT_TYPE "
            "FROM encounters WHERE "
            "CASE WHEN ADMIT_DATE IS NOT NULL AND ADMIT_DATE != '' THEN ADMIT_DATE ELSE SERVICE_DATE END IS NOT NULL "
            "AND CASE WHEN ADMIT_DATE IS NOT NULL AND ADMIT_DATE != '' THEN ADMIT_DATE ELSE SERVICE_DATE END != ''"):
            if r["MEMBER_ID"] in member_set:
                by_m_enc.setdefault(r["MEMBER_ID"], []).append(
                    (r["d"] or "", (r["VISIT_TYPE"] or "").upper()))
        claims_by_m: Dict[str, List[Tuple[str, float]]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, SERVICE_DATE, COALESCE(PAID_AMOUNT,0) p FROM claims"):
            if r["MEMBER_ID"] in member_set:
                claims_by_m.setdefault(r["MEMBER_ID"], []).append(
                    (r["SERVICE_DATE"] or "", float(r["p"])))

    from datetime import datetime
    def _d(s: str) -> Optional[datetime]:
        try: return datetime.fromisoformat(str(s)[:10])
        except Exception: return None

    rows: List[Dict[str, Any]] = []
    ip_by_m: Dict[str, List[sqlite3.Row]] = {}
    for r in ip: ip_by_m.setdefault(r["MEMBER_ID"], []).append(r)

    for mid, stays in ip_by_m.items():
        stays.sort(key=lambda x: x["DISCHARGE_DATE"] or "")
        for i, s in enumerate(stays):
            disch = _d(s["DISCHARGE_DATE"])
            admit = _d(s["ADMIT_DATE"])
            if disch is None: continue

            treat = 0
            for ed, v in by_m_enc.get(mid, []):
                edt = _d(ed)
                if edt and 0 < (edt - disch).days <= 14 \
                   and v in ("OUTPATIENT", "OFFICE", "TELEHEALTH", "PREVENTIVE"):
                    treat = 1; break

            outcome = 0
            for j in range(i + 1, len(stays)):
                nadm = _d(stays[j]["ADMIT_DATE"])
                if nadm is None: continue
                dd = (nadm - disch).days
                if 0 <= dd <= 30: outcome = 1; break
                if dd > 30: break

            try:
                age = int(s["member_age"])
            except Exception:
                age = 40
            prior_er = 0; prior_ip = 0; prior_cost = 0.0
            for ed, v in by_m_enc.get(mid, []):
                edt = _d(ed)
                if edt and 0 < (disch - edt).days <= 180:
                    if v == "EMERGENCY": prior_er += 1
                    if v == "INPATIENT": prior_ip += 1
            for sd, paid in claims_by_m.get(mid, []):
                sdt = _d(sd)
                if sdt and 0 < (disch - sdt).days <= 180: prior_cost += paid

            rows.append({
                "age": age,
                "chronic_conditions": float(s["chr"]),
                "LOS": float(s["LOS"]),
                "prior_er_180d": prior_er,
                "prior_ip_180d": prior_ip,
                "prior_cost_180d": prior_cost,
                "followup_14d": treat,
                "readmit_30d": outcome,
            })

    if len(rows) < 30:
        return {"error": "insufficient_data", "n": len(rows)}

    total_assembled = len(rows)
    rows = _stratified_sample(rows, "followup_14d", max_rows)

    try:
        res = estimate_ate(
            rows,
            outcome="readmit_30d",
            treatment="followup_14d",
            covariates=["age", "chronic_conditions", "LOS",
                        "prior_er_180d", "prior_ip_180d", "prior_cost_180d"],
            method="doubleml",
            binary_outcome=True,
        )
    except Exception as e:
        return {"error": str(e), "n": len(rows)}

    d = res.to_dict()
    d["total_discharges"] = total_assembled
    d["interpretation"] = (
        f"Post-discharge follow-up within 14 days changes 30-day readmission "
        f"probability by {res.ate*100:+.2f} percentage points "
        f"(95% CI {res.ate_ci95[0]*100:+.2f} .. {res.ate_ci95[1]*100:+.2f} pp). "
        f"Based on {res.n} discharges ({total_assembled} total population), "
        f"{res.n_treated} with follow-up."
    )
    return d


def estimate_telehealth_effect(db_path: str,
                                max_rows: int = 50000) -> Dict[str, Any]:
    _mem_limit = max(max_rows * 2, 30000)
    with _conn(db_path) as c:
        members: Dict[str, Dict[str, Any]] = {}
        for r in c.execute("""
            SELECT MEMBER_ID,
                   CAST(COALESCE((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH)) / 365.25, 40) AS INTEGER) AS member_age,
                   GENDER,
                   COALESCE(CHRONIC_CONDITIONS,0) AS chr FROM members
            ORDER BY RANDOM() LIMIT ?""", (_mem_limit,)):
            members[r["MEMBER_ID"]] = {
                "age": float(r["member_age"]),
                "gender": (r["GENDER"] or "").upper(),
                "chronic": float(r["chr"]),
            }
        for r in c.execute(
            "SELECT MEMBER_ID, COALESCE(SUM(PAID_AMOUNT),0) cost, COUNT(*) cnt "
            "FROM claims GROUP BY MEMBER_ID"):
            if r["MEMBER_ID"] in members:
                members[r["MEMBER_ID"]]["cost"] = float(r["cost"])
                members[r["MEMBER_ID"]]["claims"] = float(r["cnt"])
        for r in c.execute("""
            SELECT MEMBER_ID,
                   SUM(CASE WHEN VISIT_TYPE='TELEHEALTH' THEN 1 ELSE 0 END) th,
                   SUM(CASE WHEN VISIT_TYPE='EMERGENCY'  THEN 1 ELSE 0 END) er,
                   SUM(CASE WHEN VISIT_TYPE='INPATIENT'  THEN 1 ELSE 0 END) ip
            FROM encounters GROUP BY MEMBER_ID"""):
            if r["MEMBER_ID"] in members:
                m = members[r["MEMBER_ID"]]
                m["telehealth"] = 1 if (r["th"] or 0) > 0 else 0
                m["er"] = float(r["er"] or 0)
                m["ip"] = float(r["ip"] or 0)
        for r in c.execute("""
            SELECT MEMBER_ID, COUNT(DISTINCT PRIMARY_DIAGNOSIS) ud FROM encounters
            GROUP BY MEMBER_ID"""):
            if r["MEMBER_ID"] in members:
                members[r["MEMBER_ID"]]["unique_dx"] = float(r["ud"])

    rows = []
    for mid, m in members.items():
        if "telehealth" not in m or "cost" not in m:
            continue
        rows.append({
            "age": m.get("age", 40.0),
            "is_female": 1.0 if m["gender"].startswith("F") else 0.0,
            "chronic_conditions": m["chronic"],
            "er_count": m.get("er", 0.0),
            "ip_count": m.get("ip", 0.0),
            "unique_dx": m.get("unique_dx", 0.0),
            "telehealth": m.get("telehealth", 0),
            "total_paid": m["cost"],
        })

    if len(rows) < 30:
        return {"error": "insufficient_data", "n": len(rows)}

    total_assembled = len(rows)
    rows = _stratified_sample(rows, "telehealth", max_rows)

    try:
        res = estimate_ate(
            rows,
            outcome="total_paid",
            treatment="telehealth",
            covariates=["age", "is_female", "chronic_conditions",
                        "er_count", "ip_count", "unique_dx"],
            method="doubleml",
            binary_outcome=False,
        )
    except Exception as e:
        return {"error": str(e), "n": len(rows)}
    d = res.to_dict()
    d["interpretation"] = (
        f"Any telehealth usage changes total annual paid cost by "
        f"${res.ate:,.0f} (95% CI ${res.ate_ci95[0]:,.0f} .. "
        f"${res.ate_ci95[1]:,.0f}) after controlling for age, sex, "
        f"chronic conditions, prior ER/IP utilization, and diagnosis breadth."
    )
    return d
