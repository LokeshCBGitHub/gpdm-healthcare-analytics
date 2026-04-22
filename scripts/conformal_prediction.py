from __future__ import annotations

import math
import os
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
CONFORMAL_DIR = MODELS_DIR / "conformal"
CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)

CONFORMAL_LOG_TABLE = "gpdm_conformal_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CONFORMAL_LOG_TABLE} (
            LOG_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP     TEXT NOT NULL,
            PREDICTOR     TEXT,
            METHOD        TEXT,
            ALPHA         REAL,
            N_CAL         INTEGER,
            Q_HAT         REAL,
            COVERAGE_TEST REAL,
            GROUP_KEY     TEXT,
            NOTES         TEXT
        )
        """
    )
    conn.commit()


@dataclass
class BinaryConformal:
    alpha: float
    q_hat: float
    n_cal: int
    nonconformity: str = "absolute"
    fitted_at: float = field(default_factory=time.time)


def _quantile_correction(n: int, alpha: float) -> float:
    return math.ceil((n + 1) * (1.0 - alpha)) / float(n)


def fit_split_conformal_binary(
    p_cal: "np.ndarray",
    y_cal: "np.ndarray",
    alpha: float = 0.1,
    nonconformity: str = "absolute",
) -> BinaryConformal:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    p = np.clip(np.asarray(p_cal, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(y_cal, dtype=np.float64)
    if nonconformity == "absolute":
        s = np.abs(y - p)
    elif nonconformity == "log_loss":
        s = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    else:
        raise ValueError(f"unknown nonconformity: {nonconformity}")
    n = len(s)
    level = min(_quantile_correction(n, alpha), 1.0)
    q_hat = float(np.quantile(s, level, method="higher"))
    return BinaryConformal(alpha=alpha, q_hat=q_hat, n_cal=n, nonconformity=nonconformity)


def predict_interval_binary(p_hat: float, cal: BinaryConformal) -> Tuple[float, float]:
    if cal.nonconformity == "absolute":
        lo = max(0.0, p_hat - cal.q_hat)
        hi = min(1.0, p_hat + cal.q_hat)
    else:
        pmin = math.exp(-cal.q_hat)
        lo = max(0.0, p_hat - (1.0 - pmin))
        hi = min(1.0, p_hat + (1.0 - pmin))
    return lo, hi


@dataclass
class MondrianConformal:
    alpha: float
    by_group: Dict[str, BinaryConformal] = field(default_factory=dict)
    global_fallback: Optional[BinaryConformal] = None
    min_group_size: int = 30
    fitted_at: float = field(default_factory=time.time)


def fit_mondrian_conformal(
    p_cal: "np.ndarray",
    y_cal: "np.ndarray",
    groups_cal: Sequence[str],
    alpha: float = 0.1,
    nonconformity: str = "absolute",
    min_group_size: int = 30,
) -> MondrianConformal:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    p = np.asarray(p_cal)
    y = np.asarray(y_cal)
    g = np.asarray(list(groups_cal))

    result = MondrianConformal(alpha=alpha, min_group_size=min_group_size)
    result.global_fallback = fit_split_conformal_binary(p, y, alpha, nonconformity)

    for gk in np.unique(g):
        mask = g == gk
        if mask.sum() >= min_group_size:
            result.by_group[str(gk)] = fit_split_conformal_binary(
                p[mask], y[mask], alpha, nonconformity
            )
    return result


def predict_mondrian(p_hat: float, group: str, cal: MondrianConformal) -> Tuple[float, float]:
    sub = cal.by_group.get(str(group), cal.global_fallback)
    if sub is None:
        return 0.0, 1.0
    return predict_interval_binary(p_hat, sub)


@dataclass
class APSCalibrator:
    alpha: float
    q_hat: float
    n_cal: int
    randomize: bool = True
    fitted_at: float = field(default_factory=time.time)


def fit_aps(
    probs_cal: "np.ndarray",
    y_cal: "np.ndarray",
    alpha: float = 0.1,
    randomize: bool = True,
) -> APSCalibrator:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    P = np.asarray(probs_cal, dtype=np.float64)
    y = np.asarray(y_cal, dtype=int)
    n, K = P.shape
    order = np.argsort(-P, axis=1)
    ranks = np.argsort(order, axis=1)
    sorted_probs = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)

    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = ranks[i, y[i]]
        above = cum[i, r]
        if randomize:
            u = np.random.uniform(0, 1)
            s[i] = above - u * sorted_probs[i, r]
        else:
            s[i] = above
    level = min(_quantile_correction(n, alpha), 1.0)
    q_hat = float(np.quantile(s, level, method="higher"))
    return APSCalibrator(alpha=alpha, q_hat=q_hat, n_cal=n, randomize=randomize)


def predict_aps(probs: "np.ndarray", cal: APSCalibrator) -> List[List[int]]:
    P = np.asarray(probs, dtype=np.float64)
    n, K = P.shape
    order = np.argsort(-P, axis=1)
    sorted_probs = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    sets: List[List[int]] = []
    for i in range(n):
        keep = cum[i] <= cal.q_hat
        if not keep.any():
            keep[0] = True
        first_over = np.searchsorted(cum[i], cal.q_hat, side="left")
        if first_over < K:
            keep[first_over] = True
        sets.append(order[i, keep].tolist())
    return sets


@dataclass
class CQRAdjustment:
    alpha: float
    delta: float
    n_cal: int
    fitted_at: float = field(default_factory=time.time)


def fit_cqr(
    q_lo_cal: "np.ndarray",
    q_hi_cal: "np.ndarray",
    y_cal: "np.ndarray",
    alpha: float = 0.1,
) -> CQRAdjustment:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    ql = np.asarray(q_lo_cal, dtype=np.float64)
    qh = np.asarray(q_hi_cal, dtype=np.float64)
    y = np.asarray(y_cal, dtype=np.float64)
    E = np.maximum(ql - y, y - qh)
    n = len(E)
    level = min(_quantile_correction(n, alpha), 1.0)
    delta = float(np.quantile(E, level, method="higher"))
    return CQRAdjustment(alpha=alpha, delta=delta, n_cal=n)


def predict_cqr(q_lo: float, q_hi: float, adj: CQRAdjustment) -> Tuple[float, float]:
    return (q_lo - adj.delta, q_hi + adj.delta)


def empirical_coverage(intervals: Sequence[Tuple[float, float]], truths: Sequence[float]) -> Dict:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    lo = np.array([iv[0] for iv in intervals])
    hi = np.array([iv[1] for iv in intervals])
    y = np.array(truths, dtype=np.float64)
    covered = (y >= lo) & (y <= hi)
    widths = hi - lo
    return {
        "n": int(len(y)),
        "coverage": float(covered.mean()) if len(y) else None,
        "mean_width": float(widths.mean()) if len(y) else None,
        "median_width": float(np.median(widths)) if len(y) else None,
    }


def save_calibrator(name: str, cal) -> Path:
    path = CONFORMAL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(cal, f)
    return path


def load_calibrator(name: str):
    path = CONFORMAL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def fit_risk_conformal_from_sqlite(
    db_path: str,
    alpha: float = 0.1,
    holdout_frac: float = 0.3,
    subgroup_col: Optional[str] = "AGE_BAND",
) -> Dict:
    if not _NP_OK:
        return {"status": "numpy_unavailable"}
    try:
        try:
            from . import ml_models as mm
        except Exception:
            import ml_models as mm
    except Exception as e:
        return {"status": "ml_models_unavailable", "error": str(e)}

    if not hasattr(mm, "build_scoring_frame"):
        return {"status": "no_scoring_helper"}
    try:
        X, y, groups = mm.build_scoring_frame(db_path, target="risk_high_cost",
                                              subgroup=subgroup_col)
        preds = mm.predict_risk_batch(X)
    except Exception as e:
        return {"status": "scoring_failed", "error": str(e)}

    p = np.asarray(preds, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(groups)

    rng = np.random.default_rng(42)
    idx = np.arange(len(p))
    rng.shuffle(idx)
    n_cal = int(len(idx) * (1 - holdout_frac))
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    marginal = fit_split_conformal_binary(p[cal_idx], y[cal_idx], alpha=alpha)
    mondrian = fit_mondrian_conformal(p[cal_idx], y[cal_idx], g[cal_idx], alpha=alpha)

    marg_intervals = [predict_interval_binary(float(pi), marginal) for pi in p[test_idx]]
    mond_intervals = [predict_mondrian(float(p[i]), str(g[i]), mondrian) for i in test_idx]
    cov_marg = empirical_coverage(marg_intervals, y[test_idx])
    cov_mond = empirical_coverage(mond_intervals, y[test_idx])

    save_calibrator("risk_marginal", marginal)
    save_calibrator("risk_mondrian", mondrian)

    conn = sqlite3.connect(db_path)
    _ensure_log_table(conn)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        f"""INSERT INTO {CONFORMAL_LOG_TABLE}
            (TIMESTAMP, PREDICTOR, METHOD, ALPHA, N_CAL, Q_HAT, COVERAGE_TEST, GROUP_KEY, NOTES)
            VALUES (?, 'risk_stratifier', 'split_marginal', ?, ?, ?, ?, NULL, ?)""",
        (ts, alpha, marginal.n_cal, marginal.q_hat, cov_marg["coverage"],
         f"mean_width={cov_marg['mean_width']:.3f}"),
    )
    for gk, sub in mondrian.by_group.items():
        conn.execute(
            f"""INSERT INTO {CONFORMAL_LOG_TABLE}
                (TIMESTAMP, PREDICTOR, METHOD, ALPHA, N_CAL, Q_HAT, COVERAGE_TEST, GROUP_KEY, NOTES)
                VALUES (?, 'risk_stratifier', 'split_mondrian', ?, ?, ?, NULL, ?, NULL)""",
            (ts, alpha, sub.n_cal, sub.q_hat, gk),
        )
    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "alpha": alpha,
        "marginal": {"q_hat": marginal.q_hat, "n_cal": marginal.n_cal,
                     "test_coverage": cov_marg["coverage"],
                     "mean_width": cov_marg["mean_width"]},
        "mondrian": {
            "n_groups": len(mondrian.by_group),
            "test_coverage": cov_mond["coverage"],
            "mean_width": cov_mond["mean_width"],
        },
    }


def predict_risk_conformal(p_hat: float, group: Optional[str] = None) -> Dict:
    marg = load_calibrator("risk_marginal")
    mond = load_calibrator("risk_mondrian")
    out: Dict = {"p_hat": p_hat}
    if marg is not None:
        lo, hi = predict_interval_binary(p_hat, marg)
        out["marginal"] = {"lo": lo, "hi": hi, "alpha": marg.alpha,
                           "coverage_target": 1 - marg.alpha}
    if mond is not None and group is not None:
        lo, hi = predict_mondrian(p_hat, group, mond)
        out["mondrian"] = {"group": group, "lo": lo, "hi": hi,
                           "alpha": mond.alpha, "coverage_target": 1 - mond.alpha}
    return out


def get_conformal_status(db_path: str) -> Dict:
    out: Dict = {"calibrators_on_disk": [], "recent_fits": []}
    for p in CONFORMAL_DIR.glob("*.pkl") if CONFORMAL_DIR.exists() else []:
        out["calibrators_on_disk"].append({"name": p.stem, "size_kb": p.stat().st_size // 1024})
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, PREDICTOR, METHOD, ALPHA, N_CAL, Q_HAT, COVERAGE_TEST, GROUP_KEY, NOTES
                FROM {CONFORMAL_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 30"""
        )
        out["recent_fits"] = [
            {"ts": r[0], "predictor": r[1], "method": r[2], "alpha": r[3],
             "n_cal": r[4], "q_hat": r[5], "coverage": r[6], "group": r[7], "notes": r[8]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
