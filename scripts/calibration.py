from __future__ import annotations

import math
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
CAL_DIR = MODELS_DIR / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)

CAL_LOG_TABLE = "gpdm_calibration_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CAL_LOG_TABLE} (
            LOG_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP  TEXT,
            PREDICTOR  TEXT,
            METHOD     TEXT,
            N          INTEGER,
            ECE_BEFORE REAL,
            ECE_AFTER  REAL,
            MCE_BEFORE REAL,
            MCE_AFTER  REAL,
            BRIER_BEFORE REAL,
            BRIER_AFTER REAL,
            CHECKPOINT TEXT
        )
        """
    )
    conn.commit()


@dataclass
class PlattCalibrator:
    a: float
    b: float
    n: int
    fitted_at: float = field(default_factory=time.time)


def fit_platt(scores, labels, max_iter: int = 200, lr: float = 0.1) -> PlattCalibrator:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, len(y) - n_pos)
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    a, b = -1.0, 0.0
    for _ in range(max_iter):
        z = a * s + b
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad_a = ((p - t) * s).mean()
        grad_b = (p - t).mean()
        a -= lr * grad_a
        b -= lr * grad_b
    return PlattCalibrator(a=float(a), b=float(b), n=len(s))


def apply_platt(scores, cal: PlattCalibrator):
    if not _NP_OK:
        raise RuntimeError("numpy required")
    s = np.asarray(scores, dtype=np.float64)
    z = np.clip(cal.a * s + cal.b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class IsotonicCalibrator:
    x_sorted: "np.ndarray"
    y_fit: "np.ndarray"
    n: int
    fitted_at: float = field(default_factory=time.time)


def _pav(y, w):
    n = len(y)
    y = y.astype(np.float64).copy()
    w = w.astype(np.float64).copy()
    active = list(range(n))
    vals = y.copy()
    weights = w.copy()
    i = 0
    while i < len(active) - 1:
        if vals[i] > vals[i + 1]:
            new_w = weights[i] + weights[i + 1]
            new_v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / new_w
            vals = np.concatenate([vals[:i], [new_v], vals[i + 2:]])
            weights = np.concatenate([weights[:i], [new_w], weights[i + 2:]])
            counts_map_prev = None
            if i > 0:
                i -= 1
        else:
            i += 1
    return vals, weights


def fit_isotonic(scores, labels) -> IsotonicCalibrator:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    w = np.ones_like(y_sorted)
    n = len(y_sorted)
    fit = y_sorted.copy()
    weights = w.copy()
    i = 0
    while i < n - 1:
        if fit[i] > fit[i + 1]:
            new_w = weights[i] + weights[i + 1]
            new_v = (fit[i] * weights[i] + fit[i + 1] * weights[i + 1]) / new_w
            fit[i] = new_v
            fit[i + 1] = new_v
            weights[i] = new_w
            weights[i + 1] = new_w
            j = i
            while j > 0 and fit[j - 1] > fit[j]:
                new_w = weights[j - 1] + weights[j]
                new_v = (fit[j - 1] * weights[j - 1] + fit[j] * weights[j]) / new_w
                fit[j - 1] = new_v
                fit[j] = new_v
                weights[j - 1] = new_w
                weights[j] = new_w
                j -= 1
        i += 1
    return IsotonicCalibrator(x_sorted=s_sorted, y_fit=fit, n=n)


def apply_isotonic(scores, cal: IsotonicCalibrator):
    if not _NP_OK:
        raise RuntimeError("numpy required")
    s = np.asarray(scores, dtype=np.float64)
    out = np.interp(s, cal.x_sorted, cal.y_fit,
                    left=cal.y_fit[0], right=cal.y_fit[-1])
    return np.clip(out, 0.0, 1.0)


def fit_temperature(logits, labels, t_min: float = 0.05, t_max: float = 5.0,
                    n_search: int = 100) -> float:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    ts = np.linspace(t_min, t_max, n_search)
    best_t, best_ll = 1.0, -np.inf
    for t in ts:
        if z.ndim == 1:
            p = 1.0 / (1.0 + np.exp(-z / t))
            ll = (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum()
        else:
            zt = z / t
            zt -= zt.max(axis=1, keepdims=True)
            e = np.exp(zt)
            p = e / e.sum(axis=1, keepdims=True)
            ll = np.log(p[np.arange(len(y)), y] + 1e-12).sum()
        if ll > best_ll:
            best_ll = ll
            best_t = float(t)
    return best_t


def apply_temperature(logits, T: float):
    if not _NP_OK:
        raise RuntimeError("numpy required")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        return 1.0 / (1.0 + np.exp(-z / T))
    zt = z / T
    zt -= zt.max(axis=1, keepdims=True)
    e = np.exp(zt)
    return e / e.sum(axis=1, keepdims=True)


def _bin_stats(probs, labels, n_bins: int = 15):
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            out.append({"bin": i, "lo": lo, "hi": hi, "n": 0,
                        "avg_p": None, "avg_y": None})
            continue
        out.append({
            "bin": i, "lo": float(lo), "hi": float(hi),
            "n": int(mask.sum()),
            "avg_p": float(p[mask].mean()),
            "avg_y": float(y[mask].mean()),
        })
    return out


def ece(probs, labels, n_bins: int = 15) -> float:
    if not _NP_OK:
        return float("nan")
    stats = _bin_stats(probs, labels, n_bins)
    n_total = sum(s["n"] for s in stats)
    if n_total == 0:
        return float("nan")
    return float(sum(
        (s["n"] / n_total) * abs(s["avg_p"] - s["avg_y"])
        for s in stats if s["n"] > 0
    ))


def mce(probs, labels, n_bins: int = 15) -> float:
    if not _NP_OK:
        return float("nan")
    stats = _bin_stats(probs, labels, n_bins)
    diffs = [abs(s["avg_p"] - s["avg_y"]) for s in stats if s["n"] > 0]
    return float(max(diffs)) if diffs else float("nan")


def brier(probs, labels) -> float:
    if not _NP_OK:
        return float("nan")
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(((p - y) ** 2).mean())


def reliability_bins(probs, labels, n_bins: int = 15) -> List[Dict]:
    return _bin_stats(probs, labels, n_bins)


def save_calibrator(name: str, cal) -> Path:
    path = CAL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(cal, f)
    return path


def load_calibrator(name: str):
    path = CAL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrate_risk_from_sqlite(
    db_path: str,
    method: str = "isotonic",
    holdout_frac: float = 0.3,
) -> Dict:
    if not _NP_OK:
        return {"status": "numpy_unavailable"}
    try:
        try:
            from . import ml_models as mm
        except Exception:
            import ml_models as mm
        if not hasattr(mm, "build_scoring_frame"):
            return {"status": "no_scoring_helper"}
        X, y, _ = mm.build_scoring_frame(db_path, target="risk_high_cost")
        preds = np.asarray(mm.predict_risk_batch(X), dtype=np.float64)
    except Exception as e:
        return {"status": "scoring_failed", "error": str(e)}

    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(42)
    idx = np.arange(len(preds))
    rng.shuffle(idx)
    n_cal = int(len(idx) * (1 - holdout_frac))
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    p_cal, y_cal = preds[cal_idx], y[cal_idx]
    p_test, y_test = preds[test_idx], y[test_idx]

    ece_before = ece(p_test, y_test)
    mce_before = mce(p_test, y_test)
    br_before = brier(p_test, y_test)

    if method == "platt":
        cal = fit_platt(p_cal, y_cal)
        p_cal_after = apply_platt(p_test, cal)
    elif method == "isotonic":
        cal = fit_isotonic(p_cal, y_cal)
        p_cal_after = apply_isotonic(p_test, cal)
    elif method == "temperature":
        logits = np.log(np.clip(p_cal, 1e-9, 1 - 1e-9) / (1 - np.clip(p_cal, 1e-9, 1 - 1e-9)))
        T = fit_temperature(logits, y_cal.astype(int))
        test_logits = np.log(np.clip(p_test, 1e-9, 1 - 1e-9) / (1 - np.clip(p_test, 1e-9, 1 - 1e-9)))
        p_cal_after = apply_temperature(test_logits, T)
        cal = {"temperature": T, "n": len(p_cal)}
    else:
        return {"status": "bad_method", "method": method}

    ece_after = ece(p_cal_after, y_test)
    mce_after = mce(p_cal_after, y_test)
    br_after = brier(p_cal_after, y_test)

    ckpt = save_calibrator(f"risk_{method}", cal)

    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        conn.execute(
            f"""INSERT INTO {CAL_LOG_TABLE}
                (TIMESTAMP, PREDICTOR, METHOD, N,
                 ECE_BEFORE, ECE_AFTER, MCE_BEFORE, MCE_AFTER,
                 BRIER_BEFORE, BRIER_AFTER, CHECKPOINT)
                VALUES (?, 'risk_stratifier', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.strftime("%Y-%m-%d %H:%M:%S"), method, len(preds),
             ece_before, ece_after, mce_before, mce_after,
             br_before, br_after, str(ckpt)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {
        "status": "ok",
        "method": method,
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "ece_before": ece_before, "ece_after": ece_after,
        "mce_before": mce_before, "mce_after": mce_after,
        "brier_before": br_before, "brier_after": br_after,
        "checkpoint": str(ckpt),
        "reliability_before": reliability_bins(p_test, y_test, 10),
        "reliability_after": reliability_bins(p_cal_after, y_test, 10),
    }


def get_calibration_status(db_path: str) -> Dict:
    out: Dict = {"calibrators_on_disk": [], "recent_fits": []}
    for p in CAL_DIR.glob("*.pkl") if CAL_DIR.exists() else []:
        out["calibrators_on_disk"].append({"name": p.stem, "size_kb": p.stat().st_size // 1024})
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, PREDICTOR, METHOD, N,
                       ECE_BEFORE, ECE_AFTER, BRIER_BEFORE, BRIER_AFTER, CHECKPOINT
                FROM {CAL_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 20"""
        )
        out["recent_fits"] = [
            {"ts": r[0], "predictor": r[1], "method": r[2], "n": r[3],
             "ece_before": r[4], "ece_after": r[5],
             "brier_before": r[6], "brier_after": r[7], "checkpoint": r[8]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
