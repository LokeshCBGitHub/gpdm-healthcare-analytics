from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


DRIFT_LOG_TABLE = "gpdm_drift_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DRIFT_LOG_TABLE} (
            LOG_ID      INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP   TEXT,
            FEATURE     TEXT,
            KIND        TEXT,          -- numeric | categorical | prediction
            PSI         REAL,
            KS_STAT     REAL,
            KS_P        REAL,
            JSD         REAL,
            CHI2_STAT   REAL,
            CHI2_P      REAL,
            VERDICT     TEXT,          -- ok | moderate | significant
            N_REF       INTEGER,
            N_CUR       INTEGER
        )
        """
    )
    conn.commit()


def psi(reference, current, n_bins: int = 10, eps: float = 1e-6) -> float:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(current, dtype=np.float64)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(r, qs))
    if len(edges) < 2:
        return 0.0
    r_hist, _ = np.histogram(r, bins=edges)
    c_hist, _ = np.histogram(c, bins=edges)
    r_pct = r_hist / max(r_hist.sum(), 1) + eps
    c_pct = c_hist / max(c_hist.sum(), 1) + eps
    return float(((c_pct - r_pct) * np.log(c_pct / r_pct)).sum())


def ks_two_sample(reference, current) -> Dict:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(current, dtype=np.float64)
    if len(r) == 0 or len(c) == 0:
        return {"stat": 0.0, "p_value": 1.0, "n_ref": len(r), "n_cur": len(c)}
    if _SCIPY_OK:
        res = _scipy_stats.ks_2samp(r, c)
        return {"stat": float(res.statistic), "p_value": float(res.pvalue),
                "n_ref": len(r), "n_cur": len(c)}
    all_vals = np.concatenate([r, c])
    all_vals.sort()
    cdf_r = np.searchsorted(np.sort(r), all_vals, side="right") / len(r)
    cdf_c = np.searchsorted(np.sort(c), all_vals, side="right") / len(c)
    stat = float(np.max(np.abs(cdf_r - cdf_c)))
    n_eff = len(r) * len(c) / (len(r) + len(c))
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * stat
    p = 2.0 * math.exp(-2.0 * lam * lam)
    p = min(max(p, 0.0), 1.0)
    return {"stat": stat, "p_value": p, "n_ref": len(r), "n_cur": len(c)}


def chi_square(reference, current) -> Dict:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    r = np.asarray(list(reference))
    c = np.asarray(list(current))
    cats = sorted(set(r.tolist()) | set(c.tolist()))
    r_counts = np.array([int((r == k).sum()) for k in cats], dtype=np.float64)
    c_counts = np.array([int((c == k).sum()) for k in cats], dtype=np.float64)
    total = r_counts.sum() + c_counts.sum()
    if total == 0:
        return {"stat": 0.0, "p_value": 1.0, "df": 0}
    row_tot = np.array([r_counts.sum(), c_counts.sum()])
    col_tot = r_counts + c_counts
    expected_r = row_tot[0] * col_tot / total
    expected_c = row_tot[1] * col_tot / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_r = np.where(expected_r > 0, (r_counts - expected_r) ** 2 / expected_r, 0)
        chi_c = np.where(expected_c > 0, (c_counts - expected_c) ** 2 / expected_c, 0)
    stat = float(chi_r.sum() + chi_c.sum())
    df = max(len(cats) - 1, 1)
    if _SCIPY_OK:
        p = float(1.0 - _scipy_stats.chi2.cdf(stat, df))
    else:
        x = stat / df
        z = (math.pow(x, 1.0 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = float(0.5 * math.erfc(z / math.sqrt(2)))
    return {"stat": stat, "p_value": p, "df": df,
            "n_ref": int(row_tot[0]), "n_cur": int(row_tot[1])}


def jensen_shannon(reference, current, n_bins: int = 20, eps: float = 1e-12) -> float:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(current, dtype=np.float64)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    lo = float(min(r.min(), c.min()))
    hi = float(max(r.max(), c.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pr, _ = np.histogram(r, bins=edges, density=False)
    pc, _ = np.histogram(c, bins=edges, density=False)
    pr = pr / max(pr.sum(), 1) + eps
    pc = pc / max(pc.sum(), 1) + eps
    m = 0.5 * (pr + pc)
    kl_r = (pr * np.log(pr / m)).sum()
    kl_c = (pc * np.log(pc / m)).sum()
    return float(0.5 * (kl_r + kl_c))


def _verdict_from_psi(p: float) -> str:
    if p < 0.10:
        return "ok"
    if p < 0.25:
        return "moderate"
    return "significant"


def feature_drift_report(
    ref: Dict[str, Sequence],
    cur: Dict[str, Sequence],
    feature_types: Dict[str, str],
) -> List[Dict]:
    rows = []
    for feat, kind in feature_types.items():
        r = ref.get(feat, [])
        c = cur.get(feat, [])
        if kind == "numeric":
            p = psi(r, c)
            ks = ks_two_sample(r, c)
            js = jensen_shannon(r, c)
            rows.append({
                "feature": feat, "kind": "numeric",
                "psi": p, "ks_stat": ks["stat"], "ks_p": ks["p_value"],
                "jsd": js, "n_ref": ks["n_ref"], "n_cur": ks["n_cur"],
                "verdict": _verdict_from_psi(p),
            })
        else:
            ch = chi_square(r, c)
            rows.append({
                "feature": feat, "kind": "categorical",
                "chi2_stat": ch["stat"], "chi2_p": ch["p_value"],
                "df": ch["df"],
                "n_ref": ch["n_ref"], "n_cur": ch["n_cur"],
                "verdict": "significant" if ch["p_value"] < 0.01
                           else ("moderate" if ch["p_value"] < 0.05 else "ok"),
            })
    return rows


DEFAULT_NUMERIC_FEATURES = ["age", "total_paid_12m", "n_encounters_12m",
                            "n_admits_12m", "n_er_12m", "los_days"]
DEFAULT_CATEGORICAL_FEATURES = ["gender", "region", "chronic_flag"]


def monitor_and_log(
    db_path: str,
    ref_window_days: int = 365,
    cur_window_days: int = 30,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    psi_threshold: float = 0.25,
    pred_psi_threshold: float = 0.20,
) -> Dict:
    if not _NP_OK:
        return {"status": "numpy_unavailable"}

    numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
    categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES

    conn = sqlite3.connect(db_path)
    _ensure_log_table(conn)
    cur = conn.cursor()

    def _fetch(col, days):
        try:
            cur.execute(
                f"""SELECT {col} FROM members m
                    LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID
                    WHERE e.ADMIT_DATE >= date('now', ?)
                    AND {col} IS NOT NULL""",
                (f"-{days} days",),
            )
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return []

    ref: Dict[str, List] = {}
    cur_d: Dict[str, List] = {}
    types: Dict[str, str] = {}
    for f in numeric_features:
        rv = _fetch(f, ref_window_days)
        cv = _fetch(f, cur_window_days)
        if rv and cv:
            ref[f] = rv
            cur_d[f] = cv
            types[f] = "numeric"
    for f in categorical_features:
        rv = _fetch(f, ref_window_days)
        cv = _fetch(f, cur_window_days)
        if rv and cv:
            ref[f] = rv
            cur_d[f] = cv
            types[f] = "categorical"

    report = feature_drift_report(ref, cur_d, types)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for row in report:
        conn.execute(
            f"""INSERT INTO {DRIFT_LOG_TABLE}
                (TIMESTAMP, FEATURE, KIND, PSI, KS_STAT, KS_P, JSD, CHI2_STAT, CHI2_P,
                 VERDICT, N_REF, N_CUR)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, row.get("feature"), row.get("kind"),
             row.get("psi"), row.get("ks_stat"), row.get("ks_p"),
             row.get("jsd"), row.get("chi2_stat"), row.get("chi2_p"),
             row.get("verdict"), row.get("n_ref"), row.get("n_cur")),
        )

    pred_row = None
    try:
        cur.execute(
            """SELECT probability FROM gpdm_predictions
               WHERE timestamp >= date('now', ?)""",
            (f"-{cur_window_days} days",),
        )
        cur_preds = [r[0] for r in cur.fetchall() if r[0] is not None]
        cur.execute(
            """SELECT probability FROM gpdm_predictions
               WHERE timestamp >= date('now', ?)
                 AND timestamp <  date('now', ?)""",
            (f"-{ref_window_days} days", f"-{cur_window_days} days"),
        )
        ref_preds = [r[0] for r in cur.fetchall() if r[0] is not None]
        if len(ref_preds) > 20 and len(cur_preds) > 20:
            p_psi = psi(ref_preds, cur_preds)
            p_ks = ks_two_sample(ref_preds, cur_preds)
            pred_row = {
                "feature": "__prediction__", "kind": "prediction",
                "psi": p_psi, "ks_stat": p_ks["stat"], "ks_p": p_ks["p_value"],
                "verdict": _verdict_from_psi(p_psi),
                "n_ref": len(ref_preds), "n_cur": len(cur_preds),
            }
            conn.execute(
                f"""INSERT INTO {DRIFT_LOG_TABLE}
                    (TIMESTAMP, FEATURE, KIND, PSI, KS_STAT, KS_P, VERDICT, N_REF, N_CUR)
                    VALUES (?, '__prediction__', 'prediction', ?, ?, ?, ?, ?, ?)""",
                (ts, p_psi, p_ks["stat"], p_ks["p_value"], pred_row["verdict"],
                 len(ref_preds), len(cur_preds)),
            )
    except Exception:
        pass

    conn.commit()
    conn.close()

    significant = [r for r in report if r["verdict"] == "significant"]
    pred_sig = (pred_row and pred_row["psi"] >= pred_psi_threshold)
    should_retrain = bool(significant) or bool(pred_sig)

    return {
        "status": "ok",
        "ts": ts,
        "n_features_monitored": len(report),
        "n_significant": len(significant),
        "report": report,
        "prediction_drift": pred_row,
        "retrain_recommended": should_retrain,
        "reason": (
            [f"{r['feature']} PSI={r.get('psi'):.3f}" for r in significant]
            + ([f"prediction PSI={pred_row['psi']:.3f}"] if pred_sig else [])
        ),
    }


def should_retrain(db_path: str, psi_threshold: float = 0.25) -> Dict:
    out = monitor_and_log(db_path, psi_threshold=psi_threshold)
    return {
        "retrain": out.get("retrain_recommended", False),
        "reasons": out.get("reason", []),
        "n_significant": out.get("n_significant", 0),
    }


def get_drift_status(db_path: str) -> Dict:
    out: Dict = {"recent": []}
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, FEATURE, KIND, PSI, KS_STAT, KS_P, JSD, VERDICT,
                       N_REF, N_CUR
                FROM {DRIFT_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 50"""
        )
        out["recent"] = [
            {"ts": r[0], "feature": r[1], "kind": r[2], "psi": r[3],
             "ks_stat": r[4], "ks_p": r[5], "jsd": r[6], "verdict": r[7],
             "n_ref": r[8], "n_cur": r[9]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
