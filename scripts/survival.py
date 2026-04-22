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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
SURV_DIR = MODELS_DIR / "survival"
SURV_DIR.mkdir(parents=True, exist_ok=True)

SURV_LOG_TABLE = "gpdm_survival_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SURV_LOG_TABLE} (
            LOG_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP    TEXT,
            MODEL        TEXT,
            OUTCOME      TEXT,
            N            INTEGER,
            N_EVENTS     INTEGER,
            HORIZON_DAYS INTEGER,
            C_INDEX      REAL,
            LL           REAL,
            CHECKPOINT   TEXT,
            NOTES        TEXT
        )
        """
    )
    conn.commit()


@dataclass
class CoxModel:
    beta: "np.ndarray"
    feature_names: List[str]
    baseline_times: "np.ndarray"
    baseline_cum_hazard: "np.ndarray"
    c_index: float
    ll: float
    fitted_at: float = field(default_factory=time.time)


def _cox_loglik(beta, X, T, E):
    order = np.argsort(-T)
    Xo, To, Eo = X[order], T[order], E[order]
    eta = Xo @ beta
    exp_eta = np.exp(eta - eta.max())

    ll = 0.0
    n = len(To)
    order_a = np.argsort(To)
    Ta, Ea, eXa = To[order_a][::-1], Eo[order_a][::-1], exp_eta[order_a][::-1]
    etaA = eta[order_a][::-1]
    asc_order = np.argsort(T)
    Tasc, Easc, etasc = T[asc_order], E[asc_order], (X @ beta)[asc_order]
    exp_etasc = np.exp(etasc - etasc.max())
    cum_rev = np.flip(np.cumsum(np.flip(exp_etasc)))

    i = 0
    while i < n:
        t_i = Tasc[i]
        j = i
        while j < n and Tasc[j] == t_i:
            j += 1
        idx = np.arange(i, j)
        d = int(Easc[idx].sum())
        if d > 0:
            sum_eta_events = etasc[idx][Easc[idx] == 1].sum()
            risk = cum_rev[i]
            tied_risk = exp_etasc[idx][Easc[idx] == 1].sum()
            sub = 0.0
            for k in range(d):
                denom = risk - (k / d) * tied_risk
                denom = max(denom, 1e-12)
                sub += math.log(denom)
            ll += sum_eta_events - sub
        i = j
    return float(ll)


def fit_cox(
    X: "np.ndarray",
    T: "np.ndarray",
    E: "np.ndarray",
    feature_names: Optional[List[str]] = None,
    lr: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-6,
    l2: float = 1e-3,
) -> CoxModel:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    prev = -np.inf
    for it in range(max_iter):
        eta = X @ beta
        exp_eta = np.exp(eta - eta.max())
        order = np.argsort(T)
        Xo, To, Eo, eXo = X[order], T[order], E[order], exp_eta[order]
        cum_exp = np.flip(np.cumsum(np.flip(eXo)))
        cum_xexp = np.flip(np.cumsum(np.flip(Xo * eXo[:, None]), axis=0))
        grad = np.zeros(p)
        for i in range(n):
            if Eo[i] == 1:
                mean_x = cum_xexp[i] / max(cum_exp[i], 1e-12)
                grad += Xo[i] - mean_x
        grad -= l2 * beta
        beta += lr * grad / max(n, 1)

        if it % 10 == 0:
            ll = _cox_loglik(beta, X, T, E) - 0.5 * l2 * (beta @ beta)
            if abs(ll - prev) < tol:
                break
            prev = ll
            lr = lr * 0.9

    eta = X @ beta
    exp_eta = np.exp(eta - eta.max())
    order = np.argsort(T)
    Tasc, Easc, eXasc = T[order], E[order], exp_eta[order]
    cum_exp = np.flip(np.cumsum(np.flip(eXasc)))
    baseline_times = []
    baseline_cum_hazard = []
    H = 0.0
    i = 0
    while i < n:
        t_i = Tasc[i]
        j = i
        while j < n and Tasc[j] == t_i:
            j += 1
        idx = np.arange(i, j)
        d = int(Easc[idx].sum())
        if d > 0:
            H += d / max(cum_exp[i], 1e-12)
            baseline_times.append(float(t_i))
            baseline_cum_hazard.append(float(H))
        i = j

    ll_final = _cox_loglik(beta, X, T, E) - 0.5 * l2 * (beta @ beta)
    risk_scores = X @ beta
    c = concordance_index(T, E, risk_scores)

    names = feature_names or [f"x{k}" for k in range(p)]
    return CoxModel(
        beta=beta,
        feature_names=names,
        baseline_times=np.asarray(baseline_times),
        baseline_cum_hazard=np.asarray(baseline_cum_hazard),
        c_index=c,
        ll=ll_final,
    )


def cox_survival_curve(model: CoxModel, x: "np.ndarray", times: Sequence[float]) -> "np.ndarray":
    eta = float(np.asarray(x) @ model.beta)
    hr = math.exp(eta)
    out = np.ones(len(times))
    for i, t in enumerate(times):
        mask = model.baseline_times <= t
        H0 = model.baseline_cum_hazard[mask][-1] if mask.any() else 0.0
        out[i] = math.exp(-H0 * hr)
    return out


def cox_hazard_ratios(model: CoxModel) -> Dict[str, float]:
    return {n: float(math.exp(b)) for n, b in zip(model.feature_names, model.beta)}


if _TORCH_OK:

    class DeepHitNet(nn.Module):
        def __init__(self, in_dim: int, n_bins: int, hidden: int = 128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            )
            self.head = nn.Linear(hidden, n_bins + 1)

        def forward(self, x):
            return F.log_softmax(self.head(self.shared(x)), dim=-1)


def _discretize(T, E, n_bins, horizon):
    bins = np.linspace(0, horizon, n_bins + 1)
    idx = np.digitize(T, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    labels = np.where(E == 1, idx, n_bins)
    return labels.astype(np.int64), bins


def fit_deephit(
    X: "np.ndarray",
    T: "np.ndarray",
    E: "np.ndarray",
    n_bins: int = 20,
    horizon: float = 90.0,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    alpha_rank: float = 0.2,
) -> Dict:
    if not _TORCH_OK or not _NP_OK:
        return {"status": "torch_or_numpy_unavailable"}

    device = torch.device("cpu")
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    E = np.asarray(E, dtype=np.float32)
    y, bins = _discretize(T, E, n_bins, horizon)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    Tt = torch.from_numpy(T)
    Et = torch.from_numpy(E)

    model = DeepHitNet(in_dim=X.shape[1], n_bins=n_bins, hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb, tb, eb = Xt[idx], yt[idx], Tt[idx], Et[idx]
            log_p = model(xb)
            nll = F.nll_loss(log_p, yb)
            probs = log_p.exp()
            event_bins = probs[:, :n_bins]
            cif = torch.cumsum(event_bins, dim=-1)
            rank_loss = torch.tensor(0.0)
            if len(xb) > 1:
                i_idx = torch.arange(len(xb))
                j_idx = torch.randperm(len(xb))
                t_i = tb[i_idx]
                t_j = tb[j_idx]
                e_i = eb[i_idx]
                valid = (t_i < t_j) & (e_i == 1)
                if valid.any():
                    k_i = torch.clamp(yb[i_idx], max=n_bins - 1)
                    cif_i_at_ti = cif[i_idx, k_i]
                    cif_j_at_ti = cif[j_idx, k_i]
                    diff = cif_j_at_ti - cif_i_at_ti
                    rank_loss = torch.exp(diff[valid] / 0.1).mean()
            loss = nll + alpha_rank * rank_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1

    model.eval()
    with torch.no_grad():
        probs = model(Xt).exp().numpy()
    cif_full = np.cumsum(probs[:, :n_bins], axis=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    exp_times = (probs[:, :n_bins] * bin_centers).sum(axis=1)
    risk = -exp_times
    cidx = concordance_index(T, E, risk)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt = SURV_DIR / f"deephit_{ts}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X.shape[1],
        "n_bins": n_bins,
        "horizon": horizon,
        "bins": bins.tolist(),
    }, ckpt)

    return {
        "status": "ok",
        "c_index": float(cidx),
        "n_bins": n_bins,
        "horizon": horizon,
        "checkpoint": str(ckpt),
        "final_loss": ep_loss / max(1, n_batches),
    }


def deephit_survival_curve(checkpoint: str, x: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    if not _TORCH_OK or not _NP_OK:
        raise RuntimeError("torch + numpy required")
    blob = torch.load(checkpoint, map_location="cpu")
    model = DeepHitNet(in_dim=blob["in_dim"], n_bins=blob["n_bins"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = model(xt).exp().numpy()[0]
    n_bins = blob["n_bins"]
    bins = np.asarray(blob["bins"])
    cif = np.cumsum(probs[:n_bins])
    survival = 1.0 - cif
    times = 0.5 * (bins[:-1] + bins[1:])
    return times, survival


def concordance_index(T, E, risk_scores) -> float:
    if not _NP_OK:
        return 0.0
    T = np.asarray(T)
    E = np.asarray(E)
    r = np.asarray(risk_scores)
    n = len(T)
    num = 0
    den = 0
    for i in range(n):
        if E[i] != 1:
            continue
        for j in range(n):
            if T[j] > T[i]:
                den += 1
                if r[i] > r[j]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5
    return float(num) / den if den > 0 else 0.5


def readmission_survival_from_sqlite(
    db_path: str,
    horizon_days: int = 90,
    method: str = "cox",
) -> Dict:
    if not _NP_OK:
        return {"status": "numpy_unavailable"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT MEMBER_ID, ENCOUNTER_ID, ADMIT_DATE, DISCHARGE_DATE,
                   COALESCE(LENGTH_OF_STAY,0) AS LOS_DAYS, VISIT_TYPE
            FROM encounters
            WHERE ADMIT_DATE IS NOT NULL AND DISCHARGE_DATE IS NOT NULL
            ORDER BY MEMBER_ID, ADMIT_DATE
        """)
        rows = cur.fetchall()
    except Exception as e:
        conn.close()
        return {"status": "query_failed", "error": str(e)}

    demo: Dict[str, Dict] = {}
    try:
        cur.execute("""
            SELECT MEMBER_ID,
                   CAST(COALESCE((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH)) / 365.25, 60) AS INTEGER) AS AGE,
                   GENDER,
                   COALESCE(CHRONIC_CONDITIONS, 0) AS CHRONIC,
                   COALESCE(RISK_SCORE, 1.0) AS RISK_SCORE
            FROM members
        """)
        for r in cur.fetchall():
            demo[r[0]] = {
                'age': float(r[1] or 60), 'gender': r[2] or 'U',
                'chronic': float(r[3] or 0), 'risk_score': float(r[4] or 1.0)
            }
    except Exception:
        pass

    member_costs: Dict[str, Dict] = {}
    try:
        cur.execute("""
            SELECT MEMBER_ID,
                   COALESCE(SUM(CAST(PAID_AMOUNT AS REAL)), 0) AS total_paid,
                   COUNT(*) AS claim_count
            FROM claims GROUP BY MEMBER_ID
        """)
        for r in cur.fetchall():
            member_costs[r[0]] = {'total_paid': float(r[1]), 'claim_count': float(r[2])}
    except Exception:
        pass

    member_dx: Dict[str, float] = {}
    try:
        cur.execute("SELECT MEMBER_ID, COUNT(DISTINCT ICD10_CODE) FROM diagnoses GROUP BY MEMBER_ID")
        for r in cur.fetchall():
            member_dx[r[0]] = float(r[1])
    except Exception:
        pass

    member_er: Dict[str, float] = {}
    try:
        cur.execute("""
            SELECT MEMBER_ID, COUNT(*) FROM encounters
            WHERE UPPER(VISIT_TYPE) LIKE '%EMERGENCY%'
            GROUP BY MEMBER_ID
        """)
        for r in cur.fetchall():
            member_er[r[0]] = float(r[1])
    except Exception:
        pass

    conn.close()

    from datetime import datetime as _dt
    def _parse(d):
        try:
            return _dt.fromisoformat(d[:10])
        except Exception:
            return None

    by_member: Dict[str, List] = {}
    for r in rows:
        mid = r[0]
        by_member.setdefault(mid, []).append(r)

    X_rows: List[List[float]] = []
    T_rows: List[float] = []
    E_rows: List[int] = []
    feature_names = [
        "age", "gender_f", "los_days", "chronic_conditions",
        "risk_score", "log_total_paid", "diagnosis_count",
        "er_visit_count", "prior_admits", "is_emergency"
    ]

    for mid, encs in by_member.items():
        encs_sorted = sorted(encs, key=lambda x: x[2] or "")
        for i, enc in enumerate(encs_sorted):
            disch = _parse(enc[3])
            if disch is None:
                continue
            next_admit = None
            for j in range(i + 1, len(encs_sorted)):
                na = _parse(encs_sorted[j][2])
                if na is not None and na > disch:
                    next_admit = na
                    break
            if next_admit is not None:
                days = (next_admit - disch).days
                if days <= 0:
                    continue
                if days <= horizon_days:
                    T = days
                    E = 1
                else:
                    T = horizon_days
                    E = 0
            else:
                T = horizon_days
                E = 0

            d = demo.get(mid, {'age': 60, 'gender': 'U', 'chronic': 0, 'risk_score': 1.0})
            costs = member_costs.get(mid, {'total_paid': 0, 'claim_count': 0})
            dx_count = member_dx.get(mid, 0)
            er_count = member_er.get(mid, 0)
            visit_type = (enc[5] or '').upper()

            feats = [
                d['age'],
                1.0 if d['gender'].upper().startswith('F') else 0.0,
                float(enc[4] or 0),
                d['chronic'],
                d['risk_score'],
                math.log1p(costs['total_paid']),
                dx_count,
                er_count,
                float(i),
                1.0 if 'EMERGENCY' in visit_type else 0.0,
            ]
            X_rows.append(feats)
            T_rows.append(float(T))
            E_rows.append(E)

    if not X_rows:
        return {"status": "no_data"}

    X = np.asarray(X_rows, dtype=np.float64)
    T = np.asarray(T_rows, dtype=np.float64)
    E = np.asarray(E_rows, dtype=np.int64)

    n_events = int(E.sum())
    if method == "cox":
        model = fit_cox(X, T, E.astype(np.float64), feature_names=feature_names)
        ckpt = SURV_DIR / f"cox_readmit_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(ckpt, "wb") as f:
            pickle.dump(model, f)
        time_points = [7, 14, 30, 60, 90]
        avg_patient = X.mean(axis=0)
        surv_probs_arr = cox_survival_curve(model, avg_patient, time_points)
        survival_probabilities = {int(t): round(float(p), 4)
                                   for t, p in zip(time_points, surv_probs_arr)}

        risk_idx = feature_names.index('risk_score') if 'risk_score' in feature_names else -1
        if risk_idx >= 0:
            high_risk = avg_patient.copy()
            high_risk[risk_idx] = np.percentile(X[:, risk_idx], 75)
            hr_probs = cox_survival_curve(model, high_risk, time_points)
            high_risk_survival = {int(t): round(float(p), 4)
                                   for t, p in zip(time_points, hr_probs)}
        else:
            high_risk_survival = {}

        result = {
            "status": "ok",
            "method": "cox",
            "n": len(T),
            "n_events": n_events,
            "horizon_days": horizon_days,
            "c_index": model.c_index,
            "ll": model.ll,
            "hazard_ratios": cox_hazard_ratios(model),
            "survival_probabilities": survival_probabilities,
            "high_risk_survival": high_risk_survival,
            "event_rate": round(n_events / len(T) * 100, 2),
            "feature_names": feature_names,
            "checkpoint": str(ckpt),
        }
    elif method == "deephit":
        if not _TORCH_OK:
            return {"status": "torch_unavailable"}
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        r = fit_deephit(Xs, T, E, n_bins=20, horizon=float(horizon_days))
        r["method"] = "deephit"
        r["n"] = len(T)
        r["n_events"] = n_events
        result = r
    else:
        return {"status": "bad_method", "method": method}

    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        conn.execute(
            f"""INSERT INTO {SURV_LOG_TABLE}
                (TIMESTAMP, MODEL, OUTCOME, N, N_EVENTS, HORIZON_DAYS, C_INDEX, LL, CHECKPOINT, NOTES)
                VALUES (?, ?, 'readmission', ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                method,
                len(T),
                n_events,
                horizon_days,
                result.get("c_index"),
                result.get("ll"),
                result.get("checkpoint"),
                f"n_features={X.shape[1]}",
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return result


def get_survival_status(db_path: str) -> Dict:
    out: Dict = {"checkpoints_on_disk": [], "recent_fits": []}
    for p in SURV_DIR.glob("*") if SURV_DIR.exists() else []:
        out["checkpoints_on_disk"].append({"name": p.name, "size_kb": p.stat().st_size // 1024})
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, MODEL, OUTCOME, N, N_EVENTS, HORIZON_DAYS, C_INDEX, LL, CHECKPOINT
                FROM {SURV_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 20"""
        )
        out["recent_fits"] = [
            {"ts": r[0], "model": r[1], "outcome": r[2], "n": r[3], "n_events": r[4],
             "horizon_days": r[5], "c_index": r[6], "ll": r[7], "checkpoint": r[8]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
