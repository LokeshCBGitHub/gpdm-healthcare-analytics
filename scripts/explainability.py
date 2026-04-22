from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


ROOT = Path(__file__).resolve().parent.parent
EXPLAIN_LOG_TABLE = "gpdm_explain_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EXPLAIN_LOG_TABLE} (
            LOG_ID      INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP   TEXT,
            MODEL       TEXT,
            METHOD      TEXT,
            PATIENT_ID  TEXT,
            PREDICTION  REAL,
            TOP_FEATURES TEXT,   -- JSON
            NOTES       TEXT
        )
        """
    )
    conn.commit()


def integrated_gradients(
    model_fn: Callable,
    x: "np.ndarray",
    baseline: Optional["np.ndarray"] = None,
    steps: int = 50,
    target_idx: Optional[int] = None,
) -> "np.ndarray":
    if not _TORCH_OK or not _NP_OK:
        raise RuntimeError("torch + numpy required")

    x_arr = np.asarray(x, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(x_arr)
    baseline = np.asarray(baseline, dtype=np.float32)

    alphas = np.linspace(0.0, 1.0, steps + 1)
    gradients = []
    for a in alphas:
        xi = baseline + a * (x_arr - baseline)
        xt = torch.tensor(xi, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        out = model_fn(xt)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if out.ndim > 1 and target_idx is not None:
            scalar = out[0, target_idx]
        else:
            scalar = out.sum()
        grad = torch.autograd.grad(scalar, xt)[0]
        gradients.append(grad.detach().cpu().numpy()[0])
    grads = np.stack(gradients, axis=0)
    avg_grads = 0.5 * (grads[:-1] + grads[1:]).mean(axis=0)
    return (x_arr - baseline) * avg_grads


def smoothgrad(
    model_fn: Callable,
    x: "np.ndarray",
    sigma: float = 0.15,
    n: int = 25,
    target_idx: Optional[int] = None,
) -> "np.ndarray":
    if not _TORCH_OK or not _NP_OK:
        raise RuntimeError("torch + numpy required")
    x_arr = np.asarray(x, dtype=np.float32)
    grads = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        noise = rng.normal(0, sigma * (np.abs(x_arr).max() + 1e-6), size=x_arr.shape)
        xi = x_arr + noise.astype(np.float32)
        xt = torch.tensor(xi, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        out = model_fn(xt)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        scalar = out[0, target_idx] if (out.ndim > 1 and target_idx is not None) else out.sum()
        g = torch.autograd.grad(scalar, xt)[0].detach().cpu().numpy()[0]
        grads.append(g)
    return np.stack(grads, axis=0).mean(axis=0)


def permutation_importance(
    predict_fn: Callable,
    X: "np.ndarray",
    y: "np.ndarray",
    metric: str = "auc",
    n_repeats: int = 5,
    feature_names: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict]:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    baseline_preds = np.asarray(predict_fn(X))
    base_score = _score(metric, y, baseline_preds)

    n, p = X.shape
    names = feature_names or [f"x{i}" for i in range(p)]
    results = []
    for j in range(p):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            preds = np.asarray(predict_fn(Xp))
            drops.append(base_score - _score(metric, y, preds))
        results.append({
            "feature": names[j],
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })
    results.sort(key=lambda r: -r["importance_mean"])
    return results


def _score(metric: str, y, preds):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(preds, dtype=np.float64)
    if metric == "accuracy":
        yhat = (p >= 0.5).astype(int) if p.ndim == 1 else p.argmax(axis=1)
        return float((yhat == y).mean())
    if metric == "logloss":
        pp = np.clip(p, 1e-9, 1 - 1e-9)
        if p.ndim == 1:
            return float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
        return float(-np.log(pp[np.arange(len(y)), y.astype(int)]).mean())
    if p.ndim == 1:
        order = np.argsort(-p)
        y_sorted = y[order]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y) + 1)
        r_pos = ranks[y == 1].sum()
        return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return 0.0


def kernel_shap(
    predict_fn: Callable,
    x: "np.ndarray",
    background: "np.ndarray",
    n_samples: int = 200,
    seed: int = 0,
) -> "np.ndarray":
    if not _NP_OK:
        raise RuntimeError("numpy required")
    x = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    p = x.shape[0] if x.ndim == 1 else x.shape[-1]
    rng = np.random.default_rng(seed)
    phi = np.zeros(p, dtype=np.float64)

    x1 = x if x.ndim == 1 else x[0]

    for _ in range(n_samples):
        perm = rng.permutation(p)
        b_idx = rng.integers(0, len(bg))
        b = bg[b_idx].copy()
        with_x = b.copy()
        without_x = b.copy()
        for k, j in enumerate(perm):
            without_x[j] = b[j] if k == 0 else with_x[j]
            without_x_copy = with_x.copy()
            with_x[j] = x1[j]
            diff = predict_fn(with_x.reshape(1, -1))[0] - predict_fn(without_x_copy.reshape(1, -1))[0]
            phi[j] += float(diff)
    phi /= n_samples
    return phi


CLINICAL_FEATURE_LABELS = {
    "age": "Age",
    "gender_f": "Gender = Female",
    "gender_m": "Gender = Male",
    "n_encounters_12m": "Encounters, last 12 months",
    "n_admits_12m": "Inpatient admissions, last 12 months",
    "n_er_12m": "ER visits, last 12 months",
    "los_days": "Length of stay (days)",
    "log_total_paid": "Log total paid",
    "prior_admits": "Prior admission count",
    "dx_diabetes": "Dx: Diabetes (E11.x)",
    "dx_chf": "Dx: Congestive heart failure (I50.x)",
    "dx_copd": "Dx: COPD (J44.x)",
    "dx_ckd": "Dx: Chronic kidney disease (N18.x)",
    "dx_mi": "Dx: Acute MI (I21.x)",
    "dx_cancer": "Dx: Neoplasm",
    "dx_depression": "Dx: Depression (F32/F33)",
    "total_paid_12m": "Total paid, last 12 months",
    "rx_count_12m": "Rx fills, last 12 months",
}


def label_feature(name: str) -> str:
    return CLINICAL_FEATURE_LABELS.get(name, name)


def top_features(
    attributions: "np.ndarray",
    feature_names: List[str],
    k: int = 10,
    values: Optional["np.ndarray"] = None,
) -> List[Dict]:
    if not _NP_OK:
        return []
    a = np.asarray(attributions, dtype=np.float64).flatten()
    order = np.argsort(-np.abs(a))[:k]
    out = []
    for i in order:
        entry = {
            "feature": feature_names[i],
            "label": label_feature(feature_names[i]),
            "attribution": float(a[i]),
            "direction": "increases risk" if a[i] > 0 else "decreases risk",
        }
        if values is not None:
            entry["value"] = float(np.asarray(values).flatten()[i])
        out.append(entry)
    return out


def explain_risk(
    patient_id: str,
    db_path: str,
    method: str = "permutation",
    k: int = 10,
) -> Dict:
    out: Dict = {"patient_id": patient_id, "method_tried": method}

    if method in ("ig", "integrated_gradients", "auto") and _TORCH_OK:
        try:
            try:
                from . import patient_transformer as pt
            except Exception:
                import patient_transformer as pt
            if hasattr(pt, "build_single_patient_input") and hasattr(pt, "load_trained_model"):
                inputs = pt.build_single_patient_input(patient_id, db_path)
                model = pt.load_trained_model()
                if isinstance(model, tuple):
                    model = model[0]

                def _fn(xt):
                    return model(xt) if not isinstance(inputs, dict) else model(**{
                        **inputs, "inputs_embeds": xt
                    })

                if isinstance(inputs, dict) and any(
                    isinstance(v, torch.Tensor) and v.dtype in (torch.long, torch.int64)
                    for v in inputs.values()
                ):
                    pass
                else:
                    x_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else np.asarray(inputs)
                    attrs = integrated_gradients(_fn, x_np, steps=30, target_idx=0)
                    feature_names = getattr(pt, "FEATURE_NAMES", [f"f{i}" for i in range(len(attrs))])
                    out.update({
                        "method": "integrated_gradients",
                        "top_features": top_features(attrs, feature_names, k, x_np),
                    })
                    _log_explain(db_path, "patient_transformer", "ig", patient_id, out)
                    return out
        except Exception as e:
            out["ig_error"] = str(e)

    try:
        try:
            from . import ml_models as mm
        except Exception:
            import ml_models as mm
        if hasattr(mm, "build_scoring_frame") and hasattr(mm, "predict_risk_batch"):
            X, y, _ = mm.build_scoring_frame(db_path, target="risk_high_cost")
            feats = getattr(mm, "RISK_FEATURE_NAMES", [f"f{i}" for i in range(X.shape[1])])
            x_row = None
            if hasattr(mm, "build_single_patient_row"):
                x_row = mm.build_single_patient_row(patient_id, db_path)
            if x_row is not None:
                global_imp = permutation_importance(
                    mm.predict_risk_batch, X, y, metric="auc", n_repeats=3,
                    feature_names=feats,
                )
                imp_map = {d["feature"]: d["importance_mean"] for d in global_imp}
                cohort_mean = X.mean(axis=0)
                x_row_arr = np.asarray(x_row, dtype=np.float64)
                local = (x_row_arr - cohort_mean) * np.array([imp_map.get(f, 0.0) for f in feats])
                out.update({
                    "method": "permutation_local",
                    "top_features": top_features(local, feats, k, x_row_arr),
                    "global_importance": global_imp[:k],
                })
                prediction = float(mm.predict_risk_batch(x_row_arr.reshape(1, -1))[0])
                out["prediction"] = prediction
                _log_explain(db_path, "ml_models", "permutation_local", patient_id, out)
                return out
    except Exception as e:
        out["permutation_error"] = str(e)

    out["status"] = "no_explainer_available"
    return out


def _log_explain(db_path: str, model: str, method: str, patient_id: str, payload: Dict):
    try:
        import json as _json
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        conn.execute(
            f"""INSERT INTO {EXPLAIN_LOG_TABLE}
                (TIMESTAMP, MODEL, METHOD, PATIENT_ID, PREDICTION, TOP_FEATURES, NOTES)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.strftime("%Y-%m-%d %H:%M:%S"),
             model, method, patient_id,
             payload.get("prediction"),
             _json.dumps(payload.get("top_features", []))[:4000],
             None),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_explain_status(db_path: str) -> Dict:
    out: Dict = {"recent_explanations": []}
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, MODEL, METHOD, PATIENT_ID, PREDICTION
                FROM {EXPLAIN_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 20"""
        )
        out["recent_explanations"] = [
            {"ts": r[0], "model": r[1], "method": r[2], "patient_id": r[3], "prediction": r[4]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
