from __future__ import annotations

import copy
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_OK = True
except Exception:
    _NUMPY_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
ENSEMBLE_DIR = MODELS_DIR / "ensembles"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAINTY_LOG_TABLE = "gpdm_uncertainty_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {UNCERTAINTY_LOG_TABLE} (
            LOG_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP     TEXT NOT NULL,
            MODEL         TEXT,
            METHOD        TEXT,          -- 'mc_dropout' | 'ensemble' | 'combined'
            PATIENT_ID    TEXT,
            MEAN_P        REAL,
            STD_P         REAL,
            CI_LOW        REAL,
            CI_HIGH       REAL,
            N_SAMPLES     INTEGER
        )
        """
    )
    conn.commit()


def _enable_mc_dropout(model: "nn.Module") -> None:
    if not _TORCH_OK:
        return
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout_predict(
    model: "nn.Module",
    inputs,
    n_samples: int = 50,
    head: Optional[str] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    if not _TORCH_OK or not _NUMPY_OK:
        raise RuntimeError("torch + numpy required")

    _enable_mc_dropout(model)
    samples: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            if isinstance(inputs, dict):
                out = model(**inputs)
            else:
                out = model(inputs)
            if isinstance(out, dict):
                out = out[head] if head else next(iter(out.values()))
            probs = torch.sigmoid(out).detach().cpu().numpy()
            samples.append(probs)
    arr = np.stack(samples, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def train_ensemble(
    model_factory: Callable[[int], "nn.Module"],
    train_batches: List[Tuple],
    n_members: int = 5,
    epochs: int = 5,
    lr: float = 1e-3,
    seeds: Optional[List[int]] = None,
    loss_fn: Optional[Callable] = None,
) -> List["nn.Module"]:
    if not _TORCH_OK:
        raise RuntimeError("torch not available")

    seeds = seeds or list(range(n_members))
    members: List[nn.Module] = []
    for seed in seeds[:n_members]:
        torch.manual_seed(seed)
        if _NUMPY_OK:
            np.random.seed(seed)
        model = model_factory(seed)
        model.to(torch.device("cpu"))
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_batches:
                optim.zero_grad()
                out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                if loss_fn is not None:
                    loss = loss_fn(out, targets)
                elif isinstance(out, dict):
                    loss = sum(
                        F.binary_cross_entropy_with_logits(v, targets[k].float())
                        for k, v in out.items()
                        if k in targets
                    ) / max(1, len(out))
                else:
                    loss = F.binary_cross_entropy_with_logits(out, targets.float())
                loss.backward()
                optim.step()
        members.append(model)
    return members


def ensemble_predict(
    members: List["nn.Module"],
    inputs,
    head: Optional[str] = None,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    if not _TORCH_OK or not _NUMPY_OK:
        raise RuntimeError("torch + numpy required")

    per_member: List[np.ndarray] = []
    with torch.no_grad():
        for m in members:
            m.eval()
            out = m(**inputs) if isinstance(inputs, dict) else m(inputs)
            if isinstance(out, dict):
                out = out[head] if head else next(iter(out.values()))
            per_member.append(torch.sigmoid(out).detach().cpu().numpy())
    arr = np.stack(per_member, axis=0)
    return arr.mean(axis=0), arr.std(axis=0), arr


def save_ensemble(members: List["nn.Module"], dir_path: Path) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(members):
        torch.save(m.state_dict(), dir_path / f"member_{i:02d}.pt")
    with open(dir_path / "meta.pkl", "wb") as f:
        pickle.dump({"n": len(members), "saved_at": time.time()}, f)


def load_ensemble(model_factory: Callable[[int], "nn.Module"], dir_path: Path) -> List["nn.Module"]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    meta_path = dir_path / "meta.pkl"
    if not meta_path.exists():
        return []
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    members: List[nn.Module] = []
    for i in range(meta["n"]):
        ckpt = dir_path / f"member_{i:02d}.pt"
        if not ckpt.exists():
            continue
        m = model_factory(i)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        members.append(m)
    return members


def combined_predict(
    members: List["nn.Module"],
    inputs,
    mc_samples: int = 20,
    head: Optional[str] = None,
) -> Dict:
    if not _TORCH_OK or not _NUMPY_OK:
        raise RuntimeError("torch + numpy required")

    per_member_means: List[np.ndarray] = []
    per_member_stds: List[np.ndarray] = []
    for m in members:
        mu, sd = mc_dropout_predict(m, inputs, n_samples=mc_samples, head=head)
        per_member_means.append(mu)
        per_member_stds.append(sd)
    mu_stack = np.stack(per_member_means, axis=0)
    sd_stack = np.stack(per_member_stds, axis=0)

    mean = mu_stack.mean(axis=0)
    epistemic = mu_stack.std(axis=0)
    aleatoric = sd_stack.mean(axis=0)
    total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

    low = np.clip(mean - 1.96 * total, 0.0, 1.0)
    high = np.clip(mean + 1.96 * total, 0.0, 1.0)

    return {
        "mean": mean.tolist() if mean.ndim else float(mean),
        "std_total": total.tolist() if total.ndim else float(total),
        "std_epistemic": epistemic.tolist() if epistemic.ndim else float(epistemic),
        "std_aleatoric": aleatoric.tolist() if aleatoric.ndim else float(aleatoric),
        "ci95_low": low.tolist() if low.ndim else float(low),
        "ci95_high": high.tolist() if high.ndim else float(high),
        "n_members": len(members),
        "mc_samples": mc_samples,
    }


def predict_risk_with_uncertainty(patient_id: str, db_path: str) -> Dict:
    out: Dict = {"patient_id": patient_id, "methods_tried": []}

    if _TORCH_OK:
        try:
            try:
                from . import patient_transformer as pt
            except Exception:
                import patient_transformer as pt

            if hasattr(pt, "build_single_patient_input") and hasattr(pt, "model_factory"):
                inputs = pt.build_single_patient_input(patient_id, db_path)
                members = load_ensemble(pt.model_factory, ENSEMBLE_DIR / "patient_transformer")
                if members:
                    result = combined_predict(members, inputs, mc_samples=20)
                    out.update({"method": "ensemble+mc_dropout", **result})
                    out["methods_tried"].append("ensemble+mc_dropout")
                    _log_uncertainty(db_path, "patient_transformer", "combined",
                                     patient_id, result)
                    return out
        except Exception as exc:
            out["methods_tried"].append(f"ensemble_failed:{exc}")

        try:
            try:
                from . import patient_transformer as pt
            except Exception:
                import patient_transformer as pt

            if hasattr(pt, "load_trained_model") and hasattr(pt, "build_single_patient_input"):
                model, _tok = pt.load_trained_model() if not isinstance(
                    pt.load_trained_model(), tuple
                ) else pt.load_trained_model()
                inputs = pt.build_single_patient_input(patient_id, db_path)
                mean, std = mc_dropout_predict(model, inputs, n_samples=50)
                out.update({
                    "method": "mc_dropout",
                    "mean": float(np.atleast_1d(mean).mean()),
                    "std_total": float(np.atleast_1d(std).mean()),
                    "n_samples": 50,
                })
                out["methods_tried"].append("mc_dropout")
                _log_uncertainty(db_path, "patient_transformer", "mc_dropout",
                                 patient_id, out)
                return out
        except Exception as exc:
            out["methods_tried"].append(f"mc_dropout_failed:{exc}")

    try:
        try:
            from . import ml_models as mm
        except Exception:
            import ml_models as mm

        p = mm.predict_risk(patient_id) if hasattr(mm, "predict_risk") else None
        if p is not None:
            mean_p = float(p.get("probability", p.get("p", 0.0)))
            std = 0.08 + 0.1 * abs(0.5 - mean_p)
            out.update({
                "method": "sklearn_heuristic",
                "mean": mean_p,
                "std_total": std,
                "ci95_low": max(0.0, mean_p - 1.96 * std),
                "ci95_high": min(1.0, mean_p + 1.96 * std),
            })
            out["methods_tried"].append("sklearn_heuristic")
            _log_uncertainty(db_path, "ml_models", "heuristic", patient_id, out)
            return out
    except Exception as exc:
        out["methods_tried"].append(f"sklearn_failed:{exc}")

    out["status"] = "no_predictor_available"
    return out


def _log_uncertainty(db_path: str, model: str, method: str, patient_id: str, payload: Dict):
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        mean_p = payload.get("mean")
        std_p = payload.get("std_total")
        ci_low = payload.get("ci95_low")
        ci_high = payload.get("ci95_high")
        n_samples = payload.get("n_samples") or payload.get("mc_samples")
        if isinstance(mean_p, list) and mean_p:
            mean_p = float(sum(mean_p) / len(mean_p))
        if isinstance(std_p, list) and std_p:
            std_p = float(sum(std_p) / len(std_p))
        if isinstance(ci_low, list) and ci_low:
            ci_low = float(sum(ci_low) / len(ci_low))
        if isinstance(ci_high, list) and ci_high:
            ci_high = float(sum(ci_high) / len(ci_high))
        conn.execute(
            f"""INSERT INTO {UNCERTAINTY_LOG_TABLE}
                (TIMESTAMP, MODEL, METHOD, PATIENT_ID, MEAN_P, STD_P, CI_LOW, CI_HIGH, N_SAMPLES)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                model,
                method,
                patient_id,
                mean_p,
                std_p,
                ci_low,
                ci_high,
                n_samples,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_uncertainty_status(db_path: str) -> Dict:
    out: Dict = {
        "ensembles_on_disk": [],
        "recent_predictions": [],
        "total_predictions": 0,
    }
    try:
        for sub in ENSEMBLE_DIR.iterdir() if ENSEMBLE_DIR.exists() else []:
            meta = sub / "meta.pkl"
            if meta.exists():
                with open(meta, "rb") as f:
                    m = pickle.load(f)
                out["ensembles_on_disk"].append({
                    "name": sub.name,
                    "n_members": m.get("n", 0),
                    "saved_at": m.get("saved_at"),
                })
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT COUNT(*) FROM {UNCERTAINTY_LOG_TABLE}"""
        )
        out["total_predictions"] = int(cur.fetchone()[0])
        cur = conn.execute(
            f"""SELECT TIMESTAMP, MODEL, METHOD, PATIENT_ID, MEAN_P, STD_P, CI_LOW, CI_HIGH
                FROM {UNCERTAINTY_LOG_TABLE}
                ORDER BY LOG_ID DESC LIMIT 20"""
        )
        out["recent_predictions"] = [
            {
                "ts": r[0], "model": r[1], "method": r[2], "patient_id": r[3],
                "mean_p": r[4], "std_p": r[5], "ci95_low": r[6], "ci95_high": r[7],
            }
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as exc:
        out["error"] = str(exc)
    return out
