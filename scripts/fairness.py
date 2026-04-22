from __future__ import annotations

import os
import json
import math
import sqlite3
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    _SK = True
except Exception:
    _SK = False


_LOG_TABLE = 'gpdm_fairness_log'


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            attribute TEXT NOT NULL,
            metric TEXT NOT NULL,
            group_name TEXT,
            value REAL,
            ci_lo REAL,
            ci_hi REAL,
            n INTEGER,
            details TEXT
        )
    """)
    conn.commit()


def _confusion_at_threshold(p: np.ndarray, y: np.ndarray, t: float = 0.5) -> Dict[str, int]:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def tpr_fpr(p: np.ndarray, y: np.ndarray, t: float = 0.5) -> Tuple[float, float]:
    c = _confusion_at_threshold(p, y, t)
    tpr = c['tp'] / max(1, c['tp'] + c['fn'])
    fpr = c['fp'] / max(1, c['fp'] + c['tn'])
    return tpr, fpr


def positive_rate(p: np.ndarray, t: float = 0.5) -> float:
    return float((p >= t).mean()) if len(p) else 0.0


def auc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    if _SK:
        try:
            return float(roc_auc_score(y, p))
        except Exception:
            pass
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if len(p) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        k = int(mask.sum())
        if k == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        total += (k / n) * abs(acc - conf)
    return float(total)


@dataclass
class GroupMetrics:
    group: str
    n: int
    n_pos: int
    prevalence: float
    positive_rate: float
    tpr: float
    fpr: float
    auc: float
    brier: float
    ece: float


def per_group_metrics(p: np.ndarray, y: np.ndarray, groups: Sequence,
                       threshold: float = 0.5, n_bins: int = 10) -> List[GroupMetrics]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(list(groups))
    out = []
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        pg, yg = p[mask], y[mask]
        if len(pg) == 0:
            continue
        tpr, fpr = tpr_fpr(pg, yg, threshold)
        out.append(GroupMetrics(
            group=str(g),
            n=int(len(pg)),
            n_pos=int(yg.sum()),
            prevalence=float(yg.mean()) if len(yg) else float('nan'),
            positive_rate=positive_rate(pg, threshold),
            tpr=tpr, fpr=fpr,
            auc=auc(pg, yg),
            brier=brier(pg, yg),
            ece=ece(pg, yg, n_bins=n_bins),
        ))
    return out


@dataclass
class FairnessReport:
    attribute: str
    groups: List[GroupMetrics]
    dpd: float
    eod_tpr: float
    eod_fpr: float
    calibration_gap: float
    auc_gap: float
    reference_group: str
    verdict: str
    ts: float = field(default_factory=lambda: time.time())


def fairness_report(p: np.ndarray, y: np.ndarray, groups: Sequence,
                     attribute: str, threshold: float = 0.5,
                     reference: Optional[str] = None) -> FairnessReport:
    gm = per_group_metrics(p, y, groups, threshold=threshold)
    if not gm:
        return FairnessReport(attribute=attribute, groups=[], dpd=0.0, eod_tpr=0.0,
                               eod_fpr=0.0, calibration_gap=0.0, auc_gap=0.0,
                               reference_group='', verdict='insufficient_data')
    ref = reference or max(gm, key=lambda m: m.n).group
    ref_m = next(m for m in gm if m.group == ref)
    dpd = max(m.positive_rate for m in gm) - min(m.positive_rate for m in gm)
    eod_tpr = max(abs(m.tpr - ref_m.tpr) for m in gm)
    eod_fpr = max(abs(m.fpr - ref_m.fpr) for m in gm)
    cal_gap = max(m.ece for m in gm) - min(m.ece for m in gm)
    auc_vals = [m.auc for m in gm if not math.isnan(m.auc)]
    auc_gap = (max(auc_vals) - min(auc_vals)) if len(auc_vals) >= 2 else 0.0

    worst = max(dpd, eod_tpr, eod_fpr)
    if worst > 0.20:
        verdict = 'severe_disparity'
    elif worst > 0.10:
        verdict = 'material_disparity'
    elif worst > 0.05:
        verdict = 'minor_disparity'
    else:
        verdict = 'acceptable'

    return FairnessReport(
        attribute=attribute, groups=gm, dpd=float(dpd),
        eod_tpr=float(eod_tpr), eod_fpr=float(eod_fpr),
        calibration_gap=float(cal_gap), auc_gap=float(auc_gap),
        reference_group=ref, verdict=verdict,
    )


def bootstrap_metric(p: np.ndarray, y: np.ndarray, fn, n_boot: int = 500,
                     seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(p)
    if n == 0:
        return (float('nan'),) * 3
    vals = []
    base = fn(p, y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(p[idx], y[idx]))
        except Exception:
            continue
    if not vals:
        return float(base), float('nan'), float('nan')
    vals = np.array(vals, dtype=float)
    lo = float(np.nanpercentile(vals, 2.5))
    hi = float(np.nanpercentile(vals, 97.5))
    return float(base), lo, hi


def _fetch_predictions_with_demographics(db_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.prob, p.actual, m.gender, m.region, m.race, m.age
            FROM gpdm_predictions p
            LEFT JOIN members m ON m.member_id = p.entity_id
            WHERE p.actual IS NOT NULL
              AND p.prob IS NOT NULL
            LIMIT 200000
        """)
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    if not rows:
        return None
    p = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    gender = np.array([r[2] or 'unknown' for r in rows])
    region = np.array([r[3] or 'unknown' for r in rows])
    race = np.array([r[4] or 'unknown' for r in rows])
    age = np.array([r[5] if r[5] is not None else -1 for r in rows], dtype=float)
    buckets = np.full(len(age), 'unknown', dtype=object)
    for i, a in enumerate(age):
        if a < 0:
            continue
        elif a < 18:
            buckets[i] = '<18'
        elif a < 40:
            buckets[i] = '18-39'
        elif a < 65:
            buckets[i] = '40-64'
        else:
            buckets[i] = '65+'
    return {'p': p, 'y': y, 'gender': gender, 'region': region,
            'race': race, 'age_bucket': buckets}


def audit_from_sqlite(db_path: str, threshold: float = 0.5,
                      n_boot: int = 200) -> Dict:
    data = _fetch_predictions_with_demographics(db_path)
    if data is None:
        return {'status': 'no_data',
                'reason': 'gpdm_predictions or members unavailable, or no labeled rows'}

    conn = sqlite3.connect(db_path)
    _ensure_log_table(conn)
    reports: Dict[str, Dict] = {}

    for attr in ['gender', 'region', 'race', 'age_bucket']:
        groups = data[attr]
        if len(set(groups.tolist())) < 2:
            continue
        rep = fairness_report(data['p'], data['y'], groups,
                               attribute=attr, threshold=threshold)
        cis = {}
        for g in rep.groups:
            mask = groups == g.group
            _, lo, hi = bootstrap_metric(data['p'][mask], data['y'][mask], auc,
                                          n_boot=n_boot)
            cis[g.group] = (lo, hi)

        ts = time.time()
        for metric, value in [('dpd', rep.dpd), ('eod_tpr', rep.eod_tpr),
                               ('eod_fpr', rep.eod_fpr), ('auc_gap', rep.auc_gap),
                               ('calibration_gap', rep.calibration_gap)]:
            conn.execute(
                f"INSERT INTO {_LOG_TABLE}(ts,attribute,metric,value,n,details) "
                f"VALUES(?,?,?,?,?,?)",
                (ts, attr, metric, float(value), int(len(data['p'])),
                 json.dumps({'verdict': rep.verdict, 'reference': rep.reference_group}))
            )
        for g in rep.groups:
            lo, hi = cis.get(g.group, (float('nan'), float('nan')))
            conn.execute(
                f"INSERT INTO {_LOG_TABLE}(ts,attribute,metric,group_name,value,"
                f"ci_lo,ci_hi,n,details) VALUES(?,?,?,?,?,?,?,?,?)",
                (ts, attr, 'auc', g.group, float(g.auc), float(lo), float(hi),
                 int(g.n), json.dumps(asdict(g)))
            )
        conn.commit()
        reports[attr] = {
            'verdict': rep.verdict,
            'dpd': rep.dpd,
            'eod_tpr': rep.eod_tpr,
            'eod_fpr': rep.eod_fpr,
            'auc_gap': rep.auc_gap,
            'calibration_gap': rep.calibration_gap,
            'reference_group': rep.reference_group,
            'groups': [asdict(g) for g in rep.groups],
            'ci_auc': {k: {'lo': v[0], 'hi': v[1]} for k, v in cis.items()},
        }

    conn.close()
    worst = 'acceptable'
    order = ['acceptable', 'minor_disparity', 'material_disparity', 'severe_disparity', 'insufficient_data']
    for rep in reports.values():
        if order.index(rep['verdict']) > order.index(worst):
            worst = rep['verdict']
    return {'status': 'ok', 'overall': worst, 'reports': reports,
            'n_predictions': int(len(data['p'])), 'threshold': threshold}


def get_fairness_status(db_path: str) -> Dict:
    if not os.path.exists(db_path):
        return {'status': 'no_db'}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*), MAX(ts) FROM {_LOG_TABLE}")
        n, last_ts = cur.fetchone()
        cur.execute(
            f"SELECT attribute, metric, value, ts FROM {_LOG_TABLE} "
            f"WHERE metric IN ('dpd','eod_tpr','eod_fpr') "
            f"ORDER BY ts DESC LIMIT 60"
        )
        recent = [{'attribute': r[0], 'metric': r[1], 'value': r[2], 'ts': r[3]}
                  for r in cur.fetchall()]
    except sqlite3.OperationalError:
        conn.close()
        return {'status': 'never_run'}
    conn.close()
    return {'status': 'ok', 'n_log_rows': int(n or 0),
            'last_audit_ts': float(last_ts) if last_ts else None,
            'recent_metrics': recent}
