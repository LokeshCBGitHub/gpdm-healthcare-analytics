from __future__ import annotations

import math
import pickle
import random
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
FOREST_DIR = MODELS_DIR / "causal_forest"
FOREST_DIR.mkdir(parents=True, exist_ok=True)

FOREST_LOG_TABLE = "gpdm_causal_forest_log"


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FOREST_LOG_TABLE} (
            LOG_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
            TIMESTAMP  TEXT,
            OUTCOME    TEXT,
            TREATMENT  TEXT,
            N          INTEGER,
            ATE        REAL,
            ATE_SE     REAL,
            ATE_CI_LO  REAL,
            ATE_CI_HI  REAL,
            POLICY_VALUE REAL,
            CHECKPOINT TEXT
        )
        """
    )
    conn.commit()


@dataclass
class TreeNode:
    is_leaf: bool = True
    feature: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    tau_hat: float = 0.0
    n_treat: int = 0
    n_ctrl: int = 0
    indices: Optional[List[int]] = None


def _split_criterion_tau(Y, T, mask, feat, thr):
    left = mask & (feat < thr)
    right = mask & (feat >= thr)
    if left.sum() < 5 or right.sum() < 5:
        return -np.inf
    if not ((T[left] == 1).any() and (T[left] == 0).any()):
        return -np.inf
    if not ((T[right] == 1).any() and (T[right] == 0).any()):
        return -np.inf
    tau_l = Y[left & (T == 1)].mean() - Y[left & (T == 0)].mean()
    tau_r = Y[right & (T == 1)].mean() - Y[right & (T == 0)].mean()
    nl, nr = left.sum(), right.sum()
    return float(nl * nr / (nl + nr) * (tau_l - tau_r) ** 2)


def _grow_tree(X, Y, T, indices, max_depth, min_leaf, n_features_try, rng):
    node = TreeNode(is_leaf=True, indices=list(indices))
    if max_depth == 0 or len(indices) < 2 * min_leaf:
        return node
    n = len(indices)
    p = X.shape[1]
    feat_subset = rng.choice(p, size=min(n_features_try, p), replace=False)

    best_gain = -np.inf
    best_feat = -1
    best_thr = 0.0
    mask_full = np.zeros(X.shape[0], dtype=bool)
    mask_full[indices] = True
    for j in feat_subset:
        vals = X[indices, j]
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        cands = uniq if len(uniq) <= 20 else np.quantile(uniq, np.linspace(0.1, 0.9, 20))
        for thr in cands:
            gain = _split_criterion_tau(Y, T, mask_full, X[:, j], thr)
            if gain > best_gain:
                best_gain = gain
                best_feat = int(j)
                best_thr = float(thr)
    if best_feat < 0 or best_gain <= 0:
        return node

    left_mask = X[indices, best_feat] < best_thr
    left_idx = [indices[i] for i in range(n) if left_mask[i]]
    right_idx = [indices[i] for i in range(n) if not left_mask[i]]
    if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
        return node

    node.is_leaf = False
    node.feature = best_feat
    node.threshold = best_thr
    node.indices = None
    node.left = _grow_tree(X, Y, T, left_idx, max_depth - 1, min_leaf, n_features_try, rng)
    node.right = _grow_tree(X, Y, T, right_idx, max_depth - 1, min_leaf, n_features_try, rng)
    return node


def _fill_leaf_estimates(node, X, Y, T, honest_idx):
    if node.is_leaf:
        leaf_idx = [i for i in honest_idx if _in_leaf(node, X[i], node)]
        return



def _route(root: TreeNode, x: "np.ndarray") -> TreeNode:
    node = root
    while not node.is_leaf:
        if x[node.feature] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node


def _leaves(root: TreeNode) -> List[TreeNode]:
    out: List[TreeNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            out.append(n)
        else:
            stack.append(n.left)
            stack.append(n.right)
    return out


def _assign_honest_leaves(root: TreeNode, X, Y, T, honest_idx):
    leaves = _leaves(root)
    for leaf in leaves:
        leaf.indices = []
    for i in honest_idx:
        leaf = _route(root, X[i])
        leaf.indices.append(i)
    for leaf in leaves:
        if not leaf.indices:
            leaf.tau_hat = 0.0
            leaf.n_treat = 0
            leaf.n_ctrl = 0
            continue
        Yl = Y[leaf.indices]
        Tl = T[leaf.indices]
        tr = Yl[Tl == 1]
        ct = Yl[Tl == 0]
        if len(tr) > 0 and len(ct) > 0:
            leaf.tau_hat = float(tr.mean() - ct.mean())
        else:
            leaf.tau_hat = 0.0
        leaf.n_treat = int(len(tr))
        leaf.n_ctrl = int(len(ct))


@dataclass
class CausalForest:
    trees: List[TreeNode]
    feature_names: List[str]
    n: int
    oob_ate: float = 0.0
    oob_ate_se: float = 0.0
    fitted_at: float = field(default_factory=time.time)

    def predict(self, x: "np.ndarray") -> Dict:
        preds = np.array([_route(t, x).tau_hat for t in self.trees])
        tau = float(preds.mean())
        if len(preds) > 1:
            se = float(preds.std(ddof=1) / math.sqrt(len(preds)))
        else:
            se = 0.0
        return {
            "tau": tau,
            "se": se,
            "ci95_lo": tau - 1.96 * se,
            "ci95_hi": tau + 1.96 * se,
        }

    def predict_many(self, X: "np.ndarray") -> "np.ndarray":
        return np.array([self.predict(X[i])["tau"] for i in range(len(X))])


def fit_causal_forest(
    X: "np.ndarray",
    Y: "np.ndarray",
    T: "np.ndarray",
    n_trees: int = 200,
    max_depth: int = 8,
    min_leaf: int = 10,
    n_features_try: Optional[int] = None,
    subsample_frac: float = 0.5,
    feature_names: Optional[List[str]] = None,
    seed: int = 42,
) -> CausalForest:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    n, p = X.shape
    n_features_try = n_features_try or max(1, int(math.sqrt(p)))
    rng = np.random.default_rng(seed)

    trees: List[TreeNode] = []
    for b in range(n_trees):
        m = int(n * subsample_frac)
        sub = rng.choice(n, size=m, replace=False)
        rng.shuffle(sub)
        half = m // 2
        struct_idx = sub[:half].tolist()
        honest_idx = sub[half:].tolist()
        if len(set(T[struct_idx].tolist())) < 2 or len(set(T[honest_idx].tolist())) < 2:
            continue
        tree = _grow_tree(X, Y, T, struct_idx, max_depth, min_leaf,
                          n_features_try, rng)
        _assign_honest_leaves(tree, X, Y, T, honest_idx)
        trees.append(tree)

    forest = CausalForest(
        trees=trees,
        feature_names=feature_names or [f"x{i}" for i in range(p)],
        n=n,
    )
    taus = forest.predict_many(X)
    forest.oob_ate = float(taus.mean())
    forest.oob_ate_se = float(taus.std(ddof=1) / math.sqrt(len(taus))) if len(taus) > 1 else 0.0
    return forest


@dataclass
class DRLearner:
    base: CausalForest
    n: int
    fitted_at: float = field(default_factory=time.time)

    def predict(self, x: "np.ndarray") -> Dict:
        return self.base.predict(x)


def fit_dr_learner(
    X: "np.ndarray",
    Y: "np.ndarray",
    T: "np.ndarray",
    mu0_hat: "np.ndarray",
    mu1_hat: "np.ndarray",
    e_hat: "np.ndarray",
    n_trees: int = 200,
) -> DRLearner:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    mu0 = np.asarray(mu0_hat, dtype=np.float64)
    mu1 = np.asarray(mu1_hat, dtype=np.float64)
    e = np.clip(np.asarray(e_hat, dtype=np.float64), 0.02, 0.98)
    Z = (mu1 - mu0
         + T * (Y - mu1) / e
         - (1 - T) * (Y - mu0) / (1 - e))
    Xb = np.vstack([X, X])
    Yb = np.concatenate([Z, np.zeros_like(Z)])
    Tb = np.concatenate([np.ones_like(T), np.zeros_like(T)])
    base = fit_causal_forest(Xb, Yb, Tb, n_trees=n_trees, max_depth=6, min_leaf=20)
    return DRLearner(base=base, n=len(X))


def evaluate_policy(
    tau_hat: "np.ndarray",
    Y: "np.ndarray",
    T: "np.ndarray",
    e_hat: Optional["np.ndarray"] = None,
    top_frac: float = 0.2,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> Dict:
    if not _NP_OK:
        raise RuntimeError("numpy required")
    tau = np.asarray(tau_hat, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    n = len(tau)
    if e_hat is None:
        e_hat = np.full(n, float(T.mean()))
    e = np.clip(e_hat, 0.02, 0.98)
    thresh = np.quantile(tau, 1 - top_frac)
    pi = (tau >= thresh).astype(int)
    val = pi * T * Y / e + (1 - pi) * (1 - T) * Y / (1 - e)
    point = float(val.mean())

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots.append(float(val[idx].mean()))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))

    baseline = float(((1 - T) * Y / (1 - e)).mean())
    return {
        "policy_value": point,
        "policy_ci95": (lo, hi),
        "baseline_untreated": baseline,
        "improvement": point - baseline,
        "top_frac": top_frac,
        "n_bootstrap": n_bootstrap,
        "n_treated_by_policy": int(pi.sum()),
    }


def estimate_heterogeneous_followup_effect(db_path: str) -> Dict:
    if not _NP_OK:
        return {"status": "numpy_unavailable"}
    try:
        try:
            from . import causal_inference as ci
        except Exception:
            import causal_inference as ci
        if not hasattr(ci, "_build_followup_frame"):
            X, Y, T, names = _build_followup_frame(db_path)
        else:
            X, Y, T, names = ci._build_followup_frame(db_path)
    except Exception as e:
        return {"status": "frame_build_failed", "error": str(e)}

    if X is None or len(X) < 200:
        return {"status": "insufficient_data", "n": 0 if X is None else len(X)}

    forest = fit_causal_forest(X, Y, T, n_trees=200, max_depth=7,
                               min_leaf=15, feature_names=names)
    tau = forest.predict_many(X)
    pol = evaluate_policy(tau, Y, T, top_frac=0.2)

    order = np.argsort(tau)
    k = max(10, len(tau) // 10)
    bottom = tau[order[:k]].mean()
    top = tau[order[-k:]].mean()

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt = FOREST_DIR / f"followup_forest_{ts}.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(forest, f)

    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        conn.execute(
            f"""INSERT INTO {FOREST_LOG_TABLE}
                (TIMESTAMP, OUTCOME, TREATMENT, N, ATE, ATE_SE, ATE_CI_LO, ATE_CI_HI,
                 POLICY_VALUE, CHECKPOINT)
                VALUES (?, 'readmit_30d', 'followup_14d', ?, ?, ?, ?, ?, ?, ?)""",
            (time.strftime("%Y-%m-%d %H:%M:%S"),
             len(X), forest.oob_ate, forest.oob_ate_se,
             forest.oob_ate - 1.96 * forest.oob_ate_se,
             forest.oob_ate + 1.96 * forest.oob_ate_se,
             pol["policy_value"], str(ckpt)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {
        "status": "ok",
        "n": int(len(X)),
        "ate": forest.oob_ate,
        "ate_se": forest.oob_ate_se,
        "ate_ci95": (forest.oob_ate - 1.96 * forest.oob_ate_se,
                     forest.oob_ate + 1.96 * forest.oob_ate_se),
        "cate_bottom_decile": float(bottom),
        "cate_top_decile": float(top),
        "heterogeneity_ratio": float(abs(top - bottom) / max(abs(forest.oob_ate), 1e-6)),
        "policy": pol,
        "checkpoint": str(ckpt),
    }


def _build_followup_frame(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
              e.MEMBER_ID, e.DISCHARGE_DATE,
              COALESCE(e.LENGTH_OF_STAY,0) AS LOS_DAYS,
              0 AS TOTAL_PAID,
              CAST(COALESCE((JULIANDAY('now') - JULIANDAY(m.DATE_OF_BIRTH)) / 365.25, 40) AS INTEGER) AS AGE,
              m.GENDER
            FROM encounters e
            JOIN members m ON e.MEMBER_ID = m.MEMBER_ID
            WHERE e.DISCHARGE_DATE IS NOT NULL
        """)
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return None, None, None, []
    X_list, Y_list, T_list = [], [], []
    for mid, ddate, los, paid, age, gender in rows:
        if not ddate:
            continue
        try:
            cur.execute("""
                SELECT COUNT(*) FROM encounters
                WHERE MEMBER_ID = ? AND ADMIT_DATE > ? AND ADMIT_DATE <= date(?, '+14 day')
                  AND VISIT_TYPE = 'OUTPATIENT'
            """, (mid, ddate, ddate))
            T = 1 if cur.fetchone()[0] > 0 else 0
            cur.execute("""
                SELECT COUNT(*) FROM encounters
                WHERE MEMBER_ID = ? AND ADMIT_DATE > ? AND ADMIT_DATE <= date(?, '+30 day')
                  AND VISIT_TYPE IN ('INPATIENT', 'EMERGENCY')
            """, (mid, ddate, ddate))
            Y = 1 if cur.fetchone()[0] > 0 else 0
            X_list.append([
                float(age or 60),
                1.0 if gender == "F" else 0.0,
                float(los or 0),
                math.log1p(float(paid or 0)),
                0.0,
                0.0,
            ])
            Y_list.append(Y)
            T_list.append(T)
        except Exception:
            continue
    conn.close()
    names = ["age", "gender_f", "los_days", "log_total_paid",
             "prior_admits", "chronic_count"]
    return (np.array(X_list), np.array(Y_list), np.array(T_list), names)


def get_causal_forest_status(db_path: str) -> Dict:
    out: Dict = {"checkpoints_on_disk": [], "recent_fits": []}
    for p in FOREST_DIR.glob("*.pkl") if FOREST_DIR.exists() else []:
        out["checkpoints_on_disk"].append({"name": p.name, "size_kb": p.stat().st_size // 1024})
    try:
        conn = sqlite3.connect(db_path)
        _ensure_log_table(conn)
        cur = conn.execute(
            f"""SELECT TIMESTAMP, OUTCOME, TREATMENT, N, ATE, ATE_SE,
                       ATE_CI_LO, ATE_CI_HI, POLICY_VALUE, CHECKPOINT
                FROM {FOREST_LOG_TABLE} ORDER BY LOG_ID DESC LIMIT 20"""
        )
        out["recent_fits"] = [
            {"ts": r[0], "outcome": r[1], "treatment": r[2], "n": r[3],
             "ate": r[4], "ate_se": r[5], "ate_ci95": (r[6], r[7]),
             "policy_value": r[8], "checkpoint": r[9]}
            for r in cur.fetchall()
        ]
        conn.close()
    except Exception as e:
        out["error"] = str(e)
    return out
