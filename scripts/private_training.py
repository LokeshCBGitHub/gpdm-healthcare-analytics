from __future__ import annotations

import os, json, math, sqlite3, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Sequence

try:
    import numpy as np
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:
    _TORCH = False


_LOG_TABLE = 'gpdm_dp_training_log'
_MODEL_DIR = Path('data/models/dp_sgd')
_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _compute_rdp_gaussian(q: float, sigma: float, steps: int,
                           orders: Sequence[float]) -> List[float]:
    def rdp_gauss(alpha, sigma):
        return alpha / (2 * sigma * sigma)

    def log_add(a, b):
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))

    def log_sub(a, b):
        if a <= b:
            return -float('inf')
        return a + math.log1p(-math.exp(b - a))

    def rdp_subsampled(alpha, q, sigma):
        if q == 0.0:
            return 0.0
        if q == 1.0:
            return rdp_gauss(alpha, sigma)
        if not float(alpha).is_integer():
            lo = int(math.floor(alpha)); hi = int(math.ceil(alpha))
            if hi == lo:
                return rdp_subsampled(int(alpha), q, sigma)
            return ((hi - alpha) * rdp_subsampled(lo, q, sigma)
                    + (alpha - lo) * rdp_subsampled(hi, q, sigma))
        alpha = int(alpha)
        log_terms = []
        log_q = math.log(q)
        log_1mq = math.log(1 - q)
        for k in range(alpha + 1):
            log_coef = (math.lgamma(alpha + 1)
                         - math.lgamma(k + 1)
                         - math.lgamma(alpha - k + 1))
            log_terms.append(log_coef + (alpha - k) * log_1mq + k * log_q
                              + (k * k - k) / (2 * sigma * sigma))
        m = max(log_terms)
        total = m + math.log(sum(math.exp(x - m) for x in log_terms))
        return total / (alpha - 1)

    return [steps * rdp_subsampled(a, q, sigma) for a in orders]


def _rdp_to_eps(rdp: Sequence[float], orders: Sequence[float],
                 delta: float) -> float:
    eps_vals = [r + math.log(1 / delta) / (a - 1) if a > 1 else float('inf')
                for r, a in zip(rdp, orders)]
    return float(min(eps_vals))


@dataclass
class DPAccountant:
    orders: List[float] = field(default_factory=lambda:
        [1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 7., 8., 10., 12., 15., 20., 32., 64.])
    rdp: List[float] = field(default_factory=list)

    def step(self, q: float, sigma: float, n_steps: int = 1) -> None:
        new_rdp = _compute_rdp_gaussian(q, sigma, n_steps, self.orders)
        if not self.rdp:
            self.rdp = list(new_rdp)
        else:
            self.rdp = [a + b for a, b in zip(self.rdp, new_rdp)]

    def epsilon(self, delta: float) -> float:
        if not self.rdp:
            return 0.0
        return _rdp_to_eps(self.rdp, self.orders, delta)


def dp_sgd_step(model, loss_fn, inputs, targets, *,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.1,
                 optimizer,
                 microbatch_size: int = 1) -> float:
    if not _TORCH:
        raise RuntimeError("torch not available")

    params = [p for p in model.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in params]
    B = inputs.shape[0]
    n_micro = math.ceil(B / microbatch_size)
    total_loss = 0.0

    for mb in range(n_micro):
        s = mb * microbatch_size
        e = min(B, s + microbatch_size)
        x_mb = inputs[s:e]
        y_mb = targets[s:e]

        optimizer.zero_grad(set_to_none=True)
        out = model(x_mb)
        loss = loss_fn(out, y_mb)
        loss.backward()
        total_loss += float(loss.detach()) * (e - s)

        grads = [p.grad.detach().clone() if p.grad is not None
                 else torch.zeros_like(p) for p in params]
        gnorm = torch.sqrt(sum((g * g).sum() for g in grads) + 1e-12)
        factor = min(1.0, max_grad_norm / float(gnorm))
        for a, g in zip(accum, grads):
            a.add_(g * factor)

    sigma = noise_multiplier * max_grad_norm
    optimizer.zero_grad(set_to_none=True)
    for p, a in zip(params, accum):
        noise = torch.randn_like(p) * sigma
        p.grad = (a + noise) / float(B)
    optimizer.step()

    return total_loss / max(1, B)


def train_risk_net_dp(db_path: str, epochs: int = 5,
                        batch_size: int = 256, lr: float = 5e-4,
                        max_grad_norm: float = 1.0,
                        noise_multiplier: float = 1.1,
                        target_delta: float = 1e-5) -> Dict:
    if not _TORCH:
        return {'status': 'torch_missing'}

    if not os.path.exists(db_path):
        return {'status': 'no_db'}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT features_json, label FROM gpdm_feature_frame "
            "WHERE label IS NOT NULL LIMIT 500000"
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()

    if len(rows) < 200:
        return {'status': 'insufficient_data', 'n': len(rows)}

    import json as _json
    X = np.array([list(_json.loads(r[0]).values()) for r in rows], dtype=np.float32)
    y = np.array([int(r[1]) for r in rows], dtype=np.float32)
    d = X.shape[1]

    model = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    acct = DPAccountant()
    N = len(X)
    q = batch_size / N
    steps_per_epoch = max(1, N // batch_size)
    history = []

    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(N)
        ep_loss = 0.0
        for s in range(0, N, batch_size):
            b_idx = idx[s:s + batch_size]
            xb = torch.from_numpy(X[b_idx])
            yb = torch.from_numpy(y[b_idx]).unsqueeze(-1)
            loss = dp_sgd_step(
                model, loss_fn, xb, yb,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                optimizer=opt,
                microbatch_size=1,
            )
            ep_loss += loss
            acct.step(q, noise_multiplier, n_steps=1)
        eps = acct.epsilon(target_delta)
        history.append({'epoch': ep, 'train_loss': ep_loss / max(1, steps_per_epoch),
                         'epsilon': eps, 'delta': target_delta})

    torch.save(model.state_dict(), _MODEL_DIR / 'risk_net_dp.pt')
    (_MODEL_DIR / 'meta.json').write_text(json.dumps({
        'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
        'noise_multiplier': noise_multiplier, 'max_grad_norm': max_grad_norm,
        'final_epsilon': history[-1]['epsilon'] if history else None,
        'delta': target_delta, 'N': int(N), 'features': int(d),
    }))

    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, epoch INTEGER, train_loss REAL,
            epsilon REAL, delta REAL
        )
    """)
    ts = time.time()
    for h in history:
        conn.execute(
            f"INSERT INTO {_LOG_TABLE}(ts,epoch,train_loss,epsilon,delta) "
            f"VALUES(?,?,?,?,?)",
            (ts, h['epoch'], h['train_loss'], h['epsilon'], h['delta'])
        )
    conn.commit(); conn.close()

    return {
        'status': 'ok',
        'history': history,
        'final_epsilon': history[-1]['epsilon'] if history else None,
        'delta': target_delta,
        'guarantee': f"({history[-1]['epsilon']:.2f}, {target_delta:.1e})-DP"
                      if history else None,
        'n': int(N), 'features': int(d),
    }


def get_dp_status(db_path: str) -> Dict:
    out = {'status': 'ok', 'torch': _TORCH,
            'model_saved': (_MODEL_DIR / 'risk_net_dp.pt').exists()}
    if not os.path.exists(db_path):
        out['db'] = 'missing'; return out
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT MAX(ts), MIN(epsilon), MAX(epsilon) FROM {_LOG_TABLE}")
        r = cur.fetchone()
        out['last_run_ts'] = r[0]; out['eps_min'] = r[1]; out['eps_final'] = r[2]
    except Exception:
        pass
    conn.close()
    meta = _MODEL_DIR / 'meta.json'
    if meta.exists():
        out['meta'] = json.loads(meta.read_text())
    return out
