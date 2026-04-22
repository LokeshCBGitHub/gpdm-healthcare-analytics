from __future__ import annotations

import os, json, sqlite3, time, copy, hashlib
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


_ROOT = Path('data/models/federated')
_ROOT.mkdir(parents=True, exist_ok=True)
_LOG_TABLE = 'gpdm_federated_log'


def _sd_zero_like(sd):
    return {k: torch.zeros_like(v) for k, v in sd.items()}


def _sd_add_(acc, other, weight: float = 1.0):
    for k in acc:
        acc[k].add_(other[k].to(acc[k].dtype) * weight)
    return acc


def _sd_scale_(sd, s: float):
    for k in sd:
        sd[k].mul_(s)
    return sd


def _sd_diff(a, b):
    return {k: (a[k] - b[k]) for k in a}


def _sd_norm(sd) -> float:
    total = 0.0
    for v in sd.values():
        total += float((v.float() ** 2).sum())
    return float(total ** 0.5)


@dataclass
class ClientState:
    client_id: str
    n_samples: int
    root: Path

    @classmethod
    def make(cls, client_id: str, n_samples: int) -> "ClientState":
        root = _ROOT / 'clients' / client_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(client_id=client_id, n_samples=n_samples, root=root)

    def save_update(self, round_id: int, state_dict) -> Path:
        p = self.root / f"update_r{round_id:03d}.pt"
        torch.save(state_dict, p)
        return p

    def load_update(self, round_id: int):
        p = self.root / f"update_r{round_id:03d}.pt"
        return torch.load(p, map_location='cpu') if p.exists() else None


def _derive_mask(seed_a: str, seed_b: str, shapes) -> Dict:
    h = hashlib.sha256((seed_a + '|' + seed_b).encode()).digest()
    seed = int.from_bytes(h[:8], 'little')
    rng = np.random.default_rng(seed)
    mask = {}
    for k, shape in shapes.items():
        mask[k] = torch.from_numpy(rng.standard_normal(shape).astype('float32'))
    return mask


def apply_secure_aggregation_masks(updates: List[Dict],
                                     client_ids: List[str]) -> List[Dict]:
    if len(updates) != len(client_ids):
        raise ValueError("updates and client_ids must match")
    shapes = {k: tuple(v.shape) for k, v in updates[0].items()}
    masked = [copy.deepcopy(u) for u in updates]
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):
            m = _derive_mask(client_ids[i], client_ids[j], shapes)
            for k in masked[i]:
                masked[i][k] = masked[i][k] + m[k].to(masked[i][k].dtype)
                masked[j][k] = masked[j][k] - m[k].to(masked[j][k].dtype)
    return masked


def fedavg_aggregate(updates: List[Dict], weights: List[float]) -> Dict:
    tot = float(sum(weights)) or 1.0
    acc = _sd_zero_like(updates[0])
    for u, w in zip(updates, weights):
        _sd_add_(acc, u, weight=w / tot)
    return acc


def client_update_fedprox(model, loss_fn, data_iter, *,
                           global_state, mu: float = 0.01,
                           local_epochs: int = 1, lr: float = 1e-3) -> Dict:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    global_copy = {k: v.detach().clone() for k, v in global_state.items()}

    for _ in range(local_epochs):
        for xb, yb in data_iter:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            prox = 0.0
            for name, p in model.named_parameters():
                if name in global_copy:
                    prox = prox + ((p - global_copy[name].to(p.dtype)) ** 2).sum()
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def orchestrate_round(round_id: int, global_state: Dict,
                       client_states: List[ClientState],
                       use_secure_agg: bool = False) -> Dict:
    updates = []
    weights = []
    ids = []
    for cs in client_states:
        u = cs.load_update(round_id)
        if u is None:
            continue
        updates.append(u)
        weights.append(float(cs.n_samples))
        ids.append(cs.client_id)

    if not updates:
        return {'status': 'no_updates', 'round_id': round_id}

    if use_secure_agg and len(updates) >= 2:
        updates = apply_secure_aggregation_masks(updates, ids)

    new_global = fedavg_aggregate(updates, weights)
    gpath = _ROOT / f"global_r{round_id:03d}.pt"
    torch.save(new_global, gpath)

    drift = float(_sd_norm(_sd_diff(new_global, global_state)))
    return {'status': 'ok', 'round_id': round_id,
            'n_clients': len(updates),
            'total_samples': int(sum(weights)),
            'drift_vs_prev_global': drift,
            'global_path': str(gpath),
            'secure_agg': use_secure_agg,
            'client_ids': ids}


def log_round(db_path: str, summary: Dict) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            round_id INTEGER,
            n_clients INTEGER,
            total_samples INTEGER,
            drift REAL,
            secure_agg INTEGER,
            details TEXT
        )
    """)
    conn.execute(
        f"INSERT INTO {_LOG_TABLE}(ts,round_id,n_clients,total_samples,drift,"
        f"secure_agg,details) VALUES(?,?,?,?,?,?,?)",
        (time.time(), int(summary.get('round_id', -1)),
         int(summary.get('n_clients', 0)),
         int(summary.get('total_samples', 0)),
         float(summary.get('drift_vs_prev_global', 0.0)),
         1 if summary.get('secure_agg') else 0,
         json.dumps({k: v for k, v in summary.items()
                      if k not in ('global_path',)}))
    )
    conn.commit(); conn.close()


def simulate_regions(db_path: str, n_rounds: int = 3,
                      use_secure_agg: bool = True) -> Dict:
    if not _TORCH:
        return {'status': 'torch_missing'}
    if not os.path.exists(db_path):
        return {'status': 'no_db'}

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT features_json, label FROM gpdm_feature_frame "
            "WHERE label IS NOT NULL LIMIT 50000"
        ).fetchall()
    except Exception:
        rows = []
    conn.close()
    if len(rows) < 300:
        return {'status': 'insufficient_data', 'n': len(rows)}

    import json as _json
    X = np.array([list(_json.loads(r[0]).values()) for r in rows], dtype=np.float32)
    y = np.array([int(r[1]) for r in rows], dtype=np.float32)
    d = X.shape[1]

    rng = np.random.default_rng(0)
    assign = rng.integers(0, 3, size=len(X))
    partitions = []
    for k in range(3):
        idx = np.where(assign == k)[0]
        partitions.append((X[idx], y[idx]))

    def make_model():
        return nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    global_model = make_model()
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    torch.save(global_state, _ROOT / 'global_r000.pt')

    client_states = [ClientState.make(f'region_{i}', n_samples=len(partitions[i][0]))
                     for i in range(3)]

    loss_fn = nn.BCEWithLogitsLoss()
    history = []

    for r in range(1, n_rounds + 1):
        for ci, cs in enumerate(client_states):
            Xp, yp = partitions[ci]
            if len(Xp) < 10:
                continue
            local = make_model()
            local.load_state_dict({k: v.clone() for k, v in global_state.items()})
            bs = 64
            order = rng.permutation(len(Xp))
            batches = []
            for s in range(0, len(Xp), bs):
                b = order[s:s + bs]
                batches.append((torch.from_numpy(Xp[b]),
                                 torch.from_numpy(yp[b]).unsqueeze(-1)))
            new_state = client_update_fedprox(
                local, loss_fn, batches,
                global_state=global_state, mu=0.01,
                local_epochs=1, lr=1e-2,
            )
            cs.save_update(r, new_state)

        summary = orchestrate_round(r, global_state, client_states,
                                      use_secure_agg=use_secure_agg)
        if summary.get('status') == 'ok':
            global_state = torch.load(summary['global_path'], map_location='cpu')
            log_round(db_path, summary)
            history.append(summary)

    return {'status': 'ok', 'rounds': n_rounds, 'history': history,
            'n_regions': 3, 'n_samples': int(len(X)),
            'secure_agg': use_secure_agg}


def get_federated_status(db_path: str) -> Dict:
    out = {'status': 'ok', 'torch': _TORCH,
            'global_models': sorted(p.name for p in _ROOT.glob('global_r*.pt')),
            'clients': sorted(p.name for p in (_ROOT / 'clients').glob('*')
                               if p.is_dir()) if (_ROOT / 'clients').exists() else []}
    if not os.path.exists(db_path):
        out['db'] = 'missing'; return out
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*), MAX(round_id), MAX(ts) FROM {_LOG_TABLE}")
        r = cur.fetchone()
        out['n_log_rows'] = int(r[0] or 0)
        out['last_round'] = r[1]; out['last_ts'] = r[2]
    except Exception:
        pass
    conn.close()
    return out
