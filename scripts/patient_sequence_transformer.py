from __future__ import annotations

import os, json, sqlite3, time, math, pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except Exception:
    _TORCH = False


_MODEL_DIR = Path('data/models/seq_transformer')
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
_LOG_TABLE = 'gpdm_seq_transformer_log'


@dataclass
class SeqVocab:
    code2id: Dict[str, int] = field(default_factory=dict)
    id2code: Dict[int, str] = field(default_factory=dict)
    special: Dict[str, int] = field(default_factory=lambda: {
        '[PAD]': 0, '[CLS]': 1, '[UNK]': 2, '[SEP]': 3,
    })

    def build(self, codes: Sequence[str], min_freq: int = 2) -> None:
        from collections import Counter
        c = Counter(codes)
        self.code2id = dict(self.special)
        next_id = max(self.special.values()) + 1
        for code, cnt in c.most_common():
            if cnt < min_freq or code in self.code2id:
                continue
            self.code2id[code] = next_id
            next_id += 1
        self.id2code = {v: k for k, v in self.code2id.items()}

    def encode(self, code: str) -> int:
        return self.code2id.get(code, self.special['[UNK]'])

    def __len__(self) -> int:
        return len(self.code2id)


def _load_encounter_sequences(db_path: str, horizon_days: int = 30,
                               max_seq_len: int = 128) -> Tuple[List, List, SeqVocab]:
    if not os.path.exists(db_path):
        return [], [], SeqVocab()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(encounters)")
        cols = [r[1] for r in cur.fetchall()]
    except Exception:
        conn.close(); return [], [], SeqVocab()
    if 'member_id' not in cols:
        conn.close(); return [], [], SeqVocab()

    date_col = next((c for c in ['encounter_date', 'date', 'svc_date', 'service_date']
                     if c in cols), None)
    code_col = next((c for c in ['primary_dx', 'dx1', 'diagnosis', 'dx_code',
                                  'cpt', 'cpt_code'] if c in cols), None)
    type_col = next((c for c in ['encounter_type', 'type', 'setting', 'pos']
                     if c in cols), None)
    admit_col = next((c for c in ['is_admission', 'inpatient', 'admit_flag']
                      if c in cols), None)

    if not date_col or not code_col:
        conn.close(); return [], [], SeqVocab()

    type_expr = ("COALESCE(" + type_col + ",'OP')") if type_col else "'OP'"
    admit_expr = admit_col if admit_col else '0'
    q = (f"SELECT member_id, {date_col}, COALESCE({code_col},'UNK'), "
         f"{type_expr}, {admit_expr} "
         f"FROM encounters WHERE {date_col} IS NOT NULL "
         f"ORDER BY member_id, {date_col} LIMIT 2000000")
    rows = cur.execute(q).fetchall()

    demo = {}
    try:
        cur.execute("SELECT member_id, gender, age, region FROM members")
        for m, g, a, r in cur.fetchall():
            demo[m] = {'gender': g or 'U', 'age': a if a is not None else -1,
                       'region': r or 'U'}
    except Exception:
        pass
    conn.close()

    from collections import defaultdict
    per_member: Dict[str, List[Tuple]] = defaultdict(list)
    for mid, date, code, etype, admit in rows:
        per_member[mid].append((date, str(code), str(etype), int(admit or 0)))

    all_tokens: List[str] = []
    for seq in per_member.values():
        for _, code, etype, _ in seq:
            all_tokens.append(f"DX::{code}")
            all_tokens.append(f"TYPE::{etype}")
    vocab = SeqVocab()
    vocab.build(all_tokens, min_freq=2)

    sequences = []
    labels = []
    for mid, seq in per_member.items():
        if len(seq) < 2:
            continue
        last_date = seq[-1][0]
        try:
            import datetime as _dt
            last_dt = _dt.datetime.fromisoformat(str(last_date)[:19])
        except Exception:
            continue
        label = 0
        for (d, _c, _t, admit) in seq:
            try:
                d_dt = _dt.datetime.fromisoformat(str(d)[:19])
            except Exception:
                continue
            if admit and 0 < (d_dt - last_dt).days <= horizon_days:
                label = 1
                break

        tokens = []
        prev_dt = None
        for (d, c, t, _a) in seq[:-1][-max_seq_len:]:
            try:
                d_dt = _dt.datetime.fromisoformat(str(d)[:19])
            except Exception:
                continue
            delta = 0 if prev_dt is None else max(0, (d_dt - prev_dt).days)
            tokens.append({
                'code_id': vocab.encode(f"DX::{c}"),
                'type_id': vocab.encode(f"TYPE::{t}"),
                'delta_bucket': _delta_bucket(delta),
            })
            prev_dt = d_dt

        if not tokens:
            continue

        d = demo.get(mid, {})
        sequences.append({
            'tokens': tokens,
            'age_bucket': _age_bucket(d.get('age', -1)),
            'gender': 0 if (d.get('gender') or 'U').upper().startswith('F') else 1,
        })
        labels.append(label)

    return sequences, labels, vocab


def _age_bucket(age) -> int:
    try:
        a = float(age)
    except Exception:
        return 0
    if a < 0: return 0
    if a < 18: return 1
    if a < 40: return 2
    if a < 65: return 3
    return 4


def _delta_bucket(days: int) -> int:
    if days <= 0: return 0
    if days <= 7: return 1
    if days <= 30: return 2
    if days <= 90: return 3
    if days <= 365: return 4
    return 5


if _TORCH:
    class EncounterTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                      n_layers: int = 2, ffn: int = 256, max_len: int = 130,
                      n_delta: int = 6, n_age: int = 5, n_gender: int = 2,
                      dropout: float = 0.2):
            super().__init__()
            self.code_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.type_emb = nn.Embedding(vocab_size, d_model // 2, padding_idx=0)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.delta_emb = nn.Embedding(n_delta, d_model // 2)
            self.age_emb = nn.Embedding(n_age, d_model)
            self.gender_emb = nn.Embedding(n_gender, d_model)
            self.fuse = nn.Linear(d_model + d_model // 2 + d_model // 2, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
                dropout=dropout, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.max_len = max_len

        def forward(self, code_ids, type_ids, delta_ids, age_id, gender_id,
                     attention_mask):
            B, L = code_ids.shape
            pos_ids = torch.arange(L, device=code_ids.device).unsqueeze(0).expand(B, L)
            code = self.code_emb(code_ids)
            typ = self.type_emb(type_ids)
            delta = self.delta_emb(delta_ids)
            fused = torch.cat([code, typ, delta], dim=-1)
            fused = self.fuse(fused) + self.pos_emb(pos_ids)
            cls = (self.age_emb(age_id) + self.gender_emb(gender_id)).unsqueeze(1)
            x = torch.cat([cls, fused], dim=1)
            pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                               device=code_ids.device),
                                    attention_mask], dim=1)
            h = self.encoder(x, src_key_padding_mask=pad_mask)
            logit = self.head(h[:, 0, :]).squeeze(-1)
            return logit


    class _SeqDataset(Dataset):
        def __init__(self, sequences, labels, max_len=128):
            self.seq = sequences
            self.y = labels
            self.max_len = max_len

        def __len__(self):
            return len(self.seq)

        def __getitem__(self, i):
            s = self.seq[i]
            toks = s['tokens'][-self.max_len:]
            L = len(toks)
            pad = self.max_len - L
            code_ids = [t['code_id'] for t in toks] + [0] * pad
            type_ids = [t['type_id'] for t in toks] + [0] * pad
            delta_ids = [t['delta_bucket'] for t in toks] + [0] * pad
            attn = [False] * L + [True] * pad
            return (
                torch.tensor(code_ids, dtype=torch.long),
                torch.tensor(type_ids, dtype=torch.long),
                torch.tensor(delta_ids, dtype=torch.long),
                torch.tensor(s['age_bucket'], dtype=torch.long),
                torch.tensor(s['gender'], dtype=torch.long),
                torch.tensor(attn, dtype=torch.bool),
                torch.tensor(float(self.y[i]), dtype=torch.float),
            )


def train_from_sqlite(db_path: str, horizon_days: int = 30,
                       epochs: int = 3, batch_size: int = 64,
                       lr: float = 3e-4, max_seq_len: int = 128) -> Dict:
    if not _TORCH:
        return {'status': 'torch_missing'}
    torch.manual_seed(0)
    np.random.seed(0)
    seqs, labels, vocab = _load_encounter_sequences(
        db_path, horizon_days=horizon_days, max_seq_len=max_seq_len)
    if len(seqs) < 50:
        return {'status': 'insufficient_data', 'n': len(seqs)}

    idx = np.random.permutation(len(seqs))
    split = int(0.8 * len(seqs))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_seqs = [seqs[i] for i in tr_idx]; tr_lab = [labels[i] for i in tr_idx]
    va_seqs = [seqs[i] for i in va_idx]; va_lab = [labels[i] for i in va_idx]

    train_ds = _SeqDataset(tr_seqs, tr_lab, max_len=max_seq_len)
    val_ds = _SeqDataset(va_seqs, va_lab, max_len=max_seq_len)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cpu')
    model = EncounterTransformer(
        vocab_size=max(4, len(vocab)),
        max_len=max_seq_len + 2,
    ).to(device)

    pos_w = float((len(tr_lab) - sum(tr_lab)) / max(1, sum(tr_lab)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best = {'val_auc': -1.0, 'epoch': -1}
    history = []

    from fairness import auc as _auc, brier as _brier, ece as _ece

    for ep in range(epochs):
        model.train()
        total = 0.0; n = 0
        for batch in train_ld:
            c, t, d, a, g, m, y = [b.to(device) for b in batch]
            opt.zero_grad()
            logit = model(c, t, d, a, g, m)
            loss = loss_fn(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(y); n += len(y)
        sched.step()

        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for batch in val_ld:
                c, t, d, a, g, m, y = [b.to(device) for b in batch]
                logit = model(c, t, d, a, g, m)
                ps.extend(torch.sigmoid(logit).cpu().numpy().tolist())
                ys.extend(y.cpu().numpy().tolist())
        ys_a = np.array(ys); ps_a = np.array(ps)
        va_auc = float(_auc(ps_a, ys_a.astype(int)))
        va_brier = float(_brier(ps_a, ys_a))
        va_ece = float(_ece(ps_a, ys_a.astype(int)))
        history.append({'epoch': ep, 'train_loss': total / max(1, n),
                         'val_auc': va_auc, 'val_brier': va_brier, 'val_ece': va_ece})
        if va_auc > best['val_auc']:
            best = {'val_auc': va_auc, 'val_brier': va_brier, 'val_ece': va_ece,
                    'epoch': ep}
            torch.save(model.state_dict(), _MODEL_DIR / 'model.pt')
            with open(_MODEL_DIR / 'vocab.pkl', 'wb') as f:
                pickle.dump(vocab, f)
            with open(_MODEL_DIR / 'meta.json', 'w') as f:
                json.dump({
                    'horizon_days': horizon_days,
                    'max_seq_len': max_seq_len,
                    'vocab_size': len(vocab),
                    'd_model': 128, 'n_heads': 4, 'n_layers': 2,
                }, f)

    _log(db_path, history, best)
    return {'status': 'ok', 'best': best, 'history': history,
            'n_train': len(tr_lab), 'n_val': len(va_lab)}


def _log(db_path: str, history: List[Dict], best: Dict) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            epoch INTEGER,
            val_auc REAL, val_brier REAL, val_ece REAL,
            train_loss REAL, best INTEGER
        )
    """)
    ts = time.time()
    for h in history:
        is_best = 1 if h.get('epoch') == best.get('epoch') else 0
        conn.execute(
            f"INSERT INTO {_LOG_TABLE}(ts,epoch,val_auc,val_brier,val_ece,train_loss,best) "
            f"VALUES(?,?,?,?,?,?,?)",
            (ts, h['epoch'], h['val_auc'], h['val_brier'], h['val_ece'],
             h['train_loss'], is_best)
        )
    conn.commit(); conn.close()


def predict_for_patient(member_id: str, db_path: str,
                          max_seq_len: int = 128) -> Dict:
    if not _TORCH:
        return {'status': 'torch_missing'}
    model_path = _MODEL_DIR / 'model.pt'
    vocab_path = _MODEL_DIR / 'vocab.pkl'
    meta_path = _MODEL_DIR / 'meta.json'
    if not (model_path.exists() and vocab_path.exists()):
        return {'status': 'untrained'}

    with open(vocab_path, 'rb') as f:
        vocab: SeqVocab = pickle.load(f)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(encounters)")
    cols = [r[1] for r in cur.fetchall()]
    date_col = next((c for c in ['encounter_date', 'date', 'svc_date'] if c in cols), None)
    code_col = next((c for c in ['primary_dx', 'dx1', 'diagnosis', 'dx_code'] if c in cols), None)
    type_col = next((c for c in ['encounter_type', 'type', 'setting'] if c in cols), None)
    if not date_col or not code_col:
        conn.close(); return {'status': 'schema_unsupported'}
    type_expr = ("COALESCE(" + type_col + ",'OP')") if type_col else "'OP'"
    q = (f"SELECT {date_col}, COALESCE({code_col},'UNK'), {type_expr} "
         f"FROM encounters WHERE member_id=? ORDER BY {date_col} LIMIT ?")
    rows = cur.execute(q, (member_id, max_seq_len * 2)).fetchall()
    try:
        cur.execute("SELECT gender, age FROM members WHERE member_id=?", (member_id,))
        d = cur.fetchone() or ('U', -1)
    except Exception:
        d = ('U', -1)
    conn.close()
    if not rows:
        return {'status': 'no_history', 'member_id': member_id}

    import datetime as _dt
    toks = []; prev = None
    for r in rows[-max_seq_len:]:
        try:
            d_dt = _dt.datetime.fromisoformat(str(r[0])[:19])
        except Exception:
            continue
        delta = 0 if prev is None else max(0, (d_dt - prev).days)
        toks.append({
            'code_id': vocab.encode(f"DX::{r[1]}"),
            'type_id': vocab.encode(f"TYPE::{r[2]}"),
            'delta_bucket': _delta_bucket(delta),
        })
        prev = d_dt
    if not toks:
        return {'status': 'no_history'}

    L = len(toks); pad = max_seq_len - L
    c = torch.tensor([[t['code_id'] for t in toks] + [0] * pad], dtype=torch.long)
    t = torch.tensor([[t['type_id'] for t in toks] + [0] * pad], dtype=torch.long)
    dd = torch.tensor([[t['delta_bucket'] for t in toks] + [0] * pad], dtype=torch.long)
    m = torch.tensor([[False] * L + [True] * pad], dtype=torch.bool)
    a = torch.tensor([_age_bucket(d[1])], dtype=torch.long)
    g = torch.tensor([0 if str(d[0] or 'U').upper().startswith('F') else 1],
                      dtype=torch.long)

    model = EncounterTransformer(
        vocab_size=max(4, len(vocab)),
        max_len=max_seq_len + 2,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logit = model(c, t, dd, a, g, m)
        prob = float(torch.sigmoid(logit).item())

    return {'status': 'ok', 'member_id': member_id, 'risk': prob,
             'horizon_days': int(meta.get('horizon_days', 30)),
             'n_tokens': L}


def get_seq_status(db_path: str) -> Dict:
    out = {'status': 'ok', 'torch_available': _TORCH,
            'model_trained': (_MODEL_DIR / 'model.pt').exists()}
    if not os.path.exists(db_path):
        out['db'] = 'missing'; return out
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT MAX(ts), MAX(val_auc) FROM {_LOG_TABLE}")
        r = cur.fetchone()
        out['last_trained_ts'] = r[0]; out['best_auc'] = r[1]
    except Exception:
        pass
    conn.close()
    return out
