from __future__ import annotations

import json
import logging
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _repo_root() -> str:
    return (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _models_dir() -> str:
    d = os.path.join(_repo_root(), "data", "models")
    os.makedirs(d, exist_ok=True)
    return d


def _default_db() -> str:
    return os.path.join(_repo_root(), "data", "healthcare_demo.db")


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _TORCH = (torch, nn, F)
    except Exception as e:
        logger.warning("torch unavailable: %s", e)
        _TORCH = False
    return _TORCH


MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"

VISIT_TYPES = ["PAD", "UNK", "EMERGENCY", "INPATIENT", "OUTPATIENT",
               "URGENT_CARE", "TELEHEALTH", "PREVENTIVE", "OFFICE"]
AGE_BUCKETS = ["0-17", "18-29", "30-44", "45-54", "55-64", "65-74", "75-84", "85+"]


def _age_bucket(dob: str, event_date: str) -> int:
    try:
        by = int((dob or "")[:4])
        ey = int((event_date or "")[:4])
        age = max(0, ey - by)
    except Exception:
        age = 40
    if age < 18: return 0
    if age < 30: return 1
    if age < 45: return 2
    if age < 55: return 3
    if age < 65: return 4
    if age < 75: return 5
    if age < 85: return 6
    return 7


def _delta_bucket(days: int) -> int:
    if days <= 0: return 0
    if days <= 7: return 1
    if days <= 30: return 2
    if days <= 90: return 3
    if days <= 180: return 4
    if days <= 365: return 5
    if days <= 730: return 6
    return 7


def _visit_idx(v: str) -> int:
    v = (v or "").upper().strip()
    try:
        return VISIT_TYPES.index(v)
    except ValueError:
        return 1


@dataclass
class PatientTimeline:
    member_id: str
    codes: List[str]
    visits: List[int]
    ages: List[int]
    deltas: List[int]
    labels: Dict[str, int] = field(default_factory=dict)


def build_timelines(db_path: str) -> List[PatientTimeline]:
    timelines: Dict[str, PatientTimeline] = {}
    dob: Dict[str, str] = {}
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT MEMBER_ID, DATE_OF_BIRTH FROM members"):
            dob[r["MEMBER_ID"]] = r["DATE_OF_BIRTH"] or ""
            timelines[r["MEMBER_ID"]] = PatientTimeline(
                member_id=r["MEMBER_ID"], codes=[], visits=[], ages=[], deltas=[])

        enc_events: Dict[str, List[Tuple[str, str]]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, COALESCE(ADMIT_DATE, SERVICE_DATE) d, VISIT_TYPE "
            "FROM encounters WHERE COALESCE(ADMIT_DATE, SERVICE_DATE) IS NOT NULL"):
            enc_events.setdefault(r["MEMBER_ID"], []).append(
                (r["d"] or "", r["VISIT_TYPE"] or ""))

        dx_by_m: Dict[str, List[Tuple[str, str]]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, ICD10_CODE, COALESCE(DIAGNOSIS_DATE, '') d "
            "FROM diagnoses WHERE ICD10_CODE IS NOT NULL"):
            dx_by_m.setdefault(r["MEMBER_ID"], []).append(
                (r["d"] or "", r["ICD10_CODE"]))

        costs: Dict[str, float] = {}
        ipd: Dict[str, float] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, COALESCE(SUM(PAID_AMOUNT),0) c FROM claims GROUP BY MEMBER_ID"):
            costs[r["MEMBER_ID"]] = float(r["c"])
        for r in c.execute(
            "SELECT MEMBER_ID, COALESCE(SUM(LENGTH_OF_STAY),0) d FROM encounters "
            "WHERE VISIT_TYPE='INPATIENT' GROUP BY MEMBER_ID"):
            ipd[r["MEMBER_ID"]] = float(r["d"])

        ip_rows: Dict[str, List[Tuple[str, str]]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, ADMIT_DATE, DISCHARGE_DATE FROM encounters "
            "WHERE VISIT_TYPE='INPATIENT' AND DISCHARGE_DATE IS NOT NULL "
            "ORDER BY MEMBER_ID, DISCHARGE_DATE"):
            ip_rows.setdefault(r["MEMBER_ID"], []).append(
                (r["ADMIT_DATE"] or "", r["DISCHARGE_DATE"] or ""))

        er_rows: Dict[str, List[str]] = {}
        for r in c.execute(
            "SELECT MEMBER_ID, COALESCE(ADMIT_DATE, SERVICE_DATE) d FROM encounters "
            "WHERE VISIT_TYPE='EMERGENCY'"):
            er_rows.setdefault(r["MEMBER_ID"], []).append(r["d"] or "")

    all_costs = list(costs.values()) or [0.0]
    all_ipd = list(ipd.values()) or [0.0]
    cost_thr = float(np.quantile(all_costs, 0.80))
    ipd_thr = float(np.quantile(all_ipd, 0.80))

    def _parse(d: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(d)[:10])
        except Exception:
            return None

    def _readmit(mid: str) -> int:
        rows = ip_rows.get(mid, [])
        for i in range(len(rows) - 1):
            d1, d2 = _parse(rows[i][1]), _parse(rows[i + 1][0])
            if d1 and d2 and 0 <= (d2 - d1).days <= 30:
                return 1
        return 0

    def _future_er(mid: str, cutoff: Optional[datetime]) -> int:
        if cutoff is None:
            return 0
        for d in er_rows.get(mid, []):
            ed = _parse(d)
            if ed and 0 < (ed - cutoff).days <= 90:
                return 1
        return 0

    out: List[PatientTimeline] = []
    for mid, tl in timelines.items():
        events: List[Tuple[str, str, str]] = []
        for d, v in enc_events.get(mid, []):
            events.append((d, v, ""))
        for d, c_ in dx_by_m.get(mid, []):
            events.append((d, "", c_))
        if not events:
            continue
        events.sort(key=lambda x: x[0] or "")

        prev = None
        for d, v, code in events:
            dt = _parse(d)
            delta_days = 0 if prev is None or dt is None else max(0, (dt - prev).days)
            tl.codes.append(code or (f"VISIT_{v}" if v else UNK_TOKEN))
            tl.visits.append(_visit_idx(v))
            tl.ages.append(_age_bucket(dob.get(mid, ""), d))
            tl.deltas.append(_delta_bucket(delta_days))
            prev = dt or prev

        last = _parse(events[-1][0])
        cutoff = last
        tl.labels = {
            "high_cost": 1 if costs.get(mid, 0.0) >= cost_thr else 0,
            "high_util": 1 if ipd.get(mid, 0.0) >= ipd_thr else 0,
            "readmit_30d": _readmit(mid),
            "future_er_90d": _future_er(mid, cutoff),
        }
        if tl.codes:
            out.append(tl)
    logger.info("Built %d patient timelines (median len=%.0f)", len(out),
                float(np.median([len(t.codes) for t in out])) if out else 0)
    return out


class CodeVocab:
    def __init__(self):
        self.id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1,
                                    CLS_TOKEN: 2, MASK_TOKEN: 3}

    def fit(self, timelines: List[PatientTimeline], min_count: int = 1) -> None:
        from collections import Counter
        ct = Counter()
        for t in timelines:
            ct.update(t.codes)
        for tok, n in ct.most_common():
            if n >= min_count and tok not in self.id:
                self.id[tok] = len(self.id)

    def __len__(self):
        return len(self.id)

    def get(self, tok: str) -> int:
        return self.id.get(tok, 1)

    def to_json(self) -> str:
        return json.dumps(self.id)

    @classmethod
    def from_json(cls, s: str) -> "CodeVocab":
        v = cls()
        v.id = json.loads(s)
        return v


MAX_LEN = 128


def _build_model(n_codes: int, d_model: int = 192, nhead: int = 6,
                 nlayers: int = 4, dropout: float = 0.1):
    t = _torch()
    if not t:
        return None
    torch, nn, F = t

    class PatientTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
            self.visit_emb = nn.Embedding(len(VISIT_TYPES), d_model)
            self.age_emb = nn.Embedding(len(AGE_BUCKETS), d_model)
            self.delta_emb = nn.Embedding(8, d_model)
            self.pos_emb = nn.Embedding(MAX_LEN + 1, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 3,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

            self.mlm_head = nn.Linear(d_model, n_codes)
            self.heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                ) for name in ("high_cost", "high_util", "readmit_30d", "future_er_90d")
            })

        def _embed(self, codes, visits, ages, deltas):
            B, T = codes.shape
            pos = torch.arange(T, device=codes.device).unsqueeze(0).expand(B, T)
            x = (self.code_emb(codes) + self.visit_emb(visits)
                 + self.age_emb(ages) + self.delta_emb(deltas)
                 + self.pos_emb(pos))
            return self.drop(self.ln(x))

        def forward(self, codes, visits, ages, deltas, mask, mlm: bool = False):
            x = self._embed(codes, visits, ages, deltas)
            kpm = ~mask
            h = self.encoder(x, src_key_padding_mask=kpm)
            cls = h[:, 0, :]
            head_logits = {name: layer(cls).squeeze(-1) for name, layer in self.heads.items()}
            if mlm:
                return head_logits, self.mlm_head(h)
            return head_logits

    return PatientTransformer()


def _encode_timeline(tl: PatientTimeline, vocab: CodeVocab) -> Dict[str, np.ndarray]:
    codes = [vocab.get(CLS_TOKEN)] + [vocab.get(c) for c in tl.codes][: MAX_LEN - 1]
    visits = [0] + tl.visits[: MAX_LEN - 1]
    ages = [0] + tl.ages[: MAX_LEN - 1]
    deltas = [0] + tl.deltas[: MAX_LEN - 1]
    L = len(codes)
    pad = MAX_LEN - L
    codes += [0] * pad; visits += [0] * pad; ages += [0] * pad; deltas += [0] * pad
    mask = [True] * L + [False] * pad
    return {
        "codes": np.array(codes, dtype=np.int64),
        "visits": np.array(visits, dtype=np.int64),
        "ages": np.array(ages, dtype=np.int64),
        "deltas": np.array(deltas, dtype=np.int64),
        "mask": np.array(mask, dtype=np.bool_),
    }


def _mlm_mask(codes: np.ndarray, mask: np.ndarray, vocab: CodeVocab,
              p: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.full_like(codes, -100)
    mask_id = vocab.get(MASK_TOKEN)
    pad_id = 0
    cls_id = vocab.get(CLS_TOKEN)
    out_codes = codes.copy()
    for i in range(codes.shape[0]):
        for j in range(codes.shape[1]):
            if not mask[i, j] or codes[i, j] in (pad_id, cls_id):
                continue
            if random.random() < p:
                labels[i, j] = codes[i, j]
                r = random.random()
                if r < 0.8:
                    out_codes[i, j] = mask_id
                elif r < 0.9:
                    out_codes[i, j] = random.randint(4, len(vocab) - 1)
    return out_codes, labels


def train(db_path: Optional[str] = None, pretrain_epochs: int = 3,
          finetune_epochs: int = 8, batch_size: int = 32,
          lr: float = 3e-4) -> Dict[str, Any]:
    t = _torch()
    if not t:
        return {"error": "torch_unavailable"}
    torch, nn, F = t
    device = torch.device("cpu")

    db_path = db_path or _default_db()
    timelines = build_timelines(db_path)
    if len(timelines) < 40:
        return {"error": "insufficient_data", "n_patients": len(timelines)}

    vocab = CodeVocab(); vocab.fit(timelines, min_count=1)

    enc = [_encode_timeline(t_, vocab) for t_ in timelines]
    codes = np.stack([e["codes"] for e in enc])
    visits = np.stack([e["visits"] for e in enc])
    ages = np.stack([e["ages"] for e in enc])
    deltas = np.stack([e["deltas"] for e in enc])
    masks = np.stack([e["mask"] for e in enc])
    y = {
        k: np.array([tl.labels.get(k, 0) for tl in timelines], dtype=np.float32)
        for k in ("high_cost", "high_util", "readmit_30d", "future_er_90d")
    }

    n = len(timelines)
    idx = np.arange(n); np.random.default_rng(42).shuffle(idx)
    split = max(8, int(n * 0.15))
    val_idx, tr_idx = idx[:split], idx[split:]

    model = _build_model(n_codes=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pos_weight = {
        k: torch.tensor([max(1.0, (y[k] == 0).sum() / max(1, (y[k] == 1).sum()))],
                        dtype=torch.float32, device=device).clamp(1.0, 10.0)
        for k in y
    }

    def _slice(ix):
        return (
            torch.from_numpy(codes[ix]).long().to(device),
            torch.from_numpy(visits[ix]).long().to(device),
            torch.from_numpy(ages[ix]).long().to(device),
            torch.from_numpy(deltas[ix]).long().to(device),
            torch.from_numpy(masks[ix]).bool().to(device),
            {k: torch.from_numpy(y[k][ix]).float().to(device) for k in y},
        )

    history: List[Dict[str, Any]] = []

    logger.info("MLM pretrain on %d patients for %d epochs", len(tr_idx), pretrain_epochs)
    model.train()
    for ep in range(pretrain_epochs):
        rng = np.random.default_rng(ep)
        sh = tr_idx.copy(); rng.shuffle(sh)
        total = 0.0; steps = 0
        for i in range(0, len(sh), batch_size):
            b = sh[i:i + batch_size]
            c_, v_, a_, d_, m_, _ = _slice(b)
            c_np = c_.cpu().numpy(); m_np = m_.cpu().numpy()
            c_masked, mlm_lbls = _mlm_mask(c_np, m_np, vocab, p=0.15)
            c_inp = torch.from_numpy(c_masked).long().to(device)
            mlm_lbl = torch.from_numpy(mlm_lbls).long().to(device)
            _, logits = model(c_inp, v_, a_, d_, m_, mlm=True)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), mlm_lbl.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()); steps += 1
        history.append({"phase": "mlm", "epoch": ep, "loss": total / max(steps, 1)})
        logger.info("  MLM epoch %d loss=%.4f", ep, total / max(steps, 1))

    best_val = 1e9; best_state = None; bad = 0
    logger.info("Multi-task fine-tune for %d epochs", finetune_epochs)
    for ep in range(finetune_epochs):
        model.train()
        rng = np.random.default_rng(ep + 100)
        sh = tr_idx.copy(); rng.shuffle(sh)
        total = 0.0; steps = 0
        for i in range(0, len(sh), batch_size):
            b = sh[i:i + batch_size]
            c_, v_, a_, d_, m_, ybatch = _slice(b)
            head_logits = model(c_, v_, a_, d_, m_, mlm=False)
            loss = 0.0
            for k, lg in head_logits.items():
                loss = loss + F.binary_cross_entropy_with_logits(
                    lg, ybatch[k], pos_weight=pos_weight[k])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()); steps += 1

        model.eval()
        with torch.no_grad():
            c_, v_, a_, d_, m_, ybatch = _slice(val_idx)
            head_logits = model(c_, v_, a_, d_, m_, mlm=False)
            vloss = 0.0; aucs = {}
            for k, lg in head_logits.items():
                vloss = vloss + float(F.binary_cross_entropy_with_logits(
                    lg, ybatch[k], pos_weight=pos_weight[k]).item())
                probs = torch.sigmoid(lg).cpu().numpy()
                true = ybatch[k].cpu().numpy()
                try:
                    from sklearn.metrics import roc_auc_score
                    aucs[k] = float(roc_auc_score(true, probs)) if len(set(true)) > 1 else float("nan")
                except Exception:
                    aucs[k] = float("nan")
        history.append({"phase": "ft", "epoch": ep,
                        "train_loss": total / max(steps, 1),
                        "val_loss": vloss, "val_auc": aucs})
        logger.info("  FT epoch %d tr=%.3f val=%.3f aucs=%s",
                    ep, total / max(steps, 1), vloss, aucs)
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 3:
                logger.info("Early stop at FT epoch %d", ep)
                break

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(_models_dir(), "patient_transformer.pt"))
    with open(os.path.join(_models_dir(), "patient_transformer.meta.json"), "w") as f:
        json.dump({
            "vocab": vocab.id,
            "n_codes": len(vocab),
            "max_len": MAX_LEN,
            "visit_types": VISIT_TYPES,
            "age_buckets": AGE_BUCKETS,
            "trained_at": time.time(),
            "n_patients": n,
            "best_val_loss": best_val,
            "history_tail": history[-5:],
        }, f)
    final_val = history[-1]["val_auc"] if history else {}
    return {
        "trained": True,
        "n_patients": n,
        "vocab_size": len(vocab),
        "best_val_loss": best_val,
        "final_auc": final_val,
        "epochs_run": len(history),
        "architecture": "BEHRT-style (MLM pretrain + multi-task fine-tune)",
    }


_MODEL = None
_VOCAB: Optional[CodeVocab] = None
_META: Optional[Dict[str, Any]] = None


def load() -> bool:
    global _MODEL, _VOCAB, _META
    t = _torch()
    if not t:
        return False
    torch, _, _ = t
    wp = os.path.join(_models_dir(), "patient_transformer.pt")
    mp = os.path.join(_models_dir(), "patient_transformer.meta.json")
    if not (os.path.exists(wp) and os.path.exists(mp)):
        return False
    _META = json.load(open(mp))
    _VOCAB = CodeVocab(); _VOCAB.id = _META["vocab"]
    _MODEL = _build_model(n_codes=len(_VOCAB))
    _MODEL.load_state_dict(torch.load(wp, map_location="cpu"))
    _MODEL.eval()
    return True


def predict(member_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    if _MODEL is None and not load():
        return {"error": "model_not_trained"}
    t = _torch(); torch, _, _ = t
    db_path = db_path or _default_db()

    tls = [tl for tl in build_timelines(db_path) if tl.member_id == member_id]
    if not tls:
        return {"error": "member_not_found_or_no_events"}
    enc = _encode_timeline(tls[0], _VOCAB)
    with torch.no_grad():
        head_logits = _MODEL(
            torch.from_numpy(enc["codes"]).long().unsqueeze(0),
            torch.from_numpy(enc["visits"]).long().unsqueeze(0),
            torch.from_numpy(enc["ages"]).long().unsqueeze(0),
            torch.from_numpy(enc["deltas"]).long().unsqueeze(0),
            torch.from_numpy(enc["mask"]).bool().unsqueeze(0),
            mlm=False,
        )
        probs = {k: float(torch.sigmoid(v)[0].item()) for k, v in head_logits.items()}
    return {
        "member_id": member_id,
        "probabilities": probs,
        "model": "patient_transformer_v1",
        "architecture": "BEHRT-style 4-layer encoder, d=192, h=6",
        "n_events": int(enc["mask"].sum()),
        "vocab_size": len(_VOCAB),
    }


def status() -> Dict[str, Any]:
    loaded = _MODEL is not None or load()
    return {
        "trained": loaded,
        "torch_available": bool(_torch()),
        "meta": _META if loaded else None,
    }
