from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "models", "embeddings"))
os.environ.setdefault("HF_HOME",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "models", "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_TORCH = None
_SBERT = None
_FAISS = None


def _try_import_torch():
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


def _try_import_sbert():
    global _SBERT
    if _SBERT is not None:
        return _SBERT
    try:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer
    except Exception as e:
        logger.warning("sentence-transformers unavailable: %s", e)
        _SBERT = False
    return _SBERT


def _try_import_faiss():
    global _FAISS
    if _FAISS is not None:
        return _FAISS
    try:
        import faiss
        _FAISS = faiss
    except Exception as e:
        logger.info("faiss unavailable, using numpy fallback: %s", e)
        _FAISS = False
    return _FAISS


def _repo_root() -> str:
    return (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _data_dir() -> str:
    return os.path.join(_repo_root(), "data")


def _models_dir() -> str:
    d = os.path.join(_data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def _default_db() -> str:
    return os.path.join(_data_dir(), "healthcare_demo.db")


class ClinicalEmbedder:

    DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "models", "embeddings", "all-MiniLM-L6-v2",
    )
    DIM = 384

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.backend = "none"
        self._model = None
        self._torch = None
        self._warmed = False
        self._hash_dim = self.DIM

        sbert_cls = _try_import_sbert()
        torch_mod = _try_import_torch()

        local_ok = os.path.isdir(model_name) if isinstance(model_name, str) else False
        if sbert_cls and torch_mod and local_ok:
            try:
                self._model = sbert_cls(model_name, local_files_only=True)
                self._torch = torch_mod[0]
                self.backend = "sbert"
                self.DIM = self._model.get_sentence_embedding_dimension()
                logger.info("ClinicalEmbedder: local SBERT '%s' dim=%d", model_name, self.DIM)
            except TypeError:
                try:
                    self._model = sbert_cls(model_name)
                    self._torch = torch_mod[0]
                    self.backend = "sbert"
                    self.DIM = self._model.get_sentence_embedding_dimension()
                    logger.info("ClinicalEmbedder: local SBERT '%s' dim=%d", model_name, self.DIM)
                except Exception as e:
                    logger.warning("SBERT load failed (%s); using hash embeddings.", e)
                    self._model = None
            except Exception as e:
                logger.warning("SBERT load failed (%s); using hash embeddings.", e)
                self._model = None
        elif sbert_cls and torch_mod and not local_ok:
            logger.warning(
                "SBERT skipped: no local model at %s. Air-gap policy forbids "
                "downloading. Using hash embedder. To enable SBERT, stage the "
                "model files on-prem at that path.", model_name,
            )

        if self._model is None:
            self.backend = "hash"
            logger.info("ClinicalEmbedder: hashed n-gram fallback (dim=%d)", self._hash_dim)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        if self.backend == "sbert":
            vecs = self._model.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=normalize,
            )
            return vecs.astype(np.float32)
        out = np.zeros((len(texts), self._hash_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f"  {(t or '').lower()}  "
            for j in range(len(s) - 2):
                ngram = s[j:j+3]
                h = (hash(ngram) & 0xFFFFFFFF)
                idx = h % self._hash_dim
                sign = 1.0 if (h >> 16) & 1 else -1.0
                out[i, idx] += sign
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            out = out / norms
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def domain_warmup(self, pairs: List[Tuple[str, str]], epochs: int = 1,
                      lr: float = 2e-5, batch_size: int = 16) -> Dict[str, Any]:
        if self.backend != "sbert" or not self._torch:
            return {"warmed": False, "reason": "sbert_unavailable"}
        if not pairs:
            return {"warmed": False, "reason": "no_pairs"}

        torch = self._torch
        nn = __import__("torch.nn", fromlist=["_"])
        st_model = self._model
        st_model.train()
        opt = torch.optim.AdamW(st_model.parameters(), lr=lr)

        losses = []
        steps = 0
        for ep in range(epochs):
            random.shuffle(pairs)
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                if len(batch) < 2:
                    continue
                anchors = [a for a, _ in batch]
                positives = [b for _, b in batch]

                a_tok = st_model.tokenize(anchors)
                p_tok = st_model.tokenize(positives)
                device = next(st_model.parameters()).device
                a_tok = {k: v.to(device) for k, v in a_tok.items()}
                p_tok = {k: v.to(device) for k, v in p_tok.items()}

                a_out = st_model(a_tok)["sentence_embedding"]
                p_out = st_model(p_tok)["sentence_embedding"]
                a_n = torch.nn.functional.normalize(a_out, dim=1)
                p_n = torch.nn.functional.normalize(p_out, dim=1)

                scores = a_n @ p_n.T * 20.0
                labels = torch.arange(len(batch), device=device)
                loss = torch.nn.functional.cross_entropy(scores, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                steps += 1

        st_model.eval()
        self._warmed = True
        return {
            "warmed": True,
            "steps": steps,
            "final_loss": losses[-1] if losses else float("nan"),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        }


@dataclass
class CodeEntry:
    code: str
    description: str
    synonyms: List[str] = field(default_factory=list)
    kind: str = "ICD10"


class ClinicalVocabulary:

    def __init__(self, embedder: ClinicalEmbedder):
        self.embedder = embedder
        self.entries: List[CodeEntry] = []
        self.matrix: Optional[np.ndarray] = None
        self._faiss_index = None

    def load_from_cms_module(self) -> int:
        try:
            import importlib
            cms = importlib.import_module("cms_expand_data")
        except Exception as e:
            logger.warning("cms_expand_data import failed: %s", e)
            return 0

        added = 0
        for attr, kind in (
            ("ICD10_MASTER", "ICD10"),
            ("CPT_MASTER", "CPT"),
            ("DRG_MASTER", "DRG"),
            ("MEDICATION_MASTER", "DRUG"),
        ):
            m = getattr(cms, attr, None)
            if not isinstance(m, dict):
                continue
            for code, info in m.items():
                if isinstance(info, dict):
                    desc = info.get("description") or info.get("name") or str(code)
                    syns = info.get("synonyms") or info.get("aliases") or []
                else:
                    desc = str(info); syns = []
                if isinstance(syns, str):
                    syns = [syns]
                self.entries.append(
                    CodeEntry(code=str(code), description=desc, synonyms=list(syns), kind=kind)
                )
                added += 1

        if not added:
            try:
                with sqlite3.connect(_default_db()) as c:
                    for row in c.execute(
                        "SELECT DISTINCT ICD10_CODE, ICD10_DESCRIPTION FROM diagnoses "
                        "WHERE ICD10_DESCRIPTION IS NOT NULL"):
                        self.entries.append(CodeEntry(
                            code=row[0], description=row[1] or row[0], kind="ICD10"))
                        added += 1
            except Exception:
                pass

        logger.info("ClinicalVocabulary: loaded %d entries", added)
        return added

    def build_index(self) -> None:
        if not self.entries:
            return
        docs = [
            f"{e.code} {e.description} " + " ".join(e.synonyms)
            for e in self.entries
        ]
        self.matrix = self.embedder.encode(docs, normalize=True)
        faiss = _try_import_faiss()
        if faiss:
            self._faiss_index = faiss.IndexFlatIP(self.matrix.shape[1])
            self._faiss_index.add(self.matrix.astype(np.float32))
        logger.info("ClinicalVocabulary: indexed %d entries (faiss=%s)",
                    len(self.entries), bool(faiss))

    def resolve(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not text or self.matrix is None or not self.entries:
            return []
        q = self.embedder.encode([text], normalize=True).astype(np.float32)
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, min(k, len(self.entries)))
            scores, idx = scores[0], idx[0]
        else:
            sims = (self.matrix @ q.T).ravel()
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        return [{
            "code": self.entries[int(i)].code,
            "description": self.entries[int(i)].description,
            "kind": self.entries[int(i)].kind,
            "score": float(s),
        } for i, s in zip(idx, scores) if int(i) >= 0]

    def harvest_pairs(self) -> List[Tuple[str, str]]:
        out = []
        for e in self.entries:
            for s in e.synonyms:
                if s and e.description:
                    out.append((e.description, s))
                    out.append((s, e.description))
        return out


class SemanticSchemaIndex:
    def __init__(self, embedder: ClinicalEmbedder):
        self.embedder = embedder
        self.docs: List[Dict[str, str]] = []
        self.matrix: Optional[np.ndarray] = None
        self._faiss = None

    def build(self, db_path: str) -> int:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
            for t in tables:
                cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})")]
                self.docs.append({
                    "table": t, "column": "*", "kind": "table",
                    "text": f"{t} table with columns: {', '.join(cols)}",
                })
                for col in cols:
                    self.docs.append({
                        "table": t, "column": col, "kind": "column",
                        "text": f"column {col} in table {t}",
                    })
        if not self.docs:
            return 0
        self.matrix = self.embedder.encode([d["text"] for d in self.docs])
        faiss = _try_import_faiss()
        if faiss:
            self._faiss = faiss.IndexFlatIP(self.matrix.shape[1])
            self._faiss.add(self.matrix.astype(np.float32))
        logger.info("SemanticSchemaIndex: %d docs indexed", len(self.docs))
        return len(self.docs)

    def lookup(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.matrix is None:
            return []
        vec = self.embedder.encode([q]).astype(np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(vec, k)
            scores, idx = scores[0], idx[0]
        else:
            sims = (self.matrix @ vec.T).ravel()
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        return [{**self.docs[int(i)], "score": float(s)}
                for i, s in zip(idx, scores) if int(i) >= 0]


class NeuralIntentClassifier:
    MODEL_FILE = "intent_mlp.pt"
    META_FILE = "intent_mlp.meta.json"

    def __init__(self, embedder: ClinicalEmbedder):
        self.embedder = embedder
        self.labels: List[str] = []
        self._model = None
        self._torch_mod = None

    def _build_net(self, input_dim: int, n_classes: int):
        t = _try_import_torch()
        if not t:
            return None
        torch, nn, F = t
        self._torch_mod = (torch, nn, F)

        class MLP(nn.Module):
            def __init__(self, d_in, d_hidden, n_out):
                super().__init__()
                self.fc1 = nn.Linear(d_in, d_hidden)
                self.bn1 = nn.BatchNorm1d(d_hidden)
                self.fc2 = nn.Linear(d_hidden, d_hidden // 2)
                self.bn2 = nn.BatchNorm1d(d_hidden // 2)
                self.drop = nn.Dropout(0.3)
                self.out = nn.Linear(d_hidden // 2, n_out)

            def forward(self, x):
                x = F.gelu(self.bn1(self.fc1(x)))
                x = self.drop(x)
                x = F.gelu(self.bn2(self.fc2(x)))
                return self.out(x)

        return MLP(input_dim, 256, n_classes)

    def train(self, texts: List[str], labels: List[str],
              epochs: int = 30, lr: float = 1e-3, patience: int = 5) -> Dict[str, Any]:
        t = _try_import_torch()
        if not t:
            return {"error": "torch_unavailable"}
        torch, nn, F = t

        uniq = sorted(set(labels))
        self.labels = uniq
        y_idx = np.array([uniq.index(l) for l in labels], dtype=np.int64)
        X = self.embedder.encode(texts, normalize=True)

        rng = np.random.default_rng(42)
        order = rng.permutation(len(texts))
        split = max(1, int(len(texts) * 0.15))
        val_ix, tr_ix = order[:split], order[split:]

        net = self._build_net(X.shape[1], len(uniq))
        if net is None:
            return {"error": "torch_unavailable"}
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(y_idx).long()

        best_val, best_state, bad = 1e9, None, 0
        history = []
        for ep in range(epochs):
            net.train()
            logits = net(Xt[tr_ix])
            loss = F.cross_entropy(logits, yt[tr_ix], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()

            net.eval()
            with torch.no_grad():
                vloss = F.cross_entropy(net(Xt[val_ix]), yt[val_ix]).item()
                vacc = (net(Xt[val_ix]).argmax(1) == yt[val_ix]).float().mean().item()
            history.append({"epoch": ep, "train_loss": float(loss.item()),
                            "val_loss": vloss, "val_acc": vacc})
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        self._save()
        return {
            "trained": True,
            "n_samples": len(texts),
            "n_classes": len(uniq),
            "val_acc": history[-1]["val_acc"] if history else None,
            "best_val_loss": best_val,
            "epochs_run": len(history),
            "embedding_backend": self.embedder.backend,
        }

    def _save(self) -> None:
        if self._model is None:
            return
        torch = self._torch_mod[0]
        torch.save(self._model.state_dict(), os.path.join(_models_dir(), self.MODEL_FILE))
        with open(os.path.join(_models_dir(), self.META_FILE), "w") as f:
            json.dump({"labels": self.labels,
                       "input_dim": self.embedder.DIM}, f)

    def load(self) -> bool:
        meta_p = os.path.join(_models_dir(), self.META_FILE)
        w_p = os.path.join(_models_dir(), self.MODEL_FILE)
        if not (os.path.exists(meta_p) and os.path.exists(w_p)):
            return False
        t = _try_import_torch()
        if not t:
            return False
        torch, _, _ = t
        meta = json.load(open(meta_p))
        self.labels = meta["labels"]
        self._model = self._build_net(meta["input_dim"], len(self.labels))
        self._model.load_state_dict(torch.load(w_p, map_location="cpu"))
        self._model.eval()
        return True

    def predict(self, text: str) -> Dict[str, Any]:
        if self._model is None and not self.load():
            return {"error": "model_not_trained"}
        torch = self._torch_mod[0]
        F = self._torch_mod[2]
        x = torch.from_numpy(self.embedder.encode([text])).float()
        with torch.no_grad():
            probs = F.softmax(self._model(x), dim=1)[0].cpu().numpy()
        i = int(np.argmax(probs))
        return {
            "intent": self.labels[i],
            "confidence": float(probs[i]),
            "neural": True,
            "all_scores": {lbl: float(p) for lbl, p in zip(self.labels, probs)},
        }


class ClinicalRiskNet:
    MODEL_FILE = "risk_net.pt"
    META_FILE = "risk_net.meta.json"

    PAD = 0
    UNK = 1

    def __init__(self):
        self.icd_vocab: Dict[str, int] = {}
        self._model = None
        self._torch_mod = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.max_dx = 32

    def _build_net(self, n_icd: int, n_demo: int):
        t = _try_import_torch()
        if not t:
            return None
        torch, nn, F = t
        self._torch_mod = (torch, nn, F)

        class Net(nn.Module):
            def __init__(self, n_icd, n_demo, emb=64, hidden=128):
                super().__init__()
                self.icd_emb = nn.Embedding(n_icd, emb, padding_idx=0)
                self.attn_q = nn.Linear(emb, emb)
                self.attn_k = nn.Linear(emb, emb)
                self.attn_v = nn.Linear(emb, emb)
                self.demo_proj = nn.Linear(n_demo, emb)
                self.head = nn.Sequential(
                    nn.Linear(emb * 2, hidden),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(0.3),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden // 2, 3),
                )

            def forward(self, icd_ids, mask, demo):
                e = self.icd_emb(icd_ids)
                q = self.attn_q(e); k = self.attn_k(e); v = self.attn_v(e)
                scores = (q @ k.transpose(1, 2)) / math.sqrt(e.size(-1))
                scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
                w = torch.softmax(scores, dim=-1)
                ctx = w @ v
                m = mask.unsqueeze(-1).float()
                pooled = (ctx * m).sum(1) / (m.sum(1).clamp(min=1.0))
                d = self.demo_proj(demo)
                z = torch.cat([pooled, d], dim=1)
                return self.head(z)

        return Net(n_icd, n_demo)

    def _build_dataset(self, db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row

            members: Dict[str, Dict[str, Any]] = {}
            for r in c.execute("""
              SELECT m.MEMBER_ID, m.DATE_OF_BIRTH, m.GENDER,
                     COALESCE(m.CHRONIC_CONDITIONS,0) AS chr
              FROM members m"""):
                members[r["MEMBER_ID"]] = {
                    "dob": r["DATE_OF_BIRTH"] or "",
                    "gender": (r["GENDER"] or "").upper(),
                    "chronic": float(r["chr"]),
                    "dx": [],
                }
            for r in c.execute(
                "SELECT MEMBER_ID, ICD10_CODE FROM diagnoses WHERE ICD10_CODE IS NOT NULL"):
                if r["MEMBER_ID"] in members:
                    members[r["MEMBER_ID"]]["dx"].append(r["ICD10_CODE"])
            for r in c.execute("""
              SELECT MEMBER_ID,
                SUM(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 ELSE 0 END) er,
                SUM(CASE WHEN VISIT_TYPE='INPATIENT' THEN 1 ELSE 0 END) ip,
                COALESCE(SUM(LENGTH_OF_STAY),0) ipd
              FROM encounters GROUP BY MEMBER_ID"""):
                if r["MEMBER_ID"] in members:
                    m = members[r["MEMBER_ID"]]
                    m["er"] = float(r["er"] or 0)
                    m["ip"] = float(r["ip"] or 0)
                    m["ipd"] = float(r["ipd"] or 0)
            for r in c.execute("""
              SELECT MEMBER_ID, COALESCE(SUM(PAID_AMOUNT),0) cost,
                     COUNT(*) cnt,
                     SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) denied
              FROM claims GROUP BY MEMBER_ID"""):
                if r["MEMBER_ID"] in members:
                    m = members[r["MEMBER_ID"]]
                    m["cost"] = float(r["cost"])
                    m["claims"] = float(r["cnt"])
                    m["denied"] = float(r["denied"])
            by_m_ip: Dict[str, List[Tuple[str, str]]] = {}
            for r in c.execute(
                "SELECT MEMBER_ID, ADMIT_DATE, DISCHARGE_DATE FROM encounters "
                "WHERE VISIT_TYPE='INPATIENT' AND DISCHARGE_DATE IS NOT NULL "
                "ORDER BY MEMBER_ID, DISCHARGE_DATE"):
                by_m_ip.setdefault(r["MEMBER_ID"], []).append(
                    (r["ADMIT_DATE"] or "", r["DISCHARGE_DATE"] or ""))

        def _readmit(mid: str) -> int:
            rows = by_m_ip.get(mid, [])
            from datetime import datetime
            for i in range(len(rows) - 1):
                try:
                    d = (datetime.fromisoformat(rows[i+1][0][:10]) -
                         datetime.fromisoformat(rows[i][1][:10])).days
                    if 0 <= d <= 30:
                        return 1
                except Exception:
                    continue
            return 0

        all_codes = set()
        for m in members.values():
            all_codes.update(m["dx"])
        self.icd_vocab = {"<pad>": 0, "<unk>": 1}
        for i, c_ in enumerate(sorted(all_codes)):
            self.icd_vocab[c_] = i + 2

        from datetime import datetime
        def _age(dob: str) -> float:
            try:
                return max(0, datetime.now().year - int(dob[:4]))
            except Exception:
                return 40.0

        ids_rows, mask_rows, demo_rows, y_rows = [], [], [], []
        all_costs = [m.get("cost", 0.0) for m in members.values()]
        all_ipd = [m.get("ipd", 0.0) for m in members.values()]
        cost_thr = float(np.quantile(all_costs, 0.80)) if all_costs else 0.0
        ipd_thr = float(np.quantile(all_ipd, 0.80)) if all_ipd else 0.0

        for mid, m in members.items():
            dx = m["dx"][: self.max_dx]
            ids = [self.icd_vocab.get(c_, self.UNK) for c_ in dx]
            pad = self.max_dx - len(ids)
            mask = [1] * len(ids) + [0] * pad
            ids += [self.PAD] * pad
            ids_rows.append(ids)
            mask_rows.append(mask)

            demo = [
                _age(m["dob"]),
                1.0 if m["gender"].startswith("F") else 0.0,
                m.get("chronic", 0.0),
                m.get("er", 0.0),
                m.get("ip", 0.0),
                m.get("ipd", 0.0),
                m.get("claims", 0.0),
                m.get("denied", 0.0),
                (m.get("denied", 0.0) / m.get("claims", 1.0)) if m.get("claims") else 0.0,
                math.log1p(m.get("cost", 0.0)),
            ]
            demo_rows.append(demo)
            y_rows.append([
                1 if m.get("cost", 0.0) >= cost_thr else 0,
                _readmit(mid),
                1 if m.get("ipd", 0.0) >= ipd_thr else 0,
            ])

        X_ids = np.array(ids_rows, dtype=np.int64)
        X_mask = np.array(mask_rows, dtype=np.bool_)
        X_demo = np.array(demo_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)

        self.feature_means = X_demo.mean(0)
        self.feature_stds = X_demo.std(0) + 1e-6
        X_demo = (X_demo - self.feature_means) / self.feature_stds
        return X_ids, X_mask, X_demo, y

    def train(self, db_path: str, epochs: int = 25, lr: float = 1e-3,
              batch_size: int = 64) -> Dict[str, Any]:
        t = _try_import_torch()
        if not t:
            return {"error": "torch_unavailable"}
        torch, nn, F = t

        X_ids, X_mask, X_demo, y = self._build_dataset(db_path)
        n = X_ids.shape[0]
        if n < 40:
            return {"error": "insufficient_data", "n_samples": n}

        net = self._build_net(len(self.icd_vocab), X_demo.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = torch.tensor(
            [(y[:, j] == 0).sum() / max(1, (y[:, j] == 1).sum()) for j in range(3)],
            dtype=torch.float32,
        ).clamp(1.0, 8.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        idx = np.arange(n); np.random.default_rng(42).shuffle(idx)
        split = max(4, int(n * 0.15))
        val_idx, tr_idx = idx[:split], idx[split:]

        def _tensor_slice(ix):
            return (
                torch.from_numpy(X_ids[ix]).long(),
                torch.from_numpy(X_mask[ix]).bool(),
                torch.from_numpy(X_demo[ix]).float(),
                torch.from_numpy(y[ix]).float(),
            )

        best_val, best_state, bad = 1e9, None, 0
        history = []
        for ep in range(epochs):
            net.train()
            rng = np.random.default_rng(ep)
            tr_shuf = tr_idx.copy(); rng.shuffle(tr_shuf)
            tot = 0.0; steps = 0
            for i in range(0, len(tr_shuf), batch_size):
                b = tr_shuf[i:i + batch_size]
                if len(b) < 2:
                    continue
                iid, im, de, yb = _tensor_slice(b)
                logits = net(iid, im, de)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item()); steps += 1

            net.eval()
            with torch.no_grad():
                iid, im, de, yb = _tensor_slice(val_idx)
                vloss = float(loss_fn(net(iid, im, de), yb).item())
                probs = torch.sigmoid(net(iid, im, de)).cpu().numpy()
                aucs = []
                for j in range(3):
                    try:
                        from sklearn.metrics import roc_auc_score
                        aucs.append(float(roc_auc_score(y[val_idx, j], probs[:, j])))
                    except Exception:
                        aucs.append(float("nan"))
            history.append({"epoch": ep, "train_loss": tot / max(steps, 1),
                            "val_loss": vloss, "val_auc": aucs})
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 5:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        self._save()
        return {
            "trained": True,
            "n_samples": n,
            "n_icd_vocab": len(self.icd_vocab),
            "best_val_loss": best_val,
            "final_val_auc": history[-1]["val_auc"] if history else None,
            "epochs_run": len(history),
        }

    def _save(self) -> None:
        torch = self._torch_mod[0]
        torch.save(self._model.state_dict(), os.path.join(_models_dir(), self.MODEL_FILE))
        with open(os.path.join(_models_dir(), self.META_FILE), "w") as f:
            json.dump({
                "icd_vocab": self.icd_vocab,
                "max_dx": self.max_dx,
                "feature_means": self.feature_means.tolist(),
                "feature_stds": self.feature_stds.tolist(),
            }, f)

    def load(self) -> bool:
        mp = os.path.join(_models_dir(), self.META_FILE)
        wp = os.path.join(_models_dir(), self.MODEL_FILE)
        if not (os.path.exists(mp) and os.path.exists(wp)):
            return False
        t = _try_import_torch()
        if not t:
            return False
        torch, _, _ = t
        meta = json.load(open(mp))
        self.icd_vocab = meta["icd_vocab"]
        self.max_dx = int(meta["max_dx"])
        self.feature_means = np.array(meta["feature_means"], dtype=np.float32)
        self.feature_stds = np.array(meta["feature_stds"], dtype=np.float32)
        n_demo = len(self.feature_means)
        self._model = self._build_net(len(self.icd_vocab), n_demo)
        self._model.load_state_dict(torch.load(wp, map_location="cpu"))
        self._model.eval()
        return True

    def predict(self, db_path: str, member_id: str) -> Dict[str, Any]:
        if self._model is None and not self.load():
            return {"error": "model_not_trained"}
        torch = self._torch_mod[0]

        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            m = c.execute(
                "SELECT DATE_OF_BIRTH, GENDER, COALESCE(CHRONIC_CONDITIONS,0) chr "
                "FROM members WHERE MEMBER_ID=?", (member_id,)).fetchone()
            if m is None:
                return {"error": "member_not_found"}
            dx = [r[0] for r in c.execute(
                "SELECT ICD10_CODE FROM diagnoses WHERE MEMBER_ID=?", (member_id,))]
            util = c.execute("""
                SELECT SUM(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 ELSE 0 END) er,
                       SUM(CASE WHEN VISIT_TYPE='INPATIENT' THEN 1 ELSE 0 END) ip,
                       COALESCE(SUM(LENGTH_OF_STAY),0) ipd
                FROM encounters WHERE MEMBER_ID=?""", (member_id,)).fetchone()
            cost_row = c.execute("""
                SELECT COALESCE(SUM(PAID_AMOUNT),0) cost, COUNT(*) cnt,
                       SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) denied
                FROM claims WHERE MEMBER_ID=?""", (member_id,)).fetchone()

        ids = [self.icd_vocab.get(c_, self.UNK) for c_ in dx][: self.max_dx]
        mask = [1] * len(ids) + [0] * (self.max_dx - len(ids))
        ids += [self.PAD] * (self.max_dx - len(ids))

        from datetime import datetime
        try:
            age = max(0, datetime.now().year - int((m["DATE_OF_BIRTH"] or "")[:4]))
        except Exception:
            age = 40
        claims = float(cost_row["cnt"] or 0)
        denied = float(cost_row["denied"] or 0)
        demo = np.array([
            age,
            1.0 if (m["GENDER"] or "").upper().startswith("F") else 0.0,
            float(m["chr"]),
            float(util["er"] or 0),
            float(util["ip"] or 0),
            float(util["ipd"] or 0),
            claims,
            denied,
            (denied / claims) if claims else 0.0,
            math.log1p(float(cost_row["cost"] or 0.0)),
        ], dtype=np.float32)
        demo = (demo - self.feature_means) / self.feature_stds

        iid = torch.from_numpy(np.array([ids], dtype=np.int64))
        im = torch.from_numpy(np.array([mask], dtype=np.bool_))
        de = torch.from_numpy(demo.reshape(1, -1)).float()
        with torch.no_grad():
            logits = self._model(iid, im, de)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return {
            "member_id": member_id,
            "high_cost_probability": float(probs[0]),
            "readmit_30d_probability": float(probs[1]),
            "high_utilization_probability": float(probs[2]),
            "model": "clinical_risk_net_v1",
            "architecture": "ICD embedding + self-attention pool + multi-task MLP",
            "n_dx": int(sum(mask)),
        }


_EMBEDDER: Optional[ClinicalEmbedder] = None
_VOCAB: Optional[ClinicalVocabulary] = None
_SCHEMA: Optional[SemanticSchemaIndex] = None
_INTENT: Optional[NeuralIntentClassifier] = None
_RISK: Optional[ClinicalRiskNet] = None


def init_neural_stack(db_path: Optional[str] = None, warmup: bool = True) -> Dict[str, Any]:
    global _EMBEDDER, _VOCAB, _SCHEMA, _INTENT, _RISK
    db_path = db_path or _default_db()
    _EMBEDDER = ClinicalEmbedder()

    _VOCAB = ClinicalVocabulary(_EMBEDDER)
    n_codes = _VOCAB.load_from_cms_module()
    warm_info = {}
    if warmup and n_codes and _EMBEDDER.backend == "sbert":
        pairs = _VOCAB.harvest_pairs()
        if pairs:
            warm_info = _EMBEDDER.domain_warmup(pairs, epochs=1)
    _VOCAB.build_index()

    _SCHEMA = SemanticSchemaIndex(_EMBEDDER)
    n_schema = _SCHEMA.build(db_path)

    _INTENT = NeuralIntentClassifier(_EMBEDDER)
    _INTENT.load()

    _RISK = ClinicalRiskNet()
    _RISK.load()

    return {
        "embedder_backend": _EMBEDDER.backend,
        "embedder_dim": _EMBEDDER.DIM,
        "vocabulary_entries": n_codes,
        "schema_docs": n_schema,
        "intent_loaded": _INTENT._model is not None,
        "risk_loaded": _RISK._model is not None,
        "warmup": warm_info,
        "faiss": bool(_try_import_faiss()),
        "torch": bool(_try_import_torch()),
    }


def get_neural_status(db_path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "stack_initialised": _EMBEDDER is not None,
        "embedder_backend": getattr(_EMBEDDER, "backend", None),
        "vocabulary_entries": len(_VOCAB.entries) if _VOCAB else 0,
        "schema_docs": len(_SCHEMA.docs) if _SCHEMA else 0,
        "intent_loaded": (_INTENT is not None and _INTENT._model is not None),
        "intent_labels": _INTENT.labels if _INTENT else [],
        "risk_loaded": (_RISK is not None and _RISK._model is not None),
        "torch_available": bool(_try_import_torch()),
        "sbert_available": bool(_try_import_sbert()),
        "faiss_available": bool(_try_import_faiss()),
    }


def embed_text(text: str) -> List[float]:
    if _EMBEDDER is None:
        init_neural_stack()
    return _EMBEDDER.encode_one(text).tolist()


def resolve_clinical_code(text: str, k: int = 5) -> List[Dict[str, Any]]:
    if _VOCAB is None:
        init_neural_stack()
    return _VOCAB.resolve(text, k)


def semantic_schema_lookup(q: str, k: int = 5) -> List[Dict[str, Any]]:
    if _SCHEMA is None:
        init_neural_stack()
    return _SCHEMA.lookup(q, k)


def classify_intent_neural(text: str) -> Dict[str, Any]:
    if _INTENT is None:
        init_neural_stack()
    return _INTENT.predict(text)


def train_intent_mlp(db_path: Optional[str] = None, epochs: int = 30) -> Dict[str, Any]:
    db_path = db_path or _default_db()
    if _INTENT is None:
        init_neural_stack(db_path)
    try:
        from semantic_layer import IntentClassifier
        base = IntentClassifier.TRAINING_DATA
    except Exception as e:
        return {"error": f"no_training_data: {e}"}
    try:
        from learning_loop import load_feedback_examples
        learned = load_feedback_examples(db_path)
    except Exception:
        learned = {}
    texts: List[str] = []; labels: List[str] = []
    for intent, qs in base.items():
        for q in qs:
            texts.append(q); labels.append(intent)
    for intent, qs in learned.items():
        for q in qs:
            texts.append(q); labels.append(intent)
    return _INTENT.train(texts, labels, epochs=epochs)


def train_risk_net(db_path: Optional[str] = None, epochs: int = 25) -> Dict[str, Any]:
    db_path = db_path or _default_db()
    if _RISK is None:
        init_neural_stack(db_path)
    return _RISK.train(db_path, epochs=epochs)


def predict_risk_neural(db_path: Optional[str], member_id: str) -> Dict[str, Any]:
    db_path = db_path or _default_db()
    if _RISK is None:
        init_neural_stack(db_path)
    return _RISK.predict(db_path, member_id)
