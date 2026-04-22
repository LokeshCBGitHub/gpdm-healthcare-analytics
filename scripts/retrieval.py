from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _Encoder:
    name = "base"
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class _HashEncoder(_Encoder):
    name = "hash"
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        t = (text or "").lower()
        words = self._TOKEN_RE.findall(t)
        joined = "_".join(words)
        trigrams = [joined[i:i + 3] for i in range(0, max(0, len(joined) - 2))]
        return words + trigrams

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int.from_bytes(
                    hashlib.blake2s(tok.encode("utf-8"), digest_size=4).digest(),
                    "big",
                )
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return out / norms


class _STEncoder(_Encoder):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.dim = int(self._m.get_sentence_embedding_dimension() or 384)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self._m.encode(list(texts), normalize_embeddings=True,
                           show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def _pick_encoder() -> _Encoder:
    if os.environ.get("GPDM_RETRIEVAL_FORCE_HASH", "").lower() in ("1", "true"):
        return _HashEncoder()
    try:
        import sentence_transformers
        return _STEncoder()
    except Exception as e:
        logger.info("[retrieval] sentence-transformers unavailable; "
                    "falling back to hash encoder (%s)", e)
        return _HashEncoder()


@dataclass
class Hit:
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        return d


class Index:
    def __init__(self, encoder: _Encoder):
        self.encoder = encoder
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._vecs: Optional[np.ndarray] = None
        self._built_at: float = 0.0
        self._lock = threading.Lock()

    def size(self) -> int:
        return len(self._texts)

    def stats(self) -> Dict[str, Any]:
        by_src: Dict[str, int] = {}
        for s in self._sources:
            by_src[s] = by_src.get(s, 0) + 1
        return {
            "encoder": self.encoder.name,
            "dim": self.encoder.dim,
            "total": len(self._texts),
            "by_source": by_src,
            "built_at": self._built_at,
        }

    def search(self, query: str, k: int = 5,
               sources: Optional[Iterable[str]] = None,
               min_score: float = 0.05,
               rerank: Optional[bool] = None,
               recall_n: int = 30) -> List[Hit]:
        if not query or self._vecs is None or self._vecs.shape[0] == 0:
            return []
        q = self.encoder.encode([query])[0]
        sims = self._vecs @ q
        if sources:
            src_set = set(sources)
            mask = np.array([s in src_set for s in self._sources], dtype=bool)
            if not mask.any():
                return []
            sims = np.where(mask, sims, -1.0)

        do_rerank = rerank
        if do_rerank is None:
            do_rerank = os.environ.get("GPDM_RERANK_ENABLED", "0").lower() \
                         not in ("0", "false", "no", "off")

        if do_rerank:
            n_recall = min(max(recall_n, k), len(sims))
            cand_idx = np.argpartition(-sims, n_recall - 1)[:n_recall]
            cand_idx = cand_idx[np.argsort(-sims[cand_idx])]
            cand_texts = [self._texts[i] for i in cand_idx]
            cand_sims  = [float(sims[i]) for i in cand_idx]
            try:
                from reranker import get_reranker
                rr = get_reranker().rerank(query, cand_texts,
                                           base_scores=cand_sims, k=k)
                final_idx = [int(cand_idx[i]) for i in rr.indices]
                final_scores = list(rr.scores)
                hits: List[Hit] = []
                for i, s in zip(final_idx, final_scores):
                    if s < min_score:
                        continue
                    md = dict(self._metas[i])
                    md["reranked"] = True
                    md["rerank_backend"] = rr.backend
                    hits.append(Hit(
                        text=self._texts[i],
                        source=self._sources[i],
                        score=float(s),
                        metadata=md,
                    ))
                return hits
            except Exception as e:
                logger.debug("[retrieval] rerank failed, using bi-encoder "
                             "results: %s", e)

        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        hits: List[Hit] = []
        for i in idx:
            s = float(sims[i])
            if s < min_score:
                continue
            hits.append(Hit(
                text=self._texts[i],
                source=self._sources[i],
                score=s,
                metadata=dict(self._metas[i]),
            ))
        return hits

    def rebuild(self, db_path: Optional[str]) -> None:
        with self._lock:
            texts, sources, metas = [], [], []
            for t, s, m in _iter_sources(db_path):
                if not t or not str(t).strip():
                    continue
                texts.append(str(t).strip())
                sources.append(s)
                metas.append(m)
            self._texts, self._sources, self._metas = texts, sources, metas
            if texts:
                self._vecs = self.encoder.encode(texts)
            else:
                self._vecs = np.zeros((0, self.encoder.dim), dtype=np.float32)
            self._built_at = time.time()
            logger.info("[retrieval] index built: %s", self.stats())


def _iter_query_log(db_path: str, limit: int = 5000
                    ) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    if not db_path:
        return
    data_dir = os.path.dirname(db_path) if os.path.isfile(db_path) else db_path
    candidate = os.path.join(data_dir, "query_tracker.db")
    if not os.path.exists(candidate):
        candidate = db_path
    try:
        conn = sqlite3.connect(candidate)
        try:
            cur = conn.execute(
                "SELECT question, intent, COUNT(*) as n, MAX(timestamp) as t "
                "FROM query_log WHERE success = 1 GROUP BY question "
                "ORDER BY n DESC LIMIT ?", (limit,)
            )
            for row in cur.fetchall():
                q, intent, n, t = row
                yield q, "query_log", {
                    "intent": intent, "count": int(n or 0),
                    "last_ts": float(t or 0),
                }
        finally:
            conn.close()
    except Exception as e:
        logger.debug("[retrieval] query_log unavailable: %s", e)


def _iter_benchmarks() -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    try:
        from cms_reference import load_benchmarks
        cat = load_benchmarks() or {}
    except Exception as e:
        logger.debug("[retrieval] cms_reference unavailable: %s", e)
        return
    for key, entry in cat.items():
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("name") or key
        desc  = entry.get("description") or entry.get("source") or ""
        good  = entry.get("good"); warn = entry.get("warn"); bad = entry.get("bad")
        text = f"{label}. Benchmark with good={good}, warn={warn}, bad={bad}. {desc}"
        yield text, "benchmark", {
            "key": key, "label": label,
            "good": good, "warn": warn, "bad": bad,
            "source": entry.get("source", ""),
        }


def _iter_metrics() -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    try:
        from healthcare_forecasting import FORECAST_METRICS
    except Exception as e:
        logger.debug("[retrieval] healthcare_forecasting unavailable: %s", e)
        return
    for key, meta in (FORECAST_METRICS or {}).items():
        label = meta.get("label") or key
        ctx = meta.get("business_context") or ""
        yield f"{label}. {ctx}", "metric", {
            "key": key, "label": label, "unit": meta.get("unit", ""),
        }


def _iter_kpi_registry() -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    try:
        import executive_kpis as ek
    except Exception:
        return
    reg = getattr(ek, "KPI_REGISTRY", None) or getattr(ek, "KPIS", None)
    if not reg:
        return
    items = reg.items() if isinstance(reg, dict) else enumerate(reg)
    for k, entry in items:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("name") or str(k)
        so    = entry.get("so_what") or entry.get("description") or ""
        yield f"{label}. {so}", "kpi", {"key": str(k), "label": label}


def _iter_sources(db_path: Optional[str]
                  ) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    if db_path:
        yield from _iter_query_log(db_path)
    yield from _iter_benchmarks()
    yield from _iter_metrics()
    yield from _iter_kpi_registry()


_INDEX: Optional[Index] = None
_INDEX_LOCK = threading.Lock()
_INDEX_DB_PATH: Optional[str] = None


def get_index(db_path: Optional[str] = None,
              force_rebuild: bool = False) -> Index:
    global _INDEX, _INDEX_DB_PATH
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = Index(_pick_encoder())
            _INDEX_DB_PATH = db_path
            _INDEX.rebuild(db_path)
        elif force_rebuild or (db_path and db_path != _INDEX_DB_PATH):
            _INDEX_DB_PATH = db_path or _INDEX_DB_PATH
            _INDEX.rebuild(_INDEX_DB_PATH)
    return _INDEX


def reset_index() -> None:
    global _INDEX, _INDEX_DB_PATH
    with _INDEX_LOCK:
        _INDEX = None
        _INDEX_DB_PATH = None


__all__ = ["Hit", "Index", "get_index", "reset_index"]
