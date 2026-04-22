from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _RerankBackend:
    name = "base"

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class _CrossEncoderBackend(_RerankBackend):
    name = "cross_encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self._m = CrossEncoder(model_name)
        self.model_name = model_name

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, d) for d in docs]
        raw = self._m.predict(pairs)
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        return 1.0 / (1.0 + np.exp(-arr))


class _LexicalBackend(_RerankBackend):
    name = "lexical"

    @staticmethod
    def _words(s: str) -> set:
        return set(_TOKEN_RE.findall((s or "").lower()))

    @staticmethod
    def _trigrams(s: str) -> set:
        s2 = "_".join(_TOKEN_RE.findall((s or "").lower()))
        return {s2[i:i + 3] for i in range(max(0, len(s2) - 2))}

    def _one(self, q: str, d: str) -> float:
        qw, dw = self._words(q), self._words(d)
        qt, dt = self._trigrams(q), self._trigrams(d)
        j_w = len(qw & dw) / max(1, len(qw | dw))
        j_t = len(qt & dt) / max(1, len(qt | dt))
        lq, ld = max(1, len(q)), max(1, len(d))
        lr = 1.0 - min(1.0, abs(math.log(lq) - math.log(ld)) / 2.0)
        return float(0.45 * j_w + 0.45 * j_t + 0.10 * lr)

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        return np.array([self._one(query, d) for d in docs], dtype=np.float32)


def _pick_backend() -> _RerankBackend:
    if os.environ.get("GPDM_RERANK_FORCE_LEX", "").lower() in ("1", "true"):
        return _LexicalBackend()
    model_name = os.environ.get("GPDM_RERANK_MODEL",
                                "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        return _CrossEncoderBackend(model_name)
    except Exception as e:
        logger.info("[reranker] CrossEncoder unavailable (%s); using lexical",
                    e)
        return _LexicalBackend()


@dataclass
class RerankResult:
    indices: List[int]
    scores: List[float]
    backend: str
    blend: float
    base_scores: List[float]
    cross_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indices": list(self.indices),
            "scores": [round(float(s), 4) for s in self.scores],
            "backend": self.backend,
            "blend": self.blend,
            "base_scores": [round(float(s), 4) for s in self.base_scores],
            "cross_scores": [round(float(s), 4) for s in self.cross_scores],
        }


class Reranker:

    def __init__(self):
        self.enabled = os.environ.get("GPDM_RERANK_ENABLED", "0").lower() \
                        not in ("0", "false", "no", "off")
        try:
            blend = float(os.environ.get("GPDM_RERANK_BLEND", "0.70"))
        except Exception:
            blend = 0.70
        self.blend = max(0.0, min(1.0, blend))
        self._backend: Optional[_RerankBackend] = None
        self._backend_name = ""
        self._last_ms: float = 0.0
        self._call_count: int = 0
        self._pair_count: int = 0

    def _get(self) -> _RerankBackend:
        if self._backend is None:
            self._backend = _pick_backend()
            self._backend_name = self._backend.name
        return self._backend

    def rerank(self, query: str, candidates: Sequence[str],
               base_scores: Optional[Sequence[float]] = None,
               k: Optional[int] = None) -> RerankResult:
        n = len(candidates)
        base = np.asarray(base_scores if base_scores is not None
                           else [0.5] * n, dtype=np.float32).reshape(-1)
        if n == 0:
            return RerankResult([], [], self._backend_name or "none",
                                self.blend, [], [])
        t0 = time.time()
        cross = self._get().score(query, candidates)
        self._last_ms = (time.time() - t0) * 1000.0
        self._call_count += 1
        self._pair_count += n

        b = (base + 1.0) / 2.0 if base.min() < 0 else base.copy()
        b = np.clip(b, 0.0, 1.0)
        blended = self.blend * cross + (1.0 - self.blend) * b

        k = k if k is not None else n
        order = np.argsort(-blended)[:k]
        return RerankResult(
            indices=[int(i) for i in order],
            scores=[float(blended[i]) for i in order],
            backend=self._backend_name or (self._backend.name if self._backend else "none"),
            blend=self.blend,
            base_scores=[float(b[i]) for i in order],
            cross_scores=[float(cross[i]) for i in order],
        )

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self._backend_name or "uninitialized",
            "blend": self.blend,
            "last_call_ms": round(self._last_ms, 2),
            "calls": self._call_count,
            "pairs_scored": self._pair_count,
        }


_RERANKER: Optional[Reranker] = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> Reranker:
    global _RERANKER
    with _RERANKER_LOCK:
        if _RERANKER is None:
            _RERANKER = Reranker()
    return _RERANKER


def reset_reranker() -> None:
    global _RERANKER
    with _RERANKER_LOCK:
        _RERANKER = None


__all__ = ["Reranker", "RerankResult", "get_reranker", "reset_reranker"]
