from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundedFollowUp:
    prompt: str
    score: float
    support: int
    from_intent: str
    to_intent: str
    avg_recency_days: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score": round(float(self.score), 4),
            "support": int(self.support),
            "from_intent": self.from_intent,
            "to_intent": self.to_intent,
            "avg_recency_days": round(float(self.avg_recency_days), 2),
            "kind": "grounded",
        }


class TransitionGraph:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "query_tracker.db")
        self._lock = threading.RLock()

        self.enabled = os.environ.get("GPDM_GROUNDED_FU_ENABLED", "1").lower() \
                        not in ("0", "false", "no", "off")
        try:
            self.max_pivots = int(os.environ.get("GPDM_GROUNDED_FU_MAX_PIVOTS",
                                                 "3000"))
        except Exception:
            self.max_pivots = 3000
        try:
            self.min_sim = float(os.environ.get("GPDM_GROUNDED_FU_MIN_SIM",
                                                "0.40"))
        except Exception:
            self.min_sim = 0.40
        try:
            self.window_sec = int(os.environ.get("GPDM_GROUNDED_FU_WINDOW_SEC",
                                                 "600"))
        except Exception:
            self.window_sec = 600

        self._pivot_texts: List[str] = []
        self._pivot_vecs: Optional[np.ndarray] = None
        self._pivot_intents: List[str] = []
        self._transitions: List[List[Tuple[str, str, float, str]]] = []
        self._built_at: float = 0.0
        self._encoder = None
        self._encoder_name = ""
        self._dim = 0

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from retrieval import _pick_encoder
            self._encoder = _pick_encoder()
            self._encoder_name = getattr(self._encoder, "name", "none")
            self._dim = int(getattr(self._encoder, "dim", 0) or 0)
        except Exception as e:
            logger.warning("[grounded_fu] encoder unavailable: %s", e)
            self._encoder = None
        return self._encoder

    def rebuild(self) -> None:
        if not self.enabled:
            return
        enc = self._get_encoder()
        if enc is None:
            return
        if not os.path.exists(self.db_path):
            logger.info("[grounded_fu] query_tracker.db not present; skipping")
            return

        rows: List[Tuple[str, str, str, float]] = []
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT question, COALESCE(intent,''), "
                    "       COALESCE(user_id,'anon'), timestamp "
                    "FROM query_log WHERE success = 1 "
                    "ORDER BY user_id ASC, timestamp ASC "
                    "LIMIT ?", (self.max_pivots * 4,),
                )
                rows = [(r[0], r[1], r[2], float(r[3] or 0.0))
                        for r in cur.fetchall()
                        if r and r[0]]
            finally:
                conn.close()
        except Exception as e:
            logger.debug("[grounded_fu] query_log read failed: %s", e)
            return

        if len(rows) < 4:
            logger.info("[grounded_fu] insufficient history (%d rows)", len(rows))
            return

        now = time.time()
        pivots: Dict[str, Dict[str, Any]] = {}
        for i in range(len(rows) - 1):
            q_prev, intent_prev, u_prev, t_prev = rows[i]
            q_next, intent_next, u_next, t_next = rows[i + 1]
            if u_prev != u_next:
                continue
            if (t_next - t_prev) <= 0 or (t_next - t_prev) > self.window_sec:
                continue
            if q_prev.strip().lower() == q_next.strip().lower():
                continue
            age_sec = max(0.0, now - t_prev)
            pivot = pivots.setdefault(q_prev, {
                "intent": intent_prev,
                "transitions": [],
            })
            pivot["transitions"].append(
                (q_next, intent_next, age_sec, u_prev)
            )

        top = sorted(pivots.items(),
                     key=lambda kv: -len(kv[1]["transitions"]))
        top = top[: self.max_pivots]
        if not top:
            logger.info("[grounded_fu] no transitions found")
            return

        texts = [t for t, _ in top]
        vecs = enc.encode(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs = vecs / norms

        with self._lock:
            self._pivot_texts   = texts
            self._pivot_vecs    = vecs
            self._pivot_intents = [p[1]["intent"] for p in top]
            self._transitions   = [p[1]["transitions"] for p in top]
            self._built_at      = now

        logger.info("[grounded_fu] built graph: %d pivots, %d transitions, "
                    "encoder=%s dim=%d",
                    len(texts),
                    sum(len(t) for t in self._transitions),
                    self._encoder_name, self._dim)

    def suggest(self, question: str, k: int = 4,
                min_score: float = 0.25) -> List[GroundedFollowUp]:
        if not self.enabled:
            return []
        if self._pivot_vecs is None or len(self._pivot_texts) == 0:
            return []
        enc = self._get_encoder()
        if enc is None:
            return []
        q = enc.encode([question])[0].astype(np.float32)
        n = float(np.linalg.norm(q))
        if n > 0.0:
            q = q / n
        sims = self._pivot_vecs @ q
        top_k = min(max(k * 4, 8), sims.size)
        top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        agg: Dict[str, Dict[str, Any]] = {}
        for i in top_idx:
            s = float(sims[i])
            if s < self.min_sim:
                continue
            from_intent = self._pivot_intents[i] or ""
            for q_next, intent_next, age_sec, user_id in self._transitions[i]:
                key = q_next.strip().lower()
                if not key:
                    continue
                if key == (question or "").strip().lower():
                    continue
                age_days = age_sec / 86400.0
                decay = max(0.3, 1.0 - min(age_days / 90.0, 0.7))
                e = agg.setdefault(key, {
                    "prompt": q_next.strip(),
                    "score": 0.0,
                    "users": set(),
                    "count": 0,
                    "to_intent": intent_next or "",
                    "from_intent": from_intent,
                    "age_sum": 0.0,
                })
                e["score"]  += s * decay
                e["count"]  += 1
                e["users"].add(user_id)
                e["age_sum"] += age_days

        if not agg:
            return []
        max_score = max(v["score"] for v in agg.values()) or 1.0
        out: List[GroundedFollowUp] = []
        for v in agg.values():
            norm = v["score"] / max_score
            if norm < min_score:
                continue
            out.append(GroundedFollowUp(
                prompt=v["prompt"],
                score=float(norm),
                support=len(v["users"]),
                from_intent=v["from_intent"],
                to_intent=v["to_intent"],
                avg_recency_days=(v["age_sum"] / max(1, v["count"])),
            ))
        out.sort(key=lambda g: (-g.score, -g.support))
        return out[:k]

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "encoder": self._encoder_name or "none",
            "dim": self._dim,
            "pivots": len(self._pivot_texts),
            "transitions": sum(len(t) for t in self._transitions),
            "built_at": self._built_at,
            "config": {
                "max_pivots": self.max_pivots,
                "min_sim": self.min_sim,
                "window_sec": self.window_sec,
            },
        }


_GRAPH: Optional[TransitionGraph] = None
_GRAPH_LOCK = threading.Lock()
_REBUILD_EVERY_SEC = 900


def get_graph(data_dir: Optional[str] = None,
              force_rebuild: bool = False) -> Optional[TransitionGraph]:
    global _GRAPH
    if data_dir is None:
        return _GRAPH
    with _GRAPH_LOCK:
        if _GRAPH is None or _GRAPH.data_dir != data_dir:
            try:
                _GRAPH = TransitionGraph(data_dir)
                _GRAPH.rebuild()
            except Exception as e:
                logger.error("[grounded_fu] init failed: %s", e)
                _GRAPH = None
                return None
        elif force_rebuild or (time.time() - _GRAPH._built_at
                                > _REBUILD_EVERY_SEC):
            try:
                _GRAPH.rebuild()
            except Exception as e:
                logger.error("[grounded_fu] rebuild failed: %s", e)
    return _GRAPH


def reset_graph() -> None:
    global _GRAPH
    with _GRAPH_LOCK:
        _GRAPH = None


__all__ = ["GroundedFollowUp", "TransitionGraph",
           "get_graph", "reset_graph"]
