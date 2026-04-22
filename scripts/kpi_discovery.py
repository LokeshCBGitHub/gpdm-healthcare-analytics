from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were",
    "of", "in", "on", "to", "for", "by", "with", "from", "at", "as",
    "what", "how", "when", "where", "which", "who", "show", "tell",
    "give", "get", "find", "list", "do", "does", "did", "has", "have",
    "can", "could", "would", "should", "will", "me", "my", "our",
    "this", "that", "these", "those", "it", "its", "over", "about",
    "last", "next", "total", "number", "count", "please",
}


@dataclass
class DiscoveredKPI:
    label: str
    signal_tokens: List[str]
    support: int
    frequency: int
    users: int
    dominant_intent: str
    recency_days: float
    novelty_score: float
    discovery_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "signal_tokens": list(self.signal_tokens),
            "support": int(self.support),
            "frequency": int(self.frequency),
            "users": int(self.users),
            "dominant_intent": self.dominant_intent,
            "recency_days": round(float(self.recency_days), 2),
            "novelty_score": round(float(self.novelty_score), 3),
            "discovery_score": round(float(self.discovery_score), 4),
        }


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOP and len(t) > 2]


class KPIDiscovery:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "query_tracker.db")
        self._lock = threading.RLock()

        self.enabled = os.environ.get("GPDM_KPI_DISCOVERY_ENABLED", "1").lower() \
                        not in ("0", "false", "no", "off")
        def _i(n: str, d: int) -> int:
            try: return int(os.environ.get(n, d))
            except Exception: return d
        def _f(n: str, d: float) -> float:
            try: return float(os.environ.get(n, d))
            except Exception: return d

        self.max_queries  = _i("GPDM_KPI_DISC_MAX_QUERIES", 10000)
        self.cluster_sim  = _f("GPDM_KPI_DISC_CLUSTER_SIM", 0.55)
        self.min_support  = _i("GPDM_KPI_DISC_MIN_SUPPORT", 5)
        self.novelty_sim  = _f("GPDM_KPI_DISC_NOVELTY_SIM", 0.75)

        self._encoder = None
        self._encoder_name = ""
        self._last_run_ts: float = 0.0
        self._last_count: int = 0

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from retrieval import _pick_encoder
            self._encoder = _pick_encoder()
            self._encoder_name = getattr(self._encoder, "name", "none")
        except Exception as e:
            logger.warning("[kpi_discovery] encoder unavailable: %s", e)
            self._encoder = None
        return self._encoder

    def _load_queries(self) -> List[Tuple[str, int, int, float, str]]:
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT question, "
                    "       COUNT(*) AS n, "
                    "       COUNT(DISTINCT user_id) AS u, "
                    "       MAX(timestamp) AS t, "
                    "       COALESCE(intent, '') AS it "
                    "FROM query_log WHERE success = 1 "
                    "GROUP BY question "
                    "ORDER BY n DESC "
                    "LIMIT ?", (self.max_queries,),
                )
                return [(r[0], int(r[1] or 0), int(r[2] or 0),
                         float(r[3] or 0), str(r[4] or ""))
                        for r in cur.fetchall() if r and r[0]]
            finally:
                conn.close()
        except Exception as e:
            logger.debug("[kpi_discovery] query_log read failed: %s", e)
            return []

    def _known_kpi_texts(self) -> List[str]:
        out: List[str] = []
        try:
            import executive_kpis as _ek
            reg = getattr(_ek, "KPI_REGISTRY", None) or getattr(_ek, "KPIS", None)
            if isinstance(reg, dict):
                for v in reg.values():
                    if isinstance(v, dict):
                        for k in ("label", "question", "name", "description"):
                            if v.get(k):
                                out.append(str(v[k]))
                                break
            elif isinstance(reg, list):
                for v in reg:
                    if isinstance(v, dict):
                        for k in ("label", "question", "name", "description"):
                            if v.get(k):
                                out.append(str(v[k]))
                                break
                    elif isinstance(v, str):
                        out.append(v)
        except Exception:
            pass
        return out

    def _cluster(self, vecs: np.ndarray, freqs: List[int]
                 ) -> List[List[int]]:
        n = vecs.shape[0]
        if n == 0:
            return []
        assigned = np.zeros(n, dtype=bool)
        order = sorted(range(n), key=lambda i: -freqs[i])
        clusters: List[List[int]] = []
        for seed in order:
            if assigned[seed]:
                continue
            sims = vecs @ vecs[seed]
            members = np.where((sims >= self.cluster_sim) & (~assigned))[0]
            if members.size == 0:
                members = np.array([seed], dtype=int)
            assigned[members] = True
            clusters.append(members.tolist())
        return clusters

    def discover(self, limit: int = 20) -> List[DiscoveredKPI]:
        if not self.enabled:
            return []
        enc = self._get_encoder()
        if enc is None:
            return []
        rows = self._load_queries()
        if len(rows) < self.min_support:
            return []

        texts   = [r[0] for r in rows]
        freqs   = [r[1] for r in rows]
        users   = [r[2] for r in rows]
        stamps  = [r[3] for r in rows]
        intents = [r[4] for r in rows]

        vecs = np.asarray(enc.encode(texts), dtype=np.float32)
        nrm = np.linalg.norm(vecs, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
        vecs = vecs / nrm

        clusters = self._cluster(vecs, freqs)

        known_vecs = None
        known_texts = self._known_kpi_texts()
        if known_texts:
            try:
                kv = np.asarray(enc.encode(known_texts), dtype=np.float32)
                kn = np.linalg.norm(kv, axis=1, keepdims=True); kn[kn == 0] = 1.0
                known_vecs = kv / kn
            except Exception:
                known_vecs = None

        now = time.time()
        out: List[DiscoveredKPI] = []
        for members in clusters:
            support = len(members)
            seed = max(members, key=lambda i: freqs[i])
            label = texts[seed]

            freq_total = sum(freqs[i] for i in members)
            if freq_total < self.min_support:
                continue
            user_total = sum(users[i] for i in members)
            latest = max(stamps[i] for i in members)
            avg_age_days = np.mean([
                (now - stamps[i]) / 86400.0 for i in members if stamps[i] > 0
            ]) if any(stamps[i] > 0 for i in members) else 0.0

            intent_counter = Counter(intents[i] for i in members if intents[i])
            dom_intent = intent_counter.most_common(1)[0][0] if intent_counter else ""

            tok_counter: Counter = Counter()
            for i in members:
                tok_counter.update(_tokens(texts[i]))
            signal_tokens = [w for w, _ in tok_counter.most_common(5)]

            if known_vecs is not None and known_vecs.shape[0] > 0:
                centroid = vecs[members].mean(axis=0)
                cn = float(np.linalg.norm(centroid))
                if cn > 0: centroid = centroid / cn
                max_sim = float((known_vecs @ centroid).max())
                novelty = max(0.0, 1.0 - max_sim)
                if max_sim >= self.novelty_sim:
                    continue
            else:
                novelty = 1.0

            recency = max(0.3, 1.0 - min(avg_age_days / 90.0, 0.7))
            disc_score = math.log1p(freq_total) * recency * (0.5 + 0.5 * novelty)

            out.append(DiscoveredKPI(
                label=label,
                signal_tokens=signal_tokens,
                support=support,
                frequency=freq_total,
                users=user_total,
                dominant_intent=dom_intent,
                recency_days=float(avg_age_days),
                novelty_score=float(novelty),
                discovery_score=float(disc_score),
            ))

        out.sort(key=lambda d: -d.discovery_score)
        self._last_run_ts = now
        self._last_count = len(out)
        return out[:limit]

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "encoder": self._encoder_name or "uninitialized",
            "last_run_ts": self._last_run_ts,
            "last_count": self._last_count,
            "config": {
                "max_queries": self.max_queries,
                "cluster_sim": self.cluster_sim,
                "min_support": self.min_support,
                "novelty_sim": self.novelty_sim,
            },
        }


_DISC: Optional[KPIDiscovery] = None
_LOCK = threading.Lock()


def get_discovery(data_dir: Optional[str] = None) -> Optional[KPIDiscovery]:
    global _DISC
    if data_dir is None:
        return _DISC
    with _LOCK:
        if _DISC is None or _DISC.data_dir != data_dir:
            try:
                _DISC = KPIDiscovery(data_dir)
            except Exception as e:
                logger.error("[kpi_discovery] init failed: %s", e)
                _DISC = None
    return _DISC


def reset_discovery() -> None:
    global _DISC
    with _LOCK:
        _DISC = None


__all__ = ["DiscoveredKPI", "KPIDiscovery",
           "get_discovery", "reset_discovery"]
