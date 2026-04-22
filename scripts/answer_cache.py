from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT    NOT NULL,
    normalized  TEXT    NOT NULL UNIQUE,
    vec         BLOB    NOT NULL,
    dim         INTEGER NOT NULL,
    encoder     TEXT    NOT NULL,
    answer_json TEXT    NOT NULL,
    created_ts  REAL    NOT NULL,
    last_hit_ts REAL    NOT NULL,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    byte_size   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_answer_cache_last_hit
    ON answer_cache(last_hit_ts);
CREATE INDEX IF NOT EXISTS ix_answer_cache_created
    ON answer_cache(created_ts);
"""

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(q: str) -> str:
    q = (q or "").lower().strip()
    q = _PUNCT.sub(" ", q)
    return _WS.sub(" ", q).strip()


@dataclass
class _Config:
    enabled: bool
    threshold: float
    max_entries: int
    ttl_sec: int

    @classmethod
    def from_env(cls) -> "_Config":
        def _f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return default

        def _i(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return default

        return cls(
            enabled=os.environ.get("GPDM_ANSWER_CACHE_ENABLED", "1").lower()
                     not in ("0", "false", "no", "off"),
            threshold=_f("GPDM_ANSWER_CACHE_THRESHOLD", 0.85),
            max_entries=_i("GPDM_ANSWER_CACHE_MAX", 5000),
            ttl_sec=_i("GPDM_ANSWER_CACHE_TTL_SEC", 86400),
        )


class AnswerCache:

    def __init__(self, data_dir: str, cfg: Optional[_Config] = None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "answer_cache.db")
        self.cfg = cfg or _Config.from_env()
        self._lock = threading.RLock()

        self._ids: list = []
        self._norms: list = []
        self._vecs: Optional[np.ndarray] = None
        self._last_hits: list = []

        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

        self._encoder = None
        self._encoder_name = ""
        self._dim = 0
        self._init_db()
        self._warm()

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from retrieval import _pick_encoder
            enc = _pick_encoder()
        except Exception as e:
            logger.warning("[answer_cache] retrieval encoder unavailable (%s); "
                           "cache will run in exact-match-only mode", e)
            enc = None
        self._encoder = enc
        self._encoder_name = getattr(enc, "name", "none")
        self._dim = int(getattr(enc, "dim", 0) or 0)
        if "GPDM_ANSWER_CACHE_THRESHOLD" not in os.environ and enc is not None:
            if self._encoder_name == "hash":
                self.cfg.threshold = 0.75
        return enc

    def _encode(self, text: str) -> Optional[np.ndarray]:
        enc = self._get_encoder()
        if enc is None:
            return None
        v = enc.encode([text])[0].astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0.0:
            v = v / n
        return v

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def _warm(self) -> None:
        if not self.cfg.enabled:
            return
        enc = self._get_encoder()
        enc_name = self._encoder_name or "none"

        cutoff = time.time() - self.cfg.ttl_sec
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM answer_cache WHERE created_ts < ? OR encoder != ?",
                (cutoff, enc_name),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT id, normalized, vec, dim, last_hit_ts "
                "FROM answer_cache ORDER BY last_hit_ts DESC"
            ).fetchall()
            ids, norms, last, vecs = [], [], [], []
            for rid, nq, blob, dim, lh in rows:
                try:
                    v = np.frombuffer(blob, dtype=np.float32)
                    if v.size != dim:
                        continue
                    ids.append(rid)
                    norms.append(nq)
                    last.append(float(lh))
                    vecs.append(v)
                except Exception:
                    continue
            if vecs:
                self._vecs = np.vstack(vecs)
                self._ids, self._norms, self._last_hits = ids, norms, last
            else:
                self._vecs = None
                self._ids, self._norms, self._last_hits = [], [], []
            logger.info("[answer_cache] warmed %d entries (encoder=%s dim=%s)",
                        len(self._ids), enc_name, self._dim)

    def lookup(self, question: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled:
            return None
        norm = _normalize(question)
        if not norm:
            return None

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT id, answer_json, hit_count FROM answer_cache "
                    "WHERE normalized = ? LIMIT 1", (norm,),
                ).fetchone()
                if row:
                    rid, ans_json, hc = row
                    now = time.time()
                    conn.execute(
                        "UPDATE answer_cache SET hit_count = ?, "
                        "last_hit_ts = ? WHERE id = ?",
                        (int(hc) + 1, now, rid),
                    )
                    conn.commit()
                    try:
                        i = self._ids.index(rid)
                        self._last_hits[i] = now
                    except ValueError:
                        pass
                    self._hits_exact += 1
                    answer = json.loads(ans_json)
                    return self._decorate(answer, 1.0, "exact", rid)
            finally:
                conn.close()

        if self._vecs is None or self._vecs.shape[0] == 0:
            self._misses += 1
            return None
        q = self._encode(question)
        if q is None or q.size == 0 or q.size != self._vecs.shape[1]:
            self._misses += 1
            return None
        sims = self._vecs @ q
        j = int(np.argmax(sims))
        best = float(sims[j])
        if best < self.cfg.threshold:
            self._misses += 1
            return None
        rid = self._ids[j]
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT answer_json, hit_count FROM answer_cache WHERE id = ?",
                    (rid,),
                ).fetchone()
                if not row:
                    self._misses += 1
                    return None
                ans_json, hc = row
                conn.execute(
                    "UPDATE answer_cache SET hit_count = ?, "
                    "last_hit_ts = ? WHERE id = ?",
                    (int(hc) + 1, now, rid),
                )
                conn.commit()
            finally:
                conn.close()
            self._last_hits[j] = now
            self._hits_semantic += 1
            return self._decorate(json.loads(ans_json), best, "semantic", rid)

    def store(self, question: str, answer: Dict[str, Any]) -> bool:
        if not self.cfg.enabled or not question or not isinstance(answer, dict):
            return False
        if answer.get("error") or answer.get("auto_repair"):
            return False
        norm = _normalize(question)
        if not norm:
            return False
        v = self._encode(question)
        if v is None:
            v = np.zeros((self._dim or 1,), dtype=np.float32)

        compact = {k: v2 for k, v2 in answer.items()
                   if k not in ("_trace", "_debug", "_raw_llm")}
        try:
            ans_json = json.dumps(compact, default=str)
        except Exception as e:
            logger.debug("[answer_cache] not cacheable (%s)", e)
            return False
        byte_size = len(ans_json.encode("utf-8"))
        if byte_size > 2_000_000:
            return False

        now = time.time()
        blob = v.astype(np.float32).tobytes()
        enc_name = self._encoder_name or "none"

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                existing = conn.execute(
                    "SELECT id FROM answer_cache WHERE normalized = ?",
                    (norm,),
                ).fetchone()
                if existing:
                    rid = existing[0]
                    conn.execute(
                        "UPDATE answer_cache SET question = ?, vec = ?, "
                        "dim = ?, encoder = ?, answer_json = ?, "
                        "created_ts = ?, last_hit_ts = ?, byte_size = ? "
                        "WHERE id = ?",
                        (question, blob, int(v.size), enc_name, ans_json,
                         now, now, byte_size, rid),
                    )
                else:
                    cur = conn.execute(
                        "INSERT INTO answer_cache "
                        "(question, normalized, vec, dim, encoder, answer_json, "
                        " created_ts, last_hit_ts, hit_count, byte_size) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                        (question, norm, blob, int(v.size), enc_name,
                         ans_json, now, now, byte_size),
                    )
                    rid = int(cur.lastrowid or 0)
                conn.commit()
            finally:
                conn.close()

            self._stores += 1
            if existing:
                try:
                    i = self._ids.index(rid)
                    self._vecs[i] = v
                    self._last_hits[i] = now
                except (ValueError, TypeError, AttributeError):
                    self._warm()
            else:
                if self._vecs is None or self._vecs.size == 0:
                    self._vecs = v.reshape(1, -1).astype(np.float32)
                else:
                    self._vecs = np.vstack([self._vecs, v.reshape(1, -1)])
                self._ids.append(rid)
                self._norms.append(norm)
                self._last_hits.append(now)

            if len(self._ids) > self.cfg.max_entries:
                self._evict_lru()

        return True

    def _evict_lru(self) -> None:
        target = max(1, int(self.cfg.max_entries * 0.95))
        n = len(self._ids)
        to_drop = n - target
        if to_drop <= 0:
            return
        order = np.argsort(np.asarray(self._last_hits))[:to_drop]
        drop_ids = [self._ids[i] for i in order]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[order] = False

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "DELETE FROM answer_cache WHERE id = ?",
                [(int(i),) for i in drop_ids],
            )
            conn.commit()
        finally:
            conn.close()

        self._ids     = [self._ids[i]       for i in range(n) if keep_mask[i]]
        self._norms   = [self._norms[i]     for i in range(n) if keep_mask[i]]
        self._last_hits = [self._last_hits[i] for i in range(n) if keep_mask[i]]
        if self._vecs is not None:
            self._vecs = self._vecs[keep_mask]
        self._evictions += len(drop_ids)
        logger.info("[answer_cache] LRU evicted %d entries", len(drop_ids))

    def clear(self) -> int:
        with self._lock:
            n = len(self._ids)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM answer_cache")
                conn.commit()
            finally:
                conn.close()
            self._ids, self._norms, self._last_hits = [], [], []
            self._vecs = None
            return n

    def status(self) -> Dict[str, Any]:
        with self._lock:
            total_bytes = 0
            try:
                conn = sqlite3.connect(self.db_path)
                row = conn.execute(
                    "SELECT COALESCE(SUM(byte_size),0), COUNT(*) FROM answer_cache"
                ).fetchone()
                conn.close()
                if row:
                    total_bytes = int(row[0] or 0)
            except Exception:
                pass
            total_lookups = self._hits_exact + self._hits_semantic + self._misses
            hit_rate = ((self._hits_exact + self._hits_semantic) / total_lookups
                        if total_lookups else 0.0)
            return {
                "enabled": self.cfg.enabled,
                "encoder": self._encoder_name or "none",
                "dim": self._dim,
                "entries": len(self._ids),
                "bytes": total_bytes,
                "max_entries": self.cfg.max_entries,
                "threshold": self.cfg.threshold,
                "ttl_sec": self.cfg.ttl_sec,
                "hits_exact": self._hits_exact,
                "hits_semantic": self._hits_semantic,
                "misses": self._misses,
                "stores": self._stores,
                "evictions": self._evictions,
                "hit_rate": round(hit_rate, 4),
                "db_path": self.db_path,
            }

    def _decorate(self, answer: Dict[str, Any], score: float,
                  match_type: str, rid: int) -> Dict[str, Any]:
        out = dict(answer)
        out["cached"] = True
        out["cache_match"] = match_type
        out["cache_score"] = round(float(score), 4)
        out["cache_id"]    = int(rid)
        return out


_CACHE: Optional[AnswerCache] = None
_CACHE_LOCK = threading.Lock()


def get_cache(data_dir: Optional[str] = None) -> Optional[AnswerCache]:
    global _CACHE
    if data_dir is None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE.data_dir != data_dir:
            try:
                _CACHE = AnswerCache(data_dir)
            except Exception as e:
                logger.error("[answer_cache] init failed: %s", e)
                _CACHE = None
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


__all__ = ["AnswerCache", "get_cache", "reset_cache"]
