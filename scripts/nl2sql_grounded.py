from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_INTENT_PATTERNS = [
    ("count",   re.compile(r"\b(how many|count of|number of|total number)\b")),
    ("sum",     re.compile(r"\b(total|sum of|overall|aggregate)\b")),
    ("avg",     re.compile(r"\b(average|mean|typical|per\s+member|per\s+admit)\b")),
    ("trend",   re.compile(r"\b(trend|over time|by month|by quarter|by year|monthly|quarterly|yearly)\b")),
    ("topN",    re.compile(r"\b(top\s+\d+|highest|largest|worst|best)\b")),
    ("compare", re.compile(r"\b(vs|versus|compared? to|difference between)\b")),
]

_TIME_HINTS = [
    (re.compile(r"last\s+(\d+)\s+(day|week|month|year)s?"),
     lambda m: (int(m.group(1)), m.group(2))),
    (re.compile(r"(?:this|current)\s+(quarter|year|month)"),
     lambda m: (1, m.group(1))),
    (re.compile(r"\b(ytd|year\s*to\s*date)\b"),
     lambda m: (1, "year")),
]


@dataclass
class ColumnPick:
    table: str
    column: str
    score: float
    data_type: str = ""
    semantic_type: str = ""
    top_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table, "column": self.column,
            "score": round(float(self.score), 4),
            "data_type": self.data_type,
            "semantic_type": self.semantic_type,
            "top_values": list(self.top_values)[:3],
        }


class SchemaGrounder:
    def __init__(self, schema_registry):
        self.schema = schema_registry
        self._encoder = None
        self._encoder_name = ""
        self._texts: List[str] = []
        self._cols: List[ColumnPick] = []
        self._vecs: Optional[np.ndarray] = None
        self._built_at: float = 0.0
        self._lock = threading.RLock()

        def _i(n, d):
            try: return int(os.environ.get(n, d))
            except Exception: return d
        self.top_cols   = _i("GPDM_NL2SQL_TOP_COLS", 8)
        self.top_tables = _i("GPDM_NL2SQL_TOP_TABLES", 4)
        self.use_rerank = os.environ.get("GPDM_NL2SQL_RERANK", "1").lower() \
                           not in ("0", "false", "no", "off")

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from retrieval import _pick_encoder
            self._encoder = _pick_encoder()
            self._encoder_name = getattr(self._encoder, "name", "none")
        except Exception as e:
            logger.warning("[nl2sql] encoder unavailable: %s", e)
            self._encoder = None
        return self._encoder

    def _col_text(self, table: str, col: Dict[str, Any]) -> str:
        name = col.get("name", "")
        sem  = col.get("semantic_type", "") or col.get("data_type", "")
        name_readable = re.sub(r"[_\-]+", " ",
                               re.sub(r"([a-z])([A-Z])", r"\1 \2", name)).lower()
        parts = [f"{table} {name_readable}"]
        if sem:
            parts.append(f"type {sem}")
        tv = col.get("top_values") or []
        if tv:
            sample = ", ".join(str(v) for v in tv[:3])
            parts.append(f"examples: {sample}")
        return ". ".join(parts)

    def rebuild(self) -> None:
        enc = self._get_encoder()
        if enc is None:
            return
        texts, cols = [], []
        tables = getattr(self.schema, "tables", {}) or {}
        for tname, cdefs in tables.items():
            for c in (cdefs or []):
                texts.append(self._col_text(tname, c))
                cols.append(ColumnPick(
                    table=tname,
                    column=c.get("name", ""),
                    score=0.0,
                    data_type=c.get("data_type", ""),
                    semantic_type=c.get("semantic_type", ""),
                    top_values=list(c.get("top_values") or []),
                ))
        if not texts:
            logger.info("[nl2sql] empty schema; nothing to index")
            with self._lock:
                self._texts, self._cols, self._vecs = [], [], None
            return
        vecs = np.asarray(enc.encode(texts), dtype=np.float32)
        n = np.linalg.norm(vecs, axis=1, keepdims=True); n[n == 0] = 1.0
        vecs = vecs / n
        with self._lock:
            self._texts, self._cols = texts, cols
            self._vecs = vecs
            self._built_at = time.time()
        logger.info("[nl2sql] schema index built: %d columns, encoder=%s",
                    len(texts), self._encoder_name)

    def ground(self, question: str, top_cols: Optional[int] = None,
               top_tables: Optional[int] = None,
               rerank: Optional[bool] = None) -> Dict[str, Any]:
        if self._vecs is None:
            self.rebuild()
        if self._vecs is None or self._vecs.shape[0] == 0 or not question:
            return self._empty_bundle(question, reason="schema_unavailable")

        enc = self._get_encoder()
        q = enc.encode([question])[0].astype(np.float32)
        nn = float(np.linalg.norm(q))
        if nn > 0: q = q / nn
        sims = self._vecs @ q

        kc = max(1, int(top_cols if top_cols is not None else self.top_cols))
        recall = min(max(kc * 4, 20), sims.size)
        cand = np.argpartition(-sims, recall - 1)[:recall]
        cand = cand[np.argsort(-sims[cand])]
        base_scores = [float(sims[i]) for i in cand]
        texts_cand  = [self._texts[i] for i in cand]

        do_rerank = rerank if rerank is not None else self.use_rerank
        final_idx: List[int]
        final_scores: List[float]
        rerank_backend = None
        if do_rerank:
            try:
                from reranker import get_reranker
                rr = get_reranker().rerank(question, texts_cand,
                                           base_scores=base_scores, k=kc)
                final_idx    = [int(cand[i]) for i in rr.indices]
                final_scores = list(rr.scores)
                rerank_backend = rr.backend
            except Exception as e:
                logger.debug("[nl2sql] rerank unavailable: %s", e)
                final_idx    = [int(i) for i in cand[:kc]]
                final_scores = [float(sims[i]) for i in final_idx]
        else:
            final_idx    = [int(i) for i in cand[:kc]]
            final_scores = [float(sims[i]) for i in final_idx]

        picks: List[ColumnPick] = []
        for i, s in zip(final_idx, final_scores):
            cp = self._cols[i]
            picks.append(ColumnPick(
                table=cp.table, column=cp.column, score=s,
                data_type=cp.data_type, semantic_type=cp.semantic_type,
                top_values=list(cp.top_values)[:3],
            ))

        seen, tables = set(), []
        for p in picks:
            if p.table not in seen:
                seen.add(p.table); tables.append(p.table)
        kt = max(1, int(top_tables if top_tables is not None else self.top_tables))
        tables = tables[:kt]
        picks  = [p for p in picks if p.table in set(tables)]

        join_path: List[Tuple[str, str, str]] = []
        try:
            if hasattr(self.schema, "find_join_path"):
                join_path = self.schema.find_join_path(tables) or []
        except Exception:
            join_path = []

        intent = self._detect_intent(question)
        time_window = self._detect_time_window(question)

        candidate_sql, sql_confidence = self._template_sql(
            question, intent, time_window, tables, picks, join_path,
        )

        return {
            "question": question,
            "intent": intent,
            "time_window": time_window,
            "selected_tables": list(tables),
            "selected_columns": [p.to_dict() for p in picks],
            "join_path": [
                {"from": a, "to": b, "on": c} for (a, b, c) in join_path
            ],
            "rerank_backend": rerank_backend,
            "candidate_sql": candidate_sql,
            "sql_confidence": sql_confidence,
            "prompt_context": self._build_prompt_context(
                question, intent, time_window, tables, picks, join_path,
            ),
            "encoder": self._encoder_name,
        }

    def _detect_intent(self, q: str) -> str:
        ql = (q or "").lower()
        for name, pat in _INTENT_PATTERNS:
            if pat.search(ql):
                return name
        return "select"

    def _detect_time_window(self, q: str) -> Optional[Dict[str, Any]]:
        ql = (q or "").lower()
        for pat, fn in _TIME_HINTS:
            m = pat.search(ql)
            if m:
                n, unit = fn(m)
                return {"n": int(n), "unit": str(unit)}
        return None

    def _template_sql(self, question: str, intent: str,
                      tw: Optional[Dict[str, Any]],
                      tables: List[str], picks: List[ColumnPick],
                      join_path: List[Tuple[str, str, str]],
                      ) -> Tuple[Optional[str], float]:
        if not tables or not picks:
            return None, 0.0

        base = tables[0]
        score_mean = float(np.mean([p.score for p in picks])) if picks else 0.0

        numeric_cols = [p for p in picks
                        if p.data_type.lower() in
                        ("int", "integer", "bigint", "numeric", "number",
                         "float", "real", "double", "decimal")
                        or "amount" in p.column.lower()
                        or "cost" in p.column.lower()
                        or "spend" in p.column.lower()
                        or "count" in p.column.lower()]
        date_cols    = [p for p in picks
                        if "date" in p.data_type.lower()
                        or "date" in p.column.lower()
                        or "time" in p.column.lower()]
        groupby_cols = [p for p in picks
                        if "region" in p.column.lower()
                        or "state"  in p.column.lower()
                        or "dept"   in p.column.lower()
                        or "department" in p.column.lower()
                        or "provider" in p.column.lower()
                        or "plan"   in p.column.lower()
                        or "category" in p.column.lower()]

        from_clause = base
        for (a, b, col) in join_path:
            if "=" in col:
                from_clause += f"\nJOIN {b} ON {a}.{col.split('=')[0]} = {b}.{col.split('=')[1]}"
            else:
                from_clause += f"\nJOIN {b} ON {a}.{col} = {b}.{col}"

        where_parts = []
        if tw and date_cols:
            dc = date_cols[0]
            unit_map = {"day": "days", "week": "days", "month": "months",
                        "quarter": "months", "year": "years"}
            n = tw["n"]; unit = tw["unit"]
            if unit == "week":
                n = n * 7
            where_parts.append(f"{dc.table}.{dc.column} >= date('now', '-{n} {unit_map.get(unit, 'days')}')")

        where_sql = ("\nWHERE " + " AND ".join(where_parts)) if where_parts else ""

        if intent == "count":
            sql = (f"SELECT COUNT(*) AS cnt FROM {from_clause}{where_sql}")
            conf = min(0.8, 0.45 + score_mean * 0.5)
            return sql, conf
        if intent == "sum" and numeric_cols:
            nc = numeric_cols[0]
            sql = (f"SELECT SUM({nc.table}.{nc.column}) AS total_{nc.column.lower()} "
                   f"FROM {from_clause}{where_sql}")
            return sql, min(0.8, 0.45 + score_mean * 0.5)
        if intent == "avg" and numeric_cols:
            nc = numeric_cols[0]
            sql = (f"SELECT AVG({nc.table}.{nc.column}) AS avg_{nc.column.lower()} "
                   f"FROM {from_clause}{where_sql}")
            return sql, min(0.8, 0.45 + score_mean * 0.5)
        if intent == "topN" and numeric_cols:
            nc = numeric_cols[0]
            grp = groupby_cols[0] if groupby_cols else None
            m = re.search(r"top\s+(\d+)", (question or "").lower())
            n_top = int(m.group(1)) if m else 10
            if grp:
                sql = (f"SELECT {grp.table}.{grp.column}, "
                       f"SUM({nc.table}.{nc.column}) AS total "
                       f"FROM {from_clause}{where_sql} "
                       f"GROUP BY {grp.table}.{grp.column} "
                       f"ORDER BY total DESC LIMIT {n_top}")
            else:
                sql = (f"SELECT * FROM {from_clause}{where_sql} "
                       f"ORDER BY {nc.table}.{nc.column} DESC LIMIT {n_top}")
            return sql, min(0.75, 0.4 + score_mean * 0.5)
        if intent == "trend" and numeric_cols and date_cols:
            nc, dc = numeric_cols[0], date_cols[0]
            sql = (f"SELECT strftime('%Y-%m', {dc.table}.{dc.column}) AS period, "
                   f"SUM({nc.table}.{nc.column}) AS total "
                   f"FROM {from_clause}{where_sql} "
                   f"GROUP BY period ORDER BY period")
            return sql, min(0.75, 0.4 + score_mean * 0.5)

        cols_sql = ", ".join(
            f"{p.table}.{p.column}"
            for p in picks[: min(6, len(picks))]
        )
        sql = (f"SELECT {cols_sql} FROM {from_clause}{where_sql} LIMIT 100")
        return sql, min(0.55, 0.3 + score_mean * 0.4)

    def _build_prompt_context(self, question: str, intent: str,
                              tw: Optional[Dict[str, Any]],
                              tables: List[str], picks: List[ColumnPick],
                              join_path: List[Tuple[str, str, str]]) -> str:
        lines: List[str] = []
        lines.append("# Grounded schema context")
        lines.append(f"Question: {question}")
        lines.append(f"Detected intent: {intent}")
        if tw:
            lines.append(f"Time window: last {tw['n']} {tw['unit']}(s)")
        lines.append("")
        lines.append("## Relevant tables")
        for t in tables:
            lines.append(f"- {t}")
        lines.append("")
        lines.append("## Relevant columns (ranked)")
        for p in picks:
            extras = []
            if p.semantic_type:
                extras.append(f"type={p.semantic_type}")
            elif p.data_type:
                extras.append(f"type={p.data_type}")
            if p.top_values:
                extras.append(f"examples={p.top_values[:3]}")
            extra_str = "  (" + "; ".join(extras) + ")" if extras else ""
            lines.append(f"- {p.table}.{p.column}  [score {p.score:.3f}]{extra_str}")
        if join_path:
            lines.append("")
            lines.append("## Join path")
            for (a, b, col) in join_path:
                lines.append(f"- {a} JOIN {b} ON {col}")
        return "\n".join(lines)

    def _empty_bundle(self, question: str, reason: str) -> Dict[str, Any]:
        return {
            "question": question,
            "intent": "select",
            "time_window": None,
            "selected_tables": [],
            "selected_columns": [],
            "join_path": [],
            "rerank_backend": None,
            "candidate_sql": None,
            "sql_confidence": 0.0,
            "prompt_context": f"# Grounded schema context\n\n(unavailable: {reason})",
            "encoder": self._encoder_name,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "encoder": self._encoder_name or "uninitialized",
            "columns_indexed": len(self._cols),
            "tables_indexed": len(getattr(self.schema, "tables", {}) or {}),
            "built_at": self._built_at,
            "top_cols": self.top_cols,
            "top_tables": self.top_tables,
            "use_rerank": self.use_rerank,
        }


_GROUNDERS: "Dict[int, SchemaGrounder]" = {}
_LOCK = threading.Lock()


def get_grounder(schema_registry) -> SchemaGrounder:
    key = id(schema_registry)
    with _LOCK:
        g = _GROUNDERS.get(key)
        if g is None:
            g = SchemaGrounder(schema_registry)
            g.rebuild()
            _GROUNDERS[key] = g
    return g


def reset_grounders() -> None:
    with _LOCK:
        _GROUNDERS.clear()


__all__ = ["ColumnPick", "SchemaGrounder", "get_grounder", "reset_grounders"]
