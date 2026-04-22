from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("LLAMA_CPP_DISABLE_DOWNLOAD", "1")

DEFAULT_MODEL_PATH = os.environ.get(
    "CLINICAL_LLM_MODEL",
    "",
)
CANDIDATE_FILENAMES = [
    "openbiollm-8b-q4_k_m.gguf",
    "meditron-7b-q4_k_m.gguf",
    "llama-3.1-8b-instruct-q4_k_m.gguf",
    "phi-3-mini-medical-q4_k_m.gguf",
    "llama-3.2-3b-instruct-q4_k_m.gguf",
]

DEFAULT_CTX = int(os.environ.get("CLINICAL_LLM_CTX", "8192"))
DEFAULT_THREADS = int(os.environ.get("CLINICAL_LLM_THREADS", str(max(4, os.cpu_count() or 4))))
DEFAULT_MAX_TOKENS = int(os.environ.get("CLINICAL_LLM_MAX_TOKENS", "512"))
DEFAULT_GPU_LAYERS = 0


def _repo_root() -> str:
    return (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _models_dir() -> str:
    d = os.path.join(_repo_root(), "data", "models", "llm")
    os.makedirs(d, exist_ok=True)
    return d


def _default_db() -> str:
    return os.path.join(_repo_root(), "data", "healthcare_demo.db")


_LLAMA = None
_LLM = None
_MODEL_PATH: Optional[str] = None


def _try_import_llama():
    global _LLAMA
    if _LLAMA is not None:
        return _LLAMA
    try:
        from llama_cpp import Llama
        _LLAMA = Llama
    except Exception as e:
        logger.warning("llama-cpp-python unavailable: %s", e)
        _LLAMA = False
    return _LLAMA


def _find_model_file() -> Optional[str]:
    if DEFAULT_MODEL_PATH and os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    for fn in CANDIDATE_FILENAMES:
        p = os.path.join(_models_dir(), fn)
        if os.path.exists(p):
            return p
    try:
        for fn in os.listdir(_models_dir()):
            if fn.lower().endswith(".gguf"):
                return os.path.join(_models_dir(), fn)
    except Exception:
        pass
    return None


def load_llm(force: bool = False) -> Dict[str, Any]:
    global _LLM, _MODEL_PATH
    if _LLM is not None and not force:
        return {"loaded": True, "model_path": _MODEL_PATH}
    Llama = _try_import_llama()
    if not Llama:
        return {"loaded": False, "error": "llama_cpp_not_installed"}
    path = _find_model_file()
    if not path:
        return {
            "loaded": False,
            "error": "no_gguf_model_found",
            "search_dir": _models_dir(),
            "sideload_instructions": (
                f"Place any medical .gguf (e.g. OpenBioLLM-8B Q4_K_M) into "
                f"{_models_dir()}. Or set env CLINICAL_LLM_MODEL=/abs/path.gguf"
            ),
        }
    try:
        logger.info("Loading clinical LLM: %s (ctx=%d threads=%d gpu_layers=%d)",
                    path, DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_GPU_LAYERS)
        _LLM = Llama(
            model_path=path,
            n_ctx=DEFAULT_CTX,
            n_threads=DEFAULT_THREADS,
            n_gpu_layers=DEFAULT_GPU_LAYERS,
            verbose=False,
            chat_format="llama-3" if "llama-3" in path.lower() or "meditron" in path.lower()
                       else "chatml",
        )
        _MODEL_PATH = path
        return {"loaded": True, "model_path": path,
                "ctx": DEFAULT_CTX, "threads": DEFAULT_THREADS}
    except Exception as e:
        logger.exception("LLM load failed")
        return {"loaded": False, "error": str(e)}


SQL_MEMORY_TABLE = "gpdm_sql_memory"


def _init_sql_memory(db_path: str) -> None:
    try:
        with sqlite3.connect(db_path, timeout=10) as c:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {SQL_MEMORY_TABLE} (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL NOT NULL,
                    question  TEXT NOT NULL,
                    sql       TEXT NOT NULL,
                    row_count INTEGER,
                    success   INTEGER NOT NULL,
                    ms        REAL
                )
            """)
            c.execute(f"CREATE INDEX IF NOT EXISTS ix_sqlmem_ts ON {SQL_MEMORY_TABLE}(ts)")
            c.commit()
    except Exception:
        pass


def _remember_query(db_path: str, q: str, sql: str,
                    row_count: int, success: bool, ms: float) -> None:
    _init_sql_memory(db_path)
    try:
        with sqlite3.connect(db_path, timeout=10) as c:
            c.execute(
                f"INSERT INTO {SQL_MEMORY_TABLE} (ts, question, sql, row_count, success, ms)"
                f" VALUES (?,?,?,?,?,?)",
                (time.time(), q, sql, row_count, 1 if success else 0, ms),
            )
            c.commit()
    except Exception:
        pass


def _recent_successful_pairs(db_path: str, limit: int = 200) -> List[Tuple[str, str]]:
    _init_sql_memory(db_path)
    try:
        with sqlite3.connect(db_path, timeout=10) as c:
            return [(r[0], r[1]) for r in c.execute(
                f"SELECT question, sql FROM {SQL_MEMORY_TABLE} "
                f"WHERE success=1 ORDER BY ts DESC LIMIT ?", (limit,))]
    except Exception:
        return []


def _schema_digest(db_path: str, question: str, k: int = 8) -> str:
    try:
        import neural_engine
        if neural_engine._SCHEMA is None:
            neural_engine.init_neural_stack(db_path, warmup=False)
        hits = neural_engine.semantic_schema_lookup(question, k=k)
    except Exception:
        hits = []

    tables = set()
    for h in hits:
        tables.add(h.get("table"))
    if not tables:
        with sqlite3.connect(db_path) as c:
            tables = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'gpdm_%' AND name NOT LIKE 'sqlite_%'")}

    lines = []
    with sqlite3.connect(db_path) as c:
        for t in sorted(tables):
            if not t:
                continue
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})")]
            if cols:
                lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)


def _clinical_hints(question: str, k: int = 5) -> str:
    try:
        import neural_engine
        if neural_engine._VOCAB is None:
            neural_engine.init_neural_stack(warmup=False)
        hits = neural_engine.resolve_clinical_code(question, k=k)
    except Exception:
        hits = []
    if not hits:
        return ""
    return "\n".join(f"- {h['kind']} {h['code']}: {h['description']}" for h in hits)


def _example_pairs(db_path: str, k: int = 4) -> str:
    pairs = _recent_successful_pairs(db_path, limit=100)[:k]
    if not pairs:
        return ""
    out = []
    for q, sql in pairs:
        out.append(f"-- Q: {q}\n{sql.strip()}\n")
    return "\n".join(out)


FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|REINDEX|REPLACE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


def _extract_sql(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        m = re.search(r"(SELECT\b.+?;)", text, re.DOTALL | re.IGNORECASE)
        candidate = m.group(1).strip() if m else text.strip()
    parts = [p.strip() for p in candidate.split(";") if p.strip()]
    if not parts:
        return None
    return parts[0] + ";"


def _validate_sql(sql: str, db_path: str) -> Tuple[bool, str]:
    if not sql:
        return False, "empty"
    up = sql.upper().strip()
    if not up.startswith("SELECT") and not up.startswith("WITH"):
        return False, "not_select"
    if FORBIDDEN.search(sql):
        return False, "forbidden_keyword"
    try:
        with sqlite3.connect(db_path) as c:
            tables = {r[0].lower() for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    except Exception:
        return True, "ok_no_check"
    refs = set(re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE))
    for r in refs:
        if r.lower() not in tables:
            return False, f"unknown_table:{r}"
    return True, "ok"


SYSTEM_PROMPT = """You are GPDM-Clinical, a careful analytics assistant for a
GPDM-style healthcare claims database. You translate natural-language clinical
questions into a single, valid, read-only SQLite SELECT statement.

Rules:
 1. Return ONLY a SQL block. No commentary outside the SQL.
 2. SELECT (or WITH ... SELECT) only. Never write, modify, or delete.
 3. Use the exact table/column names listed in SCHEMA.
 4. Prefer explicit JOINs on MEMBER_ID or ENCOUNTER_ID.
 5. For date filters use SERVICE_DATE / ADMIT_DATE / DISCHARGE_DATE as appropriate.
 6. When the user implies "high-risk", "frequent flyer", or "outlier", compute
    it with percentiles/aggregates, never hardcoded numbers.
 7. Every claim/encounter total must be COALESCE(..., 0) to avoid NULL drift.
 8. End the statement with a semicolon.
"""


def _build_prompt(question: str, db_path: str) -> str:
    schema = _schema_digest(db_path, question, k=10)
    hints = _clinical_hints(question, k=5)
    examples = _example_pairs(db_path, k=4)

    p = [SYSTEM_PROMPT, "\n# SCHEMA\n" + schema]
    if hints:
        p.append("\n# CLINICAL CODE HINTS\n" + hints)
    if examples:
        p.append("\n# EXAMPLES (prior successful translations)\n" + examples)
    p.append(f"\n# USER QUESTION\n{question}\n\n# SQL\n```sql\n")
    return "\n".join(p)


def _generate(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = 0.1, stop: Optional[List[str]] = None) -> str:
    info = load_llm()
    if not info.get("loaded"):
        raise RuntimeError(info.get("error") or "llm_unloaded")
    stop = stop or ["```", "\n\n# ", "</s>", "<|eot_id|>"]
    out = _LLM(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop,
               top_p=0.9, repeat_penalty=1.05)
    return out["choices"][0]["text"]


@dataclass
class AskResult:
    question: str
    sql: Optional[str]
    rows: List[Dict[str, Any]]
    columns: List[str]
    answer: Optional[str]
    used_fallback: bool
    error: Optional[str]
    elapsed_ms: float
    validation: str
    model: Optional[str]


def ask(question: str, db_path: Optional[str] = None,
        execute: bool = True, summarise: bool = True,
        max_rows: int = 200) -> Dict[str, Any]:
    t0 = time.time()
    db_path = db_path or _default_db()
    info = load_llm()
    if not info.get("loaded"):
        return _fallback(question, db_path, reason=info.get("error"))

    prompt = _build_prompt(question, db_path)
    try:
        raw = _generate(prompt, max_tokens=400, temperature=0.05)
    except Exception as e:
        return _fallback(question, db_path, reason=f"generation_failed:{e}")

    sql = _extract_sql(raw)
    ok, reason = _validate_sql(sql or "", db_path)
    if not ok:
        logger.info("LLM SQL invalid (%s): %s", reason, sql)
        fb = _fallback(question, db_path, reason=f"validation:{reason}")
        fb["llm_sql_attempt"] = sql
        return fb

    rows, cols, err = [], [], None
    if execute:
        try:
            with sqlite3.connect(db_path, timeout=15) as c:
                cur = c.execute(sql if sql.rstrip(";").strip() else "SELECT 1")
                cols = [d[0] for d in (cur.description or [])]
                rows = [dict(zip(cols, r)) for r in cur.fetchmany(max_rows)]
        except Exception as e:
            err = str(e)

    answer = None
    if summarise and rows and not err:
        try:
            answer = _summarise(question, sql, rows, cols)
        except Exception:
            pass

    elapsed = (time.time() - t0) * 1000.0
    _remember_query(db_path, question, sql or "", len(rows), err is None, elapsed)

    return {
        "question": question,
        "sql": sql,
        "columns": cols,
        "rows": rows,
        "row_count": len(rows),
        "answer": answer,
        "used_fallback": False,
        "error": err,
        "elapsed_ms": round(elapsed, 1),
        "validation": reason,
        "model": _MODEL_PATH,
    }


def _summarise(question: str, sql: str, rows: List[Dict[str, Any]],
               cols: List[str]) -> Optional[str]:
    sample = rows[:20]
    prompt = (
        "You are a clinical analytics assistant. Given the user question, the "
        "SQL we ran, and the first rows of the result, write a concise, "
        "factual 2-3 sentence answer in plain English. Cite numbers from the "
        "rows. Do not speculate beyond the data.\n\n"
        f"QUESTION: {question}\n\nSQL:\n{sql}\n\nROWS (first {len(sample)} of "
        f"{len(rows)}):\n{json.dumps(sample, default=str)[:4000]}\n\nANSWER: "
    )
    try:
        out = _generate(prompt, max_tokens=220, temperature=0.2,
                        stop=["\n\n", "QUESTION:", "SQL:"])
        return out.strip()
    except Exception:
        return None


def _fallback(question: str, db_path: str, reason: str = "") -> Dict[str, Any]:
    try:
        import semantic_sql_engine as sse
        for fn_name in ("answer_question", "run", "translate_and_run", "process"):
            fn = getattr(sse, fn_name, None)
            if fn:
                try:
                    r = fn(question, db_path=db_path) if "db_path" in fn.__code__.co_varnames \
                        else fn(question)
                    return {
                        "question": question,
                        "sql": r.get("sql") if isinstance(r, dict) else None,
                        "rows": r.get("rows", []) if isinstance(r, dict) else [],
                        "columns": r.get("columns", []) if isinstance(r, dict) else [],
                        "row_count": len(r.get("rows", [])) if isinstance(r, dict) else 0,
                        "answer": r.get("answer") if isinstance(r, dict) else None,
                        "used_fallback": True,
                        "fallback_reason": reason,
                        "error": None,
                        "model": "rule_based_fallback",
                    }
                except Exception:
                    continue
    except Exception:
        pass
    return {
        "question": question,
        "sql": None, "rows": [], "columns": [], "row_count": 0,
        "answer": None, "used_fallback": True,
        "fallback_reason": reason, "error": "no_llm_and_no_fallback",
        "model": None,
    }


def status() -> Dict[str, Any]:
    Llama = _try_import_llama()
    found = _find_model_file()
    return {
        "llama_cpp_installed": bool(Llama),
        "model_found": bool(found),
        "model_path": found or None,
        "loaded": _LLM is not None,
        "ctx": DEFAULT_CTX,
        "threads": DEFAULT_THREADS,
        "gpu_layers": DEFAULT_GPU_LAYERS,
        "search_dir": _models_dir(),
        "candidate_names": CANDIDATE_FILENAMES,
    }
