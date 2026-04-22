from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

_log = logging.getLogger("gpdm.sneakpeek")


def _safe_count(db_path: str, sql: str) -> Optional[int]:
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute(sql).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()
    except Exception:
        return None


def _data_dir() -> str:
    return os.path.join(
        (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
    )


def _learning_loop_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        from learning_loop import get_accuracy_stats
        stats = get_accuracy_stats() or {}
        acc = stats.get("accuracy")
        n = stats.get("sample_size") or stats.get("count")
        if acc is not None and n:
            out.append({
                "title": "Answer accuracy",
                "body": f"{round(float(acc) * 100)}% on the last {int(n)} questions you and your team rated.",
                "icon": "",
            })
    except Exception as e:
        _log.debug("learning_loop unavailable: %s", e)
    return out


def _query_tracker_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    db = os.path.join(_data_dir(), "query_tracker.db")
    if not os.path.exists(db):
        return out
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        try:
            tbls = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
            cand = [t for t in ("query_log", "queries", "query_tracker", "questions")
                    if t in tbls]
            for tbl in cand[:1]:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl}").fetchone()
                if row and row[0]:
                    out.append({
                        "title": "Questions seen",
                        "body": f"{int(row[0]):,} total — the model picks the most frequent ones for self-review.",
                        "icon": "",
                    })
                cols = [c[1] for c in conn.execute(
                    f"PRAGMA table_info({tbl})")]
                for col in ("intent", "concept", "topic", "tables_used"):
                    if col in cols:
                        rows = conn.execute(
                            f"SELECT {col}, COUNT(*) c FROM {tbl} "
                            f"WHERE {col} IS NOT NULL AND {col} != '' "
                            f"GROUP BY {col} ORDER BY c DESC LIMIT 1"
                        ).fetchone()
                        if rows and rows[0]:
                            out.append({
                                "title": "Hot topic",
                                "body": f"Most-asked area this period: {rows[0]} ({int(rows[1])} questions).",
                                "icon": "",
                            })
                        break
        finally:
            conn.close()
    except Exception as e:
        _log.debug("query_tracker scan failed: %s", e)
    return out


def _drift_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        import drift_monitor as dm
        status = dm.get_drift_status(
            os.path.join(_data_dir(), "healthcare_demo.db")
        ) or {}
        score = status.get("overall_drift") or status.get("drift_score")
        flagged = status.get("flagged_columns") or []
        if score is not None:
            level = "low" if score < 0.1 else ("moderate" if score < 0.3 else "high")
            out.append({
                "title": "Data drift",
                "body": f"Drift level looks {level} — distributions match recent history within {round((1 - float(score)) * 100)}%.",
                "icon": "",
            })
        if flagged:
            out.append({
                "title": "Watching",
                "body": f"{len(flagged)} field(s) shifted recently; the model is reweighting them.",
                "icon": "",
            })
    except Exception as e:
        _log.debug("drift_monitor unavailable: %s", e)
    return out


def _continual_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        import continual_learning as cl
        st = cl.get_continual_status(
            os.path.join(_data_dir(), "healthcare_demo.db")
        ) or {}
        last = st.get("last_update") or st.get("last_run")
        cycles = st.get("cycles") or st.get("updates")
        if cycles:
            out.append({
                "title": "Self-tuning",
                "body": f"{int(cycles)} background learning cycles completed.",
                "icon": "",
            })
        if last:
            out.append({
                "title": "Last refresh",
                "body": f"Knowledge base last refreshed {last}.",
                "icon": "",
            })
    except Exception as e:
        _log.debug("continual_learning unavailable: %s", e)
    return out


def _catalog_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        from catalog_registry import CatalogRegistry
        reg = CatalogRegistry.get_instance()
        if not reg._initialized:
            return out
        n_tables = sum(len(env.tables) for env in reg.environments.values()
                       if hasattr(env, "tables"))
        if n_tables:
            out.append({
                "title": "Tables understood",
                "body": f"{n_tables} table(s) auto-classified into business concepts.",
                "icon": "",
            })
    except Exception as e:
        _log.debug("catalog scan failed: %s", e)
    return out


def _phi_bullets() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    db = os.path.join(_data_dir(), "phi_lineage.db")
    if not os.path.exists(db):
        return out
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT phi_type) FROM phi_lineage"
            ).fetchone()
            if row and row[0]:
                out.append({
                    "title": "Privacy guardrail",
                    "body": f"{int(row[0])} sensitive field(s) across {int(row[1])} HIPAA category type(s) are masked before any answer is shown.",
                    "icon": "",
                })
        finally:
            conn.close()
    except Exception as e:
        _log.debug("phi sneakpeek failed: %s", e)
    return out


def get_sneakpeek(max_items: int = 6) -> Dict[str, Any]:
    bullets: List[Dict[str, str]] = []
    for fn in (_learning_loop_bullets, _query_tracker_bullets, _drift_bullets,
               _continual_bullets, _catalog_bullets, _phi_bullets):
        try:
            bullets.extend(fn())
        except Exception as e:
            _log.debug("sneakpeek source %s failed: %s", fn.__name__, e)
    if not bullets:
        bullets = [{
            "title": "Warming up",
            "body": "The system is gathering its first signals — check back in a few minutes.",
            "icon": "",
        }]
    return {
        "ok": True,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "items": bullets[:max_items],
    }
