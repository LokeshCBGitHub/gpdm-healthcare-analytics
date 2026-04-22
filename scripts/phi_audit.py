from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

_log = logging.getLogger(__name__)

_DEFAULT_AUDIT_PATH = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "phi_audit.db",
)


class PhiAuditLog:

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_AUDIT_PATH
        self._lock = threading.Lock()
        self._salt = os.environ.get("GPDM_PHI_AUDIT_SALT", "gpdm-phi-audit-v1").encode()
        self._init()

    def _init(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phi_audit (
                    audit_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id               TEXT,
                    role                  TEXT,
                    question_sha256       TEXT,
                    question_masked       TEXT,
                    resolved_sql          TEXT,
                    tables_touched        TEXT,     -- JSON array
                    phi_cols_touched      TEXT,     -- JSON array
                    phi_tokens_in_result  INTEGER,
                    rows_returned         INTEGER,
                    elapsed_ms            INTEGER,
                    source_ip             TEXT,
                    outcome               TEXT      -- ok | blocked | error
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_phi_audit_user_ts "
                "ON phi_audit(user_id, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_phi_audit_outcome "
                "ON phi_audit(outcome)"
            )
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self._db_path, 0o600)
        except OSError:
            pass

    def _hash(self, text: str) -> str:
        return hashlib.sha256(self._salt + (text or "").encode("utf-8")).hexdigest()

    def record(self,
               user_id: str = "",
               role: str = "",
               question_raw: str = "",
               question_masked: str = "",
               resolved_sql: str = "",
               tables_touched: Iterable[str] = (),
               phi_cols_touched: Iterable[str] = (),
               phi_tokens_in_result: int = 0,
               rows_returned: int = 0,
               elapsed_ms: int = 0,
               source_ip: str = "",
               outcome: str = "ok") -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "INSERT INTO phi_audit "
                "(user_id, role, question_sha256, question_masked, resolved_sql, "
                " tables_touched, phi_cols_touched, phi_tokens_in_result, "
                " rows_returned, elapsed_ms, source_ip, outcome) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id or "",
                    role or "",
                    self._hash(question_raw),
                    question_masked or "",
                    resolved_sql or "",
                    json.dumps(list(tables_touched)),
                    json.dumps(list(phi_cols_touched)),
                    int(phi_tokens_in_result),
                    int(rows_returned),
                    int(elapsed_ms),
                    source_ip or "",
                    outcome,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def recent(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM phi_audit"
        params: List[Any] = []
        if user_id:
            q += " WHERE user_id = ?"
            params.append(user_id)
        q += " ORDER BY audit_id DESC LIMIT ?"
        params.append(limit)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(q, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def vacuum_older_than(self, days: int) -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "DELETE FROM phi_audit WHERE ts < datetime('now', ?)",
                (f"-{int(days)} days",),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


_instance: Optional[PhiAuditLog] = None


def get_audit_log() -> PhiAuditLog:
    global _instance
    if _instance is None:
        _instance = PhiAuditLog()
    return _instance
