from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional

_DEFAULT_LINEAGE_PATH = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "phi_lineage.db",
)


class PhiLineage:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_LINEAGE_PATH
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phi_lineage (
                    lineage_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_env       TEXT,
                    source_catalog   TEXT,
                    source_schema    TEXT,
                    source_table     TEXT,
                    source_col       TEXT,
                    masked_table     TEXT,
                    masked_col       TEXT,
                    phi_type         TEXT,
                    masking_fn       TEXT,
                    masking_version  TEXT,
                    sample_preview   TEXT,
                    first_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_phi_lineage_key "
                "ON phi_lineage(source_table, source_col, masked_table, masked_col)"
            )
            conn.commit()
        finally:
            conn.close()

    def record(self,
               source_env: str = "",
               source_catalog: str = "",
               source_schema: str = "",
               source_table: str = "",
               source_col: str = "",
               masked_table: str = "",
               masked_col: str = "",
               phi_type: str = "",
               masking_fn: str = "hmac_sha256",
               masking_version: str = "v1",
               sample_preview: str = "") -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                INSERT INTO phi_lineage
                    (source_env, source_catalog, source_schema, source_table, source_col,
                     masked_table, masked_col, phi_type, masking_fn, masking_version,
                     sample_preview)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_table, source_col, masked_table, masked_col)
                DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    phi_type = excluded.phi_type,
                    sample_preview = excluded.sample_preview
            """, (source_env, source_catalog, source_schema, source_table, source_col,
                  masked_table, masked_col, phi_type, masking_fn, masking_version,
                  sample_preview))
            conn.commit()
        finally:
            conn.close()

    def all(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM phi_lineage ORDER BY masked_table, masked_col"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def for_table(self, masked_table: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM phi_lineage WHERE masked_table = ? ORDER BY masked_col",
                (masked_table,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


_instance: Optional[PhiLineage] = None


def get_lineage() -> PhiLineage:
    global _instance
    if _instance is None:
        _instance = PhiLineage()
    return _instance
