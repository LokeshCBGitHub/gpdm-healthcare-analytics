from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


CUBE_TABLE = "gpdm_member_month_fact"
CUBE_META  = "gpdm_cube_meta"


def _tables(conn) -> Dict[str, str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return {r[0].lower(): r[0] for r in rows}


def _cols(conn, table: str) -> Dict[str, str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1].lower(): r[1] for r in rows}
    except sqlite3.Error:
        return {}


def _first(m: Dict[str, str], *aliases: str) -> Optional[str]:
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    return None


class _Schema:
    def __init__(self, conn):
        t = _tables(conn)
        self.members = t.get('members') or t.get('member')
        self.enc     = t.get('encounters') or t.get('encounter') or t.get('visits')
        self.claims  = t.get('claims') or t.get('claim')

        ec = _cols(conn, self.enc) if self.enc else {}
        mc = _cols(conn, self.members) if self.members else {}
        cc = _cols(conn, self.claims) if self.claims else {}

        self.enc_mid   = _first(ec, 'member_id')
        self.enc_date  = _first(ec, 'admit_date', 'encounter_date',
                                 'service_date', 'visit_date', 'date')
        self.enc_disc  = _first(ec, 'discharge_date', 'disch_date')
        self.enc_type  = _first(ec, 'visit_type', 'encounter_type', 'type')
        self.enc_los   = _first(ec, 'length_of_stay', 'los')
        self.enc_paid  = _first(ec, 'total_paid', 'paid_amount', 'cost')
        self.enc_dx    = _first(ec, 'primary_dx', 'dx', 'diagnosis')

        self.cl_mid    = _first(cc, 'member_id')
        self.cl_date   = _first(cc, 'service_date', 'claim_date',
                                 'adjudicated_date', 'submitted_date')
        self.cl_status = _first(cc, 'claim_status', 'status')
        self.cl_paid   = _first(cc, 'paid_amount', 'paid')
        self.cl_billed = _first(cc, 'billed_amount', 'billed')

        self.mem_id    = _first(mc, 'member_id', 'id')
        self.mem_reg   = _first(mc, 'region', 'region')
        self.mem_lob   = _first(mc, 'plan_type', 'lob')


def _ensure_cube(conn) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {CUBE_TABLE} (
            member_id     TEXT    NOT NULL,
            year_month    TEXT    NOT NULL,
            region        TEXT,
            lob           TEXT,
            admits        INTEGER DEFAULT 0,
            er_visits     INTEGER DEFAULT 0,
            preventive    INTEGER DEFAULT 0,
            total_visits  INTEGER DEFAULT 0,
            los_days      REAL    DEFAULT 0,
            paid_usd      REAL    DEFAULT 0,
            billed_usd    REAL    DEFAULT 0,
            denials       INTEGER DEFAULT 0,
            PRIMARY KEY (member_id, year_month)
        )
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS ix_{CUBE_TABLE}_ym
            ON {CUBE_TABLE}(year_month)
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS ix_{CUBE_TABLE}_reg_ym
            ON {CUBE_TABLE}(region, year_month)
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {CUBE_META} (
            k TEXT PRIMARY KEY,
            v TEXT
        )
    """)


def _get_meta(conn, k: str) -> Optional[str]:
    r = conn.execute(
        f"SELECT v FROM {CUBE_META} WHERE k = ?", (k,)
    ).fetchone()
    return r[0] if r else None


def _set_meta(conn, k: str, v: str) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {CUBE_META}(k, v) VALUES (?, ?)", (k, v)
    )


def _max_ym_in_cube(conn) -> Optional[str]:
    r = conn.execute(
        f"SELECT MAX(year_month) FROM {CUBE_TABLE}"
    ).fetchone()
    return r[0] if r and r[0] else None


def refresh_cube(db_path: str, *, full: bool = False,
                  ym_floor: Optional[str] = None) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        _ensure_cube(conn)
        s = _Schema(conn)
        if not (s.enc and s.enc_mid and s.enc_date):
            return {'error': 'encounters/date column missing',
                     'rows_written': 0}

        if full:
            conn.execute(f"DELETE FROM {CUBE_TABLE}")
            floor = '1900-01'
        else:
            existing = _max_ym_in_cube(conn)
            if ym_floor:
                floor = ym_floor
            elif existing:
                floor = existing
            else:
                floor = '1900-01'
            conn.execute(
                f"DELETE FROM {CUBE_TABLE} WHERE year_month >= ?", (floor,)
            )

        type_col = s.enc_type or "''"
        los_expr = f"CAST({s.enc_los} AS REAL)" if s.enc_los else "0"
        paid_expr = f"CAST({s.enc_paid} AS REAL)" if s.enc_paid else "0"
        disch_expr = s.enc_disc if s.enc_disc else "NULL"

        enc_select = f"""
            SELECT
              e.{s.enc_mid}                                   AS member_id,
              strftime('%Y-%m', e.{s.enc_date})               AS year_month,
              SUM(CASE WHEN UPPER({type_col}) IN
                   ('INPATIENT','IP','ADMIT') OR {los_expr} > 0
                   THEN 1 ELSE 0 END)                         AS admits,
              SUM(CASE WHEN UPPER({type_col}) IN
                   ('EMERGENCY','ED','ER')
                   THEN 1 ELSE 0 END)                         AS er_visits,
              SUM(CASE WHEN UPPER({type_col}) IN
                   ('PREVENTIVE','WELLNESS','AWV')
                   THEN 1 ELSE 0 END)                         AS preventive,
              COUNT(*)                                         AS total_visits,
              COALESCE(SUM({los_expr}), 0)                     AS los_days,
              COALESCE(SUM({paid_expr}), 0)                    AS enc_paid
            FROM {s.enc} e
            WHERE e.{s.enc_date} IS NOT NULL
              AND strftime('%Y-%m', e.{s.enc_date}) >= ?
            GROUP BY member_id, year_month
        """

        enc_rows = conn.execute(enc_select, (floor,)).fetchall()
        fact: Dict = {}
        for r in enc_rows:
            mid, ym, admits, er, prev, total, los, enc_paid = r
            if not mid or not ym:
                continue
            fact[(mid, ym)] = {
                'admits': admits or 0,
                'er_visits': er or 0,
                'preventive': prev or 0,
                'total_visits': total or 0,
                'los_days': float(los or 0),
                'paid_usd': float(enc_paid or 0),
                'billed_usd': 0.0,
                'denials': 0,
            }

        if s.claims and s.cl_mid and s.cl_date:
            paid_col = f"CAST({s.cl_paid} AS REAL)" if s.cl_paid else "0"
            billed_col = f"CAST({s.cl_billed} AS REAL)" if s.cl_billed else "0"
            status_pred = (f"UPPER({s.cl_status}) IN ('DENIED','REJECTED')"
                            if s.cl_status else "0")
            claims_sql = f"""
                SELECT
                  c.{s.cl_mid}                                 AS member_id,
                  strftime('%Y-%m', c.{s.cl_date})             AS year_month,
                  COALESCE(SUM({paid_col}), 0)                 AS paid_usd,
                  COALESCE(SUM({billed_col}), 0)               AS billed_usd,
                  SUM(CASE WHEN {status_pred} THEN 1 ELSE 0 END) AS denials
                FROM {s.claims} c
                WHERE c.{s.cl_date} IS NOT NULL
                  AND strftime('%Y-%m', c.{s.cl_date}) >= ?
                GROUP BY member_id, year_month
            """
            for r in conn.execute(claims_sql, (floor,)).fetchall():
                mid, ym, paid, billed, denials = r
                if not mid or not ym:
                    continue
                key = (mid, ym)
                if key not in fact:
                    fact[key] = {
                        'admits': 0, 'er_visits': 0, 'preventive': 0,
                        'total_visits': 0, 'los_days': 0.0,
                        'paid_usd': 0.0, 'billed_usd': 0.0, 'denials': 0,
                    }
                fact[key]['paid_usd'] = float(paid or 0)
                fact[key]['billed_usd'] = float(billed or 0)
                fact[key]['denials'] = int(denials or 0)

        reg_map: Dict[str, tuple] = {}
        if s.members and s.mem_id:
            reg_col = s.mem_reg or "''"
            lob_col = s.mem_lob or "''"
            for mid, reg, lob in conn.execute(
                f"SELECT {s.mem_id}, {reg_col}, {lob_col} FROM {s.members}"
            ).fetchall():
                reg_map[mid] = (reg or '', lob or '')

        rows = []
        for (mid, ym), m in fact.items():
            reg, lob = reg_map.get(mid, ('', ''))
            rows.append((mid, ym, reg, lob,
                          m['admits'], m['er_visits'], m['preventive'],
                          m['total_visits'], m['los_days'],
                          m['paid_usd'], m['billed_usd'], m['denials']))
        if rows:
            conn.executemany(
                f"INSERT OR REPLACE INTO {CUBE_TABLE} "
                f"(member_id, year_month, region, lob, admits, er_visits, "
                f"preventive, total_visits, los_days, paid_usd, billed_usd, denials) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )

        import time as _t
        _set_meta(conn, 'last_refresh_epoch', str(_t.time()))
        _set_meta(conn, 'last_refresh_floor', floor)
        _set_meta(conn, 'row_count', str(len(rows)))
        conn.commit()

        return {
            'rows_written': len(rows),
            'floor_ym': floor,
            'full_rebuild': bool(full),
            'latest_ym_in_cube': _max_ym_in_cube(conn),
            'tables_used': {'members': s.members, 'encounters': s.enc,
                             'claims': s.claims},
        }
    finally:
        conn.close()


def get_cube_status(db_path: str) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(db_path)
        try:
            _ensure_cube(conn)
            row = conn.execute(
                f"SELECT COUNT(*), MIN(year_month), MAX(year_month) "
                f"FROM {CUBE_TABLE}"
            ).fetchone()
            return {
                'row_count': row[0],
                'min_ym': row[1],
                'max_ym': row[2],
                'last_refresh_epoch': _get_meta(conn, 'last_refresh_epoch'),
            }
        finally:
            conn.close()
    except sqlite3.Error as e:
        return {'error': str(e)}


def refresh_from_gateway(db_path: str, *, full: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import connection_config as _cc
        import data_gateway as _dg
        if _cc.is_backend_enabled('iics'):
            try:
                out['iics'] = _dg.run_taskflow('refresh_member_month_fact')
            except Exception as e:
                out['iics_error'] = str(e)
        else:
            out['iics'] = {'status': 'disabled'}
    except ImportError as e:
        out['gateway_unavailable'] = str(e)

    try:
        out['cube'] = refresh_cube(db_path, full=full)
    except Exception as e:
        out['cube_error'] = str(e)
    return out
