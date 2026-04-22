from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA cache_size = -524288",
    "PRAGMA mmap_size = 34359738368",
    "PRAGMA page_size = 8192",
    "PRAGMA foreign_keys = ON",
    "PRAGMA wal_autocheckpoint = 2000",
    "PRAGMA threads = 4",
]


_INDEX_SPECS: List[Dict] = [
    {'table': ['members', 'member'], 'cols': [['member_id', 'mbr_id', 'id']]},
    {'table': ['members', 'member'], 'cols': [['region']]},
    {'table': ['members', 'member'], 'cols': [['plan_type', 'lob']]},
    {'table': ['members', 'member'], 'cols': [['risk_score']]},

    {'table': ['encounters', 'visits'], 'cols': [['member_id', 'MEMBER_ID']]},
    {'table': ['encounters', 'visits'], 'cols': [['admit_date', 'encounter_date',
                                                   'service_date', 'visit_date']]},
    {'table': ['encounters', 'visits'], 'cols': [['discharge_date']]},
    {'table': ['encounters', 'visits'], 'cols': [['visit_type', 'encounter_type']]},
    {'table': ['encounters', 'visits'], 'cols': [['primary_dx', 'dx', 'diagnosis']]},
    {'table': ['encounters', 'visits'], 'cols': [['provider_id', 'rendering_npi', 'npi']]},

    {'table': ['encounters', 'visits'],
     'cols': [['member_id', 'MEMBER_ID'], ['admit_date', 'encounter_date',
                                             'service_date', 'visit_date']]},

    {'table': ['claims', 'claim'], 'cols': [['member_id']]},
    {'table': ['claims', 'claim'], 'cols': [['claim_status', 'status']]},
    {'table': ['claims', 'claim'], 'cols': [['service_date', 'claim_date',
                                               'adjudicated_date']]},
    {'table': ['claims', 'claim'], 'cols': [['submitted_date']]},
    {'table': ['claims', 'claim'], 'cols': [['rendering_npi', 'provider_id', 'npi']]},
    {'table': ['claims', 'claim'], 'cols': [['region']]},

    {'table': ['claims', 'claim'],
     'cols': [['member_id'], ['service_date', 'claim_date',
                                'adjudicated_date', 'submitted_date']]},

    {'table': ['providers', 'provider'], 'cols': [['npi', 'provider_id']]},
    {'table': ['providers', 'provider'], 'cols': [['specialty']]},
]


def _tables(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return {r[0].lower(): r[0] for r in rows}


def _cols(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1].lower(): r[1] for r in rows}
    except sqlite3.Error:
        return {}


def _first_match(lookup: Dict[str, str], aliases: List[str]) -> Optional[str]:
    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]
    return None


def _apply_pragmas(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for pragma in _PRAGMAS:
        try:
            row = conn.execute(pragma).fetchone()
            val = row[0] if row else 'ok'
            results.append((pragma, str(val)))
        except sqlite3.Error as e:
            results.append((pragma, f'skip: {e}'))
    return results


def _resolve_index(conn: sqlite3.Connection, spec: Dict) -> Optional[Dict]:
    t_map = _tables(conn)
    table = _first_match(t_map, spec['table'])
    if not table:
        return None
    c_map = _cols(conn, table)
    resolved_cols: List[str] = []
    for alias_list in spec['cols']:
        c = _first_match(c_map, alias_list)
        if not c:
            return None
        resolved_cols.append(c)
    idx_name = "ix_gpdm_" + table.lower() + "_" + "_".join(
        c.lower() for c in resolved_cols
    )
    return {'name': idx_name, 'table': table, 'cols': resolved_cols}


def _create_index(conn: sqlite3.Connection, idx: Dict) -> str:
    cols_sql = ", ".join(idx['cols'])
    sql = (f"CREATE INDEX IF NOT EXISTS {idx['name']} "
           f"ON {idx['table']} ({cols_sql})")
    try:
        conn.execute(sql)
        return 'ok'
    except sqlite3.Error as e:
        return f'error: {e}'


def optimize_db(db_path: str, *, analyze: bool = True,
                vacuum: bool = False) -> Dict:
    report: Dict = {
        'db_path': db_path,
        'pragmas': [],
        'indexes_created': [],
        'indexes_skipped': [],
        'analyze': None,
        'vacuum': None,
    }
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        report['error'] = f'open failed: {e}'
        return report

    try:
        report['pragmas'] = _apply_pragmas(conn)

        conn.execute("BEGIN")
        try:
            for spec in _INDEX_SPECS:
                resolved = _resolve_index(conn, spec)
                if not resolved:
                    report['indexes_skipped'].append(
                        {'aliases': spec, 'reason': 'table or col missing'}
                    )
                    continue
                status = _create_index(conn, resolved)
                report['indexes_created'].append({**resolved, 'status': status})
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        if analyze:
            try:
                conn.execute("ANALYZE")
                report['analyze'] = 'ok'
            except sqlite3.Error as e:
                report['analyze'] = f'error: {e}'

        if vacuum:
            try:
                conn.execute("VACUUM")
                report['vacuum'] = 'ok'
            except sqlite3.Error as e:
                report['vacuum'] = f'error: {e}'
    finally:
        conn.close()

    _log.info("perf_indexes: %d indexes created/verified, %d skipped",
              len(report['indexes_created']), len(report['indexes_skipped']))
    return report


def apply_runtime_pragmas(conn: sqlite3.Connection) -> None:
    for p in ("PRAGMA temp_store = MEMORY",
              "PRAGMA cache_size = -262144",
              "PRAGMA mmap_size = 17179869184"):
        try:
            conn.execute(p)
        except sqlite3.Error:
            pass
