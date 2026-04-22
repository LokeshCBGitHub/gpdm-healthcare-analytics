from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


_DEFAULT_DROP = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'cms_drop'
)


_SHAPES: List[Tuple[str, List[str]]] = [
    ('cms_hospital_readmission',
     ['facility id', 'measure name', 'excess readmission ratio']),
    ('cms_hospital_readmission',
     ['facility_id', 'measure_id', 'excess_readmission_ratio']),
    ('cms_hospital_readmission',
     ['provider id', 'measure id', 'score']),

    ('cms_physician_utilization',
     ['rndrng_npi', 'hcpcs_cd', 'tot_benes']),
    ('cms_physician_utilization',
     ['npi', 'hcpcs_code', 'bene_unique_cnt']),

    ('cms_partd_prescriber',
     ['prscrbr_npi', 'brnd_name', 'tot_clms']),
    ('cms_partd_prescriber',
     ['npi', 'drug_name', 'total_claim_count']),

    ('cms_chronic_conditions',
     ['bene_geo_cd', 'bene_cc_bh_diab_v2_pct']),
    ('cms_chronic_conditions',
     ['state', 'diabetes_prevalence']),

    ('cms_ed_utilization',
     ['state', 'ed_visits_per_1000']),
    ('cms_ed_utilization',
     ['state', 'ed_rate']),
]


def _snake(name: str) -> str:
    n = re.sub(r'[^0-9a-zA-Z]+', '_', name or '').strip('_').lower()
    return n or 'col'


def _match_shape(headers: List[str]) -> Optional[str]:
    hs = [h.strip().lower() for h in headers]
    for table, needles in _SHAPES:
        if all(any(n in h for h in hs) for n in needles):
            return table
    return None


def _sniff(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(65536)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',\t|;')
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(f, dialect=dialect)
            try:
                headers = next(reader)
            except StopIteration:
                return None
            return {'headers': headers, 'dialect': dialect}
    except OSError as e:
        _log.warning("cms_loader: cannot read %s: %s", path, e)
        return None


def _file_fingerprint(path: str) -> str:
    st = os.stat(path)
    return hashlib.md5(
        f"{path}:{st.st_size}:{int(st.st_mtime)}".encode('utf-8')
    ).hexdigest()[:16]


def _load_file(conn: sqlite3.Connection, path: str) -> Dict[str, Any]:
    sniff = _sniff(path)
    if not sniff:
        return {'file': path, 'status': 'skipped', 'reason': 'unreadable'}
    headers = sniff['headers']
    table = _match_shape(headers)
    if not table:
        return {'file': path, 'status': 'skipped',
                 'reason': 'no matching CMS shape'}

    snake_cols = [_snake(h) for h in headers]
    seen: Dict[str, int] = {}
    unique_cols = []
    for c in snake_cols:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)

    ddl_cols = ', '.join(f'"{c}" TEXT' for c in unique_cols)
    conn.execute(f'DROP TABLE IF EXISTS {table}')
    conn.execute(f'CREATE TABLE {table} ({ddl_cols})')

    placeholders = ','.join(['?'] * len(unique_cols))
    col_list_sql = ','.join('"' + c + '"' for c in unique_cols)
    ins_sql = f'INSERT INTO {table}({col_list_sql}) VALUES ({placeholders})'

    rows_loaded = 0
    batch: List[Tuple] = []
    BATCH = 5000
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, dialect=sniff['dialect'])
        next(reader, None)
        for row in reader:
            if len(row) < len(unique_cols):
                row = row + [''] * (len(unique_cols) - len(row))
            elif len(row) > len(unique_cols):
                row = row[:len(unique_cols)]
            batch.append(tuple(row))
            if len(batch) >= BATCH:
                conn.executemany(ins_sql, batch)
                rows_loaded += len(batch); batch.clear()
        if batch:
            conn.executemany(ins_sql, batch)
            rows_loaded += len(batch)

    first_col = unique_cols[0]
    try:
        conn.execute(f'CREATE INDEX IF NOT EXISTS ix_{table}_{first_col} '
                     f'ON {table}("{first_col}")')
    except sqlite3.Error:
        pass

    return {
        'file': os.path.basename(path),
        'table': table,
        'rows_loaded': rows_loaded,
        'columns': unique_cols,
        'fingerprint': _file_fingerprint(path),
        'status': 'loaded',
    }


def _ensure_ingest_log(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cms_ingest_log (
          file TEXT PRIMARY KEY,
          table_name TEXT,
          rows_loaded INTEGER,
          fingerprint TEXT,
          loaded_at REAL
        )
    """)


def scan_and_load(db_path: str,
                   drop_dir: Optional[str] = None,
                   *, reload_all: bool = False) -> Dict[str, Any]:
    drop_dir = drop_dir or os.environ.get('CMS_DROP_DIR', _DEFAULT_DROP)
    if not os.path.isdir(drop_dir):
        return {'drop_dir': drop_dir,
                 'status': 'no-directory',
                 'files_loaded': [], 'files_skipped': []}

    files = [
        os.path.join(drop_dir, f) for f in sorted(os.listdir(drop_dir))
        if f.lower().endswith(('.csv', '.tsv', '.txt'))
    ]
    if not files:
        return {'drop_dir': drop_dir, 'files_loaded': [], 'files_skipped': []}

    conn = sqlite3.connect(db_path)
    loaded: List[Dict] = []
    skipped: List[Dict] = []
    try:
        _ensure_ingest_log(conn)
        for path in files:
            fp = _file_fingerprint(path)
            if not reload_all:
                row = conn.execute(
                    "SELECT fingerprint FROM cms_ingest_log WHERE file = ?",
                    (os.path.basename(path),)
                ).fetchone()
                if row and row[0] == fp:
                    skipped.append({'file': os.path.basename(path),
                                     'reason': 'unchanged'})
                    continue
            try:
                res = _load_file(conn, path)
                if res.get('status') == 'loaded':
                    conn.execute(
                        "INSERT OR REPLACE INTO cms_ingest_log "
                        "(file, table_name, rows_loaded, fingerprint, loaded_at) "
                        "VALUES (?,?,?,?,?)",
                        (os.path.basename(path), res['table'],
                          res['rows_loaded'], res['fingerprint'], time.time())
                    )
                    loaded.append(res)
                else:
                    skipped.append(res)
            except Exception as e:
                _log.warning("cms_loader: failed to load %s: %s", path, e)
                skipped.append({'file': os.path.basename(path),
                                 'error': str(e)})
        conn.commit()
    finally:
        conn.close()

    return {'drop_dir': drop_dir,
             'files_loaded': loaded,
             'files_skipped': skipped}


def ingest_status(db_path: str) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(db_path)
        try:
            _ensure_ingest_log(conn)
            rows = conn.execute(
                "SELECT file, table_name, rows_loaded, loaded_at "
                "FROM cms_ingest_log ORDER BY loaded_at DESC"
            ).fetchall()
            return {
                'drop_dir': os.environ.get('CMS_DROP_DIR', _DEFAULT_DROP),
                'ingested': [
                    {'file': r[0], 'table': r[1], 'rows': r[2],
                      'loaded_at': r[3]} for r in rows
                ],
            }
        finally:
            conn.close()
    except sqlite3.Error as e:
        return {'error': str(e)}
