import os
import json
import hashlib
import sqlite3
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger('gpdm.schema_persist')


def compute_db_fingerprint(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        parts = []
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for tbl in tables:
            cols = conn.execute(f"PRAGMA table_info([{tbl}])").fetchall()
            col_sig = '|'.join(f"{c[1]}:{c[2]}" for c in cols)
            try:
                rc = conn.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]
            except Exception:
                rc = -1
            parts.append(f"{tbl}:{col_sig}:{rc}")
        conn.close()
        fp_str = '\n'.join(parts)
        return hashlib.sha256(fp_str.encode()).hexdigest()[:32]
    except Exception as e:
        logger.warning("Failed to compute DB fingerprint: %s", e)
        return ''


def _profile_to_dict(profile) -> dict:
    return {
        'table': profile.table,
        'name': profile.name,
        'data_type': profile.data_type,
        'sample_values': profile.sample_values,
        'distinct_count': profile.distinct_count,
        'null_pct': profile.null_pct,
        'is_numeric': profile.is_numeric,
        'is_date': profile.is_date,
        'is_categorical': profile.is_categorical,
        'is_id': profile.is_id,
        'is_text': profile.is_text,
        'min_val': profile.min_val,
        'max_val': profile.max_val,
        'semantic_tags': profile.semantic_tags,
    }


def _dict_to_profile(d: dict, ColumnProfile):
    p = ColumnProfile(d['table'], d['name'])
    p.data_type = d.get('data_type', 'text')
    p.sample_values = d.get('sample_values', [])
    p.distinct_count = d.get('distinct_count', 0)
    p.null_pct = d.get('null_pct', 0.0)
    p.is_numeric = d.get('is_numeric', False)
    p.is_date = d.get('is_date', False)
    p.is_categorical = d.get('is_categorical', False)
    p.is_id = d.get('is_id', False)
    p.is_text = d.get('is_text', False)
    p.min_val = d.get('min_val')
    p.max_val = d.get('max_val')
    p.semantic_tags = d.get('semantic_tags', [])
    return p


def save_schema_cache(learner, cache_path: str, db_path: str):
    try:
        fingerprint = compute_db_fingerprint(db_path)
        data = {
            'fingerprint': fingerprint,
            'saved_at': time.time(),
            'tables': {},
            'table_row_counts': dict(learner.table_row_counts),
            'join_graph': {k: dict(v) for k, v in learner.join_graph.items()},
        }
        for table, profiles in learner.tables.items():
            data['tables'][table] = [_profile_to_dict(p) for p in profiles]

        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logger.info(
            "Schema cache saved: %d tables, fingerprint=%s",
            len(data['tables']), fingerprint
        )
    except Exception as e:
        logger.warning("Failed to save schema cache: %s", e)


def load_schema_cache(cache_path: str, db_path: str, ColumnProfile_cls) -> Optional[dict]:
    if not os.path.exists(cache_path):
        logger.debug("No schema cache found at %s", cache_path)
        return None

    try:
        with open(cache_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to read schema cache: %s", e)
        return None

    current_fp = compute_db_fingerprint(db_path)
    cached_fp = data.get('fingerprint', '')
    if current_fp != cached_fp:
        logger.info(
            "Schema cache stale (fp %s != %s), will rebuild",
            cached_fp[:8], current_fp[:8]
        )
        return None

    try:
        tables = {}
        for table, profile_dicts in data.get('tables', {}).items():
            tables[table] = [
                _dict_to_profile(d, ColumnProfile_cls) for d in profile_dicts
            ]

        age = time.time() - data.get('saved_at', 0)
        logger.info(
            "Schema cache loaded: %d tables, age=%.0fs, fingerprint=%s",
            len(tables), age, cached_fp[:8]
        )

        return {
            'tables': tables,
            'table_row_counts': data.get('table_row_counts', {}),
            'join_graph': data.get('join_graph', {}),
        }
    except Exception as e:
        logger.warning("Failed to deserialize schema cache: %s", e)
        return None
