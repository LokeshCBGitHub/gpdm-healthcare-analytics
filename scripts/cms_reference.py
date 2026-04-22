from __future__ import annotations

import csv
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


_DEFAULT_DIR = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'cms_reference'
)


def _dir() -> str:
    return os.environ.get('CMS_REFERENCE_DIR', _DEFAULT_DIR)


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    except OSError as e:
        _log.warning("cms_reference: cannot read %s: %s", path, e)
        return []


def _coerce(v: str, t: type):
    if v is None or v == '':
        return None
    try:
        return t(v)
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def load_benchmarks() -> Dict[str, Dict[str, Any]]:
    rows = _read_csv(os.path.join(_dir(), 'benchmarks.csv'))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = (r.get('metric_key') or '').strip()
        if not k:
            continue
        out[k] = {
            'label': r.get('label'),
            'value': _coerce(r.get('value'), float),
            'unit': r.get('unit'),
            'population': r.get('population'),
            'source': r.get('source'),
            'year': _coerce(r.get('year'), int),
        }
    _log.info("cms_reference: loaded %d benchmarks", len(out))
    return out


@lru_cache(maxsize=1)
def load_icd_hcc_map() -> Dict[str, Dict[str, Any]]:
    rows = _read_csv(os.path.join(_dir(), 'icd_to_hcc.csv'))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = (r.get('icd_prefix') or '').strip().upper()
        if not k:
            continue
        out[k] = {
            'hcc_code': r.get('hcc_code'),
            'hcc_label': r.get('hcc_label'),
            'hcc_weight': _coerce(r.get('hcc_weight'), float) or 0.0,
        }
    _log.info("cms_reference: loaded %d ICD→HCC mappings", len(out))
    return out


@lru_cache(maxsize=1)
def load_hedis_measures() -> List[Dict[str, Any]]:
    rows = _read_csv(os.path.join(_dir(), 'hedis_measures.csv'))
    out = []
    for r in rows:
        out.append({
            'code': r.get('measure_code'),
            'name': r.get('measure_name'),
            'domain': r.get('domain'),
            'num': r.get('numerator_definition'),
            'den': r.get('denominator_definition'),
            'age_min': _coerce(r.get('age_min'), int),
            'age_max': _coerce(r.get('age_max'), int),
            'source': r.get('source'),
        })
    return out


@lru_cache(maxsize=1)
def load_condition_prevalence() -> List[Dict[str, Any]]:
    rows = _read_csv(os.path.join(_dir(), 'chronic_conditions_prevalence.csv'))
    out = []
    for r in rows:
        out.append({
            'condition': r.get('condition'),
            'prev_medicare': _coerce(r.get('prevalence_medicare'), float),
            'prev_commercial': _coerce(r.get('prevalence_commercial'), float),
            'source': r.get('source'),
            'year': _coerce(r.get('year'), int),
        })
    return out


def benchmark_for(metric: str, lob: str = 'commercial') -> Optional[Dict]:
    bm = load_benchmarks()
    key_ordered = [
        f"{metric}_{lob.lower()}",
        f"{metric}",
        f"{metric}_national",
    ]
    for k in key_ordered:
        if k in bm:
            return bm[k]
    return None


def reload_all() -> Dict[str, int]:
    load_benchmarks.cache_clear()
    load_icd_hcc_map.cache_clear()
    load_hedis_measures.cache_clear()
    load_condition_prevalence.cache_clear()
    return {
        'benchmarks': len(load_benchmarks()),
        'icd_hcc': len(load_icd_hcc_map()),
        'hedis_measures': len(load_hedis_measures()),
        'condition_prevalence': len(load_condition_prevalence()),
    }


def reference_status() -> Dict[str, Any]:
    return {
        'directory': _dir(),
        'benchmarks': len(load_benchmarks()),
        'icd_hcc_map': len(load_icd_hcc_map()),
        'hedis_measures': len(load_hedis_measures()),
        'condition_prevalence': len(load_condition_prevalence()),
    }
