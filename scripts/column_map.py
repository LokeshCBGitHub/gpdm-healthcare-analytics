from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None

_log = logging.getLogger("gpdm.column_map")


_DEFAULT_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "claims": {
        "claim_id":       ["claim_id", "CLM_ID", "CLAIM_ID", "clm_id"],
        "member_id":      ["member_id", "MBR_ID", "MEMBER_ID", "mbr_id", "patient_id"],
        "encounter_id":   ["encounter_id", "ENCOUNTER_ID", "enc_id", "VISIT_ID"],
        "paid_amount":    ["paid_amount", "paid", "CLM_PMT_AMT", "PAID_AMT",
                            "PAID_USD", "AMT_PAID", "PAID"],
        "billed_amount":  ["billed_amount", "billed", "charge_amount",
                            "CHG_AMT", "CHARGE_AMT"],
        "service_date":   ["service_date", "svc_date", "SVC_DT", "DOS",
                            "claim_date", "CLAIM_DATE"],
        "submitted_date": ["submitted_date", "sub_date", "SUBMIT_DT"],
        "adjudicated_date": ["adjudicated_date", "adj_date", "paid_date", "PAID_DT"],
        "claim_status":   ["claim_status", "status", "CLM_STS_CD"],
        "diagnosis":      ["diagnosis", "diag_code", "ICD_DIAG_CD", "ICD10",
                            "PRIMARY_DIAG"],
        "provider_id":    ["provider_id", "PROV_NPI", "NPI", "PROVIDER_ID"],
        "region":         ["region", "KP_REGION", "REGION", "REGION_CD"],
    },
    "encounters": {
        "encounter_id":    ["encounter_id", "ENCOUNTER_ID", "enc_id", "VISIT_ID"],
        "member_id":       ["member_id", "MBR_ID", "MEMBER_ID", "patient_id"],
        "provider_id":     ["provider_id", "PROV_NPI", "NPI", "PROVIDER_ID",
                             "rendering_provider_id"],
        "encounter_date":  ["encounter_date", "admit_date", "ADMIT_DT",
                             "service_date", "visit_date", "date",
                             "SVC_DT", "DOS"],
        "admit_date":      ["admit_date", "ADMIT_DT", "encounter_date",
                             "service_date", "visit_date", "date"],
        "discharge_date":  ["discharge_date", "disch_date", "DISCH_DT"],
        "visit_type":      ["visit_type", "encounter_type", "type", "ENC_TYP_CD"],
        "length_of_stay":  ["length_of_stay", "los", "LOS_DAYS"],
        "total_paid":      ["total_paid", "paid_amount", "cost", "allowed",
                             "PAID_AMT", "TOT_PAID", "CLM_PMT_AMT"],
        "facility_id":     ["facility_id", "FAC_ID", "HOSPITAL_ID"],
        "region":          ["region", "KP_REGION", "REGION", "REGION_CD"],
        "primary_dx":      ["primary_dx", "diagnosis", "diag_code",
                             "ICD_DIAG_CD", "ICD10", "PRIMARY_DIAG",
                             "principal_dx"],
        "is_admission":    ["is_admission", "admit_flag", "IP_FLAG",
                             "inpatient_flag"],
        "readmit_30d":     ["readmit_30d", "READMIT_30D_FLG", "readmit_flag"],
    },
    "members": {
        "member_id":       ["member_id", "MBR_ID", "MEMBER_ID", "id",
                             "patient_id", "PAT_ID"],
        "dob":             ["dob", "DOB", "birth_date", "DT_OF_BIRTH"],
        "age":             ["age", "AGE", "member_age", "age_yrs"],
        "gender":          ["gender", "sex", "GNDR_CD", "SEX_CD"],
        "region":          ["region", "KP_REGION", "state", "ST_CD", "REGION_CD",
                            "REGION"],
        "lob":             ["lob", "line_of_business", "LOB_CD", "product",
                             "PLAN_LINE"],
        "plan":             ["plan", "plan_type", "PLN_TYP_CD", "plan_id"],
        "enrollment_date":  ["enrollment_date", "ENROLL_DT", "effective_date"],
        "risk_score":       ["risk_score", "RISK_SCR", "hcc_score",
                             "predicted_risk"],
        "pcp_provider_id":  ["pcp_provider_id", "PCP_NPI", "pcp_id",
                             "primary_pcp", "PCP_PROV_ID"],
    },
    "providers": {
        "provider_id":    ["provider_id", "PROV_NPI", "NPI", "PROVIDER_ID"],
        "provider_name":  ["provider_name", "name", "PROV_NAME",
                            "PROVIDER_NM"],
        "specialty":      ["specialty", "SPCLTY_CD", "taxonomy"],
        "network_status": ["network_status", "NTWK_STS", "in_network"],
        "name":           ["name", "PROV_NAME", "provider_name"],
        "region":         ["region", "KP_REGION", "REGION", "REGION_CD"],
    },
    "prescriptions": {
        "rx_id":       ["rx_id", "RX_ID", "script_id"],
        "member_id":   ["member_id", "MBR_ID", "MEMBER_ID"],
        "ndc":         ["ndc", "NDC_CD", "ndc_code"],
        "fill_date":   ["fill_date", "FILL_DT", "dispense_date"],
        "days_supply": ["days_supply", "DAYS_SUPPLY_QTY"],
        "paid_amount": ["paid_amount", "paid", "PAID_AMT"],
    },
}

_DEFAULT_TABLE_ALIASES: Dict[str, List[str]] = {
    "claims":        ["claims", "claim", "CLAIMS", "KP_CLAIMS_MASTER",
                       "gpdm.claims_silver", "claims_adjudicated"],
    "encounters":    ["encounters", "encounter", "visits", "ENCOUNTERS",
                       "KP_ENCOUNTERS", "gpdm.encounters_silver"],
    "members":       ["members", "member", "MEMBERS", "KP_MEMBER_MASTER",
                       "patients", "PATIENTS"],
    "providers":     ["providers", "provider", "PROVIDERS", "KP_PROVIDERS"],
    "prescriptions": ["prescriptions", "pharmacy", "RX", "KP_RX"],
}


@dataclass
class TableMap:
    logical:       str
    physical_name: Optional[str] = None
    columns:       Dict[str, str] = field(default_factory=dict)


@dataclass
class ColumnMap:
    source: str = "default"
    tables: Dict[str, TableMap] = field(default_factory=dict)

    def merge_defaults(self) -> None:
        for tbl_logical, col_aliases in _DEFAULT_ALIASES.items():
            tmap = self.tables.setdefault(tbl_logical, TableMap(tbl_logical))
            for col_logical, aliases in col_aliases.items():
                if col_logical not in tmap.columns:
                    tmap.columns[col_logical] = aliases[0]


def _config_dir() -> str:
    override = os.environ.get("GPDM_COLUMN_MAP_DIR")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "config"))


def _load_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as fh:
            if path.endswith((".yaml", ".yml")):
                if yaml is None:
                    _log.warning("PyYAML not installed; skipping %s", path)
                    return None
                return yaml.safe_load(fh)
            return json.load(fh)
    except Exception as exc:
        _log.warning("Failed to load %s: %s", path, exc)
        return None


def _discover_map_file() -> Optional[str]:
    d = _config_dir()
    if not os.path.isdir(d):
        return None
    source = os.environ.get("GPDM_COLUMN_MAP_SOURCE", "auto").strip()
    candidates: List[str] = []
    for f in sorted(os.listdir(d)):
        if not f.startswith("column_map."):
            continue
        if not f.endswith((".yaml", ".yml", ".json")):
            continue
        if ".draft." in f:
            continue
        if source != "auto" and f"column_map.{source}." not in f:
            continue
        candidates.append(os.path.join(d, f))
    return candidates[0] if candidates else None


def _parse(raw: dict) -> ColumnMap:
    cm = ColumnMap(source=str(raw.get("source", "default")))
    for tbl_logical, tbl_spec in (raw.get("tables") or {}).items():
        tm = TableMap(logical=tbl_logical)
        if isinstance(tbl_spec, dict):
            tm.physical_name = tbl_spec.get("physical_name") or tbl_spec.get("physical")
            for col_logical, col_physical in (tbl_spec.get("columns") or {}).items():
                tm.columns[col_logical] = str(col_physical)
        cm.tables[tbl_logical] = tm
    cm.merge_defaults()
    return cm


_LOCK = threading.RLock()
_CACHE: Dict[str, ColumnMap] = {}


def load(source: Optional[str] = None, force: bool = False) -> ColumnMap:
    key = source or os.environ.get("GPDM_COLUMN_MAP_SOURCE", "auto")
    with _LOCK:
        if not force and key in _CACHE:
            return _CACHE[key]
        path = _discover_map_file()
        if path:
            raw = _load_file(path)
            if raw:
                cm = _parse(raw)
                _log.info("Loaded column map: source=%s file=%s tables=%d",
                          cm.source, os.path.basename(path), len(cm.tables))
                _CACHE[key] = cm
                return cm
        cm = ColumnMap(source="default")
        cm.merge_defaults()
        _CACHE[key] = cm
        return cm


def _physical_candidates(cm: ColumnMap, logical_table: str,
                         logical_col: str) -> List[str]:
    out: List[str] = []
    tm = cm.tables.get(logical_table)
    if tm and logical_col in tm.columns:
        out.append(tm.columns[logical_col])
    for alias in _DEFAULT_ALIASES.get(logical_table, {}).get(logical_col, []):
        if alias not in out:
            out.append(alias)
    if logical_col not in out:
        out.append(logical_col)
    return out


def resolve_column(conn: Optional[sqlite3.Connection],
                   logical_table: str,
                   logical_col: str,
                   physical_table: Optional[str] = None) -> Optional[str]:
    cm = load()
    candidates = _physical_candidates(cm, logical_table, logical_col)
    if conn is None or physical_table is None:
        return candidates[0]
    try:
        cur = conn.execute(f'PRAGMA table_info("{physical_table}")')
        existing = {row[1].lower(): row[1] for row in cur.fetchall()}
    except sqlite3.Error:
        return candidates[0]
    for cand in candidates:
        if cand.lower() in existing:
            if os.environ.get("GPDM_COLUMN_MAP_DEBUG") == "1":
                _log.info("resolve_column %s.%s -> %s (via %s)",
                          logical_table, logical_col, existing[cand.lower()], cand)
            return existing[cand.lower()]
    return None


def resolve_table(conn: Optional[sqlite3.Connection],
                  logical_table: str) -> Optional[str]:
    cm = load()
    out: List[str] = []
    tm = cm.tables.get(logical_table)
    if tm and tm.physical_name:
        out.append(tm.physical_name)
    for alias in _DEFAULT_TABLE_ALIASES.get(logical_table, []):
        if alias not in out:
            out.append(alias)
    if logical_table not in out:
        out.append(logical_table)
    if conn is None:
        return out[0]
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        existing = {row[0].lower(): row[0] for row in cur.fetchall()}
    except sqlite3.Error:
        return out[0]
    for cand in out:
        if cand.lower() in existing:
            return existing[cand.lower()]
    return None


def resolve_many(conn: sqlite3.Connection,
                 logical_table: str,
                 logical_cols: List[str]) -> Dict[str, Optional[str]]:
    phys_table = resolve_table(conn, logical_table)
    return {
        lc: resolve_column(conn, logical_table, lc, phys_table)
        for lc in logical_cols
    }


def status() -> Dict[str, object]:
    cm = load()
    return {
        "source":  cm.source,
        "tables":  {t: {"physical": m.physical_name,
                        "columns":  dict(m.columns)} for t, m in cm.tables.items()},
        "config_dir":  _config_dir(),
        "active_file": _discover_map_file(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(status())
