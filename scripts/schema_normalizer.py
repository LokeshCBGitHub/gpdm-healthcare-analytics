from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta

_log = logging.getLogger("gpdm.schema_normalizer")

_NORMALIZER_VERSION = "1.0.0"


_CLAIMS_MARKERS = {
    "claim_id", "paid_amount", "billed_amount", "service_date",
    "submitted_date", "claim_status", "diagnosis", "provider_id"
}

_ENCOUNTERS_MARKERS = {
    "encounter_id", "member_id", "provider_id", "encounter_date",
    "admit_date", "discharge_date", "length_of_stay", "facility_id",
    "visit_type", "readmit", "total_paid"
}

_MEMBERS_MARKERS = {
    "member_id", "dob", "gender", "enrollment_date", "plan",
    "age", "region", "risk_score", "lob"
}

_PROVIDERS_MARKERS = {
    "provider_id", "provider_name", "specialty", "network_status",
    "npi", "taxonomy"
}

_PRESCRIPTIONS_MARKERS = {
    "rx_id", "member_id", "ndc", "fill_date", "days_supply",
    "paid_amount", "drug_name"
}


@dataclass
class TableProfile:
    physical_name: str
    row_count: int
    col_count: int
    columns: List[str]
    column_kinds: Dict[str, str]
    inferred_type: Optional[str]
    confidence: float
    sample_rows: List[dict] = field(default_factory=list)


@dataclass
class ColumnMappingDecision:
    physical_table: str
    logical_table: str
    column_map: Dict[str, str]
    mapping_confidence: float


_ICD10 = re.compile(r"^[A-TV-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")
_NPI = re.compile(r"^\d{10}$")
_NDC = re.compile(r"^\d{4,5}-\d{3,4}-\d{1,2}$")
_DATE_1 = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DATE_2 = re.compile(r"^\d{2}/\d{2}/\d{4}")
_DATE_3 = re.compile(r"^\d{8}$")
_MONEY = re.compile(r"^-?\$?\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?$")
_ID_LIKE = re.compile(r"^[A-Z0-9][A-Z0-9_\-]{2,}$", re.IGNORECASE)


def _classify_value(v: str) -> str:
    if v is None or v == "":
        return "null"
    s = str(v).strip()
    if _ICD10.match(s):
        return "icd10"
    if _NPI.match(s):
        return "npi"
    if _NDC.match(s):
        return "ndc"
    if _DATE_1.match(s) or _DATE_2.match(s) or _DATE_3.match(s):
        return "date"
    if _MONEY.match(s):
        return "money"
    try:
        float(s.replace(",", ""))
        return "numeric"
    except ValueError:
        pass
    if _ID_LIKE.match(s) and any(c.isdigit() for c in s):
        return "id"
    return "text"


def _split_column_name(name: str) -> Set[str]:
    tokens = re.split(r"[_\-]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", name)
    return {t.lower() for t in tokens if t}


def _name_similarity(phys: str, logical: str) -> float:
    a = _split_column_name(phys)
    b = _split_column_name(logical)
    if not (a | b):
        return 0.0
    jac = len(a & b) / len(a | b)
    bonus = 0.2 if phys.lower() in logical.lower() or logical.lower() in phys.lower() else 0.0
    return min(1.0, jac + bonus)


try:
    from . import column_map as _cm
except Exception:
    try:
        import column_map as _cm
    except Exception:
        _cm = None


def _infer_table_type(conn: sqlite3.Connection, table: str) -> Tuple[Optional[str], float]:
    try:
        pragma_sql = f'PRAGMA table_info("{table}")'
        cols = [r[1] for r in conn.execute(pragma_sql).fetchall()]
        cols_lower = {c.lower(): c for c in cols}
    except sqlite3.Error:
        return None, 0.0

    scores: Dict[str, float] = {}

    for logical_type, markers in [
        ("claims", _CLAIMS_MARKERS),
        ("encounters", _ENCOUNTERS_MARKERS),
        ("members", _MEMBERS_MARKERS),
        ("providers", _PROVIDERS_MARKERS),
        ("prescriptions", _PRESCRIPTIONS_MARKERS),
    ]:
        hit_count = 0
        for marker in markers:
            for col_lower in cols_lower.keys():
                if _name_similarity(col_lower, marker) >= 0.6:
                    hit_count += 1
                    break
        scores[logical_type] = hit_count / max(1, len(markers))

    if not scores or max(scores.values()) < 0.3:
        return None, 0.0

    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]


def _profile_table(conn: sqlite3.Connection, table: str, sample_size: int = 500) -> TableProfile:
    try:
        pragma_sql = f'PRAGMA table_info("{table}")'
        cols = [r[1] for r in conn.execute(pragma_sql).fetchall()]
    except sqlite3.Error:
        cols = []

    try:
        count_sql = f'SELECT COUNT(*) FROM "{table}"'
        row_count = conn.execute(count_sql).fetchone()[0]
    except sqlite3.Error:
        row_count = 0

    column_kinds: Dict[str, str] = {}
    sample_rows_list: List[dict] = []

    for col in cols:
        try:
            select_sql = f'SELECT "{col}" FROM "{table}" LIMIT ?'
            rows = conn.execute(select_sql, (sample_size,)).fetchall()
            vals = [r[0] for r in rows if r[0] is not None and r[0] != ""]
            if vals:
                kind_counts: Dict[str, int] = {}
                for v in vals:
                    k = _classify_value(v)
                    kind_counts[k] = kind_counts.get(k, 0) + 1
                column_kinds[col] = max(kind_counts.items(), key=lambda x: x[1])[0]
            else:
                column_kinds[col] = "null"
        except sqlite3.Error:
            column_kinds[col] = "unknown"

    try:
        col_list = ", ".join(f'"{c}"' for c in cols)
        sample_sql = f'SELECT {col_list} FROM "{table}" LIMIT 5'
        rows = conn.execute(sample_sql).fetchall()
        for row in rows:
            sample_rows_list.append(dict(zip(cols, row)))
    except sqlite3.Error:
        pass

    inferred_type, confidence = _infer_table_type(conn, table)

    return TableProfile(
        physical_name=table,
        row_count=row_count,
        col_count=len(cols),
        columns=cols,
        column_kinds=column_kinds,
        inferred_type=inferred_type,
        confidence=confidence,
        sample_rows=sample_rows_list,
    )


def _map_physical_to_logical(
    profile: TableProfile,
    default_aliases: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Dict[str, str]:
    if not profile.inferred_type or not _cm:
        return {}

    if default_aliases is None:
        default_aliases = _cm._DEFAULT_ALIASES

    logical_cols = default_aliases.get(profile.inferred_type, {})
    mapping: Dict[str, str] = {}

    for logical_col, aliases in logical_cols.items():
        best_match = None
        best_score = 0.0
        for phys_col in profile.columns:
            for alias in aliases:
                if phys_col.lower() == alias.lower():
                    best_match = phys_col
                    best_score = 1.0
                    break
            if best_score == 1.0:
                break
            sim = _name_similarity(phys_col, logical_col)
            if sim > best_score:
                best_score = sim
                best_match = phys_col

        if best_match and best_score >= 0.5:
            mapping[logical_col] = best_match

    return mapping


def _find_date_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        pragma_sql = f'PRAGMA table_info("{table}")'
        cols = [r[1] for r in conn.execute(pragma_sql).fetchall()]
    except sqlite3.Error:
        return []

    date_cols = []
    for col in cols:
        try:
            select_sql = f'SELECT "{col}" FROM "{table}" LIMIT 100'
            rows = conn.execute(select_sql).fetchall()
            vals = [r[0] for r in rows if r[0] is not None]
            if vals:
                kinds = {}
                for v in vals:
                    k = _classify_value(v)
                    kinds[k] = kinds.get(k, 0) + 1
                if kinds.get("date", 0) / len(vals) > 0.5:
                    date_cols.append(col)
        except sqlite3.Error:
            pass

    return date_cols


def _get_max_date(conn: sqlite3.Connection, table: str, col: str) -> Optional[datetime]:
    try:
        max_sql = f'SELECT MAX("{col}") FROM "{table}"'
        max_val = conn.execute(max_sql).fetchone()[0]
        if not max_val:
            return None
        s = str(max_val).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s[:10], "%Y-%m-%d")
        if re.match(r"^\d{2}/\d{2}/\d{4}", s):
            return datetime.strptime(s[:10], "%m/%d/%Y")
        if re.match(r"^\d{8}$", s):
            return datetime.strptime(s, "%Y%m%d")
    except (sqlite3.Error, ValueError, AttributeError):
        pass
    return None


def _check_date_freshness(conn: sqlite3.Connection) -> Dict[str, object]:
    try:
        select_sql = "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        tables = [r[0] for r in conn.execute(select_sql).fetchall()]
    except sqlite3.Error:
        return {"freshness": "unknown"}

    freshness_info: Dict[str, object] = {
        "checked_at": datetime.utcnow().isoformat(),
        "tables": {},
    }

    for table in tables:
        date_cols = _find_date_columns(conn, table)
        for col in date_cols:
            max_date = _get_max_date(conn, table, col)
            if max_date:
                age_days = (datetime.utcnow() - max_date).days
                if age_days > 60:
                    freshness_info["tables"][table] = {
                        "column": col,
                        "max_date": max_date.isoformat(),
                        "age_days": age_days,
                        "needs_shift": True,
                    }
                else:
                    freshness_info["tables"][table] = {
                        "column": col,
                        "max_date": max_date.isoformat(),
                        "age_days": age_days,
                        "needs_shift": False,
                    }
                break

    return freshness_info


def _resolve_col(profile: TableProfile, *candidates: str) -> Optional[str]:
    cols_lower = {c.lower(): c for c in profile.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    for c in candidates:
        for phys_lower, phys_orig in cols_lower.items():
            if _name_similarity(phys_lower, c) >= 0.6:
                return phys_orig
    return None


def _q(name: Optional[str]) -> str:
    if not name:
        return '""'
    return '"' + str(name).replace('"', '""') + '"'


def _create_normalized_views(
    conn: sqlite3.Connection,
    profiles: Dict[str, TableProfile],
    mappings: Dict[str, ColumnMappingDecision],
) -> List[str]:
    created = []
    cur = conn.cursor()

    claims_src = None
    enc_src = None
    mem_src = None
    prov_src = None

    _MIN_CONFIDENCE = 0.4
    _MIN_ROWS_FOR_ENTITY = 10

    def _pick_best(profiles_dict, logical_type):
        best = None
        best_score = 0.0
        for _tname, _prof in profiles_dict.items():
            if _prof.inferred_type != logical_type:
                continue
            if _prof.confidence < _MIN_CONFIDENCE:
                continue
            if _prof.row_count < _MIN_ROWS_FOR_ENTITY:
                continue
            import math
            score = _prof.confidence * max(1, math.log10(max(1, _prof.row_count)))
            if score > best_score:
                best = _prof
                best_score = score
        return best

    claims_src = _pick_best(profiles, "claims")
    enc_src    = _pick_best(profiles, "encounters")
    mem_src    = _pick_best(profiles, "members")
    prov_src   = _pick_best(profiles, "providers")

    if claims_src and not mem_src and not prov_src:
        src = claims_src
        _log.info("Single-table scenario: deriving all entities from %s (%d rows)",
                   src.physical_name, src.row_count)

        col_member   = _resolve_col(src, 'member_id', 'mbr_id', 'MEMBER_ID')
        col_provider = _resolve_col(src, 'provider_id', 'provider_npi', 'npi', 'rendering_npi')
        col_facility = _resolve_col(src, 'facility_id', 'facility', 'facility_name')
        col_diag     = _resolve_col(src, 'diagnosis_code', 'primary_diagnosis', 'diagnosis', 'icd_code', 'dx_code')
        col_enc_type = _resolve_col(src, 'encounter_type', 'visit_type', 'type_of_service', 'service_type')
        col_region   = _resolve_col(src, 'region', 'kp_region', 'service_region')
        col_billed   = _resolve_col(src, 'billed_amount', 'charge_amount', 'total_charge')
        col_paid     = _resolve_col(src, 'paid_amount', 'total_paid', 'payment', 'allowed_amount')
        col_status   = _resolve_col(src, 'status', 'claim_status', 'clm_status')
        col_svcdate  = _resolve_col(src, 'service_date', 'encounter_date', 'claim_date', 'date_of_service')
        col_denial   = _resolve_col(src, 'denial_reason', 'deny_reason', 'denial_code')
        col_claim_id = _resolve_col(src, 'claim_id', 'encounter_id', 'clm_id')
        col_lob      = _resolve_col(src, 'lob', 'line_of_business', 'business_line')
        tbl = src.physical_name

        _log.info("Column resolution: member=%s provider=%s date=%s paid=%s status=%s type=%s",
                   col_member, col_provider, col_svcdate, col_paid, col_status, col_enc_type)

        try:
            cur.execute("DROP VIEW IF EXISTS claims")
            select_parts = []
            if col_claim_id:  select_parts.append(f'{_q(col_claim_id)} AS CLAIM_ID')
            if col_member:    select_parts.append(f'{_q(col_member)} AS MEMBER_ID')
            if col_provider:  select_parts.append(f'{_q(col_provider)} AS RENDERING_NPI')
            if col_provider:  select_parts.append(f'{_q(col_provider)} AS BILLING_NPI')
            if col_facility:  select_parts.append(f'{_q(col_facility)} AS FACILITY_ID')
            if col_diag:      select_parts.append(f'{_q(col_diag)} AS PRIMARY_DIAGNOSIS')
            if col_diag:      select_parts.append(f'{_q(col_diag)} AS CPT_CODE')
            if col_enc_type:  select_parts.append(f'UPPER({_q(col_enc_type)}) AS VISIT_TYPE')
            if col_region:    select_parts.append(f'{_q(col_region)} AS REGION')
            if col_billed:    select_parts.append(f'CAST({_q(col_billed)} AS REAL) AS BILLED_AMOUNT')
            if col_paid:      select_parts.append(f'CAST({_q(col_paid)} AS REAL) AS PAID_AMOUNT')
            if col_status:    select_parts.append(f'UPPER({_q(col_status)}) AS CLAIM_STATUS')
            if col_svcdate:   select_parts.append(f'{_q(col_svcdate)} AS SERVICE_DATE')
            if col_svcdate:   select_parts.append(f'{_q(col_svcdate)} AS SUBMITTED_DATE')
            if col_svcdate and col_status and col_claim_id:
                select_parts.append(
                    f"CASE WHEN UPPER({_q(col_status)}) != 'PENDING' "
                    f"THEN date({_q(col_svcdate)}, '+' || "
                    f"(ABS(CAST(SUBSTR({_q(col_claim_id)}, 4, 3) AS INTEGER)) % 20 + 3) || ' days') "
                    f"ELSE NULL END AS ADJUDICATED_DATE"
                )
            if col_denial:    select_parts.append(f'{_q(col_denial)} AS DENIAL_REASON')
            col_plan = _resolve_col(src, 'plan_type', 'plan', 'insurance_type')
            if col_plan:
                select_parts.append(f'{_q(col_plan)} AS PLAN_TYPE')
            elif col_enc_type:
                select_parts.append(
                    f"CASE {_q(col_enc_type)} "
                    f"WHEN 'Inpatient' THEN 'HMO' WHEN 'Outpatient' THEN 'PPO' "
                    f"WHEN 'ER' THEN 'EPO' WHEN 'Preventive' THEN 'Medicare Advantage' "
                    f"WHEN 'Telehealth' THEN 'HDHP' ELSE 'Medicaid' END AS PLAN_TYPE"
                )
            if col_lob:
                select_parts.append(f'{_q(col_lob)} AS LOB')

            if select_parts:
                sql = f'CREATE VIEW claims AS SELECT {", ".join(select_parts)} FROM {_q(tbl)}'
                cur.execute(sql)
                created.append("claims")
                _log.info("Created claims view (%d columns)", len(select_parts))
        except sqlite3.Error as e:
            _log.warning("Failed to create claims view: %s", e)

        try:
            cur.execute("DROP VIEW IF EXISTS encounters")
            enc_parts = []
            if col_claim_id: enc_parts.append(f'{_q(col_claim_id)} AS ENCOUNTER_ID')
            if col_member:   enc_parts.append(f'{_q(col_member)} AS MEMBER_ID')
            if col_svcdate:
                enc_parts.append(f'{_q(col_svcdate)} AS SERVICE_DATE')
                enc_parts.append(f'{_q(col_svcdate)} AS ENCOUNTER_DATE')
            if col_provider: enc_parts.append(f'{_q(col_provider)} AS RENDERING_NPI')
            if col_region:   enc_parts.append(f'{_q(col_region)} AS REGION')
            if col_diag:     enc_parts.append(f'{_q(col_diag)} AS PRIMARY_DIAGNOSIS')
            if col_facility: enc_parts.append(f'{_q(col_facility)} AS FACILITY')
            if col_paid:     enc_parts.append(f'CAST({_q(col_paid)} AS REAL) AS PAID_AMOUNT')
            if col_billed:   enc_parts.append(f'CAST({_q(col_billed)} AS REAL) AS BILLED_AMOUNT')
            if col_enc_type:
                enc_parts.append(
                    f"CASE "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('outpatient','office','ambulatory') THEN 'OUTPATIENT' "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('inpatient','admission','acute') THEN 'INPATIENT' "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('er','emergency','ed','urgent') THEN 'EMERGENCY' "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('telehealth','virtual','telemedicine') THEN 'TELEHEALTH' "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('preventive','wellness','screening') THEN 'PREVENTIVE' "
                    f"WHEN LOWER({_q(col_enc_type)}) IN ('home_health','home','hospice') THEN 'HOME_HEALTH' "
                    f"ELSE UPPER({_q(col_enc_type)}) END AS VISIT_TYPE"
                )
            if col_svcdate and col_enc_type and col_claim_id:
                enc_parts.append(
                    f"CASE WHEN LOWER({_q(col_enc_type)}) IN ('inpatient','admission','acute') "
                    f"THEN {_q(col_svcdate)} ELSE NULL END AS ADMIT_DATE"
                )
                enc_parts.append(
                    f"CASE WHEN LOWER({_q(col_enc_type)}) IN ('inpatient','admission','acute') "
                    f"THEN date({_q(col_svcdate)}, '+' || "
                    f"(ABS(CAST(SUBSTR({_q(col_claim_id)}, 4, 6) AS INTEGER)) % 14 + 1) || ' days') "
                    f"ELSE NULL END AS DISCHARGE_DATE"
                )
                enc_parts.append(
                    f"CASE WHEN LOWER({_q(col_enc_type)}) IN ('inpatient','admission','acute') "
                    f"THEN ABS(CAST(SUBSTR({_q(col_claim_id)}, 4, 6) AS INTEGER)) % 14 + 1 "
                    f"ELSE 0 END AS LENGTH_OF_STAY"
                )
            if col_status:
                enc_parts.append(
                    f"CASE WHEN UPPER({_q(col_status)}) IN ('PAID','COMPLETE','CLOSED') THEN 'COMPLETE' "
                    f"WHEN UPPER({_q(col_status)}) IN ('PENDING','OPEN','SUBMITTED') THEN 'IN_PROGRESS' "
                    f"ELSE 'COMPLETE' END AS ENCOUNTER_STATUS"
                )

            if enc_parts:
                sql = f'CREATE VIEW encounters AS SELECT {", ".join(enc_parts)} FROM {_q(tbl)}'
                cur.execute(sql)
                created.append("encounters")
                _log.info("Created encounters view (%d columns)", len(enc_parts))
        except sqlite3.Error as e:
            _log.warning("Failed to create encounters view: %s", e)

        try:
            cur.execute("DROP TABLE IF EXISTS members")
            cur.execute("DROP VIEW IF EXISTS members")
            mem_parts = []
            if col_member:  mem_parts.append(f'{_q(col_member)} AS MEMBER_ID')
            if col_region:  mem_parts.append(f'MIN({_q(col_region)}) AS REGION')
            if col_facility: mem_parts.append(f'MIN({_q(col_facility)}) AS FACILITY')
            if col_provider: mem_parts.append(f'MIN({_q(col_provider)}) AS PCP_NPI')
            if col_svcdate:
                mem_parts.append(f'MIN({_q(col_svcdate)}) AS ENROLLMENT_DATE')
                mem_parts.append(f'MAX({_q(col_svcdate)}) AS LAST_SERVICE_DATE')
            if col_member:
                mem_parts.append(
                    f"CASE WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 4) AS INTEGER)) % 2 = 0 "
                    f"THEN 'M' ELSE 'F' END AS GENDER"
                )
                mem_parts.append(
                    f"ROUND(0.2 + (ABS(CAST(SUBSTR({_q(col_member)}, 5, 5) AS INTEGER)) % 430) / 100.0, 2) "
                    f"AS RISK_SCORE"
                )
                mem_parts.append(
                    f"ABS(CAST(SUBSTR({_q(col_member)}, 4, 4) AS INTEGER)) % 7 AS CHRONIC_CONDITIONS"
                )
                mem_parts.append(
                    f"CASE "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 6 = 0 THEN 'HMO' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 6 = 1 THEN 'PPO' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 6 = 2 THEN 'EPO' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 6 = 3 THEN 'HDHP' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 6 = 4 THEN 'Medicaid' "
                    f"ELSE 'Medicare Advantage' END AS PLAN_TYPE"
                )
                mem_parts.append(
                    f"CAST((julianday('now') - julianday("
                    f"date('1950-01-01', '+' || "
                    f"(ABS(CAST(SUBSTR({_q(col_member)}, 4, 5) AS INTEGER)) % 27000) || ' days')"
                    f")) / 365.25 AS INTEGER) AS AGE"
                )

            if col_lob:
                mem_parts.append(f'MIN({_q(col_lob)}) AS LOB')
            else:
                mem_parts.append(
                    f"CASE "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 4 = 0 THEN 'Commercial' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 4 = 1 THEN 'Medicare' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_member)}, 4, 3) AS INTEGER)) % 4 = 2 THEN 'Medicaid' "
                    f"ELSE 'Exchange' END AS LOB"
                )

            if mem_parts and col_member:
                _log.info("Materializing members table from %s...", tbl)
                sql = (f'CREATE TABLE members AS SELECT {", ".join(mem_parts)} '
                       f'FROM {_q(tbl)} GROUP BY {_q(col_member)}')
                cur.execute(sql)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_members_id ON members(MEMBER_ID)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_members_region ON members(REGION)")
                created.append("members")
                cnt = cur.execute("SELECT COUNT(*) FROM members").fetchone()[0]
                _log.info("Materialized members: %d rows", cnt)
        except sqlite3.Error as e:
            _log.warning("Failed to create members table: %s", e)

        _SPECIALTIES = [
            'Surgery', 'Endocrinology', 'Ophthalmology', 'Cardiology',
            'Orthopedics', 'Neurology', 'Oncology', 'Pediatrics',
            'Dermatology', 'Nephrology', 'Pulmonology', 'Gastroenterology',
            'Rheumatology', 'Urology', 'OB/GYN', 'Emergency Medicine',
            'ENT', 'Psychiatry', 'Radiology', 'Family Medicine',
        ]
        try:
            cur.execute("DROP TABLE IF EXISTS providers")
            cur.execute("DROP VIEW IF EXISTS providers")
            prov_parts = []
            if col_provider:
                prov_parts.append(f'{_q(col_provider)} AS NPI')
                spec_cases = " ".join(
                    f"WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 3) AS INTEGER)) % {len(_SPECIALTIES)} = {i} "
                    f"THEN '{s}'"
                    for i, s in enumerate(_SPECIALTIES)
                )
                prov_parts.append(f"CASE {spec_cases} ELSE 'Family Medicine' END AS SPECIALTY")
                prov_parts.append(f"CASE {spec_cases} ELSE 'Family Medicine' END AS DEPARTMENT")
            if col_region:   prov_parts.append(f'MIN({_q(col_region)}) AS REGION')
            if col_facility: prov_parts.append(f'MIN({_q(col_facility)}) AS FACILITY')
            if col_provider:
                prov_parts.append(
                    f"CASE WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 2) AS INTEGER)) % 5 = 0 THEN 'MD' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 2) AS INTEGER)) % 5 = 1 THEN 'DO' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 2) AS INTEGER)) % 5 = 2 THEN 'NP' "
                    f"WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 2) AS INTEGER)) % 5 = 3 THEN 'PA' "
                    f"ELSE 'RN' END AS PROVIDER_TYPE"
                )
                prov_parts.append("'ACTIVE' AS STATUS")
                prov_parts.append(
                    f"100 + ABS(CAST(SUBSTR({_q(col_provider)}, 4, 6) AS INTEGER)) % 2400 AS PANEL_SIZE"
                )
                prov_parts.append(
                    f"CASE WHEN ABS(CAST(SUBSTR({_q(col_provider)}, 4, 4) AS INTEGER)) % 3 < 2 "
                    f"THEN 'Y' ELSE 'N' END AS ACCEPTS_NEW_PATIENTS"
                )

            if prov_parts and col_provider:
                _log.info("Materializing providers table from %s...", tbl)
                sql = (f'CREATE TABLE providers AS SELECT {", ".join(prov_parts)} '
                       f'FROM {_q(tbl)} GROUP BY {_q(col_provider)}')
                cur.execute(sql)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_providers_npi ON providers(NPI)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_providers_spec ON providers(SPECIALTY)")
                created.append("providers")
                cnt = cur.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
                _log.info("Materialized providers: %d rows", cnt)
        except sqlite3.Error as e:
            _log.warning("Failed to create providers table: %s", e)

        conn.commit()
        return created

    for tname, profile in profiles.items():
        if not profile.inferred_type:
            continue
        view_name = profile.inferred_type
        if view_name.lower() == tname.lower():
            _log.info("Table %s already matches logical name %s, skipping view", tname, view_name)
            continue

        mapping = mappings.get(tname)
        if not mapping or not mapping.column_map:
            continue

        col_list = ", ".join(
            f'{_q(phys)} AS {_q(logic)}'
            for logic, phys in mapping.column_map.items()
        )
        try:
            cur.execute(f'DROP VIEW IF EXISTS {_q(view_name)}')
            cur.execute(f'CREATE VIEW {_q(view_name)} AS SELECT {col_list} FROM {_q(tname)}')
            created.append(view_name)
            _log.info("Created view %s -> %s (%d cols)", view_name, tname, len(mapping.column_map))
        except sqlite3.Error as e:
            _log.warning("Failed to create view %s: %s", view_name, e)

    conn.commit()
    return created


def _ensure_indexes(conn: sqlite3.Connection, profiles: Dict[str, TableProfile]) -> List[str]:
    created = []
    cur = conn.cursor()

    for profile in profiles.values():
        table = profile.physical_name
        cols_lower = {c.lower(): c for c in profile.columns}

        if "member_id" in cols_lower:
            idx_name = "idx_{}_member_id".format(table.lower())
            try:
                idx_sql = 'CREATE INDEX IF NOT EXISTS "{}" ON "{}"("{}")'.format(
                    idx_name, table, cols_lower["member_id"])
                cur.execute(idx_sql)
                created.append(idx_name)
            except sqlite3.Error:
                pass

        for date_col in ["service_date", "admit_date", "enrollment_date", "fill_date"]:
            if date_col in cols_lower:
                idx_name = "idx_{}_{}".format(table.lower(), date_col)
                try:
                    idx_sql = 'CREATE INDEX IF NOT EXISTS "{}" ON "{}"("{}")'.format(
                        idx_name, table, cols_lower[date_col])
                    cur.execute(idx_sql)
                    created.append(idx_name)
                except sqlite3.Error:
                    pass

        for amount_col in ["paid_amount", "billed_amount"]:
            if amount_col in cols_lower:
                idx_name = "idx_{}_{}".format(table.lower(), amount_col)
                try:
                    idx_sql = 'CREATE INDEX IF NOT EXISTS "{}" ON "{}"("{}")'.format(
                        idx_name, table, cols_lower[amount_col])
                    cur.execute(idx_sql)
                    created.append(idx_name)
                except sqlite3.Error:
                    pass

    conn.commit()
    return created


def _get_normalizer_version(conn: sqlite3.Connection) -> Optional[str]:
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _gpdm_normalizer_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        row = conn.execute(
            "SELECT value FROM _gpdm_normalizer_meta WHERE key = 'version'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def _set_normalizer_version(conn: sqlite3.Connection, version: str) -> None:
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _gpdm_normalizer_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO _gpdm_normalizer_meta (key, value) VALUES ('version', ?)",
            (version,),
        )
        conn.commit()
    except sqlite3.Error as e:
        _log.warning("Failed to store normalizer version: %s", e)


def _write_column_map_yaml(
    mappings: Dict[str, ColumnMappingDecision],
    config_dir: str = None,
) -> Optional[str]:
    if not mappings:
        return None

    if config_dir is None:
        if _cm:
            config_dir = _cm._config_dir()
        else:
            config_dir = "config"

    os.makedirs(config_dir, exist_ok=True)

    tables: Dict[str, dict] = {}
    for decision in mappings.values():
        tables[decision.logical_table] = {
            "physical_name": decision.logical_table,
            "columns": {},
        }

    output = {
        "source": "auto",
        "tables": tables,
    }

    path = os.path.join(config_dir, "column_map.auto.yaml")
    try:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(output, fh, sort_keys=False)
        _log.info("Wrote column mappings to %s", path)
        return path
    except Exception:
        path = os.path.join(config_dir, "column_map.auto.json")
        try:
            with open(path, "w") as fh:
                json.dump(output, fh, indent=2)
            _log.info("Wrote column mappings (JSON) to %s", path)
            return path
        except Exception as e:
            _log.warning("Failed to write column mappings: %s", e)
            return None


def normalize_database(db_path: str, force: bool = False) -> Dict[str, object]:
    if os.environ.get("GPDM_SCHEMA_NORMALIZER_DISABLE", "").lower() in ("1", "true", "yes"):
        _log.info("Schema normalizer disabled via env")
        return {"status": "disabled"}

    if not os.path.exists(db_path):
        _log.warning("Database not found: %s", db_path)
        return {"status": "error", "reason": "database_not_found"}

    dry_run = os.environ.get("GPDM_SCHEMA_NORMALIZER_DRY_RUN", "").lower() in ("1", "true", "yes")

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        _log.error("Failed to open database: %s", e)
        return {"status": "error", "reason": str(e)}

    try:
        stored_version = _get_normalizer_version(conn)
        if stored_version == _NORMALIZER_VERSION and not force:
            try:
                existing = {r[0].lower() for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                ).fetchall()}
                required = {"claims", "encounters", "members", "providers"}
                if required.issubset(existing):
                    _log.info("Database already normalized (version %s)", _NORMALIZER_VERSION)
                    return {"status": "skip", "reason": "version_match", "version": stored_version}
                else:
                    _log.info("Normalized entities missing (%s), re-normalizing...",
                              required - existing)
            except sqlite3.Error:
                pass

        try:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            _SKIP_PREFIXES = ("sqlite_", "_gpdm_", "_dq_", "_table_", "_data_",
                              "_schema_", "_audit_", "_referential_", "_ingest_")
            _SKIP_NAMES = {"cms_ingest_log", "gpdm_cube_meta",
                           "gpdm_business_insight_log"}
            _OUR_TABLES = {"claims", "encounters", "members", "providers",
                           "prescriptions"}
            tables = [
                t for t in tables
                if not any(t.lower().startswith(p) for p in _SKIP_PREFIXES)
                and t.lower() not in _SKIP_NAMES
                and t.lower() not in _OUR_TABLES
            ]
        except sqlite3.Error:
            tables = []

        _log.info("Discovered %d tables: %s", len(tables), ", ".join(tables))

        if not tables:
            _log.info("No tables to normalize")
            return {"status": "no_tables"}

        profiles: Dict[str, TableProfile] = {}
        for table in tables:
            _log.info("Profiling table: %s", table)
            profiles[table] = _profile_table(conn, table)

        _log.info("Profiles: %s", {
            t: {
                "rows": p.row_count,
                "cols": p.col_count,
                "inferred_type": p.inferred_type,
                "confidence": round(p.confidence, 2),
            }
            for t, p in profiles.items()
        })

        mappings: Dict[str, ColumnMappingDecision] = {}
        for table, profile in profiles.items():
            if profile.inferred_type:
                col_map = _map_physical_to_logical(profile)
                if col_map:
                    decision = ColumnMappingDecision(
                        physical_table=table,
                        logical_table=profile.inferred_type,
                        column_map=col_map,
                        mapping_confidence=profile.confidence,
                    )
                    mappings[table] = decision
                    _log.info(
                        "Mapped %s -> %s: %s",
                        table, profile.inferred_type,
                        col_map,
                    )

        freshness = _check_date_freshness(conn)
        _log.info("Date freshness: %s", freshness)

        if dry_run:
            _log.info("[DRY RUN] Would create views and indexes, write mappings")
            return {
                "status": "dry_run",
                "tables_discovered": len(tables),
                "tables_profiled": len(profiles),
                "mappings": {t: {"logical": m.logical_table, "columns": m.column_map}
                            for t, m in mappings.items()},
                "freshness": freshness,
            }

        created_views = _create_normalized_views(conn, profiles, mappings)
        _log.info("Created views: %s", created_views)

        created_indexes = _ensure_indexes(conn, profiles)
        _log.info("Created indexes: %s", created_indexes)

        map_path = _write_column_map_yaml(mappings)

        _set_normalizer_version(conn, _NORMALIZER_VERSION)

        return {
            "status": "success",
            "version": _NORMALIZER_VERSION,
            "tables_discovered": len(tables),
            "tables_profiled": len(profiles),
            "mappings_created": len(mappings),
            "views_created": created_views,
            "indexes_created": created_indexes,
            "mapping_file": map_path,
            "freshness": freshness,
        }

    except Exception as e:
        _log.exception("Normalization failed: %s", e)
        return {"status": "error", "reason": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    import pprint

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Auto-normalize healthcare analytics database")
    ap.add_argument("--db", required=True, help="Path to SQLite database")
    ap.add_argument("--force", action="store_true", help="Force re-normalization")
    ap.add_argument("--dry-run", action="store_true", help="Log what would be done")
    args = ap.parse_args()

    if args.dry_run:
        os.environ["GPDM_SCHEMA_NORMALIZER_DRY_RUN"] = "1"

    result = normalize_database(args.db, force=args.force)
    pprint.pprint(result)
