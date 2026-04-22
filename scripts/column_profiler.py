from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from . import column_map as _cm
except Exception:
    import column_map as _cm

_log = logging.getLogger("gpdm.column_profiler")


_ICD10   = re.compile(r"^[A-TV-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")
_NPI     = re.compile(r"^\d{10}$")
_NDC     = re.compile(r"^\d{4,5}-\d{3,4}-\d{1,2}$")
_DATE_1  = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DATE_2  = re.compile(r"^\d{2}/\d{2}/\d{4}")
_DATE_3  = re.compile(r"^\d{8}$")
_MONEY   = re.compile(r"^-?\$?\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?$")
_ID_LIKE = re.compile(r"^[A-Z0-9][A-Z0-9_\-]{2,}$", re.IGNORECASE)


def _classify_value(v: str) -> str:
    if v is None or v == "":
        return "null"
    s = str(v).strip()
    if _ICD10.match(s):   return "icd10"
    if _NPI.match(s):     return "npi"
    if _NDC.match(s):     return "ndc"
    if _DATE_1.match(s) or _DATE_2.match(s) or _DATE_3.match(s): return "date"
    if _MONEY.match(s):   return "money"
    try:
        float(s.replace(",", ""))
        return "numeric"
    except ValueError:
        pass
    if _ID_LIKE.match(s) and any(c.isdigit() for c in s):
        return "id"
    return "text"


@dataclass
class ColumnProfile:
    table:         str
    column:        str
    row_count:     int = 0
    null_rate:     float = 0.0
    distinct_rate: float = 0.0
    dominant_kind: str = "text"
    kind_mix:      Dict[str, float] = field(default_factory=dict)
    sample:        List[str]        = field(default_factory=list)
    proposed_logical: Optional[str] = None
    confidence:       float = 0.0
    reason:           str = ""

    def to_dict(self) -> dict:
        return {
            "column":        self.column,
            "row_count":     self.row_count,
            "null_rate":     round(self.null_rate, 3),
            "distinct_rate": round(self.distinct_rate, 3),
            "dominant_kind": self.dominant_kind,
            "kind_mix":      {k: round(v, 2) for k, v in self.kind_mix.items()},
            "sample":        self.sample[:4],
            "proposed_logical": self.proposed_logical,
            "confidence":       round(self.confidence, 3),
            "reason":           self.reason,
        }


_SPLIT = re.compile(r"[_\-]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokens(name: str) -> List[str]:
    return [t.lower() for t in _SPLIT.split(name) if t]


def _name_similarity(phys: str, logical_variants: List[str]) -> float:
    a = set(_tokens(phys))
    best = 0.0
    for v in logical_variants:
        b = set(_tokens(v))
        if not (a | b):
            continue
        jac = len(a & b) / len(a | b)
        bonus = 0.0
        if phys.lower() in v.lower() or v.lower() in phys.lower():
            bonus = 0.2
        best = max(best, min(1.0, jac + bonus))
    return best


_KIND_FIT: Dict[str, set] = {
    "date":    {"service_date", "admit_date", "discharge_date",
                 "submitted_date", "adjudicated_date", "dob",
                 "enrollment_date", "fill_date"},
    "money":   {"paid_amount", "billed_amount", "total_paid"},
    "numeric": {"paid_amount", "billed_amount", "total_paid",
                 "length_of_stay", "days_supply"},
    "id":      {"member_id", "claim_id", "encounter_id", "provider_id",
                 "facility_id", "rx_id"},
    "icd10":   {"diagnosis"},
    "npi":     {"provider_id"},
    "ndc":     {"ndc"},
    "text":    {"gender", "region", "plan", "visit_type", "specialty",
                 "claim_status", "network_status", "name"},
}


def _score_mapping(phys_col: str, profile: ColumnProfile,
                   logical_table: str) -> Tuple[Optional[str], float, str]:
    aliases = _cm._DEFAULT_ALIASES.get(logical_table, {})
    if not aliases:
        return None, 0.0, "unknown logical table"

    candidates = _KIND_FIT.get(profile.dominant_kind, set()) & set(aliases.keys())
    if not candidates:
        candidates = set(aliases.keys())

    best_logical: Optional[str] = None
    best_score = 0.0
    best_reason = ""
    for logical_col in candidates:
        variants = [logical_col] + aliases.get(logical_col, [])
        name_sim = _name_similarity(phys_col, variants)
        kind_bonus = 0.3 if logical_col in _KIND_FIT.get(
            profile.dominant_kind, set()) else 0.0
        uniq_bonus = 0.0
        if logical_col.endswith("_id") and profile.distinct_rate > 0.9:
            uniq_bonus = 0.15
        score = min(1.0, 0.7 * name_sim + kind_bonus + uniq_bonus)
        if score > best_score:
            best_score = score
            best_logical = logical_col
            best_reason = (f"name_sim={name_sim:.2f} kind={profile.dominant_kind} "
                           f"+kind_bonus={kind_bonus} +uniq={uniq_bonus}")
    return best_logical, best_score, best_reason


def _logical_table_for(phys_table: str) -> Optional[str]:
    ph = phys_table.lower()
    for logical, aliases in _cm._DEFAULT_TABLE_ALIASES.items():
        for a in aliases:
            if a.lower() == ph:
                return logical
    phys_tokens = set(_tokens(phys_table))
    best = (None, 0.0)
    for logical, aliases in _cm._DEFAULT_TABLE_ALIASES.items():
        for a in aliases:
            sim = _name_similarity(phys_table, [a])
            if sim > best[1]:
                best = (logical, sim)
    return best[0] if best[1] >= 0.4 else None


def profile_database(conn: sqlite3.Connection,
                      sample: Optional[int] = None) -> Dict[str, List[ColumnProfile]]:
    sample = sample or int(os.environ.get("GPDM_PROFILER_SAMPLE", "1000"))
    out: Dict[str, List[ColumnProfile]] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        " AND name NOT LIKE 'sqlite_%'").fetchall()]
    for t in tables:
        try:
            cols = [r[1] for r in conn.execute(
                f'PRAGMA table_info("{t}")').fetchall()]
        except sqlite3.Error:
            continue
        try:
            total = conn.execute(
                f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error:
            total = 0
        profiles: List[ColumnProfile] = []
        for c in cols:
            try:
                rows = conn.execute(
                    f'SELECT "{c}" FROM "{t}" LIMIT ?', (sample,)).fetchall()
            except sqlite3.Error:
                continue
            vals = [r[0] for r in rows]
            n = len(vals)
            nulls = sum(1 for v in vals if v is None or v == "")
            non_null = [v for v in vals if v is not None and v != ""]
            distinct = len(set(str(v) for v in non_null))
            kinds: Dict[str, int] = {}
            for v in non_null:
                k = _classify_value(v)
                kinds[k] = kinds.get(k, 0) + 1
            if kinds:
                dominant = max(kinds.items(), key=lambda kv: kv[1])[0]
                mix = {k: v / max(1, len(non_null)) for k, v in kinds.items()}
            else:
                dominant = "text"; mix = {}
            prof = ColumnProfile(
                table=t, column=c, row_count=total or n,
                null_rate=nulls / n if n else 0.0,
                distinct_rate=distinct / len(non_null) if non_null else 0.0,
                dominant_kind=dominant, kind_mix=mix,
                sample=[str(v) for v in non_null[:4]],
            )
            profiles.append(prof)
        out[t] = profiles
    return out


def propose_mapping(profiles: Dict[str, List[ColumnProfile]],
                    threshold: Optional[float] = None) -> Dict[str, dict]:
    threshold = threshold if threshold is not None else float(
        os.environ.get("GPDM_PROFILER_THRESHOLD", "0.45"))

    out_tables: Dict[str, dict] = {}
    for phys_table, col_profiles in profiles.items():
        logical_table = _logical_table_for(phys_table)
        if not logical_table:
            continue
        entry = out_tables.setdefault(logical_table, {
            "physical_name": phys_table,
            "columns": {},
        })
        best_per_logical: Dict[str, Tuple[str, float]] = {}
        for p in col_profiles:
            logical, score, reason = _score_mapping(p.column, p, logical_table)
            p.proposed_logical = logical
            p.confidence = score
            p.reason = reason
            if logical and score >= threshold:
                prev = best_per_logical.get(logical)
                if not prev or score > prev[1]:
                    best_per_logical[logical] = (p.column, score)
        for logical_col, (phys_col, _score) in best_per_logical.items():
            entry["columns"][logical_col] = phys_col
    return out_tables


def write_draft(tables: Dict[str, dict],
                source: str,
                config_dir: Optional[str] = None) -> str:
    d = config_dir or _cm._config_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"column_map.{source}.draft.yaml")
    try:
        import yaml
        body = yaml.safe_dump({"source": source, "tables": tables},
                              sort_keys=False)
    except Exception:
        body = json.dumps({"source": source, "tables": tables}, indent=2)
        path = path.replace(".yaml", ".json")
    with open(path, "w") as fh:
        fh.write(body)
    return path


def run(db_path: str, source: str = "unknown") -> dict:
    conn = sqlite3.connect(db_path)
    try:
        profiles = profile_database(conn)
        tables = propose_mapping(profiles)
        draft_path = write_draft(tables, source=source)
        return {
            "db_path":    db_path,
            "source":     source,
            "draft_path": draft_path,
            "profiles":   {t: [p.to_dict() for p in plist]
                           for t, plist in profiles.items()},
            "mapping":    tables,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    import pprint
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--source", default="unknown")
    args = ap.parse_args()
    result = run(args.db, args.source)
    pprint.pprint({k: v for k, v in result.items() if k != "profiles"})
    print("\nDraft written to:", result["draft_path"])
