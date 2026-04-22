from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_db() -> str:
    return os.path.join(
        (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "healthcare_demo.db")


INGEST_LOG_TABLE = "gpdm_fhir_ingest_log"


def _ensure_tables(c: sqlite3.Connection) -> None:
    cur = c.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS members (
            MEMBER_ID TEXT PRIMARY KEY,
            FIRST_NAME TEXT, LAST_NAME TEXT,
            DATE_OF_BIRTH TEXT, GENDER TEXT,
            REGION TEXT, PLAN_TYPE TEXT,
            RISK_SCORE REAL, CHRONIC_CONDITIONS INTEGER,
            ENROLLMENT_DATE TEXT,
            ADDRESS TEXT, CITY TEXT, STATE TEXT, POSTAL_CODE TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS encounters (
            ENCOUNTER_ID TEXT PRIMARY KEY,
            MEMBER_ID TEXT, VISIT_TYPE TEXT,
            ADMIT_DATE TEXT, DISCHARGE_DATE TEXT, SERVICE_DATE TEXT,
            LENGTH_OF_STAY INTEGER, RENDERING_NPI TEXT,
            FACILITY TEXT, DISPOSITION TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            CLAIM_ID TEXT PRIMARY KEY,
            MEMBER_ID TEXT, ENCOUNTER_ID TEXT,
            SERVICE_DATE TEXT, CLAIM_STATUS TEXT,
            BILLED_AMOUNT REAL, PAID_AMOUNT REAL,
            CPT_CODE TEXT, DRG TEXT, RENDERING_NPI TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            DIAGNOSIS_ID TEXT PRIMARY KEY,
            MEMBER_ID TEXT, ENCOUNTER_ID TEXT,
            ICD10_CODE TEXT, ICD10_DESCRIPTION TEXT,
            DIAGNOSIS_DATE TEXT, IS_CHRONIC TEXT, RANK INTEGER
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            PRESCRIPTION_ID TEXT PRIMARY KEY,
            MEMBER_ID TEXT, ENCOUNTER_ID TEXT,
            RXNORM TEXT, MEDICATION_NAME TEXT, MEDICATION_CLASS TEXT,
            DAYS_SUPPLY INTEGER, COST REAL, FILL_DATE TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            NPI TEXT PRIMARY KEY,
            NAME TEXT, SPECIALTY TEXT, STATUS TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            OBSERVATION_ID TEXT PRIMARY KEY,
            MEMBER_ID TEXT, ENCOUNTER_ID TEXT,
            LOINC_CODE TEXT, DESCRIPTION TEXT,
            VALUE_NUMERIC REAL, VALUE_STRING TEXT, UNIT TEXT,
            OBSERVATION_DATE TEXT
        )""")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {INGEST_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            source TEXT,
            resources_total INTEGER,
            counts_json TEXT,
            errors INTEGER,
            elapsed_ms REAL
        )""")
    c.commit()


ICD10_SYS = ("http://hl7.org/fhir/sid/icd-10",
             "http://hl7.org/fhir/sid/icd-10-cm",
             "http://hl7.org/fhir/sid/icd-10-pcs")
CPT_SYS = ("http://www.ama-assn.org/go/cpt",)
RXNORM_SYS = ("http://www.nlm.nih.gov/research/umls/rxnorm",)
LOINC_SYS = ("http://loinc.org",)
SNOMED_SYS = ("http://snomed.info/sct",)
NPI_SYS = ("http://hl7.org/fhir/sid/us-npi",)


def _coding_match(codeable: Optional[Dict[str, Any]],
                  systems: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    if not codeable: return None
    for coding in codeable.get("coding", []) or []:
        sys = coding.get("system", "")
        if sys in systems:
            return coding.get("code", ""), coding.get("display", "") or ""
    codings = codeable.get("coding", []) or []
    if codings:
        return codings[0].get("code", ""), codings[0].get("display", "") or ""
    return None


def _text(codeable: Optional[Dict[str, Any]]) -> str:
    if not codeable: return ""
    return codeable.get("text", "") or ""


def _ref_id(ref: Any) -> Optional[str]:
    if ref is None: return None
    if isinstance(ref, str): return ref.split("/")[-1] or None
    if isinstance(ref, dict):
        r = ref.get("reference")
        if isinstance(r, str): return r.split("/")[-1] or None
        if ref.get("identifier", {}).get("value"):
            return ref["identifier"]["value"]
    return None


def _period_start(res: Dict[str, Any]) -> str:
    p = res.get("period") or {}
    return (p.get("start") or res.get("onsetDateTime") or res.get("effectiveDateTime")
            or res.get("recordedDate") or res.get("authoredOn") or "")[:10]


def _period_end(res: Dict[str, Any]) -> str:
    p = res.get("period") or {}
    return (p.get("end") or "")[:10]


def _write_patient(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    names = r.get("name") or [{}]
    first = " ".join((names[0].get("given") or [""]))
    last = names[0].get("family", "")
    address = (r.get("address") or [{}])[0]
    c.execute("""
        INSERT OR REPLACE INTO members
          (MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER,
           REGION, PLAN_TYPE, RISK_SCORE, CHRONIC_CONDITIONS,
           ENROLLMENT_DATE, ADDRESS, CITY, STATE, POSTAL_CODE)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rid, first, last, (r.get("birthDate") or "")[:10],
         (r.get("gender") or "").upper(),
         (address.get("state") or ""), "", None, None,
         "",
         " ".join(address.get("line") or []),
         address.get("city") or "",
         address.get("state") or "",
         address.get("postalCode") or ""))


def _write_encounter(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("subject"))
    cls = (r.get("class") or {}).get("code", "") or ""
    visit_map = {"EMER": "EMERGENCY", "IMP": "INPATIENT", "AMB": "OUTPATIENT",
                 "HH": "HOME_HEALTH", "VR": "TELEHEALTH",
                 "PRENC": "PREVENTIVE", "SS": "URGENT_CARE"}
    visit = visit_map.get(cls, cls or "OUTPATIENT")
    start = (r.get("period") or {}).get("start", "")[:10]
    end = (r.get("period") or {}).get("end", "")[:10]
    los = 0
    if start and end:
        try:
            from datetime import datetime
            los = max(0, (datetime.fromisoformat(end) -
                          datetime.fromisoformat(start)).days)
        except Exception:
            los = 0
    npi = ""
    for p in r.get("participant") or []:
        ind = p.get("individual") or {}
        for ident in (ind.get("identifier") and [ind["identifier"]] or []):
            if ident.get("system") in NPI_SYS:
                npi = ident.get("value", "") or ""; break
        if npi: break

    c.execute("""
        INSERT OR REPLACE INTO encounters
          (ENCOUNTER_ID, MEMBER_ID, VISIT_TYPE, ADMIT_DATE, DISCHARGE_DATE,
           SERVICE_DATE, LENGTH_OF_STAY, RENDERING_NPI, FACILITY, DISPOSITION)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (rid, patient, visit, start, end, start, los, npi,
         _text((r.get("serviceProvider") or {})),
         (r.get("hospitalization") or {}).get("dischargeDisposition", {}).get(
             "text", "")))


def _write_condition(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("subject"))
    enc = _ref_id(r.get("encounter"))
    m = _coding_match(r.get("code"), ICD10_SYS + SNOMED_SYS)
    code, disp = (m or ("", ""))
    onset = (r.get("onsetDateTime") or "")[:10] or _period_start(r)
    categories = [_text(cat) for cat in (r.get("category") or [])]
    is_chronic = "TRUE" if any("chronic" in (c_ or "").lower() for c_ in categories) else "FALSE"
    c.execute("""
        INSERT OR REPLACE INTO diagnoses
          (DIAGNOSIS_ID, MEMBER_ID, ENCOUNTER_ID, ICD10_CODE, ICD10_DESCRIPTION,
           DIAGNOSIS_DATE, IS_CHRONIC, RANK)
        VALUES (?,?,?,?,?,?,?,?)""",
        (rid, patient, enc, code, disp or _text(r.get("code")),
         onset, is_chronic, 1))


def _write_medication(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("subject"))
    enc = _ref_id(r.get("encounter") or r.get("context"))
    mc = r.get("medicationCodeableConcept") or {}
    m = _coding_match(mc, RXNORM_SYS)
    code, disp = (m or ("", _text(mc)))
    days_supply = 0
    dispense = r.get("dispenseRequest") or {}
    if dispense.get("expectedSupplyDuration", {}).get("unit") == "d":
        try:
            days_supply = int(dispense["expectedSupplyDuration"].get("value") or 0)
        except Exception:
            pass
    c.execute("""
        INSERT OR REPLACE INTO prescriptions
          (PRESCRIPTION_ID, MEMBER_ID, ENCOUNTER_ID, RXNORM, MEDICATION_NAME,
           MEDICATION_CLASS, DAYS_SUPPLY, COST, FILL_DATE)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (rid, patient, enc, code, disp, "", days_supply, 0.0,
         (r.get("authoredOn") or "")[:10]))


def _write_observation(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("subject"))
    enc = _ref_id(r.get("encounter"))
    m = _coding_match(r.get("code"), LOINC_SYS + SNOMED_SYS)
    code, disp = (m or ("", _text(r.get("code"))))
    vq = r.get("valueQuantity") or {}
    vnum = vq.get("value")
    unit = vq.get("unit", "") or vq.get("code", "") or ""
    vstr = r.get("valueString") or _text(r.get("valueCodeableConcept"))
    date = (r.get("effectiveDateTime") or "")[:10] or _period_start(r)
    c.execute("""
        INSERT OR REPLACE INTO observations
          (OBSERVATION_ID, MEMBER_ID, ENCOUNTER_ID, LOINC_CODE, DESCRIPTION,
           VALUE_NUMERIC, VALUE_STRING, UNIT, OBSERVATION_DATE)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (rid, patient, enc, code, disp,
         float(vnum) if vnum is not None else None,
         vstr, unit, date))


def _write_claim(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("patient"))
    enc = None
    for item in r.get("item") or []:
        enc = _ref_id((item.get("encounter") or [None])[0])
        if enc: break
    status = (r.get("status") or "active").upper()
    total = r.get("total") or {}
    billed = float(total.get("value") or 0.0)
    paid = 0.0
    for adj in r.get("adjudication") or []:
        if (adj.get("category") or {}).get("text", "").lower() in ("benefit", "eligible"):
            paid = float(adj.get("amount", {}).get("value") or paid)
    cpt = ""
    for item in r.get("item") or []:
        m = _coding_match(item.get("productOrService"), CPT_SYS)
        if m: cpt = m[0]; break
    c.execute("""
        INSERT OR REPLACE INTO claims
          (CLAIM_ID, MEMBER_ID, ENCOUNTER_ID, SERVICE_DATE, CLAIM_STATUS,
           BILLED_AMOUNT, PAID_AMOUNT, CPT_CODE, DRG, RENDERING_NPI)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (rid, patient, enc, (r.get("created") or "")[:10],
         status, billed, paid, cpt, "", ""))


def _write_eob(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    _write_claim(c, r)


def _write_procedure(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    patient = _ref_id(r.get("subject"))
    enc = _ref_id(r.get("encounter"))
    m = _coding_match(r.get("code"), CPT_SYS + SNOMED_SYS)
    code, disp = (m or ("", _text(r.get("code"))))
    date = _period_start(r)
    c.execute("""
        INSERT OR REPLACE INTO encounters
          (ENCOUNTER_ID, MEMBER_ID, VISIT_TYPE, ADMIT_DATE, DISCHARGE_DATE,
           SERVICE_DATE, LENGTH_OF_STAY, RENDERING_NPI, FACILITY, DISPOSITION)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (enc or rid, patient, "OUTPATIENT", date, date, date, 0, "", disp, code))


def _write_practitioner(c: sqlite3.Connection, r: Dict[str, Any]) -> None:
    rid = r.get("id") or str(uuid.uuid4())
    npi = rid
    for ident in r.get("identifier") or []:
        if ident.get("system") in NPI_SYS:
            npi = ident.get("value") or rid; break
    names = r.get("name") or [{}]
    name = " ".join(
        [(" ".join(names[0].get("given") or [])), names[0].get("family", "")]).strip()
    spec = ""
    for q in r.get("qualification") or []:
        spec = _text(q.get("code")) or spec
    c.execute("""INSERT OR REPLACE INTO providers
                 (NPI, NAME, SPECIALTY, STATUS) VALUES (?,?,?,?)""",
              (npi, name, spec, "ACTIVE"))


WRITERS: Dict[str, Callable[[sqlite3.Connection, Dict[str, Any]], None]] = {
    "Patient": _write_patient,
    "Encounter": _write_encounter,
    "Condition": _write_condition,
    "MedicationRequest": _write_medication,
    "MedicationStatement": _write_medication,
    "MedicationDispense": _write_medication,
    "Observation": _write_observation,
    "Claim": _write_claim,
    "ExplanationOfBenefit": _write_eob,
    "Procedure": _write_procedure,
    "Practitioner": _write_practitioner,
}


def _iter_bundle_resources(bundle: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if bundle.get("resourceType") == "Bundle":
        for e in bundle.get("entry") or []:
            r = e.get("resource")
            if r and isinstance(r, dict):
                yield r
    elif bundle.get("resourceType"):
        yield bundle


def ingest_bundle_dict(bundle: Dict[str, Any], db_path: Optional[str] = None,
                       source: str = "memory") -> Dict[str, Any]:
    db_path = db_path or _default_db()
    t0 = time.time()
    counts: Dict[str, int] = {}
    errors = 0
    total = 0
    with sqlite3.connect(db_path, timeout=30) as c:
        _ensure_tables(c)
        for res in _iter_bundle_resources(bundle):
            total += 1
            rt = res.get("resourceType", "")
            fn = WRITERS.get(rt)
            if not fn:
                continue
            try:
                fn(c, res)
                counts[rt] = counts.get(rt, 0) + 1
            except Exception as e:
                errors += 1
                logger.debug("write %s failed: %s", rt, e)
        c.execute(
            f"INSERT INTO {INGEST_LOG_TABLE} "
            f"(ts, source, resources_total, counts_json, errors, elapsed_ms) "
            f"VALUES (?,?,?,?,?,?)",
            (time.time(), source, total, json.dumps(counts), errors,
             (time.time() - t0) * 1000))
        c.commit()
    return {
        "resources_total": total,
        "written": counts,
        "errors": errors,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "db_path": db_path,
    }


def ingest_path(path: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    db_path = db_path or _default_db()
    if not os.path.exists(path):
        return {"error": "path_not_found", "path": path}

    files: List[str] = []
    if os.path.isdir(path):
        for root, _, fn in os.walk(path):
            for f in fn:
                if f.lower().endswith((".json", ".ndjson", ".jsonl")):
                    files.append(os.path.join(root, f))
    else:
        files = [path]

    totals = {"resources_total": 0, "written": {}, "errors": 0,
              "files_processed": 0, "files_failed": 0}
    for fp in files:
        try:
            lower = fp.lower()
            if lower.endswith((".ndjson", ".jsonl")):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            res = json.loads(line)
                        except Exception:
                            totals["errors"] += 1; continue
                        r = ingest_bundle_dict(
                            {"resourceType": res.get("resourceType", ""), **res
                             } if res.get("resourceType") != "Bundle" else res,
                            db_path=db_path, source=fp)
                        totals["resources_total"] += r["resources_total"]
                        for k, v in r["written"].items():
                            totals["written"][k] = totals["written"].get(k, 0) + v
                        totals["errors"] += r["errors"]
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                r = ingest_bundle_dict(bundle, db_path=db_path, source=fp)
                totals["resources_total"] += r["resources_total"]
                for k, v in r["written"].items():
                    totals["written"][k] = totals["written"].get(k, 0) + v
                totals["errors"] += r["errors"]
            totals["files_processed"] += 1
        except Exception as e:
            logger.warning("ingest %s failed: %s", fp, e)
            totals["files_failed"] += 1
    return totals


def get_ingestion_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    db_path = db_path or _default_db()
    try:
        with sqlite3.connect(db_path) as c:
            _ensure_tables(c)
            rows = list(c.execute(
                f"SELECT ts, source, resources_total, counts_json, errors, elapsed_ms "
                f"FROM {INGEST_LOG_TABLE} ORDER BY ts DESC LIMIT 20"))
        return {
            "recent_runs": [
                {"ts": r[0], "source": r[1], "resources": r[2],
                 "counts": json.loads(r[3] or "{}"), "errors": r[4],
                 "elapsed_ms": r[5]}
                for r in rows
            ],
            "n_runs_logged": len(rows),
        }
    except Exception as e:
        return {"error": str(e)}
