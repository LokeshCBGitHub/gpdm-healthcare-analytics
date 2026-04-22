import os
import sqlite3
import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')


STAR_SCHEMA = {
    'dimensions': {
        'dim_member': {
            'source': 'members',
            'grain': 'One row per member',
            'business_key': 'MEMBER_ID',
            'scd_type': 2,
            'columns': {
                'MEMBER_ID': {'type': 'TEXT', 'nullable': False, 'role': 'business_key'},
                'MRN': {'type': 'TEXT', 'nullable': False, 'role': 'natural_key'},
                'FIRST_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'LAST_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'DATE_OF_BIRTH': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'GENDER': {'type': 'TEXT', 'nullable': False, 'role': 'attribute', 'valid_values': ['M', 'F']},
                'RACE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'LANGUAGE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'ADDRESS': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'CITY': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'STATE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'ZIP_CODE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'PHONE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'EMAIL': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'REGION': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'FACILITY': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'PLAN_TYPE': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'ENROLLMENT_DATE': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'DISENROLLMENT_DATE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'PCP_NPI': {'type': 'TEXT', 'nullable': True, 'role': 'foreign_key', 'references': 'providers.NPI'},
                'RISK_SCORE': {'type': 'REAL', 'nullable': True, 'role': 'measure'},
                'CHRONIC_CONDITIONS': {'type': 'INTEGER', 'nullable': True, 'role': 'measure'},
            },
        },
        'dim_provider': {
            'source': 'providers',
            'grain': 'One row per provider NPI',
            'business_key': 'NPI',
            'scd_type': 1,
            'columns': {
                'NPI': {'type': 'TEXT', 'nullable': False, 'role': 'business_key'},
                'PROVIDER_FIRST_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'PROVIDER_LAST_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'SPECIALTY': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'DEPARTMENT': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'REGION': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'FACILITY': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'PROVIDER_TYPE': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'DEA_NUMBER': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'LICENSE_STATE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'HIRE_DATE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'STATUS': {'type': 'TEXT', 'nullable': False, 'role': 'attribute', 'valid_values': ['Active', 'Inactive', 'On Leave']},
                'PANEL_SIZE': {'type': 'INTEGER', 'nullable': True, 'role': 'measure'},
                'ACCEPTS_NEW_PATIENTS': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
            },
        },
        'dim_diagnosis': {
            'source': 'diagnoses',
            'grain': 'One row per diagnosis event',
            'business_key': 'DIAGNOSIS_ID',
            'columns': {
                'DIAGNOSIS_ID': {'type': 'TEXT', 'nullable': False, 'role': 'business_key'},
                'ICD10_CODE': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'ICD10_DESCRIPTION': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'DIAGNOSIS_TYPE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'IS_CHRONIC': {'type': 'TEXT', 'nullable': True, 'role': 'attribute', 'valid_values': ['Y', 'N']},
                'SEVERITY': {'type': 'REAL', 'nullable': True, 'role': 'measure'},
                'HCC_CODE': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'HCC_CATEGORY': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
            },
        },
        'dim_cpt': {
            'source': 'cpt_codes',
            'grain': 'One row per CPT code',
            'business_key': 'CPT_CODE',
            'columns': {
                'CPT_CODE': {'type': 'TEXT', 'nullable': False, 'role': 'business_key'},
                'DESCRIPTION': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'CATEGORY': {'type': 'TEXT', 'nullable': True, 'role': 'attribute'},
                'RVU': {'type': 'REAL', 'nullable': True, 'role': 'measure'},
            },
        },
        'dim_date': {
            'source': 'derived',
            'grain': 'One row per calendar date',
            'business_key': 'DATE_KEY',
            'columns': {
                'DATE_KEY': {'type': 'TEXT', 'nullable': False, 'role': 'business_key'},
                'YEAR': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'QUARTER': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'MONTH': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'MONTH_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'DAY_OF_WEEK': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'DAY_NAME': {'type': 'TEXT', 'nullable': False, 'role': 'attribute'},
                'IS_WEEKEND': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'FISCAL_YEAR': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
                'FISCAL_QUARTER': {'type': 'INTEGER', 'nullable': False, 'role': 'attribute'},
            },
        },
    },
    'facts': {
        'fact_claim': {
            'source': 'claims',
            'grain': 'One row per claim line',
            'business_key': 'CLAIM_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'ENCOUNTER_ID': 'encounters.ENCOUNTER_ID',
                'RENDERING_NPI': 'providers.NPI',
                'BILLING_NPI': 'providers.NPI',
            },
            'measures': ['BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'PAID_AMOUNT',
                         'MEMBER_RESPONSIBILITY', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE'],
            'degenerate_dims': ['CLAIM_STATUS', 'DENIAL_REASON', 'CLAIM_TYPE', 'PLAN_TYPE', 'REGION', 'FACILITY'],
        },
        'fact_encounter': {
            'source': 'encounters',
            'grain': 'One row per encounter',
            'business_key': 'ENCOUNTER_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'RENDERING_NPI': 'providers.NPI',
                'SUPERVISING_NPI': 'providers.NPI',
            },
            'measures': ['LENGTH_OF_STAY'],
            'degenerate_dims': ['VISIT_TYPE', 'DEPARTMENT', 'FACILITY', 'REGION',
                                'DISPOSITION', 'ENCOUNTER_STATUS'],
        },
        'fact_prescription': {
            'source': 'prescriptions',
            'grain': 'One row per prescription',
            'business_key': 'RX_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'PRESCRIBING_NPI': 'providers.NPI',
            },
            'measures': ['COST', 'COPAY', 'QUANTITY', 'DAYS_SUPPLY', 'REFILLS_AUTHORIZED', 'REFILLS_USED'],
            'degenerate_dims': ['MEDICATION_NAME', 'MEDICATION_CLASS', 'STATUS', 'PHARMACY'],
        },
        'fact_referral': {
            'source': 'referrals',
            'grain': 'One row per referral',
            'business_key': 'REFERRAL_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'REFERRING_NPI': 'providers.NPI',
                'REFERRED_TO_NPI': 'providers.NPI',
            },
            'measures': [],
            'degenerate_dims': ['URGENCY', 'REFERRAL_TYPE', 'STATUS', 'SPECIALTY'],
        },
        'fact_appointment': {
            'source': 'appointments',
            'grain': 'One row per appointment',
            'business_key': 'APPOINTMENT_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'PROVIDER_NPI': 'providers.NPI',
            },
            'measures': ['DURATION_MINUTES'],
            'degenerate_dims': ['APPOINTMENT_TYPE', 'DEPARTMENT', 'FACILITY', 'REGION', 'STATUS', 'REASON'],
        },
        'fact_diagnosis': {
            'source': 'diagnoses',
            'grain': 'One row per diagnosis event',
            'business_key': 'DIAGNOSIS_ID',
            'foreign_keys': {
                'MEMBER_ID': 'members.MEMBER_ID',
                'ENCOUNTER_ID': 'encounters.ENCOUNTER_ID',
                'DIAGNOSING_NPI': 'providers.NPI',
            },
            'measures': ['SEVERITY'],
            'degenerate_dims': ['ICD10_CODE', 'ICD10_DESCRIPTION', 'DIAGNOSIS_TYPE', 'IS_CHRONIC', 'HCC_CODE', 'HCC_CATEGORY'],
        },
    },
    'conformed_dimensions': {
        'MEMBER_ID': 'dim_member',
        'NPI': 'dim_provider',
        'RENDERING_NPI': 'dim_provider',
        'BILLING_NPI': 'dim_provider',
        'PRESCRIBING_NPI': 'dim_provider',
        'REFERRING_NPI': 'dim_provider',
        'REFERRED_TO_NPI': 'dim_provider',
        'SUPERVISING_NPI': 'dim_provider',
        'DIAGNOSING_NPI': 'dim_provider',
        'PROVIDER_NPI': 'dim_provider',
        'PCP_NPI': 'dim_provider',
        'CPT_CODE': 'dim_cpt',
    },
}


DQ_RULES = {
    'referential_integrity': [
        {
            'name': 'claims_member_exists',
            'description': 'Every claim must reference a valid member',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM claims_4m c WHERE c.member_id NOT IN (SELECT MEMBER_ID FROM members) AND c.member_id IS NOT NULL AND c.member_id != ''",
            'threshold': 0,
        },
        {
            'name': 'encounters_member_exists',
            'description': 'Every encounter must reference a valid member',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM encounters e WHERE e.MEMBER_ID NOT IN (SELECT MEMBER_ID FROM members) AND e.MEMBER_ID IS NOT NULL AND e.MEMBER_ID != ''",
            'threshold': 0,
        },
        {
            'name': 'claims_provider_exists',
            'description': 'Claims provider should exist in providers table',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM claims_4m c WHERE c.provider_id NOT IN (SELECT NPI FROM providers) AND c.provider_id IS NOT NULL AND c.provider_id != ''",
            'threshold': 0,
        },
        {
            'name': 'encounters_provider_exists',
            'description': 'Encounter provider should exist in providers table',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM encounters e WHERE e.RENDERING_NPI NOT IN (SELECT NPI FROM providers) AND e.RENDERING_NPI IS NOT NULL AND e.RENDERING_NPI != ''",
            'threshold': 0,
        },
        {
            'name': 'members_pcp_exists',
            'description': 'Member PCP should exist in providers table',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM members m WHERE m.PCP_NPI NOT IN (SELECT NPI FROM providers) AND m.PCP_NPI IS NOT NULL AND m.PCP_NPI != ''",
            'threshold': 0,
        },
    ],
    'uniqueness': [
        {
            'name': 'claim_id_unique',
            'description': 'claim_id must be unique in claims',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM (SELECT claim_id FROM claims_4m GROUP BY claim_id HAVING COUNT(*) > 1)",
            'threshold': 0,
        },
        {
            'name': 'encounter_id_unique',
            'description': 'ENCOUNTER_ID must be unique in encounters',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM (SELECT ENCOUNTER_ID FROM encounters GROUP BY ENCOUNTER_ID HAVING COUNT(*) > 1)",
            'threshold': 0,
        },
        {
            'name': 'member_id_unique',
            'description': 'MEMBER_ID must be unique in members',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM (SELECT MEMBER_ID FROM members GROUP BY MEMBER_ID HAVING COUNT(*) > 1)",
            'threshold': 0,
        },
        {
            'name': 'npi_unique',
            'description': 'NPI must be unique in providers',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM (SELECT NPI FROM providers GROUP BY NPI HAVING COUNT(*) > 1)",
            'threshold': 0,
        },
    ],
    'completeness': [
        {
            'name': 'claims_member_id_not_null',
            'description': 'member_id in claims cannot be null',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE member_id IS NULL OR member_id = ''",
            'threshold': 0,
        },
        {
            'name': 'claims_billed_amount_not_null',
            'description': 'billed_amount in claims cannot be null',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE billed_amount IS NULL OR billed_amount = ''",
            'threshold': 0,
        },
        {
            'name': 'claims_service_date_not_null',
            'description': 'service_date in claims cannot be null',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE service_date IS NULL OR service_date = ''",
            'threshold': 0,
        },
        {
            'name': 'encounters_service_date_not_null',
            'description': 'SERVICE_DATE in encounters cannot be null',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM encounters WHERE SERVICE_DATE IS NULL OR SERVICE_DATE = ''",
            'threshold': 0,
        },
    ],
    'validity': [
        {
            'name': 'claims_status_valid',
            'description': 'Claim status must be a known value',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE UPPER(status) NOT IN ('APPROVED','DENIED','PENDING','ADJUSTED','APPEALED','VOIDED','PAID') AND status IS NOT NULL AND status != ''",
            'threshold': 0,
        },
        {
            'name': 'encounters_visit_type_valid',
            'description': 'Visit type must be a known value',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM encounters WHERE UPPER(VISIT_TYPE) NOT IN ('OUTPATIENT','TELEHEALTH','INPATIENT','URGENT_CARE','EMERGENCY','HOME_HEALTH','PREVENTIVE','ER','RX','OBSERVATION','REHAB','SNF') AND VISIT_TYPE IS NOT NULL AND VISIT_TYPE != ''",
            'threshold': 0,
        },
        {
            'name': 'claims_no_negative_amounts',
            'description': 'Financial amounts cannot be negative',
            'severity': 'critical',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE CAST(billed_amount AS REAL) < 0 OR CAST(paid_amount AS REAL) < 0",
            'threshold': 0,
        },
        {
            'name': 'encounters_discharge_after_admit',
            'description': 'Discharge date must be on or after admit date',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM encounters WHERE DISCHARGE_DATE < ADMIT_DATE AND DISCHARGE_DATE IS NOT NULL AND DISCHARGE_DATE != '' AND ADMIT_DATE IS NOT NULL AND ADMIT_DATE != ''",
            'threshold': 0,
        },
        {
            'name': 'members_gender_valid',
            'description': 'Gender must be M or F',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM members WHERE UPPER(GENDER) NOT IN ('M', 'F') AND GENDER IS NOT NULL AND GENDER != ''",
            'threshold': 0,
        },
        {
            'name': 'claims_paid_lte_billed',
            'description': 'Paid amount should not exceed billed amount by more than 10%',
            'severity': 'warning',
            'sql': "SELECT COUNT(*) FROM claims_4m WHERE CAST(paid_amount AS REAL) > CAST(billed_amount AS REAL) * 1.1 AND paid_amount IS NOT NULL AND billed_amount IS NOT NULL AND CAST(paid_amount AS REAL) > 0",
            'threshold': 0,
        },
    ],
    'consistency': [
        {
            'name': 'member_region_consistency',
            'description': 'Member region in claims vs member table (cross-region visits are normal)',
            'severity': 'info',
            'sql': (
                "SELECT ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM claims_4m), 0), 0) "
                "FROM claims_4m c JOIN members m ON c.member_id = m.MEMBER_ID "
                "WHERE c.region != m.REGION AND c.region IS NOT NULL AND m.REGION IS NOT NULL AND c.region != '' AND m.REGION != ''"
            ),
            'threshold': 90,
        },
    ],
}


REQUIRED_INDEXES = [
    ('idx_gov_c4m_member',     'claims_4m', 'member_id'),
    ('idx_gov_c4m_provider',   'claims_4m', 'provider_id'),
    ('idx_gov_c4m_svcdate',    'claims_4m', 'service_date'),
    ('idx_gov_c4m_status',     'claims_4m', 'status'),
    ('idx_gov_c4m_region',     'claims_4m', 'region'),
    ('idx_gov_c4m_facility',   'claims_4m', 'facility_id'),
    ('idx_gov_c4m_diagnosis',  'claims_4m', 'diagnosis_code'),
    ('idx_gov_mem_id',         'members', 'MEMBER_ID'),
    ('idx_gov_mem_region',     'members', 'REGION'),
    ('idx_gov_mem_plan',       'members', 'PLAN_TYPE'),
    ('idx_gov_mem_pcp',        'members', 'PCP_NPI'),
    ('idx_gov_prov_npi',       'providers', 'NPI'),
    ('idx_gov_prov_specialty', 'providers', 'SPECIALTY'),
    ('idx_gov_prov_region',    'providers', 'REGION'),
]


class DataGovernanceEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def ensure_metadata_tables(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS _dq_run_log (
                run_id TEXT PRIMARY KEY,
                run_timestamp TEXT NOT NULL,
                total_rules INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                warnings INTEGER NOT NULL,
                overall_score REAL NOT NULL,
                duration_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS _dq_rule_results (
                run_id TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                violation_count INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (run_id, rule_name),
                FOREIGN KEY (run_id) REFERENCES _dq_run_log(run_id)
            );

            CREATE TABLE IF NOT EXISTS _table_metadata (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                data_type TEXT,
                is_nullable TEXT,
                is_primary_key TEXT,
                references_table TEXT,
                references_column TEXT,
                star_schema_role TEXT,
                description TEXT,
                last_profiled TEXT,
                distinct_count INTEGER,
                null_count INTEGER,
                min_value TEXT,
                max_value TEXT,
                PRIMARY KEY (table_name, column_name)
            );

            CREATE TABLE IF NOT EXISTS _data_lineage (
                lineage_id TEXT PRIMARY KEY,
                source_table TEXT NOT NULL,
                target_table TEXT,
                operation TEXT NOT NULL,
                row_count INTEGER,
                timestamp TEXT NOT NULL,
                user_session TEXT,
                checksum TEXT,
                details TEXT
            );

            CREATE TABLE IF NOT EXISTS _schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS _audit_trail (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                action TEXT NOT NULL,
                table_name TEXT,
                record_id TEXT,
                old_values TEXT,
                new_values TEXT,
                user_session TEXT,
                ip_address TEXT
            );

            CREATE TABLE IF NOT EXISTS _referential_integrity_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                parent_table TEXT NOT NULL,
                child_table TEXT NOT NULL,
                fk_column TEXT NOT NULL,
                orphan_count INTEGER NOT NULL,
                sample_orphan_ids TEXT,
                remediation TEXT
            );
        """)
        conn.commit()
        log.info("Metadata tables created/verified")

    def create_indexes(self) -> Dict[str, int]:
        conn = self._get_conn()
        created = 0
        skipped = 0
        for idx_name, table, column in REQUIRED_INDEXES:
            try:
                existing = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,)
                ).fetchone()
                if existing:
                    skipped += 1
                    continue
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
                created += 1
            except Exception as e:
                log.warning("Failed to create index %s on %s.%s: %s", idx_name, table, column, e)
        conn.commit()
        log.info("Indexes: %d created, %d already existed", created, skipped)
        return {'created': created, 'skipped': skipped, 'total': len(REQUIRED_INDEXES)}

    def run_dq_checks(self) -> Dict[str, Any]:
        conn = self._get_conn()
        t0 = datetime.now(timezone.utc)
        run_id = hashlib.sha256(t0.isoformat().encode()).hexdigest()[:16]

        results = []
        passed = 0
        failed = 0
        warnings = 0

        for category, rules in DQ_RULES.items():
            for rule in rules:
                try:
                    row = conn.execute(rule['sql']).fetchone()
                    violation_count = row[0] if row else 0
                    status = 'PASS' if violation_count <= rule['threshold'] else 'FAIL'
                    if status == 'FAIL':
                        if rule['severity'] == 'critical':
                            failed += 1
                        elif rule['severity'] == 'warning':
                            warnings += 1
                        else:
                            warnings += 1
                    else:
                        passed += 1
                    results.append({
                        'rule_name': rule['name'],
                        'category': category,
                        'severity': rule['severity'],
                        'description': rule['description'],
                        'violation_count': violation_count,
                        'threshold': rule['threshold'],
                        'status': status,
                    })
                except Exception as e:
                    log.warning("DQ rule %s failed: %s", rule['name'], e)
                    results.append({
                        'rule_name': rule['name'],
                        'category': category,
                        'severity': rule['severity'],
                        'description': rule['description'],
                        'violation_count': -1,
                        'threshold': rule['threshold'],
                        'status': 'ERROR',
                    })
                    failed += 1

        duration_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        total_rules = len(results)
        overall_score = round(100.0 * passed / total_rules, 1) if total_rules > 0 else 0

        conn.execute(
            "INSERT OR REPLACE INTO _dq_run_log VALUES (?,?,?,?,?,?,?,?)",
            (run_id, t0.isoformat(), total_rules, passed, failed, warnings,
             overall_score, duration_ms)
        )
        for r in results:
            conn.execute(
                "INSERT OR REPLACE INTO _dq_rule_results VALUES (?,?,?,?,?,?,?,?)",
                (run_id, r['rule_name'], r['category'], r['severity'],
                 r['description'], r['violation_count'], r['threshold'], r['status'])
            )
        conn.commit()

        return {
            'run_id': run_id,
            'timestamp': t0.isoformat(),
            'total_rules': total_rules,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'overall_score': overall_score,
            'duration_ms': duration_ms,
            'results': results,
        }

    def fix_referential_integrity(self) -> Dict[str, Any]:
        conn = self._get_conn()
        fixes = {}

        try:
            dup_claims = conn.execute(
                "SELECT claim_id, COUNT(*) as cnt FROM claims_4m GROUP BY claim_id HAVING cnt > 1"
            ).fetchall()
            if dup_claims:
                for claim_id, cnt in dup_claims:
                    conn.execute(
                        "DELETE FROM claims_4m WHERE rowid NOT IN "
                        "(SELECT MIN(rowid) FROM claims_4m WHERE claim_id = ?) AND claim_id = ?",
                        (claim_id, claim_id)
                    )
                fixes['duplicate_claims_removed'] = len(dup_claims)
        except Exception as e:
            log.warning("Claims dedup failed: %s", e)

        try:
            dup_encounters = conn.execute(
                "SELECT ENCOUNTER_ID, COUNT(*) as cnt FROM encounters GROUP BY ENCOUNTER_ID HAVING cnt > 1"
            ).fetchall()
            if dup_encounters:
                for enc_id, cnt in dup_encounters:
                    conn.execute(
                        "DELETE FROM encounters WHERE rowid NOT IN "
                        "(SELECT MIN(rowid) FROM encounters WHERE ENCOUNTER_ID = ?) AND ENCOUNTER_ID = ?",
                        (enc_id, enc_id)
                    )
                fixes['duplicate_encounters_removed'] = len(dup_encounters)
        except Exception as e:
            log.warning("Encounters dedup failed: %s", e)

        conn.commit()

        try:
            conn.execute(
                "INSERT INTO _data_lineage (lineage_id, source_table, operation, row_count, timestamp, details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (hashlib.sha256(datetime.now(timezone.utc).isoformat().encode()).hexdigest()[:16],
                 'claims_4m,encounters', 'referential_integrity_fix',
                 sum(fixes.values()) if fixes else 0, datetime.now(timezone.utc).isoformat(),
                 json.dumps(fixes))
            )
            conn.commit()
        except Exception as e:
            log.warning("Lineage log write failed: %s", e)

        return fixes

    def profile_metadata(self) -> Dict[str, Any]:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        tables_profiled = 0
        columns_profiled = 0

        for table_name in ['members', 'claims_4m', 'encounters', 'providers']:
            try:
                cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            except Exception:
                continue
            tables_profiled += 1

            for col in cols:
                col_name = col[1]
                col_type = col[2]
                is_nullable = 'Y' if col[3] == 0 else 'N'
                is_pk = 'Y' if col[5] == 1 else 'N'

                ref_table = None
                ref_column = None
                star_role = 'attribute'

                for fact_name, fact_def in STAR_SCHEMA['facts'].items():
                    if fact_def['source'] == table_name:
                        if col_name in fact_def.get('foreign_keys', {}):
                            ref = fact_def['foreign_keys'][col_name]
                            ref_table, ref_column = ref.split('.')
                            star_role = 'foreign_key'
                        elif col_name in fact_def.get('measures', []):
                            star_role = 'measure'
                        elif col_name == fact_def.get('business_key'):
                            star_role = 'business_key'
                        elif col_name in fact_def.get('degenerate_dims', []):
                            star_role = 'degenerate_dimension'

                for dim_name, dim_def in STAR_SCHEMA['dimensions'].items():
                    if dim_def['source'] == table_name:
                        col_meta = dim_def['columns'].get(col_name, {})
                        if col_meta:
                            star_role = col_meta.get('role', star_role)
                            if col_meta.get('references'):
                                parts = col_meta['references'].split('.')
                                ref_table = parts[0]
                                ref_column = parts[1] if len(parts) > 1 else None

                try:
                    stats = conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}), "
                        f"SUM(CASE WHEN {col_name} IS NULL OR {col_name} = '' THEN 1 ELSE 0 END), "
                        f"MIN({col_name}), MAX({col_name}) "
                        f"FROM {table_name}"
                    ).fetchone()
                    distinct_count = stats[0]
                    null_count = stats[1]
                    min_val = str(stats[2])[:100] if stats[2] is not None else None
                    max_val = str(stats[3])[:100] if stats[3] is not None else None
                except Exception:
                    distinct_count = null_count = 0
                    min_val = max_val = None

                conn.execute(
                    "INSERT OR REPLACE INTO _table_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (table_name, col_name, col_type, is_nullable, is_pk,
                     ref_table, ref_column, star_role, None, now,
                     distinct_count, null_count, min_val, max_val)
                )
                columns_profiled += 1

        conn.commit()
        return {'tables_profiled': tables_profiled, 'columns_profiled': columns_profiled}

    def create_dim_date(self) -> int:
        conn = self._get_conn()
        conn.execute("DROP TABLE IF EXISTS dim_date")
        conn.execute("""
            CREATE TABLE dim_date (
                DATE_KEY TEXT PRIMARY KEY,
                YEAR INTEGER NOT NULL,
                QUARTER INTEGER NOT NULL,
                MONTH INTEGER NOT NULL,
                MONTH_NAME TEXT NOT NULL,
                DAY_OF_WEEK INTEGER NOT NULL,
                DAY_NAME TEXT NOT NULL,
                IS_WEEKEND INTEGER NOT NULL,
                FISCAL_YEAR INTEGER NOT NULL,
                FISCAL_QUARTER INTEGER NOT NULL
            )
        """)

        min_date = conn.execute(
            "SELECT MIN(d) FROM (SELECT MIN(service_date) as d FROM claims_4m "
            "UNION SELECT MIN(SERVICE_DATE) FROM encounters "
            "UNION SELECT MIN(ENROLLMENT_DATE) FROM members)"
        ).fetchone()[0]
        max_date = conn.execute(
            "SELECT MAX(d) FROM (SELECT MAX(service_date) as d FROM claims_4m "
            "UNION SELECT MAX(SERVICE_DATE) FROM encounters)"
        ).fetchone()[0]

        if not min_date or not max_date:
            return 0

        from datetime import timedelta
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        start = datetime.strptime(min_date[:10], '%Y-%m-%d')
        end = datetime.strptime(max_date[:10], '%Y-%m-%d')
        current = start
        rows = []
        while current <= end:
            date_key = current.strftime('%Y-%m-%d')
            year = current.year
            month = current.month
            quarter = (month - 1) // 3 + 1
            day_of_week = current.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            fiscal_year = year if month >= 7 else year - 1
            fiscal_quarter = ((month - 7) % 12) // 3 + 1
            rows.append((date_key, year, quarter, month, month_names[month],
                         day_of_week, day_names[day_of_week], is_weekend,
                         fiscal_year, fiscal_quarter))
            current += timedelta(days=1)

        conn.executemany(
            "INSERT INTO dim_date VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_date_year ON dim_date(YEAR)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_date_month ON dim_date(YEAR, MONTH)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_date_quarter ON dim_date(YEAR, QUARTER)")
        conn.commit()
        return len(rows)

    def get_dq_summary_for_query(self, tables_used: List[str]) -> Dict:
        conn = self._get_conn()
        try:
            latest = conn.execute(
                "SELECT run_id, overall_score FROM _dq_run_log ORDER BY run_timestamp DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return {'overall_score': 0, 'table_issues': [], 'warnings': [], 'tables': []}
        if not latest:
            return {'overall_score': 0, 'table_issues': [], 'warnings': [], 'tables': []}

        run_id, overall_score = latest
        relevant_rules = conn.execute(
            "SELECT rule_name, severity, violation_count, status FROM _dq_rule_results "
            "WHERE run_id = ? AND status = 'FAIL' ORDER BY "
            "CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END",
            (run_id,)
        ).fetchall()

        table_issues = []
        warnings = []
        for rule_name, severity, violations, status in relevant_rules:
            if tables_used:
                for t in tables_used:
                    if t.lower() in rule_name.lower():
                        table_issues.append({
                            'rule': rule_name,
                            'severity': severity,
                            'violations': violations,
                        })
                        if severity in ('critical', 'warning'):
                            warnings.append(f"{rule_name}: {violations} violations")
            else:
                table_issues.append({
                    'rule': rule_name,
                    'severity': severity,
                    'violations': violations,
                })
                if severity in ('critical', 'warning'):
                    warnings.append(f"{rule_name}: {violations} violations")

        return {
            'overall_score': overall_score,
            'table_issues': table_issues[:5],
            'warnings': warnings[:5],
            'tables': tables_used,
        }

    def get_star_schema_doc(self) -> Dict:
        return STAR_SCHEMA

    def full_governance_run(self) -> Dict[str, Any]:
        t0 = datetime.now(timezone.utc)
        report = {}

        log.info("Step 1/6: Creating metadata tables...")
        self.ensure_metadata_tables()
        report['metadata_tables'] = 'created'

        log.info("Step 2/6: Creating indexes...")
        report['indexes'] = self.create_indexes()

        log.info("Step 3/6: Fixing referential integrity...")
        report['integrity_fixes'] = self.fix_referential_integrity()

        log.info("Step 4/6: Running DQ checks...")
        report['dq'] = self.run_dq_checks()

        log.info("Step 5/6: Profiling metadata...")
        report['metadata_profile'] = self.profile_metadata()

        log.info("Step 6/6: Creating dim_date...")
        report['dim_date_rows'] = self.create_dim_date()

        duration = (datetime.now(timezone.utc) - t0).total_seconds()
        report['total_duration_seconds'] = round(duration, 2)

        log.info("Full governance run completed in %.1fs", duration)
        return report


def run_governance(db_path: str = None) -> Dict[str, Any]:
    if db_path is None:
        db_path = os.path.join(DATA_DIR, 'healthcare_production.db')
    engine = DataGovernanceEngine(db_path)
    try:
        return engine.full_governance_run()
    finally:
        engine.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    db = os.path.join(DATA_DIR, 'healthcare_production.db')
    report = run_governance(db)

    print("\n" + "=" * 80)
    print("DATA GOVERNANCE REPORT")
    print("=" * 80)

    print(f"\nDuration: {report['total_duration_seconds']}s")

    idx = report['indexes']
    print(f"\nINDEXES: {idx['created']} created, {idx['skipped']} existed, {idx['total']} total")

    fixes = report['integrity_fixes']
    if fixes:
        print("\nREFERENTIAL INTEGRITY FIXES:")
        for k, v in fixes.items():
            print(f"  {k}: {v}")
    else:
        print("\nREFERENTIAL INTEGRITY: No issues found")

    dq = report['dq']
    print(f"\nDATA QUALITY SCORE: {dq['overall_score']}/100")
    print(f"  Rules: {dq['total_rules']} total, {dq['passed']} passed, {dq['failed']} failed, {dq['warnings']} warnings")

    failed_rules = [r for r in dq['results'] if r['status'] != 'PASS']
    if failed_rules:
        print("\n  FAILED RULES:")
        for r in failed_rules:
            icon = 'CRITICAL' if r['severity'] == 'critical' else 'WARNING' if r['severity'] == 'warning' else 'INFO'
            print(f"    [{icon}] {r['rule_name']}: {r['violation_count']} violations - {r['description']}")

    meta = report['metadata_profile']
    print(f"\nMETADATA: {meta['tables_profiled']} tables, {meta['columns_profiled']} columns profiled")
    print(f"\nDIM_DATE: {report['dim_date_rows']} date rows generated")

    print("\n" + "=" * 80)
