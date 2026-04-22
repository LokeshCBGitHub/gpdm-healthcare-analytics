import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

@dataclass
class CheckResult:
    category: str
    check_name: str
    status: str
    affected_count: int
    total_count: int
    percentage: float
    sql_query: str
    sample_records: List[Dict[str, Any]]
    message: str
    severity: str


class DataQualityEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.results = []
        self.table_counts = {}
        self.category_scores = {
            'referential_integrity': {'score': 0, 'max_checks': 0},
            'orphan_records': {'score': 0, 'max_checks': 0},
            'completeness': {'score': 0, 'max_checks': 0},
            'validity': {'score': 0, 'max_checks': 0},
            'duplicates': {'score': 0, 'max_checks': 0},
            'temporal': {'score': 0, 'max_checks': 0},
            'cross_table': {'score': 0, 'max_checks': 0},
            'business_rules': {'score': 0, 'max_checks': 0},
        }

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def _execute_query(self, sql: str) -> List[Tuple]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        except Exception as e:
            print(f"Query error: {e}\nSQL: {sql}")
            return []

    def _get_row_count(self, table: str) -> int:
        if table not in self.table_counts:
            sql = f"SELECT COUNT(*) as cnt FROM {table}"
            result = self._execute_query(sql)
            self.table_counts[table] = result[0][0] if result else 0
        return self.table_counts[table]

    def _add_result(self, category: str, check_name: str, status: str,
                    affected_count: int, total_count: int, sql_query: str,
                    sample_records: List[Dict] = None, message: str = "",
                    severity: str = "LOW"):
        percentage = (affected_count / total_count * 100) if total_count > 0 else 0

        if sample_records is None:
            sample_records = []

        result = CheckResult(
            category=category,
            check_name=check_name,
            status=status,
            affected_count=affected_count,
            total_count=total_count,
            percentage=percentage,
            sql_query=sql_query,
            sample_records=sample_records,
            message=message,
            severity=severity
        )

        self.results.append(result)

        if category in self.category_scores:
            self.category_scores[category]['max_checks'] += 1
            if status == 'PASS':
                self.category_scores[category]['score'] += 1

    def _get_sample_records(self, sql: str, limit: int = 5) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql + f" LIMIT {limit}")
            cols = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        except:
            return []

    def run_all_checks(self) -> Dict:
        self.connect()

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = {'MEMBERS', 'CLAIMS', 'ENCOUNTERS', 'DIAGNOSES',
                             'PROVIDERS', 'PRESCRIPTIONS', 'REFERRALS', 'appointments'}

            found_tables = set(t.upper() for t in tables)

            if not found_tables:
                print("Warning: Database has unusual schema, attempting generic checks")

            self._check_referential_integrity(found_tables)
            self._check_orphan_records(found_tables)
            self._check_completeness(found_tables)
            self._check_validity(found_tables)
            self._check_duplicates(found_tables)
            self._check_temporal_consistency(found_tables)
            self._check_cross_table_consistency(found_tables)
            self._check_business_rules(found_tables)

        finally:
            self.disconnect()

        return self._compile_results()

    def _check_referential_integrity(self, tables: set):
        category = 'referential_integrity'

        if 'CLAIMS' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            AND MEMBER_ID IS NOT NULL AND MEMBER_ID != ''
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = f"""
            SELECT * FROM CLAIMS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            AND MEMBER_ID IS NOT NULL AND MEMBER_ID != ''
            """

            self._add_result(category, 'Claims.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims reference non-existent members", severity)

        if 'CLAIMS' in tables and 'ENCOUNTERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
            AND ENCOUNTER_ID NOT IN (SELECT ENCOUNTER_ID FROM ENCOUNTERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
            AND ENCOUNTER_ID NOT IN (SELECT ENCOUNTER_ID FROM ENCOUNTERS)
            """

            self._add_result(category, 'Claims.ENCOUNTER_ID → Encounters.ENCOUNTER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims reference non-existent encounters", severity)

        if 'CLAIMS' in tables and 'PROVIDERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != ''
            AND RENDERING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != ''
            AND RENDERING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """

            self._add_result(category, 'Claims.RENDERING_NPI → Providers.NPI',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims reference non-existent rendering providers", severity)

        if 'CLAIMS' in tables and 'PROVIDERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE BILLING_NPI IS NOT NULL AND BILLING_NPI != ''
            AND BILLING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE BILLING_NPI IS NOT NULL AND BILLING_NPI != ''
            AND BILLING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """

            self._add_result(category, 'Claims.BILLING_NPI → Providers.NPI',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims reference non-existent billing providers", severity)

        if 'DIAGNOSES' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM DIAGNOSES
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('DIAGNOSES')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM DIAGNOSES
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """

            self._add_result(category, 'Diagnoses.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} diagnoses reference non-existent members", severity)

        if 'DIAGNOSES' in tables and 'ENCOUNTERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM DIAGNOSES
            WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
            AND ENCOUNTER_ID NOT IN (SELECT ENCOUNTER_ID FROM ENCOUNTERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('DIAGNOSES')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM DIAGNOSES
            WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
            AND ENCOUNTER_ID NOT IN (SELECT ENCOUNTER_ID FROM ENCOUNTERS)
            """

            self._add_result(category, 'Diagnoses.ENCOUNTER_ID → Encounters.ENCOUNTER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} diagnoses reference non-existent encounters", severity)

        if 'ENCOUNTERS' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM ENCOUNTERS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('ENCOUNTERS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """

            self._add_result(category, 'Encounters.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} encounters reference non-existent members", severity)

        if 'ENCOUNTERS' in tables and 'PROVIDERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM ENCOUNTERS
            WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != ''
            AND RENDERING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('ENCOUNTERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != ''
            AND RENDERING_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """

            self._add_result(category, 'Encounters.RENDERING_NPI → Providers.NPI',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} encounters reference non-existent rendering providers", severity)

        if 'PRESCRIPTIONS' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM PRESCRIPTIONS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PRESCRIPTIONS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PRESCRIPTIONS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """

            self._add_result(category, 'Prescriptions.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} prescriptions reference non-existent members", severity)

        if 'REFERRALS' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM REFERRALS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('REFERRALS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM REFERRALS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """

            self._add_result(category, 'Referrals.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} referrals reference non-existent members", severity)

        if 'APPOINTMENTS' in tables and 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM APPOINTMENTS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('APPOINTMENTS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM APPOINTMENTS
            WHERE MEMBER_ID NOT IN (SELECT MEMBER_ID FROM MEMBERS)
            """

            self._add_result(category, 'Appointments.MEMBER_ID → Members.MEMBER_ID',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} appointments reference non-existent members", severity)

        if 'MEMBERS' in tables and 'PROVIDERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM MEMBERS
            WHERE PCP_NPI IS NOT NULL AND PCP_NPI != ''
            AND PCP_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE PCP_NPI IS NOT NULL AND PCP_NPI != ''
            AND PCP_NPI NOT IN (SELECT NPI FROM PROVIDERS)
            """

            self._add_result(category, 'Members.PCP_NPI → Providers.NPI',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} members assigned to non-existent PCPs", severity)

    def _check_orphan_records(self, tables: set):
        category = 'orphan_records'

        if 'MEMBERS' in tables and 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as orphan_count FROM MEMBERS
            WHERE MEMBER_ID NOT IN (SELECT DISTINCT MEMBER_ID FROM CLAIMS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'WARN' if affected > 0 else 'PASS'
            severity = 'MEDIUM'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE MEMBER_ID NOT IN (SELECT DISTINCT MEMBER_ID FROM CLAIMS)
            """

            self._add_result(category, 'Members without any claims',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} members have no claim history", severity)

        if 'MEMBERS' in tables and 'ENCOUNTERS' in tables:
            sql = """
            SELECT COUNT(*) as orphan_count FROM MEMBERS
            WHERE MEMBER_ID NOT IN (SELECT DISTINCT MEMBER_ID FROM ENCOUNTERS)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'WARN' if affected > 0 else 'PASS'
            severity = 'MEDIUM'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE MEMBER_ID NOT IN (SELECT DISTINCT MEMBER_ID FROM ENCOUNTERS)
            """

            self._add_result(category, 'Members without any encounters',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} members have no encounter history", severity)

        if 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as orphan_count FROM MEMBERS
            WHERE PCP_NPI IS NULL OR PCP_NPI = ''
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'WARN' if affected > 0 else 'PASS'
            severity = 'MEDIUM'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE PCP_NPI IS NULL OR PCP_NPI = ''
            """

            self._add_result(category, 'Members without PCP assignment',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} members lack PCP assignment", severity)

        if 'ENCOUNTERS' in tables and 'DIAGNOSES' in tables:
            sql = """
            SELECT COUNT(*) as orphan_count FROM ENCOUNTERS
            WHERE ENCOUNTER_ID NOT IN (SELECT DISTINCT ENCOUNTER_ID FROM DIAGNOSES
                                       WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != '')
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('ENCOUNTERS')

            status = 'WARN' if affected > 0 else 'PASS'
            severity = 'MEDIUM'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE ENCOUNTER_ID NOT IN (SELECT DISTINCT ENCOUNTER_ID FROM DIAGNOSES
                                       WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != '')
            """

            self._add_result(category, 'Encounters without any diagnoses',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} encounters lack diagnosis information", severity)

        if 'PROVIDERS' in tables and 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as orphan_count FROM PROVIDERS
            WHERE NPI NOT IN (SELECT DISTINCT RENDERING_NPI FROM CLAIMS
                             WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != '')
            AND NPI NOT IN (SELECT DISTINCT BILLING_NPI FROM CLAIMS
                           WHERE BILLING_NPI IS NOT NULL AND BILLING_NPI != '')
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PROVIDERS')

            status = 'WARN' if affected > 0 else 'PASS'
            severity = 'LOW'
            sample_sql = """
            SELECT * FROM PROVIDERS
            WHERE NPI NOT IN (SELECT DISTINCT RENDERING_NPI FROM CLAIMS
                             WHERE RENDERING_NPI IS NOT NULL AND RENDERING_NPI != '')
            AND NPI NOT IN (SELECT DISTINCT BILLING_NPI FROM CLAIMS
                           WHERE BILLING_NPI IS NOT NULL AND BILLING_NPI != '')
            """

            self._add_result(category, 'Providers never billed',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} providers have no claims history", severity)

    def _check_completeness(self, tables: set):
        category = 'completeness'

        critical_fields = {
            'MEMBERS': ['DATE_OF_BIRTH', 'GENDER', 'ENROLLMENT_DATE', 'PCP_NPI',
                       'RISK_SCORE', 'KP_REGION', 'PLAN_TYPE', 'ZIP_CODE'],
            'CLAIMS': ['SERVICE_DATE', 'BILLED_AMOUNT', 'PAID_AMOUNT', 'CLAIM_STATUS',
                      'CPT_CODE', 'ICD10_CODE', 'RENDERING_NPI'],
            'ENCOUNTERS': ['SERVICE_DATE', 'VISIT_TYPE', 'RENDERING_NPI',
                          'DEPARTMENT', 'ENCOUNTER_STATUS'],
            'DIAGNOSES': ['ICD10_CODE', 'DIAGNOSIS_TYPE', 'SEVERITY', 'DIAGNOSIS_DATE'],
            'PROVIDERS': ['SPECIALTY', 'DEPARTMENT', 'STATUS', 'PANEL_SIZE'],
            'PRESCRIPTIONS': ['MEDICATION_NAME', 'PRESCRIPTION_DATE', 'COST', 'DAYS_SUPPLY'],
            'REFERRALS': ['STATUS', 'REFERRAL_DATE', 'URGENCY'],
        }

        for table, fields in critical_fields.items():
            if table not in tables:
                continue

            total = self._get_row_count(table)

            for field in fields:
                sql = f"""
                SELECT COUNT(*) as null_count FROM {table}
                WHERE {field} IS NULL OR {field} = ''
                """
                result = self._execute_query(sql)
                affected = result[0][0] if result else 0

                status = 'PASS' if affected == 0 else ('WARN' if affected < total * 0.05 else 'FAIL')
                severity = 'CRITICAL' if affected > total * 0.1 else 'MEDIUM' if affected > 0 else 'LOW'

                sample_sql = f"""
                SELECT * FROM {table}
                WHERE {field} IS NULL OR {field} = ''
                """

                percentage = affected/total*100 if total > 0 else 0
                self._add_result(category, f'{table}.{field} - Completeness',
                               status, affected, total, sql,
                               self._get_sample_records(sample_sql),
                               f"{affected} ({percentage:.1f}%) records missing {field}",
                               severity)

    def _check_validity(self, tables: set):
        category = 'validity'

        if 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM MEMBERS
            WHERE DATE_OF_BIRTH IS NOT NULL AND DATE_OF_BIRTH != ''
            AND (
                DATE_OF_BIRTH NOT LIKE '____-__-__' OR
                DATE_OF_BIRTH > DATE('now') OR
                DATE_OF_BIRTH < '1900-01-01'
            )
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE DATE_OF_BIRTH IS NOT NULL AND DATE_OF_BIRTH != ''
            AND (
                DATE_OF_BIRTH NOT LIKE '____-__-__' OR
                DATE_OF_BIRTH > DATE('now') OR
                DATE_OF_BIRTH < '1900-01-01'
            )
            """

            self._add_result(category, 'Members - Invalid DATE_OF_BIRTH format or values',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} invalid date values in DATE_OF_BIRTH", severity)

        if 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM MEMBERS
            WHERE GENDER IS NOT NULL AND GENDER != ''
            AND UPPER(GENDER) NOT IN ('M', 'F', 'U', 'O')
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE GENDER IS NOT NULL AND GENDER != ''
            AND UPPER(GENDER) NOT IN ('M', 'F', 'U', 'O')
            """

            self._add_result(category, 'Members - Invalid GENDER values',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} invalid gender values", severity)

        if 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE (CAST(BILLED_AMOUNT AS REAL) < CAST(ALLOWED_AMOUNT AS REAL)
                   OR CAST(ALLOWED_AMOUNT AS REAL) < CAST(PAID_AMOUNT AS REAL))
            AND BILLED_AMOUNT IS NOT NULL AND ALLOWED_AMOUNT IS NOT NULL AND PAID_AMOUNT IS NOT NULL
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE (CAST(BILLED_AMOUNT AS REAL) < CAST(ALLOWED_AMOUNT AS REAL)
                   OR CAST(ALLOWED_AMOUNT AS REAL) < CAST(PAID_AMOUNT AS REAL))
            AND BILLED_AMOUNT IS NOT NULL AND ALLOWED_AMOUNT IS NOT NULL AND PAID_AMOUNT IS NOT NULL
            """

            self._add_result(category, 'Claims - Invalid financial waterfall',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims violate financial waterfall logic", severity)

        if 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE PAID_AMOUNT IS NOT NULL AND PAID_AMOUNT != ''
            AND CAST(PAID_AMOUNT AS REAL) < 0
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE PAID_AMOUNT IS NOT NULL AND PAID_AMOUNT != ''
            AND CAST(PAID_AMOUNT AS REAL) < 0
            """

            self._add_result(category, 'Claims - Negative paid amounts',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims have negative paid amounts", severity)

        if 'MEMBERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM MEMBERS
            WHERE RISK_SCORE IS NOT NULL AND RISK_SCORE != ''
            AND (CAST(RISK_SCORE AS REAL) < 0 OR CAST(RISK_SCORE AS REAL) > 10)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('MEMBERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE RISK_SCORE IS NOT NULL AND RISK_SCORE != ''
            AND (CAST(RISK_SCORE AS REAL) < 0 OR CAST(RISK_SCORE AS REAL) > 10)
            """

            self._add_result(category, 'Members - Invalid RISK_SCORE range',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} members have risk scores outside 0-10 range", severity)

        if 'PRESCRIPTIONS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM PRESCRIPTIONS
            WHERE DAYS_SUPPLY IS NOT NULL AND DAYS_SUPPLY != ''
            AND CAST(DAYS_SUPPLY AS REAL) <= 0
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PRESCRIPTIONS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PRESCRIPTIONS
            WHERE DAYS_SUPPLY IS NOT NULL AND DAYS_SUPPLY != ''
            AND CAST(DAYS_SUPPLY AS REAL) <= 0
            """

            self._add_result(category, 'Prescriptions - Invalid DAYS_SUPPLY',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} prescriptions have invalid days supply", severity)

        if 'PRESCRIPTIONS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM PRESCRIPTIONS
            WHERE REFILLS_USED IS NOT NULL AND REFILLS_AUTHORIZED IS NOT NULL
            AND CAST(REFILLS_USED AS INTEGER) > CAST(REFILLS_AUTHORIZED AS INTEGER)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PRESCRIPTIONS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PRESCRIPTIONS
            WHERE REFILLS_USED IS NOT NULL AND REFILLS_AUTHORIZED IS NOT NULL
            AND CAST(REFILLS_USED AS INTEGER) > CAST(REFILLS_AUTHORIZED AS INTEGER)
            """

            self._add_result(category, 'Prescriptions - Refills used > authorized',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} prescriptions have more refills used than authorized", severity)

    def _check_duplicates(self, tables: set):
        category = 'duplicates'

        if 'CLAIMS' in tables:
            sql = """
            SELECT CLAIM_ID, COUNT(*) as cnt FROM CLAIMS
            WHERE CLAIM_ID IS NOT NULL AND CLAIM_ID != ''
            GROUP BY CLAIM_ID HAVING COUNT(*) > 1
            """
            result = self._execute_query(sql)
            affected = sum(row[1] - 1 for row in result)
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE CLAIM_ID IN (
                SELECT CLAIM_ID FROM CLAIMS
                WHERE CLAIM_ID IS NOT NULL AND CLAIM_ID != ''
                GROUP BY CLAIM_ID HAVING COUNT(*) > 1
            )
            ORDER BY CLAIM_ID
            """

            self._add_result(category, 'Claims - Duplicate CLAIM_IDs',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} duplicate claim records found", severity)

        if 'MEMBERS' in tables:
            sql = """
            SELECT MEMBER_ID, COUNT(*) as cnt FROM MEMBERS
            WHERE MEMBER_ID IS NOT NULL AND MEMBER_ID != ''
            GROUP BY MEMBER_ID HAVING COUNT(*) > 1
            """
            result = self._execute_query(sql)
            affected = sum(row[1] - 1 for row in result)
            total = self._get_row_count('MEMBERS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM MEMBERS
            WHERE MEMBER_ID IN (
                SELECT MEMBER_ID FROM MEMBERS
                WHERE MEMBER_ID IS NOT NULL AND MEMBER_ID != ''
                GROUP BY MEMBER_ID HAVING COUNT(*) > 1
            )
            ORDER BY MEMBER_ID
            """

            self._add_result(category, 'Members - Duplicate MEMBER_IDs',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} duplicate member records found", severity)

        if 'ENCOUNTERS' in tables:
            sql = """
            SELECT ENCOUNTER_ID, COUNT(*) as cnt FROM ENCOUNTERS
            WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
            GROUP BY ENCOUNTER_ID HAVING COUNT(*) > 1
            """
            result = self._execute_query(sql)
            affected = sum(row[1] - 1 for row in result)
            total = self._get_row_count('ENCOUNTERS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE ENCOUNTER_ID IN (
                SELECT ENCOUNTER_ID FROM ENCOUNTERS
                WHERE ENCOUNTER_ID IS NOT NULL AND ENCOUNTER_ID != ''
                GROUP BY ENCOUNTER_ID HAVING COUNT(*) > 1
            )
            ORDER BY ENCOUNTER_ID
            """

            self._add_result(category, 'Encounters - Duplicate ENCOUNTER_IDs',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} duplicate encounter records found", severity)

        if 'PROVIDERS' in tables:
            sql = """
            SELECT NPI, COUNT(*) as cnt FROM PROVIDERS
            WHERE NPI IS NOT NULL AND NPI != ''
            GROUP BY NPI HAVING COUNT(*) > 1
            """
            result = self._execute_query(sql)
            affected = sum(row[1] - 1 for row in result)
            total = self._get_row_count('PROVIDERS')

            status = 'PASS' if affected == 0 else 'FAIL'
            severity = 'CRITICAL' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PROVIDERS
            WHERE NPI IN (
                SELECT NPI FROM PROVIDERS
                WHERE NPI IS NOT NULL AND NPI != ''
                GROUP BY NPI HAVING COUNT(*) > 1
            )
            ORDER BY NPI
            """

            self._add_result(category, 'Providers - Duplicate NPIs',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} duplicate provider records found", severity)

    def _check_temporal_consistency(self, tables: set):
        category = 'temporal'

        if 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE SERVICE_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL
            AND SERVICE_DATE != '' AND SUBMITTED_DATE != ''
            AND SUBMITTED_DATE < SERVICE_DATE
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE SERVICE_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL
            AND SERVICE_DATE != '' AND SUBMITTED_DATE != ''
            AND SUBMITTED_DATE < SERVICE_DATE
            """

            self._add_result(category, 'Claims - SUBMITTED_DATE before SERVICE_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims submitted before service date", severity)

        if 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE SUBMITTED_DATE IS NOT NULL AND ADJUDICATED_DATE IS NOT NULL
            AND SUBMITTED_DATE != '' AND ADJUDICATED_DATE != ''
            AND ADJUDICATED_DATE < SUBMITTED_DATE
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE SUBMITTED_DATE IS NOT NULL AND ADJUDICATED_DATE IS NOT NULL
            AND SUBMITTED_DATE != '' AND ADJUDICATED_DATE != ''
            AND ADJUDICATED_DATE < SUBMITTED_DATE
            """

            self._add_result(category, 'Claims - ADJUDICATED_DATE before SUBMITTED_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims adjudicated before submission", severity)

        if 'ENCOUNTERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM ENCOUNTERS
            WHERE ADMIT_DATE IS NOT NULL AND DISCHARGE_DATE IS NOT NULL
            AND ADMIT_DATE != '' AND DISCHARGE_DATE != ''
            AND ADMIT_DATE > DISCHARGE_DATE
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('ENCOUNTERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'HIGH' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE ADMIT_DATE IS NOT NULL AND DISCHARGE_DATE IS NOT NULL
            AND ADMIT_DATE != '' AND DISCHARGE_DATE != ''
            AND ADMIT_DATE > DISCHARGE_DATE
            """

            self._add_result(category, 'Encounters - ADMIT_DATE after DISCHARGE_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} encounters have invalid admission/discharge dates", severity)

        if 'PRESCRIPTIONS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM PRESCRIPTIONS
            WHERE PRESCRIPTION_DATE IS NOT NULL AND FILL_DATE IS NOT NULL
            AND PRESCRIPTION_DATE != '' AND FILL_DATE != ''
            AND PRESCRIPTION_DATE > FILL_DATE
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PRESCRIPTIONS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PRESCRIPTIONS
            WHERE PRESCRIPTION_DATE IS NOT NULL AND FILL_DATE IS NOT NULL
            AND PRESCRIPTION_DATE != '' AND FILL_DATE != ''
            AND PRESCRIPTION_DATE > FILL_DATE
            """

            self._add_result(category, 'Prescriptions - PRESCRIPTION_DATE after FILL_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} prescriptions have prescription date after fill date", severity)

        if 'REFERRALS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM REFERRALS
            WHERE REFERRAL_DATE IS NOT NULL AND APPOINTMENT_DATE IS NOT NULL
            AND REFERRAL_DATE != '' AND APPOINTMENT_DATE != ''
            AND REFERRAL_DATE > APPOINTMENT_DATE
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('REFERRALS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM REFERRALS
            WHERE REFERRAL_DATE IS NOT NULL AND APPOINTMENT_DATE IS NOT NULL
            AND REFERRAL_DATE != '' AND APPOINTMENT_DATE != ''
            AND REFERRAL_DATE > APPOINTMENT_DATE
            """

            self._add_result(category, 'Referrals - REFERRAL_DATE after APPOINTMENT_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} referrals have invalid date sequences", severity)

    def _check_cross_table_consistency(self, tables: set):
        category = 'cross_table'

        if 'MEMBERS' in tables and 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as mismatch_count
            FROM CLAIMS c
            JOIN MEMBERS m ON c.MEMBER_ID = m.MEMBER_ID
            WHERE c.KP_REGION IS NOT NULL AND m.KP_REGION IS NOT NULL
            AND c.KP_REGION != '' AND m.KP_REGION != ''
            AND c.KP_REGION != m.KP_REGION
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT c.*, m.KP_REGION as MEMBER_REGION
            FROM CLAIMS c
            JOIN MEMBERS m ON c.MEMBER_ID = m.MEMBER_ID
            WHERE c.KP_REGION IS NOT NULL AND m.KP_REGION IS NOT NULL
            AND c.KP_REGION != '' AND m.KP_REGION != ''
            AND c.KP_REGION != m.KP_REGION
            """

            self._add_result(category, 'Claims KP_REGION matches Member KP_REGION',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} claims have mismatched KP_REGION values", severity)

    def _check_business_rules(self, tables: set):
        category = 'business_rules'

        if 'CLAIMS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM CLAIMS
            WHERE CLAIM_STATUS = 'DENIED' OR CLAIM_STATUS = 'Denied'
            AND (DENIAL_REASON IS NULL OR DENIAL_REASON = '')
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('CLAIMS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM CLAIMS
            WHERE CLAIM_STATUS = 'DENIED' OR CLAIM_STATUS = 'Denied'
            AND (DENIAL_REASON IS NULL OR DENIAL_REASON = '')
            """

            self._add_result(category, 'Denied claims have DENIAL_REASON',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} denied claims missing denial reason", severity)

        if 'ENCOUNTERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM ENCOUNTERS
            WHERE (VISIT_TYPE = 'INPATIENT' OR VISIT_TYPE = 'Inpatient')
            AND (LENGTH_OF_STAY IS NULL OR LENGTH_OF_STAY = '' OR CAST(LENGTH_OF_STAY AS INTEGER) <= 0)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('ENCOUNTERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM ENCOUNTERS
            WHERE (VISIT_TYPE = 'INPATIENT' OR VISIT_TYPE = 'Inpatient')
            AND (LENGTH_OF_STAY IS NULL OR LENGTH_OF_STAY = '' OR CAST(LENGTH_OF_STAY AS INTEGER) <= 0)
            """

            self._add_result(category, 'Inpatient encounters have valid LENGTH_OF_STAY',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} inpatient encounters have invalid length of stay", severity)

        if 'PROVIDERS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM PROVIDERS
            WHERE (STATUS = 'ACTIVE' OR STATUS = 'Active')
            AND (PANEL_SIZE IS NULL OR PANEL_SIZE = '' OR CAST(PANEL_SIZE AS INTEGER) <= 0)
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('PROVIDERS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'LOW' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM PROVIDERS
            WHERE (STATUS = 'ACTIVE' OR STATUS = 'Active')
            AND (PANEL_SIZE IS NULL OR PANEL_SIZE = '' OR CAST(PANEL_SIZE AS INTEGER) <= 0)
            """

            self._add_result(category, 'Active providers have positive PANEL_SIZE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} active providers have zero or missing panel size", severity)

        if 'REFERRALS' in tables:
            sql = """
            SELECT COUNT(*) as invalid_count FROM REFERRALS
            WHERE (STATUS = 'COMPLETED' OR STATUS = 'Completed')
            AND (APPOINTMENT_DATE IS NULL OR APPOINTMENT_DATE = '')
            """
            result = self._execute_query(sql)
            affected = result[0][0] if result else 0
            total = self._get_row_count('REFERRALS')

            status = 'PASS' if affected == 0 else 'WARN'
            severity = 'MEDIUM' if affected > 0 else 'LOW'
            sample_sql = """
            SELECT * FROM REFERRALS
            WHERE (STATUS = 'COMPLETED' OR STATUS = 'Completed')
            AND (APPOINTMENT_DATE IS NULL OR APPOINTMENT_DATE = '')
            """

            self._add_result(category, 'Completed referrals have APPOINTMENT_DATE',
                           status, affected, total, sql,
                           self._get_sample_records(sample_sql),
                           f"{affected} completed referrals missing appointment date", severity)

    def _compile_results(self) -> Dict:

        weights = {
            'referential_integrity': 0.25,
            'completeness': 0.20,
            'validity': 0.20,
            'orphan_records': 0.15,
            'temporal': 0.10,
            'cross_table': 0.05,
            'business_rules': 0.05,
        }

        weighted_score = 0
        total_weight = 0

        for category, weight in weights.items():
            if category in self.category_scores:
                max_checks = self.category_scores[category]['max_checks']
                score = self.category_scores[category]['score']

                if max_checks > 0:
                    category_score = (score / max_checks) * 100
                    weighted_score += category_score * weight
                    total_weight += weight

        overall_score = (weighted_score / total_weight) if total_weight > 0 else 0

        status_counts = {
            'PASS': sum(1 for r in self.results if r.status == 'PASS'),
            'WARN': sum(1 for r in self.results if r.status == 'WARN'),
            'FAIL': sum(1 for r in self.results if r.status == 'FAIL'),
        }

        by_category = defaultdict(list)
        for result in self.results:
            by_category[result.category].append(result)

        return {
            'overall_score': round(overall_score, 1),
            'status_counts': status_counts,
            'total_checks': len(self.results),
            'category_scores': self.category_scores,
            'results': self.results,
            'by_category': dict(by_category),
            'timestamp': datetime.now().isoformat(),
        }

    def generate_report_html(self, results: Dict, output_path: str) -> str:

        overall_score = results['overall_score']
        status_counts = results['status_counts']

        if overall_score >= 90:
            score_color = '#27AE60'
            score_status = 'Excellent'
        elif overall_score >= 75:
            score_color = '#F39C12'
            score_status = 'Good'
        elif overall_score >= 60:
            score_color = '#E67E22'
            score_status = 'Fair'
        else:
            score_color = '#E74C3C'
            score_status = 'Poor'

        category_html = ''
        for category, scores in results['category_scores'].items():
            if scores['max_checks'] > 0:
                pct = (scores['score'] / scores['max_checks']) * 100
                category_name = category.replace('_', ' ').title()

                if pct >= 90:
                    cat_color = '#27AE60'
                elif pct >= 75:
                    cat_color = '#F39C12'
                else:
                    cat_color = '#E74C3C'

                category_html += f"""
                <div class="category-score">
                    <div class="category-name">{category_name}</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {pct}%; background-color: {cat_color};"></div>
                    </div>
                    <div class="score-value">{pct:.0f}% ({int(scores['score'])}/{int(scores['max_checks'])})</div>
                </div>
                """

        results_html = ''
        for category in sorted(results['by_category'].keys()):
            category_results = results['by_category'][category]
            category_name = category.replace('_', ' ').title()

            results_html += f'<h3 style="margin-top: 2rem; color: #002B5C; border-bottom: 2px solid #002B5C; padding-bottom: 0.5rem;">{category_name}</h3>'

            for result in category_results:
                status_color = {
                    'PASS': '#27AE60',
                    'WARN': '#F39C12',
                    'FAIL': '#E74C3C',
                }[result.status]

                severity_colors = {"CRITICAL": "#E74C3C", "HIGH": "#E67E22", "MEDIUM": "#F39C12", "LOW": "#3498DB"}
                severity_color = severity_colors.get(result.severity, "#95A5A6")
                severity_badge = f'<span style="background-color: {severity_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; margin-left: 0.5rem;">{result.severity}</span>'

                results_html += f"""
                <div class="check-result" style="border-left: 4px solid {status_color}; padding: 1rem; margin: 0.5rem 0; background-color: #f9f9f9; border-radius: 0.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>{result.check_name}</strong>
                        <div>
                            <span style="background-color: {status_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem; font-weight: bold; font-size: 0.85rem;">{result.status}</span>
                            {severity_badge}
                        </div>
                    </div>
                    <div style="color: #555; margin-bottom: 0.5rem;">
                        <strong>Impact:</strong> {result.affected_count:,} of {result.total_count:,} records ({result.percentage:.1f}%)
                    </div>
                    <div style="color: #555; margin-bottom: 0.5rem;">
                        <strong>Message:</strong> {result.message}
                    </div>
                    <div style="background-color: #fff; border: 1px solid #ddd; border-radius: 0.25rem; padding: 0.5rem; margin: 0.5rem 0; max-height: 200px; overflow-y: auto;">
                        <div style="font-size: 0.75rem; color: #666; font-family: monospace; word-break: break-all;">
                            <strong>SQL Query:</strong><br/>
                            {result.sql_query.strip()}
                        </div>
                    </div>

                    {self._render_sample_records(result.sample_records) if result.sample_records else '<div style="color: #999; font-size: 0.85rem;">No affected records in sample.</div>'}
                </div>
                """

        critical_issues = [r for r in results['results'] if r.severity == 'CRITICAL' and r.status != 'PASS']
        high_issues = [r for r in results['results'] if r.severity == 'HIGH' and r.status != 'PASS']

        recommendations = ''
        if critical_issues:
            recommendations += '<div style="background-color: #FADBD8; border-left: 4px solid #E74C3C; padding: 1rem; margin-bottom: 1rem; border-radius: 0.25rem;"><h4 style="margin: 0 0 0.5rem 0; color: #C0392B;">Critical Issues (Immediate Action Required)</h4><ul style="margin: 0; padding-left: 1.5rem;">'
            for issue in critical_issues[:5]:
                recommendations += f'<li>{issue.check_name}: {issue.message}</li>'
            recommendations += '</ul></div>'

        if high_issues:
            recommendations += '<div style="background-color: #FEF5E7; border-left: 4px solid #F39C12; padding: 1rem; margin-bottom: 1rem; border-radius: 0.25rem;"><h4 style="margin: 0 0 0.5rem 0; color: #D68910;">High Priority Issues</h4><ul style="margin: 0; padding-left: 1.5rem;">'
            for issue in high_issues[:5]:
                recommendations += f'<li>{issue.check_name}: {issue.message}</li>'
            recommendations += '</ul></div>'

        if not critical_issues and not high_issues:
            recommendations += '<div style="background-color: #D5F4E6; border-left: 4px solid #27AE60; padding: 1rem; border-radius: 0.25rem;"><h4 style="margin: 0 0 0.5rem 0; color: #1E8449;">Data Quality Status</h4><p style="margin: 0;">Your dataset has passed all critical checks. Continue monitoring for data quality issues.</p></div>'

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Data Quality Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .header {{
            background: linear-gradient(135deg, #002B5C 0%, #004B87 100%);
            color: white;
            padding: 3rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            font-size: 1rem;
            opacity: 0.9;
        }}

        .score-card {{
            background: white;
            border-radius: 0.5rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
            align-items: center;
        }}

        .overall-score {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}

        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background-color: {score_color};
            color: white;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .score-status {{
            font-size: 1.25rem;
            font-weight: bold;
            color: {score_color};
        }}

        .status-summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }}

        .status-item {{
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 0.25rem;
            text-align: center;
            border-top: 3px solid #002B5C;
        }}

        .status-item strong {{
            display: block;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .status-item.pass strong {{ color: #27AE60; }}
        .status-item.warn strong {{ color: #F39C12; }}
        .status-item.fail strong {{ color: #E74C3C; }}

        .status-item span {{
            color: #666;
            font-size: 0.9rem;
        }}

        .category-scores {{
            background: white;
            border-radius: 0.5rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .category-scores h2 {{
            color: #002B5C;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #002B5C;
            padding-bottom: 0.5rem;
        }}

        .category-score {{
            margin-bottom: 1.5rem;
        }}

        .category-name {{
            font-weight: bold;
            color: #002B5C;
            margin-bottom: 0.5rem;
        }}

        .score-bar {{
            background-color: #e0e0e0;
            border-radius: 0.25rem;
            height: 20px;
            overflow: hidden;
        }}

        .score-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .score-value {{
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
        }}

        .recommendations {{
            background: white;
            border-radius: 0.5rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .recommendations h2 {{
            color: #002B5C;
            margin-bottom: 1rem;
            border-bottom: 2px solid #002B5C;
            padding-bottom: 0.5rem;
        }}

        .results-section {{
            background: white;
            border-radius: 0.5rem;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .results-section h2 {{
            color: #002B5C;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #002B5C;
            padding-bottom: 0.5rem;
        }}

        .check-result {{
            transition: all 0.2s ease;
        }}

        .check-result:hover {{
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }}

        table th {{
            background-color: #f0f0f0;
            padding: 0.5rem;
            text-align: left;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }}

        table td {{
            padding: 0.5rem;
            border-bottom: 1px solid #eee;
        }}

        table tr:hover {{
            background-color: #f9f9f9;
        }}

        .footer {{
            text-align: center;
            color: #999;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Healthcare Data Quality Report</h1>
            <p>Comprehensive data quality assessment based on Tuva Project framework</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem;">Generated: {results['timestamp']}</p>
        </div>

        <div class="score-card">
            <div class="overall-score">
                <div class="score-circle">{overall_score:.0f}</div>
                <div class="score-status">{score_status}</div>
            </div>
            <div>
                <h3 style="color: #002B5C; margin-bottom: 1rem;">Check Summary</h3>
                <div class="status-summary">
                    <div class="status-item pass">
                        <strong>{status_counts['PASS']}</strong>
                        <span>Passed</span>
                    </div>
                    <div class="status-item warn">
                        <strong>{status_counts['WARN']}</strong>
                        <span>Warnings</span>
                    </div>
                    <div class="status-item fail">
                        <strong>{status_counts['FAIL']}</strong>
                        <span>Failed</span>
                    </div>
                </div>
                <p style="margin-top: 1rem; color: #666; font-size: 0.9rem;">
                    <strong>Total Checks:</strong> {results['total_checks']}
                </p>
            </div>
        </div>

        <div class="category-scores">
            <h2>Category Scores</h2>
            {category_html}
        </div>

        <div class="recommendations">
            <h2>Recommendations & Priority Actions</h2>
            {recommendations}
        </div>

        <div class="results-section">
            <h2>Detailed Check Results</h2>
            {results_html}
        </div>

        <div class="footer">
            <p>This report was generated by the Healthcare Data Quality Engine.</p>
            <p>For remediation assistance, review the SQL queries and sample records for each failed check.</p>
        </div>
    </div>
</body>
</html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Report generated: {output_path}")
        return output_path

    def _render_sample_records(self, records: List[Dict]) -> str:
        if not records:
            return ''

        html = '<div style="margin-top: 0.5rem;"><strong>Sample Records:</strong><table>'
        html += '<tr>'
        for key in records[0].keys():
            html += f'<th>{key}</th>'
        html += '</tr>'

        for record in records:
            html += '<tr>'
            for value in record.values():
                val_str = str(value)[:50] if value else '(NULL)'
                html += f'<td>{val_str}</td>'
            html += '</tr>'

        html += '</table></div>'
        return html


if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/scripts/healthcare_demo.db'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data_quality_report.html'

    print(f"Starting Data Quality Engine")
    print(f"Database: {db_path}")
    print(f"Output: {output_path}")

    engine = DataQualityEngine(db_path)
    results = engine.run_all_checks()

    print(f"\nResults Summary:")
    print(f"  Overall Score: {results['overall_score']}/100")
    print(f"  Passed: {results['status_counts']['PASS']}")
    print(f"  Warnings: {results['status_counts']['WARN']}")
    print(f"  Failed: {results['status_counts']['FAIL']}")

    engine.generate_report_html(results, output_path)
    print(f"\nData quality assessment complete!")
