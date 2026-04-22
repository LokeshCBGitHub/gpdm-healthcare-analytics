import re
import math
import time
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger('gpdm.concept_expander')


class SchemaProfiler:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables = {}
        self.col_index = {}
        self._profile()

    def _profile(self):
        t0 = time.time()
        try:
            conn = sqlite3.connect(self.db_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            for t in tables:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
                row_count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                table_profile = {'row_count': row_count, 'columns': {}}

                for c in cols:
                    cname = c[1]
                    try:
                        distinct = conn.execute(
                            f"SELECT COUNT(DISTINCT [{cname}]) FROM {t}"
                        ).fetchone()[0]
                        nulls = conn.execute(
                            f'SELECT COUNT(*) FROM {t} WHERE [{cname}] IS NULL OR [{cname}] = ""'
                        ).fetchone()[0]
                        samples = [str(s[0]) for s in conn.execute(
                            f'SELECT DISTINCT [{cname}] FROM {t} WHERE [{cname}] IS NOT NULL AND [{cname}] != "" LIMIT 8'
                        ).fetchall()]

                        col_type = self._classify_column(
                            cname, distinct, row_count, samples
                        )

                        profile = {
                            'name': cname,
                            'table': t,
                            'type': col_type,
                            'distinct': distinct,
                            'null_pct': round(nulls / max(row_count, 1) * 100, 1),
                            'samples': samples[:5],
                            'row_count': row_count,
                        }
                        table_profile['columns'][cname] = profile
                        self.col_index[f"{t}.{cname}"] = profile
                    except Exception:
                        pass

                self.tables[t] = table_profile
            conn.close()
            elapsed = (time.time() - t0) * 1000
            total_cols = sum(len(t['columns']) for t in self.tables.values())
            logger.info("Schema profiled: %d tables, %d columns in %dms",
                       len(self.tables), total_cols, elapsed)
        except Exception as e:
            logger.error("Schema profiling failed: %s", e)

    def _classify_column(self, name: str, distinct: int, row_count: int,
                         samples: List[str]) -> str:
        n = name.upper()
        ratio = distinct / max(row_count, 1)

        if any(s and len(s) == 10 and s.count('-') == 2 and s[:4].isdigit()
               for s in samples):
            return 'temporal'
        if any(kw in n for kw in ('DATE', 'TIME', '_DT', 'TIMESTAMP')):
            return 'temporal'

        clean_samples = [s for s in samples if s and s not in ('', 'None')]
        is_numeric = all(
            s.replace('.', '', 1).replace('-', '', 1).isdigit()
            for s in clean_samples
        ) if clean_samples else False

        if ratio > 0.5 and not is_numeric:
            return 'identifier'
        if any(kw in n for kw in ('_ID', 'MRN', 'NPI', 'EMAIL', 'PHONE',
                                   'ADDRESS', 'ZIP_CODE', 'DEA_', 'AUTH')):
            return 'identifier'

        if is_numeric and distinct > 20:
            return 'metric'
        if any(kw in n for kw in ('AMOUNT', 'COST', 'PAID', 'BILLED',
                                   'SCORE', 'PRICE', 'RVU', 'PANEL_SIZE')):
            return 'metric'

        if 2 <= distinct <= 50:
            return 'categorical'

        if is_numeric:
            return 'metric'
        return 'identifier'

    def get_dimensions(self, table: str) -> List[Dict]:
        if table not in self.tables:
            return []
        return [
            prof for prof in self.tables[table]['columns'].values()
            if prof['type'] == 'categorical'
        ]

    def get_metrics(self, table: str) -> List[Dict]:
        if table not in self.tables:
            return []
        return [
            prof for prof in self.tables[table]['columns'].values()
            if prof['type'] == 'metric'
        ]

    def get_temporals(self, table: str) -> List[Dict]:
        if table not in self.tables:
            return []
        return [
            prof for prof in self.tables[table]['columns'].values()
            if prof['type'] == 'temporal'
        ]


COLUMN_CONTEXT = {
    'GENDER': ('Gender Distribution', 'Gender distribution affects preventive screening protocols (mammography, prostate), risk stratification, and HEDIS measure denominators.'),
    'RACE': ('Race & Ethnicity', 'CMS and NCQA require health equity reporting stratified by race/ethnicity. Disparities in outcomes and access are key quality indicators.'),
    'LANGUAGE': ('Preferred Language', 'CMS requires Language Access Plans for LEP members. NCQA CLAS standards measure culturally appropriate services.'),
    'PLAN_TYPE': ('Plan Type', 'Plan mix determines revenue composition, risk corridor exposure, and distinct HEDIS/Stars reporting obligations per line of business.'),
    'KP_REGION': ('Geographic Region', 'Regional variation affects provider network adequacy, facility utilization, and state-specific regulatory compliance.'),
    'FACILITY': ('Facility', 'Facility-level analysis reveals capacity utilization, cost variation, and network adequacy across service areas.'),
    'SPECIALTY': ('Provider Specialty', 'Specialty distribution affects referral patterns, access metrics, and network adequacy per CMS time-and-distance standards.'),
    'DEPARTMENT': ('Department', 'Department utilization informs staffing models, facility planning, and capital allocation.'),
    'VISIT_TYPE': ('Visit Type', 'Visit type mix (inpatient, outpatient, ED, telehealth) indicates care model maturity and access patterns.'),
    'CLAIM_STATUS': ('Claim Status', 'Claim status distribution reveals revenue cycle health. High denial or pending rates indicate process gaps.'),
    'CLAIM_TYPE': ('Claim Type', 'Institutional vs. Professional cost split benchmarks: HFMA reports institutional typically 60-70% of total spend.'),
    'DENIAL_REASON': ('Denial Reason', 'CARC/RARC denial codes identify preventable vs. clinical denials. CO-16 and CO-4 are often preventable through front-end edits.'),
    'PROVIDER_TYPE': ('Provider Type', 'MD/DO vs. APP ratios indicate care model maturity. MGMA benchmarks: 1.5-2.0 APPs per physician for primary care.'),
    'STATUS': ('Status', 'Status distribution reveals operational throughput and potential bottlenecks in workflows.'),
    'MEDICATION_CLASS': ('Medication Class', 'Medication class utilization drives formulary management and Stars Part D adherence measures.'),
    'MEDICATION_NAME': ('Medication', 'Individual medication volume informs pharmacy benefit design and therapeutic substitution opportunities.'),
    'PHARMACY': ('Pharmacy Channel', 'Pharmacy channel mix (mail order vs. retail) affects cost and adherence. Mail order typically improves PDC.'),
    'APPOINTMENT_TYPE': ('Appointment Type', 'Appointment type distribution reveals care delivery patterns and resource allocation needs.'),
    'DIAGNOSIS_TYPE': ('Diagnosis Type', 'Primary vs. secondary diagnosis coding affects DRG assignment, risk adjustment, and quality measure capture.'),
    'SEVERITY': ('Severity Level', 'Severity distribution drives acuity-adjusted benchmarking and resource allocation.'),
    'IS_CHRONIC': ('Chronic vs. Acute', 'Chronic condition identification is critical for care management enrollment and HCC risk capture.'),
    'HCC_CATEGORY': ('HCC Risk Category', 'HCC categories drive Medicare Advantage risk adjustment revenue. Accurate coding directly impacts capitation.'),
    'CHIEF_COMPLAINT': ('Chief Complaint', 'Chief complaint analysis identifies top presenting conditions and ED triage patterns.'),
    'DISPOSITION': ('Discharge Disposition', 'Disposition patterns reveal admission rates, observation utilization, and transfer patterns.'),
    'URGENCY': ('Urgency Level', 'Urgency distribution indicates clinical acuity mix and scheduling pressure.'),
    'REFERRAL_TYPE': ('Referral Type', 'Internal vs. external referral ratios indicate network leakage and self-referral patterns.'),
    'REFERRAL_REASON': ('Referral Reason', 'Referral reasons inform specialist demand planning and care coordination needs.'),
    'IS_PCP_VISIT': ('PCP vs. Specialist', 'PCP visit ratios indicate primary care utilization and gatekeeping effectiveness.'),
    'CHRONIC_CONDITIONS': ('Chronic Condition Count', 'Multiple chronic conditions (MCC) drive 75%+ of total healthcare spend. CMS CCM billing applies to 2+ conditions.'),
    'ENCOUNTER_STATUS': ('Encounter Status', 'Encounter completion rates indicate scheduling efficiency and care delivery throughput.'),
    'ACCEPTS_NEW_PATIENTS': ('Accepting New Patients', 'New patient acceptance rates directly impact member access and appointment availability metrics.'),
    'LICENSE_STATE': ('License State', 'License state distribution affects telehealth eligibility and cross-state practice capabilities.'),
    'CATEGORY': ('CPT Category', 'Service category mix reveals procedure volume distribution and revenue composition by service line.'),
    'LENGTH_OF_STAY': ('Length of Stay', 'LOS benchmarking against CMS geometric mean reveals efficiency. Prolonged stays increase cost and readmission risk.'),
    'COPAY': ('Copay Tier', 'Copay distribution indicates member cost-sharing burden and plan benefit design impact on utilization.'),
    'DAYS_SUPPLY': ('Days Supply', 'Days supply patterns indicate adherence behavior. 90-day fills correlate with better PDC outcomes.'),
    'QUANTITY': ('Prescription Quantity', 'Quantity patterns inform dispensing efficiency and formulary compliance.'),
    'CITY': ('City', 'City-level analysis reveals urban vs. suburban utilization patterns and access equity.'),
    'STATE': ('State', 'State-level variation reflects regulatory environment, Medicaid expansion status, and market dynamics.'),
    'RISK_SCORE': ('Risk Score', 'CMS HCC risk scores drive Medicare Advantage capitation. Accurate risk capture represents material revenue opportunity.'),
}

TABLE_LABELS = {
    'claims': 'Claims', 'members': 'Members', 'encounters': 'Encounters',
    'providers': 'Providers', 'appointments': 'Appointments',
    'prescriptions': 'Prescriptions', 'diagnoses': 'Diagnoses',
    'referrals': 'Referrals', 'cpt_codes': 'CPT Codes',
}

TABLE_KEYWORD_MAP = {
    'claims':        ['claim', 'claims', 'denial', 'denied', 'billed', 'paid', 'cost',
                      'revenue', 'financial', 'money', 'reimbursement'],
    'members':       ['member', 'patient', 'enrollee', 'demographic', 'population',
                      'enrollment', 'who', 'people', 'age', 'gender', 'race', 'language'],
    'encounters':    ['encounter', 'visit', 'admission', 'discharge', 'inpatient',
                      'outpatient', 'emergency', 'ed ', 'utilization', 'los', 'length of stay'],
    'providers':     ['doctor', 'physician', 'specialist', 'npi', 'practitioner',
                      'network', 'staffing', 'workforce', 'panel', 'clinician'],
    'appointments':  ['appointment', 'schedule', 'no show', 'no-show', 'cancel',
                      'booking', 'access', 'wait time'],
    'prescriptions': ['prescription', 'medication', 'drug', 'pharmacy', 'rx',
                      'refill', 'adherence', 'formulary'],
    'diagnoses':     ['diagnosis', 'diagnoses', 'condition', 'disease', 'icd',
                      'chronic', 'hcc', 'comorbid', 'prevalence'],
    'referrals':     ['referral', 'refer', 'specialist referral', 'authorization',
                      'auth', 'network leakage'],
}

CONCEPT_TRIGGERS = {
    'demographics': ['demographic', 'demographics', 'population', 'who are our members',
                     'member profile', 'member breakdown', 'patient population',
                     'member characteristics'],
    'financials':   ['financial', 'financials', 'revenue', 'cost overview',
                     'financial summary', 'cost analysis', 'spending',
                     'give me the numbers', 'financial health', 'fiscal'],
    'quality':      ['quality', 'quality metrics', 'outcomes', 'performance',
                     'hedis', 'stars', 'clinical quality', 'how are we doing',
                     'quality scores'],
    'utilization':  ['utilization overview', 'utilization summary',
                     'service utilization', 'visit patterns', 'care utilization'],
    'provider_network': ['provider network', 'network adequacy',
                         'provider overview', 'workforce overview'],
    'patient_journey':  ['patient journey', 'member journey', 'care journey',
                         'member timeline', 'patient timeline', 'care timeline',
                         'episode of care', 'care pathway', 'treatment pathway',
                         'patient experience', 'care continuum',
                         'what happened to', 'sequence of events',
                         'patient history', 'care history'],
}


STATIC_CONCEPTS = {
    'demographics': {
        'label': 'Member Demographics',
        'description': 'Population characteristics across all demographic dimensions',
        'primary_table': 'members',
        'curated_dimensions': [
            {'label': 'Gender Distribution', 'sql': "SELECT GENDER as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY GENDER ORDER BY member_count DESC", 'chart_type': 'pie'},
            {'label': 'Race & Ethnicity', 'sql': "SELECT RACE as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY RACE ORDER BY member_count DESC", 'chart_type': 'bar'},
            {'label': 'Age Distribution', 'sql': "SELECT CASE WHEN CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT)<18 THEN 'Pediatric (0-17)' WHEN CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 18 AND 25 THEN 'Young Adult (18-25)' WHEN CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 26 AND 40 THEN 'Adult (26-40)' WHEN CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 41 AND 55 THEN 'Middle Age (41-55)' WHEN CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 56 AND 64 THEN 'Pre-Medicare (56-64)' ELSE 'Medicare (65+)' END as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY category ORDER BY MIN(CAST((julianday('now')-julianday(DATE_OF_BIRTH))/365.25 AS INT))", 'chart_type': 'bar'},
            {'label': 'Preferred Language', 'sql': "SELECT LANGUAGE as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY LANGUAGE ORDER BY member_count DESC", 'chart_type': 'bar'},
            {'label': 'KP Region', 'sql': "SELECT KP_REGION as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY KP_REGION ORDER BY member_count DESC", 'chart_type': 'bar'},
            {'label': 'Plan Type', 'sql': "SELECT PLAN_TYPE as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY PLAN_TYPE ORDER BY member_count DESC", 'chart_type': 'pie'},
            {'label': 'Risk Score Distribution', 'sql': "SELECT CASE WHEN CAST(RISK_SCORE AS FLOAT)<1.0 THEN 'Low (<1.0)' WHEN CAST(RISK_SCORE AS FLOAT) BETWEEN 1.0 AND 2.0 THEN 'Moderate (1.0-2.0)' WHEN CAST(RISK_SCORE AS FLOAT) BETWEEN 2.0 AND 3.5 THEN 'High (2.0-3.5)' ELSE 'Very High (>3.5)' END as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct, ROUND(AVG(CAST(RISK_SCORE AS FLOAT)),2) as avg_risk FROM members GROUP BY category ORDER BY MIN(CAST(RISK_SCORE AS FLOAT))", 'chart_type': 'bar'},
            {'label': 'Chronic Condition Burden', 'sql': "SELECT CASE WHEN CAST(CHRONIC_CONDITIONS AS INT)=0 THEN 'None' WHEN CAST(CHRONIC_CONDITIONS AS INT)=1 THEN '1 Condition' WHEN CAST(CHRONIC_CONDITIONS AS INT)=2 THEN '2 Conditions' WHEN CAST(CHRONIC_CONDITIONS AS INT) BETWEEN 3 AND 4 THEN '3-4 Conditions' ELSE '5+ (Complex)' END as category, COUNT(*) as member_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM members),1) as pct FROM members GROUP BY category ORDER BY MIN(CAST(CHRONIC_CONDITIONS AS INT))", 'chart_type': 'bar'},
        ],
    },
    'financials': {
        'label': 'Financial Overview',
        'description': 'Comprehensive financial analysis across cost and revenue dimensions',
        'primary_table': 'claims',
        'curated_dimensions': [
            {'label': 'Claims Summary', 'sql': "SELECT COUNT(*) as total_claims, ROUND(SUM(CAST(BILLED_AMOUNT AS FLOAT)),2) as total_billed, ROUND(SUM(CAST(PAID_AMOUNT AS FLOAT)),2) as total_paid, ROUND(AVG(CAST(PAID_AMOUNT AS FLOAT)),2) as avg_paid, ROUND(100.0*SUM(CAST(PAID_AMOUNT AS FLOAT))/NULLIF(SUM(CAST(BILLED_AMOUNT AS FLOAT)),0),1) as payment_rate_pct FROM claims WHERE CLAIM_STATUS='PAID'", 'chart_type': 'kpi'},
            {'label': 'Cost by Plan Type', 'sql': "SELECT PLAN_TYPE as category, COUNT(*) as claims, ROUND(SUM(CAST(PAID_AMOUNT AS FLOAT)),2) as total_paid, ROUND(AVG(CAST(PAID_AMOUNT AS FLOAT)),2) as avg_paid FROM claims GROUP BY PLAN_TYPE ORDER BY total_paid DESC", 'chart_type': 'bar'},
            {'label': 'Cost by Claim Type', 'sql': "SELECT CLAIM_TYPE as category, COUNT(*) as claims, ROUND(SUM(CAST(PAID_AMOUNT AS FLOAT)),2) as total_paid, ROUND(AVG(CAST(PAID_AMOUNT AS FLOAT)),2) as avg_paid FROM claims GROUP BY CLAIM_TYPE ORDER BY total_paid DESC", 'chart_type': 'pie'},
            {'label': 'Cost by Region', 'sql': "SELECT KP_REGION as category, COUNT(*) as claims, ROUND(SUM(CAST(PAID_AMOUNT AS FLOAT)),2) as total_paid, ROUND(AVG(CAST(PAID_AMOUNT AS FLOAT)),2) as avg_paid FROM claims GROUP BY KP_REGION ORDER BY total_paid DESC", 'chart_type': 'bar'},
            {'label': 'Denial Financial Impact', 'sql': "SELECT COUNT(*) as denied_claims, ROUND(SUM(CAST(BILLED_AMOUNT AS FLOAT)),2) as denied_billed, ROUND(AVG(CAST(BILLED_AMOUNT AS FLOAT)),2) as avg_denied, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM claims),1) as denial_rate_pct FROM claims WHERE CLAIM_STATUS='DENIED'", 'chart_type': 'kpi'},
            {'label': 'Cost by Facility', 'sql': "SELECT FACILITY as category, COUNT(*) as claims, ROUND(SUM(CAST(PAID_AMOUNT AS FLOAT)),2) as total_paid FROM claims GROUP BY FACILITY ORDER BY total_paid DESC LIMIT 15", 'chart_type': 'bar'},
        ],
    },
    'quality': {
        'label': 'Clinical Quality Overview',
        'description': 'Quality measures spanning outcomes, process metrics, and patient safety',
        'primary_table': 'claims',
        'curated_dimensions': [
            {'label': 'Denial by Reason', 'sql': "SELECT DENIAL_REASON as category, COUNT(*) as count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'),1) as pct FROM claims WHERE CLAIM_STATUS='DENIED' AND DENIAL_REASON!='' GROUP BY DENIAL_REASON ORDER BY count DESC", 'chart_type': 'bar'},
            {'label': 'Top Diagnoses', 'sql': "SELECT ICD10_DESCRIPTION as category, COUNT(*) as count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM diagnoses),1) as pct FROM diagnoses GROUP BY ICD10_DESCRIPTION ORDER BY count DESC LIMIT 10", 'chart_type': 'bar'},
            {'label': 'HCC Categories', 'sql': "SELECT HCC_CATEGORY as category, COUNT(*) as count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM diagnoses WHERE HCC_CATEGORY!='None'),1) as pct FROM diagnoses WHERE HCC_CATEGORY!='None' AND HCC_CATEGORY!='' GROUP BY HCC_CATEGORY ORDER BY count DESC", 'chart_type': 'bar'},
            {'label': 'Medication Adherence', 'sql': "SELECT MEDICATION_CLASS as category, COUNT(*) as rx_count, ROUND(AVG(CAST(REFILLS_USED AS FLOAT)/NULLIF(CAST(REFILLS_AUTHORIZED AS FLOAT),0)*100),1) as adherence_pct FROM prescriptions WHERE STATUS='FILLED' GROUP BY MEDICATION_CLASS ORDER BY rx_count DESC", 'chart_type': 'bar'},
            {'label': 'Diagnosis Severity', 'sql': "SELECT SEVERITY as category, COUNT(*) as count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM diagnoses),1) as pct FROM diagnoses GROUP BY SEVERITY ORDER BY count DESC", 'chart_type': 'pie'},
        ],
    },
    'patient_journey': {
        'label': 'Patient Journey — Cross-System Care Timeline',
        'description': 'Unified view across all care touchpoints: encounters, claims, prescriptions, diagnoses, referrals, and appointments',
        'primary_table': 'members',
        'curated_dimensions': [
            {'label': 'Care Touchpoint Volume',
             'sql': (
                 "SELECT 'Encounters' as touchpoint, COUNT(*) as events FROM encounters "
                 "UNION ALL SELECT 'Claims', COUNT(*) FROM claims "
                 "UNION ALL SELECT 'Prescriptions', COUNT(*) FROM prescriptions "
                 "UNION ALL SELECT 'Diagnoses', COUNT(*) FROM diagnoses "
                 "UNION ALL SELECT 'Referrals', COUNT(*) FROM referrals "
                 "UNION ALL SELECT 'Appointments', COUNT(*) FROM appointments "
                 "ORDER BY events DESC"
             ),
             'chart_type': 'bar',
             'medical_context': 'Touchpoint volume reveals system utilization patterns. High claim-to-encounter ratios may indicate unbundling; low referral completion signals care gaps.'},
            {'label': 'Patient Reach by System',
             'sql': (
                 "SELECT 'Encounters' as system, COUNT(DISTINCT MEMBER_ID) as unique_patients FROM encounters "
                 "UNION ALL SELECT 'Claims', COUNT(DISTINCT MEMBER_ID) FROM claims "
                 "UNION ALL SELECT 'Prescriptions', COUNT(DISTINCT MEMBER_ID) FROM prescriptions "
                 "UNION ALL SELECT 'Diagnoses', COUNT(DISTINCT MEMBER_ID) FROM diagnoses "
                 "UNION ALL SELECT 'Referrals', COUNT(DISTINCT MEMBER_ID) FROM referrals "
                 "UNION ALL SELECT 'Appointments', COUNT(DISTINCT MEMBER_ID) FROM appointments "
                 "ORDER BY unique_patients DESC"
             ),
             'chart_type': 'bar',
             'medical_context': 'Patient reach shows coverage gaps. Members appearing in encounters but not claims may have billing issues; those with diagnoses but no prescriptions may lack treatment follow-through.'},
            {'label': 'Avg Events per Patient',
             'sql': (
                 "SELECT 'Encounters' as type, ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) as avg_events FROM encounters "
                 "UNION ALL SELECT 'Claims', ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) FROM claims "
                 "UNION ALL SELECT 'Prescriptions', ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) FROM prescriptions "
                 "UNION ALL SELECT 'Diagnoses', ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) FROM diagnoses "
                 "UNION ALL SELECT 'Referrals', ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) FROM referrals "
                 "UNION ALL SELECT 'Appointments', ROUND(1.0*COUNT(*)/COUNT(DISTINCT MEMBER_ID),1) FROM appointments "
                 "ORDER BY avg_events DESC"
             ),
             'chart_type': 'bar',
             'medical_context': 'Event intensity per patient signals care complexity. High ratios indicate frequent utilizers; low ratios may indicate underserved populations.'},
            {'label': 'Monthly Care Timeline',
             'sql': (
                 "SELECT month, SUM(encounters) as encounters, SUM(claims) as claims, "
                 "SUM(prescriptions) as prescriptions, SUM(appointments) as appointments "
                 "FROM ("
                 "  SELECT strftime('%Y-%m', SERVICE_DATE) as month, COUNT(*) as encounters, 0 as claims, 0 as prescriptions, 0 as appointments FROM encounters WHERE SERVICE_DATE != '' GROUP BY month "
                 "  UNION ALL SELECT strftime('%Y-%m', SERVICE_DATE), 0, COUNT(*), 0, 0 FROM claims WHERE SERVICE_DATE != '' GROUP BY 1 "
                 "  UNION ALL SELECT strftime('%Y-%m', FILL_DATE), 0, 0, COUNT(*), 0 FROM prescriptions WHERE FILL_DATE != '' GROUP BY 1 "
                 "  UNION ALL SELECT strftime('%Y-%m', APPOINTMENT_DATE), 0, 0, 0, COUNT(*) FROM appointments WHERE APPOINTMENT_DATE != '' GROUP BY 1 "
                 ") GROUP BY month ORDER BY month"
             ),
             'chart_type': 'line',
             'medical_context': 'Temporal alignment of care events reveals operational cadence, seasonal patterns, and system throughput. Divergence between encounters and claims may indicate billing lag.'},
            {'label': 'Journey Completeness',
             'sql': (
                 "SELECT ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM encounters)/(SELECT COUNT(*) FROM members),1) as pct_with_encounters, "
                 "ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM claims)/(SELECT COUNT(*) FROM members),1) as pct_with_claims, "
                 "ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM prescriptions)/(SELECT COUNT(*) FROM members),1) as pct_with_prescriptions, "
                 "ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM diagnoses)/(SELECT COUNT(*) FROM members),1) as pct_with_diagnoses, "
                 "ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM referrals)/(SELECT COUNT(*) FROM members),1) as pct_with_referrals, "
                 "ROUND(100.0*(SELECT COUNT(DISTINCT MEMBER_ID) FROM appointments)/(SELECT COUNT(*) FROM members),1) as pct_with_appointments"
             ),
             'chart_type': 'kpi',
             'medical_context': 'Journey completeness shows what proportion of the member population has each type of care event. Low percentages reveal engagement gaps; 100% penetration in claims but low in prescriptions may indicate a population not on maintenance therapy.'},
            {'label': 'Encounter Type Distribution',
             'sql': "SELECT VISIT_TYPE as category, COUNT(*) as count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM encounters),1) as pct FROM encounters GROUP BY VISIT_TYPE ORDER BY count DESC",
             'chart_type': 'pie',
             'medical_context': 'Visit type mix reveals care delivery model. High ED ratios signal access issues; high outpatient indicates managed care effectiveness.'},
            {'label': 'Top Conditions Driving Care',
             'sql': "SELECT ICD10_DESCRIPTION as condition, COUNT(DISTINCT MEMBER_ID) as patients, COUNT(*) as total_diagnoses, ROUND(100.0*COUNT(DISTINCT MEMBER_ID)/(SELECT COUNT(*) FROM members),1) as pct_population FROM diagnoses GROUP BY ICD10_DESCRIPTION ORDER BY patients DESC LIMIT 10",
             'chart_type': 'bar',
             'medical_context': 'Top conditions represent the primary drivers of patient journeys. These determine care pathway design, resource allocation, and clinical program priorities.'},
            {'label': 'Sample Patient Timeline (1 Patient)',
             'sql': (
                 "WITH target AS (SELECT MEMBER_ID FROM encounters GROUP BY MEMBER_ID HAVING COUNT(*) >= 3 ORDER BY COUNT(*) DESC LIMIT 1), "
                 "events AS ("
                 "  SELECT e.MEMBER_ID, e.SERVICE_DATE as event_date, 'Encounter' as event_type, e.VISIT_TYPE || ' - ' || e.DEPARTMENT as detail FROM encounters e JOIN target t ON e.MEMBER_ID = t.MEMBER_ID "
                 "  UNION ALL SELECT c.MEMBER_ID, c.SERVICE_DATE, 'Claim', c.CLAIM_TYPE || ' $' || c.PAID_AMOUNT FROM claims c JOIN target t ON c.MEMBER_ID = t.MEMBER_ID "
                 "  UNION ALL SELECT p.MEMBER_ID, p.PRESCRIPTION_DATE, 'Rx', p.MEDICATION_NAME FROM prescriptions p JOIN target t ON p.MEMBER_ID = t.MEMBER_ID "
                 "  UNION ALL SELECT d.MEMBER_ID, d.DIAGNOSIS_DATE, 'Diagnosis', d.ICD10_DESCRIPTION FROM diagnoses d JOIN target t ON d.MEMBER_ID = t.MEMBER_ID "
                 "  UNION ALL SELECT r.MEMBER_ID, r.REFERRAL_DATE, 'Referral', r.SPECIALTY || ' - ' || r.REFERRAL_REASON FROM referrals r JOIN target t ON r.MEMBER_ID = t.MEMBER_ID "
                 "  UNION ALL SELECT a.MEMBER_ID, a.APPOINTMENT_DATE, 'Appointment', a.APPOINTMENT_TYPE || ' - ' || a.DEPARTMENT FROM appointments a JOIN target t ON a.MEMBER_ID = t.MEMBER_ID"
                 ") SELECT event_date, event_type, detail FROM events WHERE event_date IS NOT NULL AND event_date != '' ORDER BY event_date, event_type LIMIT 50"
             ),
             'chart_type': 'table',
             'medical_context': 'A sample patient timeline illustrates the actual care pathway: how events chain together chronologically. This reveals care coordination quality, follow-up adherence, and care fragmentation.'},
        ],
    },
}


class ConceptExpander:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.profiler = SchemaProfiler(db_path)

    def detect_concept(self, question: str) -> Optional[str]:
        q = question.lower().strip()

        for concept_key, triggers in CONCEPT_TRIGGERS.items():
            for trigger in triggers:
                if trigger in q:
                    return concept_key

        specific_intent_words = {
            'rate', 'average', 'avg', 'cost', 'total', 'count', 'sum',
            'top', 'highest', 'lowest', 'best', 'worst', 'compare', 'vs',
            'trend', 'monthly', 'by', 'per', 'between', 'correlation',
            'why', 'reason', 'which', 'specific', 'particular',
            'how', 'number', 'percentage', 'ratio', 'each', 'group',
            'denied', 'approved', 'rejected', 'pending', 'cancelled',
            'more', 'less', 'above', 'below', 'than', 'over', 'under',
            'most', 'least', 'many', 'few', 'where',
            'maximum', 'minimum', 'max', 'min', 'largest', 'smallest',
            'biggest', 'single', 'median', 'std', 'deviation', 'variance',
        }
        q_words_set = set(re.findall(r'\b\w+\b', q))
        has_specific_intent = bool(q_words_set & specific_intent_words)

        if not has_specific_intent:
            broad_patterns = [
                r'show\s+(?:me\s+)?(?:the\s+)?(\w+)\s*(?:data|info|information|breakdown|overview|analysis)?$',
                r'(?:tell|show)\s+(?:me\s+)?about\s+(?:the\s+)?(\w+)$',
                r'what\s+(?:is|about|are)\s+(?:the\s+)?(?:a\s+)?(\w+)$',
                r'give\s+(?:me\s+)?(?:the\s+)?(\w+)\s*(?:data|info|breakdown|overview)?$',
                r'(?:analyze|analyse)\s+(?:the\s+)?(\w+)$',
                r'(\w+)\s+(?:overview|summary|breakdown|analysis|data)$',
            ]

            for pattern in broad_patterns:
                match = re.search(pattern, q)
                if match:
                    word = match.group(1).lower()
                    table = self._word_to_table(word)
                    if table:
                        return f'dynamic:{table}'

        min_matches = 3 if has_specific_intent else 2
        q_words = set(re.findall(r'\b\w+\b', q))
        for table, keywords in TABLE_KEYWORD_MAP.items():
            matched_q_words = set()
            for kw in keywords:
                if kw in q_words:
                    matched_q_words.add(kw)
                elif len(kw) > 3:
                    for w in q_words:
                        if w.startswith(kw):
                            matched_q_words.add(w)
            if len(matched_q_words) >= min_matches:
                return f'dynamic:{table}'

        return None

    def _word_to_table(self, word: str) -> Optional[str]:
        for table in self.profiler.tables:
            if word == table.lower() or word == table.lower() + 's' or word + 's' == table.lower():
                return table
        for table, keywords in TABLE_KEYWORD_MAP.items():
            if word in keywords or word + 's' in keywords:
                return table
        return None

    def expand(self, question: str, concept_key: str = None) -> Optional[Dict]:
        if not concept_key:
            concept_key = self.detect_concept(question)
        if not concept_key:
            return None

        t0 = time.time()

        if concept_key in STATIC_CONCEPTS:
            result = self._expand_static(concept_key)
        elif concept_key.startswith('dynamic:'):
            table = concept_key.split(':', 1)[1]
            result = self._expand_dynamic(table, question)
        else:
            return None

        if not result:
            return None

        result['execution_time_ms'] = round((time.time() - t0) * 1000)
        return result


    def _expand_static(self, concept_key: str) -> Optional[Dict]:
        concept = STATIC_CONCEPTS[concept_key]
        results = []

        try:
            conn = sqlite3.connect(self.db_path)
            for dim_def in concept['curated_dimensions']:
                dim_result = self._execute_dimension(conn, dim_def)
                results.append(dim_result)
            conn.close()
        except Exception as e:
            logger.error("Static expansion error: %s", e)
            return None

        return {
            'is_concept': True,
            'concept': concept_key,
            'label': concept['label'],
            'description': concept['description'],
            'dimensions': results,
            'total_dimensions': len(results),
        }


    def _expand_dynamic(self, table: str, question: str) -> Optional[Dict]:
        if table not in self.profiler.tables:
            return None

        dimensions = self.profiler.get_dimensions(table)
        if not dimensions:
            return None

        table_label = TABLE_LABELS.get(table, table.title())
        row_count = self.profiler.tables[table]['row_count']
        results = []

        try:
            conn = sqlite3.connect(self.db_path)

            for dim_prof in dimensions:
                col_name = dim_prof['name']
                distinct = dim_prof['distinct']

                skip_cols = {
                    'CHECK_IN_TIME', 'CHECK_OUT_TIME', 'APPOINTMENT_TIME',
                    'CPT_CODE', 'CPT_DESCRIPTION', 'ICD10_CODE', 'ICD10_DESCRIPTION',
                    'MEDICATION_NAME', 'PRIMARY_DIAGNOSIS', 'DIAGNOSIS_DESCRIPTION',
                    'CHIEF_COMPLAINT', 'REASON',
                }
                if col_name.upper() in skip_cols:
                    continue

                human_label, medical_context = self._get_column_context(col_name)
                limit_clause = 'LIMIT 20' if distinct > 20 else ''

                sql = (
                    f"SELECT [{col_name}] as category, COUNT(*) as count, "
                    f"ROUND(100.0 * COUNT(*) / {row_count}, 1) as pct "
                    f"FROM {table} "
                    f"WHERE [{col_name}] IS NOT NULL AND [{col_name}] != '' "
                    f"GROUP BY [{col_name}] ORDER BY count DESC {limit_clause}"
                )

                chart_type = 'pie' if distinct <= 4 else 'bar'

                dim_def = {
                    'label': human_label,
                    'sql': sql,
                    'chart_type': chart_type,
                    'medical_context': medical_context,
                }
                dim_result = self._execute_dimension(conn, dim_def)
                if dim_result and dim_result['row_count'] > 0:
                    results.append(dim_result)

            temporals = self.profiler.get_temporals(table)
            if temporals:
                main_date = temporals[0]['name']
                temporal_sql = (
                    f"SELECT strftime('%Y-%m', [{main_date}]) as month, "
                    f"COUNT(*) as count "
                    f"FROM {table} WHERE [{main_date}] != '' "
                    f"GROUP BY month ORDER BY month"
                )
                dim_def = {
                    'label': f'Monthly Trend ({main_date.replace("_", " ").title()})',
                    'sql': temporal_sql,
                    'chart_type': 'line',
                    'medical_context': 'Temporal patterns reveal seasonality, growth trends, and operational throughput changes.',
                }
                dim_result = self._execute_dimension(conn, dim_def)
                if dim_result and dim_result['row_count'] > 1:
                    results.append(dim_result)

            metrics = self.profiler.get_metrics(table)
            metric_cols = [m for m in metrics if m['name'].upper() not in
                          ('NPI', 'RENDERING_NPI', 'BILLING_NPI', 'SUPERVISING_NPI',
                           'PCP_NPI', 'PRESCRIBING_NPI', 'REFERRING_NPI',
                           'REFERRED_TO_NPI', 'DIAGNOSING_NPI', 'ZIP_CODE',
                           'CPT_CODE')]
            if metric_cols:
                agg_parts = []
                for m in metric_cols[:5]:
                    mn = m['name']
                    agg_parts.append(
                        f"ROUND(AVG(CAST([{mn}] AS FLOAT)),2) as avg_{mn.lower()}, "
                        f"ROUND(MIN(CAST([{mn}] AS FLOAT)),2) as min_{mn.lower()}, "
                        f"ROUND(MAX(CAST([{mn}] AS FLOAT)),2) as max_{mn.lower()}"
                    )
                metric_sql = f"SELECT COUNT(*) as total_records, {', '.join(agg_parts)} FROM {table}"
                dim_def = {
                    'label': 'Key Metrics Summary',
                    'sql': metric_sql,
                    'chart_type': 'kpi',
                    'medical_context': 'Summary statistics provide baseline reference for identifying outliers and benchmarking.',
                }
                dim_result = self._execute_dimension(conn, dim_def)
                if dim_result:
                    results.insert(0, dim_result)

            conn.close()
        except Exception as e:
            logger.error("Dynamic expansion error for %s: %s", table, e)
            return None

        if not results:
            return None

        return {
            'is_concept': True,
            'concept': f'dynamic:{table}',
            'label': f'{table_label} — Comprehensive Analysis',
            'description': f'Auto-discovered {len(results)} dimensions across {row_count:,} {table_label.lower()} records',
            'dimensions': results,
            'total_dimensions': len(results),
        }


    def _get_column_context(self, col_name: str) -> Tuple[str, str]:
        upper = col_name.upper()
        if upper in COLUMN_CONTEXT:
            return COLUMN_CONTEXT[upper]
        human_label = col_name.replace('_', ' ').title()
        return (human_label, '')

    def _execute_dimension(self, conn, dim_def: Dict) -> Optional[Dict]:
        try:
            cursor = conn.execute(dim_def['sql'])
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            insight = self._generate_insight(rows, columns)

            return {
                'label': dim_def['label'],
                'rows': rows,
                'columns': columns,
                'chart_type': dim_def.get('chart_type', 'bar'),
                'insight': insight,
                'medical_context': dim_def.get('medical_context', ''),
                'row_count': len(rows),
                'sql': dim_def['sql'],
                'error': None,
            }
        except Exception as e:
            return {
                'label': dim_def['label'],
                'rows': [], 'columns': [], 'chart_type': 'bar',
                'insight': f"Unable to compute: {str(e)[:80]}",
                'medical_context': '', 'row_count': 0,
                'sql': dim_def['sql'], 'error': str(e),
            }

    def _generate_insight(self, rows: List, columns: List) -> str:
        if not rows:
            return "No data available."

        if len(rows) == 1 and len(columns) > 3:
            parts = []
            for i, col in enumerate(columns):
                val = rows[0][i]
                if val is not None and isinstance(val, (int, float)):
                    label = col.replace('_', ' ').title()
                    if isinstance(val, float):
                        parts.append(f"{label}: {val:,.2f}")
                    else:
                        parts.append(f"{label}: {val:,}")
            return " | ".join(parts) if parts else f"Found {len(columns)} metrics."

        count_idx = 1 if len(columns) > 1 else 0
        total = sum(r[count_idx] for r in rows
                    if len(r) > count_idx and isinstance(r[count_idx], (int, float)))

        if total == 0:
            return f"Found {len(rows)} categories."

        sorted_rows = sorted(
            [r for r in rows if len(r) > count_idx and isinstance(r[count_idx], (int, float))],
            key=lambda r: r[count_idx], reverse=True
        )

        top = sorted_rows[0] if sorted_rows else rows[0]
        second = sorted_rows[1] if len(sorted_rows) > 1 else None

        top_cat = str(top[0])
        top_count = top[count_idx]
        top_pct = round(top_count / total * 100, 1) if total > 0 else 0

        pct_idx = next((i for i, c in enumerate(columns) if 'pct' in c.lower()), None)
        if pct_idx is not None and len(top) > pct_idx and isinstance(top[pct_idx], (int, float)):
            top_pct = top[pct_idx]

        record_label = 'members' if any('member' in c.lower() for c in columns) else 'records'

        parts = [f"{top_cat} leads at {top_pct}% ({top_count:,} {record_label})"]
        if second:
            s_cat = str(second[0])
            s_pct = round(second[count_idx] / total * 100, 1) if total > 0 else 0
            if pct_idx is not None and len(second) > pct_idx and isinstance(second[pct_idx], (int, float)):
                s_pct = second[pct_idx]
            parts[0] += f", followed by {s_cat} ({s_pct}%)."
        else:
            parts[0] += "."

        parts.append(f"{len(rows)} categories across {total:,} total {record_label}.")

        if top_pct > 50:
            parts.append("High concentration in top category — investigate distribution equity.")
        elif len(rows) > 3 and top_pct < 20:
            parts.append("Relatively even distribution across categories.")

        return " ".join(parts)

    def get_available_concepts(self) -> List[Dict]:
        results = []
        for key, concept in STATIC_CONCEPTS.items():
            results.append({
                'key': key,
                'label': concept['label'],
                'description': concept['description'],
                'dimension_count': len(concept['curated_dimensions']),
            })
        for table, profile in self.profiler.tables.items():
            dims = [c for c in profile['columns'].values() if c['type'] == 'categorical']
            if dims:
                results.append({
                    'key': f'dynamic:{table}',
                    'label': f'{TABLE_LABELS.get(table, table.title())} Analysis',
                    'description': f'{len(dims)} dimensions, {profile["row_count"]:,} records',
                    'dimension_count': len(dims),
                })
        return results
