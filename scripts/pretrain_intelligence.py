import json
import logging
import os
import sqlite3
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger('pretrain')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


TRAINING_QUERIES = [
    {"q": "How many claims do we have?", "intent": "count",
     "sql": 'SELECT COUNT(*) AS total_claims FROM claims',
     "tables": ["claims"], "columns": ["CLAIM_ID"]},
    {"q": "How many members are enrolled?", "intent": "count",
     "sql": 'SELECT COUNT(DISTINCT MEMBER_ID) AS total_members FROM members',
     "tables": ["members"], "columns": ["MEMBER_ID"]},
    {"q": "How many encounters happened?", "intent": "count",
     "sql": 'SELECT COUNT(*) AS total_encounters FROM encounters',
     "tables": ["encounters"], "columns": ["ENCOUNTER_ID"]},
    {"q": "How many providers do we have?", "intent": "count",
     "sql": 'SELECT COUNT(*) AS total_providers FROM providers',
     "tables": ["providers"], "columns": ["NPI"]},
    {"q": "How many denied claims are there?", "intent": "count",
     "sql": "SELECT COUNT(*) AS denied_claims FROM claims WHERE CLAIM_STATUS = 'DENIED'",
     "tables": ["claims"], "columns": ["CLAIM_ID", "CLAIM_STATUS"]},
    {"q": "How many ER visits?", "intent": "count",
     "sql": "SELECT COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY'",
     "tables": ["encounters"], "columns": ["ENCOUNTER_ID", "VISIT_TYPE"]},
    {"q": "How many inpatient admissions?", "intent": "count",
     "sql": "SELECT COUNT(*) AS inpatient_admissions FROM encounters WHERE VISIT_TYPE = 'INPATIENT'",
     "tables": ["encounters"], "columns": ["ENCOUNTER_ID", "VISIT_TYPE"]},
    {"q": "How many chronic diagnoses?", "intent": "count",
     "sql": "SELECT COUNT(*) AS chronic_diagnoses FROM diagnoses WHERE IS_CHRONIC = 'Y'",
     "tables": ["diagnoses"], "columns": ["DIAGNOSIS_ID", "IS_CHRONIC"]},
    {"q": "How many prescriptions were written?", "intent": "count",
     "sql": 'SELECT COUNT(*) AS total_prescriptions FROM prescriptions',
     "tables": ["prescriptions"], "columns": ["RX_ID"]},
    {"q": "How many referrals are pending?", "intent": "count",
     "sql": "SELECT COUNT(*) AS pending_referrals FROM referrals WHERE STATUS = 'PENDING'",
     "tables": ["referrals"], "columns": ["REFERRAL_ID", "STATUS"]},
    {"q": "Total number of appointments", "intent": "count",
     "sql": 'SELECT COUNT(*) AS total_appointments FROM appointments',
     "tables": ["appointments"], "columns": ["APPOINTMENT_ID"]},
    {"q": "Count of unique diagnoses", "intent": "count",
     "sql": 'SELECT COUNT(DISTINCT ICD10_CODE) AS unique_diagnoses FROM diagnoses',
     "tables": ["diagnoses"], "columns": ["ICD10_CODE"]},

    {"q": "What is the total cost of all claims?", "intent": "aggregate",
     "sql": 'SELECT SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims',
     "tables": ["claims"], "columns": ["PAID_AMOUNT"]},
    {"q": "What is the average claim amount?", "intent": "aggregate",
     "sql": 'SELECT AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims WHERE PAID_AMOUNT > 0',
     "tables": ["claims"], "columns": ["PAID_AMOUNT"]},
    {"q": "What is the total billed amount?", "intent": "aggregate",
     "sql": 'SELECT SUM(CAST(BILLED_AMOUNT AS REAL)) AS total_billed FROM claims',
     "tables": ["claims"], "columns": ["BILLED_AMOUNT"]},
    {"q": "Average length of stay", "intent": "aggregate",
     "sql": "SELECT AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND LENGTH_OF_STAY > 0",
     "tables": ["encounters"], "columns": ["LENGTH_OF_STAY"]},
    {"q": "What is the average risk score?", "intent": "aggregate",
     "sql": "SELECT AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk_score FROM members WHERE RISK_SCORE IS NOT NULL",
     "tables": ["members"], "columns": ["RISK_SCORE"]},
    {"q": "Total member responsibility", "intent": "aggregate",
     "sql": 'SELECT SUM(CAST(MEMBER_RESPONSIBILITY AS REAL)) AS total_member_resp FROM claims',
     "tables": ["claims"], "columns": ["MEMBER_RESPONSIBILITY"]},
    {"q": "Average prescription cost", "intent": "aggregate",
     "sql": 'SELECT AVG(CAST(COST AS REAL)) AS avg_rx_cost FROM prescriptions WHERE COST > 0',
     "tables": ["prescriptions"], "columns": ["COST"]},
    {"q": "Total copay collected", "intent": "aggregate",
     "sql": 'SELECT SUM(CAST(COPAY AS REAL)) AS total_copay FROM claims',
     "tables": ["claims"], "columns": ["COPAY"]},

    {"q": "What is our denial rate?", "intent": "rate",
     "sql": "SELECT ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct FROM claims",
     "tables": ["claims"], "columns": ["CLAIM_STATUS"]},
    {"q": "What is the yield rate?", "intent": "rate",
     "sql": "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) * 100.0 / NULLIF(SUM(CAST(BILLED_AMOUNT AS REAL)), 0), 2) AS yield_rate_pct FROM claims WHERE BILLED_AMOUNT > 0",
     "tables": ["claims"], "columns": ["PAID_AMOUNT", "BILLED_AMOUNT"]},
    {"q": "What is the no-show rate for appointments?", "intent": "rate",
     "sql": "SELECT ROUND(SUM(CASE WHEN STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS no_show_rate_pct FROM appointments",
     "tables": ["appointments"], "columns": ["STATUS"]},
    {"q": "What percentage of diagnoses are chronic?", "intent": "rate",
     "sql": "SELECT ROUND(SUM(CASE WHEN IS_CHRONIC = 'Y' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS chronic_pct FROM diagnoses",
     "tables": ["diagnoses"], "columns": ["IS_CHRONIC"]},
    {"q": "ER utilization rate per 1000 members", "intent": "rate",
     "sql": "SELECT ROUND(COUNT(CASE WHEN VISIT_TYPE = 'EMERGENCY' THEN 1 END) * 1000.0 / NULLIF((SELECT COUNT(DISTINCT MEMBER_ID) FROM members), 0), 2) AS er_per_1000 FROM encounters",
     "tables": ["encounters", "members"], "columns": ["VISIT_TYPE", "MEMBER_ID"]},

    {"q": "Claims by status", "intent": "breakdown",
     "sql": "SELECT CLAIM_STATUS, COUNT(*) AS claim_count FROM claims GROUP BY CLAIM_STATUS ORDER BY claim_count DESC",
     "tables": ["claims"], "columns": ["CLAIM_STATUS"]},
    {"q": "Cost by region", "intent": "breakdown",
     "sql": "SELECT KP_REGION, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, COUNT(*) AS claim_count FROM claims GROUP BY KP_REGION ORDER BY total_paid DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "PAID_AMOUNT"]},
    {"q": "Members by plan type", "intent": "breakdown",
     "sql": "SELECT PLAN_TYPE, COUNT(*) AS member_count FROM members GROUP BY PLAN_TYPE ORDER BY member_count DESC",
     "tables": ["members"], "columns": ["PLAN_TYPE"]},
    {"q": "Encounters by visit type", "intent": "breakdown",
     "sql": "SELECT VISIT_TYPE, COUNT(*) AS encounter_count FROM encounters GROUP BY VISIT_TYPE ORDER BY encounter_count DESC",
     "tables": ["encounters"], "columns": ["VISIT_TYPE"]},
    {"q": "Providers by specialty", "intent": "breakdown",
     "sql": "SELECT SPECIALTY, COUNT(*) AS provider_count FROM providers GROUP BY SPECIALTY ORDER BY provider_count DESC",
     "tables": ["providers"], "columns": ["SPECIALTY"]},
    {"q": "Cost breakdown by plan type", "intent": "breakdown",
     "sql": "SELECT PLAN_TYPE, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid, COUNT(*) AS claim_count FROM claims GROUP BY PLAN_TYPE ORDER BY total_paid DESC",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT"]},
    {"q": "Denial rate by region", "intent": "breakdown",
     "sql": "SELECT KP_REGION, ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct, COUNT(*) AS total_claims FROM claims GROUP BY KP_REGION ORDER BY denial_rate_pct DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "CLAIM_STATUS"]},
    {"q": "Diagnoses by severity", "intent": "breakdown",
     "sql": "SELECT SEVERITY, COUNT(*) AS diagnosis_count FROM diagnoses GROUP BY SEVERITY ORDER BY diagnosis_count DESC",
     "tables": ["diagnoses"], "columns": ["SEVERITY"]},
    {"q": "Claims by claim type", "intent": "breakdown",
     "sql": "SELECT CLAIM_TYPE, COUNT(*) AS claim_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY CLAIM_TYPE ORDER BY total_paid DESC",
     "tables": ["claims"], "columns": ["CLAIM_TYPE", "PAID_AMOUNT"]},
    {"q": "Appointments by department", "intent": "breakdown",
     "sql": "SELECT DEPARTMENT, COUNT(*) AS appt_count FROM appointments GROUP BY DEPARTMENT ORDER BY appt_count DESC",
     "tables": ["appointments"], "columns": ["DEPARTMENT"]},
    {"q": "Prescriptions by medication class", "intent": "breakdown",
     "sql": "SELECT MEDICATION_CLASS, COUNT(*) AS rx_count, AVG(CAST(COST AS REAL)) AS avg_cost FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY rx_count DESC",
     "tables": ["prescriptions"], "columns": ["MEDICATION_CLASS", "COST"]},
    {"q": "Referrals by specialty", "intent": "breakdown",
     "sql": "SELECT SPECIALTY, COUNT(*) AS referral_count FROM referrals GROUP BY SPECIALTY ORDER BY referral_count DESC",
     "tables": ["referrals"], "columns": ["SPECIALTY"]},
    {"q": "Members by gender", "intent": "breakdown",
     "sql": "SELECT GENDER, COUNT(*) AS member_count FROM members GROUP BY GENDER ORDER BY member_count DESC",
     "tables": ["members"], "columns": ["GENDER"]},
    {"q": "Members by state", "intent": "breakdown",
     "sql": "SELECT STATE, COUNT(*) AS member_count FROM members GROUP BY STATE ORDER BY member_count DESC LIMIT 20",
     "tables": ["members"], "columns": ["STATE"]},

    {"q": "Monthly claim volume trend", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, COUNT(*) AS claim_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims WHERE SERVICE_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["claims"], "columns": ["SERVICE_DATE", "PAID_AMOUNT"]},
    {"q": "How has cost trended over time?", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims WHERE SERVICE_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["claims"], "columns": ["SERVICE_DATE", "PAID_AMOUNT"]},
    {"q": "Denial rate trend by month", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct FROM claims WHERE SERVICE_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["claims"], "columns": ["SERVICE_DATE", "CLAIM_STATUS"]},
    {"q": "ER visit trend over time", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' AND SERVICE_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["encounters"], "columns": ["SERVICE_DATE", "VISIT_TYPE"]},
    {"q": "Monthly enrollment trend", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', ENROLLMENT_DATE) AS month, COUNT(*) AS new_enrollments FROM members WHERE ENROLLMENT_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["members"], "columns": ["ENROLLMENT_DATE"]},
    {"q": "PMPM trend over time", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', c.SERVICE_DATE) AS month, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT c.MEMBER_ID), 0), 2) AS pmpm FROM claims c WHERE c.SERVICE_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["claims"], "columns": ["SERVICE_DATE", "PAID_AMOUNT", "MEMBER_ID"]},
    {"q": "Prescription volume by month", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', PRESCRIPTION_DATE) AS month, COUNT(*) AS rx_count FROM prescriptions WHERE PRESCRIPTION_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["prescriptions"], "columns": ["PRESCRIPTION_DATE"]},

    {"q": "Top 10 highest cost members", "intent": "ranking",
     "sql": "SELECT c.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, COUNT(*) AS claim_count FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY c.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME ORDER BY total_cost DESC LIMIT 10",
     "tables": ["claims", "members"], "columns": ["MEMBER_ID", "PAID_AMOUNT", "FIRST_NAME", "LAST_NAME"]},
    {"q": "Top providers by claim volume", "intent": "ranking",
     "sql": "SELECT c.RENDERING_NPI, p.PROVIDER_LAST_NAME, p.SPECIALTY, COUNT(*) AS claim_count, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_paid FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY c.RENDERING_NPI, p.PROVIDER_LAST_NAME, p.SPECIALTY ORDER BY claim_count DESC LIMIT 10",
     "tables": ["claims", "providers"], "columns": ["RENDERING_NPI", "NPI", "PROVIDER_LAST_NAME", "SPECIALTY", "PAID_AMOUNT"]},
    {"q": "Most expensive diagnoses", "intent": "ranking",
     "sql": "SELECT c.ICD10_CODE, c.ICD10_DESCRIPTION, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, COUNT(*) AS claim_count FROM claims c WHERE c.ICD10_CODE IS NOT NULL GROUP BY c.ICD10_CODE, c.ICD10_DESCRIPTION ORDER BY total_cost DESC LIMIT 15",
     "tables": ["claims"], "columns": ["ICD10_CODE", "ICD10_DESCRIPTION", "PAID_AMOUNT"]},
    {"q": "Highest cost facilities", "intent": "ranking",
     "sql": "SELECT FACILITY, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, COUNT(*) AS claim_count, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY FACILITY ORDER BY total_paid DESC LIMIT 10",
     "tables": ["claims"], "columns": ["FACILITY", "PAID_AMOUNT"]},
    {"q": "Top 10 most common procedures", "intent": "ranking",
     "sql": "SELECT CPT_CODE, CPT_DESCRIPTION, COUNT(*) AS procedure_count FROM claims WHERE CPT_CODE IS NOT NULL GROUP BY CPT_CODE, CPT_DESCRIPTION ORDER BY procedure_count DESC LIMIT 10",
     "tables": ["claims"], "columns": ["CPT_CODE", "CPT_DESCRIPTION"]},
    {"q": "Which regions have the highest denial rates?", "intent": "ranking",
     "sql": "SELECT KP_REGION, ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct, COUNT(*) AS total_claims FROM claims GROUP BY KP_REGION ORDER BY denial_rate_pct DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "CLAIM_STATUS"]},
    {"q": "Bottom 10 providers by panel size", "intent": "ranking",
     "sql": "SELECT NPI, PROVIDER_LAST_NAME, SPECIALTY, PANEL_SIZE FROM providers WHERE PANEL_SIZE IS NOT NULL AND CAST(PANEL_SIZE AS INTEGER) > 0 ORDER BY CAST(PANEL_SIZE AS INTEGER) ASC LIMIT 10",
     "tables": ["providers"], "columns": ["NPI", "PROVIDER_LAST_NAME", "SPECIALTY", "PANEL_SIZE"]},
    {"q": "Highest risk score members", "intent": "ranking",
     "sql": "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, RISK_SCORE, CHRONIC_CONDITIONS, PLAN_TYPE FROM members WHERE RISK_SCORE IS NOT NULL ORDER BY CAST(RISK_SCORE AS REAL) DESC LIMIT 20",
     "tables": ["members"], "columns": ["MEMBER_ID", "RISK_SCORE", "CHRONIC_CONDITIONS"]},

    {"q": "Compare costs between HMO and PPO", "intent": "comparison",
     "sql": "SELECT PLAN_TYPE, COUNT(*) AS claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims WHERE PLAN_TYPE IN ('HMO', 'PPO') GROUP BY PLAN_TYPE",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT"]},
    {"q": "Compare denial rates between commercial and Medicare", "intent": "comparison",
     "sql": "SELECT PLAN_TYPE, ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct, COUNT(*) AS total_claims FROM claims WHERE PLAN_TYPE IN ('HMO', 'PPO', 'EPO', 'HDHP', 'Medicare Advantage', 'Medicaid') GROUP BY PLAN_TYPE ORDER BY denial_rate_pct DESC",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "CLAIM_STATUS"]},
    {"q": "Compare ER visits across regions", "intent": "comparison",
     "sql": "SELECT KP_REGION, COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY KP_REGION ORDER BY er_visits DESC",
     "tables": ["encounters"], "columns": ["KP_REGION", "VISIT_TYPE"]},
    {"q": "Inpatient vs outpatient cost comparison", "intent": "comparison",
     "sql": "SELECT e.VISIT_TYPE, COUNT(DISTINCT c.CLAIM_ID) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_paid FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE c.ENCOUNTER_ID != '' AND e.VISIT_TYPE IN ('INPATIENT', 'OUTPATIENT') GROUP BY e.VISIT_TYPE",
     "tables": ["claims", "encounters"], "columns": ["VISIT_TYPE", "PAID_AMOUNT"]},

    {"q": "Show me all denied claims over $10000", "intent": "detail",
     "sql": "SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, ICD10_DESCRIPTION, BILLED_AMOUNT, PAID_AMOUNT, DENIAL_REASON, KP_REGION FROM claims WHERE CLAIM_STATUS = 'DENIED' AND CAST(BILLED_AMOUNT AS REAL) > 10000 ORDER BY CAST(BILLED_AMOUNT AS REAL) DESC LIMIT 100",
     "tables": ["claims"], "columns": ["CLAIM_ID", "MEMBER_ID", "BILLED_AMOUNT", "CLAIM_STATUS", "DENIAL_REASON"]},
    {"q": "List high risk patients", "intent": "detail",
     "sql": "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, RISK_SCORE, CHRONIC_CONDITIONS, PLAN_TYPE, KP_REGION FROM members WHERE CAST(RISK_SCORE AS REAL) > 2.0 ORDER BY CAST(RISK_SCORE AS REAL) DESC LIMIT 50",
     "tables": ["members"], "columns": ["MEMBER_ID", "RISK_SCORE", "CHRONIC_CONDITIONS"]},
    {"q": "Show me pending referrals", "intent": "detail",
     "sql": "SELECT REFERRAL_ID, MEMBER_ID, REFERRAL_DATE, REFERRAL_REASON, URGENCY, SPECIALTY, STATUS FROM referrals WHERE STATUS = 'PENDING' ORDER BY REFERRAL_DATE DESC LIMIT 100",
     "tables": ["referrals"], "columns": ["REFERRAL_ID", "STATUS", "URGENCY"]},
    {"q": "Find members with diabetes", "intent": "detail",
     "sql": "SELECT DISTINCT d.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, d.ICD10_CODE, d.ICD10_DESCRIPTION, m.RISK_SCORE FROM diagnoses d JOIN members m ON d.MEMBER_ID = m.MEMBER_ID WHERE d.ICD10_CODE LIKE 'E11%' OR d.ICD10_DESCRIPTION LIKE '%diabetes%' ORDER BY CAST(m.RISK_SCORE AS REAL) DESC LIMIT 50",
     "tables": ["diagnoses", "members"], "columns": ["MEMBER_ID", "ICD10_CODE", "ICD10_DESCRIPTION", "RISK_SCORE"]},
    {"q": "Which claims have no encounter?", "intent": "detail",
     "sql": "SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, PAID_AMOUNT, CLAIM_STATUS FROM claims WHERE ENCOUNTER_ID IS NULL OR ENCOUNTER_ID = '' LIMIT 100",
     "tables": ["claims"], "columns": ["CLAIM_ID", "ENCOUNTER_ID"]},

    {"q": "What is our PMPM?", "intent": "rate",
     "sql": "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT MEMBER_ID || '-' || strftime('%Y-%m', SERVICE_DATE)), 0), 2) AS pmpm FROM claims WHERE SERVICE_DATE IS NOT NULL",
     "tables": ["claims"], "columns": ["PAID_AMOUNT", "MEMBER_ID", "SERVICE_DATE"]},
    {"q": "PMPM by plan type", "intent": "breakdown",
     "sql": "SELECT PLAN_TYPE, ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT MEMBER_ID || '-' || strftime('%Y-%m', SERVICE_DATE)), 0), 2) AS pmpm FROM claims WHERE SERVICE_DATE IS NOT NULL GROUP BY PLAN_TYPE ORDER BY pmpm DESC",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT", "MEMBER_ID", "SERVICE_DATE"]},
    {"q": "Average length of stay by department", "intent": "breakdown",
     "sql": "SELECT DEPARTMENT, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los, COUNT(*) AS encounters FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND CAST(LENGTH_OF_STAY AS REAL) > 0 GROUP BY DEPARTMENT ORDER BY avg_los DESC",
     "tables": ["encounters"], "columns": ["DEPARTMENT", "LENGTH_OF_STAY"]},
    {"q": "Yield rate by region", "intent": "breakdown",
     "sql": "SELECT KP_REGION, ROUND(SUM(CAST(PAID_AMOUNT AS REAL)) * 100.0 / NULLIF(SUM(CAST(BILLED_AMOUNT AS REAL)), 0), 2) AS yield_rate_pct FROM claims WHERE CAST(BILLED_AMOUNT AS REAL) > 0 GROUP BY KP_REGION ORDER BY yield_rate_pct DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "PAID_AMOUNT", "BILLED_AMOUNT"]},

    {"q": "Claims with member details", "intent": "detail",
     "sql": "SELECT c.CLAIM_ID, c.SERVICE_DATE, m.FIRST_NAME, m.LAST_NAME, m.PLAN_TYPE, c.PAID_AMOUNT, c.CLAIM_STATUS FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID ORDER BY c.SERVICE_DATE DESC LIMIT 50",
     "tables": ["claims", "members"], "columns": ["CLAIM_ID", "MEMBER_ID", "FIRST_NAME", "LAST_NAME", "PAID_AMOUNT"]},
    {"q": "Provider performance summary", "intent": "breakdown",
     "sql": "SELECT p.SPECIALTY, COUNT(DISTINCT p.NPI) AS providers, COUNT(c.CLAIM_ID) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_paid FROM providers p JOIN claims c ON p.NPI = c.RENDERING_NPI GROUP BY p.SPECIALTY ORDER BY total_paid DESC",
     "tables": ["providers", "claims"], "columns": ["SPECIALTY", "NPI", "RENDERING_NPI", "PAID_AMOUNT"]},
    {"q": "Diagnosis costs by condition", "intent": "ranking",
     "sql": "SELECT d.ICD10_DESCRIPTION, COUNT(DISTINCT d.MEMBER_ID) AS patients, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM diagnoses d JOIN claims c ON d.ENCOUNTER_ID = c.ENCOUNTER_ID WHERE d.ENCOUNTER_ID IS NOT NULL AND c.ENCOUNTER_ID != '' GROUP BY d.ICD10_DESCRIPTION ORDER BY total_cost DESC LIMIT 20",
     "tables": ["diagnoses", "claims"], "columns": ["ICD10_DESCRIPTION", "MEMBER_ID", "PAID_AMOUNT"]},

    {"q": "Claims in California", "intent": "detail",
     "sql": "SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, PAID_AMOUNT, CLAIM_STATUS, KP_REGION FROM claims WHERE KP_REGION IN ('NCAL', 'SCAL') ORDER BY SERVICE_DATE DESC LIMIT 100",
     "tables": ["claims"], "columns": ["CLAIM_ID", "KP_REGION"]},
    {"q": "Members over 65", "intent": "detail",
     "sql": "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, PLAN_TYPE, RISK_SCORE FROM members WHERE DATE_OF_BIRTH <= date('now', '-65 years') ORDER BY DATE_OF_BIRTH ASC LIMIT 100",
     "tables": ["members"], "columns": ["MEMBER_ID", "DATE_OF_BIRTH"]},
    {"q": "Claims from last year", "intent": "detail",
     "sql": "SELECT CLAIM_ID, SERVICE_DATE, PAID_AMOUNT, CLAIM_STATUS FROM claims WHERE SERVICE_DATE >= date('now', '-1 year') ORDER BY SERVICE_DATE DESC LIMIT 200",
     "tables": ["claims"], "columns": ["CLAIM_ID", "SERVICE_DATE"]},
    {"q": "High cost claims over 50000", "intent": "detail",
     "sql": "SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, ICD10_DESCRIPTION, BILLED_AMOUNT, PAID_AMOUNT, KP_REGION FROM claims WHERE CAST(BILLED_AMOUNT AS REAL) > 50000 ORDER BY CAST(BILLED_AMOUNT AS REAL) DESC LIMIT 50",
     "tables": ["claims"], "columns": ["CLAIM_ID", "BILLED_AMOUNT"]},
    {"q": "Urgent referrals", "intent": "detail",
     "sql": "SELECT REFERRAL_ID, MEMBER_ID, REFERRAL_DATE, REFERRAL_REASON, SPECIALTY, STATUS FROM referrals WHERE URGENCY = 'URGENT' ORDER BY REFERRAL_DATE DESC LIMIT 50",
     "tables": ["referrals"], "columns": ["REFERRAL_ID", "URGENCY"]},

    {"q": "Give me a summary of our claims", "intent": "executive",
     "sql": "SELECT COUNT(*) AS total_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid, ROUND(SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS denial_rate_pct, COUNT(DISTINCT MEMBER_ID) AS unique_members, COUNT(DISTINCT RENDERING_NPI) AS unique_providers FROM claims",
     "tables": ["claims"], "columns": ["CLAIM_ID", "PAID_AMOUNT", "CLAIM_STATUS", "MEMBER_ID", "RENDERING_NPI"]},
    {"q": "Executive dashboard KPIs", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Unique Members', COUNT(DISTINCT MEMBER_ID) FROM members UNION ALL SELECT 'ER Visits', COUNT(*) FROM encounters WHERE VISIT_TYPE='EMERGENCY'",
     "tables": ["claims", "members", "encounters"], "columns": ["PAID_AMOUNT", "CLAIM_STATUS", "MEMBER_ID", "VISIT_TYPE"]},
    {"q": "Overall health system performance", "intent": "executive",
     "sql": "SELECT COUNT(DISTINCT m.MEMBER_ID) AS total_members, COUNT(DISTINCT e.ENCOUNTER_ID) AS total_encounters, COUNT(DISTINCT c.CLAIM_ID) AS total_claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, AVG(CAST(m.RISK_SCORE AS REAL)) AS avg_risk FROM members m LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID",
     "tables": ["members", "encounters", "claims"], "columns": ["MEMBER_ID", "PAID_AMOUNT", "RISK_SCORE"]},

    {"q": "Why are denial rates high in the Northwest?", "intent": "root_cause",
     "sql": "SELECT DENIAL_REASON, COUNT(*) AS denial_count, ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM claims WHERE KP_REGION = 'NW' AND CLAIM_STATUS = 'DENIED'), 2) AS pct_of_denials FROM claims WHERE KP_REGION = 'NW' AND CLAIM_STATUS = 'DENIED' AND DENIAL_REASON IS NOT NULL AND DENIAL_REASON != '' GROUP BY DENIAL_REASON ORDER BY denial_count DESC LIMIT 10",
     "tables": ["claims"], "columns": ["DENIAL_REASON", "KP_REGION", "CLAIM_STATUS"]},
    {"q": "What's driving high costs?", "intent": "root_cause",
     "sql": "SELECT ICD10_DESCRIPTION, COUNT(*) AS claim_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_cost, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_cost FROM claims WHERE CAST(PAID_AMOUNT AS REAL) > 5000 GROUP BY ICD10_DESCRIPTION ORDER BY total_cost DESC LIMIT 15",
     "tables": ["claims"], "columns": ["ICD10_DESCRIPTION", "PAID_AMOUNT"]},
    {"q": "Which departments have the most ER visits?", "intent": "root_cause",
     "sql": "SELECT DEPARTMENT, COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY DEPARTMENT ORDER BY er_visits DESC LIMIT 10",
     "tables": ["encounters"], "columns": ["DEPARTMENT", "VISIT_TYPE"]},

    {"q": "Revenue by facility", "intent": "breakdown",
     "sql": "SELECT FACILITY, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_revenue, COUNT(*) AS claim_count, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_claim FROM claims WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY total_revenue DESC",
     "tables": ["claims"], "columns": ["FACILITY", "PAID_AMOUNT"]},
    {"q": "Cost by facility", "intent": "breakdown",
     "sql": "SELECT FACILITY, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_cost, COUNT(*) AS claims FROM claims WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY total_cost DESC",
     "tables": ["claims"], "columns": ["FACILITY", "PAID_AMOUNT"]},
    {"q": "Top facilities by volume", "intent": "ranking",
     "sql": "SELECT FACILITY, COUNT(*) AS encounter_count FROM encounters WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY encounter_count DESC LIMIT 15",
     "tables": ["encounters"], "columns": ["FACILITY"]},
    {"q": "Facility performance comparison", "intent": "comparison",
     "sql": "SELECT FACILITY, COUNT(*) AS encounters, COUNT(DISTINCT MEMBER_ID) AS unique_patients, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los FROM encounters WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY encounters DESC LIMIT 20",
     "tables": ["encounters"], "columns": ["FACILITY", "LENGTH_OF_STAY"]},

    {"q": "Providers with most claims", "intent": "ranking",
     "sql": "SELECT c.RENDERING_NPI, p.SPECIALTY, COUNT(*) AS claim_count, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY c.RENDERING_NPI, p.SPECIALTY ORDER BY claim_count DESC LIMIT 15",
     "tables": ["claims", "providers"], "columns": ["RENDERING_NPI", "PAID_AMOUNT"]},
    {"q": "Average days to adjudicate claims", "intent": "aggregate",
     "sql": "SELECT ROUND(AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)), 1) AS avg_days_to_adjudicate, MIN(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)) AS min_days, MAX(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)) AS max_days FROM claims WHERE ADJUDICATED_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL",
     "tables": ["claims"], "columns": ["ADJUDICATED_DATE", "SUBMITTED_DATE"]},
    {"q": "Claims processing time", "intent": "aggregate",
     "sql": "SELECT ROUND(AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)), 1) AS avg_processing_days FROM claims WHERE ADJUDICATED_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL",
     "tables": ["claims"], "columns": ["ADJUDICATED_DATE", "SUBMITTED_DATE"]},
    {"q": "Referral volume by month", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', REFERRAL_DATE) AS month, COUNT(*) AS referral_count, SUM(CASE WHEN STATUS = 'COMPLETED' THEN 1 ELSE 0 END) AS completed, SUM(CASE WHEN STATUS = 'PENDING' THEN 1 ELSE 0 END) AS pending FROM referrals GROUP BY month ORDER BY month",
     "tables": ["referrals"], "columns": ["REFERRAL_DATE", "STATUS"]},
    {"q": "Risk score distribution by plan type", "intent": "breakdown",
     "sql": "SELECT PLAN_TYPE, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, MIN(CAST(RISK_SCORE AS REAL)) AS min_risk, MAX(CAST(RISK_SCORE AS REAL)) AS max_risk, COUNT(*) AS members FROM members GROUP BY PLAN_TYPE ORDER BY avg_risk DESC",
     "tables": ["members"], "columns": ["PLAN_TYPE", "RISK_SCORE"]},
    {"q": "Members with chronic conditions list", "intent": "detail",
     "sql": "SELECT m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, m.RISK_SCORE, m.CHRONIC_CONDITIONS, m.PLAN_TYPE FROM members m WHERE m.CHRONIC_CONDITIONS IS NOT NULL AND m.CHRONIC_CONDITIONS > 0 ORDER BY m.CHRONIC_CONDITIONS DESC, CAST(m.RISK_SCORE AS REAL) DESC LIMIT 50",
     "tables": ["members"], "columns": ["MEMBER_ID", "CHRONIC_CONDITIONS"]},
    {"q": "Inpatient admissions by month", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', ADMIT_DATE) AS month, COUNT(*) AS admissions, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los FROM encounters WHERE VISIT_TYPE = 'INPATIENT' AND ADMIT_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["encounters"], "columns": ["ADMIT_DATE", "VISIT_TYPE", "LENGTH_OF_STAY"]},
    {"q": "Claim type distribution", "intent": "breakdown",
     "sql": "SELECT CLAIM_TYPE, COUNT(*) AS claim_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY CLAIM_TYPE ORDER BY claim_count DESC",
     "tables": ["claims"], "columns": ["CLAIM_TYPE", "PAID_AMOUNT"]},
    {"q": "Average cost per encounter type", "intent": "aggregate",
     "sql": "SELECT e.VISIT_TYPE, COUNT(*) AS encounters, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID GROUP BY e.VISIT_TYPE ORDER BY total_cost DESC",
     "tables": ["encounters", "claims"], "columns": ["VISIT_TYPE", "PAID_AMOUNT"]},
    {"q": "Denied claims by reason", "intent": "breakdown",
     "sql": "SELECT DENIAL_REASON, COUNT(*) AS denied_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_amount FROM claims WHERE CLAIM_STATUS = 'DENIED' AND DENIAL_REASON IS NOT NULL GROUP BY DENIAL_REASON ORDER BY denied_count DESC",
     "tables": ["claims"], "columns": ["DENIAL_REASON", "CLAIM_STATUS"]},
    {"q": "Patient demographics summary", "intent": "executive",
     "sql": "SELECT GENDER, COUNT(*) AS member_count, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic FROM members GROUP BY GENDER",
     "tables": ["members"], "columns": ["GENDER", "RISK_SCORE", "CHRONIC_CONDITIONS"]},
    {"q": "Telehealth utilization", "intent": "aggregate",
     "sql": "SELECT COUNT(*) AS telehealth_visits, COUNT(DISTINCT MEMBER_ID) AS unique_patients, ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM encounters), 2) AS pct_of_all_visits FROM encounters WHERE VISIT_TYPE = 'TELEHEALTH'",
     "tables": ["encounters"], "columns": ["VISIT_TYPE", "MEMBER_ID"]},

    {"q": "Sum of all copays", "intent": "aggregate",
     "sql": "SELECT SUM(CAST(COPAY AS REAL)) AS total_copays FROM claims",
     "tables": ["claims"], "columns": ["COPAY"]},
    {"q": "Total copay amount", "intent": "aggregate",
     "sql": "SELECT SUM(CAST(COPAY AS REAL)) AS total_copays FROM claims",
     "tables": ["claims"], "columns": ["COPAY"]},
    {"q": "How is our organization performing?", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Unique Members', COUNT(DISTINCT MEMBER_ID) FROM members UNION ALL SELECT 'ER Visits', COUNT(*) FROM encounters WHERE VISIT_TYPE='EMERGENCY'",
     "tables": ["claims", "members", "encounters"], "columns": ["PAID_AMOUNT", "CLAIM_STATUS"]},
    {"q": "Clinical quality scorecard", "intent": "executive",
     "sql": "SELECT 'Denial Rate %' AS metric, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS value FROM claims UNION ALL SELECT 'Avg LOS', ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)),1) FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND LENGTH_OF_STAY > 0 UNION ALL SELECT 'ER per 1000', ROUND(COUNT(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 END)*1000.0/(SELECT COUNT(DISTINCT MEMBER_ID) FROM members),1) FROM encounters UNION ALL SELECT 'No-Show Rate %', ROUND(SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) FROM appointments UNION ALL SELECT 'Avg Risk Score', ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members",
     "tables": ["claims", "encounters", "members", "appointments"], "columns": ["CLAIM_STATUS", "LENGTH_OF_STAY", "VISIT_TYPE"]},
    {"q": "Prescriptions by drug type", "intent": "breakdown",
     "sql": "SELECT MEDICATION_CLASS, COUNT(*) AS rx_count, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY rx_count DESC",
     "tables": ["prescriptions"], "columns": ["MEDICATION_CLASS", "COST"]},
    {"q": "Break down prescriptions by drug type", "intent": "breakdown",
     "sql": "SELECT MEDICATION_CLASS, COUNT(*) AS rx_count, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY rx_count DESC",
     "tables": ["prescriptions"], "columns": ["MEDICATION_CLASS", "COST"]},

    {"q": "Show me the numbers", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Members', COUNT(*) FROM members UNION ALL SELECT 'Encounters', COUNT(*) FROM encounters",
     "tables": ["claims", "members", "encounters"], "columns": ["PAID_AMOUNT"]},
    {"q": "How are we doing?", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Unique Members', COUNT(DISTINCT MEMBER_ID) FROM members UNION ALL SELECT 'ER Visits', COUNT(*) FROM encounters WHERE VISIT_TYPE='EMERGENCY'",
     "tables": ["claims", "members", "encounters"], "columns": ["PAID_AMOUNT"]},
    {"q": "Give me some data", "intent": "executive",
     "sql": "SELECT COUNT(*) AS total_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_pct, COUNT(DISTINCT MEMBER_ID) AS unique_members FROM claims",
     "tables": ["claims"], "columns": ["PAID_AMOUNT", "CLAIM_STATUS"]},
    {"q": "What's the situation?", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Unique Members', COUNT(DISTINCT MEMBER_ID) FROM members",
     "tables": ["claims", "members"], "columns": ["PAID_AMOUNT"]},
    {"q": "Show me trends", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, COUNT(*) AS claim_count, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_rate FROM claims WHERE SERVICE_DATE >= date('now', '-12 months') GROUP BY month ORDER BY month",
     "tables": ["claims"], "columns": ["SERVICE_DATE", "PAID_AMOUNT"]},
    {"q": "What should I know?", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'ER Visits', COUNT(*) FROM encounters WHERE VISIT_TYPE='EMERGENCY' UNION ALL SELECT 'Avg Risk Score', ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members",
     "tables": ["claims", "encounters", "members"], "columns": ["PAID_AMOUNT"]},
    {"q": "Any issues?", "intent": "executive",
     "sql": "SELECT 'Denial Rate %' AS metric, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS value FROM claims UNION ALL SELECT 'ER per 1000', ROUND(COUNT(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 END)*1000.0/(SELECT COUNT(DISTINCT MEMBER_ID) FROM members),1) FROM encounters UNION ALL SELECT 'No-Show Rate %', ROUND(SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) FROM appointments UNION ALL SELECT 'Pending Referrals', COUNT(*) FROM referrals WHERE STATUS='PENDING'",
     "tables": ["claims", "encounters", "appointments", "referrals"], "columns": ["CLAIM_STATUS"]},
    {"q": "The big picture", "intent": "executive",
     "sql": "SELECT 'Total Claims' AS metric, COUNT(*) AS value FROM claims UNION ALL SELECT 'Total Paid', SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims UNION ALL SELECT 'Denial Rate %', ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) FROM claims UNION ALL SELECT 'Members', COUNT(*) FROM members UNION ALL SELECT 'Providers', COUNT(*) FROM providers",
     "tables": ["claims", "members", "providers"], "columns": ["PAID_AMOUNT"]},


    {"q": "Monthly appointment volume", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', APPOINTMENT_DATE) AS month, COUNT(*) AS appointment_count FROM appointments WHERE APPOINTMENT_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["appointments"], "columns": ["APPOINTMENT_DATE"]},
    {"q": "Appointment trends by month", "intent": "trend",
     "sql": "SELECT strftime('%Y-%m', APPOINTMENT_DATE) AS month, COUNT(*) AS appointment_count FROM appointments WHERE APPOINTMENT_DATE IS NOT NULL GROUP BY month ORDER BY month",
     "tables": ["appointments"], "columns": ["APPOINTMENT_DATE"]},

    {"q": "Commercial vs Medicare costs", "intent": "comparison",
     "sql": "SELECT CASE WHEN PLAN_TYPE IN ('Medicare Advantage','Medicaid') THEN 'Government' ELSE 'Commercial' END AS plan_category, COUNT(*) AS claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid, ROUND(SUM(CAST(PAID_AMOUNT AS REAL))/NULLIF(COUNT(DISTINCT MEMBER_ID),0),2) AS cost_per_member FROM claims GROUP BY plan_category ORDER BY total_paid DESC",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT"]},
    {"q": "Compare commercial and Medicare plan costs", "intent": "comparison",
     "sql": "SELECT CASE WHEN PLAN_TYPE IN ('Medicare Advantage','Medicaid') THEN 'Government' ELSE 'Commercial' END AS plan_category, COUNT(*) AS claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY plan_category",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT"]},

    {"q": "HMO vs Medicare Advantage PMPM", "intent": "comparison",
     "sql": "SELECT c.PLAN_TYPE, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT c.MEMBER_ID),0), 2) AS pmpm, COUNT(*) AS claims, COUNT(DISTINCT c.MEMBER_ID) AS members FROM claims c WHERE c.PLAN_TYPE IN ('HMO','Medicare Advantage') GROUP BY c.PLAN_TYPE ORDER BY pmpm DESC",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT", "MEMBER_ID"]},
    {"q": "PMPM comparison HMO vs Medicare", "intent": "comparison",
     "sql": "SELECT c.PLAN_TYPE, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT c.MEMBER_ID),0), 2) AS pmpm FROM claims c WHERE c.PLAN_TYPE IN ('HMO','Medicare Advantage') GROUP BY c.PLAN_TYPE",
     "tables": ["claims"], "columns": ["PLAN_TYPE", "PAID_AMOUNT"]},

    {"q": "How many patients expired?", "intent": "count",
     "sql": "SELECT COUNT(*) AS expired_patients FROM encounters WHERE DISPOSITION = 'Expired'",
     "tables": ["encounters"], "columns": ["DISPOSITION"]},
    {"q": "Patient expiration count", "intent": "count",
     "sql": "SELECT COUNT(*) AS expired FROM encounters WHERE DISPOSITION = 'Expired'",
     "tables": ["encounters"], "columns": ["DISPOSITION"]},

    {"q": "Chronic conditions by race", "intent": "breakdown",
     "sql": "SELECT m.RACE, AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, SUM(CASE WHEN m.CHRONIC_CONDITIONS > 0 THEN 1 ELSE 0 END) AS with_chronic, COUNT(*) AS members FROM members m GROUP BY m.RACE ORDER BY avg_chronic DESC",
     "tables": ["members"], "columns": ["RACE", "CHRONIC_CONDITIONS"]},
    {"q": "Chronic condition count by race", "intent": "breakdown",
     "sql": "SELECT RACE, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, COUNT(*) AS members FROM members GROUP BY RACE ORDER BY avg_chronic DESC",
     "tables": ["members"], "columns": ["RACE", "CHRONIC_CONDITIONS"]},

    {"q": "Readmission rate", "intent": "rate",
     "sql": "SELECT ROUND(COUNT(CASE WHEN e2.MEMBER_ID IS NOT NULL THEN 1 END)*100.0/NULLIF(COUNT(*),0),2) AS readmission_rate_pct FROM encounters e1 LEFT JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID AND e2.ENCOUNTER_ID != e1.ENCOUNTER_ID AND e2.ADMIT_DATE BETWEEN e1.DISCHARGE_DATE AND date(e1.DISCHARGE_DATE, '+30 days') WHERE e1.VISIT_TYPE = 'INPATIENT' AND e1.DISCHARGE_DATE IS NOT NULL",
     "tables": ["encounters"], "columns": ["MEMBER_ID", "ADMISSION_DATE", "DISCHARGE_DATE"]},
    {"q": "What is the readmission rate?", "intent": "rate",
     "sql": "SELECT ROUND(COUNT(CASE WHEN e2.MEMBER_ID IS NOT NULL THEN 1 END)*100.0/NULLIF(COUNT(*),0),2) AS readmission_rate_pct FROM encounters e1 LEFT JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID AND e2.ENCOUNTER_ID != e1.ENCOUNTER_ID AND e2.ADMIT_DATE BETWEEN e1.DISCHARGE_DATE AND date(e1.DISCHARGE_DATE, '+30 days') WHERE e1.VISIT_TYPE = 'INPATIENT' AND e1.DISCHARGE_DATE IS NOT NULL",
     "tables": ["encounters"], "columns": ["MEMBER_ID"]},
    {"q": "Readmission rate by department", "intent": "breakdown",
     "sql": "SELECT e1.DEPARTMENT, COUNT(*) AS discharges, SUM(CASE WHEN e2.MEMBER_ID IS NOT NULL THEN 1 ELSE 0 END) AS readmissions, ROUND(SUM(CASE WHEN e2.MEMBER_ID IS NOT NULL THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0),2) AS readmit_pct FROM encounters e1 LEFT JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID AND e2.ENCOUNTER_ID != e1.ENCOUNTER_ID AND e2.ADMIT_DATE BETWEEN e1.DISCHARGE_DATE AND date(e1.DISCHARGE_DATE, '+30 days') WHERE e1.VISIT_TYPE = 'INPATIENT' AND e1.DISCHARGE_DATE IS NOT NULL GROUP BY e1.DEPARTMENT ORDER BY readmit_pct DESC",
     "tables": ["encounters"], "columns": ["DEPARTMENT", "MEMBER_ID"]},
    {"q": "Why is the readmission rate increasing?", "intent": "root_cause",
     "sql": "SELECT e1.DEPARTMENT, e1.DIAGNOSIS_DESCRIPTION, COUNT(*) AS readmissions FROM encounters e1 INNER JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID AND e1.ENCOUNTER_ID != e2.ENCOUNTER_ID AND e1.ADMIT_DATE BETWEEN e2.DISCHARGE_DATE AND date(e2.DISCHARGE_DATE, '+30 days') WHERE e1.VISIT_TYPE = 'INPATIENT' GROUP BY e1.DEPARTMENT, e1.DIAGNOSIS_DESCRIPTION ORDER BY readmissions DESC LIMIT 20",
     "tables": ["encounters"], "columns": ["DEPARTMENT", "DIAGNOSIS_DESCRIPTION"]},

    {"q": "What drives high prescription costs?", "intent": "root_cause",
     "sql": "SELECT MEDICATION_CLASS, MEDICATION_NAME, COUNT(*) AS rx_count, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY MEDICATION_CLASS, MEDICATION_NAME ORDER BY total_cost DESC LIMIT 20",
     "tables": ["prescriptions"], "columns": ["MEDICATION_CLASS", "MEDICATION_NAME", "COST"]},
    {"q": "Prescription cost drivers", "intent": "root_cause",
     "sql": "SELECT MEDICATION_CLASS, COUNT(*) AS rx_count, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY total_cost DESC",
     "tables": ["prescriptions"], "columns": ["MEDICATION_CLASS", "COST"]},

    {"q": "What does our member population look like?", "intent": "breakdown",
     "sql": "SELECT GENDER, COUNT(*) AS members, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, COUNT(DISTINCT PLAN_TYPE) AS plan_types FROM members GROUP BY GENDER",
     "tables": ["members"], "columns": ["GENDER", "RISK_SCORE", "CHRONIC_CONDITIONS"]},
    {"q": "Member demographics summary", "intent": "breakdown",
     "sql": "SELECT GENDER, RACE, COUNT(*) AS cnt, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk FROM members GROUP BY GENDER, RACE ORDER BY cnt DESC",
     "tables": ["members"], "columns": ["GENDER", "RACE", "RISK_SCORE"]},
    {"q": "Member population breakdown", "intent": "breakdown",
     "sql": "SELECT PLAN_TYPE, GENDER, COUNT(*) AS members, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk FROM members GROUP BY PLAN_TYPE, GENDER ORDER BY members DESC",
     "tables": ["members"], "columns": ["PLAN_TYPE", "GENDER", "RISK_SCORE"]},

    {"q": "Cost per member by region", "intent": "breakdown",
     "sql": "SELECT c.KP_REGION, COUNT(DISTINCT c.MEMBER_ID) AS members, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL))/NULLIF(COUNT(DISTINCT c.MEMBER_ID),0),2) AS cost_per_member FROM claims c GROUP BY c.KP_REGION ORDER BY cost_per_member DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "MEMBER_ID", "PAID_AMOUNT"]},
    {"q": "Per member cost by region", "intent": "breakdown",
     "sql": "SELECT KP_REGION, ROUND(SUM(CAST(PAID_AMOUNT AS REAL))/NULLIF(COUNT(DISTINCT MEMBER_ID),0),2) AS cost_per_member, COUNT(DISTINCT MEMBER_ID) AS members FROM claims GROUP BY KP_REGION ORDER BY cost_per_member DESC",
     "tables": ["claims"], "columns": ["KP_REGION", "PAID_AMOUNT"]},
]


def pretrain(db_path: str):

    data_dir = os.path.dirname(db_path)

    logger.info("Step 1: Learning database schema...")
    t0 = time.time()

    from semantic_layer import SchemaLearner
    learner = SchemaLearner(db_path)
    learner.learn(sample_size=1000)

    logger.info("Schema learned in %.1fs: %d tables, %d columns, %d join paths",
                time.time() - t0,
                len(learner.tables),
                sum(len(v) for v in learner.tables.values()),
                sum(len(v) for v in learner.join_graph.values()))

    logger.info("\nStep 2: Training Deep Understanding Engine on real schema...")
    t0 = time.time()

    from deep_understanding import DeepUnderstandingEngine
    deep_engine = DeepUnderstandingEngine(
        schema_learner=learner,
        db_path=db_path,
        data_dir=data_dir,
    )
    logger.info("Deep engine ready in %.1fs: vocab=%d, dim=%d, concepts=%d",
                time.time() - t0, deep_engine.vocab_size,
                deep_engine.embedding_dim, deep_engine.concept_count)

    logger.info("\nStep 3: Initializing SQL Reasoning Engine...")
    t0 = time.time()

    from sql_reasoning import SQLReasoningEngine
    shared_embedder = deep_engine._embedder if hasattr(deep_engine, '_embedder') else None
    sql_engine = SQLReasoningEngine(schema_learner=learner, db_path=db_path, embedder=shared_embedder)
    logger.info("SQL engine ready in %.1fs: %d patterns, %d templates",
                time.time() - t0, sql_engine.pattern_count, sql_engine.template_count)

    logger.info("\nStep 4: Training on %d query examples...", len(TRAINING_QUERIES))
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    success_count = 0
    fail_count = 0

    for i, tq in enumerate(TRAINING_QUERIES):
        question = tq['q']
        expected_sql = tq['sql']
        intent = tq['intent']
        tables = tq['tables']
        columns = tq['columns']

        try:
            cursor = conn.execute(expected_sql)
            rows = cursor.fetchall()
            result_cols = [d[0] for d in cursor.description] if cursor.description else []
            row_count = len(rows)
            sql_works = True
        except Exception as e:
            logger.warning("  SKIP %d: SQL error for '%s': %s", i, question[:40], e)
            sql_works = False
            fail_count += 1
            continue

        deep_engine.learn(
            question=question,
            intent=intent,
            tables=tables,
            columns=columns,
            sql=expected_sql,
            success=True,
        )

        sql_engine.learn(
            question=question,
            sql=expected_sql,
            success=True,
        )

        success_count += 1
        if (i + 1) % 25 == 0:
            logger.info("  Trained %d/%d queries (%.0f%% success)",
                       i + 1, len(TRAINING_QUERIES),
                       success_count / (i + 1) * 100)

    conn.close()

    train_time = time.time() - t0
    logger.info("Training complete in %.1fs: %d/%d successful (%d failed)",
                train_time, success_count, len(TRAINING_QUERIES), fail_count)

    logger.info("\n" + "=" * 60)
    logger.info("Step 5: DRY RUN — Testing understanding on 30 questions")
    logger.info("=" * 60)

    test_questions = [
        "How many denied claims do we have?",
        "What is the total cost of all claims?",
        "Show me the denial rate by region",
        "Top 10 highest cost members",
        "Monthly cost trend",
        "Compare HMO vs PPO costs",
        "What is our PMPM?",
        "ER utilization rate",
        "Average length of stay",
        "Yield rate by plan type",
        "Which providers have the most claims?",
        "List high risk patients",
        "Claims breakdown by status",
        "Prescriptions by medication class",
        "How many members are enrolled?",
        "Diagnoses by severity",
        "What percentage of claims are denied?",
        "Cost per member by region",
        "Most expensive procedures",
        "Referral trends by month",
        "Members with chronic conditions",
        "Provider specialty distribution",
        "Appointment no-show rate",
        "Claims from last year",
        "Average risk score by plan type",
        "Who are our sickest patients?",
        "What's driving high costs?",
        "Revenue by facility",
        "Inpatient admission trends",
        "Executive summary of operations",
        "How many ER visits last month?",
        "Denial reasons breakdown",
        "Telehealth visit volume",
        "Average cost per visit type",
        "Which facilities have the most encounters?",
        "Risk score distribution",
        "Claim type breakdown",
        "Members by plan type",
        "Readmission rates",
        "Prescription costs by drug class",
        "Provider performance by specialty",
        "Patient demographics",
        "Top denial reasons",
        "Monthly enrollment changes",
        "Cost comparison by region",
        "High risk members over 65",
        "Pending referral count",
        "Average days to adjudicate claims",
        "Department visit volume",
        "Chronic condition prevalence",
    ]

    conn = sqlite3.connect(db_path)
    passed = 0
    total = len(test_questions)

    for q in test_questions:
        t1 = time.time()

        understanding = deep_engine.understand(q)
        u_dict = understanding.to_dict()

        sql_result = sql_engine.generate(understanding=u_dict, question=q)
        sql_text = sql_result.sql if hasattr(sql_result, 'sql') else 'SELECT 1'
        confidence = sql_result.confidence if hasattr(sql_result, 'confidence') else 0

        try:
            cursor = conn.execute(sql_text)
            rows = cursor.fetchall()
            row_count = len(rows)
            error = None
        except Exception as e:
            rows = []
            row_count = 0
            error = str(e)

        ms = (time.time() - t1) * 1000

        is_fallback = confidence < 0.20 or 'appointments.*' in sql_text
        ok = row_count > 0 and error is None and not is_fallback
        if ok:
            passed += 1

        status = "PASS" if ok else ("FALLBACK" if is_fallback else "FAIL")
        logger.info("  [%s] %s", status, q)
        logger.info("        Intent: %s (%.2f) | Tables: %s | SQL conf: %.2f | Rows: %d | %dms",
                    u_dict['intent'], u_dict['intent_confidence'],
                    u_dict['target_tables'][:3], confidence, row_count, ms)
        if error:
            logger.info("        Error: %s", error[:80])
        if sql_text and sql_text != 'SELECT 1':
            logger.info("        SQL: %s", sql_text[:150])

    conn.close()

    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN RESULTS: %d/%d passed (%.0f%%)", passed, total, passed / total * 100)
    logger.info("Deep engine: vocab=%d, memories=%d",
                deep_engine.vocab_size, len(deep_engine._memory._memories))
    logger.info("SQL engine: patterns=%d, templates=%d",
                sql_engine.pattern_count, sql_engine.template_count)
    logger.info("=" * 60)

    return passed, total


if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_production.db'

    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    passed, total = pretrain(db_path)
    sys.exit(0 if passed >= total * 0.5 else 1)
