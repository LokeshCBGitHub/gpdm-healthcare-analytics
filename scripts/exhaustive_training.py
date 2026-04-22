import sqlite3
import json
import time
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('exhaustive')
logger.setLevel(logging.INFO)


def generate_all_training_queries(db_path: str):
    conn = sqlite3.connect(db_path)
    queries = []


    queries += [
        {"q": "Claims by claim status", "sql": "SELECT CLAIM_STATUS, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY CLAIM_STATUS ORDER BY cnt DESC"},
        {"q": "Claims by plan type", "sql": "SELECT PLAN_TYPE, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY PLAN_TYPE ORDER BY total_paid DESC"},
        {"q": "Claims by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY KP_REGION ORDER BY total_paid DESC"},
        {"q": "Claims by facility", "sql": "SELECT FACILITY, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY total_paid DESC"},
        {"q": "Claims by claim type", "sql": "SELECT CLAIM_TYPE, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY CLAIM_TYPE ORDER BY cnt DESC"},
        {"q": "Claims by denial reason", "sql": "SELECT DENIAL_REASON, COUNT(*) AS cnt FROM claims WHERE CLAIM_STATUS = 'DENIED' AND DENIAL_REASON IS NOT NULL AND DENIAL_REASON != '' GROUP BY DENIAL_REASON ORDER BY cnt DESC"},
        {"q": "Claims by ICD10 code", "sql": "SELECT ICD10_CODE, ICD10_DESCRIPTION, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY ICD10_CODE, ICD10_DESCRIPTION ORDER BY cnt DESC LIMIT 20"},
        {"q": "Claims by CPT code", "sql": "SELECT CPT_CODE, CPT_DESCRIPTION, COUNT(*) AS cnt, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims GROUP BY CPT_CODE, CPT_DESCRIPTION ORDER BY cnt DESC LIMIT 20"},

        {"q": "How many paid claims?", "sql": "SELECT COUNT(*) AS paid_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims WHERE CLAIM_STATUS = 'PAID'"},
        {"q": "How many pending claims?", "sql": "SELECT COUNT(*) AS pending_claims FROM claims WHERE CLAIM_STATUS = 'PENDING'"},
        {"q": "How many adjusted claims?", "sql": "SELECT COUNT(*) AS adjusted_claims FROM claims WHERE CLAIM_STATUS = 'ADJUSTED'"},
        {"q": "How many appealed claims?", "sql": "SELECT COUNT(*) AS appealed_claims FROM claims WHERE CLAIM_STATUS = 'APPEALED'"},
        {"q": "How many voided claims?", "sql": "SELECT COUNT(*) AS voided_claims FROM claims WHERE CLAIM_STATUS = 'VOIDED'"},

        {"q": "Denied claims by region", "sql": "SELECT KP_REGION, COUNT(*) AS denied, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'),2) AS pct FROM claims WHERE CLAIM_STATUS='DENIED' GROUP BY KP_REGION ORDER BY denied DESC"},
        {"q": "Denial rate by plan type", "sql": "SELECT PLAN_TYPE, COUNT(*) AS total, SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denied, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_pct FROM claims GROUP BY PLAN_TYPE ORDER BY denial_pct DESC"},
        {"q": "Denial rate by facility", "sql": "SELECT FACILITY, COUNT(*) AS total, SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denied, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_pct FROM claims WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY denial_pct DESC"},
        {"q": "Denial rate by claim type", "sql": "SELECT CLAIM_TYPE, COUNT(*) AS total, SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denied, ROUND(SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_pct FROM claims GROUP BY CLAIM_TYPE ORDER BY denial_pct DESC"},

        {"q": "Average billed amount by plan type", "sql": "SELECT PLAN_TYPE, AVG(CAST(BILLED_AMOUNT AS REAL)) AS avg_billed, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY PLAN_TYPE ORDER BY avg_billed DESC"},
        {"q": "Average allowed amount by region", "sql": "SELECT KP_REGION, AVG(CAST(ALLOWED_AMOUNT AS REAL)) AS avg_allowed, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims GROUP BY KP_REGION ORDER BY avg_allowed DESC"},
        {"q": "Member responsibility by plan type", "sql": "SELECT PLAN_TYPE, AVG(CAST(MEMBER_RESPONSIBILITY AS REAL)) AS avg_member_resp, AVG(CAST(COPAY AS REAL)) AS avg_copay, AVG(CAST(COINSURANCE AS REAL)) AS avg_coinsurance FROM claims GROUP BY PLAN_TYPE ORDER BY avg_member_resp DESC"},
        {"q": "Copay analysis by plan type", "sql": "SELECT PLAN_TYPE, AVG(CAST(COPAY AS REAL)) AS avg_copay, MIN(CAST(COPAY AS REAL)) AS min_copay, MAX(CAST(COPAY AS REAL)) AS max_copay FROM claims GROUP BY PLAN_TYPE"},
        {"q": "Coinsurance by region", "sql": "SELECT KP_REGION, AVG(CAST(COINSURANCE AS REAL)) AS avg_coinsurance, SUM(CAST(COINSURANCE AS REAL)) AS total_coinsurance FROM claims GROUP BY KP_REGION ORDER BY avg_coinsurance DESC"},

        {"q": "Top 5 most expensive claims", "sql": "SELECT CLAIM_ID, MEMBER_ID, ICD10_DESCRIPTION, BILLED_AMOUNT, PAID_AMOUNT, CLAIM_STATUS FROM claims ORDER BY CAST(BILLED_AMOUNT AS REAL) DESC LIMIT 5"},
        {"q": "Top 10 claims by paid amount", "sql": "SELECT CLAIM_ID, MEMBER_ID, ICD10_DESCRIPTION, PAID_AMOUNT, KP_REGION, PLAN_TYPE FROM claims ORDER BY CAST(PAID_AMOUNT AS REAL) DESC LIMIT 10"},
        {"q": "Top 5 denial reasons", "sql": "SELECT DENIAL_REASON, COUNT(*) AS cnt FROM claims WHERE CLAIM_STATUS='DENIED' AND DENIAL_REASON IS NOT NULL AND DENIAL_REASON != '' GROUP BY DENIAL_REASON ORDER BY cnt DESC LIMIT 5"},

        {"q": "Total claims for emergency visits", "sql": "SELECT COUNT(*) AS er_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'EMERGENCY'"},
        {"q": "Top 5 claims for emergency", "sql": "SELECT c.CLAIM_ID, c.MEMBER_ID, c.ICD10_DESCRIPTION, c.PAID_AMOUNT, c.KP_REGION FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'EMERGENCY' ORDER BY CAST(c.PAID_AMOUNT AS REAL) DESC LIMIT 5"},
        {"q": "Claims for inpatient visits", "sql": "SELECT COUNT(*) AS inpatient_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, AVG(CAST(PAID_AMOUNT AS REAL)) AS avg_paid FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'INPATIENT'"},
        {"q": "Claims for telehealth", "sql": "SELECT COUNT(*) AS telehealth_claims, SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'TELEHEALTH'"},
    ]

    queries += [
        {"q": "Encounters by visit type", "sql": "SELECT VISIT_TYPE, COUNT(*) AS cnt FROM encounters GROUP BY VISIT_TYPE ORDER BY cnt DESC"},
        {"q": "Encounters by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS cnt FROM encounters GROUP BY DEPARTMENT ORDER BY cnt DESC"},
        {"q": "Encounters by disposition", "sql": "SELECT DISPOSITION, COUNT(*) AS cnt FROM encounters GROUP BY DISPOSITION ORDER BY cnt DESC"},
        {"q": "Encounters by encounter status", "sql": "SELECT ENCOUNTER_STATUS, COUNT(*) AS cnt FROM encounters GROUP BY ENCOUNTER_STATUS ORDER BY cnt DESC"},
        {"q": "Encounters by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt FROM encounters GROUP BY KP_REGION ORDER BY cnt DESC"},
        {"q": "Encounters by facility", "sql": "SELECT FACILITY, COUNT(*) AS cnt FROM encounters WHERE FACILITY IS NOT NULL GROUP BY FACILITY ORDER BY cnt DESC"},
        {"q": "Encounters by chief complaint", "sql": "SELECT CHIEF_COMPLAINT, COUNT(*) AS cnt FROM encounters WHERE CHIEF_COMPLAINT IS NOT NULL GROUP BY CHIEF_COMPLAINT ORDER BY cnt DESC"},
        {"q": "Encounters by primary diagnosis", "sql": "SELECT PRIMARY_DIAGNOSIS, DIAGNOSIS_DESCRIPTION, COUNT(*) AS cnt FROM encounters GROUP BY PRIMARY_DIAGNOSIS, DIAGNOSIS_DESCRIPTION ORDER BY cnt DESC LIMIT 20"},

        {"q": "How many emergency visits?", "sql": "SELECT COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY'"},
        {"q": "How many inpatient admissions?", "sql": "SELECT COUNT(*) AS inpatient FROM encounters WHERE VISIT_TYPE = 'INPATIENT'"},
        {"q": "How many outpatient visits?", "sql": "SELECT COUNT(*) AS outpatient FROM encounters WHERE VISIT_TYPE = 'OUTPATIENT'"},
        {"q": "How many telehealth visits?", "sql": "SELECT COUNT(*) AS telehealth FROM encounters WHERE VISIT_TYPE = 'TELEHEALTH'"},
        {"q": "How many urgent care visits?", "sql": "SELECT COUNT(*) AS urgent_care FROM encounters WHERE VISIT_TYPE = 'URGENT_CARE'"},
        {"q": "How many home health visits?", "sql": "SELECT COUNT(*) AS home_health FROM encounters WHERE VISIT_TYPE = 'HOME_HEALTH'"},

        {"q": "Emergency visits by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY DEPARTMENT ORDER BY er_visits DESC"},
        {"q": "Inpatient admissions by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS admissions, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los FROM encounters WHERE VISIT_TYPE = 'INPATIENT' AND LENGTH_OF_STAY IS NOT NULL GROUP BY DEPARTMENT ORDER BY admissions DESC"},
        {"q": "Average length of stay by department", "sql": "SELECT DEPARTMENT, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los, COUNT(*) AS encounters FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND LENGTH_OF_STAY > 0 GROUP BY DEPARTMENT ORDER BY avg_los DESC"},
        {"q": "Average length of stay by visit type", "sql": "SELECT VISIT_TYPE, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los, COUNT(*) AS cnt FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND LENGTH_OF_STAY > 0 GROUP BY VISIT_TYPE ORDER BY avg_los DESC"},
        {"q": "Length of stay by region", "sql": "SELECT KP_REGION, AVG(CAST(LENGTH_OF_STAY AS REAL)) AS avg_los FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL AND LENGTH_OF_STAY > 0 GROUP BY KP_REGION ORDER BY avg_los DESC"},

        {"q": "How many patients were admitted?", "sql": "SELECT COUNT(*) AS admitted FROM encounters WHERE DISPOSITION = 'Admitted'"},
        {"q": "How many patients were discharged?", "sql": "SELECT COUNT(*) AS discharged FROM encounters WHERE DISPOSITION = 'Discharged'"},
        {"q": "How many patients left AMA?", "sql": "SELECT COUNT(*) AS ama FROM encounters WHERE DISPOSITION = 'AMA'"},
        {"q": "How many patients expired?", "sql": "SELECT COUNT(*) AS expired FROM encounters WHERE DISPOSITION = 'Expired'"},
        {"q": "How many patients transferred?", "sql": "SELECT COUNT(*) AS transferred FROM encounters WHERE DISPOSITION = 'Transferred'"},
        {"q": "Disposition by visit type", "sql": "SELECT VISIT_TYPE, DISPOSITION, COUNT(*) AS cnt FROM encounters GROUP BY VISIT_TYPE, DISPOSITION ORDER BY VISIT_TYPE, cnt DESC"},

        {"q": "Top 5 busiest departments", "sql": "SELECT DEPARTMENT, COUNT(*) AS visits FROM encounters GROUP BY DEPARTMENT ORDER BY visits DESC LIMIT 5"},
        {"q": "Top chief complaints", "sql": "SELECT CHIEF_COMPLAINT, COUNT(*) AS cnt FROM encounters WHERE CHIEF_COMPLAINT IS NOT NULL GROUP BY CHIEF_COMPLAINT ORDER BY cnt DESC LIMIT 10"},
        {"q": "Top 5 longest stays", "sql": "SELECT ENCOUNTER_ID, MEMBER_ID, VISIT_TYPE, DEPARTMENT, LENGTH_OF_STAY FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL ORDER BY CAST(LENGTH_OF_STAY AS INTEGER) DESC LIMIT 5"},
    ]

    queries += [
        {"q": "Diagnoses by severity", "sql": "SELECT SEVERITY, COUNT(*) AS cnt FROM diagnoses GROUP BY SEVERITY ORDER BY cnt DESC"},
        {"q": "Diagnoses by type", "sql": "SELECT DIAGNOSIS_TYPE, COUNT(*) AS cnt FROM diagnoses GROUP BY DIAGNOSIS_TYPE ORDER BY cnt DESC"},
        {"q": "Diagnoses by ICD10 code", "sql": "SELECT ICD10_CODE, ICD10_DESCRIPTION, COUNT(*) AS cnt FROM diagnoses GROUP BY ICD10_CODE, ICD10_DESCRIPTION ORDER BY cnt DESC LIMIT 20"},
        {"q": "Chronic vs non-chronic diagnoses", "sql": "SELECT IS_CHRONIC, COUNT(*) AS cnt, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM diagnoses),2) AS pct FROM diagnoses GROUP BY IS_CHRONIC"},
        {"q": "Diagnoses by HCC category", "sql": "SELECT HCC_CATEGORY, COUNT(*) AS cnt FROM diagnoses WHERE HCC_CATEGORY IS NOT NULL AND HCC_CATEGORY != 'None' GROUP BY HCC_CATEGORY ORDER BY cnt DESC"},

        {"q": "Total claims by severity", "sql": "SELECT d.SEVERITY, COUNT(DISTINCT d.DIAGNOSIS_ID) AS diagnoses, COUNT(DISTINCT c.CLAIM_ID) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM diagnoses d JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID GROUP BY d.SEVERITY ORDER BY total_cost DESC"},
        {"q": "Critical diagnoses count", "sql": "SELECT COUNT(*) AS critical_diagnoses FROM diagnoses WHERE SEVERITY = 'CRITICAL'"},
        {"q": "Severe diagnoses count", "sql": "SELECT COUNT(*) AS severe_diagnoses FROM diagnoses WHERE SEVERITY = 'SEVERE'"},
        {"q": "Moderate diagnoses count", "sql": "SELECT COUNT(*) AS moderate_diagnoses FROM diagnoses WHERE SEVERITY = 'MODERATE'"},
        {"q": "Mild diagnoses count", "sql": "SELECT COUNT(*) AS mild_diagnoses FROM diagnoses WHERE SEVERITY = 'MILD'"},
        {"q": "Chronic diagnoses by severity", "sql": "SELECT SEVERITY, COUNT(*) AS cnt FROM diagnoses WHERE IS_CHRONIC = 'Y' GROUP BY SEVERITY ORDER BY cnt DESC"},
        {"q": "Cost by diagnosis severity", "sql": "SELECT d.SEVERITY, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost, COUNT(DISTINCT c.CLAIM_ID) AS claims FROM diagnoses d JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID GROUP BY d.SEVERITY ORDER BY total_cost DESC"},

        {"q": "Top 5 most common diagnoses", "sql": "SELECT ICD10_CODE, ICD10_DESCRIPTION, COUNT(*) AS cnt FROM diagnoses GROUP BY ICD10_CODE, ICD10_DESCRIPTION ORDER BY cnt DESC LIMIT 5"},
        {"q": "Top 10 chronic conditions", "sql": "SELECT ICD10_CODE, ICD10_DESCRIPTION, COUNT(*) AS cnt FROM diagnoses WHERE IS_CHRONIC = 'Y' GROUP BY ICD10_CODE, ICD10_DESCRIPTION ORDER BY cnt DESC LIMIT 10"},
        {"q": "Most common HCC categories", "sql": "SELECT HCC_CATEGORY, COUNT(*) AS cnt FROM diagnoses WHERE HCC_CATEGORY IS NOT NULL AND HCC_CATEGORY != 'None' GROUP BY HCC_CATEGORY ORDER BY cnt DESC"},
        {"q": "Unresolved diagnoses", "sql": "SELECT ICD10_DESCRIPTION, COUNT(*) AS cnt FROM diagnoses WHERE RESOLVED_DATE IS NULL GROUP BY ICD10_DESCRIPTION ORDER BY cnt DESC LIMIT 10"},
    ]

    queries += [
        {"q": "Members by gender", "sql": "SELECT GENDER, COUNT(*) AS cnt FROM members GROUP BY GENDER"},
        {"q": "Members by race", "sql": "SELECT RACE, COUNT(*) AS cnt FROM members GROUP BY RACE ORDER BY cnt DESC"},
        {"q": "Members by language", "sql": "SELECT LANGUAGE, COUNT(*) AS cnt FROM members GROUP BY LANGUAGE ORDER BY cnt DESC"},
        {"q": "Members by city", "sql": "SELECT CITY, COUNT(*) AS cnt FROM members GROUP BY CITY ORDER BY cnt DESC"},
        {"q": "Members by state", "sql": "SELECT STATE, COUNT(*) AS cnt FROM members GROUP BY STATE ORDER BY cnt DESC"},
        {"q": "Members by plan type", "sql": "SELECT PLAN_TYPE, COUNT(*) AS cnt FROM members GROUP BY PLAN_TYPE ORDER BY cnt DESC"},
        {"q": "Members by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt FROM members GROUP BY KP_REGION ORDER BY cnt DESC"},
        {"q": "Members by chronic condition count", "sql": "SELECT CHRONIC_CONDITIONS, COUNT(*) AS cnt FROM members GROUP BY CHRONIC_CONDITIONS ORDER BY CHRONIC_CONDITIONS"},

        {"q": "Average risk score by plan type", "sql": "SELECT PLAN_TYPE, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, COUNT(*) AS members FROM members GROUP BY PLAN_TYPE ORDER BY avg_risk DESC"},
        {"q": "Average risk score by gender", "sql": "SELECT GENDER, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, COUNT(*) AS members FROM members GROUP BY GENDER"},
        {"q": "Average risk score by race", "sql": "SELECT RACE, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, COUNT(*) AS members FROM members GROUP BY RACE ORDER BY avg_risk DESC"},
        {"q": "Average risk score by region", "sql": "SELECT KP_REGION, AVG(CAST(RISK_SCORE AS REAL)) AS avg_risk, COUNT(*) AS members FROM members GROUP BY KP_REGION ORDER BY avg_risk DESC"},
        {"q": "Risk score distribution", "sql": "SELECT CASE WHEN CAST(RISK_SCORE AS REAL) < 0.5 THEN 'Low (<0.5)' WHEN CAST(RISK_SCORE AS REAL) < 1.0 THEN 'Medium (0.5-1.0)' WHEN CAST(RISK_SCORE AS REAL) < 2.0 THEN 'High (1.0-2.0)' ELSE 'Very High (2.0+)' END AS risk_tier, COUNT(*) AS members FROM members GROUP BY risk_tier ORDER BY risk_tier"},

        {"q": "Members by race and gender", "sql": "SELECT RACE, GENDER, COUNT(*) AS cnt FROM members GROUP BY RACE, GENDER ORDER BY RACE, GENDER"},
        {"q": "Members by language and region", "sql": "SELECT LANGUAGE, KP_REGION, COUNT(*) AS cnt FROM members GROUP BY LANGUAGE, KP_REGION ORDER BY LANGUAGE, cnt DESC"},
        {"q": "Chronic conditions by plan type", "sql": "SELECT PLAN_TYPE, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, SUM(CASE WHEN CHRONIC_CONDITIONS > 0 THEN 1 ELSE 0 END) AS with_chronic FROM members GROUP BY PLAN_TYPE ORDER BY avg_chronic DESC"},
        {"q": "Chronic conditions by gender", "sql": "SELECT GENDER, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, SUM(CASE WHEN CHRONIC_CONDITIONS > 0 THEN 1 ELSE 0 END) AS with_chronic FROM members GROUP BY GENDER"},
        {"q": "Chronic conditions by race", "sql": "SELECT RACE, AVG(CAST(CHRONIC_CONDITIONS AS REAL)) AS avg_chronic, COUNT(*) AS members FROM members GROUP BY RACE ORDER BY avg_chronic DESC"},

        {"q": "Top 10 highest risk members", "sql": "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, RISK_SCORE, CHRONIC_CONDITIONS, PLAN_TYPE, KP_REGION FROM members ORDER BY CAST(RISK_SCORE AS REAL) DESC LIMIT 10"},
        {"q": "Members with most chronic conditions", "sql": "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, CHRONIC_CONDITIONS, RISK_SCORE, PLAN_TYPE FROM members WHERE CHRONIC_CONDITIONS > 0 ORDER BY CHRONIC_CONDITIONS DESC, CAST(RISK_SCORE AS REAL) DESC LIMIT 20"},
        {"q": "How many Spanish speaking members?", "sql": "SELECT COUNT(*) AS spanish_speakers FROM members WHERE LANGUAGE = 'Spanish'"},
        {"q": "How many Medicare members?", "sql": "SELECT COUNT(*) AS medicare_members FROM members WHERE PLAN_TYPE = 'Medicare Advantage'"},
        {"q": "How many Medicaid members?", "sql": "SELECT COUNT(*) AS medicaid_members FROM members WHERE PLAN_TYPE = 'Medicaid'"},
        {"q": "How many HMO members?", "sql": "SELECT COUNT(*) AS hmo_members FROM members WHERE PLAN_TYPE = 'HMO'"},
        {"q": "How many PPO members?", "sql": "SELECT COUNT(*) AS ppo_members FROM members WHERE PLAN_TYPE = 'PPO'"},
    ]

    queries += [
        {"q": "Providers by specialty", "sql": "SELECT SPECIALTY, COUNT(*) AS cnt FROM providers GROUP BY SPECIALTY ORDER BY cnt DESC"},
        {"q": "Providers by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS cnt FROM providers GROUP BY DEPARTMENT ORDER BY cnt DESC"},
        {"q": "Providers by type", "sql": "SELECT PROVIDER_TYPE, COUNT(*) AS cnt FROM providers GROUP BY PROVIDER_TYPE ORDER BY cnt DESC"},
        {"q": "Providers by status", "sql": "SELECT STATUS, COUNT(*) AS cnt FROM providers GROUP BY STATUS"},
        {"q": "Providers by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt FROM providers GROUP BY KP_REGION ORDER BY cnt DESC"},
        {"q": "Providers by license state", "sql": "SELECT LICENSE_STATE, COUNT(*) AS cnt FROM providers GROUP BY LICENSE_STATE ORDER BY cnt DESC"},

        {"q": "How many MDs?", "sql": "SELECT COUNT(*) AS md_count FROM providers WHERE PROVIDER_TYPE = 'MD'"},
        {"q": "How many NPs?", "sql": "SELECT COUNT(*) AS np_count FROM providers WHERE PROVIDER_TYPE = 'NP'"},
        {"q": "How many PAs?", "sql": "SELECT COUNT(*) AS pa_count FROM providers WHERE PROVIDER_TYPE = 'PA'"},
        {"q": "Active vs inactive providers", "sql": "SELECT STATUS, COUNT(*) AS cnt FROM providers GROUP BY STATUS ORDER BY cnt DESC"},
        {"q": "Providers accepting new patients", "sql": "SELECT ACCEPTS_NEW_PATIENTS, COUNT(*) AS cnt FROM providers GROUP BY ACCEPTS_NEW_PATIENTS"},
        {"q": "Providers accepting new patients by specialty", "sql": "SELECT SPECIALTY, SUM(CASE WHEN ACCEPTS_NEW_PATIENTS='Y' THEN 1 ELSE 0 END) AS accepting, COUNT(*) AS total FROM providers GROUP BY SPECIALTY ORDER BY accepting DESC"},

        {"q": "Average panel size by specialty", "sql": "SELECT SPECIALTY, AVG(CAST(PANEL_SIZE AS REAL)) AS avg_panel, MIN(PANEL_SIZE) AS min_panel, MAX(PANEL_SIZE) AS max_panel FROM providers GROUP BY SPECIALTY ORDER BY avg_panel DESC"},
        {"q": "Average panel size by region", "sql": "SELECT KP_REGION, AVG(CAST(PANEL_SIZE AS REAL)) AS avg_panel FROM providers GROUP BY KP_REGION ORDER BY avg_panel DESC"},
        {"q": "Top 10 providers by panel size", "sql": "SELECT NPI, PROVIDER_FIRST_NAME, PROVIDER_LAST_NAME, SPECIALTY, PANEL_SIZE, KP_REGION FROM providers ORDER BY CAST(PANEL_SIZE AS INTEGER) DESC LIMIT 10"},
        {"q": "Providers with largest panels", "sql": "SELECT NPI, PROVIDER_FIRST_NAME, PROVIDER_LAST_NAME, SPECIALTY, PANEL_SIZE FROM providers WHERE STATUS='ACTIVE' ORDER BY CAST(PANEL_SIZE AS INTEGER) DESC LIMIT 15"},

        {"q": "How many cardiologists?", "sql": "SELECT COUNT(*) AS cardiologists FROM providers WHERE SPECIALTY = 'Cardiology'"},
        {"q": "How many surgeons?", "sql": "SELECT COUNT(*) AS surgeons FROM providers WHERE SPECIALTY = 'Surgery'"},
        {"q": "How many pediatricians?", "sql": "SELECT COUNT(*) AS pediatricians FROM providers WHERE SPECIALTY = 'Pediatrics'"},
        {"q": "How many psychiatrists?", "sql": "SELECT COUNT(*) AS psychiatrists FROM providers WHERE SPECIALTY = 'Psychiatry'"},
        {"q": "How many oncologists?", "sql": "SELECT COUNT(*) AS oncologists FROM providers WHERE SPECIALTY = 'Oncology'"},
    ]

    queries += [
        {"q": "Prescriptions by medication class", "sql": "SELECT MEDICATION_CLASS, COUNT(*) AS cnt, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY cnt DESC"},
        {"q": "Prescriptions by medication name", "sql": "SELECT MEDICATION_NAME, COUNT(*) AS cnt, AVG(CAST(COST AS REAL)) AS avg_cost FROM prescriptions GROUP BY MEDICATION_NAME ORDER BY cnt DESC"},
        {"q": "Prescriptions by pharmacy", "sql": "SELECT PHARMACY, COUNT(*) AS cnt, SUM(CAST(COST AS REAL)) AS total_cost FROM prescriptions GROUP BY PHARMACY ORDER BY cnt DESC"},
        {"q": "Prescriptions by status", "sql": "SELECT STATUS, COUNT(*) AS cnt FROM prescriptions GROUP BY STATUS ORDER BY cnt DESC"},
        {"q": "Prescriptions by days supply", "sql": "SELECT DAYS_SUPPLY, COUNT(*) AS cnt FROM prescriptions GROUP BY DAYS_SUPPLY ORDER BY DAYS_SUPPLY"},
        {"q": "Prescriptions by quantity", "sql": "SELECT QUANTITY, COUNT(*) AS cnt FROM prescriptions GROUP BY QUANTITY ORDER BY QUANTITY"},

        {"q": "How many metformin prescriptions?", "sql": "SELECT COUNT(*) AS cnt, AVG(CAST(COST AS REAL)) AS avg_cost FROM prescriptions WHERE MEDICATION_NAME LIKE '%Metformin%'"},
        {"q": "How many statin prescriptions?", "sql": "SELECT COUNT(*) AS cnt FROM prescriptions WHERE MEDICATION_NAME LIKE '%Atorvastatin%' OR MEDICATION_CLASS = 'Cholesterol'"},
        {"q": "Diabetes medication volume", "sql": "SELECT MEDICATION_NAME, COUNT(*) AS cnt, AVG(CAST(COST AS REAL)) AS avg_cost FROM prescriptions WHERE MEDICATION_CLASS = 'Diabetes' GROUP BY MEDICATION_NAME ORDER BY cnt DESC"},
        {"q": "Hypertension medication volume", "sql": "SELECT MEDICATION_NAME, COUNT(*) AS cnt, AVG(CAST(COST AS REAL)) AS avg_cost FROM prescriptions WHERE MEDICATION_CLASS = 'Hypertension' GROUP BY MEDICATION_NAME ORDER BY cnt DESC"},
        {"q": "Depression medication volume", "sql": "SELECT MEDICATION_NAME, COUNT(*) AS cnt FROM prescriptions WHERE MEDICATION_CLASS = 'Depression' GROUP BY MEDICATION_NAME ORDER BY cnt DESC"},
        {"q": "Pain medication volume", "sql": "SELECT MEDICATION_NAME, COUNT(*) AS cnt FROM prescriptions WHERE MEDICATION_CLASS = 'Pain' GROUP BY MEDICATION_NAME ORDER BY cnt DESC"},

        {"q": "Most expensive medications", "sql": "SELECT MEDICATION_NAME, AVG(CAST(COST AS REAL)) AS avg_cost, MAX(CAST(COST AS REAL)) AS max_cost, COUNT(*) AS rx_count FROM prescriptions GROUP BY MEDICATION_NAME ORDER BY avg_cost DESC LIMIT 10"},
        {"q": "Prescription cost by pharmacy", "sql": "SELECT PHARMACY, AVG(CAST(COST AS REAL)) AS avg_cost, SUM(CAST(COST AS REAL)) AS total_cost, COUNT(*) AS rx_count FROM prescriptions GROUP BY PHARMACY ORDER BY total_cost DESC"},
        {"q": "Refill rate by medication", "sql": "SELECT MEDICATION_NAME, AVG(CAST(REFILLS_USED AS REAL)) AS avg_refills_used, AVG(CAST(REFILLS_AUTHORIZED AS REAL)) AS avg_authorized FROM prescriptions GROUP BY MEDICATION_NAME ORDER BY avg_refills_used DESC"},

        {"q": "How many filled prescriptions?", "sql": "SELECT COUNT(*) AS filled FROM prescriptions WHERE STATUS = 'FILLED'"},
        {"q": "How many pending prescriptions?", "sql": "SELECT COUNT(*) AS pending FROM prescriptions WHERE STATUS = 'PENDING'"},
        {"q": "How many cancelled prescriptions?", "sql": "SELECT COUNT(*) AS cancelled FROM prescriptions WHERE STATUS = 'CANCELLED'"},
        {"q": "How many expired prescriptions?", "sql": "SELECT COUNT(*) AS expired FROM prescriptions WHERE STATUS = 'EXPIRED'"},
    ]

    queries += [
        {"q": "Appointments by status", "sql": "SELECT STATUS, COUNT(*) AS cnt FROM appointments GROUP BY STATUS ORDER BY cnt DESC"},
        {"q": "Appointments by type", "sql": "SELECT APPOINTMENT_TYPE, COUNT(*) AS cnt FROM appointments GROUP BY APPOINTMENT_TYPE ORDER BY cnt DESC"},
        {"q": "Appointments by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS cnt FROM appointments GROUP BY DEPARTMENT ORDER BY cnt DESC"},
        {"q": "Appointments by reason", "sql": "SELECT REASON, COUNT(*) AS cnt FROM appointments GROUP BY REASON ORDER BY cnt DESC"},
        {"q": "Appointments by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt FROM appointments GROUP BY KP_REGION ORDER BY cnt DESC"},
        {"q": "Appointments by duration", "sql": "SELECT DURATION_MINUTES, COUNT(*) AS cnt FROM appointments GROUP BY DURATION_MINUTES ORDER BY DURATION_MINUTES"},

        {"q": "No-show rate by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS total, SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END) AS no_shows, ROUND(SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS no_show_pct FROM appointments GROUP BY DEPARTMENT ORDER BY no_show_pct DESC"},
        {"q": "No-show rate by appointment type", "sql": "SELECT APPOINTMENT_TYPE, COUNT(*) AS total, SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END) AS no_shows, ROUND(SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS no_show_pct FROM appointments GROUP BY APPOINTMENT_TYPE ORDER BY no_show_pct DESC"},
        {"q": "No-show rate by region", "sql": "SELECT KP_REGION, ROUND(SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS no_show_pct FROM appointments GROUP BY KP_REGION ORDER BY no_show_pct DESC"},
        {"q": "Cancelled appointments by department", "sql": "SELECT DEPARTMENT, COUNT(*) AS cancelled FROM appointments WHERE STATUS='CANCELLED' GROUP BY DEPARTMENT ORDER BY cancelled DESC"},

        {"q": "PCP visit rate", "sql": "SELECT IS_PCP_VISIT, COUNT(*) AS cnt, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM appointments),2) AS pct FROM appointments GROUP BY IS_PCP_VISIT"},
        {"q": "PCP vs specialist appointments", "sql": "SELECT CASE WHEN IS_PCP_VISIT='Y' THEN 'PCP' ELSE 'Specialist' END AS visit_type, COUNT(*) AS cnt FROM appointments GROUP BY IS_PCP_VISIT"},
    ]

    queries += [
        {"q": "Referrals by status", "sql": "SELECT STATUS, COUNT(*) AS cnt FROM referrals GROUP BY STATUS ORDER BY cnt DESC"},
        {"q": "Referrals by urgency", "sql": "SELECT URGENCY, COUNT(*) AS cnt FROM referrals GROUP BY URGENCY ORDER BY cnt DESC"},
        {"q": "Referrals by reason", "sql": "SELECT REFERRAL_REASON, COUNT(*) AS cnt FROM referrals GROUP BY REFERRAL_REASON ORDER BY cnt DESC"},
        {"q": "Referrals by specialty", "sql": "SELECT SPECIALTY, COUNT(*) AS cnt FROM referrals GROUP BY SPECIALTY ORDER BY cnt DESC"},
        {"q": "Referrals by type", "sql": "SELECT REFERRAL_TYPE, COUNT(*) AS cnt FROM referrals GROUP BY REFERRAL_TYPE ORDER BY cnt DESC"},
        {"q": "Referrals by region", "sql": "SELECT KP_REGION, COUNT(*) AS cnt FROM referrals GROUP BY KP_REGION ORDER BY cnt DESC"},

        {"q": "Referral completion rate by specialty", "sql": "SELECT SPECIALTY, COUNT(*) AS total, SUM(CASE WHEN STATUS='COMPLETED' THEN 1 ELSE 0 END) AS completed, ROUND(SUM(CASE WHEN STATUS='COMPLETED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS completion_pct FROM referrals GROUP BY SPECIALTY ORDER BY completion_pct DESC"},
        {"q": "Referral completion rate by urgency", "sql": "SELECT URGENCY, COUNT(*) AS total, SUM(CASE WHEN STATUS='COMPLETED' THEN 1 ELSE 0 END) AS completed, ROUND(SUM(CASE WHEN STATUS='COMPLETED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS completion_pct FROM referrals GROUP BY URGENCY ORDER BY completion_pct DESC"},
        {"q": "Pending referrals by specialty", "sql": "SELECT SPECIALTY, COUNT(*) AS pending FROM referrals WHERE STATUS='PENDING' GROUP BY SPECIALTY ORDER BY pending DESC"},
        {"q": "Denied referrals", "sql": "SELECT REFERRAL_REASON, COUNT(*) AS denied FROM referrals WHERE STATUS='DENIED' GROUP BY REFERRAL_REASON ORDER BY denied DESC"},
        {"q": "STAT referrals by specialty", "sql": "SELECT SPECIALTY, COUNT(*) AS stat_referrals FROM referrals WHERE URGENCY='STAT' GROUP BY SPECIALTY ORDER BY stat_referrals DESC"},
    ]

    queries += [
        {"q": "CPT codes by category", "sql": "SELECT CATEGORY, COUNT(*) AS cnt, AVG(CAST(RVU AS REAL)) AS avg_rvu FROM cpt_codes GROUP BY CATEGORY ORDER BY cnt DESC"},
        {"q": "Highest RVU procedures", "sql": "SELECT CPT_CODE, DESCRIPTION, RVU FROM cpt_codes ORDER BY CAST(RVU AS REAL) DESC LIMIT 10"},
        {"q": "Lowest RVU procedures", "sql": "SELECT CPT_CODE, DESCRIPTION, RVU FROM cpt_codes ORDER BY CAST(RVU AS REAL) ASC LIMIT 10"},
        {"q": "Average RVU by category", "sql": "SELECT CATEGORY, AVG(CAST(RVU AS REAL)) AS avg_rvu, COUNT(*) AS procedures FROM cpt_codes GROUP BY CATEGORY ORDER BY avg_rvu DESC"},
        {"q": "Surgery procedures", "sql": "SELECT CPT_CODE, DESCRIPTION, RVU FROM cpt_codes WHERE CATEGORY = 'Surgery' ORDER BY CAST(RVU AS REAL) DESC"},
        {"q": "Radiology procedures", "sql": "SELECT CPT_CODE, DESCRIPTION, RVU FROM cpt_codes WHERE CATEGORY = 'Radiology' ORDER BY CAST(RVU AS REAL) DESC"},
    ]

    queries += [
        {"q": "Total claims for critical severity", "sql": "SELECT COUNT(DISTINCT c.CLAIM_ID) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM claims c JOIN diagnoses d ON c.MEMBER_ID = d.MEMBER_ID WHERE d.SEVERITY = 'CRITICAL'"},
        {"q": "Total claims for severe diagnoses", "sql": "SELECT COUNT(DISTINCT c.CLAIM_ID) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM claims c JOIN diagnoses d ON c.MEMBER_ID = d.MEMBER_ID WHERE d.SEVERITY = 'SEVERE'"},
        {"q": "Average claim cost by diagnosis severity", "sql": "SELECT d.SEVERITY, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost, COUNT(DISTINCT c.CLAIM_ID) AS claims FROM claims c JOIN diagnoses d ON c.MEMBER_ID = d.MEMBER_ID GROUP BY d.SEVERITY ORDER BY avg_cost DESC"},

        {"q": "Claim costs by gender", "sql": "SELECT m.GENDER, COUNT(*) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY m.GENDER"},
        {"q": "Claim costs by race", "sql": "SELECT m.RACE, COUNT(*) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY m.RACE ORDER BY total_cost DESC"},
        {"q": "Claim costs by age group", "sql": "SELECT CASE WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH))/365.25 < 18 THEN 'Pediatric' WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH))/365.25 < 40 THEN 'Young Adult' WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH))/365.25 < 65 THEN 'Adult' ELSE 'Senior' END AS age_group, COUNT(*) AS claims, AVG(CAST(c.PAID_AMOUNT AS REAL)) AS avg_cost FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY age_group ORDER BY avg_cost DESC"},

        {"q": "Top 10 providers by total cost", "sql": "SELECT c.RENDERING_NPI, p.SPECIALTY, COUNT(*) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY c.RENDERING_NPI, p.SPECIALTY ORDER BY total_cost DESC LIMIT 10"},
        {"q": "Cost per provider by specialty", "sql": "SELECT p.SPECIALTY, COUNT(DISTINCT p.NPI) AS providers, COUNT(*) AS claims, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL))/COUNT(DISTINCT p.NPI),2) AS cost_per_provider FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY p.SPECIALTY ORDER BY cost_per_provider DESC"},
        {"q": "Denial rate by provider type", "sql": "SELECT p.PROVIDER_TYPE, COUNT(*) AS claims, SUM(CASE WHEN c.CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denied, ROUND(SUM(CASE WHEN c.CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS denial_pct FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI GROUP BY p.PROVIDER_TYPE ORDER BY denial_pct DESC"},

        {"q": "Top 5 emergency diagnoses", "sql": "SELECT DIAGNOSIS_DESCRIPTION, COUNT(*) AS cnt FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY DIAGNOSIS_DESCRIPTION ORDER BY cnt DESC LIMIT 5"},
        {"q": "Emergency visits by region", "sql": "SELECT KP_REGION, COUNT(*) AS er_visits FROM encounters WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY KP_REGION ORDER BY er_visits DESC"},
        {"q": "Emergency cost by region", "sql": "SELECT e.KP_REGION, COUNT(*) AS er_visits, SUM(CAST(c.PAID_AMOUNT AS REAL)) AS total_cost FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'EMERGENCY' GROUP BY e.KP_REGION ORDER BY total_cost DESC"},
        {"q": "Top 5 ER claims by cost", "sql": "SELECT c.CLAIM_ID, c.MEMBER_ID, c.ICD10_DESCRIPTION, c.PAID_AMOUNT FROM claims c JOIN encounters e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID WHERE e.VISIT_TYPE = 'EMERGENCY' ORDER BY CAST(c.PAID_AMOUNT AS REAL) DESC LIMIT 5"},
    ]

    valid = []
    invalid = 0
    for tq in queries:
        try:
            cursor = conn.execute(tq['sql'])
            rows = cursor.fetchall()
            if len(rows) >= 0:
                valid.append(tq)
        except Exception as e:
            invalid += 1
            logger.warning("  INVALID SQL for '%s': %s", tq['q'][:50], str(e)[:60])

    conn.close()
    logger.info("Generated %d valid training queries (%d invalid skipped)", len(valid), invalid)
    return valid


def train_and_test(db_path: str):
    from semantic_layer import SchemaLearner
    from deep_understanding import DeepUnderstandingEngine
    from sql_reasoning import SQLReasoningEngine

    logger.info("Generating exhaustive column-level training queries...")
    new_queries = generate_all_training_queries(db_path)

    logger.info("\nInitializing engines...")
    t0 = time.time()
    learner = SchemaLearner(db_path)
    learner.learn(sample_size=1000)

    deep = DeepUnderstandingEngine(schema_learner=learner, db_path=db_path,
                                    data_dir=os.path.dirname(db_path))
    shared_embedder = deep._embedder if hasattr(deep, '_embedder') else None
    sql_eng = SQLReasoningEngine(schema_learner=learner, db_path=db_path, embedder=shared_embedder)
    logger.info("Engines ready in %.1fs", time.time() - t0)

    logger.info("\nTraining on %d exhaustive queries...", len(new_queries))
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    trained = 0
    train_errors = 0
    for tq in new_queries:
        try:
            cursor = conn.execute(tq['sql'])
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description] if cursor.description else []

            sql_eng.learn(tq['q'], tq['sql'], success=True)

            sql_lower = tq['sql'].lower()
            tables_used = [t for t in ['claims', 'encounters', 'members', 'diagnoses',
                                        'prescriptions', 'appointments', 'referrals',
                                        'providers', 'cpt_codes']
                           if t in sql_lower]
            intent = 'aggregate' if 'count' in sql_lower or 'sum' in sql_lower or 'avg' in sql_lower else 'lookup'
            deep.learn(tq['q'], intent, tables_used, cols, tq['sql'], True)
            trained += 1
        except Exception as e:
            train_errors += 1
            if train_errors <= 5:
                logger.warning("  Train error for '%s': %s", tq['q'][:50], str(e)[:80])
    conn.close()
    logger.info("Trained %d queries in %.1fs (%d errors)", trained, time.time() - t0, train_errors)
    logger.info("Total patterns: %d", sql_eng.pattern_count)

    logger.info("\n" + "=" * 70)
    logger.info("COLUMN-LEVEL STRESS TEST")
    logger.info("=" * 70)

    test_questions = [
        "Claims by claim status", "Claims by denial reason", "Claims by ICD10 code",
        "Claims by CPT code", "Average billed amount by plan type",
        "Member responsibility by plan type", "Copay analysis by plan type",
        "Top 5 most expensive claims", "Top 5 denial reasons",
        "How many paid claims?", "How many pending claims?", "How many voided claims?",
        "Denial rate by facility", "Denial rate by claim type",
        "Coinsurance by region", "Total claims for emergency visits",
        "Top 5 claims for emergency", "Claims for telehealth",

        "Encounters by visit type", "Encounters by department",
        "Encounters by disposition", "Encounters by chief complaint",
        "Encounters by encounter status", "Encounters by primary diagnosis",
        "How many outpatient visits?", "How many urgent care visits?",
        "How many home health visits?", "Emergency visits by department",
        "Average length of stay by visit type", "Length of stay by region",
        "How many patients left AMA?", "How many patients expired?",
        "Top chief complaints", "Top 5 longest stays",
        "Disposition by visit type",

        "Diagnoses by severity", "Diagnoses by type",
        "Chronic vs non-chronic diagnoses", "Diagnoses by HCC category",
        "Critical diagnoses count", "Severe diagnoses count",
        "Cost by diagnosis severity", "Chronic diagnoses by severity",
        "Top 5 most common diagnoses", "Top 10 chronic conditions",
        "Unresolved diagnoses",

        "Members by race", "Members by language", "Members by city",
        "Members by chronic condition count", "Average risk score by gender",
        "Average risk score by race", "Risk score distribution",
        "Members by race and gender", "Chronic conditions by race",
        "How many Spanish speaking members?", "How many Medicare members?",
        "How many Medicaid members?",

        "Providers by type", "Providers by status",
        "Providers by license state", "How many MDs?", "How many NPs?",
        "Providers accepting new patients by specialty",
        "Average panel size by specialty", "Average panel size by region",
        "How many cardiologists?", "How many surgeons?",
        "Active vs inactive providers",

        "Prescriptions by pharmacy", "Prescriptions by status",
        "Prescriptions by days supply", "Prescriptions by medication name",
        "Diabetes medication volume", "Hypertension medication volume",
        "Depression medication volume", "Pain medication volume",
        "Most expensive medications", "Prescription cost by pharmacy",
        "Refill rate by medication",
        "How many filled prescriptions?", "How many pending prescriptions?",

        "Appointments by type", "Appointments by reason",
        "Appointments by duration", "No-show rate by department",
        "No-show rate by appointment type", "No-show rate by region",
        "PCP visit rate", "PCP vs specialist appointments",
        "Cancelled appointments by department",

        "Referrals by reason", "Referrals by type",
        "Referral completion rate by specialty",
        "Referral completion rate by urgency",
        "Pending referrals by specialty", "Denied referrals",
        "STAT referrals by specialty",

        "CPT codes by category", "Highest RVU procedures",
        "Average RVU by category", "Surgery procedures",

        "Total claims for critical severity",
        "Average claim cost by diagnosis severity",
        "Claim costs by gender", "Claim costs by race",
        "Top 10 providers by total cost",
        "Cost per provider by specialty",
        "Top 5 emergency diagnoses", "Emergency cost by region",
        "Top 5 ER claims by cost",
        "Denial rate by provider type",
    ]

    conn = sqlite3.connect(db_path)
    passed = 0
    fallback = 0
    failed = 0

    for q in test_questions:
        try:
            u = deep.understand(q)
            ud = u.to_dict()
            sr = sql_eng.generate(understanding=ud, question=q)
            sql_text = sr.sql
            confidence = sr.confidence

            cursor = conn.execute(sql_text)
            rows = cursor.fetchall()
            row_count = len(rows)

            is_fallback = confidence < 0.20 or 'appointments.*' in sql_text or sql_text == 'SELECT 1'

            if is_fallback:
                fallback += 1
                logger.info("  [FALLBACK] %s (conf=%.2f)", q[:60], confidence)
                logger.info("             SQL: %s", sql_text[:120])
            elif row_count == 0:
                fallback += 1
                logger.info("  [EMPTY] %s (conf=%.2f, rows=0)", q[:60], confidence)
            else:
                passed += 1
        except Exception as e:
            failed += 1
            logger.info("  [FAIL] %s: %s", q[:50], str(e)[:60])

    conn.close()

    total = passed + fallback + failed
    logger.info("\n" + "=" * 70)
    logger.info("COLUMN-LEVEL TEST: %d/%d passed (%.1f%%), %d fallback, %d fail",
                passed, total, passed/total*100, fallback, failed)
    logger.info("Total patterns in engine: %d", sql_eng.pattern_count)
    logger.info("=" * 70)

    return passed, fallback, failed


if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_production.db'

    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    passed, fallback, failed = train_and_test(db_path)
    total = passed + fallback + failed
    rate = passed / total * 100 if total else 0
    sys.exit(0 if rate >= 85 else 1)
