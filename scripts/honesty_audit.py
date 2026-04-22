import sqlite3
import os
import sys
import time
import re
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_layer import SchemaLearner
from deep_understanding import DeepUnderstandingEngine
from sql_reasoning import SQLReasoningEngine


EXECUTIVE_FALLBACK_MARKERS = [
    "SELECT 'Total Claims'",
    "'Total Paid'",
    "'Denial Rate %'",
    "'Unique Members'",
    "'Total Encounters'",
    "fallback_executive",
    "UNION ALL SELECT 'Total",
]

def is_executive_fallback(sql_text: str, approach: str) -> bool:
    sql_upper = sql_text.upper()
    for marker in EXECUTIVE_FALLBACK_MARKERS:
        if marker.upper() in sql_upper:
            return True
    if approach and 'fallback' in approach.lower():
        return True
    return False


def question_expects_table(question: str) -> str:
    q = question.lower()

    phrase_map = [
        ('no-show', 'appointments'), ('no show', 'appointments'), ('noshow', 'appointments'),
        ('cancelled appointment', 'appointments'), ('pcp visit', 'appointments'),
        ('appointment type', 'appointments'), ('appointment status', 'appointments'),
        ('appointment duration', 'appointments'), ('appointments by', 'appointments'),
        ('appointment no', 'appointments'), ('appointment volume', 'appointments'),
        ('length of stay', 'encounters'), ('chief complaint', 'encounters'),
        ('er visit', 'encounters'), ('urgent care', 'encounters'),
        ('home health', 'encounters'), ('visit type', 'encounters'),
        ('paid amount', 'claims'), ('billed amount', 'claims'),
        ('cost per member', 'claims'), ('pmpm', 'claims'),
        ('hmo vs', 'claims'), ('ppo vs', 'claims'), ('vs ppo', 'claims'),
        ('vs hmo', 'claims'), ('hmo cost', 'claims'), ('ppo cost', 'claims'),
        ('commercial vs', 'claims'), ('vs medicare', 'claims'),
        ('medication cost', 'prescriptions'), ('prescription cost', 'prescriptions'),
        ('drug cost', 'prescriptions'), ('pain medication', 'prescriptions'),
        ('accepting new patient', 'providers'), ('panel size', 'providers'),
        ('by specialty', 'providers'),
        ('chronic condition', 'diagnoses'), ('icd10 code', 'diagnoses'),
        ('cpt code', 'cpt_codes'), ('procedure category', 'cpt_codes'),
        ('cpt category', 'cpt_codes'),
        ('pending referral', 'referrals'), ('stat referral', 'referrals'),
        ('denied referral', 'referrals'), ('referral completion', 'referrals'),
    ]

    for phrase, table in phrase_map:
        if phrase in q:
            return table

    word_map = [
        ('appointment', 'appointments'),
        ('referral', 'referrals'),
        ('prescription', 'prescriptions'), ('medication', 'prescriptions'),
        ('pharmacy', 'prescriptions'), ('drug', 'prescriptions'),
        ('refill', 'prescriptions'), ('metformin', 'prescriptions'),
        ('statin', 'prescriptions'),
        ('provider', 'providers'), ('specialist', 'providers'),
        ('cardiologist', 'providers'), ('surgeon', 'providers'),
        ('pediatrician', 'providers'), ('psychiatrist', 'providers'),
        ('oncologist', 'providers'), ('npi', 'providers'),
        ('diagnosis', 'diagnoses'), ('diagnoses', 'diagnoses'),
        ('severity', 'diagnoses'), ('chronic', 'diagnoses'),
        ('hcc', 'diagnoses'),
        ('encounter', 'encounters'), ('admission', 'encounters'),
        ('inpatient', 'encounters'), ('outpatient', 'encounters'),
        ('telehealth', 'encounters'), ('emergency', 'encounters'),
        ('disposition', 'encounters'), ('discharged', 'encounters'),
        ('expired', 'encounters'), ('transferred', 'encounters'),
        ('ama', 'encounters'),
        ('claim', 'claims'), ('denial', 'claims'), ('denied', 'claims'),
        ('copay', 'claims'), ('coinsurance', 'claims'),
        ('adjudicate', 'claims'), ('billed', 'claims'),
        ('rvu', 'cpt_codes'), ('cpt', 'cpt_codes'),
        ('member', 'members'), ('patient', 'members'),
        ('enrolled', 'members'), ('risk score', 'members'),
        ('gender', 'members'), ('race', 'members'),
        ('language', 'members'), ('demographics', 'members'),
        ('population', 'members'),
    ]

    for word, table in word_map:
        if word in q:
            return table

    if 'department' in q:
        if 'appointment' in q or 'no-show' in q or 'no show' in q or 'cancel' in q:
            return 'appointments'
        return 'encounters'

    if 'cost' in q or 'paid' in q or 'expense' in q:
        return 'claims'

    return 'unknown'


def sql_uses_table(sql_text: str, expected_table: str) -> bool:
    sql_lower = sql_text.lower()
    if expected_table in sql_lower:
        return True

    related = {
        'claims': ['prescriptions', 'encounters'],
        'encounters': ['appointments', 'claims'],
        'members': ['claims', 'encounters'],
        'diagnoses': ['members', 'claims', 'encounters'],
        'providers': ['claims', 'encounters', 'prescriptions'],
        'prescriptions': ['claims'],
        'appointments': ['encounters'],
        'cpt_codes': ['claims'],
    }
    for alt_table in related.get(expected_table, []):
        if alt_table in sql_lower:
            return True
    return False


def run_audit(db_path: str):
    all_questions = []
    
    stress_questions = [
        "How many claims do we have?", "Total number of members", "Count of encounters",
        "How many providers are there?", "Number of prescriptions", "How many appointments total?",
        "Count all referrals", "How many diagnoses?", "Total claims count", "Number of unique members",
        "What is the total paid amount?", "Average claim cost", "Total billed amount across all claims",
        "What is the average length of stay?", "Average risk score for all members",
        "Total prescription costs", "Sum of all copays", "Maximum paid amount on a single claim",
        "Minimum length of stay", "Average days supply for prescriptions",
        "What is our denial rate?", "What is the yield rate?", "Appointment no-show rate",
        "What percentage of diagnoses are chronic?", "ER utilization rate per 1000 members",
        "What percentage of claims are denied?", "Claim approval rate",
        "What fraction of referrals are pending?", "Readmission rate",
        "What percent of encounters are emergency?",
        "Claims by status", "Cost by region", "Members by plan type", "Encounters by visit type",
        "Providers by specialty", "Prescriptions by medication class", "Referrals by urgency",
        "Diagnoses by severity", "Claims by claim type", "Appointments by department",
        "Members by gender", "Members by state", "Claims by facility",
        "Encounters by department", "Referrals by status",
        "Monthly claim volume trend", "Cost trend over time", "Denial rate trend by month",
        "Monthly enrollment changes", "ER visits by month", "Prescription volume trend",
        "Referral trends by month", "Monthly revenue trend", "Inpatient admissions by month",
        "Monthly appointment volume",
        "Top 10 highest cost members", "Top providers by claim volume", "Most expensive diagnoses",
        "Top 10 most common procedures", "Highest cost facilities", "Top denial reasons",
        "Most prescribed medications", "Highest risk score members", "Top specialties by claim count",
        "Busiest departments",
        "Compare HMO vs PPO costs", "Compare denial rates between regions",
        "Inpatient vs outpatient cost comparison", "Compare ER visits across regions",
        "Male vs female claim costs", "Compare medication costs by class",
        "Commercial vs Medicare costs", "Compare appointment no-show rates by department",
        "Compare length of stay by visit type", "HMO vs Medicare Advantage PMPM",
        "Show me all denied claims over $10000", "List high risk patients",
        "Show me pending referrals", "Find members with diabetes", "Show urgent referrals",
        "List claims from last year", "Show me members over 65", "Find claims with no encounter",
        "Show high cost claims over 50000", "List active providers",
        "What is our PMPM?", "PMPM by plan type", "Average length of stay by department",
        "Yield rate by region", "ER per 1000 by region", "Revenue by facility",
        "Cost per member by region", "Denial rate by plan type",
        "Average days to adjudicate claims", "Readmission rate by department",
        "Claims with member details", "Provider performance summary",
        "Diagnosis costs by condition", "Average cost per encounter type",
        "Provider specialty distribution with claim counts",
        "Member claims with diagnosis information",
        "Top providers with their specialty and claim volume",
        "Encounters with primary diagnosis breakdown",
        "Prescription costs by provider specialty", "Referral completion rate by specialty",
        "Give me a summary of our claims", "Executive dashboard KPIs",
        "Overall health system performance", "Executive summary of operations",
        "What are our key performance indicators?", "System-wide dashboard metrics",
        "How is our organization performing?", "Give me the big picture on costs",
        "Financial overview of the health system", "Clinical quality scorecard",
        "Why are denial rates high in the Northwest?", "What's driving high costs?",
        "Which departments have the most ER visits?", "Why is the readmission rate increasing?",
        "What's causing high costs in California?", "Which conditions contribute most to cost?",
        "Why do some facilities have more denials?", "What drives high prescription costs?",
        "Which plan types have the worst outcomes?", "Root cause of high ER utilization",
        "how much did we spend on claims?", "give me a breakdown of costs by plan",
        "which members cost us the most?", "show me everything about denials",
        "what are our biggest cost drivers?", "i need to know about ER utilization",
        "tell me about our high risk population", "break down prescriptions by drug type",
        "summarize our financial performance", "what does our member population look like?",
        "how many clams do we have?", "total deniel rate", "membrs by plan type",
        "presciption costs", "averge length of stay", "encountr volume by month",
        "top provders by claims", "referall trends", "diagnoseis by severity",
        "appoitment no show rate",
        "show me the numbers", "how are we doing?", "give me some data",
        "what's the situation?", "tell me about costs", "show me trends",
        "what should I know?", "any issues?", "performance summary", "the big picture",
        "claims", "denials", "PMPM", "cost", "members",
    ]
    
    column_questions = [
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
        "Top chief complaints", "Top 5 longest stays", "Disposition by visit type",
        "Diagnoses by severity", "Diagnoses by type",
        "Chronic vs non-chronic diagnoses", "Diagnoses by HCC category",
        "Critical diagnoses count", "Severe diagnoses count",
        "Cost by diagnosis severity", "Chronic diagnoses by severity",
        "Top 5 most common diagnoses", "Top 10 chronic conditions", "Unresolved diagnoses",
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
        "How many cardiologists?", "How many surgeons?", "Active vs inactive providers",
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
        "Referral completion rate by specialty", "Referral completion rate by urgency",
        "Pending referrals by specialty", "Denied referrals", "STAT referrals by specialty",
        "CPT codes by category", "Highest RVU procedures",
        "Average RVU by category", "Surgery procedures",
        "Total claims for critical severity", "Average claim cost by diagnosis severity",
        "Claim costs by gender", "Claim costs by race",
        "Top 10 providers by total cost", "Cost per provider by specialty",
        "Top 5 emergency diagnoses", "Emergency cost by region",
        "Top 5 ER claims by cost", "Denial rate by provider type",
    ]
    
    seen = set()
    for q in stress_questions + column_questions:
        ql = q.lower().strip()
        if ql not in seen:
            seen.add(ql)
            all_questions.append(q)
    
    print(f"\n{'='*80}")
    print(f"HONESTY AUDIT — {len(all_questions)} unique questions")
    print(f"{'='*80}\n")
    
    learner = SchemaLearner(db_path)
    learner.learn(sample_size=1000)
    deep = DeepUnderstandingEngine(schema_learner=learner, db_path=db_path,
                                    data_dir=os.path.dirname(db_path))
    shared_embedder = deep._embedder if hasattr(deep, '_embedder') else None
    sql_eng = SQLReasoningEngine(schema_learner=learner, db_path=db_path, embedder=shared_embedder)
    
    conn = sqlite3.connect(db_path)
    
    fallback_queries = []
    low_confidence = []
    wrong_table = []
    generic_sql = []
    errors = []
    good_queries = []
    
    for q in all_questions:
        try:
            u = deep.understand(q)
            ud = u.to_dict()
            sr = sql_eng.generate(understanding=ud, question=q)
            sql_text = sr.sql
            confidence = sr.confidence
            approach = getattr(sr, 'approach', '') or ''
            
            problems = []
            
            if is_executive_fallback(sql_text, approach):
                problems.append("EXECUTIVE_FALLBACK")
                fallback_queries.append((q, confidence, approach, sql_text[:200]))
            
            if confidence < 0.40:
                problems.append(f"LOW_CONF({confidence:.2f})")
                low_confidence.append((q, confidence, approach, sql_text[:200]))
            
            expected = question_expects_table(q)
            if expected != 'unknown' and not sql_uses_table(sql_text, expected):
                problems.append(f"WRONG_TABLE(expected={expected})")
                wrong_table.append((q, expected, sql_text[:200]))
            
            if 'SELECT *' in sql_text or 'appointments.*' in sql_text:
                problems.append("GENERIC_SQL")
                generic_sql.append((q, sql_text[:200]))
            
            try:
                cursor = conn.execute(sql_text)
                rows = cursor.fetchall()
                row_count = len(rows)
            except Exception as e:
                problems.append(f"SQL_ERROR({str(e)[:50]})")
                errors.append((q, sql_text[:200], str(e)[:80]))
                row_count = -1
            
            if not problems:
                good_queries.append((q, confidence, approach))
            else:
                print(f"  [PROBLEM] {q}")
                for p in problems:
                    print(f"            {p}")
                print(f"            conf={confidence:.2f} approach={approach}")
                print(f"            SQL: {sql_text[:150]}")
                print()
                
        except Exception as e:
            print(f"  [CRASH] {q}: {str(e)[:80]}")
            errors.append((q, '', str(e)[:80]))
    
    conn.close()
    
    total = len(all_questions)
    print(f"\n{'='*80}")
    print(f"HONESTY AUDIT RESULTS")
    print(f"{'='*80}")
    print(f"Total questions:      {total}")
    print(f"Truly good:           {len(good_queries)} ({len(good_queries)*100/total:.1f}%)")
    print(f"Executive fallback:   {len(fallback_queries)}")
    print(f"Low confidence:       {len(low_confidence)}")
    print(f"Wrong table:          {len(wrong_table)}")
    print(f"Generic SQL:          {len(generic_sql)}")
    print(f"SQL errors:           {len(errors)}")
    print()
    
    if fallback_queries:
        print(f"\n--- EXECUTIVE FALLBACK QUERIES ({len(fallback_queries)}) ---")
        for q, conf, approach, sql in fallback_queries:
            print(f"  [{conf:.2f}] {q}")
            print(f"         approach={approach}")
            print(f"         SQL: {sql[:120]}")
            print()
    
    if wrong_table:
        print(f"\n--- WRONG TABLE QUERIES ({len(wrong_table)}) ---")
        for q, expected, sql in wrong_table:
            print(f"  {q}")
            print(f"    expected={expected}, SQL: {sql[:120]}")
            print()
    
    if low_confidence:
        print(f"\n--- LOW CONFIDENCE QUERIES ({len(low_confidence)}) ---")
        for q, conf, approach, sql in low_confidence:
            print(f"  [{conf:.2f}] {q}")
            print(f"         SQL: {sql[:120]}")
            print()
    
    if errors:
        print(f"\n--- SQL ERRORS ({len(errors)}) ---")
        for q, sql, err in errors:
            print(f"  {q}")
            print(f"    ERROR: {err}")
            print()
    
    return {
        'total': total,
        'good': len(good_queries),
        'fallback': len(fallback_queries),
        'low_conf': len(low_confidence),
        'wrong_table': len(wrong_table),
        'generic': len(generic_sql),
        'errors': len(errors),
        'fallback_details': fallback_queries,
        'wrong_table_details': wrong_table,
        'low_conf_details': low_confidence,
        'error_details': errors,
    }


if __name__ == '__main__':
    db_path = '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_production.db'
    results = run_audit(db_path)
    
    problem_count = results['fallback'] + results['wrong_table'] + results['generic'] + results['errors']
    print(f"\n{'='*80}")
    print(f"TOTAL PROBLEMS: {problem_count} out of {results['total']} queries")
    print(f"TRUE ACCURACY: {results['good']}/{results['total']} ({results['good']*100/results['total']:.1f}%)")
    print(f"{'='*80}")
