import sys
import os
import time
import logging
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.WARNING)

from intelligent_pipeline import IntelligentPipeline


NOVEL_TESTS = [
    ("How much does a typical member pay out of pocket on each visit?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'COPAY']}),

    ("What is the mean reimbursement per claim in our system?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'PAID_AMOUNT']}),

    ("Can you tell me the combined deductible burden across all members?",
     "aggregate_novel",
     {'sql_contains': ['SUM', 'DEDUCTIBLE']}),

    ("What does the average patient coinsurance look like?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'COINSURANCE']}),

    ("Tell me the peak billed charge in the entire dataset",
     "aggregate_novel",
     {'sql_contains': ['MAX', 'BILLED_AMOUNT']}),

    ("What is our cumulative spending across all claims?",
     "aggregate_novel",
     {'sql_contains': ['SUM', 'PAID_AMOUNT']}),

    ("Average length of stay for our patient admissions",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'LENGTH_OF_STAY']}),

    ("Whats the floor for copay amounts we have seen?",
     "aggregate_novel",
     {'sql_contains': ['MIN', 'COPAY']}),

    ("Tell me the grand total of all charges submitted to us",
     "aggregate_novel",
     {'sql_contains': ['SUM', 'BILLED_AMOUNT']}),

    ("On average how much are we reimbursing per encounter?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'PAID_AMOUNT']}),

    ("What is the expected copayment for a standard visit?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'COPAY']}),

    ("How much do we typically pay out per claim?",
     "aggregate_novel",
     {'sql_contains': ['AVG', 'PAID_AMOUNT']}),

    ("What is our average patient acuity score?",
     "column_resolution",
     {'sql_contains': ['AVG', 'RISK_SCORE']}),

    ("How big are our typical physician panels?",
     "column_resolution",
     {'sql_contains': ['AVG', 'PANEL_SIZE']}),

    ("Show me the mean inpatient stay duration",
     "column_resolution",
     {'sql_contains': ['AVG', 'LENGTH_OF_STAY']}),

    ("What does our patient complexity distribution look like?",
     "column_resolution",
     {'sql_contains': ['RISK_SCORE']}),

    ("What is the average contracted rate per claim?",
     "column_resolution",
     {'sql_contains': ['AVG', 'ALLOWED_AMOUNT']}),

    ("How much is the average patient responsibility per visit?",
     "column_resolution",
     {'sql_contains': ['AVG']}),

    ("What is our total physician headcount?",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'providers']}),

    ("How many enrollees do we have in our system?",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'members']}),

    ("Give me the total number of billing submissions",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'claims']}),

    ("How many scripts were written last year?",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'prescriptions']}),

    ("What is our covered lives population?",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'members']}),

    ("Total number of patient admissions",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'encounters']}),

    ("How many clinicians are in our network?",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'providers']}),

    ("Give me the number of active authorizations",
     "entity_resolution",
     {'sql_contains': ['COUNT', 'referrals']}),

    ("What conditions are the most expensive to treat?",
     "rank_novel",
     {'sql_contains': ['ORDER BY'], 'min_rows': 2}),

    ("Which docs have the biggest caseload?",
     "rank_novel",
     {'sql_contains': ['ORDER BY', 'providers']}),

    ("Show me the costliest specialties in our network",
     "rank_novel",
     {'sql_contains': ['ORDER BY']}),

    ("What are the primary reasons claims get rejected?",
     "rank_novel",
     {'sql_contains': ['DENIED', 'ORDER BY']}),

    ("Which areas are generating the most healthcare utilization?",
     "rank_novel",
     {'sql_contains': ['ORDER BY']}),

    ("Most commonly diagnosed conditions across our members",
     "rank_novel",
     {'sql_contains': ['ORDER BY', 'diagnos']}),

    ("Which medications are prescribed most frequently?",
     "rank_novel",
     {'sql_contains': ['ORDER BY', 'prescriptions']}),

    ("Busiest providers by total claim volume",
     "rank_novel",
     {'sql_contains': ['ORDER BY']}),

    ("What visit types generate the most revenue?",
     "rank_novel",
     {'sql_contains': ['ORDER BY']}),

    ("Top facility locations by total paid amount",
     "rank_novel",
     {'sql_contains': ['ORDER BY', 'PAID_AMOUNT']}),

    ("How often do claims get denied in our system?",
     "rate_novel",
     {'sql_contains': ['DENIED']}),

    ("What share of our submissions are getting approved?",
     "rate_novel",
     {'sql_contains': ['CLAIM_STATUS']}),

    ("How frequently are submissions getting rejected by the plan?",
     "rate_novel",
     {'sql_contains': ['DENIED']}),

    ("What is our clean claim ratio?",
     "rate_novel",
     {'sql_contains': ['CLAIM_STATUS']}),

    ("Claim denial frequency across our network",
     "rate_novel",
     {'sql_contains': ['DENIED']}),

    ("Break down costs by geographic region",
     "group_by_novel",
     {'sql_contains': ['GROUP BY'], 'min_rows': 2}),

    ("Show me claims volume by plan type",
     "group_by_novel",
     {'sql_contains': ['GROUP BY', 'PLAN_TYPE'], 'min_rows': 2}),

    ("Average copay stratified by visit type",
     "group_by_novel",
     {'sql_contains': ['AVG', 'COPAY', 'GROUP BY']}),

    ("Member distribution across different age groups",
     "group_by_novel",
     {'sql_contains': ['GROUP BY', 'AGE_GROUP']}),

    ("Spending patterns broken out by specialty",
     "group_by_novel",
     {'sql_contains': ['GROUP BY']}),

    ("How does length of stay vary by department?",
     "group_by_novel",
     {'sql_contains': ['LENGTH_OF_STAY', 'GROUP BY']}),

    ("Claim count per insurance plan",
     "group_by_novel",
     {'sql_contains': ['COUNT', 'GROUP BY']}),

    ("Average risk score segmented by region",
     "group_by_novel",
     {'sql_contains': ['AVG', 'RISK_SCORE', 'GROUP BY']}),

    ("What is the average claim cost for diabetic patients?",
     "multi_table",
     {'sql_contains': ['AVG', 'claims', 'diagnos']}),

    ("Show me provider specialties with their average reimbursement",
     "multi_table",
     {'sql_contains': ['AVG', 'providers']}),

    ("Which regions have the most denied claims?",
     "multi_table",
     {'sql_contains': ['DENIED', 'ORDER BY']}),

    ("Average paid amount by provider specialty",
     "multi_table",
     {'sql_contains': ['AVG', 'PAID_AMOUNT', 'GROUP BY']}),

    ("Cost comparison between inpatient and outpatient encounters",
     "multi_table",
     {'sql_contains': ['encounters']}),

    ("How is our member enrollment trending quarter over quarter?",
     "trend_novel",
     {'sql_contains': ['members']}),

    ("Show me the monthly claims trajectory",
     "trend_novel",
     {'sql_contains': ['claims']}),

    ("Year over year spending analysis",
     "trend_novel",
     {'sql_contains': ['claims']}),

    ("How has our utilization changed over the past year?",
     "trend_novel",
     {'sql_contains': ['claims']}),

    ("What is our PMPM cost?",
     "jargon",
     {'sql_contains': ['members']}),

    ("Show me the loss ratio across our book of business",
     "jargon",
     {'sql_contains': ['claims']}),

    ("What does our RVU distribution look like?",
     "jargon",
     {'sql_contains': ['RVU']}),

    ("Average HCC score across our enrolled population",
     "jargon",
     {'sql_contains': ['AVG', 'RISK_SCORE']}),

    ("What is our average LOS for inpatient stays?",
     "jargon",
     {'sql_contains': ['AVG', 'LENGTH_OF_STAY']}),

    ("yo how many people do we cover",
     "casual",
     {'sql_contains': ['COUNT', 'members']}),

    ("whats the avg copay looking like",
     "casual",
     {'sql_contains': ['AVG', 'COPAY']}),

    ("give me a rundown of claims by region",
     "casual",
     {'sql_contains': ['GROUP BY']}),

    ("who are our most expensive patients",
     "casual",
     {'sql_contains': ['ORDER BY']}),

    ("how bad is our denial problem",
     "casual",
     {'sql_contains': ['DENIED']}),

    ("need the total claims count ASAP",
     "casual",
     {'sql_contains': ['COUNT', 'claims']}),

    ("quick question - avg paid per claim?",
     "casual",
     {'sql_contains': ['AVG', 'PAID_AMOUNT']}),

    ("so whats the deal with our risk scores",
     "casual",
     {'sql_contains': ['RISK_SCORE']}),

    ("pull me the top 5 diagnoses real quick",
     "casual",
     {'sql_contains': ['diagnos', 'ORDER BY']}),

    ("hmm what about average billed charges",
     "casual",
     {'sql_contains': ['AVG', 'BILLED_AMOUNT']}),

    ("What is the average cost per member stratified by region?",
     "complex",
     {'sql_contains': ['AVG', 'GROUP BY']}),

    ("Which plan types have the highest denial rates?",
     "complex",
     {'sql_contains': ['DENIED', 'PLAN_TYPE']}),

    ("Show average copay and paid amount by visit type",
     "complex",
     {'sql_contains': ['AVG', 'COPAY', 'GROUP BY']}),

    ("Top providers by number of unique patients",
     "complex",
     {'sql_contains': ['COUNT', 'ORDER BY']}),

    ("Average length of stay by diagnosis category",
     "complex",
     {'sql_contains': ['AVG', 'LENGTH_OF_STAY', 'GROUP BY']}),

    ("total claims",
     "edge_case",
     {'sql_contains': ['claims']}),

    ("providers",
     "edge_case",
     {'sql_contains': ['providers']}),

    ("average everything",
     "edge_case",
     {}),

    ("cost",
     "edge_case",
     {}),

    ("What is the average copay? Also total paid amount?",
     "edge_case",
     {'sql_contains': ['COPAY']}),

    ("show me all the data",
     "edge_case",
     {}),

    ("Tally up all the charges billed to us this year",
     "additional",
     {'sql_contains': ['SUM', 'BILLED_AMOUNT']}),

    ("Whats the going rate for reimbursement per encounter?",
     "additional",
     {'sql_contains': ['AVG', 'PAID_AMOUNT']}),

    ("Break down our drug spend by therapeutic class",
     "additional",
     {'sql_contains': ['prescriptions']}),

    ("How many distinct NPIs are in our provider file?",
     "additional",
     {'sql_contains': ['COUNT', 'DISTINCT', 'NPI']}),

    ("Average member out of pocket by plan type",
     "additional",
     {'sql_contains': ['AVG', 'GROUP BY']}),

    ("Sum total of all coinsurance collected",
     "additional",
     {'sql_contains': ['SUM', 'COINSURANCE']}),

    ("Which departments have the longest average stays?",
     "additional",
     {'sql_contains': ['AVG', 'LENGTH_OF_STAY', 'ORDER BY']}),

    ("Provider count by specialty area",
     "additional",
     {'sql_contains': ['COUNT', 'GROUP BY', 'SPECIALTY']}),

    ("Minimum copay we have charged anyone",
     "additional",
     {'sql_contains': ['MIN', 'COPAY']}),

    ("Maximum allowed amount across all claims",
     "additional",
     {'sql_contains': ['MAX', 'ALLOWED_AMOUNT']}),

    ("How many emergency visits did we have?",
     "additional",
     {'sql_contains': ['COUNT', 'encounters']}),

    ("Average risk score for our senior population",
     "additional",
     {'sql_contains': ['AVG', 'RISK_SCORE']}),

    ("What is the median claim cost?",
     "additional",
     {'sql_contains': ['claims']}),

    ("Number of unique medication classes prescribed",
     "additional",
     {'sql_contains': ['COUNT', 'DISTINCT', 'MEDICATION_CLASS']}),

    ("Total prescriptions written by our network",
     "additional",
     {'sql_contains': ['COUNT', 'prescriptions']}),

    ("What percentage of our members have high risk scores?",
     "additional",
     {'sql_contains': ['RISK_SCORE', 'members']}),

    ("Average deductible amount per member",
     "additional",
     {'sql_contains': ['AVG', 'DEDUCTIBLE']}),

    ("How does our paid amount compare to billed amount overall?",
     "additional",
     {'sql_contains': ['PAID_AMOUNT', 'BILLED_AMOUNT']}),

    ("Show me provider panel sizes ranked from largest to smallest",
     "additional",
     {'sql_contains': ['PANEL_SIZE', 'ORDER BY']}),

    ("Total member count broken down by race",
     "additional",
     {'sql_contains': ['COUNT', 'RACE', 'GROUP BY']}),
]


def run_tests():
    db_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'healthcare_demo.db')
    if not os.path.exists(db_path):
        db_path = 'data/healthcare_demo.db'

    print("=" * 80)
    print(f"NOVEL QUERY TEST SUITE — {len(NOVEL_TESTS)} tests")
    print("=" * 80)
    print()

    pipeline = IntelligentPipeline(db_path)

    passed = 0
    failed = 0
    errors = []
    category_stats = {}

    for i, (question, category, rules) in enumerate(NOVEL_TESTS, 1):
        if category not in category_stats:
            category_stats[category] = {'pass': 0, 'fail': 0}

        try:
            result = pipeline.process(question)
            sql = result.get('sql', '') or ''
            rows = result.get('rows', []) or []
            row_count = len(rows)
            sql_upper = sql.upper()

            fail_reasons = []

            min_rows = rules.get('min_rows', 1)
            if row_count < min_rows:
                fail_reasons.append(f"rows={row_count} < min={min_rows}")

            for required in rules.get('sql_contains', []):
                if required.upper() not in sql_upper:
                    fail_reasons.append(f"SQL missing '{required}'")

            for forbidden in rules.get('sql_not_contains', []):
                if forbidden.upper() in sql_upper:
                    fail_reasons.append(f"SQL has forbidden '{forbidden}'")

            if fail_reasons:
                status = 'FAIL'
                failed += 1
                category_stats[category]['fail'] += 1
                errors.append((i, question, category, fail_reasons, sql))
            else:
                status = 'PASS'
                passed += 1
                category_stats[category]['pass'] += 1

        except Exception as e:
            status = 'ERROR'
            failed += 1
            category_stats[category]['fail'] += 1
            errors.append((i, question, category, [f"EXCEPTION: {e}"], ''))

        q_display = question[:60].ljust(60)
        cat_display = category[:25].ljust(25)
        print(f"  {status:5s}: #{i:3d} [{cat_display}] \"{q_display}\"")

    print()
    print("=" * 80)
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        total = s['pass'] + s['fail']
        pct = (s['pass'] / total * 100) if total > 0 else 0
        mark = '✓' if s['fail'] == 0 else '✗'
        print(f"  {mark} {cat:30s} {s['pass']:3d}/{total:3d} ({pct:.1f}%)")

    print()
    print("=" * 80)
    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"NOVEL QUERY TEST SUITE: {passed}/{total} ({pct:.1f}%)")
    print("=" * 80)

    if errors:
        print()
        print("FAILURES:")
        print("-" * 80)
        for i, q, cat, reasons, sql in errors:
            print(f"  #{i} [{cat}] \"{q[:70]}\"")
            for r in reasons:
                print(f"       → {r}")
            if sql:
                print(f"       SQL: {sql[:120]}")
            print()

    return passed, failed, errors


if __name__ == '__main__':
    passed, failed, errors = run_tests()
    sys.exit(0 if failed == 0 else 1)
