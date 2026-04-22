import os
import sys
import time
import sqlite3
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.WARNING)

DB_PATH = '../data/healthcare_production.db'

TEST_QUERIES = [
    ("top 10 providers by volume", "ranking", "Should GROUP BY provider, ORDER BY COUNT DESC, LIMIT 10"),
    ("specialty with highest average cost", "ranking", "GROUP BY specialty, ORDER BY AVG(cost) DESC"),
    ("physicians in the Maryland Baltimore area", "filter", "Filter by location/region"),

    ("claims percentage by age group", "breakdown", "Age bucketing with percentage"),
    ("claims percentage by gender", "breakdown", "GROUP BY gender with %"),
    ("claims percentage by race", "breakdown", "GROUP BY race with %"),
    ("claims percentage by region", "breakdown", "GROUP BY KP_REGION with %"),

    ("count where one person visited emergency more than 10 times", "filter", "GROUP BY member HAVING COUNT > 10"),
    ("members who had more than 5 admissions", "filter", "GROUP BY member HAVING COUNT > 5"),
    ("members with both emergency and inpatient visits", "filter", "Members in BOTH visit types"),

    ("emergency to outpatient ratio", "aggregate", "Ratio of two visit types"),
    ("compare inpatient vs outpatient costs", "comparison", "Side-by-side cost comparison"),
    ("paid denied adjusted breakdown", "breakdown", "Breakdown by claim status"),

    ("claims with prior authorization as percentage of total", "percentage", "Claims with auth / total"),
    ("what percentage of claims are pending", "percentage", "PENDING / total claims"),
    ("denied claim percentage", "percentage", "DENIED / total claims"),

    ("high dollar claims over 10000", "filter", "BILLED_AMOUNT > 10000"),
    ("average paid amount per visit type", "aggregate", "AVG(PAID_AMOUNT) GROUP BY VISIT_TYPE"),
    ("average processing time for claims", "aggregate", "DATE diff between submitted and adjudicated"),

    ("diabetes claims by region", "breakdown", "ICD10 filter + GROUP BY region"),
    ("COVID hospitalization by region", "breakdown", "COVID ICD10 + inpatient + GROUP BY region"),
    ("readmission rate", "rate", "Complex: discharge → re-admit within 30 days"),

    ("medicaid claims", "filter", "Filter PLAN_TYPE = Medicaid"),
    ("pharmacy claims and billing", "lookup", "Prescriptions table data"),
    ("CPT code descriptions", "lookup", "From cpt_codes table"),
    ("most occurring CPT code in 2025", "ranking", "GROUP BY CPT_CODE, year filter, ORDER BY count"),

    ("upcoming PCP appointments for KP Baltimore", "filter", "PCP + region filter + future dates"),

    ("members older than 65", "filter", "Age filter from DOB"),
    ("total members by plan type", "breakdown", "GROUP BY PLAN_TYPE"),

    ("top 5 facilities by claim volume", "ranking", "GROUP BY FACILITY LIMIT 5"),
    ("denial rate by region", "rate", "DENIED / total GROUP BY region"),
]

def run_tests():
    from intelligent_pipeline import IntelligentPipeline

    print("=" * 80)
    print("COMPREHENSIVE QUERY TEST — ALL 30+ USER QUERIES")
    print("=" * 80)

    t0 = time.time()
    pipeline = IntelligentPipeline(DB_PATH, neural_dim=32)
    init_time = time.time() - t0
    print(f"\nPipeline initialized in {init_time:.1f}s\n")

    results = []
    passes = 0
    fails = 0

    for i, (query, expected_intent, notes) in enumerate(TEST_QUERIES, 1):
        try:
            result = pipeline.process(query)
            sql = result.get('sql', '')
            rows = result.get('rows', [])
            row_count = result.get('row_count', 0)
            error = result.get('error', '')
            intent = result.get('intent', '')
            strategy = result.get('strategy', '')
            columns = result.get('columns', [])

            is_pass = True
            fail_reasons = []

            if error:
                is_pass = False
                fail_reasons.append(f"SQL_ERROR: {error}")

            if not sql:
                is_pass = False
                fail_reasons.append("NO_SQL_GENERATED")

            if row_count == 0 and not error:
                fail_reasons.append("ZERO_ROWS")

            if row_count > 5000 and expected_intent in ('ranking', 'breakdown', 'aggregate', 'percentage', 'rate', 'comparison'):
                is_pass = False
                fail_reasons.append(f"TOO_MANY_ROWS({row_count}): expected aggregation")

            if 'more than' in query.lower() and ('member' in query.lower() or 'person' in query.lower()):
                if 'HAVING' not in sql.upper() and 'GROUP BY' not in sql.upper():
                    is_pass = False
                    fail_reasons.append("MISSING_HAVING: needs GROUP BY + HAVING")

            if 'ratio' in query.lower() and ('CASE' not in sql.upper() and '/' not in sql):
                is_pass = False
                fail_reasons.append("MISSING_RATIO: needs CASE WHEN or division")

            if 'vs' in query.lower() or 'compare' in query.lower():
                if 'CASE' not in sql.upper() and row_count > 1000:
                    fail_reasons.append("WEAK_COMPARISON: may need CASE WHEN pivoting")

            if is_pass:
                passes += 1
                status = "PASS"
            else:
                fails += 1
                status = "FAIL"

            results.append({
                'query': query,
                'status': status,
                'sql': sql,
                'row_count': row_count,
                'intent': intent,
                'strategy': strategy,
                'columns': columns,
                'error': error,
                'fail_reasons': fail_reasons,
                'notes': notes,
                'expected_intent': expected_intent,
            })

            icon = "✓" if is_pass else "✗"
            print(f"  {icon} [{i:2d}/{len(TEST_QUERIES)}] {query}")
            print(f"       Intent: {intent} | Strategy: {strategy} | Rows: {row_count}")
            if fail_reasons:
                print(f"       Issues: {', '.join(fail_reasons)}")
            print(f"       SQL: {sql[:120]}...")
            print()

        except Exception as e:
            fails += 1
            results.append({
                'query': query,
                'status': 'ERROR',
                'sql': '',
                'row_count': 0,
                'intent': '',
                'strategy': '',
                'columns': [],
                'error': str(e),
                'fail_reasons': [f"EXCEPTION: {e}"],
                'notes': notes,
                'expected_intent': expected_intent,
            })
            print(f"  ✗ [{i:2d}/{len(TEST_QUERIES)}] {query}")
            print(f"       EXCEPTION: {e}")
            print()

    print("=" * 80)
    print(f"RESULTS: {passes} PASS / {fails} FAIL / {len(TEST_QUERIES)} TOTAL")
    print("=" * 80)

    failure_categories = {}
    for r in results:
        if r['status'] != 'PASS':
            for reason in r['fail_reasons']:
                cat = reason.split(':')[0]
                if cat not in failure_categories:
                    failure_categories[cat] = []
                failure_categories[cat].append(r['query'])

    if failure_categories:
        print("\nFAILURE CATEGORIES:")
        for cat, queries in sorted(failure_categories.items()):
            print(f"\n  {cat} ({len(queries)} queries):")
            for q in queries:
                print(f"    - {q}")

    print()
    return results, passes, fails

if __name__ == '__main__':
    results, passes, fails = run_tests()
