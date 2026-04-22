import os
import sys
import time
import sqlite3
import logging
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.WARNING)

DB_PATH = '../data/healthcare_production.db'

TEST_QUERIES = [
    ("top 10 providers by volume", "ranking"),
    ("specialty with highest average cost", "ranking"),
    ("physicians in the Maryland Baltimore area", "filter"),
    ("claims percentage by age group", "breakdown"),
    ("claims percentage by gender", "breakdown"),
    ("claims percentage by race", "breakdown"),
    ("claims percentage by region", "breakdown"),
    ("count where one person visited emergency more than 10 times", "filter"),
    ("members who had more than 5 admissions", "filter"),
    ("members with both emergency and inpatient visits", "count"),
    ("emergency to outpatient ratio", "aggregate"),
    ("compare inpatient vs outpatient costs", "comparison"),
    ("paid denied adjusted breakdown", "breakdown"),
    ("claims with prior authorization as percentage of total", "percentage"),
    ("what percentage of claims are pending", "percentage"),
    ("denied claim percentage", "percentage"),
    ("high dollar claims over 10000", "filter"),
    ("average paid amount per visit type", "aggregate"),
    ("average processing time for claims", "aggregate"),
    ("diabetes claims by region", "breakdown"),
    ("COVID hospitalization by region", "breakdown"),
    ("readmission rate", "rate"),
    ("medicaid claims", "filter"),
    ("pharmacy claims and billing", "lookup"),
    ("CPT code descriptions", "lookup"),
    ("most occurring CPT code in 2025", "ranking"),
    ("upcoming PCP appointments for KP Baltimore", "filter"),
    ("members older than 65", "filter"),
    ("total members by plan type", "breakdown"),
    ("top 5 facilities by claim volume", "ranking"),
    ("denial rate by region", "rate"),
]

def run_tests():
    from semantic_sql_engine import SemanticSQLEngine

    print("=" * 80)
    print("DIRECT SQL ENGINE TEST — ALL QUERIES")
    print("=" * 80)

    t0 = time.time()
    engine = SemanticSQLEngine(DB_PATH)
    init_time = time.time() - t0
    print(f"\nEngine initialized in {init_time:.1f}s\n")

    conn = sqlite3.connect(DB_PATH)
    passes = 0
    fails = 0
    results = []

    for i, (query, expected_intent) in enumerate(TEST_QUERIES, 1):
        try:
            t1 = time.time()
            result = engine.generate(query)
            gen_time = time.time() - t1

            sql = result.get('sql', '')
            intent = result.get('intent', result.get('semantic_intent', ''))

            error = None
            rows = []
            columns = []
            try:
                cursor = conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
            except Exception as e:
                error = str(e)

            row_count = len(rows)

            is_pass = True
            fail_reasons = []

            if error:
                is_pass = False
                fail_reasons.append(f"SQL_ERROR: {error[:80]}")

            if not sql:
                is_pass = False
                fail_reasons.append("NO_SQL")

            if row_count > 5000 and expected_intent in ('ranking', 'breakdown', 'aggregate', 'percentage', 'rate', 'comparison'):
                is_pass = False
                fail_reasons.append(f"TOO_MANY_ROWS({row_count})")

            if 'more than' in query.lower() and ('member' in query.lower() or 'person' in query.lower()):
                if 'HAVING' not in sql.upper() and 'GROUP BY' not in sql.upper():
                    is_pass = False
                    fail_reasons.append("MISSING_HAVING")

            if 'ratio' in query.lower() and ('CASE' not in sql.upper() and '/' not in sql):
                is_pass = False
                fail_reasons.append("MISSING_RATIO_LOGIC")

            if is_pass:
                passes += 1
                icon = "✓"
            else:
                fails += 1
                icon = "✗"

            print(f"  {icon} [{i:2d}/{len(TEST_QUERIES)}] {query}")
            print(f"       Intent: {intent} | Rows: {row_count} | {gen_time*1000:.0f}ms")
            if fail_reasons:
                print(f"       ISSUES: {', '.join(fail_reasons)}")
            print(f"       SQL: {sql[:130]}")
            if rows and row_count <= 5:
                for r in rows:
                    print(f"       DATA: {r}")
            print()

            results.append({'query': query, 'pass': is_pass, 'reasons': fail_reasons, 'rows': row_count, 'sql': sql})

        except Exception as e:
            fails += 1
            print(f"  ✗ [{i:2d}/{len(TEST_QUERIES)}] {query}")
            print(f"       EXCEPTION: {e}")
            print()
            results.append({'query': query, 'pass': False, 'reasons': [f"EXCEPTION: {e}"], 'rows': 0, 'sql': ''})

    conn.close()

    print("=" * 80)
    print(f"RESULTS: {passes} PASS / {fails} FAIL / {len(TEST_QUERIES)} TOTAL")
    print(f"Pass rate: {100*passes/len(TEST_QUERIES):.0f}%")
    print("=" * 80)

    if fails > 0:
        print("\nFAILED QUERIES:")
        for r in results:
            if not r['pass']:
                print(f"  ✗ {r['query']}: {', '.join(r['reasons'])}")
                print(f"    SQL: {r['sql'][:150]}")
                print()

    return passes, fails

if __name__ == '__main__':
    run_tests()
