import os
import sys
import sqlite3
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

from query_intelligence import QueryIntelligence

TESTS = [
    {
        'q': 'How many members do we have?',
        'expect_in_sql': ['COUNT', 'MEMBER'],
        'expect_type': 'count',
        'category': 'count',
    },
    {
        'q': 'How many claims were denied?',
        'expect_in_sql': ['COUNT', 'DENIED'],
        'expect_type': 'count',
        'category': 'count',
    },
    {
        'q': 'How many distinct NPIs are in our provider file?',
        'expect_in_sql': ['COUNT', 'DISTINCT', 'NPI'],
        'expect_type': 'count',
        'category': 'count',
    },
    {
        'q': 'How many prescriptions were filled?',
        'expect_in_sql': ['COUNT', 'PRESCRIPTION'],
        'expect_type': 'count',
        'category': 'count',
    },
    {
        'q': 'What is the average paid amount?',
        'expect_in_sql': ['AVG', 'PAID_AMOUNT'],
        'expect_type': 'scalar',
        'category': 'scalar',
    },
    {
        'q': 'What is the total billed amount?',
        'expect_in_sql': ['SUM', 'BILLED_AMOUNT'],
        'expect_type': 'scalar',
        'category': 'scalar',
    },
    {
        'q': 'What is the average risk score?',
        'expect_in_sql': ['AVG', 'RISK_SCORE'],
        'expect_type': 'scalar',
        'category': 'scalar',
    },
    {
        'q': 'What is the average length of stay?',
        'expect_in_sql': ['AVG', 'LENGTH_OF_STAY'],
        'expect_type': 'scalar',
        'category': 'scalar',
    },
    {
        'q': 'What is the average days supply?',
        'expect_in_sql': ['AVG', 'DAYS_SUPPLY'],
        'expect_type': 'scalar',
        'category': 'scalar',
    },
    {
        'q': 'Average paid amount by plan type',
        'expect_in_sql': ['AVG', 'PAID_AMOUNT', 'PLAN_TYPE', 'GROUP BY'],
        'expect_type': 'grouped',
        'category': 'grouped',
    },
    {
        'q': 'Total cost by region',
        'expect_in_sql': ['SUM', 'KP_REGION', 'GROUP BY'],
        'expect_type': 'grouped',
        'category': 'grouped',
    },
    {
        'q': 'Average risk score by gender',
        'expect_in_sql': ['AVG', 'RISK_SCORE', 'GENDER', 'GROUP BY'],
        'expect_type': 'grouped',
        'category': 'grouped',
    },
    {
        'q': 'What is the most expensive medication?',
        'expect_in_sql': ['MEDICATION_NAME', 'COST', 'ORDER BY', 'DESC'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'Top 10 diagnoses by claim count',
        'expect_in_sql': ['ICD10', 'COUNT', 'ORDER BY', 'DESC', 'LIMIT 10'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'Which providers have the largest panels?',
        'expect_in_sql': ['NPI', 'PANEL_SIZE', 'ORDER BY', 'DESC'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'Most common specialties',
        'expect_in_sql': ['SPECIALTY', 'COUNT', 'ORDER BY', 'DESC'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'Which facility has the most encounters?',
        'expect_in_sql': ['FACILITY', 'COUNT', 'ORDER BY', 'DESC'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'Most prescribed medications',
        'expect_in_sql': ['MEDICATION_NAME', 'COUNT', 'ORDER BY', 'DESC'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'What is the denial rate?',
        'expect_in_sql': ['DENIED', 'CASE WHEN', 'denial_rate'],
        'expect_type': 'rate',
        'category': 'rate',
    },
    {
        'q': 'What is the no-show rate?',
        'expect_in_sql': ['No-Show', 'CASE WHEN'],
        'expect_type': 'rate',
        'category': 'rate',
    },
    {
        'q': 'How many claims are pending?',
        'expect_in_sql': ['COUNT', 'PENDING'],
        'expect_type': 'count',
        'category': 'filter',
    },
    {
        'q': 'Average cost for diabetic patients',
        'expect_in_sql': ['AVG', 'diabet'],
        'expect_type': 'scalar',
        'category': 'filter',
    },
    {
        'q': 'What is the total member count?',
        'expect_in_sql': ['COUNT'],
        'reject_in_sql': ['SUM'],
        'expect_type': 'count',
        'category': 'disambiguation',
    },
    {
        'q': 'How much did we spend on prescriptions?',
        'expect_in_sql': ['SUM', 'COST'],
        'expect_type': 'scalar',
        'category': 'disambiguation',
    },
    {
        'q': 'What is the average copay?',
        'expect_in_sql': ['AVG', 'COPAY'],
        'expect_type': 'scalar',
        'category': 'edge',
    },
    {
        'q': 'Total prescriptions written',
        'expect_in_sql': ['COUNT', 'PRESCRIPTION'],
        'reject_in_sql': ['SUM'],
        'expect_type': 'count',
        'category': 'edge',
    },
    {
        'q': 'Denial rate by plan type',
        'expect_in_sql': ['PLAN_TYPE', 'DENIED', 'CASE WHEN', 'GROUP BY'],
        'expect_type': 'rate',
        'category': 'rate_grouped',
    },
    {
        'q': 'Average paid amount by specialty',
        'expect_in_sql': ['AVG', 'PAID_AMOUNT', 'SPECIALTY', 'GROUP BY'],
        'expect_type': 'grouped',
        'category': 'grouped',
    },
    {
        'q': 'Which department has the highest average length of stay?',
        'expect_in_sql': ['DEPARTMENT', 'AVG', 'LENGTH_OF_STAY', 'ORDER BY'],
        'expect_type': 'ranked',
        'category': 'ranked',
    },
    {
        'q': 'How many unique members have prescriptions?',
        'expect_in_sql': ['COUNT', 'DISTINCT', 'MEMBER_ID'],
        'expect_type': 'count',
        'category': 'count',
    },
]


def run_tests():
    print(f"{'='*80}")
    print(f"QueryIntelligence Standalone Test — {len(TESTS)} questions, NO golden templates")
    print(f"{'='*80}\n")

    qi = QueryIntelligence(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    passed = 0
    failed = 0
    failures = []

    for i, test in enumerate(TESTS, 1):
        q = test['q']
        result = qi.generate_sql(q)
        sql = result.get('sql', '')
        sql_upper = sql.upper()
        intent = result.get('intent')
        valid = result.get('valid', False)

        missing = []
        for pattern in test.get('expect_in_sql', []):
            if pattern.upper() not in sql_upper:
                missing.append(pattern)

        rejected = []
        for pattern in test.get('reject_in_sql', []):
            if pattern.upper() in sql_upper:
                rejected.append(pattern)

        type_match = True
        if test.get('expect_type') and intent:
            type_match = intent.query_type == test['expect_type']

        exec_ok = False
        row_count = 0
        if valid and sql:
            try:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                row_count = len(rows)
                exec_ok = row_count > 0
            except Exception as e:
                exec_ok = False

        ok = not missing and not rejected and type_match and valid and exec_ok
        status = '✓' if ok else '✗'
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((i, q, missing, rejected, type_match, valid, exec_ok, sql, intent))

        print(f"  {status} #{i:2d} [{test.get('category', '?'):15s}] {q}")
        if not ok:
            if missing:
                print(f"       MISSING: {missing}")
            if rejected:
                print(f"       REJECTED: {rejected}")
            if not type_match and intent:
                print(f"       TYPE: expected={test.get('expect_type')}, got={intent.query_type}")
            if not valid:
                print(f"       INVALID SQL: {result.get('issues')}")
            if not exec_ok:
                print(f"       NO RESULTS (rows={row_count})")
            print(f"       SQL: {sql[:200]}")

    conn.close()

    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{len(TESTS)} passed, {failed} failed")
    print(f"{'='*80}")

    if failures:
        print(f"\n--- FAILURES ---")
        for i, q, missing, rejected, type_match, valid, exec_ok, sql, intent in failures:
            print(f"\n  #{i}: {q}")
            print(f"    SQL: {sql[:300]}")
            if intent:
                print(f"    Intent: type={intent.query_type}, agg={intent.aggregation}, distinct={intent.is_distinct}")

    return passed, len(TESTS)


if __name__ == '__main__':
    p, t = run_tests()
    sys.exit(0 if p == t else 1)
