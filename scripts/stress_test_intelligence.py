import os, sys, re, time, sqlite3, json, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(os.path.join(SCRIPT_DIR, '..'))

from intelligent_pipeline import IntelligentPipeline

IntelligentPipeline._try_golden_template = lambda self, *a, **kw: None

DB_PATH = os.path.join('data', 'healthcare_production.db')

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def gt(sql):
    try:
        return cur.execute(sql).fetchall()
    except Exception as e:
        return f"ERROR: {e}"

GROUND_TRUTH = {
    'total_members': gt("SELECT COUNT(*) FROM members")[0][0],
    'total_claims': gt("SELECT COUNT(*) FROM claims")[0][0],
    'total_encounters': gt("SELECT COUNT(*) FROM encounters")[0][0],
    'denied_claims': gt("SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'")[0][0],
    'female_members': gt("SELECT COUNT(*) FROM members WHERE GENDER='F'")[0][0],
    'avg_risk': gt("SELECT ROUND(AVG(CAST(RISK_SCORE AS REAL)), 2) FROM members")[0][0],
    'hmo_claims': gt("SELECT COUNT(*) FROM claims WHERE PLAN_TYPE='HMO'")[0][0],
    'er_visits': gt("SELECT COUNT(*) FROM encounters WHERE VISIT_TYPE='EMERGENCY'")[0][0],
    'inpatient_visits': gt("SELECT COUNT(*) FROM encounters WHERE VISIT_TYPE='INPATIENT'")[0][0],
    'disenrolled': gt("SELECT COUNT(*) FROM members WHERE DISENROLLMENT_DATE != ''")[0][0],
    'active_members': gt("SELECT COUNT(*) FROM members WHERE DISENROLLMENT_DATE = ''")[0][0],
    'unique_providers': gt("SELECT COUNT(DISTINCT NPI) FROM providers")[0][0],
    'ncal_claims': gt("SELECT COUNT(*) FROM claims WHERE KP_REGION='NCAL'")[0][0],
    'telehealth_visits': gt("SELECT COUNT(*) FROM encounters WHERE VISIT_TYPE='TELEHEALTH'")[0][0],
    'total_billed': gt("SELECT ROUND(SUM(CAST(BILLED_AMOUNT AS REAL)), 2) FROM claims")[0][0],
    'total_paid': gt("SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)), 2) FROM claims")[0][0],
    'prescriptions_count': gt("SELECT COUNT(*) FROM prescriptions")[0][0],
    'referrals_count': gt("SELECT COUNT(*) FROM referrals")[0][0],
}
conn.close()


TESTS = [
    {
        'q': 'How many total members do we have?',
        'category': 'simple_count',
        'checks': {
            'sql_must_have': ['members'],
            'sql_must_not_have': ['SELECT *'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'Total number of claims in the system',
        'category': 'simple_count',
        'checks': {
            'sql_must_have': ['claims'],
            'sql_must_not_have': ['SELECT *'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'how many providers are there',
        'category': 'simple_count',
        'checks': {
            'sql_must_have': ['providers'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'Count of referrals',
        'category': 'simple_count',
        'checks': {
            'sql_must_have': ['referrals'],
            'row_count_range': (1, 10),
        },
    },

    {
        'q': 'How many denied claims do we have?',
        'category': 'filtered_count',
        'checks': {
            'sql_must_have': ['claims', 'DENIED'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'How many female members are enrolled?',
        'category': 'filtered_count',
        'checks': {
            'sql_must_have': ['members', 'GENDER'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'How many emergency visits happened?',
        'category': 'filtered_count',
        'checks': {
            'sql_must_have': ['encounters'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'How many claims are still pending?',
        'category': 'filtered_count',
        'checks': {
            'sql_must_have': ['claims', 'PENDING'],
            'row_count_range': (1, 5),
        },
    },

    {
        'q': 'What is the total amount billed across all claims?',
        'category': 'aggregation',
        'checks': {
            'sql_must_have': ['BILLED_AMOUNT'],
            'sql_must_not_have': ['SELECT *'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'What is the average risk score of our members?',
        'category': 'aggregation',
        'checks': {
            'sql_must_have': ['RISK_SCORE', 'AVG'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'What is the highest paid amount on any single claim?',
        'category': 'aggregation',
        'checks': {
            'sql_must_have': ['PAID_AMOUNT'],
            'sql_must_have_any': ['MAX', 'ORDER BY'],
            'row_count_range': (1, 5),
        },
    },
    {
        'q': 'Average length of stay for inpatient visits',
        'category': 'aggregation',
        'checks': {
            'sql_must_have': ['LENGTH_OF_STAY'],
            'row_count_range': (1, 10),
        },
    },

    {
        'q': 'Break down claims by plan type',
        'category': 'group_by',
        'checks': {
            'sql_must_have': ['PLAN_TYPE', 'GROUP BY'],
            'row_count_range': (3, 10),
        },
    },
    {
        'q': 'How many encounters by visit type?',
        'category': 'group_by',
        'checks': {
            'sql_must_have': ['VISIT_TYPE', 'GROUP BY'],
            'row_count_range': (4, 10),
        },
    },
    {
        'q': 'Show claims count per region',
        'category': 'group_by',
        'checks': {
            'sql_must_have': ['KP_REGION', 'GROUP BY'],
            'row_count_range': (5, 12),
        },
    },
    {
        'q': 'Number of prescriptions by medication class',
        'category': 'group_by',
        'checks': {
            'sql_must_have': ['MEDICATION_CLASS', 'GROUP BY'],
            'row_count_range': (3, 30),
        },
    },

    {
        'q': 'Which providers have the most claims?',
        'category': 'join',
        'checks': {
            'sql_must_have': ['claims'],
            'sql_must_have_any': ['providers', 'NPI', 'RENDERING_NPI', 'GROUP BY'],
            'row_count_range': (1, 50),
        },
    },
    {
        'q': 'List the top 10 members by total paid amount',
        'category': 'join',
        'checks': {
            'sql_must_have': ['MEMBER_ID', 'PAID_AMOUNT'],
            'row_count_range': (5, 15),
        },
    },
    {
        'q': 'Which department has the most encounters?',
        'category': 'join',
        'checks': {
            'sql_must_have': ['DEPARTMENT', 'encounters'],
            'row_count_range': (1, 30),
        },
    },

    {
        'q': 'What percentage of claims are denied?',
        'category': 'conditional',
        'checks': {
            'sql_must_have': ['DENIED'],
            'sql_must_not_have': ['SELECT *'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'What is the denial rate by region?',
        'category': 'conditional',
        'checks': {
            'sql_must_have': ['KP_REGION', 'DENIED'],
            'row_count_range': (3, 12),
        },
    },
    {
        'q': 'How many members have a risk score above 2?',
        'category': 'conditional',
        'checks': {
            'sql_must_have': ['RISK_SCORE'],
            'row_count_range': (1, 10),
        },
    },

    {
        'q': 'How many claims per month in 2024?',
        'category': 'temporal',
        'checks': {
            'sql_must_have': ['SERVICE_DATE'],
            'row_count_range': (1, 15),
        },
    },
    {
        'q': 'What is the trend of encounters over time?',
        'category': 'temporal',
        'checks': {
            'sql_must_have': ['encounters'],
            'row_count_range': (1, 48),
        },
    },

    {
        'q': 'What is the average cost per encounter for HMO members?',
        'category': 'complex',
        'checks': {
            'sql_must_have': ['HMO'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'Which region has the highest denial rate?',
        'category': 'complex',
        'checks': {
            'sql_must_have': ['KP_REGION', 'DENIED'],
            'row_count_range': (1, 12),
        },
    },
    {
        'q': 'Compare telehealth vs inpatient visit counts',
        'category': 'complex',
        'checks': {
            'sql_must_have': ['VISIT_TYPE'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'What is the average copay for each plan type?',
        'category': 'complex',
        'checks': {
            'sql_must_have': ['COPAY', 'PLAN_TYPE'],
            'row_count_range': (3, 10),
        },
    },

    {
        'q': 'Show me everything about NCAL',
        'category': 'ambiguous',
        'checks': {
            'sql_must_have': ['NCAL'],
            'row_count_range': (1, 100),
        },
    },
    {
        'q': 'Who are our sickest patients?',
        'category': 'ambiguous',
        'checks': {
            'sql_must_have': ['RISK_SCORE'],
            'row_count_range': (1, 100),
        },
    },
    {
        'q': 'Give me a summary of our claims',
        'category': 'ambiguous',
        'checks': {
            'sql_must_have': ['claims'],
            'row_count_range': (1, 20),
        },
    },

    {
        'q': 'how many memebers in ncal region',
        'category': 'typo',
        'checks': {
            'sql_must_have': ['members', 'NCAL'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'whats the avg paid amnt per claim',
        'category': 'typo',
        'checks': {
            'sql_must_have': ['PAID_AMOUNT'],
            'row_count_range': (1, 10),
        },
    },
    {
        'q': 'top specialites by clam count',
        'category': 'typo',
        'checks': {
            'sql_must_have': ['SPECIALTY'],
            'row_count_range': (1, 30),
        },
    },

    {
        'q': 'What is the readmission rate for our hospital?',
        'category': 'edge_case',
        'checks': {
            'sql_must_have': [],
            'row_count_range': (0, 100),
        },
    },
    {
        'q': 'How many members have diabetes?',
        'category': 'edge_case',
        'checks': {
            'sql_must_have': ['diagnoses'],
            'row_count_range': (0, 100),
        },
    },
    {
        'q': 'What is the no-show rate for appointments?',
        'category': 'edge_case',
        'checks': {
            'sql_must_have': ['appointments'],
            'row_count_range': (0, 100),
        },
    },
    {
        'q': 'Which medications are prescribed most often?',
        'category': 'edge_case',
        'checks': {
            'sql_must_have': ['prescriptions'],
            'row_count_range': (1, 50),
        },
    },

    {
        'q': 'What is our PMPM cost?',
        'category': 'business_intel',
        'checks': {
            'sql_must_have': ['PAID_AMOUNT', 'MEMBER_ID'],
            'row_count_range': (1, 20),
        },
    },
    {
        'q': 'Which facility handles the most volume?',
        'category': 'business_intel',
        'checks': {
            'sql_must_have': ['FACILITY'],
            'row_count_range': (1, 30),
        },
    },
    {
        'q': 'What are the top 5 most common diagnoses?',
        'category': 'business_intel',
        'checks': {
            'sql_must_have': ['diagnoses'],
            'row_count_range': (3, 10),
        },
    },
]


def grade_result(test_def, result):
    issues = []
    checks = test_def['checks']

    sql = (result.get('sql', '') or '').upper()
    rows = result.get('rows', [])
    narrative = (result.get('narrative', '') or '').lower()
    error = result.get('error', '')
    source = result.get('source', '')

    if not sql and not result.get('data_gap') and 'data_gap' not in source:
        issues.append('NO SQL GENERATED')

    for kw in checks.get('sql_must_have', []):
        if kw.upper() not in sql and kw.lower() not in narrative:
            issues.append(f'Missing: {kw}')

    any_list = checks.get('sql_must_have_any', [])
    if any_list and not any(kw.upper() in sql or kw.lower() in narrative for kw in any_list):
        issues.append(f'Missing any of: {any_list}')

    for bad in checks.get('sql_must_not_have', []):
        if bad.upper() in sql:
            issues.append(f'Bad pattern: {bad}')

    min_rows, max_rows = checks.get('row_count_range', (0, 1000))
    if len(rows) < min_rows:
        issues.append(f'Too few rows: {len(rows)} (expected {min_rows}+)')
    if len(rows) > max_rows:
        issues.append(f'Too many rows: {len(rows)} (expected <{max_rows})')

    if error:
        issues.append(f'Error: {str(error)[:80]}')

    if 'SELECT *' in sql and 'LIMIT 50' in sql:
        issues.append('INTELLIGENCE FAILURE: SELECT * LIMIT 50 dump')

    if not issues:
        return 'PASS', []
    elif len(issues) == 1 and issues[0].startswith('Missing:'):
        return 'PARTIAL', issues
    else:
        return 'FAIL', issues


def main():
    print("=" * 100)
    print("  COMPREHENSIVE STRESS TEST — 40 Never-Seen Questions (NO golden templates)")
    print("  Testing: Can the system handle ANY question type with accurate SQL?")
    print("=" * 100)
    print()

    print(f"  Ground Truth: {GROUND_TRUTH['total_members']} members, {GROUND_TRUTH['total_claims']} claims, "
          f"{GROUND_TRUTH['total_encounters']} encounters, {GROUND_TRUTH['denied_claims']} denied")
    print()

    pipeline = IntelligentPipeline(db_path=DB_PATH)

    pipeline.semantic_cache.cache.clear()
    try:
        c2 = sqlite3.connect(os.path.join('data', 'answer_cache.db'))
        c2.execute("DELETE FROM answer_cache")
        c2.commit()
        c2.close()
    except: pass

    results = {'PASS': 0, 'PARTIAL': 0, 'FAIL': 0, 'CRASH': 0}
    failures = []
    category_results = {}

    total = len(TESTS)
    for i, test in enumerate(TESTS, 1):
        q = test['q']
        cat = test['category']
        if cat not in category_results:
            category_results[cat] = {'PASS': 0, 'PARTIAL': 0, 'FAIL': 0, 'CRASH': 0}

        print(f"  [{i:2d}/{total}] {q}")

        try:
            t0 = time.time()
            result = pipeline.process(q, session_id=f'stress_{i}')
            elapsed = (time.time() - t0) * 1000
        except Exception as e:
            print(f"         CRASH: {e}")
            results['CRASH'] += 1
            category_results[cat]['CRASH'] += 1
            failures.append((i, q, 'CRASH', [str(e)[:100]]))
            print()
            continue

        grade, issues = grade_result(test, result)
        results[grade] += 1
        category_results[cat][grade] += 1

        source = result.get('source', '?')
        sql_preview = (result.get('sql', '') or '')[:100].replace('\n', ' ')
        row_count = len(result.get('rows', []))

        if grade == 'PASS':
            sym = 'PASS'
        elif grade == 'PARTIAL':
            sym = 'WARN'
        else:
            sym = 'FAIL'

        print(f"         [{sym}] Source: {source} | Rows: {row_count} | {elapsed:.0f}ms")
        if issues:
            for iss in issues:
                print(f"            -> {iss}")
            failures.append((i, q, grade, issues))
        if grade != 'PASS':
            print(f"            SQL: {sql_preview}")
        print()

    total_tests = sum(results.values())
    pass_rate = results['PASS'] / total_tests * 100 if total_tests else 0

    print("=" * 100)
    print(f"  RESULTS: {results['PASS']} PASS | {results['PARTIAL']} PARTIAL | {results['FAIL']} FAIL | {results['CRASH']} CRASH")
    print(f"  PASS RATE: {pass_rate:.0f}% ({results['PASS']}/{total_tests})")
    print(f"  PASS+PARTIAL: {(results['PASS'] + results['PARTIAL']) / total_tests * 100:.0f}%")
    print("=" * 100)

    print(f"\n  CATEGORY BREAKDOWN:")
    for cat, cr in sorted(category_results.items()):
        cat_total = sum(cr.values())
        cat_pass = cr['PASS']
        print(f"    {cat:20s}: {cr['PASS']}P {cr['PARTIAL']}W {cr['FAIL']}F {cr['CRASH']}C  ({cat_pass}/{cat_total})")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for idx, q, grade, issues in failures:
            print(f"    [{idx:2d}] {grade}: {q}")
            for iss in issues[:2]:
                print(f"         -> {iss}")

    print()
    return results['PASS'], total_tests, results


if __name__ == '__main__':
    passed, total, results = main()
    sys.exit(0 if passed >= 34 else 1)
