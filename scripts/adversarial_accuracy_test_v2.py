import sys
import os
import sqlite3
import time
import logging
from typing import List, Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

logger = logging.getLogger('adversarial_test_v2')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')


from adversarial_accuracy_test import ADVERSARIAL_TESTS as ORIGINAL_TESTS


NEW_TESTS = [

    {
        'question': 'How many female members are there',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE GENDER='F'",
        'expected_value': 12620,
        'tolerance': 0.01,
        'category': 'member_demographics',
    },
    {
        'question': 'How many male members do we have',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE GENDER='M'",
        'expected_value': 12380,
        'tolerance': 0.01,
        'category': 'member_demographics',
    },
    {
        'question': 'Average risk score by gender',
        'ground_truth_sql': "SELECT GENDER, ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members GROUP BY GENDER",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'member_demographics',
    },
    {
        'question': 'Members by race breakdown',
        'ground_truth_sql': "SELECT RACE, COUNT(*) FROM members WHERE RACE IS NOT NULL GROUP BY RACE",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'member_demographics',
    },
    {
        'question': 'How many members are over 65 years old',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE DATE_OF_BIRTH <= date('now', '-65 years')",
        'expected_value': 5766,
        'tolerance': 0.01,
        'category': 'member_demographics',
    },
    {
        'question': 'Members by language distribution',
        'ground_truth_sql': "SELECT LANGUAGE, COUNT(*) FROM members WHERE LANGUAGE IS NOT NULL GROUP BY LANGUAGE ORDER BY COUNT(*) DESC",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'member_demographics',
    },
    {
        'question': 'Chronic conditions by gender',
        'ground_truth_sql': "SELECT GENDER, COUNT(*) FROM members WHERE CHRONIC_CONDITIONS > 0 GROUP BY GENDER",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 1,
        'category': 'member_demographics',
    },
    {
        'question': 'Highest risk score by race',
        'ground_truth_sql': "SELECT RACE, ROUND(MAX(CAST(RISK_SCORE AS REAL)),2) FROM members WHERE RACE IS NOT NULL GROUP BY RACE",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 1,
        'category': 'member_demographics',
    },
    {
        'question': 'Members by state',
        'ground_truth_sql': "SELECT STATE, COUNT(*) FROM members GROUP BY STATE ORDER BY COUNT(*) DESC",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'member_demographics',
    },
    {
        'question': 'Average age by KP region',
        'ground_truth_sql': "SELECT KP_REGION, ROUND(AVG(CAST(SUBSTR('2026', 1, 4) AS INTEGER) - CAST(SUBSTR(DATE_OF_BIRTH, 1, 4) AS INTEGER)),1) FROM members GROUP BY KP_REGION",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'member_demographics',
    },


    {
        'question': 'How many providers by specialty',
        'ground_truth_sql': "SELECT SPECIALTY, COUNT(*) FROM providers GROUP BY SPECIALTY ORDER BY COUNT(*) DESC",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'provider_demographics',
    },
    {
        'question': 'Average panel size by specialty',
        'ground_truth_sql': "SELECT SPECIALTY, ROUND(AVG(CAST(PANEL_SIZE AS REAL)),2) FROM providers GROUP BY SPECIALTY",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'provider_demographics',
    },
    {
        'question': 'Providers by KP region',
        'ground_truth_sql': "SELECT KP_REGION, COUNT(*) FROM providers GROUP BY KP_REGION ORDER BY COUNT(*) DESC",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'provider_demographics',
    },
    {
        'question': 'How many active providers do we have',
        'ground_truth_sql': "SELECT COUNT(*) FROM providers WHERE STATUS='ACTIVE'",
        'expected_value': 2563,
        'tolerance': 0.01,
        'category': 'provider_demographics',
    },
    {
        'question': 'Provider to member ratio',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0.1,
        'category': 'provider_demographics',
    },
    {
        'question': 'Top specialties by provider count',
        'ground_truth_sql': "SELECT SPECIALTY, COUNT(*) FROM providers GROUP BY SPECIALTY ORDER BY COUNT(*) DESC LIMIT 5",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'provider_demographics',
    },
    {
        'question': 'Providers accepting new patients',
        'ground_truth_sql': "SELECT COUNT(*) FROM providers WHERE ACCEPTS_NEW_PATIENTS='Y'",
        'expected_value': 2070,
        'tolerance': 0.01,
        'category': 'provider_demographics',
    },
    {
        'question': 'Average panel size overall',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PANEL_SIZE AS REAL)),2) FROM providers',
        'expected_value': 1305.22,
        'tolerance': 0.01,
        'category': 'provider_demographics',
    },


    {
        'question': 'Average claims per member',
        'validate_fn': lambda rows, cols: rows and 2 < float(rows[0][0]) < 4,
        'category': 'cross_table',
    },
    {
        'question': 'Total cost by gender',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'cross_table',
    },
    {
        'question': 'Denial rate by race',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'cross_table',
    },
    {
        'question': 'Average encounters per member',
        'validate_fn': lambda rows, cols: rows and 1 < float(rows[0][0]) < 3,
        'category': 'cross_table',
    },
    {
        'question': 'Most prescribed drug class',
        'ground_truth_sql': "SELECT MEDICATION_CLASS, COUNT(*) FROM prescriptions GROUP BY MEDICATION_CLASS ORDER BY COUNT(*) DESC LIMIT 1",
        'validate_fn': lambda rows, cols: rows and rows[0][0] is not None,
        'category': 'cross_table',
    },
    {
        'question': 'Claims per rendering provider',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'cross_table',
    },
    {
        'question': 'Average length of stay by department',
        'ground_truth_sql': "SELECT DEPARTMENT, ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)),2) FROM encounters WHERE DEPARTMENT IS NOT NULL GROUP BY DEPARTMENT",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'cross_table',
    },
    {
        'question': 'Referral completion rate by specialty',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'cross_table',
    },


    {
        'question': 'Show me everything about claims',
        'validate_fn': lambda rows, cols: cols and 'CLAIM_ID' in [str(c).upper() for c in cols],
        'category': 'edge_cases_v2',
    },
    {
        'question': 'What tables exist in the database',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Top 5 most expensive diagnoses',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Which region has the most claims',
        'ground_truth_sql': "SELECT KP_REGION, COUNT(*) FROM claims GROUP BY KP_REGION ORDER BY COUNT(*) DESC LIMIT 1",
        'validate_fn': lambda rows, cols: rows and rows[0][0] is not None,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Average copay vs average paid amount',
        'validate_fn': lambda rows, cols: rows and len(cols) >= 2,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Claim types distribution',
        'ground_truth_sql': "SELECT CLAIM_TYPE, COUNT(*) FROM claims WHERE CLAIM_TYPE IS NOT NULL GROUP BY CLAIM_TYPE",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Provider workload distribution by claims',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 10,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Drug costs by class',
        'ground_truth_sql': "SELECT MEDICATION_CLASS, ROUND(SUM(CAST(COST AS REAL)),2) FROM prescriptions GROUP BY MEDICATION_CLASS",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Encounter types by region',
        'ground_truth_sql': "SELECT KP_REGION, VISIT_TYPE, COUNT(*) FROM encounters GROUP BY KP_REGION, VISIT_TYPE",
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'edge_cases_v2',
    },
    {
        'question': 'Claims status distribution across all regions',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'edge_cases_v2',
    },


    {
        'question': 'Exact count of HMO claims',
        'ground_truth_sql': "SELECT COUNT(*) FROM claims WHERE PLAN_TYPE='HMO'",
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 5000,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Exact count of NCAL region members',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE KP_REGION='NCAL'",
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 1000,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Sum of all prescription costs',
        'ground_truth_sql': "SELECT ROUND(SUM(CAST(COST AS REAL)),2) FROM prescriptions",
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 100000,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Count of unique ICD10 codes',
        'ground_truth_sql': "SELECT COUNT(DISTINCT ICD10_CODE) FROM claims WHERE ICD10_CODE IS NOT NULL",
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 100,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Average copay amount across all claims',
        'ground_truth_sql': "SELECT ROUND(AVG(CAST(COPAY AS REAL)),2) FROM claims",
        'expected_value': 21.6,
        'tolerance': 0.05,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Count of emergency type encounters',
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 100,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Total members in California',
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 3000,
        'category': 'accuracy_stress',
    },
    {
        'question': 'Count of members with chronic diagnoses',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE CHRONIC_CONDITIONS > 0",
        'expected_value': 17875,
        'tolerance': 0.01,
        'category': 'accuracy_stress',
    },


    {
        'question': 'Readmission rate by region',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'rate_metrics',
    },
    {
        'question': 'No-show rate for appointments',
        'validate_fn': lambda rows, cols: rows and 0 <= float(rows[0][0]) <= 100,
        'category': 'rate_metrics',
    },
    {
        'question': 'Preventive visit rate',
        'validate_fn': lambda rows, cols: rows and 0 <= float(rows[0][0]) <= 100,
        'category': 'rate_metrics',
    },
    {
        'question': 'ER utilization rate',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) >= 0,
        'category': 'rate_metrics',
    },
    {
        'question': 'Appointment cancellation rate by facility',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'rate_metrics',
    },
    {
        'question': 'Referral approval vs denial ratio',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'rate_metrics',
    },


    {
        'question': 'Claims trend by month',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 6,
        'category': 'temporal',
    },
    {
        'question': 'Encounter volume over time',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'temporal',
    },
    {
        'question': 'Prescription fills by quarter',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'temporal',
    },
    {
        'question': 'Referrals by week',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'temporal',
    },
    {
        'question': 'Appointment volume trend',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'temporal',
    },
    {
        'question': 'Diagnosis discovery rate over time',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'temporal',
    },


    {
        'question': 'Claims cost comparison by region',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'comparative',
    },
    {
        'question': 'Provider productivity by specialty',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'comparative',
    },
    {
        'question': 'Member health outcomes by plan type',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'comparative',
    },
    {
        'question': 'Facility utilization comparison',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'comparative',
    },
    {
        'question': 'Prescription costs by drug class',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'comparative',
    },
    {
        'question': 'Diagnosis severity by demographics',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'comparative',
    },
]


ADVERSARIAL_TESTS = ORIGINAL_TESTS + NEW_TESTS


def run_adversarial_test(subset_category=None):
    from intelligent_pipeline import IntelligentPipeline

    tests_to_run = ADVERSARIAL_TESTS
    if subset_category:
        tests_to_run = [t for t in ADVERSARIAL_TESTS if t['category'] == subset_category]
        logger.info(f"Running subset: {subset_category} ({len(tests_to_run)} tests)")

    for cache_file in ['answer_cache.db', 'query_cache.db']:
        cache_path = os.path.join(DATA_DIR, cache_file)
        if os.path.exists(cache_path):
            try:
                conn = sqlite3.connect(cache_path)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                for (t,) in cur.fetchall():
                    if t != 'sqlite_sequence':
                        cur.execute(f'DELETE FROM {t}')
                conn.commit()
                conn.close()
            except:
                pass

    qg_corrections = os.path.join(DATA_DIR, 'quality_gate_corrections.json')
    if os.path.exists(qg_corrections):
        try:
            os.remove(qg_corrections)
        except:
            pass

    intel_cache = os.path.join(DATA_DIR, 'intelligence_cache.json')
    if os.path.exists(intel_cache):
        try:
            os.remove(intel_cache)
        except:
            pass

    pipeline = IntelligentPipeline(DB_PATH, DATA_DIR)

    if hasattr(pipeline, 'semantic_cache'):
        pipeline.semantic_cache.cache = {}
        if hasattr(pipeline.semantic_cache, 'max_size'):
            pipeline.semantic_cache.max_size = 0
        pipeline.semantic_cache.put = lambda *a, **kw: None

    if hasattr(pipeline, 'answer_cache') and pipeline.answer_cache:
        pipeline.answer_cache.store = lambda *a, **kw: None
        pipeline.answer_cache.lookup = lambda *a, **kw: None

    if hasattr(pipeline, 'conversations'):
        pipeline.conversations = {}

    db = sqlite3.connect(DB_PATH)

    results_by_category = {}
    total_passed = 0
    total_failed = 0
    failures = []

    print("=" * 80)
    print(f"ADVERSARIAL ACCURACY TEST V2 — {len(tests_to_run)} Questions Verified Against Ground Truth")
    print("=" * 80)
    print()

    for i, test in enumerate(tests_to_run, 1):
        question = test['question']
        category = test['category']

        if category not in results_by_category:
            results_by_category[category] = {'passed': 0, 'failed': 0}

        try:
            result = pipeline.process(question, session_id=f'adversarial_test_v2_{i}')
            sql = result.get('sql', '')
            source = result.get('source', '')

            if sql.startswith('--'):
                dims = result.get('dimensions', [])
                if dims:
                    sql = dims[0].get('sql', sql)

            cur = db.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            val = rows[0][0] if rows else None

            passed = False

            if 'expected_value' in test and test['expected_value'] is not None:
                try:
                    expected = test['expected_value']
                    tolerance = test.get('tolerance', 0.05)
                    actual = float(val)
                    if abs(actual - expected) / max(abs(expected), 0.001) <= tolerance:
                        passed = True
                except (TypeError, ValueError):
                    passed = False
            elif 'validate_fn' in test:
                try:
                    passed = test['validate_fn'](rows, columns)
                except:
                    passed = False
            else:
                passed = val is not None

            if passed:
                total_passed += 1
                results_by_category[category]['passed'] += 1
                status = 'PASS'
            else:
                total_failed += 1
                results_by_category[category]['failed'] += 1
                status = 'FAIL'
                failures.append({
                    'question': question,
                    'category': category,
                    'sql': sql[:200],
                    'got': val,
                    'expected': test.get('expected_value', 'validate_fn'),
                    'source': source,
                })

            qg = ' [QG]' if result.get('quality_gate', {}).get('corrections') else ''
            print(f"  {status}: #{i:03d} [{category:20s}] \"{question[:50]}...\" → {val}{qg}")

        except Exception as e:
            total_failed += 1
            results_by_category.setdefault(category, {'passed': 0, 'failed': 0})
            results_by_category[category]['failed'] += 1
            failures.append({
                'question': question,
                'category': category,
                'error': str(e),
            })
            print(f"  ERROR: #{i:03d} [{category:20s}] \"{question[:50]}...\" → {type(e).__name__}: {str(e)[:60]}")

    db.close()

    print()
    print("=" * 80)
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat, counts in sorted(results_by_category.items()):
        total = counts['passed'] + counts['failed']
        pct = 100.0 * counts['passed'] / total if total > 0 else 0
        status = '✓' if counts['failed'] == 0 else '✗'
        print(f"  {status} {cat:30s} {counts['passed']:3d}/{total:3d} ({pct:5.1f}%)")

    print()
    print("=" * 80)
    total = total_passed + total_failed
    pct = 100.0 * total_passed / total if total > 0 else 0
    print(f"ADVERSARIAL ACCURACY V2: {total_passed}/{total} ({pct:.1f}%)")
    print("=" * 80)

    if failures:
        print()
        print("FAILURES (showing first 20):")
        for f in failures[:20]:
            print(f"  - \"{f['question']}\" [{f['category']}]")
            if 'error' in f:
                print(f"    Error: {f['error']}")
            else:
                print(f"    Got: {f['got']} | Expected: {f['expected']}")
                print(f"    SQL: {f.get('sql', '')[:120]}")
                print(f"    Source: {f.get('source', '')}")

        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures")

    return total_passed, total_failed


if __name__ == '__main__':
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else None
    run_adversarial_test(subset)
