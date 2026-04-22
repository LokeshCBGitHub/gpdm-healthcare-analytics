import sys
import os
import sqlite3
import time
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

logger = logging.getLogger('adversarial_test')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')


def _try_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


ADVERSARIAL_TESTS = [
    {
        'question': 'Count of encounters',
        'ground_truth_sql': 'SELECT COUNT(*) FROM encounters',
        'expected_value': 50000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'Count all referrals',
        'ground_truth_sql': 'SELECT COUNT(*) FROM referrals',
        'expected_value': 5000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'How many claims are there',
        'ground_truth_sql': 'SELECT COUNT(*) FROM claims',
        'expected_value': 60000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'Total number of members',
        'ground_truth_sql': 'SELECT COUNT(*) FROM members',
        'expected_value': 25000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'How many providers are there',
        'ground_truth_sql': 'SELECT COUNT(*) FROM providers',
        'expected_value': 3000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'Count of diagnoses',
        'ground_truth_sql': 'SELECT COUNT(*) FROM diagnoses',
        'expected_value': 20000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'How many prescriptions',
        'ground_truth_sql': 'SELECT COUNT(*) FROM prescriptions',
        'expected_value': 12000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },
    {
        'question': 'Number of appointments',
        'ground_truth_sql': 'SELECT COUNT(*) FROM appointments',
        'expected_value': 10000,
        'tolerance': 0.01,
        'category': 'count_join_inflation',
    },

    {
        'question': 'Maximum paid amount on a single claim',
        'ground_truth_sql': 'SELECT MAX(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': 210600.41,
        'tolerance': 0.01,
        'category': 'aggregation_correctness',
    },
    {
        'question': 'Highest cost claim',
        'ground_truth_sql': 'SELECT MAX(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': 210600.41,
        'tolerance': 0.01,
        'category': 'aggregation_correctness',
    },
    {
        'question': 'Minimum risk score',
        'ground_truth_sql': 'SELECT MIN(CAST(RISK_SCORE AS REAL)) FROM members',
        'expected_value': 0.10,
        'tolerance': 0.05,
        'category': 'aggregation_correctness',
    },
    {
        'question': 'Average days supply for prescriptions',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(DAYS_SUPPLY AS REAL)),2) FROM prescriptions',
        'expected_value': 40.18,
        'tolerance': 0.02,
        'category': 'aggregation_correctness',
    },
    {
        'question': 'Average length of stay',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)),2) FROM encounters',
        'expected_value': 3.86,
        'tolerance': 0.05,
        'category': 'aggregation_correctness',
    },
    {
        'question': 'Average risk score',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members',
        'expected_value': 0.84,
        'tolerance': 0.05,
        'category': 'aggregation_correctness',
    },

    {
        'question': 'Average RVU by category',
        'ground_truth_sql': 'SELECT CATEGORY, ROUND(AVG(CAST(RVU AS REAL)),2) FROM cpt_codes GROUP BY CATEGORY',
        'validate_fn': lambda rows, cols: any('CATEGORY' in str(c).upper() or 'category' in str(c).lower() for c in (cols or [])) or len(rows or []) > 1,
        'category': 'table_routing',
    },
    {
        'question': 'Denied referrals',
        'ground_truth_sql': "SELECT COUNT(*) FROM referrals WHERE STATUS='DENIED'",
        'validate_fn': lambda rows, cols: rows and len(rows) > 0 and rows[0][0] is not None,
        'category': 'table_routing',
    },
    {
        'question': 'Average copay per claim',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(COPAY AS REAL)),2) FROM claims',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'table_routing',
    },
    {
        'question': 'Total prescription costs',
        'ground_truth_sql': 'SELECT ROUND(SUM(CAST(COST AS REAL)),2) FROM prescriptions',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'table_routing',
    },

    {
        'question': 'Total paid amount for all claims',
        'ground_truth_sql': 'SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)),2) FROM claims',
        'expected_value': 133923772.42,
        'tolerance': 0.05,
        'category': 'no_phantom_where',
    },
    {
        'question': 'What is the denial rate',
        'ground_truth_sql': "SELECT ROUND(100.0*SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END)/COUNT(*),2) FROM claims",
        'expected_value': 10.03,
        'tolerance': 0.05,
        'category': 'no_phantom_where',
    },
    {
        'question': 'What is the maximum billed amount',
        'ground_truth_sql': 'SELECT MAX(CAST(BILLED_AMOUNT AS REAL)) FROM claims',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 100000,
        'category': 'no_phantom_where',
    },

    {
        'question': 'Average paid amount per claim',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PAID_AMOUNT AS REAL)),2) FROM claims',
        'validate_fn': lambda rows, cols: rows and 1000 < float(rows[0][0]) < 5000,
        'category': 'substring_confusion',
    },
    {
        'question': 'Average cost per encounter',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'substring_confusion',
    },
    {
        'question': 'Percentage of denied claims',
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and 0 < _try_float(cell) < 100
            for row in rows for cell in row
        ),
        'category': 'substring_confusion',
    },

    {
        'question': 'How many referrals were denied',
        'ground_truth_sql': "SELECT COUNT(*) FROM referrals WHERE STATUS='DENIED'",
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and 100 < _try_float(cell) < 10000
            for row in rows for cell in row
        ),
        'category': 'entity_confusion',
    },
    {
        'question': 'How many appointments were cancelled',
        'ground_truth_sql': "SELECT COUNT(*) FROM appointments WHERE STATUS='CANCELLED'",
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 100,
        'category': 'entity_confusion',
    },
    {
        'question': 'How many claims were denied',
        'ground_truth_sql': "SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'",
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and 3000 < _try_float(cell) < 10000
            for row in rows for cell in row
        ),
        'category': 'entity_confusion',
    },

    {
        'question': 'HMO vs Medicare Advantage PMPM',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 1,
        'category': 'derived_metrics',
    },
    {
        'question': 'What is the average length of stay for inpatient encounters',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'derived_metrics',
    },

    {
        'question': 'Average panel size per provider',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PANEL_SIZE AS REAL)),2) FROM providers',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'specific_columns',
    },
    {
        'question': 'Total billed amount across all claims',
        'ground_truth_sql': 'SELECT ROUND(SUM(CAST(BILLED_AMOUNT AS REAL)),2) FROM claims',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 100000000,
        'category': 'specific_columns',
    },

    {
        'question': 'Claims by plan type',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'group_by',
    },
    {
        'question': 'Encounters by department',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'group_by',
    },
    {
        'question': 'Top 10 providers by panel size',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 5,
        'category': 'group_by',
    },
    {
        'question': 'Members by gender',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'group_by',
    },
    {
        'question': 'Prescription count by drug class',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'group_by',
    },

    {
        'question': 'Show me all encounter types',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'edge_cases',
    },
    {
        'question': 'What are the most common diagnoses',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 3,
        'category': 'edge_cases',
    },
    {
        'question': 'Average billed amount by region',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'edge_cases',
    },
    {
        'question': 'Claims denial rate by plan type',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'edge_cases',
    },
    {
        'question': 'How many members have chronic conditions',
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 0,
        'category': 'edge_cases',
    },
    {
        'question': 'What is the no show rate for appointments',
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and _try_float(cell) > 0
            for row in rows for cell in row
        ),
        'category': 'edge_cases',
    },
    {
        'question': 'Referral approval rate',
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and _try_float(cell) > 0
            for row in rows for cell in row
        ),
        'category': 'edge_cases',
    },
    {
        'question': 'Average cost per prescription',
        'validate_fn': lambda rows, cols: rows and 10 < float(rows[0][0]) < 500,
        'category': 'edge_cases',
    },
    {
        'question': 'Highest risk score among all members',
        'validate_fn': lambda rows, cols: rows and any(
            _try_float(cell) is not None and _try_float(cell) > 0
            for row in rows for cell in row
        ),
        'category': 'edge_cases',
    },
    {
        'question': 'Total number of unique providers',
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 1000,
        'category': 'edge_cases',
    },
    {
        'question': 'What is the most common encounter type',
        'validate_fn': lambda rows, cols: rows and rows[0][0] is not None,
        'category': 'edge_cases',
    },
    {
        'question': 'How many members are over 65',
        'validate_fn': lambda rows, cols: rows and int(rows[0][0]) > 100,
        'category': 'edge_cases',
    },
    {
        'question': 'Average allowed amount per claim',
        'validate_fn': lambda rows, cols: rows and float(rows[0][0]) > 0,
        'category': 'edge_cases',
    },
    {
        'question': 'Claims by status',
        'validate_fn': lambda rows, cols: rows and len(rows) >= 2,
        'category': 'edge_cases',
    },
]


def run_adversarial_test():
    from intelligent_pipeline import IntelligentPipeline

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
        os.remove(qg_corrections)

    intel_cache = os.path.join(DATA_DIR, 'intelligence_cache.json')
    if os.path.exists(intel_cache):
        os.remove(intel_cache)

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
    print("ADVERSARIAL ACCURACY TEST — Ground Truth Verified")
    print("=" * 80)
    print()

    for i, test in enumerate(ADVERSARIAL_TESTS, 1):
        question = test['question']
        category = test['category']

        if category not in results_by_category:
            results_by_category[category] = {'passed': 0, 'failed': 0}

        try:
            result = pipeline.process(question, session_id=f'adversarial_test_{i}')
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
                    best_match = None
                    best_err = float('inf')
                    for col_idx, cell in enumerate(rows[0] if rows else []):
                        try:
                            actual_val = float(cell)
                            err = abs(actual_val - expected) / max(abs(expected), 0.001)
                            if err < best_err:
                                best_err = err
                                best_match = actual_val
                        except (TypeError, ValueError):
                            continue
                    if best_match is not None and best_err <= tolerance:
                        passed = True
                        val = best_match
                    elif best_match is not None:
                        val = best_match
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
            print(f"  {status}: #{i:02d} [{category}] \"{question}\" → {val}{qg}")

        except Exception as e:
            total_failed += 1
            results_by_category.setdefault(category, {'passed': 0, 'failed': 0})
            results_by_category[category]['failed'] += 1
            failures.append({
                'question': question,
                'category': category,
                'error': str(e),
            })
            print(f"  ERROR: #{i:02d} [{category}] \"{question}\" → {e}")

    db.close()

    print()
    print("=" * 80)
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat, counts in sorted(results_by_category.items()):
        total = counts['passed'] + counts['failed']
        pct = 100.0 * counts['passed'] / total if total > 0 else 0
        status = '✓' if counts['failed'] == 0 else '✗'
        print(f"  {status} {cat:30s} {counts['passed']}/{total} ({pct:.0f}%)")

    print()
    print("=" * 80)
    total = total_passed + total_failed
    pct = 100.0 * total_passed / total if total > 0 else 0
    print(f"ADVERSARIAL ACCURACY: {total_passed}/{total} ({pct:.1f}%)")
    print("=" * 80)

    if failures:
        print()
        print("FAILURES:")
        for f in failures:
            print(f"  - \"{f['question']}\" [{f['category']}]")
            if 'error' in f:
                print(f"    Error: {f['error']}")
            else:
                print(f"    Got: {f['got']} | Expected: {f['expected']}")
                print(f"    SQL: {f.get('sql', '')[:120]}")
                print(f"    Source: {f.get('source', '')}")

    return total_passed, total_failed


if __name__ == '__main__':
    run_adversarial_test()
