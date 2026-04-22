import sys
import os
import sqlite3
import time
import logging
from typing import Any, Optional, Callable, List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

logger = logging.getLogger('developer_manager_test')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')


def _try_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


DEVELOPER_MANAGER_TESTS = [
    {
        'question': 'What is the highest cost claim?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': 210600.41,
        'tolerance': 0.01,
        'sql_must_have': ['MAX', 'PAID_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the lowest risk score?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MIN(CAST(RISK_SCORE AS REAL)) FROM members',
        'expected_value': 0.10,
        'tolerance': 0.05,
        'sql_must_have': ['MIN', 'RISK_SCORE', 'members'],
    },
    {
        'question': 'What is the maximum paid amount on a single claim?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': 210600.41,
        'tolerance': 0.01,
        'sql_must_have': ['MAX', 'PAID_AMOUNT'],
    },
    {
        'question': 'What is the largest panel size?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(PANEL_SIZE AS REAL)) FROM providers',
        'expected_value': 2500.0,
        'tolerance': 0.01,
        'sql_must_have': ['MAX', 'PANEL_SIZE', 'providers'],
    },
    {
        'question': 'What is the lowest copay?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MIN(CAST(COPAY AS REAL)) FROM claims',
        'expected_value': 0.0,
        'tolerance': 0.01,
        'sql_must_have': ['MIN', 'COPAY'],
    },
    {
        'question': 'What is the highest billed amount?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(BILLED_AMOUNT AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['MAX', 'BILLED_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the most expensive medication?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(COST AS REAL)) FROM prescriptions',
        'expected_value': None,
        'sql_must_have': ['MAX', 'COST', 'prescriptions'],
    },
    {
        'question': 'What is the least expensive prescription?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MIN(CAST(COST AS REAL)) FROM prescriptions',
        'expected_value': None,
        'sql_must_have': ['MIN', 'COST', 'prescriptions'],
    },
    {
        'question': 'What is the maximum length of stay?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(LENGTH_OF_STAY AS REAL)) FROM encounters',
        'expected_value': None,
        'sql_must_have': ['MAX', 'LENGTH_OF_STAY'],
    },
    {
        'question': 'What is the minimum length of stay?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MIN(CAST(LENGTH_OF_STAY AS REAL)) FROM encounters',
        'expected_value': None,
        'sql_must_have': ['MIN', 'LENGTH_OF_STAY'],
    },
    {
        'question': 'What is the highest allowed amount?',
        'category': 'superlative_disambiguation',
        'ground_truth_sql': 'SELECT MAX(CAST(ALLOWED_AMOUNT AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['MAX', 'ALLOWED_AMOUNT'],
    },

    {
        'question': 'How many claims were denied?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': "SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'",
        'expected_value': None,
        'sql_must_have': ['COUNT', 'claims', 'CLAIM_STATUS', 'DENIED'],
    },
    {
        'question': 'How many referrals were denied?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': "SELECT COUNT(*) FROM referrals WHERE STATUS='DENIED'",
        'expected_value': None,
        'sql_must_have': ['COUNT', 'referrals', 'STATUS', 'DENIED'],
    },
    {
        'question': 'What is the total cost of claims?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['SUM', 'PAID_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the total cost of prescriptions?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT SUM(CAST(COST AS REAL)) FROM prescriptions',
        'expected_value': None,
        'sql_must_have': ['SUM', 'COST', 'prescriptions'],
    },
    {
        'question': 'What is the average cost per claim?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PAID_AMOUNT AS REAL)),2) FROM claims',
        'expected_value': None,
        'sql_must_have': ['AVG', 'PAID_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the average cost per prescription?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(COST AS REAL)),2) FROM prescriptions',
        'expected_value': None,
        'sql_must_have': ['AVG', 'COST', 'prescriptions'],
    },
    {
        'question': 'Show me the denied claims count',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': "SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'",
        'expected_value': None,
        'sql_must_have': ['COUNT', 'claims', 'CLAIM_STATUS', 'DENIED'],
    },
    {
        'question': 'Show me the denied referrals count',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': "SELECT COUNT(*) FROM referrals WHERE STATUS='DENIED'",
        'expected_value': None,
        'sql_must_have': ['COUNT', 'referrals', 'STATUS', 'DENIED'],
    },
    {
        'question': 'What is the status breakdown for claims?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT CLAIM_STATUS, COUNT(*) FROM claims GROUP BY CLAIM_STATUS',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'CLAIM_STATUS', 'claims'],
    },
    {
        'question': 'What is the status breakdown for referrals?',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT STATUS, COUNT(*) FROM referrals GROUP BY STATUS',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'STATUS', 'referrals'],
    },
    {
        'question': 'Show top providers by claims',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT RENDERING_NPI, COUNT(*) FROM claims GROUP BY RENDERING_NPI ORDER BY COUNT(*) DESC LIMIT 10',
        'expected_value': None,
        'sql_must_have': ['claims', 'GROUP BY', 'RENDERING_NPI'],
    },
    {
        'question': 'Show top medications by cost',
        'category': 'same_keyword_different_tables',
        'ground_truth_sql': 'SELECT MEDICATION_NAME, SUM(CAST(COST AS REAL)) FROM prescriptions GROUP BY MEDICATION_NAME ORDER BY SUM(CAST(COST AS REAL)) DESC LIMIT 10',
        'expected_value': None,
        'sql_must_have': ['prescriptions', 'MEDICATION_NAME', 'SUM', 'COST'],
    },

    {
        'question': 'What is the average days supply for prescriptions?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(DAYS_SUPPLY AS REAL)),2) FROM prescriptions',
        'expected_value': 40.18,
        'tolerance': 0.05,
        'sql_must_have': ['AVG', 'DAYS_SUPPLY', 'prescriptions'],
    },
    {
        'question': 'What is the average length of stay?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)),2) FROM encounters',
        'expected_value': 3.86,
        'tolerance': 0.05,
        'sql_must_have': ['AVG', 'LENGTH_OF_STAY', 'encounters'],
    },
    {
        'question': 'What is the average risk score?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members',
        'expected_value': 0.84,
        'tolerance': 0.05,
        'sql_must_have': ['AVG', 'RISK_SCORE', 'members'],
    },
    {
        'question': 'What is the total paid amount?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['SUM', 'PAID_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the average copay per claim?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(COPAY AS REAL)),2) FROM claims',
        'expected_value': None,
        'sql_must_have': ['AVG', 'COPAY', 'claims'],
    },
    {
        'question': 'What is the average panel size?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PANEL_SIZE AS REAL)),2) FROM providers',
        'expected_value': None,
        'sql_must_have': ['AVG', 'PANEL_SIZE', 'providers'],
    },
    {
        'question': 'What is the maximum billed amount?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT MAX(CAST(BILLED_AMOUNT AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['MAX', 'BILLED_AMOUNT', 'claims'],
    },
    {
        'question': 'What is the total number of unique providers?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT COUNT(DISTINCT NPI) FROM providers',
        'expected_value': 3000,
        'tolerance': 0.01,
        'sql_must_have': ['COUNT', 'DISTINCT', 'NPI', 'providers'],
    },
    {
        'question': 'What is the average coinsurance amount?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(COINSURANCE AS REAL)),2) FROM claims',
        'expected_value': None,
        'sql_must_have': ['AVG', 'COINSURANCE', 'claims'],
    },
    {
        'question': 'What is the sum of deductibles?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT SUM(CAST(DEDUCTIBLE AS REAL)) FROM claims',
        'expected_value': None,
        'sql_must_have': ['SUM', 'DEDUCTIBLE', 'claims'],
    },
    {
        'question': 'What is the average RVU by category?',
        'category': 'aggregation_accuracy',
        'ground_truth_sql': 'SELECT CATEGORY, ROUND(AVG(CAST(RVU AS REAL)),2) FROM cpt_codes GROUP BY CATEGORY',
        'expected_value': None,
        'sql_must_have': ['AVG', 'RVU', 'GROUP BY', 'CATEGORY', 'cpt_codes'],
    },

    {
        'question': 'hwomany clams are there',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT COUNT(*) FROM claims',
        'expected_value': 60000,
        'tolerance': 0.01,
        'sql_must_have': ['COUNT', 'claims'],
    },
    {
        'question': 'averge payd amout per clam',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(PAID_AMOUNT AS REAL)),2) FROM claims',
        'expected_value': None,
        'sql_must_have': ['AVG', 'PAID_AMOUNT', 'claims'],
    },
    {
        'question': 'toal numbr of memebers',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT COUNT(*) FROM members',
        'expected_value': 25000,
        'tolerance': 0.01,
        'sql_must_have': ['COUNT', 'members'],
    },
    {
        'question': 'deneid clams',
        'category': 'typo_resilience',
        'ground_truth_sql': "SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS='DENIED'",
        'expected_value': None,
        'sql_must_have': ['claims', 'DENIED'],
    },
    {
        'question': 'higest rist score',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT MAX(CAST(RISK_SCORE AS REAL)) FROM members',
        'expected_value': None,
        'sql_must_have': ['MAX', 'RISK_SCORE'],
    },
    {
        'question': 'prescriptons by medicaton class',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT MEDICATION_CLASS, COUNT(*) FROM prescriptions GROUP BY MEDICATION_CLASS',
        'expected_value': None,
        'sql_must_have': ['prescriptions', 'MEDICATION_CLASS', 'GROUP BY'],
    },
    {
        'question': 'wats the denial rate',
        'category': 'typo_resilience',
        'ground_truth_sql': "SELECT ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) / COUNT(*), 2) FROM claims",
        'expected_value': None,
        'sql_must_have': ['claims'],
    },
    {
        'question': 'averge lenght of stey',
        'category': 'typo_resilience',
        'ground_truth_sql': 'SELECT ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)),2) FROM encounters',
        'expected_value': 3.86,
        'tolerance': 0.05,
        'sql_must_have': ['AVG', 'LENGTH_OF_STAY'],
    },

    {
        'question': 'Show me claims by region',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT KP_REGION, COUNT(*) FROM claims GROUP BY KP_REGION',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'KP_REGION', 'claims'],
    },
    {
        'question': 'Show me claims by plan type',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT PLAN_TYPE, COUNT(*) FROM claims GROUP BY PLAN_TYPE',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'PLAN_TYPE', 'claims'],
    },
    {
        'question': 'Show me encounters by visit type',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT VISIT_TYPE, COUNT(*) FROM encounters GROUP BY VISIT_TYPE',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'VISIT_TYPE', 'encounters'],
    },
    {
        'question': 'Average cost by department',
        'category': 'group_by_completeness',
        'ground_truth_sql': "SELECT e.DEPARTMENT, ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)),2) FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID GROUP BY e.DEPARTMENT",
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'DEPARTMENT'],
    },
    {
        'question': 'Member count by gender',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT GENDER, COUNT(*) FROM members GROUP BY GENDER',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'GENDER', 'members'],
    },
    {
        'question': 'Show prescriptions by medication class',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT MEDICATION_CLASS, COUNT(*) FROM prescriptions GROUP BY MEDICATION_CLASS',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'MEDICATION_CLASS', 'prescriptions'],
    },
    {
        'question': 'Denial rate by payer',
        'category': 'group_by_completeness',
        'ground_truth_sql': "SELECT PLAN_TYPE, ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) / COUNT(*), 2) FROM claims GROUP BY PLAN_TYPE",
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'PLAN_TYPE', 'claims'],
    },
    {
        'question': 'Show average allowed amount by plan type',
        'category': 'group_by_completeness',
        'ground_truth_sql': 'SELECT PLAN_TYPE, ROUND(AVG(CAST(ALLOWED_AMOUNT AS REAL)),2) FROM claims GROUP BY PLAN_TYPE',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'PLAN_TYPE', 'claims', 'ALLOWED_AMOUNT'],
    },

    {
        'question': 'How many members are over 65 years old?',
        'category': 'demographic_insights',
        'ground_truth_sql': "SELECT COUNT(*) FROM members WHERE DATE_OF_BIRTH <= date('now', '-65 years')",
        'expected_value': None,
        'sql_must_have': ['COUNT', 'members'],
    },
    {
        'question': 'Show member count by gender',
        'category': 'demographic_insights',
        'ground_truth_sql': 'SELECT GENDER, COUNT(*) FROM members GROUP BY GENDER',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'GENDER', 'members'],
    },
    {
        'question': 'Show members by race',
        'category': 'demographic_insights',
        'ground_truth_sql': 'SELECT RACE, COUNT(*) FROM members GROUP BY RACE',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'RACE', 'members'],
    },
    {
        'question': 'What is the average risk score by region?',
        'category': 'demographic_insights',
        'ground_truth_sql': 'SELECT KP_REGION, ROUND(AVG(CAST(RISK_SCORE AS REAL)),2) FROM members GROUP BY KP_REGION',
        'expected_value': None,
        'sql_must_have': ['AVG', 'RISK_SCORE', 'GROUP BY', 'KP_REGION', 'members'],
    },
    {
        'question': 'Show member enrollment by region',
        'category': 'demographic_insights',
        'ground_truth_sql': 'SELECT KP_REGION, COUNT(*) FROM members GROUP BY KP_REGION',
        'expected_value': None,
        'sql_must_have': ['GROUP BY', 'KP_REGION', 'members'],
    },

    {
        'question': 'Which providers have the highest denial rates?',
        'category': 'multi_table_cross_domain',
        'ground_truth_sql': "SELECT RENDERING_NPI, ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) / COUNT(*), 2) FROM claims GROUP BY RENDERING_NPI ORDER BY 2 DESC LIMIT 10",
        'expected_value': None,
        'sql_must_have': ['claims', 'RENDERING_NPI', 'GROUP BY'],
    },
    {
        'question': 'What is the cost per member by age group?',
        'category': 'multi_table_cross_domain',
        'ground_truth_sql': 'SELECT m.MEMBER_ID, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / COUNT(DISTINCT c.CLAIM_ID), 2) FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY m.MEMBER_ID LIMIT 100',
        'expected_value': None,
        'sql_must_have': ['claims', 'members', 'JOIN'],
    },
    {
        'question': 'Which region has the highest cost per member?',
        'category': 'multi_table_cross_domain',
        'ground_truth_sql': 'SELECT m.KP_REGION, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / COUNT(DISTINCT m.MEMBER_ID), 2) FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID GROUP BY m.KP_REGION ORDER BY 2 DESC LIMIT 1',
        'expected_value': None,
        'sql_must_have': ['claims', 'members', 'JOIN', 'KP_REGION', 'GROUP BY'],
    },
    {
        'question': 'What are the top diagnoses by claim cost?',
        'category': 'multi_table_cross_domain',
        'ground_truth_sql': 'SELECT d.ICD10_DESCRIPTION, ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) FROM diagnoses d JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID GROUP BY d.ICD10_DESCRIPTION ORDER BY 2 DESC LIMIT 10',
        'expected_value': None,
        'sql_must_have': ['diagnoses', 'claims', 'JOIN', 'GROUP BY'],
    },
    {
        'question': 'What is the average cost per encounter by visit type?',
        'category': 'multi_table_cross_domain',
        'ground_truth_sql': 'SELECT e.VISIT_TYPE, ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2) FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID GROUP BY e.VISIT_TYPE',
        'expected_value': None,
        'sql_must_have': ['encounters', 'claims', 'JOIN', 'VISIT_TYPE', 'GROUP BY'],
    },
]


def run_tests() -> tuple:
    print("=" * 80)
    print("DEVELOPER/MANAGER IRON-CLAD TEST SUITE")
    print("=" * 80)
    print()

    for cache_file in ['answer_cache.db', 'query_cache.db']:
        cache_path = os.path.join(DATA_DIR, cache_file)
        if os.path.exists(cache_path):
            try:
                import sqlite3 as _sq3
                _conn = _sq3.connect(cache_path)
                for tbl in _conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                    _conn.execute(f"DELETE FROM {tbl[0]}")
                _conn.commit()
                _conn.close()
                logger.info("Cleared persistent cache: %s", cache_file)
            except Exception as e:
                logger.debug("Cache clear error (non-fatal): %s", e)

    try:
        from intelligent_pipeline import IntelligentPipeline
        pipeline = IntelligentPipeline(DB_PATH)

        if hasattr(pipeline, 'sql_engine') and hasattr(pipeline.sql_engine, 'answer_cache'):
            pipeline.sql_engine.answer_cache.cache = {}
            logger.info("Cleared answer cache")
        if hasattr(pipeline, 'learning') and hasattr(pipeline.learning, 'cache'):
            pipeline.learning.cache = {}
            logger.info("Cleared learning cache")
        if hasattr(pipeline, 'answer_cache') and pipeline.answer_cache:
            try:
                pipeline.answer_cache._entries = {}
                pipeline.answer_cache._vecs = None
                pipeline.answer_cache._keys = []
                logger.info("Cleared answer_cache in-memory entries")
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 0, 0

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    total_passed = 0
    total_failed = 0
    results_by_category = {}
    failures = []

    print(f"Running {len(DEVELOPER_MANAGER_TESTS)} test cases...")
    print()

    for i, test in enumerate(DEVELOPER_MANAGER_TESTS, 1):
        question = test['question']
        category = test['category']

        if category not in results_by_category:
            results_by_category[category] = {'passed': 0, 'failed': 0}

        try:
            cur = db.cursor()
            cur.execute(test['ground_truth_sql'])
            ground_truth_rows = cur.fetchall()
            ground_truth_val = ground_truth_rows[0][0] if ground_truth_rows else None

            result = pipeline.process(question, session_id=f'dev_manager_test_{i}')
            sql = result.get('sql', '')
            source = result.get('source', '')
            row_count = result.get('row_count', 0)

            if sql.startswith('--'):
                dims = result.get('dimensions', [])
                if dims:
                    sql = dims[0].get('sql', sql)

            cur.execute(sql)
            rows = cur.fetchall()
            actual_val = rows[0][0] if rows else None

            passed = False
            validation_errors = []

            sql_upper = sql.upper()
            for keyword in test.get('sql_must_have', []):
                if keyword.upper() not in sql_upper:
                    validation_errors.append(f"Missing '{keyword}' in SQL")

            for keyword in test.get('sql_must_not_have', []):
                if keyword.upper() in sql_upper:
                    validation_errors.append(f"Found forbidden '{keyword}' in SQL")

            if test.get('expected_value') is not None and not validation_errors:
                try:
                    expected = test['expected_value']
                    tolerance = test.get('tolerance', 0.05)

                    best_match = None
                    best_err = float('inf')
                    for col_idx, cell in enumerate(rows[0] if rows else []):
                        try:
                            actual_num = float(cell)
                            err = abs(actual_num - expected) / max(abs(expected), 0.001)
                            if err < best_err:
                                best_err = err
                                best_match = actual_num
                        except (TypeError, ValueError):
                            continue

                    if best_match is not None and best_err <= tolerance:
                        passed = True
                        actual_val = best_match
                    elif best_match is not None:
                        validation_errors.append(
                            f"Value mismatch: got {best_match}, expected {expected} "
                            f"(error {best_err:.2%})"
                        )
                        actual_val = best_match
                except (TypeError, ValueError) as e:
                    validation_errors.append(f"Numeric validation error: {e}")
            elif test.get('validate_fn'):
                try:
                    passed = test['validate_fn'](rows, [d[0] for d in cur.description] if cur.description else [])
                except Exception as e:
                    validation_errors.append(f"Custom validation failed: {e}")
            elif not validation_errors:
                passed = True

            if not validation_errors and not test.get('expected_value'):
                passed = True

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
                    'got': actual_val,
                    'expected': test.get('expected_value', 'structure_check'),
                    'source': source,
                    'errors': validation_errors,
                })

            err_str = f" [{'; '.join(validation_errors[:1])}]" if validation_errors else ""
            print(f"  {status}: #{i:02d} [{category:30s}] \"{question}\" → {actual_val}{err_str}")

        except Exception as e:
            total_failed += 1
            results_by_category.setdefault(category, {'passed': 0, 'failed': 0})
            results_by_category[category]['failed'] += 1
            failures.append({
                'question': question,
                'category': category,
                'error': str(e),
            })
            print(f"  ERROR: #{i:02d} [{category:30s}] \"{question}\" → {e}")

    db.close()

    print()
    print("=" * 80)
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat, counts in sorted(results_by_category.items()):
        total = counts['passed'] + counts['failed']
        pct = 100.0 * counts['passed'] / total if total > 0 else 0
        status = '✓' if counts['failed'] == 0 else '✗'
        print(f"  {status} {cat:35s} {counts['passed']:2d}/{total:2d} ({pct:5.1f}%)")

    print()
    print("=" * 80)
    total = total_passed + total_failed
    pct = 100.0 * total_passed / total if total > 0 else 0
    print(f"DEVELOPER/MANAGER TEST SUITE: {total_passed}/{total} ({pct:.1f}%)")
    print("=" * 80)

    if failures:
        print()
        print("FAILURES:")
        print("-" * 80)
        for f in failures:
            print(f"  Question: \"{f['question']}\" [{f['category']}]")
            if 'error' in f:
                print(f"    Error: {f['error']}")
            else:
                if f.get('errors'):
                    for err in f['errors']:
                        print(f"    - {err}")
                print(f"    Got: {f['got']} | Expected: {f['expected']}")
                print(f"    SQL: {f.get('sql', '')[:120]}")
                if f.get('source'):
                    print(f"    Source: {f.get('source')}")
            print()

    return total_passed, total_failed


if __name__ == '__main__':
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
