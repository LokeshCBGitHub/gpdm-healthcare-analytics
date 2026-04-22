import os
import sys
import re
import time
import sqlite3
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

from intelligent_pipeline import IntelligentPipeline

HARD_QUESTIONS = [
    {
        'q': 'Compare NCAL vs SCAL across all KPIs',
        'must_have_sql': ['KP_REGION', 'pmpm', 'denial_rate'],
        'must_have_narrative': ['region'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'region_comparison',
    },
    {
        'q': 'What is our profit margin?',
        'must_have_sql': ['margin', 'BILLED_AMOUNT', 'PAID_AMOUNT'],
        'must_have_narrative': ['margin', '%'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'Where are we leaving money on the table?',
        'must_have_sql': ['DENIED', 'BILLED_AMOUNT'],
        'must_have_narrative': ['denied'],
        'reject_sql': ['COUNT(*) FROM claims\n', 'SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'Which specialty costs the most per member?',
        'must_have_sql': ['SPECIALTY', 'cost_per_member'],
        'must_have_narrative': ['specialty'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'provider',
    },
    {
        'q': 'Which month did we lose the most money?',
        'must_have_sql': ['strftime', 'SERVICE_DATE', 'gap'],
        'must_have_narrative': ['month'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'Are high-cost members disenrolling?',
        'must_have_sql': ['DISENROLLMENT_DATE', 'RISK_SCORE'],
        'must_have_narrative': ['disenroll', 'cost'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'retention',
    },
    {
        'q': 'How many members never visited a doctor?',
        'must_have_sql': ['LEFT JOIN', 'IS NULL'],
        'must_have_narrative': ['member'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'utilization',
    },
    {
        'q': 'How many members died last year?',
        'must_have_sql': [],
        'must_have_narrative': ['not contain', 'mortality'],
        'reject_sql': [],
        'category': 'data_gap',
        'expect_data_gap': True,
    },
    {
        'q': 'Which members should we call for retention outreach?',
        'must_have_sql': ['RISK_SCORE', 'MEMBER_ID'],
        'must_have_narrative': ['risk', 'member'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'retention',
    },
    {
        'q': 'What is the single biggest thing we can do to improve our star rating?',
        'must_have_sql': ['screening', 'star'],
        'must_have_narrative': [],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'quality',
    },
    {
        'q': 'Rank all regions by overall performance',
        'must_have_sql': ['KP_REGION', 'pmpm'],
        'must_have_narrative': ['region'],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'region_comparison',
    },
    {
        'q': 'What is our net revenue after all costs?',
        'must_have_sql': ['margin', 'BILLED_AMOUNT', 'PAID_AMOUNT'],
        'must_have_narrative': [],
        'reject_sql': ['LIMIT 50', 'SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'How much revenue can we recover from denied claims?',
        'must_have_sql': ['DENIED', 'BILLED_AMOUNT'],
        'must_have_narrative': ['denied'],
        'reject_sql': ['SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'What is the most expensive provider specialty?',
        'must_have_sql': ['SPECIALTY', 'PAID_AMOUNT'],
        'must_have_narrative': ['specialty'],
        'reject_sql': ['SELECT *'],
        'category': 'provider',
    },
    {
        'q': 'What is the mortality rate among our members?',
        'must_have_sql': [],
        'must_have_narrative': ['not contain', 'mortality'],
        'reject_sql': [],
        'category': 'data_gap',
        'expect_data_gap': True,
    },
    {
        'q': 'Which members are most likely to disenroll?',
        'must_have_sql': ['RISK_SCORE', 'MEMBER_ID'],
        'must_have_narrative': ['risk'],
        'reject_sql': ['SELECT *'],
        'category': 'retention',
    },
    {
        'q': 'Show me total revenue versus total costs',
        'must_have_sql': ['BILLED_AMOUNT', 'PAID_AMOUNT'],
        'must_have_narrative': [],
        'reject_sql': ['SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'When did we have the biggest financial loss?',
        'must_have_sql': ['strftime', 'SERVICE_DATE'],
        'must_have_narrative': ['month'],
        'reject_sql': ['SELECT *'],
        'category': 'financial',
    },
    {
        'q': 'Which members have zero utilization?',
        'must_have_sql': ['LEFT JOIN', 'IS NULL'],
        'must_have_narrative': [],
        'reject_sql': ['SELECT *'],
        'category': 'utilization',
    },
    {
        'q': 'How do we boost our star rating the fastest?',
        'must_have_sql': ['star', 'screening'],
        'must_have_narrative': ['star'],
        'reject_sql': ['SELECT *'],
        'category': 'quality',
    },
]


def score_answer(question_def, result):
    reasons = []
    score = 100

    sql = result.get('sql', '') or ''
    narrative = result.get('narrative', '') or ''
    source = result.get('source', '') or ''
    confidence = result.get('confidence', {})
    if isinstance(confidence, dict):
        grade = confidence.get('grade', '?')
        overall = confidence.get('overall', 0)
    else:
        grade = '?'
        overall = float(confidence) if confidence else 0

    is_data_gap = question_def.get('expect_data_gap', False)

    if is_data_gap:
        if 'data_gap' in source:
            reasons.append('✓ Correctly identified data gap')
        elif any(kw in narrative.lower() for kw in ['not contain', 'mortality', 'death', 'does not']):
            reasons.append('✓ Narrative acknowledges data limitation')
        else:
            reasons.append('✗ Failed to identify data gap — returned generic answer')
            score -= 50
        return _grade_from_score(score), reasons

    if 'golden_template' in source:
        reasons.append(f'✓ Golden template matched ({source})')
    elif 'fallback' in source or source == 'dynamic_sql':
        reasons.append(f'⚠ Used fallback/dynamic path ({source})')
        score -= 15

    sql_upper = sql.upper()
    for must in question_def.get('must_have_sql', []):
        if must.upper() in sql_upper:
            pass
        else:
            reasons.append(f'✗ SQL missing: {must}')
            score -= 15

    for reject in question_def.get('reject_sql', []):
        if reject.upper() in sql_upper:
            reasons.append(f'✗ SQL contains shallow pattern: {reject}')
            score -= 25

    narr_lower = narrative.lower()
    for must in question_def.get('must_have_narrative', []):
        if must.lower() in narr_lower:
            pass
        else:
            reasons.append(f'⚠ Narrative missing keyword: {must}')
            score -= 10

    rows = result.get('rows', [])
    if not rows and not is_data_gap:
        reasons.append('✗ No data rows returned')
        score -= 20

    if grade == 'A':
        reasons.append('✓ Confidence: A')
    elif grade == 'B':
        reasons.append('⚠ Confidence: B (should be A for golden template)')
        score -= 5
    else:
        reasons.append(f'✗ Confidence: {grade}')
        score -= 15

    if not reasons:
        reasons.append('✓ All checks passed')

    return _grade_from_score(score), reasons


def _grade_from_score(score):
    if score >= 90:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 50:
        return 'C'
    else:
        return 'F'


def main():
    print("=" * 80)
    print("   INTELLIGENCE DEPTH AUDIT — 20 Hardest Business Questions")
    print("=" * 80)
    print()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    print("Initializing IntelligentPipeline...")
    t0 = time.time()
    pipeline = IntelligentPipeline(db_path=DB_PATH)
    print(f"Pipeline ready in {time.time() - t0:.1f}s")
    print()

    grades = {'A': 0, 'B': 0, 'C': 0, 'F': 0}
    failures = []
    results_detail = []

    for i, qdef in enumerate(HARD_QUESTIONS, 1):
        q = qdef['q']
        print(f"  [{i:2d}/20] {q}")

        try:
            result = pipeline.process(q, session_id=f'audit_{i}')
        except Exception as e:
            print(f"         ✗ CRASH: {e}")
            grades['F'] += 1
            failures.append((i, q, 'CRASH', str(e)))
            continue

        grade, reasons = score_answer(qdef, result)
        grades[grade] += 1

        sql_preview = (result.get('sql', '') or '')[:80].replace('\n', ' ')
        source = result.get('source', '?')
        narr_preview = (result.get('narrative', '') or '')[:100].replace('\n', ' ')

        symbol = '✓' if grade == 'A' else ('⚠' if grade == 'B' else '✗')
        print(f"         {symbol} Grade: {grade} | Source: {source}")
        for r in reasons:
            print(f"           {r}")
        if grade != 'A':
            print(f"           SQL: {sql_preview}...")
            failures.append((i, q, grade, reasons))

        results_detail.append({
            'question': q,
            'grade': grade,
            'source': source,
            'sql_preview': sql_preview,
            'reasons': reasons,
            'category': qdef['category'],
        })
        print()

    total = sum(grades.values())
    a_pct = grades['A'] / total * 100 if total else 0
    print("=" * 80)
    print(f"   RESULTS: {grades['A']}A  {grades['B']}B  {grades['C']}C  {grades['F']}F")
    print(f"   A-GRADE RATE: {a_pct:.0f}%  ({grades['A']}/{total})")
    print(f"   PREVIOUS SCORE: 84% (7A, 13B)")
    print(f"   TARGET: 95%+ (19/20 A-grades)")
    print("=" * 80)

    if failures:
        print(f"\n   {len(failures)} QUESTIONS BELOW A-GRADE:")
        for idx, q, grade, detail in failures:
            print(f"     [{idx}] {q} → {grade}")

    cats = {}
    for r in results_detail:
        cat = r['category']
        if cat not in cats:
            cats[cat] = {'A': 0, 'B': 0, 'C': 0, 'F': 0}
        cats[cat][r['grade']] += 1

    print(f"\n   CATEGORY BREAKDOWN:")
    for cat, cg in sorted(cats.items()):
        cat_total = sum(cg.values())
        print(f"     {cat:25s}: {cg['A']}A {cg['B']}B {cg['C']}C {cg['F']}F  ({cg['A']}/{cat_total})")

    print()
    return grades['A'], total


if __name__ == '__main__':
    a_count, total = main()
    sys.exit(0 if a_count >= 18 else 1)
