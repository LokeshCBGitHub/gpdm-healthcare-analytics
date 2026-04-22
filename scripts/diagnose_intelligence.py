import os, sys, re, time, sqlite3, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(os.path.join(SCRIPT_DIR, '..'))

from intelligent_pipeline import IntelligentPipeline

IntelligentPipeline._try_golden_template = lambda self, *a, **kw: None

DB_PATH = os.path.join('data', 'healthcare_production.db')

HARD_QUESTIONS = [
    "What is our profit margin?",
    "Compare NCAL vs SCAL across all KPIs",
    "Where are we leaving money on the table?",
    "Which specialty costs the most per member?",
    "Which month did we lose the most money?",
    "Are high-cost members disenrolling?",
    "How many members never visited a doctor?",
    "How many members died last year?",
    "Which members should we call for retention outreach?",
    "What is the single biggest thing we can do to improve our star rating?",
    "Rank all regions by overall performance",
    "What is our net revenue after all costs?",
    "How much revenue can we recover from denied claims?",
    "What is the most expensive provider specialty?",
    "Which members are most likely to disenroll?",
    "Show me total revenue versus total costs",
    "When did we have the biggest financial loss?",
    "Which members have zero utilization?",
    "How do we boost our star rating the fastest?",
    "What is the mortality rate among our members?",
]

print("=" * 100)
print("  INTELLIGENCE DIAGNOSIS — What do the raw engines produce WITHOUT golden templates?")
print("=" * 100)
print()

pipeline = IntelligentPipeline(db_path=DB_PATH)

pipeline.semantic_cache.cache.clear()
try:
    conn = sqlite3.connect(os.path.join('data', 'answer_cache.db'))
    conn.execute("DELETE FROM answer_cache")
    conn.commit()
    conn.close()
except: pass

fail_count = 0
ok_count = 0

for i, q in enumerate(HARD_QUESTIONS, 1):
    print(f"━━━ [{i:2d}/20] {q}")
    try:
        r = pipeline.process(q, session_id=f'diag_{i}')
    except Exception as e:
        print(f"    CRASH: {e}")
        fail_count += 1
        print()
        continue

    source = r.get('source', '?')
    sql = (r.get('sql', '') or '').replace('\n', ' ')[:150]
    rows = r.get('rows', [])
    error = r.get('error', '')
    narrative = (r.get('narrative', '') or '')[:150]

    problems = []
    q_lower = q.lower()
    sql_upper = (r.get('sql', '') or '').upper()

    if 'SELECT *' in sql_upper and 'LIMIT 50' in sql_upper:
        problems.append('DUMP: SELECT * LIMIT 50 — no intelligence at all')
    if 'SELECT COUNT(*)' in sql_upper and len(sql_upper) < 80 and ('margin' in q_lower or 'revenue' in q_lower):
        problems.append('SHALLOW: Just COUNT(*) for a complex financial question')

    if 'profit margin' in q_lower or 'net revenue' in q_lower:
        if 'BILLED_AMOUNT' not in sql_upper and 'PAID_AMOUNT' not in sql_upper:
            problems.append(f'Missing financial columns — returned {source}')
    if 'compare' in q_lower and 'region' in q_lower:
        if 'KP_REGION' not in sql_upper and 'REGION' not in sql_upper:
            problems.append(f'Missing region grouping — returned {source}')
    if 'specialty' in q_lower and 'cost' in q_lower:
        if 'SPECIALTY' not in sql_upper:
            problems.append(f'Missing SPECIALTY — returned {source}')
    if 'disenroll' in q_lower:
        if 'DISENROLLMENT_DATE' not in sql_upper and 'disenroll' not in narrative.lower():
            problems.append(f'Missing disenrollment logic — returned {source}')
    if 'never visited' in q_lower or 'zero utilization' in q_lower:
        if 'LEFT JOIN' not in sql_upper or 'IS NULL' not in sql_upper:
            problems.append(f'Missing LEFT JOIN/IS NULL — returned {source}')
    if 'died' in q_lower or 'mortality' in q_lower:
        if 'not' not in narrative.lower() and 'cannot' not in narrative.lower() and 'no ' not in narrative.lower()[:50]:
            problems.append(f'Should acknowledge no mortality data — returned {source}')
    if 'month' in q_lower and ('lose' in q_lower or 'loss' in q_lower or 'biggest financial' in q_lower):
        if 'strftime' not in (r.get('sql', '') or '').lower() and 'SERVICE_DATE' not in sql_upper:
            problems.append(f'Missing date grouping — returned {source}')
    if 'money on the table' in q_lower:
        if 'DENIED' not in sql_upper:
            problems.append(f'Missing denied claims analysis — returned {source}')
    if 'retention outreach' in q_lower or 'likely to disenroll' in q_lower:
        if 'RISK_SCORE' not in sql_upper and 'risk' not in narrative.lower():
            problems.append(f'Missing risk-based retention logic — returned {source}')

    if problems:
        fail_count += 1
        sym = '❌'
    else:
        ok_count += 1
        sym = '✅'

    print(f"    {sym} Source: {source} | Rows: {len(rows)}")
    print(f"    SQL: {sql}")
    if problems:
        for p in problems:
            print(f"    ❌ {p}")
    print()

print("=" * 100)
print(f"  RAW INTELLIGENCE SCORE: {ok_count}/20 OK, {fail_count}/20 FAILED")
print(f"  These {fail_count} failures are what the golden templates were covering up.")
print(f"  To be truly intelligent, ALL 20 must pass WITHOUT golden templates.")
print("=" * 100)

print()
print("ENGINE KNOWLEDGE AUDIT:")
print()

sr = pipeline.sql_reasoner
if sr:
    print(f"  SQL Reasoner: {sr.pattern_count} patterns, {sr.template_count} templates")
    if hasattr(sr, '_schema'):
        print(f"    Schema tables: {list(sr._schema.keys())[:10]}")
    if hasattr(sr, '_business_concepts'):
        print(f"    Business concepts: {len(sr._business_concepts)}")
        for k, v in list(sr._business_concepts.items())[:5]:
            print(f"      {k}: {v[:80] if isinstance(v, str) else str(v)[:80]}")
    if hasattr(sr, '_derived_metrics'):
        print(f"    Derived metrics: {list(sr._derived_metrics.keys())[:10] if sr._derived_metrics else 'None'}")

de = pipeline.deep_engine
if de:
    print(f"\n  Deep Understanding Engine: vocab={de.vocab_size}, concepts={de.concept_count}")
    if hasattr(de, '_concept_graph'):
        print(f"    Concept graph nodes: {len(de._concept_graph)}")
    if hasattr(de, '_business_concepts'):
        print(f"    Business concepts: {list(de._business_concepts.keys())[:10] if de._business_concepts else 'None'}")

qg = pipeline.quality_gate
if qg:
    print(f"\n  SQL Quality Gate: {len(qg._table_columns)} tables, {len(qg._learned_corrections)} corrections")
    for t, cols in list(qg._table_columns.items())[:3]:
        print(f"    {t}: {cols[:5]}...")

se = pipeline.sql_engine
if se:
    print(f"\n  Semantic SQL Engine:")
    if hasattr(se, 'derived_concepts'):
        print(f"    Derived concepts: {list(se.derived_concepts.keys())[:10] if se.derived_concepts else 'None'}")
    if hasattr(se, '_schema'):
        print(f"    Schema: {list(se._schema.keys())[:5] if se._schema else 'None'}")
    attrs = [a for a in dir(se) if not a.startswith('_') and not callable(getattr(se, a, None))]
    print(f"    Public attrs: {attrs[:20]}")

print()
