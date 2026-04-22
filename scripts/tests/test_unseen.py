import sys, os, sqlite3, time, signal
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)

DB = '../data/healthcare_production.db'
from semantic_sql_engine import SemanticSQLEngine
engine = SemanticSQLEngine(DB)

UNSEEN = [
    ('claims by plan type', 'breakdown', 'Should discover PLAN_TYPE dimension'),
    ('encounters by gender', 'breakdown', 'Cross-table: encounters grouped by members.GENDER'),
    ('prescriptions by race', 'breakdown', 'Cross-table: prescriptions grouped by members.RACE'),
    ('claims percentage by specialty', 'breakdown', 'Should discover SPECIALTY dimension'),

    ('average billed amount by region', 'aggregate', 'Numeric agg by region'),
    ('total paid amount by plan type', 'aggregate', 'SUM by PLAN_TYPE'),
    ('members by race', 'breakdown', 'Direct dimension from members table'),

    ('members with more than 3 claims', 'filter', 'Threshold on claims table'),

    ('inpatient to telehealth ratio', 'aggregate', 'New ratio pair'),
    ('compare emergency vs telehealth', 'comparison', 'New comparison pair'),

    ('what percentage of claims are adjusted', 'percentage', 'Keyword percentage - ADJUSTED'),

    ('claims by visit type', 'breakdown', 'Cross-table with encounter join'),
    ('encounters per region', 'breakdown', 'Same-table dimension'),
    ('most common diagnosis code', 'ranking', 'Most occurring on diagnosis column'),
    ('readmission rate by region', 'rate', 'Readmission with GROUP BY'),
]

class Timeout(Exception): pass
def alarm_handler(s, f): raise Timeout()
signal.signal(signal.SIGALRM, alarm_handler)

p = f = 0; failures = []

for i, (q, exp, notes) in enumerate(UNSEEN, 1):
    r = engine.generate(q)
    sql = r.get('sql', '')
    err = None; cnt = 0; exec_ms = 0

    clean = sql.rstrip(';').strip()
    if not clean:
        err = 'NO_SQL'
    else:
        try:
            conn = sqlite3.connect(DB)
            signal.alarm(8)
            t0 = time.time()
            cnt = conn.execute(f'SELECT COUNT(*) FROM ({clean})').fetchone()[0]
            exec_ms = int((time.time() - t0) * 1000)
            signal.alarm(0)
            conn.close()
        except Timeout:
            signal.alarm(0)
            err = 'TIMEOUT(>8s)'
            try: conn.close()
            except: pass
        except Exception as e:
            signal.alarm(0)
            err = str(e)[:80]
            try: conn.close()
            except: pass

    ok = True; iss = []
    if err: ok = False; iss.append(err)
    if not sql: ok = False; iss.append('NO_SQL')
    if cnt > 5000 and exp in ('ranking', 'breakdown', 'aggregate', 'percentage', 'rate', 'comparison'):
        ok = False; iss.append(f'ROWS:{cnt}')
    if 'ratio' in q.lower() and ('CASE' not in sql.upper() and '/' not in sql):
        ok = False; iss.append('NO_RATIO')

    if ok:
        p += 1
        print(f'  OK [{i:2d}] {cnt:6d} rows {exec_ms:4d}ms | {q}')
        print(f'       {notes}')
    else:
        f += 1
        print(f'FAIL [{i:2d}] {cnt:6d} rows {exec_ms:4d}ms | {q} | {", ".join(iss)}')
        print(f'       {notes}')
        print(f'     SQL: {sql[:200]}')
        failures.append((q, iss, sql))
    print(f'     SQL: {sql[:150]}')
    print()

print(f'{"=" * 70}')
print(f'UNSEEN QUERIES: {p}/{len(UNSEEN)} PASS ({100 * p / len(UNSEEN):.0f}%)')
print(f'{"=" * 70}')

if failures:
    print('\nFAILURES:')
    for q, iss, sql in failures:
        print(f'  - {q}: {", ".join(iss)}')
