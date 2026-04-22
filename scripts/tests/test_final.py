import sys, os, sqlite3, time, signal
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)

DB = '../data/healthcare_production.db'

conn = sqlite3.connect(DB)
for idx_sql in [
    'CREATE INDEX IF NOT EXISTS idx_claims_member ON claims(MEMBER_ID)',
    'CREATE INDEX IF NOT EXISTS idx_members_member ON members(MEMBER_ID)',
    'CREATE INDEX IF NOT EXISTS idx_encounters_member ON encounters(MEMBER_ID)',
    'CREATE INDEX IF NOT EXISTS idx_encounters_visit ON encounters(VISIT_TYPE)',
    'CREATE INDEX IF NOT EXISTS idx_claims_encounter ON claims(ENCOUNTER_ID)',
    'CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(CLAIM_STATUS)',
    'CREATE INDEX IF NOT EXISTS idx_claims_service ON claims(SERVICE_DATE)',
    'CREATE INDEX IF NOT EXISTS idx_claims_cpt ON claims(CPT_CODE)',
    'CREATE INDEX IF NOT EXISTS idx_members_dob ON members(DATE_OF_BIRTH)',
]:
    conn.execute(idx_sql)
conn.commit()
conn.close()

from semantic_sql_engine import SemanticSQLEngine
engine = SemanticSQLEngine(DB)

TESTS = [
    ('top 10 providers by volume', 'ranking'),
    ('specialty with highest average cost', 'ranking'),
    ('physicians in the Maryland Baltimore area', 'filter'),
    ('claims percentage by age group', 'breakdown'),
    ('claims percentage by gender', 'breakdown'),
    ('claims percentage by race', 'breakdown'),
    ('claims percentage by region', 'breakdown'),
    ('count where one person visited emergency more than 10 times', 'filter'),
    ('members who had more than 5 admissions', 'filter'),
    ('members with both emergency and inpatient visits', 'count'),
    ('emergency to outpatient ratio', 'aggregate'),
    ('compare inpatient vs outpatient costs', 'comparison'),
    ('paid denied adjusted breakdown', 'breakdown'),
    ('claims with prior authorization as percentage of total', 'percentage'),
    ('what percentage of claims are pending', 'percentage'),
    ('denied claim percentage', 'percentage'),
    ('high dollar claims over 10000', 'filter'),
    ('average paid amount per visit type', 'aggregate'),
    ('average processing time for claims', 'aggregate'),
    ('diabetes claims by region', 'breakdown'),
    ('COVID hospitalization by region', 'breakdown'),
    ('readmission rate', 'rate'),
    ('medicaid claims', 'filter'),
    ('pharmacy claims and billing', 'lookup'),
    ('CPT code descriptions', 'lookup'),
    ('most occurring CPT code in 2025', 'ranking'),
    ('upcoming PCP appointments for KP Baltimore', 'filter'),
    ('members older than 65', 'filter'),
    ('total members by plan type', 'breakdown'),
    ('top 5 facilities by claim volume', 'ranking'),
    ('denial rate by region', 'rate'),
]

class Timeout(Exception): pass
def alarm_handler(s, f): raise Timeout()
signal.signal(signal.SIGALRM, alarm_handler)

p = f = 0; failures = []
total_t = time.time()

for i, (q, exp) in enumerate(TESTS, 1):
    r = engine.generate(q)
    sql = r.get('sql', '')
    err = None; cnt = 0; exec_ms = 0

    try:
        conn = sqlite3.connect(DB)
        signal.alarm(8)
        t0 = time.time()
        cnt = conn.execute(f'SELECT COUNT(*) FROM ({sql.rstrip(";")})').fetchone()[0]
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
        err = str(e)[:60]
        try: conn.close()
        except: pass

    ok = True; iss = []
    if err: ok = False; iss.append(err)
    if not sql: ok = False; iss.append('NO_SQL')
    if cnt > 5000 and exp in ('ranking', 'breakdown', 'aggregate', 'percentage', 'rate', 'comparison'):
        ok = False; iss.append(f'ROWS:{cnt}')
    if 'more than' in q.lower() and ('member' in q.lower() or 'person' in q.lower()):
        if 'HAVING' not in sql.upper() and 'GROUP BY' not in sql.upper():
            ok = False; iss.append('NO_HAVING')
    if 'ratio' in q.lower() and ('CASE' not in sql.upper() and '/' not in sql):
        ok = False; iss.append('NO_RATIO')

    if ok:
        p += 1
        print(f'  OK [{i:2d}] {cnt:6d} rows {exec_ms:4d}ms | {q}')
    else:
        f += 1
        print(f'FAIL [{i:2d}] {cnt:6d} rows {exec_ms:4d}ms | {q} | {", ".join(iss)}')
        print(f'     SQL: {sql[:150]}')
        failures.append((q, iss))

total_ms = int((time.time() - total_t) * 1000)
print(f'\n{"=" * 70}')
print(f'RESULT: {p}/{len(TESTS)} PASS ({100 * p / len(TESTS):.0f}%) in {total_ms}ms')
print(f'{"=" * 70}')

if failures:
    print(f'\nFailed ({f}):')
    for q, iss in failures:
        print(f'  {q}: {", ".join(iss)}')
