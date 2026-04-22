import sqlite3, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

db = '../data/healthcare_production.db'
if not os.path.exists(db):
    print(f"DB not found at {db}")
    sys.exit(1)

from semantic_sql_engine import SemanticSQLEngine
t0 = time.time()
engine = SemanticSQLEngine(db)
print(f"Engine init: {time.time()-t0:.1f}s\n")

QUERIES = [
    "how many claims are there",
    "count of members",
    "average paid amount by region",
    "total billed amount by claim type",
    "top 10 providers by volume",
    "which specialty has the most claims",
    "claims by type",
    "members by age group",
    "claims trend over time",
    "denial rate by region",
    "what percentage of claims are denied",
    "members older than 65",
    "claims greater than 5000",
    "show me upcoming PCP appointments",
    "what is CPT code 99213 description",
    "COVID hospitalizations by region",
    "general physicians by location",
    "claims breakdown by member demographics",
    "average processing time for claims",
    "monthly admission count",
]

conn = sqlite3.connect(db)
passed = 0
for q in QUERIES:
    r = engine.generate(q)
    sql = r['sql']
    fb = "FB" if r.get('used_fallback') else "SM"
    try:
        rows = conn.execute(sql).fetchall()
        n = len(rows)
        ok = n > 0
        tag = "OK" if ok else "EMPTY"
        if ok:
            passed += 1
        print(f"[{fb}] {tag:>5} ({n:>5}) | {q}")
    except Exception as e:
        print(f"[{fb}] FAIL         | {q}: {e}")

conn.close()
print(f"\nResult: {passed}/{len(QUERIES)} passed")
print(f"Total time: {time.time()-t0:.1f}s")
