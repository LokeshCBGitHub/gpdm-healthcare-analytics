import sqlite3, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

db = '../data/ecommerce_test.db'
if not os.path.exists(db):
    print(f"DB not found at {db}")
    sys.exit(1)

from semantic_sql_engine import SemanticSQLEngine
t0 = time.time()
engine = SemanticSQLEngine(db)
print(f"Engine init: {time.time()-t0:.1f}s\n")

QUERIES = [
    "how many orders are there",
    "count of customers",
    "total sales by category",
    "top 10 products by revenue",
    "average order amount by country",
    "orders by status",
    "monthly order trends",
    "which brand has the most products",
    "customers by loyalty tier",
    "what percentage of orders are cancelled",
    "products with stock less than 10",
    "top 5 customers by total spending",
    "average rating by brand",
    "orders greater than 500",
    "revenue by shipping method",
    "customers older than 60",
    "products by category",
    "order count by month",
    "suppliers by country",
    "cancelled orders by city",
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
        if ok:
            print(f"       SQL: {sql[:120]}")
            print(f"       Sample: {rows[0]}")
    except Exception as e:
        print(f"[{fb}] FAIL         | {q}")
        print(f"       SQL: {sql[:120]}")
        print(f"       Error: {e}")

conn.close()
print(f"\nResult: {passed}/{len(QUERIES)} passed")
print(f"Total time: {time.time()-t0:.1f}s")
