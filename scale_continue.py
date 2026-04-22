#!/usr/bin/env python3
"""Continue scaling from where it left off — DIAGNOSES onwards."""
import sqlite3, os, time

DB_PATH = 'data/healthcare_demo.db'

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-100000")
    conn.execute("PRAGMA temp_store=MEMORY")
    cur = conn.cursor()
    start = time.time()

    def exp_double(table, id_col, target, amount_cols=None):
        initial = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if initial >= target:
            print(f"  {table}: already at {initial:,}")
            return
        cols = [d[0] for d in cur.execute(f"SELECT * FROM {table} LIMIT 0").description]
        current = initial
        step = 0
        print(f"  {table}: {current:,} → {target:,}", end='', flush=True)
        while current < target:
            step += 1
            pfx = chr(64 + ((step - 1) % 26) + 1)
            needed = min(current, target - current)
            parts = []
            for c in cols:
                if c == id_col:
                    parts.append(f"'{pfx}{step}' || {c}")
                elif amount_cols and c in amount_cols:
                    mult = round(0.85 + (step % 8) * 0.04, 2)
                    parts.append(f"ROUND({c} * {mult}, 2)")
                else:
                    parts.append(c)
            sql = f"INSERT INTO {table} SELECT {', '.join(parts)} FROM {table} LIMIT {needed}"
            cur.execute(sql)
            conn.commit()
            current += needed
            print(f".", end='', flush=True)
        final = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f" ✓ {final:,}")

    def exp_double_no_id(table, target, amount_cols=None):
        initial = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if initial >= target:
            print(f"  {table}: already at {initial:,}")
            return
        cols = [d[0] for d in cur.execute(f"SELECT * FROM {table} LIMIT 0").description]
        current = initial
        step = 0
        print(f"  {table}: {current:,} → {target:,}", end='', flush=True)
        while current < target:
            step += 1
            needed = min(current, target - current)
            parts = []
            for c in cols:
                if amount_cols and c in amount_cols:
                    mult = round(0.85 + (step % 7) * 0.05, 2)
                    parts.append(f"ROUND({c} * {mult}, 2)")
                else:
                    parts.append(c)
            sql = f"INSERT INTO {table} SELECT {', '.join(parts)} FROM {table} LIMIT {needed}"
            cur.execute(sql)
            conn.commit()
            current += needed
            print(f".", end='', flush=True)
        final = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f" ✓ {final:,}")

    # Continue from where we left off
    exp_double('DIAGNOSES', 'DIAGNOSIS_ID', 200_000)
    exp_double('PRESCRIPTIONS', 'RX_ID', 150_000, amount_cols=['COST', 'COPAY'])
    exp_double('REFERRALS', 'REFERRAL_ID', 60_000)
    exp_double('appointments', 'APPOINTMENT_ID', 120_000)
    exp_double_no_id('gpdm_member_month_fact', 1_200_000, amount_cols=['paid_usd', 'billed_usd'])

    print("\nRebuilding statistics...", flush=True)
    cur.execute("ANALYZE")
    conn.commit()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SCALING CONTINUATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    total = 0
    for t in ['MEMBERS','CLAIMS','ENCOUNTERS','DIAGNOSES','PRESCRIPTIONS',
              'PROVIDERS','REFERRALS','appointments','gpdm_member_month_fact']:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total += cnt
        print(f"  {t:35s}: {cnt:>12,} rows")
    print(f"  {'TOTAL':35s}: {total:>12,} rows")
    print(f"\n  DB size: {os.path.getsize(DB_PATH)/1024/1024:.1f} MB")
    print(f"  Integrity: {conn.execute('PRAGMA integrity_check').fetchone()[0]}")
    conn.close()

if __name__ == '__main__':
    main()
