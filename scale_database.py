#!/usr/bin/env python3
"""
Scale healthcare_demo.db from ~71K to ~4M rows.
Strategy: Exponential doubling with simple ID prefixing.
NO DATE() functions (they're slow). Just prefix IDs and vary amounts.
"""
import sqlite3, os, shutil, time, sys

DB_PATH = 'data/healthcare_demo.db'
BACKUP_PATH = 'data/healthcare_demo_71k_backup.db'

def main():
    if os.path.exists(BACKUP_PATH):
        print("Restoring from backup...", flush=True)
        shutil.copy2(BACKUP_PATH, DB_PATH)
    else:
        print("Backing up original...", flush=True)
        shutil.copy2(DB_PATH, BACKUP_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-100000")
    conn.execute("PRAGMA temp_store=MEMORY")
    cur = conn.cursor()
    start = time.time()

    def exp_double(table, id_col, target, amount_cols=None):
        """Exponentially double table. Only modify ID (prefix) and optionally amount cols."""
        initial = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if initial >= target:
            print(f"  {table}: already at {initial:,}")
            return

        cols = [d[0] for d in cur.execute(f"SELECT * FROM {table} LIMIT 0").description]
        current = initial
        step = 0

        print(f"  {table}: {initial:,} → {target:,}", end='', flush=True)

        while current < target:
            step += 1
            pfx = chr(64 + ((step - 1) % 26) + 1)  # A-Z cycling
            needed = min(current, target - current)

            # Build select: prefix the ID col, vary amount cols slightly
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

            if step % 3 == 0:
                print(f" → {current:,}", end='', flush=True)

        final = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f" ✓ {final:,} ({step} steps, {time.time()-start:.0f}s)")

    def exp_double_no_id(table, target, amount_cols=None):
        """For tables without unique ID."""
        initial = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if initial >= target:
            print(f"  {table}: already at {initial:,}")
            return

        cols = [d[0] for d in cur.execute(f"SELECT * FROM {table} LIMIT 0").description]
        current = initial
        step = 0

        print(f"  {table}: {initial:,} → {target:,}", end='', flush=True)

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

            if step % 3 == 0:
                print(f" → {current:,}", end='', flush=True)

        final = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f" ✓ {final:,} ({step} steps, {time.time()-start:.0f}s)")

    # ═══════════════════════════════════════════════════
    # Target: ~4.1M total rows
    # ═══════════════════════════════════════════════════

    exp_double('MEMBERS', 'MEMBER_ID', 200_000)
    exp_double('PROVIDERS', 'NPI', 6_000)
    exp_double('ENCOUNTERS', 'ENCOUNTER_ID', 600_000,
               amount_cols=['LENGTH_OF_STAY'])
    exp_double('CLAIMS', 'CLAIM_ID', 1_500_000,
               amount_cols=['BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'PAID_AMOUNT'])
    exp_double('DIAGNOSES', 'DIAGNOSIS_ID', 200_000)
    exp_double('PRESCRIPTIONS', 'RX_ID', 150_000,
               amount_cols=['COST', 'COPAY'])
    exp_double('REFERRALS', 'REFERRAL_ID', 60_000)
    exp_double('appointments', 'APPOINTMENT_ID', 120_000)
    exp_double_no_id('gpdm_member_month_fact', 1_200_000,
                     amount_cols=['paid_usd', 'billed_usd'])

    # ── Rebuild stats ──
    print("\nRebuilding statistics...", flush=True)
    cur.execute("ANALYZE")
    conn.commit()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DATABASE SCALING COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    total = 0
    for t in ['MEMBERS','CLAIMS','ENCOUNTERS','DIAGNOSES','PRESCRIPTIONS',
              'PROVIDERS','REFERRALS','appointments','gpdm_member_month_fact']:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total += cnt
        print(f"  {t:35s}: {cnt:>12,} rows")
    print(f"  {'TOTAL':35s}: {total:>12,} rows")
    sz = os.path.getsize(DB_PATH)
    print(f"\n  Database size: {sz/1024/1024:.1f} MB")
    print(f"  Original: {BACKUP_PATH} ({os.path.getsize(BACKUP_PATH)/1024/1024:.1f} MB)")

    # Integrity check
    print("\n  Integrity check:", conn.execute("PRAGMA integrity_check").fetchone()[0])
    conn.close()

if __name__ == '__main__':
    main()
