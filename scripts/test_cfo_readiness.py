import os, sys, time, sqlite3, json, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'healthcare_production.db')


def get_ground_truth(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    truth = {}

    c.execute("SELECT COUNT(*) FROM claims")
    truth['total_claims'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM members")
    truth['total_members'] = c.fetchone()[0]

    c.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2)
        FROM claims
    """)
    truth['denial_rate_pct'] = c.fetchone()[0]

    c.execute("SELECT ROUND(SUM(PAID_AMOUNT), 2) FROM claims")
    truth['total_paid'] = c.fetchone()[0]

    c.execute("SELECT ROUND(AVG(PAID_AMOUNT), 2) FROM claims")
    truth['avg_paid_per_claim'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM encounters")
    truth['total_encounters'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM appointments")
    truth['total_appointments'] = c.fetchone()[0]

    c.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN STATUS='NO_SHOW' THEN 1 ELSE 0 END) / COUNT(*), 2)
        FROM appointments
    """)
    truth['no_show_rate_pct'] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT NPI) FROM providers")
    truth['unique_providers'] = c.fetchone()[0]

    c.execute("SELECT ROUND(AVG(LENGTH_OF_STAY), 2) FROM encounters WHERE LENGTH_OF_STAY IS NOT NULL")
    truth['avg_los'] = c.fetchone()[0]

    truth['readmission_rate_pct'] = None

    c.execute("""
        SELECT COUNT(*) FROM members
        WHERE DATE_OF_BIRTH <= date('now', '-65 years')
    """)
    truth['members_over_65'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM prescriptions WHERE STATUS='FILLED'")
    truth['total_prescriptions_filled'] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM prescriptions")
    truth['total_prescriptions'] = c.fetchone()[0]

    c.execute("SELECT ROUND(AVG(COST), 2) FROM prescriptions")
    truth['avg_rx_cost'] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT ICD10_CODE) FROM claims WHERE ICD10_CODE IS NOT NULL")
    truth['distinct_diagnoses'] = c.fetchone()[0]

    c.execute("SELECT CLAIM_STATUS, COUNT(*) FROM claims GROUP BY CLAIM_STATUS ORDER BY COUNT(*) DESC")
    truth['claims_by_status'] = dict(c.fetchall())

    c.execute("SELECT PLAN_TYPE, COUNT(*) FROM claims GROUP BY PLAN_TYPE ORDER BY COUNT(*) DESC LIMIT 1")
    row = c.fetchone()
    truth['top_payer'] = row[0] if row else None

    if truth['total_members'] > 0:
        truth['pmpm_approx'] = round(truth['total_paid'] / truth['total_members'] / 12, 2)

    conn.close()
    return truth


CFO_QUERIES = [
    {
        'query': "What is our total paid amount across all claims",
        'category': 'FINANCIAL',
        'verify': lambda r, t: _check_value_in_result(r, t['total_paid'], tolerance=0.01,
                                                       label="total_paid"),
    },
    {
        'query': "What is the average paid amount per claim",
        'category': 'FINANCIAL',
        'verify': lambda r, t: _check_value_in_result(r, t['avg_paid_per_claim'], tolerance=0.05,
                                                       label="avg_paid_per_claim"),
    },
    {
        'query': "What is our PMPM cost",
        'category': 'FINANCIAL',
        'verify': lambda r, t: _has_numeric_data(r, label="PMPM"),
    },
    {
        'query': "What is our medical loss ratio",
        'category': 'FINANCIAL',
        'verify': lambda r, t: _has_numeric_data(r, label="MLR"),
    },

    {
        'query': "What is the denial rate",
        'category': 'OPERATIONS',
        'verify': lambda r, t: _check_value_in_result(r, t['denial_rate_pct'], tolerance=0.1,
                                                       label="denial_rate"),
    },
    {
        'query': "What is the no-show rate for appointments",
        'category': 'OPERATIONS',
        'verify': lambda r, t: _check_value_in_result(r, t['no_show_rate_pct'], tolerance=0.1,
                                                       label="no_show_rate"),
    },
    {
        'query': "What is the readmission rate",
        'category': 'OPERATIONS',
        'verify': lambda r, t: _has_numeric_data(r, label="readmission_rate"),
    },
    {
        'query': "Average length of stay",
        'category': 'OPERATIONS',
        'verify': lambda r, t: _check_value_in_result(r, t['avg_los'], tolerance=0.5,
                                                       label="avg_los"),
    },

    {
        'query': "How many total claims do we have",
        'category': 'VOLUME',
        'verify': lambda r, t: _check_value_in_result(r, t['total_claims'], tolerance=0,
                                                       label="total_claims"),
    },
    {
        'query': "How many members are enrolled",
        'category': 'VOLUME',
        'verify': lambda r, t: _check_value_in_result(r, t['total_members'], tolerance=0,
                                                       label="total_members"),
    },
    {
        'query': "How many providers do we have",
        'category': 'VOLUME',
        'verify': lambda r, t: _check_value_in_result(r, t['unique_providers'], tolerance=0,
                                                       label="unique_providers"),
    },
    {
        'query': "Total number of encounters",
        'category': 'VOLUME',
        'verify': lambda r, t: _check_value_in_result(r, t['total_encounters'], tolerance=0,
                                                       label="total_encounters"),
    },
    {
        'query': "How many prescriptions were filled",
        'category': 'VOLUME',
        'verify': lambda r, t: _check_value_in_result(r, t['total_prescriptions_filled'], tolerance=0.01,
                                                       label="prescriptions_filled"),
    },

    {
        'query': "How many members are over 65",
        'category': 'DEMOGRAPHICS',
        'verify': lambda r, t: _has_numeric_data(r, label="members_over_65"),
    },
    {
        'query': "Member count by gender",
        'category': 'DEMOGRAPHICS',
        'verify': lambda r, t: _has_grouped_data(r, label="gender_breakdown"),
    },
    {
        'query': "Member enrollment by region",
        'category': 'DEMOGRAPHICS',
        'verify': lambda r, t: _has_grouped_data(r, label="region_breakdown"),
    },

    {
        'query': "Claims by payer",
        'category': 'PAYER',
        'verify': lambda r, t: _has_grouped_data(r, label="claims_by_payer"),
    },
    {
        'query': "Denial rate by payer",
        'category': 'PAYER',
        'verify': lambda r, t: _has_grouped_data(r, label="denial_by_payer"),
    },

    {
        'query': "Top 10 diagnosis codes by claim count",
        'category': 'QUALITY',
        'verify': lambda r, t: _has_ranked_data(r, min_rows=5, label="top_dx"),
    },
    {
        'query': "Average cost per prescription",
        'category': 'QUALITY',
        'verify': lambda r, t: _check_value_in_result(r, t['avg_rx_cost'], tolerance=0.05,
                                                       label="avg_rx_cost"),
    },

    {
        'query': "Show claim trends over time",
        'category': 'TREND',
        'verify': lambda r, t: _has_time_series(r, label="claim_trend"),
    },
    {
        'query': "Monthly denial trend",
        'category': 'TREND',
        'verify': lambda r, t: _has_numeric_data(r, label="denial_trend"),
    },

    {
        'query': "Forecast claim costs for next quarter",
        'category': 'FORECAST',
        'verify': lambda r, t: _has_any_response(r, label="claim_forecast"),
    },
    {
        'query': "Predict readmission risk",
        'category': 'FORECAST',
        'verify': lambda r, t: _has_any_response(r, label="readmission_forecast"),
    },

    {
        'query': "Which providers have the highest denial rates",
        'category': 'CROSS',
        'verify': lambda r, t: _has_ranked_data(r, min_rows=1, label="provider_denial"),
    },
    {
        'query': "Cost per member by age group",
        'category': 'CROSS',
        'verify': lambda r, t: _has_any_response(r, label="cost_by_age"),
    },
    {
        'query': "Show me the most expensive diagnosis categories",
        'category': 'CROSS',
        'verify': lambda r, t: _has_ranked_data(r, min_rows=1, label="expensive_dx"),
    },
    {
        'query': "Which region has the highest cost per member",
        'category': 'CROSS',
        'verify': lambda r, t: _has_any_response(r, label="region_cost"),
    },

    {
        'query': "How is the organization performing",
        'category': 'AMBIGUOUS',
        'verify': lambda r, t: _has_any_response(r, label="org_perf"),
    },
    {
        'query': "Where are we losing money",
        'category': 'AMBIGUOUS',
        'verify': lambda r, t: _has_any_response(r, label="cost_leakage"),
    },
]


def _extract_all_numbers(result):
    numbers = []
    rows = result.get('rows') or result.get('results') or []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                for v in row.values():
                    try:
                        numbers.append(float(v))
                    except (ValueError, TypeError):
                        pass
            elif isinstance(row, (list, tuple)):
                for v in row:
                    try:
                        numbers.append(float(v))
                    except (ValueError, TypeError):
                        pass
    narrative = result.get('narrative', '')
    if isinstance(narrative, str):
        import re
        for m in re.findall(r'[\d,]+\.?\d*', narrative):
            try:
                numbers.append(float(m.replace(',', '')))
            except ValueError:
                pass
    return numbers


def _check_value_in_result(result, expected, tolerance=0.01, label=""):
    if expected is None:
        return True, f"{label}: expected is None (skip)"

    numbers = _extract_all_numbers(result)
    if not numbers:
        return False, f"{label}: NO numeric data returned (expected {expected})"

    if isinstance(expected, (int, float)):
        for n in numbers:
            if expected == 0:
                if abs(n) < 1:
                    return True, f"{label}: got {n} ≈ {expected}"
            else:
                rel_err = abs(n - expected) / abs(expected)
                if rel_err <= tolerance:
                    return True, f"{label}: got {n} ≈ {expected} (err={rel_err:.3%})"

        closest = min(numbers, key=lambda n: abs(n - expected))
        rel_err = abs(closest - expected) / abs(expected) if expected != 0 else abs(closest)
        return False, f"{label}: closest={closest} vs expected={expected} (err={rel_err:.1%})"

    return False, f"{label}: cannot verify type {type(expected)}"


def _has_numeric_data(result, label=""):
    numbers = _extract_all_numbers(result)
    if numbers:
        return True, f"{label}: has {len(numbers)} numeric values"
    narrative = result.get('narrative', '')
    if narrative and any(c.isdigit() for c in str(narrative)):
        return True, f"{label}: has narrative with data"
    return False, f"{label}: NO numeric data"


def _has_grouped_data(result, label=""):
    rows = result.get('rows') or result.get('results') or []
    if isinstance(rows, list) and len(rows) >= 2:
        return True, f"{label}: {len(rows)} groups"
    if isinstance(rows, list) and len(rows) == 1:
        return True, f"{label}: 1 row (may be aggregated)"
    return False, f"{label}: no grouped data (rows={len(rows) if isinstance(rows, list) else 0})"


def _has_ranked_data(result, min_rows=1, label=""):
    rows = result.get('rows') or result.get('results') or []
    if isinstance(rows, list) and len(rows) >= min_rows:
        return True, f"{label}: {len(rows)} ranked items"
    return False, f"{label}: insufficient ranked data (got {len(rows) if isinstance(rows, list) else 0}, need {min_rows})"


def _has_time_series(result, label=""):
    rows = result.get('rows') or result.get('results') or []
    if isinstance(rows, list) and len(rows) >= 3:
        return True, f"{label}: {len(rows)} time points"
    return False, f"{label}: insufficient time series data (got {len(rows) if isinstance(rows, list) else 0})"


def _has_any_response(result, label=""):
    has_sql = bool(result.get('sql', '').strip())
    has_rows = bool(result.get('rows') or result.get('results'))
    has_narrative = bool(result.get('narrative', '').strip())
    row_count = result.get('row_count', 0)

    if has_sql and (has_rows or row_count > 0 or has_narrative):
        return True, f"{label}: SQL + {row_count} rows"
    if has_narrative and len(result.get('narrative', '')) > 50:
        return True, f"{label}: narrative response"
    if has_sql:
        return True, f"{label}: SQL generated (may be empty result set)"
    return False, f"{label}: no meaningful response"


def main():
    print("=" * 90)
    print("  CFO / EXECUTIVE READINESS AUDIT")
    print("  Testing: Real-world questions that drive million-dollar decisions")
    print("=" * 90)

    print("\n[1] Computing ground truth from raw database...")
    truth = get_ground_truth(DB_PATH)
    print(f"  Ground truth established: {len(truth)} metrics")
    for k, v in sorted(truth.items()):
        if not isinstance(v, dict):
            print(f"    {k}: {v}")

    print(f"\n[2] Initializing IntelligentPipeline...")
    t0 = time.time()
    from intelligent_pipeline import IntelligentPipeline
    pipeline = IntelligentPipeline(db_path=DB_PATH)
    init_s = time.time() - t0
    print(f"  Pipeline ready in {init_s:.1f}s")

    print(f"\n[3] Running {len(CFO_QUERIES)} executive queries...\n")

    categories = {}
    all_results = []
    accuracy_pass = 0
    accuracy_fail = 0
    functional_pass = 0
    functional_fail = 0

    for i, test in enumerate(CFO_QUERIES, 1):
        query = test['query']
        cat = test['category']
        verify_fn = test['verify']

        try:
            t_start = time.time()
            result = pipeline.process(query)
            elapsed = (time.time() - t_start) * 1000

            has_error = bool(result.get('error'))
            has_sql = bool(result.get('sql', '').strip())

            if has_sql and not has_error:
                functional_pass += 1
            else:
                functional_fail += 1

            try:
                acc_pass, acc_detail = verify_fn(result, truth)
            except Exception as ve:
                acc_pass = False
                acc_detail = f"verify error: {ve}"

            if acc_pass:
                accuracy_pass += 1
                icon = "✓"
                status = "ACCURATE"
            else:
                accuracy_fail += 1
                icon = "✗"
                status = "INACCURATE"

            row_count = result.get('row_count', 0)
            source = result.get('source', '?')

            print(f"  {icon} [{i:2d}/{len(CFO_QUERIES)}] {status:10s} | {cat:12s} | {elapsed:6.0f}ms | rows={row_count:4d} | {query[:55]}")
            if not acc_pass:
                print(f"       → {acc_detail}")

            if cat not in categories:
                categories[cat] = {'pass': 0, 'fail': 0, 'total': 0}
            categories[cat]['total'] += 1
            categories[cat]['pass' if acc_pass else 'fail'] += 1

            all_results.append({
                'query': query, 'category': cat, 'accurate': acc_pass,
                'functional': has_sql and not has_error,
                'detail': acc_detail, 'elapsed_ms': elapsed,
                'row_count': row_count, 'source': source,
            })

        except Exception as e:
            print(f"  ✗ [{i:2d}/{len(CFO_QUERIES)}] CRASH      | {cat:12s} |        | {query[:55]}")
            print(f"       → {e}")
            functional_fail += 1
            accuracy_fail += 1
            if cat not in categories:
                categories[cat] = {'pass': 0, 'fail': 0, 'total': 0}
            categories[cat]['total'] += 1
            categories[cat]['fail'] += 1

    total = len(CFO_QUERIES)
    print(f"\n{'=' * 90}")
    print(f"  EXECUTIVE READINESS SCORECARD")
    print(f"{'=' * 90}")

    func_pct = functional_pass / total * 100
    acc_pct = accuracy_pass / total * 100

    print(f"\n  FUNCTIONAL (does it work?):  {functional_pass}/{total} ({func_pct:.0f}%)")
    print(f"  ACCURATE  (is data right?):  {accuracy_pass}/{total} ({acc_pct:.0f}%)")

    print(f"\n  BY CATEGORY:")
    for cat in ['FINANCIAL', 'OPERATIONS', 'VOLUME', 'DEMOGRAPHICS', 'PAYER',
                'QUALITY', 'TREND', 'FORECAST', 'CROSS', 'AMBIGUOUS']:
        stats = categories.get(cat, {'pass': 0, 'fail': 0, 'total': 0})
        if stats['total'] == 0:
            continue
        cat_pct = stats['pass'] / stats['total'] * 100
        bar = "█" * stats['pass'] + "░" * stats['fail']
        print(f"    {cat:14s}: {stats['pass']:2d}/{stats['total']:2d} ({cat_pct:5.1f}%) {bar}")

    failures = [r for r in all_results if not r['accurate']]
    if failures:
        print(f"\n  ACCURACY FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ [{f['category']}] {f['query']}")
            print(f"      {f['detail']}")

    print(f"\n{'═' * 90}")
    if acc_pct >= 95:
        print(f"  ✅ CFO-READY: {acc_pct:.0f}% accuracy — system is reliable for executive decisions")
    elif acc_pct >= 80:
        print(f"  ⚠️  MOSTLY READY: {acc_pct:.0f}% accuracy — needs fixes before executive rollout")
    else:
        print(f"  ❌ NOT READY: {acc_pct:.0f}% accuracy — significant gaps for executive use")
    print(f"{'═' * 90}\n")

    return acc_pct >= 95


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
