import os, sys, time, traceback, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'healthcare_production.db')


JARGON_QUERIES = [
    ("what is our PMPM cost",                        "finance jargon"),
    ("show denial rate by payer",                    "insurance jargon"),
    ("average LOS for inpatient admissions",         "clinical jargon"),
    ("how many ER visits last quarter",              "patient jargon"),
    ("what is our MLR",                              "finance metric"),
    ("show readmission rates by diagnosis",          "clinical metric"),
    ("copay distribution across plans",              "patient finance"),
    ("prior auth approval rate",                     "insurance process"),
    ("capitation payments by region",                "insurance finance"),
    ("generic substitution rate for prescriptions",  "pharmacy metric"),
]

ENSEMBLE_QUERIES = [
    ("total claims amount",                          "aggregate"),
    ("how many members are enrolled",                "count"),
    ("top 5 providers by revenue",                   "rank"),
    ("denial rate by month",                         "trend"),
    ("compare male vs female claim counts",          "compare"),
    ("average cost per visit",                       "aggregate"),
    ("which diagnosis has the highest cost",         "rank"),
    ("show claim trends over time",                  "trend"),
    ("count of appointments by status",              "count"),
    ("what percentage of claims are denied",         "rate"),
]

COMPLEX_QUERIES = [
    ("how is our network performing",                                 "ambiguous"),
    ("why are denials increasing",                                    "causal"),
    ("show me everything about diabetes",                             "broad"),
    ("which region has the worst outcomes and why",                    "multi-hop"),
    ("compare Q1 vs Q2 financial performance",                        "temporal compare"),
    ("are we meeting quality benchmarks",                             "benchmark"),
    ("what should we focus on to reduce costs",                       "strategic"),
    ("show claims for patients over 65 with chronic conditions",      "multi-filter"),
    ("top denied procedures and their appeal success rate",           "cross-table"),
    ("monthly trend of ER visits vs primary care visits",             "multi-series"),
]

REGRESSION_QUERIES = [
    ("show all claims",                              "basic"),
    ("total revenue by provider",                    "aggregate"),
    ("list of denied claims",                        "filter"),
    ("average claim amount by diagnosis",            "group aggregate"),
    ("how many appointments were no-shows",          "count filter"),
    ("show member enrollment by month",              "trend"),
    ("top 10 diagnoses by frequency",                "rank"),
    ("what is the denial rate",                      "rate"),
    ("claims over $10000",                           "numeric filter"),
    ("show me provider performance summary",         "summary"),
]

ALL_QUERIES = (
    [("JARGON",    q, cat) for q, cat in JARGON_QUERIES] +
    [("ENSEMBLE",  q, cat) for q, cat in ENSEMBLE_QUERIES] +
    [("COMPLEX",   q, cat) for q, cat in COMPLEX_QUERIES] +
    [("REGRESSION",q, cat) for q, cat in REGRESSION_QUERIES]
)


def main():
    print("=" * 80)
    print("  END-TO-END PIPELINE INTEGRATION TEST")
    print("  Testing: IntelligentPipeline + DomainIntelligence + HealthcareTransformer")
    print("=" * 80)

    print("\n[1/4] Initializing IntelligentPipeline...")
    t0 = time.time()

    from intelligent_pipeline import IntelligentPipeline
    pipeline = IntelligentPipeline(db_path=DB_PATH)

    init_time = time.time() - t0
    print(f"  Pipeline initialized in {init_time:.1f}s")

    modules = {
        'semantic_graph':         pipeline.semantic_graph is not None,
        'intent_parser':          pipeline.intent_parser is not None,
        'sql_constructor':        pipeline.sql_constructor is not None,
        'domain_intelligence':    pipeline.domain_intelligence is not None,
        'healthcare_transformer': pipeline.healthcare_transformer is not None,
        'gpdm_definitions':      getattr(pipeline, 'gpdm_definitions', None) is not None,
        'predictive_engine':     getattr(pipeline, 'predictive_engine', None) is not None,
    }

    print("\n  Module Status:")
    for mod, active in modules.items():
        status = "ACTIVE" if active else "MISSING"
        icon = "✓" if active else "✗"
        print(f"    {icon} {mod}: {status}")

    critical_modules = ['semantic_graph', 'intent_parser', 'sql_constructor',
                        'domain_intelligence', 'healthcare_transformer']
    missing_critical = [m for m in critical_modules if not modules.get(m)]
    if missing_critical:
        print(f"\n  ⚠ CRITICAL MODULES MISSING: {missing_critical}")
        print("  Continuing test anyway to see what works...\n")
    else:
        print("\n  All critical modules loaded successfully!\n")

    print("[2/4] Running queries through full pipeline...\n")

    results_by_category = {}
    all_results = []
    ensemble_details = []
    domain_details = []

    for i, (category, query, subcategory) in enumerate(ALL_QUERIES, 1):
        try:
            t_start = time.time()
            result = pipeline.process(query)
            elapsed = (time.time() - t_start) * 1000

            has_sql = result.get('sql') and result['sql'].strip()
            has_rows = result.get('rows') is not None or result.get('results') is not None
            has_error = bool(result.get('error'))
            row_count = result.get('row_count', 0)

            passed = has_sql and not has_error

            t_insight = result.get('transformer_insight', {})
            has_ensemble = bool(t_insight)
            ensemble_agreed = t_insight.get('agreement', False) if has_ensemble else False
            ensemble_boosted = result.get('confidence', {}).get('ensemble_boost', False) if isinstance(result.get('confidence'), dict) else False

            status = "PASS" if passed else "FAIL"
            icon = "✓" if passed else "✗"

            entry = {
                'category': category,
                'subcategory': subcategory,
                'query': query,
                'passed': passed,
                'has_sql': has_sql,
                'has_rows': has_rows,
                'row_count': row_count,
                'has_error': has_error,
                'error': result.get('error', ''),
                'elapsed_ms': elapsed,
                'source': result.get('source', 'unknown'),
                'has_ensemble': has_ensemble,
                'ensemble_agreed': ensemble_agreed,
                'ensemble_boosted': ensemble_boosted,
                'intent': result.get('intent_parsed', {}).get('intent', ''),
                'transformer_intent': t_insight.get('intent', ''),
                'confidence': result.get('confidence_overall', 0),
            }
            all_results.append(entry)

            if category not in results_by_category:
                results_by_category[category] = {'pass': 0, 'fail': 0, 'total': 0}
            results_by_category[category]['total'] += 1
            results_by_category[category]['pass' if passed else 'fail'] += 1

            if has_ensemble:
                ensemble_details.append(entry)

            print(f"  {icon} [{i:2d}/{len(ALL_QUERIES)}] {status} | {category:10s} | {subcategory:18s} | {elapsed:6.0f}ms | rows={row_count:4d} | {query[:50]}")

            if not passed:
                print(f"       ERROR: {result.get('error', 'unknown')[:100]}")

        except Exception as e:
            print(f"  ✗ [{i:2d}/{len(ALL_QUERIES)}] CRASH | {category:10s} | {subcategory:18s} | {query[:50]}")
            print(f"       EXCEPTION: {e}")
            traceback.print_exc()
            entry = {
                'category': category, 'subcategory': subcategory, 'query': query,
                'passed': False, 'error': str(e), 'has_ensemble': False,
            }
            all_results.append(entry)
            if category not in results_by_category:
                results_by_category[category] = {'pass': 0, 'fail': 0, 'total': 0}
            results_by_category[category]['total'] += 1
            results_by_category[category]['fail'] += 1

    print("\n" + "=" * 80)
    print("[3/4] RESULTS SUMMARY")
    print("=" * 80)

    total_pass = sum(r['passed'] for r in all_results)
    total_fail = len(all_results) - total_pass
    total = len(all_results)
    pct = (total_pass / total * 100) if total > 0 else 0

    print(f"\n  OVERALL: {total_pass}/{total} PASS ({pct:.1f}%)")
    print(f"  {'─' * 60}")

    for cat in ['JARGON', 'ENSEMBLE', 'COMPLEX', 'REGRESSION']:
        stats = results_by_category.get(cat, {'pass': 0, 'fail': 0, 'total': 0})
        cat_pct = (stats['pass'] / stats['total'] * 100) if stats['total'] > 0 else 0
        bar = "█" * stats['pass'] + "░" * stats['fail']
        print(f"    {cat:12s}: {stats['pass']:2d}/{stats['total']:2d} ({cat_pct:5.1f}%) {bar}")

    print(f"\n  TRANSFORMER ENSEMBLE ANALYSIS:")
    print(f"  {'─' * 60}")
    queries_with_ensemble = [e for e in all_results if e.get('has_ensemble')]
    queries_agreed = [e for e in queries_with_ensemble if e.get('ensemble_agreed')]
    queries_boosted = [e for e in queries_with_ensemble if e.get('ensemble_boosted')]

    print(f"    Queries with transformer insight:  {len(queries_with_ensemble)}/{total}")
    print(f"    Transformer-NLU agreement:         {len(queries_agreed)}/{len(queries_with_ensemble)}")
    print(f"    Ensemble confidence boosted:        {len(queries_boosted)}/{len(queries_with_ensemble)}")

    if queries_with_ensemble:
        print(f"\n    Intent agreement details:")
        for e in queries_with_ensemble:
            agree_icon = "✓" if e.get('ensemble_agreed') else "✗"
            boost_icon = "⬆" if e.get('ensemble_boosted') else " "
            print(f"      {agree_icon}{boost_icon} NLU={e.get('intent','?'):10s} TFM={e.get('transformer_intent','?'):10s} | {e['query'][:50]}")

    failed = [r for r in all_results if not r['passed']]
    if failed:
        print(f"\n  FAILED QUERIES ({len(failed)}):")
        print(f"  {'─' * 60}")
        for f in failed:
            print(f"    ✗ [{f['category']}] {f['query']}")
            print(f"      Error: {f.get('error', 'unknown')[:120]}")

    print(f"\n{'=' * 80}")
    print("[4/4] DOMAIN INTELLIGENCE VERIFICATION")
    print("=" * 80)

    if pipeline.domain_intelligence:
        di = pipeline.domain_intelligence

        jargon_tests = [
            ("PMPM", True),
            ("MLR", True),
            ("DRG", True),
            ("copay", True),
            ("prior auth", True),
            ("LOS", True),
            ("ICD-10", True),
            ("capitation", True),
            ("EOB", True),
            ("RAF", True),
        ]

        print(f"\n  Jargon Resolution Test:")
        jargon_pass = 0
        for term, expected in jargon_tests:
            analysis = di.analyze_question(f"show me {term} data")
            recognized = False
            if isinstance(analysis, dict):
                norm_q = analysis.get('normalized_question', '')
                entities = analysis.get('entities', [])
                jargon = analysis.get('jargon_resolved', {})
                recognized = bool(jargon) or term.lower() in norm_q.lower() or len(entities) > 0

            icon = "✓" if recognized else "✗"
            jargon_pass += 1 if recognized else 0
            print(f"    {icon} '{term}' → recognized={recognized}")

        print(f"\n  Jargon Recognition: {jargon_pass}/{len(jargon_tests)} ({jargon_pass/len(jargon_tests)*100:.0f}%)")

        print(f"\n  Concept Ontology Test:")
        concept_tests = [
            "why are claims being denied",
            "what drives readmission rates",
            "how can we reduce no-shows",
        ]
        for q in concept_tests:
            analysis = di.analyze_question(q)
            concepts = analysis.get('concepts_activated', []) if isinstance(analysis, dict) else []
            print(f"    → '{q[:45]}' activated concepts: {concepts}")
    else:
        print("\n  ⚠ Domain Intelligence not loaded — skipping verification")

    print(f"\n{'═' * 80}")
    if total_pass == total:
        print(f"  ✅ ALL {total} QUERIES PASSED — Pipeline integration verified!")
    elif pct >= 95:
        print(f"  ⚠ {total_pass}/{total} PASSED ({pct:.1f}%) — Minor issues to address")
    else:
        print(f"  ❌ {total_pass}/{total} PASSED ({pct:.1f}%) — Significant regressions detected")

    active_count = sum(1 for v in modules.values() if v)
    print(f"  Modules: {active_count}/{len(modules)} active")
    print(f"  Ensemble: {len(queries_with_ensemble)} queries enriched, {len(queries_agreed)} agreed, {len(queries_boosted)} boosted")
    print(f"  Init time: {init_time:.1f}s")
    print(f"{'═' * 80}\n")

    return total_pass == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
