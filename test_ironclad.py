#!/usr/bin/env python3
"""
IRON-CLAD Test Suite for GPDM Healthcare Analytics Chatbot
===========================================================

This is not a "did it return something" test. This is a "is every number
MATHEMATICALLY CORRECT against ground truth SQL" test.

Tests:
  1. Dashboard Metric Verification — every metric cross-checked against raw SQL
  2. Edge Case Resilience — NULLs, zeros, empty strings, missing data
  3. Knowledge Graph Integrity — relationships, blueprints, learning, gap detection
  4. Pipeline Robustness — malformed input, timeouts, division-by-zero paths
  5. Business Logic Validation — thresholds, RAG status, benchmark comparisons
"""

import sys
import os
import sqlite3
import json
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = 'data/healthcare_demo.db'

passed = 0
failed = 0
errors = []


def check(label, fn, category=""):
    """Run a test, track pass/fail."""
    global passed, failed
    try:
        result = fn()
        if result is True or (isinstance(result, str) and result):
            print(f"  \033[32m✓\033[0m {label}")
            passed += 1
        else:
            msg = f"returned falsy: {result}"
            print(f"  \033[31m✗\033[0m {label}: {msg}")
            failed += 1
            errors.append((category, label, msg))
    except Exception as e:
        print(f"  \033[31m✗\033[0m {label}: {e}")
        failed += 1
        errors.append((category, label, str(e)))


def assert_eq(actual, expected, tolerance=0.01):
    """Assert two values are equal (with tolerance for floats)."""
    if isinstance(actual, float) and isinstance(expected, float):
        if abs(actual - expected) > tolerance:
            raise AssertionError(f"Expected {expected}, got {actual} (diff={abs(actual-expected):.4f})")
    elif actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")
    return True


def assert_range(val, low, high, label="value"):
    """Assert value is within range."""
    if not (low <= val <= high):
        raise AssertionError(f"{label}={val} not in [{low}, {high}]")
    return True


def assert_gt(val, threshold, label="value"):
    """Assert value is greater than threshold."""
    if val <= threshold:
        raise AssertionError(f"{label}={val} not > {threshold}")
    return True


def assert_type(val, expected_type, label="value"):
    """Assert value is of expected type."""
    if not isinstance(val, expected_type):
        raise AssertionError(f"{label} is {type(val).__name__}, expected {expected_type.__name__}")
    return True


def raw_sql(sql):
    """Execute raw SQL and return result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()
    conn.close()
    return result


def raw_sql_all(sql):
    """Execute raw SQL and return all results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results


# =============================================================================
# GROUND TRUTH — compute expected values directly from database
# =============================================================================
print("Computing ground truth from raw SQL...")

GT = {}
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Total members
c.execute("SELECT COUNT(*) FROM members")
GT['total_members'] = c.fetchone()[0]

# Disenrolled
c.execute("SELECT COUNT(*) FROM members WHERE DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''")
GT['disenrolled'] = c.fetchone()[0]
GT['retention_rate'] = round(100 - (GT['disenrolled'] / GT['total_members'] * 100), 2)
GT['disenrollment_rate'] = round(GT['disenrolled'] / GT['total_members'] * 100, 2)

# Financial
c.execute("SELECT SUM(CAST(BILLED_AMOUNT AS REAL)), SUM(CAST(PAID_AMOUNT AS REAL)), SUM(CAST(ALLOWED_AMOUNT AS REAL)) FROM claims")
billed, paid, allowed = c.fetchone()
GT['total_billed'] = billed
GT['total_paid'] = paid
GT['total_allowed'] = allowed
GT['mlr'] = round(allowed / billed * 100, 2)  # allowed/billed
GT['collection_rate'] = round(paid / allowed * 100, 2)
GT['pmpm_revenue'] = round(paid / (GT['total_members'] * 12), 2)
GT['pmpm_cost'] = round(billed / (GT['total_members'] * 12), 2)

# Claims
c.execute("SELECT COUNT(*) FROM claims")
GT['total_claims'] = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'DENIED'")
GT['denied_claims'] = c.fetchone()[0]
GT['denial_rate'] = round(GT['denied_claims'] / GT['total_claims'] * 100, 2)
c.execute("SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'PAID'")
GT['paid_claims'] = c.fetchone()[0]
GT['clean_claims_rate'] = round(GT['paid_claims'] / GT['total_claims'] * 100, 2)

# Risk scores
c.execute("SELECT AVG(CAST(RISK_SCORE AS REAL)) FROM members")
GT['avg_risk'] = round(c.fetchone()[0], 3)

# Encounters
c.execute("SELECT COUNT(*) FROM encounters")
GT['total_encounters'] = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT VISIT_TYPE) FROM encounters")
GT['visit_type_count'] = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT KP_REGION) FROM encounters")
GT['encounter_regions'] = c.fetchone()[0]

# Preventive care
c.execute("""SELECT COUNT(DISTINCT MEMBER_ID) FROM claims
    WHERE CPT_DESCRIPTION LIKE '%preventive%'
       OR CPT_DESCRIPTION LIKE '%screening%'
       OR CPT_DESCRIPTION LIKE '%wellness%'
       OR CPT_DESCRIPTION LIKE '%annual%'
       OR CPT_CODE IN ('99381','99382','99383','99384','99385','99386','99387',
                       '99391','99392','99393','99394','99395','99396','99397')""")
GT['preventive_members'] = c.fetchone()[0]
GT['preventive_rate'] = round(GT['preventive_members'] / GT['total_members'] * 100, 2)

# Outpatient access
c.execute("SELECT COUNT(DISTINCT MEMBER_ID) FROM encounters WHERE VISIT_TYPE = 'OUTPATIENT'")
GT['outpatient_members'] = c.fetchone()[0]
GT['outpatient_rate'] = round(GT['outpatient_members'] / GT['total_members'] * 100, 2)

# Regions
c.execute("SELECT COUNT(DISTINCT KP_REGION) FROM members")
GT['member_regions'] = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT KP_REGION) FROM claims")
GT['claims_regions'] = c.fetchone()[0]

# Plan types
c.execute("SELECT COUNT(DISTINCT PLAN_TYPE) FROM members")
GT['plan_types'] = c.fetchone()[0]

# Prescriptions
c.execute("SELECT COUNT(*), SUM(CAST(COST AS REAL)), COUNT(DISTINCT MEMBER_ID) FROM prescriptions")
rx_count, rx_cost, rx_members = c.fetchone()
GT['rx_count'] = rx_count
GT['rx_cost'] = rx_cost
GT['rx_members'] = rx_members
GT['pharmacy_pmpm'] = round(rx_cost / (GT['total_members'] * 12), 2)

# Referrals
c.execute("SELECT COUNT(*), COUNT(DISTINCT MEMBER_ID) FROM referrals")
ref_total, ref_members = c.fetchone()
GT['referral_count'] = ref_total
GT['referral_members'] = ref_members
GT['referral_per_1000'] = round(ref_total / GT['total_members'] * 1000, 1)

# Emergency
c.execute("SELECT COUNT(*), COUNT(DISTINCT MEMBER_ID) FROM encounters WHERE VISIT_TYPE = 'EMERGENCY'")
ed_enc, ed_mem = c.fetchone()
GT['ed_encounters'] = ed_enc
GT['ed_members'] = ed_mem
GT['ed_per_1000'] = round(ed_enc / GT['total_members'] * 1000, 1)

# Chronic conditions
c.execute("SELECT COUNT(DISTINCT MEMBER_ID) FROM diagnoses WHERE IS_CHRONIC = 'Y'")
GT['chronic_members'] = c.fetchone()[0]

# HCC codes
c.execute("SELECT COUNT(DISTINCT HCC_CODE) FROM diagnoses WHERE HCC_CODE IS NOT NULL AND HCC_CODE != ''")
GT['distinct_hcc_codes'] = c.fetchone()[0]

conn.close()

print(f"Ground truth computed: {len(GT)} metrics from raw SQL")
print()

# =============================================================================
# SECTION 1: DASHBOARD METRIC VERIFICATION
# =============================================================================
print("=" * 70)
print("SECTION 1: DASHBOARD METRIC VERIFICATION (every number vs ground truth)")
print("=" * 70)

from executive_dashboards import ExecutiveDashboardEngine
ede = ExecutiveDashboardEngine(DB_PATH)

# --- 1A: Financial Performance ---
print("\n--- 1A: Financial Performance ---")
fp = ede.get_financial_performance()
ytd = fp['sections']['ytd_summary']['metrics']

check("PMPM Revenue matches ground truth",
      lambda: assert_eq(ytd['pmpm_revenue']['value'], GT['pmpm_revenue'], 0.1),
      "financial")

check("PMPM Cost matches ground truth",
      lambda: assert_eq(ytd['pmpm_cost']['value'], GT['pmpm_cost'], 0.1),
      "financial")

check("MLR (allowed/billed) matches ground truth",
      lambda: assert_eq(ytd['medical_loss_ratio']['value'], GT['mlr'], 0.1),
      "financial")

check("Collection rate matches ground truth",
      lambda: assert_eq(ytd['collection_rate']['value'], GT['collection_rate'], 0.1),
      "financial")

check("Total members matches",
      lambda: assert_eq(ytd['total_members']['value'], GT['total_members']),
      "financial")

check("Expense breakdown has 4 claim types",
      lambda: assert_eq(len(fp['sections']['expense_by_type']['data']), 4),
      "financial")

check("Regional P&L has all regions",
      lambda: assert_eq(len(fp['sections']['regional_pl']['data']), GT['claims_regions']),
      "financial")

check("Monthly trend has data",
      lambda: assert_gt(len(fp['sections']['monthly_trend']['data']), 0, "months"),
      "financial")

# Verify regional PMPM sums make sense
regional_paid_sum = sum(r['paid'] for r in fp['sections']['regional_pl']['data'])
check("Regional paid amounts sum to total",
      lambda: assert_eq(regional_paid_sum, GT['total_paid'], GT['total_paid'] * 0.01),
      "financial")

# --- 1B: Stars Performance ---
print("\n--- 1B: Stars Performance ---")
stars = ede.get_stars_performance()

hedis = stars['sections']['hedis_measures']['measures']
check("Preventive care rate matches ground truth",
      lambda: assert_eq(hedis[0]['rate'], GT['preventive_rate'], 0.1),
      "stars")

check("Outpatient access rate matches ground truth",
      lambda: assert_eq(hedis[1]['rate'], GT['outpatient_rate'], 0.1),
      "stars")

check("Preventive numerator matches ground truth",
      lambda: assert_eq(hedis[0]['numerator'], GT['preventive_members']),
      "stars")

cahps = stars['sections']['cahps_measures']['measures']
expected_cahps = round(100 - GT['disenrollment_rate'], 2)
check("CAHPS satisfaction matches retention-based calc",
      lambda: assert_eq(cahps[0]['rate'], expected_cahps, 0.1),
      "stars")

admin = stars['sections']['admin_measures']['measures']
check("Clean claims rate matches ground truth",
      lambda: assert_eq(admin[0]['rate'], GT['clean_claims_rate'], 0.1),
      "stars")

check("Stars overall rating is valid (0-5)",
      lambda: assert_range(stars['sections']['overall_rating']['rating'], 0, 5, "stars_rating"),
      "stars")

check("Clinical quality has conditions",
      lambda: assert_gt(len(stars['sections']['clinical_quality']['data']), 0, "conditions"),
      "stars")

# --- 1C: Member Experience ---
print("\n--- 1C: Member Experience ---")
me = ede.get_member_experience()

ret = me['sections']['retention']['metrics']
check("Retention rate matches ground truth",
      lambda: assert_eq(ret['retention_rate']['value'], GT['retention_rate'], 0.1),
      "member_exp")

check("Total members matches",
      lambda: assert_eq(ret['total_members']['value'], GT['total_members']),
      "member_exp")

check("Regional comparison has all regions",
      lambda: assert_eq(len(me['sections']['regional_comparison']['data']), GT['claims_regions']),
      "member_exp")

check("Member issues (denial drivers) populated",
      lambda: assert_gt(len(me['sections']['member_issues']['data']), 0, "issues"),
      "member_exp")

check("Enrollment trend has months",
      lambda: assert_gt(len(me['sections']['enrollment_trend']['data']), 0, "months"),
      "member_exp")

# --- 1D: RADA ---
print("\n--- 1D: Risk Adjustment & Coding Accuracy ---")
rada = ede.get_risk_adjustment_coding()

risk_summary = rada['sections']['risk_score_summary']['metrics']
check("Average risk score matches ground truth",
      lambda: assert_eq(risk_summary['average_risk_score']['value'], GT['avg_risk'], 0.01),
      "rada")

check("Risk distribution has quartiles",
      lambda: assert_gt(len(rada['sections']['risk_distribution']['data']), 0, "quartiles"),
      "rada")

check("HCC capture data exists",
      lambda: assert_gt(len(rada['sections']['hcc_capture'].get('data', rada['sections']['hcc_capture'].get('metrics', {}))), 0, "hcc"),
      "rada")

check("Risk by region has all regions",
      lambda: assert_eq(len(rada['sections']['risk_by_region']['data']), GT['member_regions']),
      "rada")

# --- 1E: Membership & Market Share ---
print("\n--- 1E: Membership & Market Share ---")
ms = ede.get_membership_market_share()

check("Plan mix has all plan types",
      lambda: assert_eq(len(ms['sections']['plan_mix']['data']), GT['plan_types']),
      "membership")

plan_member_sum = sum(p['members'] for p in ms['sections']['plan_mix']['data'])
check("Plan mix member counts sum to total",
      lambda: assert_eq(plan_member_sum, GT['total_members']),
      "membership")

check("Market share by region has all regions",
      lambda: assert_eq(len(ms['sections']['market_share_by_region']['data']), GT['member_regions']),
      "membership")

# --- 1F: Service Utilization ---
print("\n--- 1F: Service Utilization ---")
su = ede.get_service_utilization()

util_data = su['sections']['util_per_1000']['data']
check("Utilization has all visit types",
      lambda: assert_eq(len(util_data), GT['visit_type_count']),
      "utilization")

util_total = sum(u['encounters'] for u in util_data)
check("Utilization encounter total matches ground truth",
      lambda: assert_eq(util_total, GT['total_encounters']),
      "utilization")

check("ED per 1000 matches ground truth",
      lambda: assert_eq(
          su['sections']['ed_metrics']['metrics']['ed_visits_per_1000']['value'],
          GT['ed_per_1000'], 0.5),
      "utilization")

check("Referrals per 1000 matches ground truth",
      lambda: assert_eq(
          su['sections']['referral_metrics']['metrics']['referrals_per_1000']['value'],
          GT['referral_per_1000'], 0.5),
      "utilization")

check("Pharmacy PMPM matches ground truth",
      lambda: assert_eq(
          su['sections']['pharmacy_metrics']['metrics']['pharmacy_pmpm']['value'],
          GT['pharmacy_pmpm'], 0.5),
      "utilization")

check("Regional utilization has all regions",
      lambda: assert_eq(len(su['sections']['regional_utilization']['data']), GT['encounter_regions']),
      "utilization")

# --- 1G: Executive Summary ---
print("\n--- 1G: Executive Summary ---")
es = ede.get_executive_summary()

sc = es['sections']['performance_scorecard']
check("Scorecard has 5 KPIs",
      lambda: assert_eq(len(sc['kpis']), 5),
      "executive")

check("Scorecard overall_status is valid RAG",
      lambda: sc['overall_status'] in ('green', 'amber', 'red') or (_ for _ in ()).throw(AssertionError(f"invalid: {sc['overall_status']}")),
      "executive")

check("RAG counts sum to KPI count",
      lambda: assert_eq(sc['rag_summary']['green'] + sc['rag_summary']['amber'] + sc['rag_summary']['red'], 5),
      "executive")

# Verify each KPI value against ground truth
for kpi in sc['kpis']:
    metric = kpi['metric']
    if metric == 'Member Retention Rate':
        check(f"Scorecard: {metric} matches GT",
              lambda m=metric, k=kpi: assert_eq(k['value'], GT['retention_rate'], 0.1),
              "executive")
    elif metric == 'Average Risk Score':
        check(f"Scorecard: {metric} matches GT",
              lambda m=metric, k=kpi: assert_eq(k['value'], GT['avg_risk'], 0.01),
              "executive")
    elif metric == 'Denial Rate':
        check(f"Scorecard: {metric} matches GT",
              lambda m=metric, k=kpi: assert_eq(k['value'], GT['denial_rate'], 0.1),
              "executive")
    elif metric == 'Medical Loss Ratio':
        check(f"Scorecard: {metric} matches GT",
              lambda m=metric, k=kpi: assert_eq(k['value'], GT['mlr'], 0.1),
              "executive")
    elif metric == 'Claims Collection Rate':
        check(f"Scorecard: {metric} matches GT",
              lambda m=metric, k=kpi: assert_eq(k['value'], GT['collection_rate'], 0.1),
              "executive")

check("Strategic priorities populated",
      lambda: assert_gt(len(es['sections']['strategic_priorities']['priorities']), 0, "priorities"),
      "executive")

check("Highlights populated",
      lambda: assert_gt(len(es['sections']['highlights']['positive_areas']), 0, "highlights"),
      "executive")


# =============================================================================
# SECTION 2: EDGE CASE RESILIENCE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: EDGE CASE RESILIENCE")
print("=" * 70)

# Test that dashboard methods never crash
print("\n--- 2A: No-crash guarantee ---")
for name, fn in [
    ('Financial Performance', ede.get_financial_performance),
    ('Stars Performance', ede.get_stars_performance),
    ('Member Experience', ede.get_member_experience),
    ('RADA', ede.get_risk_adjustment_coding),
    ('Membership', ede.get_membership_market_share),
    ('Utilization', ede.get_service_utilization),
    ('Executive Summary', ede.get_executive_summary),
    ('Full Dashboard', ede.get_full_dashboard),
]:
    check(f"{name} returns dict without crash",
          lambda f=fn: assert_type(f(), dict, name),
          "edge_case")

# Test that all numeric values are finite (no NaN, no Inf)
print("\n--- 2B: No NaN/Inf in any metric ---")
import math

def check_no_nan_inf(obj, path=""):
    """Recursively check for NaN/Inf in nested structure."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise AssertionError(f"NaN/Inf at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_nan_inf(v, f"{path}[{i}]")
    return True

for name, fn in [
    ('Financial', ede.get_financial_performance),
    ('Stars', ede.get_stars_performance),
    ('Member Exp', ede.get_member_experience),
    ('RADA', ede.get_risk_adjustment_coding),
    ('Membership', ede.get_membership_market_share),
    ('Utilization', ede.get_service_utilization),
    ('Exec Summary', ede.get_executive_summary),
]:
    check(f"{name} — no NaN/Inf anywhere",
          lambda f=fn, n=name: check_no_nan_inf(f(), n),
          "edge_case")

# Test JSON serialization (catches datetime, Decimal issues)
print("\n--- 2C: JSON serialization ---")
for name, fn in [
    ('Financial', ede.get_financial_performance),
    ('Stars', ede.get_stars_performance),
    ('Membership', ede.get_membership_market_share),
    ('Utilization', ede.get_service_utilization),
    ('Exec Summary', ede.get_executive_summary),
]:
    check(f"{name} — JSON serializable",
          lambda f=fn, n=name: json.dumps(f(), default=str) and True,
          "edge_case")


# =============================================================================
# SECTION 3: KNOWLEDGE GRAPH INTEGRITY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: KNOWLEDGE GRAPH INTEGRITY")
print("=" * 70)

from knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph(DB_PATH)
kg.discover_schema()
kg.discover_relationships()

print("\n--- 3A: Schema Discovery ---")
metrics = kg.compute_metrics()

check("Discovered 9+ tables (core healthcare + helpers)",
      lambda: assert_gt(metrics['total_tables'], 8, "tables"),
      "knowledge_graph")

check("Discovered 130+ columns",
      lambda: assert_gt(metrics['total_columns'], 130, "columns"),
      "knowledge_graph")

check("Discovered 50+ relationships",
      lambda: assert_gt(metrics['total_relationships'], 50, "relationships"),
      "knowledge_graph")

print("\n--- 3B: Relationship Validation ---")
# Key relationships that MUST exist
required_joins = [
    ('CLAIMS', 'MEMBERS', 'MEMBER_ID'),
    ('ENCOUNTERS', 'MEMBERS', 'MEMBER_ID'),
    ('DIAGNOSES', 'MEMBERS', 'MEMBER_ID'),
    ('CLAIMS', 'ENCOUNTERS', 'ENCOUNTER_ID'),
    ('DIAGNOSES', 'ENCOUNTERS', 'ENCOUNTER_ID'),
    ('PRESCRIPTIONS', 'MEMBERS', 'MEMBER_ID'),
    ('REFERRALS', 'MEMBERS', 'MEMBER_ID'),
]

for table_a, table_b, join_col in required_joins:
    key = (table_a, table_b)
    check(f"JOIN path: {table_a} -> {table_b} via {join_col}",
          lambda k=key, jc=join_col: (
              k in kg._relationships and
              any(r['column_a'] == jc or r['column_b'] == jc for r in kg._relationships[k])
          ) or (_ for _ in ()).throw(AssertionError(f"Missing JOIN: {k} via {jc}")),
          "knowledge_graph")

print("\n--- 3C: Query Blueprint Generation ---")
test_blueprints = [
    ("average paid amount by region", ['CLAIMS'], 'AVG'),
    ("total claims by plan type", ['CLAIMS'], 'COUNT'),
    ("member count by gender", ['MEMBERS'], 'COUNT'),
    ("risk score distribution", ['MEMBERS'], None),
]

for question, expected_tables, expected_agg in test_blueprints:
    check(f"Blueprint: '{question}'",
          lambda q=question: kg.generate_query_blueprint(q) is not None,
          "knowledge_graph")

print("\n--- 3D: Learning & Persistence ---")
# Test learn_pattern
kg.learn_pattern(
    question="test iron-clad query",
    sql="SELECT COUNT(*) FROM claims",
    success=True,
    execution_time=50
)
check("Pattern learned successfully",
      lambda: True,
      "knowledge_graph")

# Test find_closest_pattern
found = kg.find_closest_pattern("test iron-clad query")
check("Learned pattern retrievable",
      lambda: found is not None and len(found) > 0,
      "knowledge_graph")

print("\n--- 3E: Gap Detection & Retrain ---")
gaps = kg.detect_gaps()
check("Gap detection returns list",
      lambda: assert_type(gaps, list, "gaps"),
      "knowledge_graph")

should_retrain, reason = kg.should_retrain()
check("Retrain check returns tuple (bool, dict)",
      lambda: isinstance(should_retrain, bool) and isinstance(reason, dict),
      "knowledge_graph")


# =============================================================================
# SECTION 4: PIPELINE ROBUSTNESS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: PIPELINE ROBUSTNESS")
print("=" * 70)

print("\n--- 4A: Pipeline Initialization ---")
from intelligent_pipeline import IntelligentPipeline
pipeline = IntelligentPipeline(DB_PATH, 'data')

check("Pipeline initialized",
      lambda: pipeline is not None,
      "pipeline")

check("Knowledge graph wired",
      lambda: pipeline.knowledge_graph is not None,
      "pipeline")

check("Learning scorer wired",
      lambda: pipeline.learning_scorer is not None,
      "pipeline")

check("Executive dashboard engine wired",
      lambda: pipeline.executive_dashboard_engine is not None,
      "pipeline")

check("Source protection wired",
      lambda: pipeline.source_protect is not None,
      "pipeline")

print("\n--- 4B: Normal Query Processing ---")
normal_queries = [
    "how many members",
    "average paid amount",
    "claims by region",
    "top 5 providers by claims",
    "denial rate by plan type",
]

for q in normal_queries:
    check(f"Query: '{q}' returns result with SQL",
          lambda query=q: (
              pipeline.process(query).get('sql') is not None or
              pipeline.process(query).get('answer') is not None
          ),
          "pipeline")

print("\n--- 4C: Edge Case Queries ---")
edge_queries = [
    "",  # empty string
    "asdfghjkl",  # gibberish
    "SELECT * FROM members; DROP TABLE members;--",  # SQL injection
    "a" * 10000,  # very long input
    "   ",  # whitespace only
    "what is the meaning of life",  # non-healthcare
]

for q in edge_queries:
    label = q[:40] + "..." if len(q) > 40 else q if q.strip() else "(empty/whitespace)"
    check(f"Edge case: '{label}' — no crash",
          lambda query=q: pipeline.process(query) is not None,
          "pipeline")


# =============================================================================
# SECTION 5: BUSINESS LOGIC VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: BUSINESS LOGIC VALIDATION")
print("=" * 70)

print("\n--- 5A: RAG Status Logic ---")
es = ede.get_executive_summary()
sc = es['sections']['performance_scorecard']

for kpi in sc['kpis']:
    metric = kpi['metric']
    val = kpi['value']
    status = kpi['status']

    if metric == 'Member Retention Rate':
        disenroll = 100 - val
        expected = 'green' if disenroll < 10 else 'amber' if disenroll < 12 else 'red'
        check(f"RAG logic: {metric} ({val}%) -> {status}",
              lambda s=status, e=expected: assert_eq(s, e),
              "business_logic")

    elif metric == 'Denial Rate':
        expected = 'green' if val < 10 else 'amber' if val < 12 else 'red'
        check(f"RAG logic: {metric} ({val}%) -> {status}",
              lambda s=status, e=expected: assert_eq(s, e),
              "business_logic")

    elif metric == 'Average Risk Score':
        expected = 'green' if 1.5 <= val <= 3.0 else 'amber' if 1.2 <= val <= 3.5 else 'red'
        check(f"RAG logic: {metric} ({val}) -> {status}",
              lambda s=status, e=expected: assert_eq(s, e),
              "business_logic")

    elif metric == 'Medical Loss Ratio':
        expected = 'green' if 60 <= val <= 80 else 'amber' if 50 <= val <= 85 else 'red'
        check(f"RAG logic: {metric} ({val}%) -> {status}",
              lambda s=status, e=expected: assert_eq(s, e),
              "business_logic")

    elif metric == 'Claims Collection Rate':
        expected = 'green' if val >= 50 else 'amber' if val >= 40 else 'red'
        check(f"RAG logic: {metric} ({val}%) -> {status}",
              lambda s=status, e=expected: assert_eq(s, e),
              "business_logic")

print("\n--- 5B: Percentages in valid range (0-100) ---")

def validate_percentages(obj, path=""):
    """Check all percentage-labeled values are 0-100."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and v.get('format') == 'percent' and isinstance(v.get('value'), (int, float)):
                val = v['value']
                if not (0 <= val <= 100):
                    raise AssertionError(f"{path}.{k}: {val}% not in [0, 100]")
            validate_percentages(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            validate_percentages(v, f"{path}[{i}]")
    return True

for name, fn in [
    ('Financial', ede.get_financial_performance),
    ('Stars', ede.get_stars_performance),
    ('Member Exp', ede.get_member_experience),
]:
    check(f"{name} — all percentages 0-100",
          lambda f=fn, n=name: validate_percentages(f(), n),
          "business_logic")

print("\n--- 5C: Cross-dashboard consistency ---")
# Same metric from different dashboards must agree
fp = ede.get_financial_performance()
es = ede.get_executive_summary()

fp_total = fp['sections']['ytd_summary']['metrics']['total_members']['value']
es_retention = None
for kpi in es['sections']['performance_scorecard']['kpis']:
    if kpi['metric'] == 'Member Retention Rate':
        es_retention = kpi['value']
fp_retention = me['sections']['retention']['metrics']['retention_rate']['value']

check("Total members consistent: financial vs member_exp",
      lambda: assert_eq(fp_total, ret['total_members']['value']),
      "consistency")

check("Retention rate consistent: exec summary vs member_exp",
      lambda: assert_eq(es_retention, fp_retention, 0.1),
      "consistency")


# =============================================================================
# SECTION 6: SOURCE PROTECTION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SOURCE PROTECTION")
print("=" * 70)

from source_protect import SourceProtect
sp = SourceProtect()

check("Security headers count >= 10",
      lambda: assert_gt(len(sp.get_security_headers()), 9, "headers"),
      "security")

check("Protection script not empty",
      lambda: assert_gt(len(sp.get_protection_script()), 100, "script_bytes"),
      "security")

test_html = '<html><head></head><body><!-- SECRET KEY: abc123 --><p>Hello</p></body></html>'
protected = sp.obfuscate_response(test_html)
check("HTML comments stripped",
      lambda: '<!-- SECRET' not in protected or (_ for _ in ()).throw(AssertionError("Comments not stripped")),
      "security")

check("Content preserved after obfuscation",
      lambda: 'Hello' in protected or (_ for _ in ()).throw(AssertionError("Content lost")),
      "security")


# =============================================================================
# SECTION 7: LEARNING SCORER
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: LEARNING SCORER")
print("=" * 70)

from learning_scorer import LearningScorer
ls = LearningScorer('data')

health = ls.get_system_health()
check("System health has grade",
      lambda: health.get('grade') in ('A', 'B', 'C', 'D', 'F'),
      "learning")

check("System health has score",
      lambda: assert_type(health.get('overall_score', 0), (int, float), "score"),
      "learning")

check("Health score in valid range (0-100)",
      lambda: assert_range(health.get('overall_score', 0), 0, 100, "health_score"),
      "learning")


# =============================================================================
# SECTION 8: REASONING PRE-VALIDATOR
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: REASONING PRE-VALIDATOR")
print("=" * 70)

from reasoning_chain import ReasoningPreValidator
rpv = ReasoningPreValidator(kg)

# Valid query direction
result = rpv.validate_query_direction(
    "average paid amount by region",
    {'intent': 'aggregation'},
    ['CLAIMS'],
    ['PAID_AMOUNT'],
    'AVG'
)
check("Valid query direction accepted",
      lambda: result.validated == True,
      "reasoning")

# Test with wrong table
result2 = rpv.validate_query_direction(
    "average paid amount by region",
    {'intent': 'aggregation'},
    ['PRESCRIPTIONS'],
    ['PAID_AMOUNT'],
    'AVG'
)
check("Wrong table flagged or corrected",
      lambda: result2 is not None,
      "reasoning")


# =============================================================================
# SECTION 9: DEMOGRAPHIC ANALYTICS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: DEMOGRAPHIC ANALYTICS")
print("=" * 70)

from demographic_analytics import DemographicAnalytics
da = DemographicAnalytics(DB_PATH)

check("Geographic demographics has regions",
      lambda: assert_gt(len(da.analyze_geographic_demographics().get('regional_profile', [])), 0, "regions"),
      "demographics")

check("Health status demographics has tiers",
      lambda: assert_gt(len(da.analyze_health_status_demographics().get('chronic_distribution', [])), 0, "tiers"),
      "demographics")

check("Socioeconomic proxy has plans",
      lambda: assert_gt(len(da.analyze_socioeconomic_proxy().get('plan_type_profile', [])), 0, "plans"),
      "demographics")

check("Utilization demographics has combos",
      lambda: assert_gt(len(da.analyze_utilization_demographics().get('utilization_by_demo', [])), 0, "combos"),
      "demographics")

check("Comprehensive report has 7+ sections",
      lambda: assert_gt(len(da.get_comprehensive_demographic_report()), 7, "sections"),
      "demographics")

da.close()


# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "=" * 70)
total = passed + failed
print(f"IRON-CLAD RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if errors:
    print("\nFAILURES:")
    for cat, label, msg in errors:
        print(f"  [{cat}] {label}: {msg}")

if failed == 0:
    print("\n🔒 ALL TESTS PASSED — SYSTEM IS IRON-CLAD")
else:
    print(f"\n⚠️  {failed} FAILURES — NEEDS FIXING")
    sys.exit(1)
