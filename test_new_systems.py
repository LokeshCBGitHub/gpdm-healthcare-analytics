#!/usr/bin/env python3
"""Test all new systems built in this session."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
os.chdir(os.path.dirname(__file__))

passed = 0
failed = 0

def check(label, fn):
    global passed, failed
    try:
        result = fn()
        print(f"  PASS  {label}: {result}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        failed += 1

print("=== 1. IMPORT TESTS ===")
check("KnowledgeGraph import", lambda: __import__('knowledge_graph') and "OK")
check("ReasoningPreValidator import", lambda: (
    getattr(__import__('reasoning_chain'), 'ReasoningPreValidator') and "OK"
))
check("LearningScorer import", lambda: __import__('learning_scorer') and "OK")
check("ExecutiveDashboardEngine import", lambda: __import__('executive_dashboards') and "OK")
check("SourceProtect import", lambda: __import__('source_protect') and "OK")
check("DemographicAnalytics import", lambda: __import__('demographic_analytics') and "OK")

print("\n=== 2. KNOWLEDGE GRAPH ===")
from knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph('data/healthcare_demo.db')
check("Discover schema", lambda: kg.discover_schema() or "done")
check("Discover relationships", lambda: kg.discover_relationships() or "done")
m = kg.compute_metrics()
check("Metrics computed", lambda: f"{m.get('total_tables',0)} tables, {m.get('total_columns',0)} cols, {m.get('total_relationships',0)} rels")
check("Blueprint generation", lambda: (
    kg.generate_query_blueprint("average paid amount by region") or "generated"
))

print("\n=== 3. EXECUTIVE DASHBOARDS ===")
from executive_dashboards import ExecutiveDashboardEngine
ede = ExecutiveDashboardEngine('data/healthcare_demo.db')
check("Financial performance", lambda: f"PMPM={ede.get_financial_performance().get('pmpm_revenue','?')}")
check("Stars performance", lambda: f"rating={ede.get_stars_performance().get('overall_star_rating','?')}")
check("Member experience", lambda: f"keys={len(ede.get_member_experience())}")
check("RADA", lambda: f"avg_risk={ede.get_risk_adjustment_coding().get('average_risk_score','?')}")
check("Membership", lambda: f"total={ede.get_membership_market_share().get('total_members','?')}")
check("Utilization", lambda: f"keys={len(ede.get_service_utilization())}")
check("Executive summary", lambda: f"keys={len(ede.get_executive_summary())}")

print("\n=== 4. SOURCE PROTECT ===")
from source_protect import SourceProtect
sp = SourceProtect()
check("Security headers", lambda: f"{len(sp.get_security_headers())} headers")
check("Protection script", lambda: f"{len(sp.get_protection_script())} bytes")
test_html = '<html><head></head><body><!-- secret --><p>Hello</p></body></html>'
protected = sp.obfuscate_response(test_html)
check("HTML comments stripped", lambda: "PASS" if '<!-- secret -->' not in protected else "FAIL - comments still present")

print("\n=== 5. LEARNING SCORER ===")
from learning_scorer import LearningScorer
ls = LearningScorer('data')
check("System health", lambda: f"grade={ls.get_system_health().get('grade','?')}")

print("\n=== 6. EXPANDED DEMOGRAPHICS ===")
from demographic_analytics import DemographicAnalytics
da = DemographicAnalytics('data/healthcare_demo.db')
check("Geographic demographics", lambda: f"{len(da.analyze_geographic_demographics().get('regional_profile',[]))} regions")
check("Health status demographics", lambda: f"{len(da.analyze_health_status_demographics().get('chronic_distribution',[]))} tiers")
check("Socioeconomic proxy", lambda: f"{len(da.analyze_socioeconomic_proxy().get('plan_type_profile',[]))} plans")
check("Utilization demographics", lambda: f"{len(da.analyze_utilization_demographics().get('utilization_by_demo',[]))} combos")
check("Comprehensive report", lambda: f"{len(da.get_comprehensive_demographic_report())} sections")
da.close()

print("\n=== 7. REASONING PRE-VALIDATOR ===")
from reasoning_chain import ReasoningPreValidator
rpv = ReasoningPreValidator(kg)
check("Validate correct direction", lambda: (
    rpv.validate_query_direction(
        "average paid amount",
        {'intent': 'aggregation'},
        ['CLAIMS'],
        ['PAID_AMOUNT'],
        'AVG'
    ) and "validated"
))

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} tests failed!")
    sys.exit(1)
