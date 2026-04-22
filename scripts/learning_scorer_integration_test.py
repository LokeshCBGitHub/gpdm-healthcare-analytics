import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from learning_scorer import LearningScorer


def simulate_chatbot_session():

    scorer = LearningScorer()

    print("=" * 80)
    print("HEALTHCARE CHATBOT LEARNING SCORER - INTEGRATION TEST")
    print("=" * 80)
    print()

    print("SCENARIO 1: Processing Simple Aggregation Queries")
    print("-" * 80)

    query1 = scorer.score_query(
        question="How many total claims were submitted last month?",
        intent="aggregation",
        sql="SELECT COUNT(*) as total_claims FROM claims WHERE MONTH(submission_date) = MONTH(GETDATE()) - 1",
        sql_accuracy=0.98,
        intent_match=0.99,
        table_selection=0.98,
        column_resolution=0.97,
        join_correctness=1.0,
        latency_ms=180,
        narrative_quality=0.95,
        result={"total_claims": 4521},
        query_type="aggregation",
        tables_involved=["claims"],
        complexity_level="simple",
    )
    print(f"✓ Query 1: Overall Score={query1.overall_score:.3f}, Success={query1.success}")

    query2 = scorer.score_query(
        question="What's the average claim amount?",
        intent="aggregation",
        sql="SELECT AVG(amount) as avg_amount FROM claims",
        sql_accuracy=0.96,
        intent_match=0.98,
        table_selection=0.98,
        column_resolution=0.96,
        join_correctness=1.0,
        latency_ms=220,
        narrative_quality=0.93,
        result={"avg_amount": 1245.50},
        query_type="aggregation",
        tables_involved=["claims"],
        complexity_level="simple",
    )
    print(f"✓ Query 2: Overall Score={query2.overall_score:.3f}, Success={query2.success}")
    print()

    print("SCENARIO 2: Processing Filter and Join Queries")
    print("-" * 80)

    query3 = scorer.score_query(
        question="Which providers had claims over $5000 last quarter?",
        intent="filter_and_join",
        sql="SELECT DISTINCT p.name FROM providers p JOIN claims c ON p.id = c.provider_id WHERE c.amount > 5000 AND c.submission_date >= DATE('now', '-3 months')",
        sql_accuracy=0.88,
        intent_match=0.85,
        table_selection=0.90,
        column_resolution=0.82,
        join_correctness=0.85,
        latency_ms=750,
        narrative_quality=0.80,
        result=[],
        query_type="join",
        tables_involved=["providers", "claims"],
        complexity_level="medium",
    )
    print(f"✓ Query 3: Overall Score={query3.overall_score:.3f}, Success={query3.success}")

    query4 = scorer.score_query(
        question="Show me member demographics for high-cost claimants",
        intent="complex_analysis",
        sql="SELECT m.*, COUNT(c.id) as claim_count, SUM(c.amount) as total_cost FROM members m LEFT JOIN claims c ON m.id = c.member_id WHERE c.amount > 10000 GROUP BY m.id",
        sql_accuracy=0.65,
        intent_match=0.70,
        table_selection=0.75,
        column_resolution=0.60,
        join_correctness=0.70,
        latency_ms=4200,
        narrative_quality=0.50,
        result=None,
        query_type="join",
        tables_involved=["members", "claims"],
        complexity_level="complex",
        details={
            "error": "Ambiguous column references",
            "missing_joins": ["encounters"],
        }
    )
    print(f"✓ Query 4: Overall Score={query4.overall_score:.3f}, Success={query4.success}")
    print()

    print("SCENARIO 3: Processing Temporal and Comparative Queries")
    print("-" * 80)

    query5 = scorer.score_query(
        question="Compare claim volumes month-over-month this year",
        intent="temporal_comparison",
        sql="SELECT DATE_TRUNC('month', submission_date) as month, COUNT(*) as count FROM claims GROUP BY DATE_TRUNC('month', submission_date) ORDER BY month",
        sql_accuracy=0.75,
        intent_match=0.80,
        table_selection=0.85,
        column_resolution=0.70,
        join_correctness=1.0,
        latency_ms=2100,
        narrative_quality=0.65,
        result=[],
        query_type="temporal",
        tables_involved=["claims"],
        complexity_level="medium",
    )
    print(f"✓ Query 5: Overall Score={query5.overall_score:.3f}, Success={query5.success}")
    print()

    print("SYSTEM HEALTH ASSESSMENT")
    print("-" * 80)
    health = scorer.get_system_health()
    print(f"System Grade: {health['grade']}")
    print(f"Overall Score: {health['overall_score']:.3f} (out of 1.0)")
    print(f"Success Rate: {health['success_rate']:.1%}")
    print(f"Total Queries Processed: {health['total_queries_processed']}")
    print(f"Critical Gaps Detected: {health['critical_gaps']}")
    print(f"Pending Retrain Triggers: {health['pending_retrain_triggers']}")
    print()

    print("TOP WEAK AREAS (Needs Improvement)")
    print("-" * 80)
    weak_areas = scorer.get_weak_areas(limit=3)
    if weak_areas:
        for i, area in enumerate(weak_areas, 1):
            print(f"{i}. {area['category']} ({area['type']})")
            print(f"   Score: {area['overall_score']:.3f}")
            print(f"   Samples: {area['sample_count']}")
            print(f"   Trend: {area['trend']}")
            breakdown = area['breakdown']
            print(f"   - SQL Accuracy: {breakdown['sql_accuracy']:.3f}")
            print(f"   - Intent Match: {breakdown['intent_match']:.3f}")
            print(f"   - Column Resolution: {breakdown['column_resolution']:.3f}")
            print(f"   - JOIN Correctness: {breakdown['join_correctness']:.3f}")
            print()

    print("TOP STRONG AREAS (Performing Well)")
    print("-" * 80)
    strong_areas = scorer.get_strong_areas(limit=3)
    if strong_areas:
        for i, area in enumerate(strong_areas, 1):
            print(f"{i}. {area['category']} ({area['type']})")
            print(f"   Score: {area['overall_score']:.3f}")
            print(f"   Samples: {area['sample_count']}")
            print(f"   Trend: {area['trend']}")
            print()

    print("RETRAIN TRIGGER CHECK")
    print("-" * 80)
    should_retrain, reasons = scorer.should_retrain()
    print(f"Should Retrain: {'YES' if should_retrain else 'NO'}")

    if reasons['details']:
        print("Retrain Reasons:")
        for detail in reasons['details']:
            print(f"  - Type: {detail['type']}")
            print(f"    Severity: {detail['severity']}")
            print(f"    Affected: {detail['categories']}")
            print(f"    Recommendation: {detail['recommendation']}")
            print()

    print("IMPROVEMENT PLAN (Prioritized Fixes)")
    print("-" * 80)
    plan = scorer.get_improvement_plan()

    if plan.get('priority_gaps'):
        print(f"Priority Gaps ({len(plan['priority_gaps'])} total):")
        for gap in plan['priority_gaps'][:5]:
            priority_symbol = "CRITICAL" if gap['priority'] == 'critical' else "HIGH" if gap['priority'] == 'high' else "MEDIUM"
            print(f"  {priority_symbol} {gap['key']}")
            print(f"     Description: {gap['description']}")
            print(f"     Severity: {gap['severity']:.2f}")
            print(f"     Occurrences: {gap['occurrences']}")
            print()

    print(f"Estimated Retraining Effort: {plan.get('estimated_effort', 'unknown')}")
    print()

    print("FEEDBACK LOOP: Recording Manual Correction")
    print("-" * 80)
    scorer.learn_from_correction(
        question="What's the correlation between age and claim amount?",
        wrong_sql="SELECT age, amount FROM members m JOIN claims c ON m.id = c.member_id",
        correct_sql="SELECT m.age, AVG(c.amount) as avg_claim_amount FROM members m JOIN claims c ON m.id = c.member_id GROUP BY m.age ORDER BY m.age",
        reason="Missing GROUP BY and aggregation - need to compute average claim amount per age group"
    )
    print("✓ Correction recorded for learning")
    print()

    print("COMPREHENSIVE LEARNING REPORT")
    print("=" * 80)
    report = scorer.get_learning_report()

    print(f"Report Generated: {report['report_generated']}")
    print()

    print("System Health Summary:")
    print(f"  Grade: {report['system_health']['grade']}")
    print(f"  Score: {report['system_health']['overall_score']:.3f}")
    print(f"  Success Rate: {report['system_health']['success_rate']:.1%}")
    print(f"  Critical Gaps: {report['system_health']['critical_gaps']}")
    print()

    print(f"Should Retrain: {report['should_retrain']}")
    print(f"Weak Categories: {len(report['weak_areas'])}")
    print(f"Strong Categories: {len(report['strong_areas'])}")
    print(f"Historical Data Points: {len(report['historical_performance'])}")
    print()

    print("PIPELINE INTEGRATION EXAMPLE")
    print("=" * 80)
    print("""
# In your chatbot pipeline (intelligent_pipeline.py or similar):

from learning_scorer import LearningScorer

class ChatbotPipeline:
    def __init__(self):
        self.scorer = LearningScorer()

    def process_query(self, question, intent, sql, result, latency_ms):
        # ... your processing logic ...

        # Score this query
        score = self.scorer.score_query(
            question=question,
            intent=intent,
            sql=sql,
            sql_accuracy=compute_sql_accuracy(sql, result),
            intent_match=compute_intent_match(intent),
            table_selection=compute_table_selection(sql),
            column_resolution=compute_column_resolution(sql),
            join_correctness=compute_join_correctness(sql),
            latency_ms=latency_ms,
            narrative_quality=compute_narrative_quality(explanation),
            result=result,
            query_type=infer_query_type(sql),
            tables_involved=extract_tables(sql),
            complexity_level=assess_complexity(sql),
        )

        # Check if retraining is needed
        should_retrain, reasons = self.scorer.should_retrain()
        if should_retrain:
            logger.warning(f"Retrain triggered: {reasons}")
            trigger_retraining_pipeline()

        return result, score

    def get_diagnostics(self):
        health = self.scorer.get_system_health()
        plan = self.scorer.get_improvement_plan()
        return {
            'health': health,
            'improvement_plan': plan,
        }
""")

    scorer.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nDatabase Location: /sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/learning_scores.db")
    print("All metrics have been persisted for historical analysis.")


if __name__ == "__main__":
    simulate_chatbot_session()
