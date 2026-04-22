import json
from domain_intelligence import create_domain_intelligence


def example_1_jargon_normalization():
    print("\n" + "="*70)
    print("EXAMPLE 1: HEALTHCARE JARGON NORMALIZATION")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    user_questions = [
        "What's our PMPM vs industry?",
        "How many d/c patients had readmission?",
        "Any HCC issues with our high-risk members?",
        "What's our MLR compared to FFS competitors?",
    ]
    
    for question in user_questions:
        normalized = intelligence.normalize_question(question)
        print(f"\nOriginal:   {question}")
        print(f"Normalized: {normalized}")


def example_2_concept_understanding():
    print("\n" + "="*70)
    print("EXAMPLE 2: CONCEPT ONTOLOGY UNDERSTANDING")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    concepts_to_explore = ["claim_denial", "readmission", "risk_score"]
    
    for concept_id in concepts_to_explore:
        concept = intelligence.get_concept_definition(concept_id)
        if concept:
            print(f"\nConcept: {concept.name}")
            print(f"  Definition: {concept.definition}")
            print(f"  Related Tables: {', '.join(concept.related_tables)}")
            print(f"  SQL Pattern: {concept.sql_pattern}")
            print(f"  Measurable: {concept.metadata.get('measurable', False)}")


def example_3_question_analysis():
    print("\n" + "="*70)
    print("EXAMPLE 3: INTELLIGENT QUESTION ANALYSIS")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    complex_questions = [
        "What's the denial rate for high-risk diabetic members in NCAL this month?",
        "Are we doing well on readmissions?",
        "Which regions have concerning no-show rates?",
        "Show cost trends for elderly patients.",
    ]
    
    for question in complex_questions:
        print(f"\nQuestion: {question}")
        analysis = intelligence.analyze_question(question)
        
        print(f"  Intent: {analysis['intent']['inferred_intent']}")
        print(f"  Scope: Region={analysis['scope']['regions']}, Period={analysis['scope']['time_period']}")
        print(f"  Entities: {len(analysis['entities'])} extracted")
        
        if analysis['entities']:
            for entity in analysis['entities']:
                print(f"    - {entity['type']}: {entity['sql_where']}")
        
        if analysis['related_metrics']:
            print(f"  Metrics to calculate: {[m['metric_id'] for m in analysis['related_metrics']]}")


def example_4_sql_construction():
    print("\n" + "="*70)
    print("EXAMPLE 4: SQL CONSTRUCTION FROM NATURAL LANGUAGE")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    question = "What's the denial rate for high-risk members in NCAL last quarter?"
    analysis = intelligence.analyze_question(question)
    
    metrics = analysis['related_metrics']
    if metrics:
        metric_id = metrics[0]['metric_id']
        metric = intelligence.get_metric_definition(metric_id)
        
        where_parts = []
        
        if analysis['scope']['regions']:
            regions = analysis['scope']['regions']
            where_parts.append(f"KP_REGION IN ({','.join([repr(r) for r in regions])})")
        
        if analysis['sql_where_clause']:
            where_parts.append(f"({analysis['sql_where_clause']})")
        
        where_parts.append("SERVICE_DATE >= DATE('now', '-3 months')")
        
        where_clause = " AND ".join(where_parts)
        
        sql = f"""
            SELECT 
                DATE_TRUNC('month', SERVICE_DATE) as month,
                KP_REGION,
                {metric.numerator_sql} as claims_denied,
                {metric.denominator_sql} as total_claims,
                ROUND(100.0 * {metric.numerator_sql} / {metric.denominator_sql}, 2) as denial_rate
            FROM claims
            WHERE {where_clause}
            GROUP BY DATE_TRUNC('month', SERVICE_DATE), KP_REGION
            ORDER BY month DESC, denial_rate DESC
        """
        
        print(f"\nQuestion: {question}\n")
        print(f"Generated SQL:\n{sql}")


def example_5_benchmarking():
    print("\n" + "="*70)
    print("EXAMPLE 5: INTELLIGENT BENCHMARKING")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    metrics_to_grade = [
        ("denial_rate", 0.08),
        ("no_show_rate", 0.15),
        ("pmpm", 420),
        ("readmission_rate", 0.12),
        ("avg_los", 4.2),
    ]
    
    print("\nPerformance Report:")
    print("-" * 70)
    
    for metric_id, value in metrics_to_grade:
        metric = intelligence.get_metric_definition(metric_id)
        benchmark = intelligence.benchmark_metric(metric_id, value)
        
        print(f"\n{metric.name}:")
        print(f"  Current Value: {value} {metric.units}")
        print(f"  Grade: {benchmark['grade'].upper()}")
        print(f"  Excellent Threshold: {benchmark['excellent_threshold']} {metric.units}")
        print(f"  Average Range: {benchmark['average_range'][0]}-{benchmark['average_range'][1]} {metric.units}")
        print(f"  Percentile: {benchmark['percentile_estimate']:.0f}th")


def example_6_causal_reasoning():
    print("\n" + "="*70)
    print("EXAMPLE 6: CAUSAL REASONING - WHY ANALYSIS")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    concerning_metrics = ["denial_rate", "no_show_rate", "readmission_rate"]
    
    for metric_id in concerning_metrics:
        explanation = intelligence.explain_metric_causes(metric_id)
        
        print(f"\nMetric: {metric_id}")
        print("Investigation questions to answer:")
        
        for i, query_info in enumerate(explanation['investigation_queries'], 1):
            print(f"  {i}. {query_info['investigation']}")
            print(f"     SQL: {query_info['sql_template'][:100]}...")


def example_7_context_inference():
    print("\n" + "="*70)
    print("EXAMPLE 7: CONTEXTUAL INFERENCE ENGINE")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    ambiguous_questions = [
        "Are we doing well?",
        "What's concerning?",
        "How's NCAL?",
        "What about quality?",
        "Costs too high?",
    ]
    
    for question in ambiguous_questions:
        intent = intelligence.contextual_inference.infer_intent(question)
        scope = intelligence.contextual_inference.infer_scope(question)
        
        print(f"\nQuestion: '{question}'")
        print(f"  Inferred Intent: {intent['inferred_intent']}")
        
        if intent['metrics_to_check']:
            print(f"  Metrics to Check: {intent['metrics_to_check'][:3]}...")
        
        if scope['time_period'] != 'current':
            print(f"  Time Period: {scope['time_period']}")
        
        if scope['regions']:
            print(f"  Regions: {scope['regions']}")


def example_8_integration_workflow():
    print("\n" + "="*70)
    print("EXAMPLE 8: COMPLETE CHATBOT WORKFLOW")
    print("="*70)
    
    intelligence = create_domain_intelligence()
    
    user_question = "Are we doing well on denial rates in high-risk members?"
    print(f"\nUser: {user_question}")
    
    analysis = intelligence.analyze_question(user_question)
    print(f"\nStep 1 - Analysis:")
    print(f"  Intent: {analysis['intent']['inferred_intent']}")
    print(f"  Metrics: {[m['metric_id'] for m in analysis['related_metrics']]}")
    
    metric_id = "denial_rate"
    metric = intelligence.get_metric_definition(metric_id)
    print(f"\nStep 2 - Metric Definition:")
    print(f"  {metric.name} ({metric.units})")
    
    calculated_value = 0.09
    print(f"\nStep 3 - Calculation Result:")
    print(f"  Current Value: {calculated_value*100:.1f}%")
    
    benchmark = intelligence.benchmark_metric(metric_id, calculated_value)
    print(f"\nStep 4 - Benchmarking:")
    print(f"  Grade: {benchmark['grade']} (excellent: <5%, average: 5-10%, poor: >15%)")
    print(f"  Industry Position: {benchmark['percentile_estimate']:.0f}th percentile")
    
    if benchmark['grade'] != 'excellent':
        explanation = intelligence.explain_metric_causes(metric_id)
        print(f"\nStep 5 - Root Cause Analysis:")
        for i, q in enumerate(explanation['investigation_queries'][:2], 1):
            print(f"  {i}. {q['investigation']}")
    
    print(f"\nStep 6 - Chatbot Response:")
    if benchmark['grade'] == 'excellent':
        narrative = f"Our denial rate of {calculated_value*100:.1f}% is excellent! Well below industry benchmarks (<5%)."
    elif benchmark['grade'] == 'average':
        narrative = f"Our denial rate of {calculated_value*100:.1f}% is average for the industry (5-10% range). We should investigate specific denial reasons."
    else:
        narrative = f"Our denial rate of {calculated_value*100:.1f}% is concerning (above 15% threshold). Immediate investigation needed."
    
    print(f"  {narrative}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOMAIN INTELLIGENCE INTEGRATION EXAMPLES")
    print("="*70)
    
    example_1_jargon_normalization()
    example_2_concept_understanding()
    example_3_question_analysis()
    example_4_sql_construction()
    example_5_benchmarking()
    example_6_causal_reasoning()
    example_7_context_inference()
    example_8_integration_workflow()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
