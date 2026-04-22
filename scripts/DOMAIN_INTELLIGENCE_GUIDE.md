# Healthcare Domain Intelligence Module

## Overview

The `domain_intelligence.py` module provides deep domain understanding for the Kaiser Permanente Medicare Advantage healthcare analytics chatbot. It enables the system to:

1. **Master Healthcare Jargon** — Map synonyms, abbreviations, and alternative phrasings to canonical forms
2. **Understand Concepts** — Build semantic relationships between healthcare concepts
3. **Calculate KPIs** — Define and compute business metrics with SQL templates
4. **Infer Context** — Understand implicit meaning in ambiguous questions
5. **Extract Entities** — Convert natural language to SQL WHERE clauses
6. **Apply Benchmarks** — Grade performance against industry standards
7. **Reason Causally** — Explain "why" questions with structured investigations

## Quick Start

### Basic Initialization

```python
from domain_intelligence import create_domain_intelligence

# Create the intelligence system
intelligence = create_domain_intelligence()

# Analyze a user question
question = "What's the denial rate by region in NCAL?"
analysis = intelligence.analyze_question(question)
```

### Component Access

```python
# 1. Domain Knowledge (jargon normalization)
intelligence.domain_knowledge.normalize_term("PMPM")
# Returns: "PER_MEMBER_PER_MONTH"

# 2. Concept Ontology
concept = intelligence.ontology.get_concept("claim_denial")
# Returns: ConceptNode with definition, related tables, SQL patterns

# 3. Metrics
metric = intelligence.metric_engine.get_metric("denial_rate")
# Returns: MetricDefinition with numerator/denominator SQL

# 4. Benchmarking
benchmark = intelligence.benchmark_metric("denial_rate", 0.08)
# Returns: grade ("excellent"/"average"/"poor"), percentile, thresholds

# 5. Causal Reasoning
explanation = intelligence.explain_metric_causes("denial_rate")
# Returns: investigation queries for root cause analysis
```

## Module Components

### 1. HealthcareDomainKnowledge

Maps healthcare terminology to canonical forms across four domains:

#### Doctor Jargon
- `ICD-10` → `ICD10_CODE`
- `CPT` → `CPT_CODE`
- `DRG` → `DRG_CODE`
- `LOS` → `LENGTH_OF_STAY`
- `PCP` → `PRIMARY_CARE_PHYSICIAN`
- `HCC` → `HCC_CODE`
- `RAF` → `RISK_ADJUSTMENT_FACTOR`
- `d/c` → `DISCHARGE`

#### Patient Jargon
- `copay` → `COPAYMENT`
- `deductible` → `DEDUCTIBLE`
- `OOP` → `OUT_OF_POCKET`
- `prior auth` → `PRIOR_AUTHORIZATION`
- `ER visit` → `EMERGENCY_VISIT`
- `formulary` → `FORMULARY`

#### Insurance Jargon
- `capitation` → `CAPITATED_PAYMENT`
- `EOB` → `EXPLANATION_OF_BENEFITS`
- `utilization review` → `UTILIZATION_REVIEW`
- `medical necessity` → `MEDICAL_NECESSITY_REVIEW`
- `claim denial` → `CLAIM_DENIAL`

#### Finance Jargon
- `PMPM` → `PER_MEMBER_PER_MONTH`
- `MLR` → `MEDICAL_LOSS_RATIO`
- `FFS` → `FEE_FOR_SERVICE`
- `risk corridor` → `RISK_CORRIDOR`

**Usage:**
```python
normalized = intelligence.domain_knowledge.normalize_term("ICD-10")
aliases = intelligence.domain_knowledge.get_all_aliases_for("DISCHARGE")
is_abbrev = intelligence.domain_knowledge.is_abbreviation("PMPM")
```

### 2. ConceptOntology

Builds a semantic graph of healthcare concepts with relationships:

**Core Concepts:**
- `claim_denial` — Denied claims (uses CLAIM_STATUS, DENIAL_REASON columns)
- `readmission` — 30-day hospital readmission (temporal join on encounters)
- `no_show` — Missed appointments (STATUS='NO_SHOW')
- `hedis` — Quality measures (CMS/NCQA metrics)
- `stars_rating` — 5-star quality rating system
- `risk_score` — HCC-based risk adjustment
- `utilization` — Healthcare service volume (ED visits, inpatient days)
- `cost_analysis` — Spending analysis
- `kp_region` — Geographic service areas

**Concept Properties:**
- `definition` — Plain language description
- `related_tables` — Which tables contain this concept
- `related_columns` — Specific columns involved
- `sql_pattern` — Template WHERE clause
- `metadata` — Type, measurability, examples

**Usage:**
```python
concept = intelligence.ontology.get_concept("claim_denial")
print(f"Name: {concept.name}")
print(f"Definition: {concept.definition}")
print(f"Tables: {concept.related_tables}")
print(f"SQL Pattern: {concept.sql_pattern}")

related = intelligence.ontology.find_related_concepts("readmission")
```

### 3. BusinessMetricEngine

Defines KPIs with SQL templates and benchmarks:

**Available Metrics:**
- `denial_rate` — % of claims denied
- `no_show_rate` — % of appointments missed
- `pmpm` — Per member per month cost
- `readmission_rate` — 30-day hospital readmission %
- `avg_los` — Average length of inpatient stay
- `mlr` — Medical loss ratio
- `generic_rate` — % prescriptions filled as generic
- `er_visits_per_1000` — Emergency visits per 1000 members

**Metric Properties:**
- `numerator_sql` — Template for numerator calculation
- `denominator_sql` — Template for denominator calculation
- `time_period` — "monthly", "quarterly", or "annual"
- `units` — "%", "count", "$", "days", "per 1000"
- `benchmark_threshold` — Performance levels

**Usage:**
```python
metric = intelligence.metric_engine.get_metric("denial_rate")
print(f"Name: {metric.name}")
print(f"Numerator SQL: {metric.numerator_sql}")
print(f"Denominator SQL: {metric.denominator_sql}")
print(f"Time Period: {metric.time_period}")
print(f"Units: {metric.units}")

# Grade a value
grade = intelligence.metric_engine.grade_metric_value("denial_rate", 0.12)
# Returns: "average" (between 5-10%)
```

### 4. ContextualInferenceEngine

Understands implicit meaning in questions:

**Intent Inference:**
- `performance_check` — "Are we doing well?" → Check all metrics vs benchmarks
- `anomaly_detection` — "What's concerning?" → Find metrics worse than benchmarks
- `regional_analysis` — "How's NCAL performing?" → Filter by region
- `quality_analysis` — "Quality focus?" → Check HEDIS, Stars, readmission
- `cost_analysis` — "What's expensive?" → Cost-related queries
- `general_query` — Default catchall

**Scope Inference:**
- Time period (monthly, quarterly, YTD, last 30 days)
- Geographic regions (NCAL, SCAL, Hawaii, Colorado)
- Member segments (diabetic, heart failure, elderly, pediatric, high-risk)

**Usage:**
```python
intent = intelligence.contextual_inference.infer_intent(
    "Are we doing well on quality metrics?"
)
print(f"Intent: {intent['inferred_intent']}")
print(f"Metrics to check: {intent['metrics_to_check']}")

scope = intelligence.contextual_inference.infer_scope(
    "Denial rate in NCAL for high-risk members last 90 days"
)
print(f"Region: {scope['regions']}")
print(f"Segment: {scope['member_segments']}")
```

### 5. JargonEntityExtractor

Maps natural language patterns to SQL WHERE clauses:

**Entity Types:**
- `member_group` — "diabetic members", "high-risk", "elderly", "pediatric"
- `claim_type` — "ER visit", "emergency", "inpatient"
- `claim_status` — "denied claims", "approved"
- `appointment_status` — "no-shows", "missed"
- `visit_type` — "telehealth", "outpatient", "inpatient"

**Pattern Examples:**
```
"diabetic members" →
  MEMBER_ID IN (SELECT DISTINCT MEMBER_ID FROM diagnoses WHERE ICD10_CODE LIKE 'E11%')

"high-risk members" →
  CAST(RISK_SCORE AS REAL) > 2.0

"ER visits" →
  VISIT_TYPE='EMERGENCY'

"elderly/65+" →
  DATE_OF_BIRTH <= DATE('now', '-65 years')
```

**Usage:**
```python
entities = intelligence.entity_extractor.extract_entities(
    "denied claims for diabetic members in ER"
)
# Returns list of matched EntityMapping objects

where_clause = intelligence.entity_extractor.build_where_clause(
    [entity_obj for _, entity_obj in entities]
)
# Returns: "(CLAIM_STATUS='DENIED') AND (...diabetes query...) AND (...ER query...)"
```

### 6. BenchmarkDatabase

Grades metrics against industry standards:

**Benchmark Thresholds:**
| Metric | Excellent | Average | Poor |
|--------|-----------|---------|------|
| Denial Rate | <5% | 5-10% | >15% |
| No-Show Rate | <5% | 5-12% | >20% |
| Readmission Rate | <10% | 10-15% | >20% |
| PMPM | <$350 | $350-500 | >$750 |
| Avg LOS | <3 days | 3-5 days | >7 days |
| Generic Rate | >90% | 80-90% | <80% |
| ER Visits/1000 | <250 | 250-400 | >500 |

**Usage:**
```python
# Grade a single value
grade = intelligence.benchmark_db.grade_performance("denial_rate", 0.08)
# Returns: "average"

# Get full comparison
comparison = intelligence.benchmark_db.get_peer_comparison("denial_rate", 0.12)
# Returns:
# {
#   "metric_id": "denial_rate",
#   "current_value": 0.12,
#   "grade": "poor",
#   "excellent_threshold": 0.05,
#   "average_range": (0.05, 0.10),
#   "poor_threshold": 0.15,
#   "percentile_estimate": 25.0
# }
```

### 7. CausalReasoningEngine

Explains "why" with structured investigation queries:

**Causal Patterns:**

For **high denial rate**:
- Distribution of denial reasons
- Denial rate by plan type
- Denial rate by region
- Top denying providers

For **high costs**:
- Identify high-cost members
- Analyze costs by diagnosis
- Check inpatient utilization

For **high no-shows**:
- No-show rate by department
- No-show rate by appointment type
- No-show rate by time of day

For **high readmissions**:
- Readmission rate by diagnosis
- Discharge against medical advice

**Usage:**
```python
explanation = intelligence.causal_reasoning.explain_metric("denial_rate")
# Returns:
# {
#   "metric_id": "denial_rate",
#   "investigation_queries": [
#     {"investigation": "Check denial reasons distribution",
#      "sql_template": "SELECT DENIAL_REASON, COUNT(*) ..."},
#     ...
#   ]
# }
```

## Integration with Healthcare LLM Pipeline

### Example: Full Question Analysis

```python
from domain_intelligence import create_domain_intelligence

intelligence = create_domain_intelligence()
question = "What's the denial rate trend in NCAL for high-risk members?"

# Full analysis
analysis = intelligence.analyze_question(question)

# Extract key components
normalized_q = analysis["normalized_question"]
intent = analysis["intent"]["inferred_intent"]
scope = analysis["scope"]
entities = analysis["entities"]
where_clause = analysis["sql_where_clause"]
metrics = analysis["related_metrics"]

# Build SQL
if metrics:
    metric_id = metrics[0]["metric_id"]
    metric = intelligence.get_metric_definition(metric_id)
    
    # Build WHERE clause from scope
    filters = []
    if scope["regions"]:
        filters.append(f"KP_REGION IN ({','.join([repr(r) for r in scope['regions']])})")
    
    filters.append(where_clause)
    
    final_where = " AND ".join(filters)
    
    # Build final SQL using metric template
    sql = f"""
        SELECT KP_REGION, DATE_TRUNC('month', SERVICE_DATE) as month,
               {metric.numerator_sql} as numerator,
               {metric.denominator_sql} as denominator
        FROM claims
        WHERE {final_where}
        GROUP BY KP_REGION, DATE_TRUNC('month', SERVICE_DATE)
        ORDER BY month DESC
    """
```

### Example: Intelligent Insight Generation

```python
# After calculating a metric value
metric_id = "denial_rate"
current_value = 0.14  # 14%

# Get benchmark comparison
benchmark = intelligence.benchmark_metric(metric_id, current_value)
grade = benchmark["grade"]  # "poor"

# Get causal explanation
explanation = intelligence.explain_metric_causes(metric_id)

# Generate narrative
if grade == "poor":
    narratives = [
        f"Our denial rate is {current_value*100:.1f}%, which is {grade} compared to industry benchmarks",
        "To understand why, we should investigate:",
    ]
    
    for query_info in explanation["investigation_queries"]:
        narratives.append(f"  - {query_info['investigation']}")
```

## Database Schema Integration

The module references these tables:

**claims** (60K rows)
- CLAIM_ID, MEMBER_ID, SERVICE_DATE, CPT_CODE, ICD10_CODE
- PAID_AMOUNT, CLAIM_STATUS, DENIAL_REASON
- KP_REGION, PLAN_TYPE

**members** (25K rows)
- MEMBER_ID, GENDER, RACE, KP_REGION, PLAN_TYPE
- RISK_SCORE, CHRONIC_CONDITIONS, DATE_OF_BIRTH

**encounters** (50K rows)
- ENCOUNTER_ID, MEMBER_ID, VISIT_TYPE
- LENGTH_OF_STAY, DEPARTMENT, ADMIT_DATE, DISCHARGE_DATE

**appointments** (10K rows)
- STATUS, DEPARTMENT, APPOINTMENT_TYPE, APPOINTMENT_TIME

**diagnoses** (20K rows)
- ICD10_CODE, HCC_CODE, HCC_CATEGORY, IS_CHRONIC, SEVERITY

**prescriptions** (12K rows)
- MEDICATION_NAME, MEDICATION_CLASS, COST, STATUS

## Adding New Concepts, Metrics, and Entities

### Add a Healthcare Concept

```python
from domain_intelligence import ConceptNode, create_domain_intelligence

intelligence = create_domain_intelligence()

# Create new concept
concept = ConceptNode(
    id="preventive_screening",
    name="Preventive Screening",
    aliases=["preventive care", "screening"],
    definition="Preventive health screenings and vaccinations",
    related_tables=["encounters"],
    related_columns=["VISIT_TYPE", "DEPARTMENT"],
    sql_pattern="DEPARTMENT IN ('Preventive Care', 'Wellness')",
    metadata={"quality_indicator": True, "hedis_relevant": True}
)

intelligence.ontology.add_concept(concept)
```

### Add a New KPI

```python
from domain_intelligence import MetricDefinition

metric = MetricDefinition(
    metric_id="preventive_visit_rate",
    name="Preventive Visit Rate",
    aliases=["preventive visits", "wellness visits"],
    numerator_sql="COUNT(CASE WHEN DEPARTMENT='Preventive Care' THEN 1 END)",
    denominator_sql="COUNT(DISTINCT MEMBER_ID)",
    time_period="annual",
    units="%",
    concept_id="preventive_screening"
)

intelligence.metric_engine.add_metric(metric)
```

### Add a New Entity Pattern

```python
from domain_intelligence import EntityMapping

entity = EntityMapping(
    entity_pattern=r"preventive\s+care|wellness\s+visit",
    entity_type="visit_type",
    sql_where_clause="DEPARTMENT='Preventive Care'",
    required_tables=["encounters"]
)

intelligence.entity_extractor.entity_mappings.append(entity)
```

## Performance Characteristics

- **Initialization**: ~50ms (builds all ontologies, metrics, benchmarks in-memory)
- **Question analysis**: ~5ms (regex matching, concept lookup)
- **Entity extraction**: ~2ms (pattern matching across 20+ patterns)
- **Benchmark lookup**: <1ms (dict lookup)
- **Memory footprint**: ~5MB (all dictionaries, concepts, metrics in RAM)

## No External Dependencies

Uses only stdlib + numpy:
- `re` — Regex pattern matching
- `json` — Data serialization
- `math` — Numeric calculations
- `dataclasses` — Data structure definitions
- `collections.defaultdict` — Efficient dict operations
- `numpy` — Numeric arrays (for potential ML integration)

## Testing

Run the module directly to test all components:

```bash
python3 /sessions/great-gallant-allen/mnt/chatbot/mtp_demo/scripts/domain_intelligence.py
```

Output:
```
=== DOMAIN INTELLIGENCE MODULE TEST ===

1. JARGON NORMALIZATION
  icd-10               → ICD10_CODE
  pmpm                 → PER_MEMBER_PER_MONTH
  ...

2. QUESTION ANALYSIS
  Original: What is the denial rate by region in NCAL for high-risk members?
  Intent: regional_analysis
  Entities found: 1
  Related metrics: 5
  ...

✓ Module loaded successfully!
```

## Next Steps

1. **Integration**: Wire into `intelligent_pipeline.py` for NLU preprocessing
2. **Extension**: Add more concepts (chronic conditions, transitions of care, etc.)
3. **Learning**: Track which jargon terms users employ most, add auto-learning
4. **Validation**: Cross-check extracted entities against actual schema
5. **Refinement**: Expand causal patterns with domain expertise
