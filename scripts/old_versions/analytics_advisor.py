"""
Auto-analyze data and recommend business actions with ready-to-run SQL.
Standalone module with Pure Python - no external dependencies beyond standard library.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class Impact(Enum):
    """Business impact level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AnalyticsCategory(Enum):
    """Business domain categories."""
    PATIENT_RETENTION = "patient_retention"
    PATIENT_ACQUISITION = "patient_acquisition"
    PREVENTIVE_CARE = "preventive_care"
    COST_OPTIMIZATION = "cost_optimization"
    OPERATIONS = "operations"
    POPULATION_HEALTH = "population_health"
    DATA_QUALITY = "data_quality"


@dataclass
class AnalyticSpec:
    """Specification for a single analytic."""
    category: str
    name: str
    description: str
    required_columns: List[str]
    sql_template: str
    impact: str  # HIGH, MEDIUM, LOW
    business_action: str


# Registry of 17 analytics with ready-to-run SQL templates
ANALYTICS_REGISTRY = [
    # PATIENT RETENTION (3)
    AnalyticSpec(
        category="patient_retention",
        name="Member Churn Analysis",
        description="Identify members at risk of disenrollment based on utilization patterns",
        required_columns=["member_id", "enrollment_end_date", "claim_date", "service_date"],
        sql_template="""
SELECT
  {member_col} as member_id,
  COUNT(DISTINCT {date_col}) as visit_days,
  COUNT(*) as claim_count,
  SUM({amount_col}) as total_cost,
  MAX({date_col}) as last_service_date,
  DATEDIFF(day, MAX({date_col}), GETDATE()) as days_since_visit,
  CASE
    WHEN DATEDIFF(day, MAX({date_col}), GETDATE()) > 90 THEN 'HIGH_RISK'
    WHEN DATEDIFF(day, MAX({date_col}), GETDATE()) > 30 THEN 'MEDIUM_RISK'
    ELSE 'ACTIVE'
  END as churn_risk
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {member_col}
ORDER BY days_since_visit DESC
        """,
        impact="HIGH",
        business_action="Implement targeted engagement program for high-risk members; increase touchpoints, offer incentives, review care gaps"
    ),
    AnalyticSpec(
        category="patient_retention",
        name="Visit Frequency Decline",
        description="Detect members with declining visit trends month-over-month",
        required_columns=["member_id", "service_date", "encounter_id"],
        sql_template="""
WITH monthly_visits AS (
  SELECT
    {member_col} as member_id,
    DATETRUNC(month, {date_col}) as visit_month,
    COUNT(DISTINCT {encounter_col}) as visit_count
  FROM {table}
  WHERE {date_col} >= DATEADD(month, -6, GETDATE())
  GROUP BY {member_col}, DATETRUNC(month, {date_col})
)
SELECT
  member_id,
  MAX(visit_count) as peak_visits,
  MIN(visit_count) as trough_visits,
  ROUND(100.0 * (MIN(visit_count) - MAX(visit_count)) / MAX(visit_count), 2) as decline_pct,
  COUNT(DISTINCT visit_month) as months_tracked
FROM monthly_visits
GROUP BY member_id
HAVING ROUND(100.0 * (MIN(visit_count) - MAX(visit_count)) / MAX(visit_count), 2) < -20
ORDER BY decline_pct ASC
        """,
        impact="HIGH",
        business_action="Conduct outreach calls; assess barriers to care; offer transportation, telehealth, or specialty referrals"
    ),
    AnalyticSpec(
        category="patient_retention",
        name="No-Show Analysis",
        description="Analyze no-show patterns and predict future no-show risk",
        required_columns=["member_id", "appointment_date", "visit_status", "provider_id"],
        sql_template="""
SELECT
  {member_col} as member_id,
  {provider_col} as provider_id,
  COUNT(*) as total_appointments,
  SUM(CASE WHEN visit_status = 'NO_SHOW' THEN 1 ELSE 0 END) as no_show_count,
  ROUND(100.0 * SUM(CASE WHEN visit_status = 'NO_SHOW' THEN 1 ELSE 0 END) / COUNT(*), 1) as no_show_pct,
  MAX({date_col}) as most_recent_appointment
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {member_col}, {provider_col}
HAVING SUM(CASE WHEN visit_status = 'NO_SHOW' THEN 1 ELSE 0 END) > 0
ORDER BY no_show_pct DESC
        """,
        impact="MEDIUM",
        business_action="Send appointment reminders; implement SMS/call confirmations; analyze scheduling conflicts; offer flexible times"
    ),

    # PATIENT ACQUISITION (3)
    AnalyticSpec(
        category="patient_acquisition",
        name="High-Demand Service Lines",
        description="Identify services with high utilization and unmet demand",
        required_columns=["service_code", "cpt_code", "claim_count", "member_count", "service_date"],
        sql_template="""
SELECT
  {service_col} as service,
  COUNT(DISTINCT {member_col}) as unique_members,
  COUNT(*) as total_claims,
  ROUND(COUNT(*) / CAST(COUNT(DISTINCT {member_col}) AS FLOAT), 2) as claims_per_member,
  SUM({amount_col}) as total_cost,
  ROUND(SUM({amount_col}) / COUNT(DISTINCT {member_col}), 2) as avg_cost_per_member
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {service_col}
ORDER BY claims_per_member DESC
        """,
        impact="MEDIUM",
        business_action="Expand capacity for high-demand services; recruit specialists; market services to adjacent populations; negotiate rates with vendors"
    ),
    AnalyticSpec(
        category="patient_acquisition",
        name="External Referral Leakage",
        description="Detect members being referred outside network for high-value services",
        required_columns=["member_id", "in_network", "out_of_network", "cost", "service_code"],
        sql_template="""
SELECT
  {member_col} as member_id,
  SUM(CASE WHEN in_network = 1 THEN {amount_col} ELSE 0 END) as in_network_cost,
  SUM(CASE WHEN in_network = 0 THEN {amount_col} ELSE 0 END) as out_network_cost,
  ROUND(100.0 * SUM(CASE WHEN in_network = 0 THEN {amount_col} ELSE 0 END) /
        (SUM(CASE WHEN in_network = 1 THEN {amount_col} ELSE 0 END) +
         SUM(CASE WHEN in_network = 0 THEN {amount_col} ELSE 0 END)), 1) as out_of_network_pct
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {member_col}
HAVING SUM(CASE WHEN in_network = 0 THEN {amount_col} ELSE 0 END) > 0
ORDER BY out_network_cost DESC
        """,
        impact="HIGH",
        business_action="Expand specialty network; invest in recruiting specialists; benchmark outcomes vs external providers; improve referral workflows"
    ),
    AnalyticSpec(
        category="patient_acquisition",
        name="Geographic Gap Analysis",
        description="Identify service areas with low utilization relative to population",
        required_columns=["zip_code", "region", "member_count", "claim_count", "service_date"],
        sql_template="""
SELECT
  {region_col} as region,
  COUNT(DISTINCT {member_col}) as enrolled_members,
  COUNT(*) as total_claims,
  ROUND(COUNT(*) / CAST(COUNT(DISTINCT {member_col}) AS FLOAT), 2) as claims_per_member,
  ROUND(100.0 * COUNT(DISTINCT {member_col}) / SUM(COUNT(DISTINCT {member_col})) OVER (), 1) as pct_of_enrollment
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {region_col}
ORDER BY claims_per_member ASC
        """,
        impact="MEDIUM",
        business_action="Open clinics or partner facilities in underserved areas; enhance transportation programs; increase digital health offerings"
    ),

    # PREVENTIVE CARE (2)
    AnalyticSpec(
        category="preventive_care",
        name="Preventive Screening Gap",
        description="Identify members due for preventive services (vaccines, screenings, wellness)",
        required_columns=["member_id", "age", "gender", "last_screening_date", "screening_code"],
        sql_template="""
SELECT
  {member_col} as member_id,
  DATEDIFF(year, dob, GETDATE()) as age,
  gender,
  screening_type,
  MAX({date_col}) as last_screening_date,
  DATEDIFF(day, MAX({date_col}), GETDATE()) as days_overdue,
  CASE
    WHEN screening_type = 'mammo' AND gender = 'F' AND DATEDIFF(year, dob, GETDATE()) >= 40
      AND DATEDIFF(day, MAX({date_col}), GETDATE()) > 365 THEN 'OVERDUE'
    WHEN screening_type = 'colonoscopy' AND DATEDIFF(year, dob, GETDATE()) >= 50
      AND DATEDIFF(day, MAX({date_col}), GETDATE()) > 1095 THEN 'OVERDUE'
    WHEN screening_type = 'flu_vaccine' AND DATEDIFF(day, MAX({date_col}), GETDATE()) > 365 THEN 'OVERDUE'
    ELSE 'CURRENT'
  END as status
FROM {table}
WHERE {date_col} >= DATEADD(year, -5, GETDATE())
GROUP BY {member_col}, dob, gender, screening_type
        """,
        impact="HIGH",
        business_action="Launch screening campaigns; mail incentive cards; integrate with EMR for automated reminders; partner with community health workers"
    ),
    AnalyticSpec(
        category="preventive_care",
        name="Chronic Disease Monitoring",
        description="Identify members with chronic conditions needing ongoing management and monitoring",
        required_columns=["member_id", "diagnosis_code", "icd10", "last_visit_date", "medication"],
        sql_template="""
SELECT
  {member_col} as member_id,
  diagnosis_code,
  COUNT(DISTINCT {encounter_col}) as visit_count,
  MAX({date_col}) as last_visit_date,
  DATEDIFF(day, MAX({date_col}), GETDATE()) as days_since_visit,
  COUNT(DISTINCT medication_id) as unique_meds,
  CASE
    WHEN DATEDIFF(day, MAX({date_col}), GETDATE()) > 60 THEN 'NEEDS_ATTENTION'
    ELSE 'MANAGED'
  END as management_status
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
  AND diagnosis_code IN ('E11', 'I10', 'J44')  -- Type 2 Diabetes, Hypertension, COPD
GROUP BY {member_col}, diagnosis_code
ORDER BY days_since_visit DESC
        """,
        impact="HIGH",
        business_action="Enroll in disease management programs; increase visit frequency; optimize medication therapy; provide care coordinator support"
    ),

    # COST OPTIMIZATION (3)
    AnalyticSpec(
        category="cost_optimization",
        name="High-Cost Member Identification",
        description="Identify top utilizers and members with catastrophic costs for intervention",
        required_columns=["member_id", "total_cost", "claim_count", "service_date"],
        sql_template="""
WITH member_costs AS (
  SELECT
    {member_col} as member_id,
    SUM({amount_col}) as total_cost,
    COUNT(*) as claim_count,
    COUNT(DISTINCT {date_col}) as service_days,
    SUM(CASE WHEN icd10_code LIKE 'E11%' THEN 1 ELSE 0 END) as diabetes_claims,
    SUM(CASE WHEN icd10_code LIKE 'I10%' THEN 1 ELSE 0 END) as hypertension_claims
  FROM {table}
  WHERE {date_col} >= DATEADD(year, -1, GETDATE())
  GROUP BY {member_col}
)
SELECT
  member_id,
  total_cost,
  claim_count,
  ROUND(total_cost / claim_count, 2) as avg_cost_per_claim,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_cost) OVER () as p75_cost,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_cost) OVER () as p90_cost,
  diabetes_claims,
  hypertension_claims,
  CASE
    WHEN total_cost > PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_cost) OVER () THEN 'CATASTROPHIC'
    WHEN total_cost > PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_cost) OVER () THEN 'HIGH'
    ELSE 'STANDARD'
  END as cost_tier
FROM member_costs
ORDER BY total_cost DESC
        """,
        impact="HIGH",
        business_action="Assign case managers; implement care coordination; optimize medication therapy; negotiate specialty care rates; offer disease management programs"
    ),
    AnalyticSpec(
        category="cost_optimization",
        name="ER Non-Emergent Utilization",
        description="Detect unnecessary ER visits that could be managed in primary care",
        required_columns=["member_id", "facility_type", "visit_reason", "primary_diagnosis", "claim_cost"],
        sql_template="""
SELECT
  {member_col} as member_id,
  primary_diagnosis,
  COUNT(*) as er_visits,
  SUM({amount_col}) as total_er_cost,
  ROUND(SUM({amount_col}) / COUNT(*), 2) as avg_cost_per_visit,
  CASE
    WHEN primary_diagnosis IN ('URI', 'Migraine', 'Minor_Injury') THEN 'POTENTIALLY_NON_EMERGENT'
    ELSE 'LIKELY_EMERGENT'
  END as appropriateness
FROM {table}
WHERE facility_type = 'EMERGENCY_DEPT'
  AND {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {member_col}, primary_diagnosis
HAVING COUNT(*) > 1
ORDER BY total_er_cost DESC
        """,
        impact="MEDIUM",
        business_action="Route to urgent care; strengthen PCP relationships; implement copay strategies; improve after-hours access; provide nurse hotline"
    ),
    AnalyticSpec(
        category="cost_optimization",
        name="Hospital Readmission Cost",
        description="Identify members with frequent readmissions and their cost impact",
        required_columns=["member_id", "admission_date", "discharge_date", "readmission_within_30", "total_cost"],
        sql_template="""
WITH admissions AS (
  SELECT
    {member_col} as member_id,
    admission_date,
    discharge_date,
    SUM({amount_col}) as admission_cost,
    LEAD(admission_date) OVER (PARTITION BY {member_col} ORDER BY admission_date) as next_admission,
    DATEDIFF(day, discharge_date,
      LEAD(admission_date) OVER (PARTITION BY {member_col} ORDER BY admission_date)) as days_to_readmission
  FROM {table}
  WHERE facility_type = 'HOSPITAL'
  GROUP BY {member_col}, admission_date, discharge_date
)
SELECT
  member_id,
  COUNT(*) as admission_count,
  SUM(admission_cost) as total_admission_cost,
  SUM(CASE WHEN days_to_readmission <= 30 THEN 1 ELSE 0 END) as readmissions_30day,
  ROUND(100.0 * SUM(CASE WHEN days_to_readmission <= 30 THEN 1 ELSE 0 END) /
        NULLIF(COUNT(*), 0), 1) as readmission_rate_pct
FROM admissions
GROUP BY member_id
HAVING COUNT(*) > 1
ORDER BY readmission_rate_pct DESC
        """,
        impact="HIGH",
        business_action="Deploy transitional care team; enhance discharge planning; increase post-discharge follow-up; optimize medication reconciliation; implement remote monitoring"
    ),

    # OPERATIONS (2)
    AnalyticSpec(
        category="operations",
        name="Provider Workload Analysis",
        description="Analyze provider utilization, capacity, and burnout risk indicators",
        required_columns=["provider_id", "npi", "encounter_date", "patient_count", "visit_count"],
        sql_template="""
SELECT
  {provider_col} as provider_id,
  COUNT(DISTINCT {member_col}) as unique_patients,
  COUNT(*) as total_encounters,
  COUNT(DISTINCT {date_col}) as service_days,
  ROUND(COUNT(*) / CAST(COUNT(DISTINCT {date_col}) AS FLOAT), 2) as encounters_per_day,
  COUNT(DISTINCT DATEPART(week, {date_col})) as weeks_worked,
  ROUND(COUNT(*) / CAST(COUNT(DISTINCT DATEPART(week, {date_col})) AS FLOAT), 2) as encounters_per_week,
  CASE
    WHEN ROUND(COUNT(*) / CAST(COUNT(DISTINCT {date_col}) AS FLOAT), 2) > 20 THEN 'HIGH_LOAD'
    WHEN ROUND(COUNT(*) / CAST(COUNT(DISTINCT {date_col}) AS FLOAT), 2) > 15 THEN 'MEDIUM_LOAD'
    ELSE 'HEALTHY_LOAD'
  END as workload_status
FROM {table}
WHERE {date_col} >= DATEADD(month, -3, GETDATE())
GROUP BY {provider_col}
ORDER BY encounters_per_day DESC
        """,
        impact="MEDIUM",
        business_action="Rebalance patient panels; hire additional providers; implement team-based care; increase support staff; offer flexible schedules"
    ),
    AnalyticSpec(
        category="operations",
        name="Telehealth Adoption",
        description="Track telehealth utilization and identify expansion opportunities",
        required_columns=["visit_type", "encounter_date", "member_id", "provider_id", "visit_modality"],
        sql_template="""
WITH visit_types AS (
  SELECT
    {date_col} as visit_date,
    DATETRUNC(month, {date_col}) as visit_month,
    visit_modality,
    COUNT(*) as visit_count,
    COUNT(DISTINCT {member_col}) as unique_members
  FROM {table}
  WHERE {date_col} >= DATEADD(month, -12, GETDATE())
  GROUP BY DATETRUNC(month, {date_col}), visit_modality, {date_col}
)
SELECT
  visit_month,
  visit_modality,
  visit_count,
  unique_members,
  ROUND(100.0 * visit_count / SUM(visit_count) OVER (PARTITION BY visit_month), 1) as pct_of_month,
  LAG(visit_count) OVER (PARTITION BY visit_modality ORDER BY visit_month) as prior_month_count,
  ROUND(100.0 * (visit_count - LAG(visit_count) OVER (PARTITION BY visit_modality ORDER BY visit_month)) /
        LAG(visit_count) OVER (PARTITION BY visit_modality ORDER BY visit_month), 1) as mom_growth_pct
FROM visit_types
ORDER BY visit_month DESC, visit_modality
        """,
        impact="MEDIUM",
        business_action="Expand telehealth infrastructure; train providers; market telehealth benefits; invest in patient portal; negotiate rates with payers"
    ),

    # POPULATION HEALTH (2)
    AnalyticSpec(
        category="population_health",
        name="Disease Prevalence Analysis",
        description="Analyze prevalence of major disease categories across population",
        required_columns=["member_id", "diagnosis_code", "icd10", "service_date"],
        sql_template="""
SELECT
  diagnosis_code,
  COUNT(DISTINCT {member_col}) as affected_members,
  ROUND(100.0 * COUNT(DISTINCT {member_col}) /
        (SELECT COUNT(DISTINCT {member_col}) FROM {table}), 1) as prevalence_pct,
  ROUND(AVG({amount_col}), 2) as avg_cost_per_member,
  COUNT(*) as total_claims,
  MAX({date_col}) as most_recent_case
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY diagnosis_code
ORDER BY affected_members DESC
        """,
        impact="HIGH",
        business_action="Develop disease management programs; allocate resources to high-prevalence conditions; design prevention campaigns; benchmark against national rates"
    ),
    AnalyticSpec(
        category="population_health",
        name="Social Determinants of Health Proxy",
        description="Use claims data proxies to identify SDOH risk factors",
        required_columns=["member_id", "transportation", "housing", "zip_code", "income_quartile"],
        sql_template="""
SELECT
  {member_col} as member_id,
  zip_code,
  income_quartile,
  SUM(CASE WHEN service_type = 'TRANSPORTATION' THEN 1 ELSE 0 END) as transportation_claims,
  SUM(CASE WHEN service_type = 'HOUSING_RELATED' THEN 1 ELSE 0 END) as housing_claims,
  SUM({amount_col}) as total_sdoh_spending,
  SUM(CASE WHEN service_type IN ('ER', 'HOSPITALIZATION') THEN {amount_col} ELSE 0 END) as acute_care_cost,
  ROUND(SUM(CASE WHEN service_type IN ('ER', 'HOSPITALIZATION') THEN {amount_col} ELSE 0 END) /
        SUM({amount_col}), 2) as acute_cost_ratio
FROM {table}
WHERE {date_col} >= DATEADD(year, -1, GETDATE())
GROUP BY {member_col}, zip_code, income_quartile
HAVING SUM({amount_col}) > 0
ORDER BY acute_cost_ratio DESC
        """,
        impact="MEDIUM",
        business_action="Partner with community organizations; address transportation barriers; invest in housing stability; coordinate social services; close SDOH gaps"
    ),

    # DATA QUALITY (2)
    AnalyticSpec(
        category="data_quality",
        name="Completeness Scorecard",
        description="Measure data completeness and missing value rates by column",
        required_columns=["*"],  # Requires all columns
        sql_template="""
SELECT
  column_name,
  COUNT(*) as total_rows,
  COUNT(CASE WHEN value IS NOT NULL THEN 1 END) as non_null_count,
  COUNT(CASE WHEN value IS NULL THEN 1 END) as null_count,
  ROUND(100.0 * COUNT(CASE WHEN value IS NOT NULL THEN 1 END) / COUNT(*), 1) as completeness_pct,
  CASE
    WHEN ROUND(100.0 * COUNT(CASE WHEN value IS NOT NULL THEN 1 END) / COUNT(*), 1) >= 95 THEN 'EXCELLENT'
    WHEN ROUND(100.0 * COUNT(CASE WHEN value IS NOT NULL THEN 1 END) / COUNT(*), 1) >= 80 THEN 'GOOD'
    WHEN ROUND(100.0 * COUNT(CASE WHEN value IS NOT NULL THEN 1 END) / COUNT(*), 1) >= 60 THEN 'FAIR'
    ELSE 'POOR'
  END as quality_rating
FROM {table}
GROUP BY column_name
ORDER BY completeness_pct ASC
        """,
        impact="HIGH",
        business_action="Audit upstream data sources; implement validation rules; improve data capture workflows; schedule remediation sprints; establish data governance"
    ),
    AnalyticSpec(
        category="data_quality",
        name="Duplicate Detection",
        description="Identify potential duplicate records within datasets",
        required_columns=["member_id", "name", "dob", "claim_id"],
        sql_template="""
WITH suspect_dupes AS (
  SELECT
    {member_col} as member_id,
    name,
    dob,
    COUNT(*) as record_count,
    COUNT(DISTINCT {member_col}) as unique_ids,
    STRING_AGG(CAST({member_col} AS VARCHAR), ',') as member_ids
  FROM {table}
  WHERE {date_col} >= DATEADD(year, -1, GETDATE())
  GROUP BY name, dob
  HAVING COUNT(*) > COUNT(DISTINCT {member_col})
)
SELECT
  member_id,
  name,
  dob,
  record_count,
  unique_ids,
  member_ids,
  record_count - 1 as potential_duplicates
FROM suspect_dupes
ORDER BY record_count DESC
        """,
        impact="HIGH",
        business_action="Run deduplication scripts; merge duplicate records; implement identity resolution rules; improve data entry controls; establish master data management"
    ),
]


class AnalyticsCatalog:
    """Registry of available analytics."""

    def __init__(self):
        self.analytics = ANALYTICS_REGISTRY

    def get_by_category(self, category: str) -> List[AnalyticSpec]:
        """Get all analytics for a category."""
        return [a for a in self.analytics if a.category == category]

    def get_all(self) -> List[AnalyticSpec]:
        """Get all analytics."""
        return self.analytics

    def find_by_name(self, name: str) -> Optional[AnalyticSpec]:
        """Find a single analytic by name."""
        for a in self.analytics:
            if name.lower() in a.name.lower():
                return a
        return None


# Pattern matching for analytics questions
ADVISOR_PATTERNS = {
    "all_analytics": [
        r"what\s+analytics\s+can\s+we\s+do",
        r"analytics\s+on\s+.*",
        r"what\s+recomm",
        r"all\s+recomm",
        r"available\s+analyt",
    ],
    "retention": [
        r"increase\s+retention",
        r"reduce\s+churn",
        r"keep\s+patients",
        r"patient\s+engagement",
        r"member\s+loyalty",
        r"no.?show",
    ],
    "acquisition": [
        r"attract\s+.*patient",
        r"grow\s+.*member",
        r"new\s+patient",
        r"patient\s+acquisition",
        r"expand\s+market",
        r"geographic",
        r"leakage",
    ],
    "preventive": [
        r"preventive",
        r"screening",
        r"wellness",
        r"chronic\s+disease",
        r"monitoring",
    ],
    "cost": [
        r"reduce\s+cost",
        r"cost\s+optim",
        r"high.*cost.*member",
        r"readmission",
        r"er\s+utilization",
        r"financial\s+risk",
    ],
    "operations": [
        r"provider.*workload",
        r"telehealth",
        r"operations",
        r"capacity",
        r"staff",
    ],
    "population": [
        r"population\s+health",
        r"disease\s+prevalence",
        r"sdoh",
        r"social\s+determ",
    ],
    "quality": [
        r"data\s+quality",
        r"dq",
        r"duplicate",
        r"completeness",
        r"data\s+validation",
    ],
}


class AnalyticsAdvisor:
    """
    Recommend analytics and business actions based on available data.
    """

    def __init__(self, catalog: Optional[Any] = None):
        self.catalog = AnalyticsCatalog()
        self.semantic_catalog = catalog
        self.available_columns = self._detect_available_columns()

    def _detect_available_columns(self) -> List[str]:
        """Scan semantic catalog for available column types."""
        columns = []

        if self.semantic_catalog and hasattr(self.semantic_catalog, "tables"):
            for table in self.semantic_catalog.tables:
                if hasattr(table, "columns"):
                    for col in table.columns:
                        col_name = getattr(col, "column_name", "")
                        semantic_type = getattr(col, "semantic_type", "")
                        healthcare_type = getattr(col, "healthcare_type", "")

                        # Add both specific and generic names
                        if col_name:
                            columns.append(col_name.lower())
                        if semantic_type:
                            columns.append(semantic_type.lower())
                        if healthcare_type:
                            columns.append(healthcare_type.lower())

        return list(set(columns))

    def _can_run_analytic(self, analytic: AnalyticSpec, available_cols: List[str]) -> bool:
        """Check if required columns exist for an analytic."""
        if "*" in analytic.required_columns:
            return True

        for req_col in analytic.required_columns:
            req_lower = req_col.lower()
            # Check if any available column contains the required column concept
            if not any(req_lower in avail for avail in available_cols):
                return False

        return True

    def _fill_sql_template(
        self, analytic: AnalyticSpec, available_cols: List[str]
    ) -> str:
        """Fill SQL template with actual column names."""
        sql = analytic.sql_template

        # Replace common placeholders with defaults
        replacements = {
            "{table}": "analytics_table",
            "{member_col}": "member_id",
            "{date_col}": "service_date",
            "{amount_col}": "allowed_amount",
            "{provider_col}": "provider_id",
            "{encounter_col}": "encounter_id",
            "{service_col}": "service_code",
            "{region_col}": "region",
        }

        for placeholder, default in replacements.items():
            sql = sql.replace(placeholder, default)

        return sql

    def get_all_recommendations(self) -> List[Dict[str, Any]]:
        """Get all recommendations, prioritized by feasibility and impact."""
        recommendations = []

        for analytic in self.catalog.get_all():
            can_run = self._can_run_analytic(analytic, self.available_columns)
            impact_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(analytic.impact, 0)
            priority = (can_run, impact_score)

            sql = self._fill_sql_template(analytic, self.available_columns) if can_run else ""

            recommendations.append({
                "name": analytic.name,
                "category": analytic.category,
                "description": analytic.description,
                "can_run": can_run,
                "impact": analytic.impact,
                "business_action": analytic.business_action,
                "sql_template": sql,
                "priority": priority[0] and priority[1],
            })

        # Sort by feasibility first, then impact
        recommendations.sort(key=lambda x: (not x["can_run"], -{"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["impact"], 0)))
        return recommendations

    def get_category_recommendations(self, category: str) -> List[Dict[str, Any]]:
        """Get recommendations for a specific category."""
        all_recs = self.get_all_recommendations()
        return [r for r in all_recs if r["category"] == category]

    def is_analytics_question(self, question: str) -> Optional[str]:
        """Classify question type; return category or None."""
        question_lower = question.lower()

        for category, patterns in ADVISOR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return category

        return None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer an analytics question.
        Returns {answer: str, result_data: list}
        """
        question_type = self.is_analytics_question(question)

        if question_type is None:
            return {
                "answer": "Unable to classify as an analytics question.",
                "result_data": []
            }

        if question_type == "all_analytics":
            recs = self.get_all_recommendations()
        elif question_type in ["retention", "acquisition", "preventive", "cost", "operations", "population", "quality"]:
            recs = self.get_category_recommendations(question_type)
        else:
            recs = self.get_all_recommendations()

        # Filter to runnable analytics
        runnable = [r for r in recs if r["can_run"]]
        blocked = [r for r in recs if not r["can_run"]]

        answer_parts = [f"Analytics Recommendations ({question_type})"]

        if runnable:
            answer_parts.append(f"\nReady to Run ({len(runnable)}):")
            for i, rec in enumerate(runnable[:5], 1):
                answer_parts.append(f"  {i}. {rec['name']} ({rec['impact']})")
                answer_parts.append(f"     {rec['description']}")
                answer_parts.append(f"     Action: {rec['business_action']}")

        if blocked:
            answer_parts.append(f"\nNeed More Data ({len(blocked)}):")
            for rec in blocked[:3]:
                answer_parts.append(f"  - {rec['name']}")

        return {
            "answer": "\n".join(answer_parts),
            "result_data": runnable[:5]  # Top 5 runnable
        }


if __name__ == "__main__":
    # Demo usage
    advisor = AnalyticsAdvisor()

    print("AnalyticsAdvisor Demo")
    print("=" * 60)

    print("\n1. All Recommendations:")
    recs = advisor.get_all_recommendations()
    print(f"Total analytics available: {len(recs)}")
    print(f"Runnable: {sum(1 for r in recs if r['can_run'])}")
    print(f"Blocked: {sum(1 for r in recs if not r['can_run'])}")

    print("\n2. Sample Query - Retention Analytics:")
    result = advisor.answer_question("How can we reduce member churn?")
    print(result["answer"])

    print("\n3. Sample Query - Cost Optimization:")
    result = advisor.answer_question("What can we do to reduce costs?")
    print(result["answer"])

    print("\n4. Category - Preventive Care:")
    recs = advisor.get_category_recommendations("preventive_care")
    for rec in recs:
        print(f"  - {rec['name']}: {rec['description']}")

    print("\n5. SQL Template Example:")
    if advisor.catalog.get_all():
        spec = advisor.catalog.get_all()[0]
        sql = advisor._fill_sql_template(spec, advisor.available_columns)
        print(f"Analytics: {spec.name}")
        print(f"SQL (first 300 chars):\n{sql[:300]}...")
