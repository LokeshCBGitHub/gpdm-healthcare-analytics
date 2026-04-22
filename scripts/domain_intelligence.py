import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class ConceptNode:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    definition: str = ""
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)
    related_columns: List[str] = field(default_factory=list)
    sql_pattern: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    metric_id: str
    name: str
    aliases: List[str]
    numerator_sql: str
    denominator_sql: Optional[str]
    time_period: str
    units: str
    concept_id: Optional[str] = None
    benchmark_threshold: Optional[Dict[str, float]] = None


@dataclass
class Benchmark:
    metric_id: str
    excellent: float
    average_low: float
    average_high: float
    poor: float


@dataclass
class EntityMapping:
    entity_pattern: str
    entity_type: str
    sql_where_clause: str
    required_tables: List[str]
    aliases: List[str] = field(default_factory=list)


class HealthcareDomainKnowledge:

    def __init__(self):
        self.doctor_jargon = self._init_doctor_jargon()
        self.patient_jargon = self._init_patient_jargon()
        self.insurance_jargon = self._init_insurance_jargon()
        self.finance_jargon = self._init_finance_jargon()
        self.all_mappings = {}
        self._build_all_mappings()

    def _init_doctor_jargon(self) -> Dict[str, str]:
        return {
            "icd-10": "ICD10_CODE",
            "icd10": "ICD10_CODE",
            "cpt": "CPT_CODE",
            "cpt code": "CPT_CODE",
            "drg": "DRG_CODE",
            "diagnosis related group": "DRG_CODE",
            "los": "LENGTH_OF_STAY",
            "length of stay": "LENGTH_OF_STAY",
            "ama": "AGAINST_MEDICAL_ADVICE",
            "against medical advice": "AGAINST_MEDICAL_ADVICE",
            "acs": "ACUTE_CORONARY_SYNDROME",
            "acute coronary syndrome": "ACUTE_CORONARY_SYNDROME",
            "hcc": "HCC_CODE",
            "hierarchical condition category": "HCC_CODE",
            "raf": "RISK_ADJUSTMENT_FACTOR",
            "risk adjustment factor": "RISK_ADJUSTMENT_FACTOR",
            "mrn": "MEMBER_ID",
            "medical record number": "MEMBER_ID",
            "npi": "PROVIDER_NPI",
            "national provider identifier": "PROVIDER_NPI",
            "attending": "ATTENDING_PHYSICIAN",
            "pcp": "PRIMARY_CARE_PHYSICIAN",
            "primary care physician": "PRIMARY_CARE_PHYSICIAN",
            "specialist": "SPECIALIST_PHYSICIAN",
            "consult": "CONSULTATION",
            "admit": "ADMISSION",
            "admission": "ADMISSION",
            "d/c": "DISCHARGE",
            "dc": "DISCHARGE",
            "discharge": "DISCHARGE",
            "chief complaint": "CHIEF_COMPLAINT",
            "ros": "REVIEW_OF_SYSTEMS",
            "review of systems": "REVIEW_OF_SYSTEMS",
            "hpi": "HISTORY_OF_PRESENT_ILLNESS",
            "history of present illness": "HISTORY_OF_PRESENT_ILLNESS",
            "assessment": "ASSESSMENT_PLAN",
            "plan": "ASSESSMENT_PLAN",
            "assessment/plan": "ASSESSMENT_PLAN",
        }

    def _init_patient_jargon(self) -> Dict[str, str]:
        return {
            "copay": "COPAYMENT",
            "co-pay": "COPAYMENT",
            "deductible": "DEDUCTIBLE",
            "oop": "OUT_OF_POCKET",
            "out of pocket": "OUT_OF_POCKET",
            "out-of-pocket": "OUT_OF_POCKET",
            "prior auth": "PRIOR_AUTHORIZATION",
            "prior authorization": "PRIOR_AUTHORIZATION",
            "referral": "REFERRAL",
            "er visit": "EMERGENCY_VISIT",
            "emergency room": "EMERGENCY_VISIT",
            "er": "EMERGENCY_VISIT",
            "urgent care": "URGENT_CARE_VISIT",
            "prescription": "PRESCRIPTION",
            "rx": "PRESCRIPTION",
            "generic": "GENERIC_MEDICATION",
            "brand": "BRAND_MEDICATION",
            "brand name": "BRAND_MEDICATION",
            "formulary": "FORMULARY",
            "covered": "COVERED_MEDICATION",
            "not covered": "NOT_COVERED_MEDICATION",
            "in-network": "IN_NETWORK_PROVIDER",
            "out of network": "OUT_OF_NETWORK_PROVIDER",
        }

    def _init_insurance_jargon(self) -> Dict[str, str]:
        return {
            "capitation": "CAPITATED_PAYMENT",
            "capitated": "CAPITATED_PAYMENT",
            "adjudication": "CLAIM_ADJUDICATION",
            "eob": "EXPLANATION_OF_BENEFITS",
            "explanation of benefits": "EXPLANATION_OF_BENEFITS",
            "medical necessity": "MEDICAL_NECESSITY_REVIEW",
            "utilization review": "UTILIZATION_REVIEW",
            "ur": "UTILIZATION_REVIEW",
            "case management": "CASE_MANAGEMENT",
            "credentialing": "PROVIDER_CREDENTIALING",
            "network": "PROVIDER_NETWORK",
            "in-network": "IN_NETWORK",
            "onn": "OUT_OF_NETWORK",
            "out of network": "OUT_OF_NETWORK",
            "allowed amount": "ALLOWED_AMOUNT",
            "member responsibility": "MEMBER_COINSURANCE",
            "coinsurance": "MEMBER_COINSURANCE",
            "claim denial": "CLAIM_DENIAL",
            "denial": "CLAIM_DENIAL",
            "claim status": "CLAIM_STATUS",
            "pending": "CLAIM_PENDING",
            "approved": "CLAIM_APPROVED",
            "denied": "CLAIM_DENIED",
            "appeal": "CLAIM_APPEAL",
        }

    def _init_finance_jargon(self) -> Dict[str, str]:
        return {
            "pmpm": "PER_MEMBER_PER_MONTH",
            "per member per month": "PER_MEMBER_PER_MONTH",
            "mlr": "MEDICAL_LOSS_RATIO",
            "medical loss ratio": "MEDICAL_LOSS_RATIO",
            "actuarial": "ACTUARIAL_ANALYSIS",
            "risk corridor": "RISK_CORRIDOR",
            "risk adjustment": "RISK_ADJUSTMENT",
            "prospective": "PROSPECTIVE_ANALYSIS",
            "retrospective": "RETROSPECTIVE_ANALYSIS",
            "ffs": "FEE_FOR_SERVICE",
            "fee for service": "FEE_FOR_SERVICE",
            "reinsurance": "REINSURANCE",
            "stop loss": "STOP_LOSS",
            "stop-loss": "STOP_LOSS",
            "premium": "PREMIUM_AMOUNT",
            "medical spend": "MEDICAL_SPEND",
        }

    def _build_all_mappings(self):
        self.all_mappings = {}
        for source in [self.doctor_jargon, self.patient_jargon, self.insurance_jargon, self.finance_jargon]:
            self.all_mappings.update(source)

    def normalize_term(self, term: str) -> str:
        term_lower = term.lower().strip()
        return self.all_mappings.get(term_lower, term_lower)

    def get_all_aliases_for(self, canonical_form: str) -> List[str]:
        return [k for k, v in self.all_mappings.items() if v == canonical_form]

    def is_abbreviation(self, term: str) -> bool:
        return term.upper() in self.all_mappings or term.lower() in self.all_mappings


class ConceptOntology:

    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self._build_ontology()

    def _build_ontology(self):
        self.add_concept(ConceptNode(
            id="claim_denial",
            name="Claim Denial",
            aliases=["denied claim", "denial", "claim rejected"],
            definition="A claim that was not approved for payment",
            related_tables=["claims"],
            related_columns=["CLAIM_STATUS", "DENIAL_REASON"],
            sql_pattern="CLAIM_STATUS='DENIED'",
            metadata={
                "query_type": "filter",
                "measurable": True,
                "related_metrics": ["denial_rate", "denial_rate_by_reason"]
            }
        ))

        self.add_concept(ConceptNode(
            id="readmission",
            name="Readmission",
            aliases=["hospital readmission", "unplanned readmission", "30-day readmission"],
            definition="Patient admitted to hospital within 30 days of prior discharge",
            related_tables=["encounters"],
            related_columns=["MEMBER_ID", "VISIT_TYPE", "ADMIT_DATE", "DISCHARGE_DATE"],
            sql_pattern="VISIT_TYPE='INPATIENT' AND ADMIT_DATE <= DATE(DISCHARGE_DATE, '+30 days')",
            metadata={
                "query_type": "join_temporal",
                "measurable": True,
                "related_metrics": ["readmission_rate"],
                "quality_indicator": True
            }
        ))

        self.add_concept(ConceptNode(
            id="no_show",
            name="No-Show",
            aliases=["patient no-show", "missed appointment", "absence"],
            definition="Scheduled appointment that patient did not attend",
            related_tables=["appointments"],
            related_columns=["STATUS"],
            sql_pattern="STATUS='NO_SHOW'",
            metadata={
                "query_type": "filter",
                "measurable": True,
                "related_metrics": ["no_show_rate"]
            }
        ))

        self.add_concept(ConceptNode(
            id="hedis",
            name="HEDIS Quality Measure",
            aliases=["hedis", "healthcare effectiveness", "quality measure"],
            definition="Healthcare Effectiveness Data and Information Set measures (CMS/NCQA)",
            related_tables=["encounters", "members", "diagnoses"],
            related_columns=["VISIT_TYPE", "MEMBER_ID", "ICD10_CODE"],
            metadata={
                "query_type": "aggregate",
                "measurable": True,
                "quality_indicator": True,
                "examples": ["well visits", "preventive screenings", "chronic care management"]
            }
        ))

        self.add_concept(ConceptNode(
            id="stars_rating",
            name="CMS Star Rating",
            aliases=["stars", "cms stars", "5-star rating", "quality rating"],
            definition="CMS 5-star quality rating system for Medicare Advantage plans",
            related_tables=["members", "encounters"],
            metadata={
                "query_type": "aggregate",
                "measurable": True,
                "quality_indicator": True,
                "regulatory": True
            }
        ))

        self.add_concept(ConceptNode(
            id="risk_score",
            name="Risk Score",
            aliases=["hcc risk score", "risk adjustment", "member risk"],
            definition="Hierarchical Condition Category-based risk adjustment score (0-5+)",
            related_tables=["members"],
            related_columns=["RISK_SCORE", "HCC_CODE"],
            sql_pattern="CAST(RISK_SCORE AS REAL) > 2.0",
            metadata={
                "query_type": "filter",
                "measurable": True,
                "high_risk_threshold": 2.0
            }
        ))

        self.add_concept(ConceptNode(
            id="utilization",
            name="Healthcare Utilization",
            aliases=["utilization", "encounter volume", "service utilization"],
            definition="Volume of healthcare services provided (visits, procedures, inpatient days)",
            related_tables=["encounters"],
            related_columns=["ENCOUNTER_ID", "VISIT_TYPE", "LENGTH_OF_STAY"],
            metadata={
                "query_type": "aggregate",
                "measurable": True,
                "includes": ["ED visits", "inpatient days", "specialist visits"]
            }
        ))

        self.add_concept(ConceptNode(
            id="cost_analysis",
            name="Cost Analysis",
            aliases=["cost", "spending", "medical spend", "total cost"],
            definition="Analysis of healthcare expenditures",
            related_tables=["claims"],
            related_columns=["PAID_AMOUNT", "SERVICE_DATE"],
            metadata={
                "query_type": "aggregate",
                "measurable": True
            }
        ))

        self.add_concept(ConceptNode(
            id="kp_region",
            name="Kaiser Permanente Region",
            aliases=["region", "kp region", "service area"],
            definition="Geographical region/service area",
            related_tables=["members", "claims", "encounters"],
            related_columns=["KP_REGION"],
            metadata={
                "query_type": "filter",
                "examples": ["NCAL", "SCAL", "Hawaii", "Colorado"]
            }
        ))

    def add_concept(self, concept: ConceptNode):
        self.concepts[concept.id] = concept

    def get_concept(self, concept_id_or_alias: str) -> Optional[ConceptNode]:
        if concept_id_or_alias in self.concepts:
            return self.concepts[concept_id_or_alias]

        for concept in self.concepts.values():
            if concept_id_or_alias.lower() in [a.lower() for a in concept.aliases]:
                return concept

        return None

    def find_related_concepts(self, concept_id: str) -> List[str]:
        concept = self.concepts.get(concept_id)
        if not concept:
            return []

        related = set()
        related.update(concept.parent_concepts)
        related.update(concept.child_concepts)

        for other in self.concepts.values():
            if other.id == concept_id:
                continue
            if set(other.related_tables) & set(concept.related_tables):
                related.add(other.id)

        return list(related)


class BusinessMetricEngine:

    def __init__(self):
        self.metrics: Dict[str, MetricDefinition] = {}
        self.benchmarks: Dict[str, Benchmark] = {}
        self._build_metrics()
        self._build_benchmarks()

    def _build_metrics(self):

        self.add_metric(MetricDefinition(
            metric_id="denial_rate",
            name="Claim Denial Rate",
            aliases=["denial rate", "denial %", "claims denied"],
            numerator_sql="COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END)",
            denominator_sql="COUNT(*)",
            time_period="monthly",
            units="%",
            concept_id="claim_denial",
            benchmark_threshold={"excellent": 0.05, "average": 0.10, "poor": 0.15}
        ))

        self.add_metric(MetricDefinition(
            metric_id="no_show_rate",
            name="No-Show Rate",
            aliases=["no-show rate", "no shows", "missed appointments"],
            numerator_sql="COUNT(CASE WHEN STATUS='NO_SHOW' THEN 1 END)",
            denominator_sql="COUNT(*)",
            time_period="monthly",
            units="%",
            concept_id="no_show"
        ))

        self.add_metric(MetricDefinition(
            metric_id="pmpm",
            name="Per Member Per Month Cost",
            aliases=["pmpm", "per member per month", "monthly cost"],
            numerator_sql="SUM(PAID_AMOUNT)",
            denominator_sql="COUNT(DISTINCT MEMBER_ID) * (EXTRACT(DAY FROM DATE('now')) / 30.0)",
            time_period="monthly",
            units="$"
        ))

        self.add_metric(MetricDefinition(
            metric_id="readmission_rate",
            name="30-Day Readmission Rate",
            aliases=["readmission", "readmission rate", "30-day readmission"],
            numerator_sql="COUNT(CASE WHEN READMIT_FLAG=1 THEN 1 END)",
            denominator_sql="COUNT(CASE WHEN VISIT_TYPE='INPATIENT' THEN 1 END)",
            time_period="quarterly",
            units="%",
            concept_id="readmission"
        ))

        self.add_metric(MetricDefinition(
            metric_id="avg_los",
            name="Average Length of Stay",
            aliases=["avg los", "average los", "length of stay"],
            numerator_sql="SUM(LENGTH_OF_STAY)",
            denominator_sql="COUNT(*)",
            time_period="monthly",
            units="days"
        ))

        self.add_metric(MetricDefinition(
            metric_id="mlr",
            name="Medical Loss Ratio",
            aliases=["mlr", "medical loss ratio"],
            numerator_sql="SUM(PAID_AMOUNT)",
            denominator_sql="(SELECT SUM(premium_revenue) FROM budget_table)",
            time_period="quarterly",
            units="%"
        ))

        self.add_metric(MetricDefinition(
            metric_id="generic_rate",
            name="Generic Dispensing Rate",
            aliases=["generic rate", "generic prescriptions"],
            numerator_sql="COUNT(CASE WHEN MEDICATION_CLASS='GENERIC' THEN 1 END)",
            denominator_sql="COUNT(*)",
            time_period="monthly",
            units="%"
        ))

        self.add_metric(MetricDefinition(
            metric_id="er_visits_per_1000",
            name="ER Visits per 1000 Members",
            aliases=["er visits", "emergency visits", "ed visits"],
            numerator_sql="COUNT(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 END) * 1000",
            denominator_sql="COUNT(DISTINCT MEMBER_ID)",
            time_period="monthly",
            units="per 1000"
        ))

    def _build_benchmarks(self):
        benchmarks_data = [
            ("denial_rate", 0.05, 0.05, 0.10, 0.15),
            ("no_show_rate", 0.05, 0.05, 0.12, 0.20),
            ("readmission_rate", 0.10, 0.10, 0.15, 0.20),
            ("pmpm", 350, 350, 500, 750),
            ("avg_los", 3.0, 3.0, 5.0, 7.0),
            ("generic_rate", 0.90, 0.80, 0.90, 0.80),
            ("er_visits_per_1000", 250, 250, 400, 500),
        ]

        for metric_id, excellent, avg_low, avg_high, poor in benchmarks_data:
            self.benchmarks[metric_id] = Benchmark(
                metric_id=metric_id,
                excellent=excellent,
                average_low=avg_low,
                average_high=avg_high,
                poor=poor
            )

    def add_metric(self, metric: MetricDefinition):
        self.metrics[metric.metric_id] = metric

    def get_metric(self, metric_id_or_alias: str) -> Optional[MetricDefinition]:
        metric_id_or_alias_lower = metric_id_or_alias.lower()

        if metric_id_or_alias in self.metrics:
            return self.metrics[metric_id_or_alias]

        for metric in self.metrics.values():
            if metric_id_or_alias_lower in [a.lower() for a in metric.aliases]:
                return metric

        return None

    def get_benchmark(self, metric_id: str) -> Optional[Benchmark]:
        return self.benchmarks.get(metric_id)

    def grade_metric_value(self, metric_id: str, value: float) -> str:
        benchmark = self.get_benchmark(metric_id)
        if not benchmark:
            return "unknown"

        inverse_metrics = {"denial_rate", "no_show_rate", "readmission_rate", "avg_los", "er_visits_per_1000"}
        is_inverse = metric_id in inverse_metrics

        if is_inverse:
            if value <= benchmark.excellent:
                return "excellent"
            elif value <= benchmark.average_high:
                return "average"
            else:
                return "poor"
        else:
            if value >= benchmark.excellent:
                return "excellent"
            elif value >= benchmark.average_low:
                return "average"
            else:
                return "poor"


class ContextualInferenceEngine:

    def __init__(self, ontology: ConceptOntology, metric_engine: BusinessMetricEngine):
        self.ontology = ontology
        self.metric_engine = metric_engine

    def infer_intent(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()

        intent = {
            "original": question,
            "inferred_intent": None,
            "metrics_to_check": [],
            "concepts": [],
            "benchmarking_required": False
        }

        if any(p in question_lower for p in ["doing well", "how are we", "are we doing", "performance"]):
            intent["inferred_intent"] = "performance_check"
            intent["benchmarking_required"] = True
            intent["metrics_to_check"] = list(self.metric_engine.metrics.keys())

        elif any(p in question_lower for p in ["concerning", "red flag", "problem", "issue", "bad"]):
            intent["inferred_intent"] = "anomaly_detection"
            intent["benchmarking_required"] = True
            intent["metrics_to_check"] = list(self.metric_engine.metrics.keys())

        elif any(p in question_lower for p in ["ncal", "scal", "hawaii", "colorado", "region"]):
            intent["inferred_intent"] = "regional_analysis"
            intent["concepts"].append("kp_region")

        elif any(p in question_lower for p in ["quality", "stars", "hedis", "readmission"]):
            intent["inferred_intent"] = "quality_analysis"
            intent["concepts"].extend(["stars_rating", "hedis", "readmission"])

        elif any(p in question_lower for p in ["cost", "spending", "pmpm", "expensive", "cheap"]):
            intent["inferred_intent"] = "cost_analysis"
            intent["concepts"].append("cost_analysis")

        else:
            intent["inferred_intent"] = "general_query"

        return intent

    def infer_scope(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()

        scope = {
            "time_period": "current",
            "time_expression": None,
            "regions": [],
            "member_segments": [],
            "departments": [],
            "plan_types": []
        }

        time_patterns = {
            "ytd": ("year_to_date", None),
            "monthly": ("monthly", None),
            "quarterly": ("quarterly", None),
            "today": ("daily", None),
            "last 30": ("last_30_days", None),
            "last 90": ("last_90_days", None),
            "last year": ("last_year", None),
        }

        for pattern, (period, expr) in time_patterns.items():
            if pattern in question_lower:
                scope["time_period"] = period
                scope["time_expression"] = expr
                break

        regions = ["ncal", "scal", "hawaii", "colorado", "norcal", "socal"]
        for region in regions:
            if region in question_lower:
                scope["regions"].append(region.upper())

        segments = {
            "diabetic": "E11%",
            "heart": "I50%",
            "copd": "J44%",
            "elderly": "65+",
            "pediatric": "<18",
            "high-risk": "RISK_SCORE>2"
        }

        for segment, code in segments.items():
            if segment in question_lower:
                scope["member_segments"].append((segment, code))

        return scope


class JargonEntityExtractor:

    def __init__(self, domain_knowledge: HealthcareDomainKnowledge):
        self.domain_knowledge = domain_knowledge
        self.entity_mappings: List[EntityMapping] = []
        self._build_mappings()

    def _build_mappings(self):
        mappings = [
            EntityMapping(
                entity_pattern=r"diabetic\s+members?",
                entity_type="member_group",
                sql_where_clause="MEMBER_ID IN (SELECT DISTINCT MEMBER_ID FROM diagnoses WHERE ICD10_CODE LIKE 'E11%')",
                required_tables=["diagnoses", "members"],
                aliases=["diabetes patients", "type 2 diabetics"]
            ),
            EntityMapping(
                entity_pattern=r"heart\s+failure\s+patients?",
                entity_type="member_group",
                sql_where_clause="MEMBER_ID IN (SELECT DISTINCT MEMBER_ID FROM diagnoses WHERE HCC_CATEGORY='Heart Failure')",
                required_tables=["diagnoses", "members"],
                aliases=["CHF patients", "cardiac patients"]
            ),
            EntityMapping(
                entity_pattern=r"er\s+visit|emergency\s+visit|ed\s+visit",
                entity_type="claim_type",
                sql_where_clause="VISIT_TYPE='EMERGENCY'",
                required_tables=["encounters"],
                aliases=["emergency room", "emergency department"]
            ),
            EntityMapping(
                entity_pattern=r"high.?risk\s+members?",
                entity_type="member_group",
                sql_where_clause="CAST(RISK_SCORE AS REAL) > 2.0",
                required_tables=["members"],
                aliases=["high-risk patients", "high risk"]
            ),
            EntityMapping(
                entity_pattern=r"male\s+members?",
                entity_type="member_group",
                sql_where_clause="GENDER='M'",
                required_tables=["members"]
            ),
            EntityMapping(
                entity_pattern=r"female\s+members?",
                entity_type="member_group",
                sql_where_clause="GENDER='F'",
                required_tables=["members"]
            ),
            EntityMapping(
                entity_pattern=r"elderly|seniors?|65\+|over\s+65|age\s+65",
                entity_type="member_group",
                sql_where_clause="DATE_OF_BIRTH <= DATE('now', '-65 years')",
                required_tables=["members"]
            ),
            EntityMapping(
                entity_pattern=r"pediatric|children|kids|under\s+18|age\s+<\s*18",
                entity_type="member_group",
                sql_where_clause="DATE_OF_BIRTH >= DATE('now', '-18 years')",
                required_tables=["members"]
            ),
            EntityMapping(
                entity_pattern=r"denied\s+claims?|claim\s+denials?",
                entity_type="claim_status",
                sql_where_clause="CLAIM_STATUS='DENIED'",
                required_tables=["claims"]
            ),
            EntityMapping(
                entity_pattern=r"no.?shows?|missed\s+appointments?",
                entity_type="appointment_status",
                sql_where_clause="STATUS='NO_SHOW'",
                required_tables=["appointments"]
            ),
            EntityMapping(
                entity_pattern=r"inpatient|hospitalization|admission",
                entity_type="visit_type",
                sql_where_clause="VISIT_TYPE='INPATIENT'",
                required_tables=["encounters"]
            ),
            EntityMapping(
                entity_pattern=r"outpatient|office\s+visit",
                entity_type="visit_type",
                sql_where_clause="VISIT_TYPE='OUTPATIENT'",
                required_tables=["encounters"]
            ),
            EntityMapping(
                entity_pattern=r"telehealth|virtual|telemedicine",
                entity_type="visit_type",
                sql_where_clause="VISIT_TYPE='TELEHEALTH'",
                required_tables=["encounters"]
            ),
        ]
        self.entity_mappings = mappings

    def extract_entities(self, text: str) -> List[Tuple[str, EntityMapping]]:
        entities = []
        text_lower = text.lower()

        for mapping in self.entity_mappings:
            if re.search(mapping.entity_pattern, text_lower, re.IGNORECASE):
                entities.append((mapping.entity_pattern, mapping))

        return entities

    def build_where_clause(self, entities: List[EntityMapping]) -> str:
        if not entities:
            return ""

        where_clauses = [e.sql_where_clause for e in entities]
        return " AND ".join([f"({clause})" for clause in where_clauses])


class BenchmarkDatabase:

    def __init__(self, metric_engine: BusinessMetricEngine):
        self.metric_engine = metric_engine
        self.industry_benchmarks = metric_engine.benchmarks

    def grade_performance(self, metric_id: str, value: float) -> str:
        return self.metric_engine.grade_metric_value(metric_id, value)

    def get_peer_comparison(self, metric_id: str, value: float) -> Dict[str, Any]:
        benchmark = self.metric_engine.get_benchmark(metric_id)
        if not benchmark:
            return {}

        grade = self.grade_performance(metric_id, value)

        return {
            "metric_id": metric_id,
            "current_value": value,
            "grade": grade,
            "excellent_threshold": benchmark.excellent,
            "average_range": (benchmark.average_low, benchmark.average_high),
            "poor_threshold": benchmark.poor,
            "percentile_estimate": self._estimate_percentile(metric_id, value)
        }

    def _estimate_percentile(self, metric_id: str, value: float) -> float:
        benchmark = self.metric_engine.get_benchmark(metric_id)
        if not benchmark:
            return 50.0

        inverse_metrics = {"denial_rate", "no_show_rate", "readmission_rate", "avg_los", "er_visits_per_1000"}
        is_inverse = metric_id in inverse_metrics

        if is_inverse:
            if value <= benchmark.excellent:
                return 90.0
            elif value <= benchmark.average_high:
                return 50.0
            else:
                return 10.0
        else:
            if value >= benchmark.excellent:
                return 90.0
            elif value >= benchmark.average_low:
                return 50.0
            else:
                return 10.0


class CausalReasoningEngine:

    def __init__(self, ontology: ConceptOntology, metric_engine: BusinessMetricEngine):
        self.ontology = ontology
        self.metric_engine = metric_engine
        self.causal_patterns = self._build_causal_patterns()

    def _build_causal_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "high_denial_rate": [
                {
                    "investigation": "Check denial reasons distribution",
                    "sql_template": "SELECT DENIAL_REASON, COUNT(*) as count FROM claims WHERE CLAIM_STATUS='DENIED' GROUP BY DENIAL_REASON"
                },
                {
                    "investigation": "Check denial rate by plan type",
                    "sql_template": "SELECT PLAN_TYPE, COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) * 100.0 / COUNT(*) as denial_rate FROM claims GROUP BY PLAN_TYPE"
                },
                {
                    "investigation": "Check denial rate by region",
                    "sql_template": "SELECT KP_REGION, COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) * 100.0 / COUNT(*) as denial_rate FROM claims GROUP BY KP_REGION"
                },
                {
                    "investigation": "Check denial rate by provider",
                    "sql_template": "SELECT PROVIDER_NPI, COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) * 100.0 / COUNT(*) as denial_rate FROM claims GROUP BY PROVIDER_NPI ORDER BY denial_rate DESC LIMIT 10"
                }
            ],
            "high_costs": [
                {
                    "investigation": "Identify high-cost members",
                    "sql_template": "SELECT MEMBER_ID, SUM(PAID_AMOUNT) as total_cost FROM claims GROUP BY MEMBER_ID ORDER BY total_cost DESC LIMIT 100"
                },
                {
                    "investigation": "Check costs by diagnosis",
                    "sql_template": "SELECT d.ICD10_CODE, AVG(c.PAID_AMOUNT) as avg_cost, COUNT(*) as claim_count FROM claims c JOIN diagnoses d ON c.MEMBER_ID=d.MEMBER_ID GROUP BY d.ICD10_CODE ORDER BY avg_cost DESC"
                },
                {
                    "investigation": "Check inpatient utilization costs",
                    "sql_template": "SELECT COUNT(*) as inpatient_encounters, SUM(c.PAID_AMOUNT) as total_inpatient_cost FROM encounters e JOIN claims c ON e.ENCOUNTER_ID=c.ENCOUNTER_ID WHERE e.VISIT_TYPE='INPATIENT'"
                }
            ],
            "high_no_shows": [
                {
                    "investigation": "Check no-show rate by department",
                    "sql_template": "SELECT DEPARTMENT, COUNT(CASE WHEN STATUS='NO_SHOW' THEN 1 END) * 100.0 / COUNT(*) as no_show_rate FROM appointments GROUP BY DEPARTMENT"
                },
                {
                    "investigation": "Check no-show rate by appointment type",
                    "sql_template": "SELECT APPOINTMENT_TYPE, COUNT(CASE WHEN STATUS='NO_SHOW' THEN 1 END) * 100.0 / COUNT(*) as no_show_rate FROM appointments GROUP BY APPOINTMENT_TYPE"
                },
                {
                    "investigation": "Check no-show rate by time of day",
                    "sql_template": "SELECT EXTRACT(HOUR FROM APPOINTMENT_TIME) as hour, COUNT(CASE WHEN STATUS='NO_SHOW' THEN 1 END) * 100.0 / COUNT(*) as no_show_rate FROM appointments GROUP BY EXTRACT(HOUR FROM APPOINTMENT_TIME)"
                }
            ],
            "high_readmissions": [
                {
                    "investigation": "Check readmission by diagnosis",
                    "sql_template": "SELECT d.HCC_CATEGORY, COUNT(CASE WHEN e2.ADMIT_DATE <= DATE(e1.DISCHARGE_DATE, '+30 days') THEN 1 END) * 100.0 / COUNT(*) as readmit_rate FROM encounters e1 JOIN encounters e2 ON e1.MEMBER_ID=e2.MEMBER_ID AND e1.DISCHARGE_DATE < e2.ADMIT_DATE JOIN diagnoses d ON e1.MEMBER_ID=d.MEMBER_ID WHERE e1.VISIT_TYPE='INPATIENT' GROUP BY d.HCC_CATEGORY"
                },
                {
                    "investigation": "Check discharge against medical advice",
                    "sql_template": "SELECT COUNT(*) as ama_discharges FROM encounters WHERE AGAINST_MEDICAL_ADVICE='Y'"
                }
            ]
        }

    def explain_metric(self, metric_id: str) -> Dict[str, Any]:
        explanation = {
            "metric_id": metric_id,
            "possible_causes": [],
            "investigation_queries": []
        }

        metric_to_pattern = {
            "denial_rate": "high_denial_rate",
            "pmpm": "high_costs",
            "no_show_rate": "high_no_shows",
            "readmission_rate": "high_readmissions",
        }

        pattern_key = metric_to_pattern.get(metric_id)
        if pattern_key in self.causal_patterns:
            for pattern in self.causal_patterns[pattern_key]:
                explanation["investigation_queries"].append(pattern)

        return explanation


class HealthcareDomainIntelligence:

    def __init__(self):
        self.domain_knowledge = HealthcareDomainKnowledge()
        self.ontology = ConceptOntology()
        self.metric_engine = BusinessMetricEngine()
        self.contextual_inference = ContextualInferenceEngine(self.ontology, self.metric_engine)
        self.entity_extractor = JargonEntityExtractor(self.domain_knowledge)
        self.benchmark_db = BenchmarkDatabase(self.metric_engine)
        self.causal_reasoning = CausalReasoningEngine(self.ontology, self.metric_engine)

    def normalize_question(self, question: str) -> str:
        normalized = question
        terms = question.lower().split()

        for term in terms:
            canonical = self.domain_knowledge.normalize_term(term)
            if canonical != term.lower():
                normalized = normalized.replace(term, canonical, 1)

        return normalized

    def analyze_question(self, question: str) -> Dict[str, Any]:
        normalized = self.normalize_question(question)

        jargon_resolved = {}
        for term in question.lower().split():
            canonical = self.domain_knowledge.normalize_term(term)
            if canonical != term.lower():
                jargon_resolved[term] = canonical

        analysis = {
            "original_question": question,
            "normalized_question": normalized,
            "jargon_resolved": jargon_resolved,
            "intent": self.contextual_inference.infer_intent(question),
            "scope": self.contextual_inference.infer_scope(question),
            "entities": [],
            "sql_where_clause": "",
            "related_metrics": [],
            "related_concepts": [],
            "concepts_activated": [],
        }

        entities = self.entity_extractor.extract_entities(question)
        if entities:
            entity_mappings = [e[1] for e in entities]
            analysis["entities"] = [
                {
                    "type": e.entity_type,
                    "pattern": e.entity_pattern,
                    "sql_where": e.sql_where_clause,
                    "required_tables": e.required_tables
                }
                for e in entity_mappings
            ]
            analysis["sql_where_clause"] = self.entity_extractor.build_where_clause(entity_mappings)

        intent_keywords = analysis["intent"]["inferred_intent"]
        for metric in self.metric_engine.metrics.values():
            if any(kw in metric.name.lower() or kw in str(metric.aliases).lower() for kw in question.lower().split()):
                analysis["related_metrics"].append({
                    "metric_id": metric.metric_id,
                    "name": metric.name,
                    "aliases": metric.aliases
                })

        for concept in self.ontology.concepts.values():
            if any(kw in concept.name.lower() or kw in str(concept.aliases).lower() for kw in question.lower().split()):
                analysis["related_concepts"].append({
                    "concept_id": concept.id,
                    "name": concept.name,
                    "aliases": concept.aliases
                })
                analysis["concepts_activated"].append(concept.id)

        try:
            causal = self.causal_reasoning.find_root_causes(question)
            if causal:
                analysis["causal_factors"] = causal
        except Exception:
            pass

        return analysis

    def get_metric_definition(self, metric_id_or_alias: str) -> Optional[MetricDefinition]:
        return self.metric_engine.get_metric(metric_id_or_alias)

    def get_concept_definition(self, concept_id_or_alias: str) -> Optional[ConceptNode]:
        return self.ontology.get_concept(concept_id_or_alias)

    def explain_metric_causes(self, metric_id: str) -> Dict[str, Any]:
        return self.causal_reasoning.explain_metric(metric_id)

    def benchmark_metric(self, metric_id: str, value: float) -> Dict[str, Any]:
        return self.benchmark_db.get_peer_comparison(metric_id, value)


def create_domain_intelligence() -> HealthcareDomainIntelligence:
    return HealthcareDomainIntelligence()


def get_all_metrics() -> List[str]:
    intelligence = create_domain_intelligence()
    return list(intelligence.metric_engine.metrics.keys())


def get_all_concepts() -> List[str]:
    intelligence = create_domain_intelligence()
    return list(intelligence.ontology.concepts.keys())


if __name__ == "__main__":
    intelligence = create_domain_intelligence()

    question = "What's the denial rate by region in NCAL?"
    analysis = intelligence.analyze_question(question)
    print(f"Question Analysis:\n{json.dumps(analysis, indent=2, default=str)}\n")

    metric = intelligence.get_metric_definition("denial_rate")
    if metric:
        print(f"Metric: {metric.name}")
        print(f"  Numerator SQL: {metric.numerator_sql}")
        print(f"  Denominator SQL: {metric.denominator_sql}\n")

    benchmark = intelligence.benchmark_metric("denial_rate", 0.12)
    print(f"Benchmark Results:\n{json.dumps(benchmark, indent=2)}\n")

    explanation = intelligence.explain_metric_causes("high_denial_rate")
    print(f"Causal Patterns:\n{json.dumps(explanation, indent=2, default=str)}\n")
