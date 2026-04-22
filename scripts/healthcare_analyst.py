import re
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.healthcare_analyst')


@dataclass
class AnalyticalConcept:
    name: str
    category: str
    sql_pattern: str
    tables: List[str]
    columns: List[str]
    group_by_options: List[str]
    description: str = ''
    related_concepts: List[str] = field(default_factory=list)
    unit: str = ''


ANALYTICAL_CONCEPTS: Dict[str, AnalyticalConcept] = {
    'denial_rate': AnalyticalConcept(
        name='denial_rate',
        category='financial',
        sql_pattern="ROUND(100.0 * SUM(CASE WHEN {status_col} = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
        tables=['claims'],
        columns=['CLAIM_STATUS'],
        group_by_options=['RENDERING_NPI', 'KP_REGION', 'PLAN_TYPE', 'FACILITY',
                          'CLAIM_TYPE', 'CPT_CODE', 'ICD10_CODE'],
        description='Percentage of claims denied out of total claims',
        related_concepts=['clean_claim_rate', 'appeal_rate'],
        unit='%',
    ),
    'clean_claim_rate': AnalyticalConcept(
        name='clean_claim_rate',
        category='financial',
        sql_pattern="ROUND(100.0 * SUM(CASE WHEN {status_col} = 'APPROVED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
        tables=['claims'],
        columns=['CLAIM_STATUS'],
        group_by_options=['RENDERING_NPI', 'KP_REGION', 'PLAN_TYPE'],
        description='Percentage of claims approved on first submission',
        related_concepts=['denial_rate'],
        unit='%',
    ),
    'pmpm': AnalyticalConcept(
        name='pmpm',
        category='financial',
        sql_pattern="ROUND(SUM(CAST({cost_col} AS REAL)) / NULLIF(COUNT(DISTINCT {member_col}), 0), 2)",
        tables=['claims', 'members'],
        columns=['PAID_AMOUNT', 'MEMBER_ID'],
        group_by_options=['KP_REGION', 'PLAN_TYPE', 'AGE_GROUP'],
        description='Per Member Per Month cost — total cost divided by unique members',
        related_concepts=['cost_per_encounter', 'total_cost'],
        unit='$',
    ),
    'cost_per_encounter': AnalyticalConcept(
        name='cost_per_encounter',
        category='financial',
        sql_pattern="ROUND(SUM(CAST({cost_col} AS REAL)) / NULLIF(COUNT(DISTINCT {encounter_col}), 0), 2)",
        tables=['claims'],
        columns=['PAID_AMOUNT', 'ENCOUNTER_ID'],
        group_by_options=['VISIT_TYPE', 'FACILITY', 'KP_REGION', 'RENDERING_NPI'],
        description='Average cost per clinical encounter',
        related_concepts=['pmpm', 'total_cost'],
        unit='$',
    ),
    'total_cost': AnalyticalConcept(
        name='total_cost',
        category='financial',
        sql_pattern="ROUND(SUM(CAST({cost_col} AS REAL)), 2)",
        tables=['claims'],
        columns=['PAID_AMOUNT'],
        group_by_options=['KP_REGION', 'PLAN_TYPE', 'RENDERING_NPI', 'ICD10_CODE',
                          'CPT_CODE', 'CLAIM_TYPE', 'FACILITY'],
        description='Total paid amount across claims',
        related_concepts=['pmpm', 'average_cost'],
        unit='$',
    ),
    'average_cost': AnalyticalConcept(
        name='average_cost',
        category='financial',
        sql_pattern="ROUND(AVG(CAST({cost_col} AS REAL)), 2)",
        tables=['claims'],
        columns=['PAID_AMOUNT'],
        group_by_options=['VISIT_TYPE', 'PLAN_TYPE', 'KP_REGION', 'CLAIM_TYPE'],
        description='Average paid amount per claim',
        related_concepts=['total_cost', 'pmpm'],
        unit='$',
    ),
    'member_responsibility': AnalyticalConcept(
        name='member_responsibility',
        category='financial',
        sql_pattern="ROUND(SUM(CAST(COPAY AS REAL)) + SUM(CAST(COINSURANCE AS REAL)) + SUM(CAST(DEDUCTIBLE AS REAL)), 2)",
        tables=['claims'],
        columns=['COPAY', 'COINSURANCE', 'DEDUCTIBLE', 'MEMBER_RESPONSIBILITY'],
        group_by_options=['MEMBER_ID', 'PLAN_TYPE', 'KP_REGION'],
        description='Total out-of-pocket cost for members',
        related_concepts=['copay_analysis', 'deductible_analysis'],
        unit='$',
    ),

    'utilization_rate': AnalyticalConcept(
        name='utilization_rate',
        category='operational',
        sql_pattern="ROUND(CAST(COUNT(*) AS REAL) / NULLIF(COUNT(DISTINCT {member_col}), 0), 2)",
        tables=['encounters', 'members'],
        columns=['ENCOUNTER_ID', 'MEMBER_ID'],
        group_by_options=['VISIT_TYPE', 'KP_REGION', 'DEPARTMENT'],
        description='Average number of encounters per member',
        related_concepts=['ed_utilization', 'readmission_rate'],
        unit='encounters/member',
    ),
    'ed_utilization': AnalyticalConcept(
        name='ed_utilization',
        category='operational',
        sql_pattern="ROUND(100.0 * SUM(CASE WHEN VISIT_TYPE = 'EMERGENCY' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
        tables=['encounters'],
        columns=['VISIT_TYPE'],
        group_by_options=['KP_REGION', 'FACILITY', 'DEPARTMENT'],
        description='Percentage of encounters that are ED visits',
        related_concepts=['utilization_rate'],
        unit='%',
    ),
    'average_los': AnalyticalConcept(
        name='average_los',
        category='operational',
        sql_pattern="ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)), 2)",
        tables=['encounters'],
        columns=['LENGTH_OF_STAY'],
        group_by_options=['VISIT_TYPE', 'FACILITY', 'DEPARTMENT', 'KP_REGION'],
        description='Average length of stay for inpatient encounters',
        related_concepts=['utilization_rate'],
        unit='days',
    ),

    'chronic_prevalence': AnalyticalConcept(
        name='chronic_prevalence',
        category='quality',
        sql_pattern="ROUND(100.0 * SUM(CASE WHEN IS_CHRONIC = 'Y' THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT {member_col}), 0), 2)",
        tables=['diagnoses', 'members'],
        columns=['IS_CHRONIC', 'MEMBER_ID'],
        group_by_options=['ICD10_DESCRIPTION', 'KP_REGION', 'HCC_CATEGORY'],
        description='Percentage of members with chronic conditions',
        related_concepts=['risk_score', 'comorbidity_index'],
        unit='%',
    ),
    'risk_score': AnalyticalConcept(
        name='risk_score',
        category='quality',
        sql_pattern="ROUND(AVG(CAST(RISK_SCORE AS REAL)), 2)",
        tables=['members'],
        columns=['RISK_SCORE'],
        group_by_options=['KP_REGION', 'PLAN_TYPE', 'GENDER', 'AGE_GROUP'],
        description='Average HCC-based risk adjustment score',
        related_concepts=['chronic_prevalence', 'pmpm'],
        unit='score',
    ),

    'provider_panel': AnalyticalConcept(
        name='provider_panel',
        category='operational',
        sql_pattern="COUNT(DISTINCT {member_col})",
        tables=['claims', 'providers'],
        columns=['MEMBER_ID', 'RENDERING_NPI'],
        group_by_options=['RENDERING_NPI', 'SPECIALTY', 'FACILITY', 'KP_REGION'],
        description='Number of unique patients seen by each provider',
        related_concepts=['utilization_rate'],
        unit='members',
    ),
    'prescription_volume': AnalyticalConcept(
        name='prescription_volume',
        category='operational',
        sql_pattern="COUNT(*)",
        tables=['prescriptions'],
        columns=['PRESCRIPTION_ID'],
        group_by_options=['MEDICATION_CLASS', 'MEDICATION_NAME', 'PRESCRIBING_NPI', 'KP_REGION'],
        description='Volume of prescriptions written',
        related_concepts=['prescription_cost'],
        unit='count',
    ),
}


CONCEPT_PATTERNS = [
    (r'\b(?:denial|denied|rejection)\s*(?:rate|percentage|%)', 'denial_rate'),
    (r'\b(?:clean\s*claim|approval)\s*(?:rate|percentage|%)', 'clean_claim_rate'),

    (r'\b(?:pmpm|per\s*member\s*per\s*month|cost\s*per\s*member)', 'pmpm'),
    (r'\b(?:cost|spend)\s*(?:per|by)\s*(?:encounter|visit)', 'cost_per_encounter'),
    (r'\b(?:total|aggregate)\s*(?:cost|paid|spend|amount)', 'total_cost'),
    (r'\b(?:average|avg|mean)\s*(?:cost|paid|amount)', 'average_cost'),
    (r'\b(?:member\s*responsibility|out\s*of\s*pocket|oop)', 'member_responsibility'),

    (r'\b(?:utilization|usage)\s*(?:rate|ratio)', 'utilization_rate'),
    (r'\b(?:ed|emergency)\s*(?:utilization|usage|visit)', 'ed_utilization'),
    (r'\b(?:average|avg|mean)\s*(?:length\s*of\s*stay|los)', 'average_los'),

    (r'\b(?:chronic)\s*(?:prevalence|rate|conditions)', 'chronic_prevalence'),
    (r'\b(?:risk\s*score|raf|hcc\s*score|risk\s*adjustment)', 'risk_score'),

    (r'\b(?:panel\s*size|patient\s*panel|attributed)', 'provider_panel'),
    (r'\b(?:prescription|rx)\s*(?:volume|count)', 'prescription_volume'),
]


class HealthcareFinancialAnalyst:

    def __init__(self):
        self.concepts = ANALYTICAL_CONCEPTS
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), concept_name)
            for pattern, concept_name in CONCEPT_PATTERNS
        ]

    def identify_concept(self, question: str) -> Optional[str]:
        q = question.lower()
        for pattern, concept_name in self._compiled_patterns:
            if pattern.search(q):
                return concept_name
        return None

    def get_concept(self, name: str) -> Optional[AnalyticalConcept]:
        return self.concepts.get(name)

    def suggest_group_by(self, concept_name: str, question: str) -> Optional[Tuple[str, str]]:
        concept = self.concepts.get(concept_name)
        if not concept:
            return None

        q = question.lower()

        GROUP_BY_MAP = {
            r'\bby\s+(?:provider|doctor|physician|npi)': ('claims', 'RENDERING_NPI'),
            r'\bprovider': ('claims', 'RENDERING_NPI'),
            r'\bby\s+specialty': ('providers', 'SPECIALTY'),

            r'\bby\s+(?:region|area|geography|location)': ('members', 'KP_REGION'),
            r'\b(?:which|what)\s+region': ('members', 'KP_REGION'),
            r'\bby\s+(?:state|county|zip)': ('members', 'STATE'),
            r'\bby\s+facility': ('claims', 'FACILITY'),

            r'\bby\s+(?:plan|plan\s*type|insurance|payer)': ('claims', 'PLAN_TYPE'),
            r'\bby\s+(?:claim\s*type|service\s*type)': ('claims', 'CLAIM_TYPE'),

            r'\bby\s+(?:diagnosis|condition|icd)': ('diagnoses', 'ICD10_DESCRIPTION'),
            r'\bby\s+(?:procedure|cpt|service)': ('claims', 'CPT_DESCRIPTION'),
            r'\bby\s+(?:medication|drug|rx)\s*(?:class)?': ('prescriptions', 'MEDICATION_CLASS'),
            r'\bby\s+visit\s*type': ('encounters', 'VISIT_TYPE'),
            r'\bby\s+department': ('encounters', 'DEPARTMENT'),

            r'\bby\s+(?:age|age\s*group)': ('members', 'AGE_GROUP'),
            r'\bby\s+gender': ('members', 'GENDER'),
            r'\bby\s+(?:race|ethnicity)': ('members', 'RACE'),
        }

        for pattern, (table, col) in GROUP_BY_MAP.items():
            if re.search(pattern, q):
                if col in concept.group_by_options or table in concept.tables:
                    return (table, col)

        return None

    def validate_analytical_coherence(self, intent_dict: Dict) -> List[str]:
        warnings = []

        agg_col = intent_dict.get('agg_column', '')
        agg_fn = intent_dict.get('agg_function', '')
        intent_type = intent_dict.get('intent', '')

        CATEGORICAL_COLS = {'CLAIM_STATUS', 'VISIT_TYPE', 'PLAN_TYPE', 'GENDER',
                           'IS_CHRONIC', 'CLAIM_TYPE', 'SPECIALTY'}
        if agg_col in CATEGORICAL_COLS and agg_fn in ('SUM', 'AVG'):
            warnings.append(
                f"Cannot {agg_fn} column {agg_col} — it's categorical. "
                f"Use COUNT or rate calculation instead."
            )

        MONEY_COLS = {'PAID_AMOUNT', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT',
                      'COPAY', 'COINSURANCE', 'DEDUCTIBLE', 'COST',
                      'MEMBER_RESPONSIBILITY'}
        if agg_col in MONEY_COLS and agg_fn == 'COUNT':
            warnings.append(
                f"Counting {agg_col} is unusual — did you mean SUM({agg_col}) or AVG({agg_col})?"
            )

        if agg_col == 'RISK_SCORE' and agg_fn == 'SUM':
            warnings.append(
                f"SUM(RISK_SCORE) is meaningless — risk scores should be AVG'd across a population."
            )

        if agg_col == 'LENGTH_OF_STAY' and agg_fn == 'SUM':
            warnings.append(
                f"SUM(LENGTH_OF_STAY) = total bed days. For benchmarking, use AVG(LENGTH_OF_STAY)."
            )

        return warnings

    def enrich_intent(self, intent_dict: Dict, question: str) -> Dict:
        enriched = dict(intent_dict)
        q = question.lower()

        concept_name = self.identify_concept(question)
        if concept_name:
            concept = self.concepts[concept_name]
            enriched['analytical_concept'] = concept_name
            enriched['concept_description'] = concept.description
            enriched['result_unit'] = concept.unit
            logger.debug("Identified concept: %s — %s", concept_name, concept.description)

        warnings = self.validate_analytical_coherence(intent_dict)
        if warnings:
            enriched['analytical_warnings'] = warnings
            for w in warnings:
                logger.warning("Analytical coherence: %s", w)

        if concept_name and not enriched.get('group_by'):
            suggested = self.suggest_group_by(concept_name, question)
            if suggested:
                enriched['suggested_group_by'] = suggested

        enriched['business_context'] = self._generate_business_context(
            concept_name, enriched, question
        )

        return enriched

    def _generate_business_context(self, concept_name: Optional[str],
                                    intent: Dict, question: str) -> str:
        if not concept_name:
            return "General analytical query"

        concept = self.concepts.get(concept_name)
        if not concept:
            return "Unknown analytical concept"

        context_parts = [concept.description]

        if concept.category == 'financial':
            context_parts.append(
                "Financial metrics should be benchmarked against plan year targets "
                "and prior period performance."
            )
        elif concept.category == 'operational':
            context_parts.append(
                "Operational metrics indicate care delivery efficiency and "
                "should be compared across facilities and departments."
            )
        elif concept.category == 'quality':
            context_parts.append(
                "Quality metrics are tied to Star ratings and HEDIS measures. "
                "Improvements impact both patient outcomes and financial incentives."
            )

        if concept.related_concepts:
            context_parts.append(
                f"Related metrics to consider: {', '.join(concept.related_concepts)}"
            )

        return ' '.join(context_parts)

    def get_all_concepts(self) -> Dict[str, AnalyticalConcept]:
        return dict(self.concepts)

    def get_concepts_by_category(self, category: str) -> Dict[str, AnalyticalConcept]:
        return {k: v for k, v in self.concepts.items() if v.category == category}


_GLOBAL_ANALYST: Optional[HealthcareFinancialAnalyst] = None


def get_analyst() -> HealthcareFinancialAnalyst:
    global _GLOBAL_ANALYST
    if _GLOBAL_ANALYST is None:
        _GLOBAL_ANALYST = HealthcareFinancialAnalyst()
    return _GLOBAL_ANALYST
