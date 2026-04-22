from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger('gpdm.schema_mapper')


@dataclass
class ConceptDef:
    name: str
    canonical_name: str
    aliases: List[str]
    value_patterns: List[str] = field(default_factory=list)
    data_type: str = 'any'
    domain: str = 'healthcare'

CONCEPT_CATALOG: Dict[str, ConceptDef] = {}

def _register(name: str, canonical: str, aliases: List[str],
              patterns: List[str] = None, dtype: str = 'any'):
    CONCEPT_CATALOG[name] = ConceptDef(
        name=name, canonical_name=canonical,
        aliases=[a.lower() for a in aliases],
        value_patterns=patterns or [],
        data_type=dtype,
    )

_register('member_id', 'member_id',
          ['member_id', 'mbr_id', 'subscriber_id', 'patient_id', 'enrollee_id',
           'member_no', 'patient_no', 'person_id', 'individual_id'],
          dtype='id')
_register('claim_id', 'claim_id',
          ['claim_id', 'clm_id', 'claim_no', 'claim_number', 'claim_key'],
          dtype='id')
_register('encounter_id', 'encounter_id',
          ['encounter_id', 'enc_id', 'visit_id', 'admission_id', 'case_id'],
          dtype='id')
_register('provider_id', 'provider_id',
          ['provider_id', 'npi', 'rendering_npi', 'billing_npi', 'prov_id',
           'physician_id', 'doctor_id', 'attending_npi'],
          [r'^\d{10}$'],
          dtype='id')

_register('service_date', 'service_date',
          ['service_date', 'date_of_service', 'dos', 'svc_date',
           'claim_date', 'adjudicated_date', 'paid_date', 'process_date'],
          dtype='date')
_register('admit_date', 'admit_date',
          ['admit_date', 'admission_date', 'encounter_date', 'visit_date',
           'start_date', 'begin_date', 'entry_date'],
          dtype='date')
_register('discharge_date', 'discharge_date',
          ['discharge_date', 'disch_date', 'end_date', 'release_date'],
          dtype='date')
_register('birth_date', 'birth_date',
          ['birth_date', 'dob', 'date_of_birth', 'birthdate'],
          dtype='date')

_register('diagnosis_code', 'diagnosis_code',
          ['diagnosis_code', 'dx_code', 'primary_dx', 'icd_code', 'icd10',
           'diag_cd', 'dx', 'diagnosis', 'principal_diagnosis'],
          [r'^[A-Z]\d{2}', r'^\d{3}\.\d'],
          dtype='text')
_register('procedure_code', 'procedure_code',
          ['procedure_code', 'px_code', 'cpt_code', 'cpt', 'hcpcs',
           'proc_code', 'procedure', 'service_code'],
          [r'^\d{5}$', r'^[0-9A-Z]\d{4}$'],
          dtype='text')
_register('drug_code', 'drug_code',
          ['ndc', 'drug_code', 'ndc_code', 'drug_ndc', 'medication_code',
           'rx_code', 'formulary_id'],
          [r'^\d{11}$', r'^\d{5}-\d{4}-\d{2}$'],
          dtype='text')
_register('visit_type', 'visit_type',
          ['visit_type', 'encounter_type', 'service_type', 'facility_type',
           'place_of_service', 'pos', 'setting', 'admission_type'],
          dtype='text')

_register('paid_amount', 'paid_amount',
          ['paid_amount', 'paid_amt', 'amount_paid', 'net_payment',
           'reimbursement', 'payment_amount', 'paid'],
          dtype='numeric')
_register('billed_amount', 'billed_amount',
          ['billed_amount', 'billed_amt', 'charge_amount', 'charges',
           'submitted_amount', 'gross_charges', 'total_charge'],
          dtype='numeric')
_register('allowed_amount', 'allowed_amount',
          ['allowed_amount', 'allowed_amt', 'allowable', 'contracted_amount',
           'fee_schedule_amount'],
          dtype='numeric')
_register('copay', 'copay',
          ['copay', 'copay_amount', 'co_pay', 'member_copay'],
          dtype='numeric')
_register('deductible', 'deductible',
          ['deductible', 'deductible_amount', 'ded_amount', 'member_deductible'],
          dtype='numeric')
_register('coinsurance', 'coinsurance',
          ['coinsurance', 'coins', 'coinsurance_amount', 'member_coinsurance'],
          dtype='numeric')

_register('gender', 'gender',
          ['gender', 'sex', 'member_gender', 'patient_sex'],
          [r'^[MF]$', r'^(Male|Female)$'],
          dtype='text')
_register('age', 'age',
          ['age', 'member_age', 'patient_age', 'age_years'],
          dtype='numeric')
_register('region', 'region',
          ['region', 'market', 'service_area', 'geo', 'geography', 'state',
           'county', 'zip', 'zip_code'],
          dtype='text')
_register('plan_type', 'plan_type',
          ['plan_type', 'lob', 'line_of_business', 'product', 'plan_name',
           'benefit_plan', 'coverage_type'],
          dtype='text')

_register('claim_status', 'claim_status',
          ['claim_status', 'status', 'adjudication_status', 'claim_disposition',
           'processing_status'],
          [r'^(APPROVED|DENIED|PENDING|PAID|REJECTED)$'],
          dtype='text')
_register('risk_score', 'risk_score',
          ['risk_score', 'hcc_score', 'raf_score', 'risk_adjustment',
           'acuity_score', 'severity_score'],
          dtype='numeric')


@dataclass
class ColumnMapping:
    source_column: str
    concept: str
    canonical_name: str
    confidence: float
    match_method: str
    reason: str = ''

@dataclass
class MappingResult:
    table: str
    mappings: List[ColumnMapping] = field(default_factory=list)
    needs_review: List[ColumnMapping] = field(default_factory=list)
    auto_approved: List[ColumnMapping] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)
    summary: str = ''


class SchemaMapper:

    def __init__(self, llm_backend=None):
        self._map_count = 0

    def map_table(self, table_name: str,
                  profiles: List[Dict[str, Any]]) -> MappingResult:
        self._map_count += 1
        result = MappingResult(table=table_name)

        for profile in profiles:
            col_name = profile.get('name', profile.get('column', ''))
            mapping = self._map_column(col_name, profile)
            result.mappings.append(mapping)

            if mapping.confidence >= 0.8:
                result.auto_approved.append(mapping)
            elif mapping.concept != 'unknown':
                result.needs_review.append(mapping)
            else:
                result.unmapped.append(col_name)

        total = len(result.mappings)
        auto = len(result.auto_approved)
        review = len(result.needs_review)
        unknown = len(result.unmapped)
        result.summary = (f"Mapped {total} columns: {auto} auto-approved, "
                          f"{review} need review, {unknown} unmapped")
        logger.info("SchemaMapper: %s — %s", table_name, result.summary)

        return result

    def _map_column(self, col_name: str,
                    profile: Dict[str, Any]) -> ColumnMapping:

        mapping = self._exact_match(col_name)
        if mapping and mapping.confidence >= 0.9:
            return mapping

        mapping = self._alias_match(col_name)
        if mapping and mapping.confidence >= 0.7:
            return mapping

        samples = profile.get('sample_values', [])
        if samples:
            mapping = self._value_pattern_match(col_name, samples)
            if mapping and mapping.confidence >= 0.7:
                return mapping

        if profile.get('dtype') in ('numeric', 'float', 'integer', 'REAL', 'INTEGER'):
            mapping = self._statistical_match(col_name, profile)
            if mapping and mapping.confidence >= 0.6:
                return mapping

        mapping = self._fuzzy_match(col_name, profile)
        if mapping and mapping.confidence >= 0.5:
            return mapping

        return ColumnMapping(
            source_column=col_name,
            concept='unknown',
            canonical_name=col_name.lower(),
            confidence=0.0,
            match_method='none',
            reason='No matching concept found',
        )

    def _exact_match(self, col_name: str) -> Optional[ColumnMapping]:
        lower = col_name.lower().strip()
        for concept_name, concept in CONCEPT_CATALOG.items():
            if lower == concept.canonical_name or lower in concept.aliases:
                return ColumnMapping(
                    source_column=col_name,
                    concept=concept_name,
                    canonical_name=concept.canonical_name,
                    confidence=0.95,
                    match_method='exact',
                    reason=f"Exact match to concept '{concept_name}'",
                )
        return None

    def _alias_match(self, col_name: str) -> Optional[ColumnMapping]:
        normalized = col_name.lower().replace('_', '').replace('-', '').replace(' ', '')

        best_match = None
        best_score = 0.0

        for concept_name, concept in CONCEPT_CATALOG.items():
            for alias in concept.aliases:
                alias_norm = alias.replace('_', '').replace('-', '').replace(' ', '')
                if alias_norm in normalized or normalized in alias_norm:
                    score = min(len(alias_norm), len(normalized)) / max(len(alias_norm), len(normalized))
                    if score > best_score:
                        best_score = score
                        best_match = ColumnMapping(
                            source_column=col_name,
                            concept=concept_name,
                            canonical_name=concept.canonical_name,
                            confidence=min(0.85, 0.5 + score * 0.4),
                            match_method='alias',
                            reason=f"Alias match: '{col_name}' ≈ '{alias}'",
                        )

        return best_match

    def _value_pattern_match(self, col_name: str,
                             samples: List) -> Optional[ColumnMapping]:
        if not samples:
            return None

        sample_strs = [str(s) for s in samples[:20] if s is not None]
        if not sample_strs:
            return None

        best_match = None
        best_count = 0

        for concept_name, concept in CONCEPT_CATALOG.items():
            for pattern in concept.value_patterns:
                compiled = re.compile(pattern)
                match_count = sum(1 for s in sample_strs if compiled.match(s))
                match_rate = match_count / len(sample_strs) if sample_strs else 0

                if match_rate > 0.5 and match_count > best_count:
                    best_count = match_count
                    best_match = ColumnMapping(
                        source_column=col_name,
                        concept=concept_name,
                        canonical_name=concept.canonical_name,
                        confidence=min(0.90, 0.5 + match_rate * 0.4),
                        match_method='value_pattern',
                        reason=f"Value pattern match: {match_count}/{len(sample_strs)} "
                               f"samples match {concept_name} pattern",
                    )

        return best_match

    def _statistical_match(self, col_name: str,
                           profile: Dict) -> Optional[ColumnMapping]:
        min_val = profile.get('min_val', profile.get('min', None))
        max_val = profile.get('max_val', profile.get('max', None))
        mean_val = profile.get('mean', None)
        distinct = profile.get('distinct_count', profile.get('cardinality', 0))
        null_pct = profile.get('null_pct', 0)

        if min_val is None or max_val is None:
            return None

        if (isinstance(min_val, (int, float)) and min_val >= 0 and
                isinstance(max_val, (int, float)) and max_val > 100 and
                isinstance(mean_val, (int, float)) and mean_val > 10):
            name_lower = col_name.lower()
            if any(h in name_lower for h in ['amt', 'amount', 'cost', 'charge', 'paid',
                                              'billed', 'allowed', 'price', 'fee']):
                return ColumnMapping(
                    source_column=col_name,
                    concept='paid_amount',
                    canonical_name=col_name.lower(),
                    confidence=0.65,
                    match_method='statistical',
                    reason=f"Numeric with cost-like distribution (range {min_val}-{max_val})",
                )

        if (isinstance(min_val, (int, float)) and 0 <= min_val <= 120 and
                isinstance(max_val, (int, float)) and max_val <= 120):
            name_lower = col_name.lower()
            if 'age' in name_lower:
                return ColumnMapping(
                    source_column=col_name,
                    concept='age',
                    canonical_name='age',
                    confidence=0.85,
                    match_method='statistical',
                    reason=f"Age-like range (0-120), name contains 'age'",
                )

        if (isinstance(min_val, (int, float)) and 0 <= min_val and
                isinstance(max_val, (int, float)) and max_val <= 10):
            name_lower = col_name.lower()
            if any(h in name_lower for h in ['risk', 'score', 'hcc', 'raf', 'acuity']):
                return ColumnMapping(
                    source_column=col_name,
                    concept='risk_score',
                    canonical_name='risk_score',
                    confidence=0.7,
                    match_method='statistical',
                    reason=f"Score-like range (0-{max_val}), name suggests risk/score",
                )

        return None

    def _fuzzy_match(self, col_name: str,
                     profile: Dict) -> Optional[ColumnMapping]:
        try:
            col_lower = col_name.lower().replace('_', ' ')
            col_tokens = set(col_lower.split())
            best_concept = None
            best_score = 0.0
            best_reason = ''

            for concept_name, concept_def in CONCEPT_CATALOG.items():
                concept_tokens = set(concept_name.lower().replace('_', ' ').split())
                overlap = len(col_tokens & concept_tokens)
                if overlap > 0:
                    token_score = overlap / max(len(col_tokens), len(concept_tokens))
                else:
                    token_score = 0.0

                all_names = [concept_def.canonical_name.lower()] + [
                    a.lower() for a in concept_def.aliases
                ]
                for name in all_names:
                    name_tokens = set(name.replace('_', ' ').split())
                    name_overlap = len(col_tokens & name_tokens)
                    if name_overlap > 0:
                        ns = name_overlap / max(len(col_tokens), len(name_tokens))
                        token_score = max(token_score, ns * 0.8)

                for name in all_names:
                    if name in col_lower or col_lower in name:
                        token_score = max(token_score, 0.55)

                if len(col_lower) < 20:
                    for name in all_names:
                        common = sum(1 for c in set(col_lower) if c in name)
                        ratio = common / max(len(set(col_lower)), len(set(name)), 1)
                        token_score = max(token_score, ratio * 0.5)

                if token_score > best_score:
                    best_score = token_score
                    best_concept = concept_def
                    best_reason = f"Fuzzy match: token overlap with '{concept_name}'"

            if best_concept and best_score >= 0.5:
                return ColumnMapping(
                    source_column=col_name,
                    concept=best_concept.canonical_name,
                    canonical_name=best_concept.canonical_name,
                    confidence=min(best_score, 0.75),
                    match_method='fuzzy',
                    reason=best_reason,
                )
        except Exception as e:
            logger.debug("Fuzzy column mapping failed for %s: %s", col_name, e)

        return None
