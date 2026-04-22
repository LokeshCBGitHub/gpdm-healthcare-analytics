import re
import math
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger('gpdm.tuva')


class ClinicalConceptRegistry:

    CLINICAL_CONCEPTS = {
        'diabetes': {
            'icd10_prefixes': ['E10', 'E11', 'E13'],
            'hcc_categories': ['Diabetes'],
            'synonyms': ['diabetic', 'dm', 'dm2', 'type 2 diabetes', 'type 1 diabetes',
                          'a1c', 'hba1c', 'blood sugar', 'glucose', 'insulin'],
            'tuva_concept': 'chronic_condition',
        },
        'hypertension': {
            'icd10_prefixes': ['I10', 'I11', 'I12', 'I13', 'I15'],
            'hcc_categories': [],
            'synonyms': ['htn', 'high blood pressure', 'bp', 'blood pressure'],
            'tuva_concept': 'chronic_condition',
        },
        'heart failure': {
            'icd10_prefixes': ['I50'],
            'hcc_categories': ['Heart Failure'],
            'synonyms': ['chf', 'congestive heart failure', 'hf', 'cardiac failure',
                          'heart disease', 'cardiovascular'],
            'tuva_concept': 'chronic_condition',
        },
        'copd': {
            'icd10_prefixes': ['J44', 'J43'],
            'hcc_categories': ['COPD'],
            'synonyms': ['chronic obstructive pulmonary', 'emphysema',
                          'chronic bronchitis', 'lung disease'],
            'tuva_concept': 'chronic_condition',
        },
        'ckd': {
            'icd10_prefixes': ['N18'],
            'hcc_categories': ['CKD'],
            'synonyms': ['chronic kidney disease', 'renal failure', 'kidney disease',
                          'dialysis', 'renal', 'kidney'],
            'tuva_concept': 'chronic_condition',
        },
        'cancer': {
            'icd10_prefixes': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05',
                               'C06', 'C07', 'C08', 'C09', 'C10', 'C11',
                               'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
                               'C18', 'C19', 'C20', 'C21', 'C22', 'C25',
                               'C34', 'C43', 'C50', 'C53', 'C56', 'C61',
                               'C64', 'C67', 'C71', 'C73', 'C78', 'C79',
                               'C80', 'C90', 'C91', 'C92'],
            'hcc_categories': ['Cancer'],
            'synonyms': ['malignant', 'neoplasm', 'tumor', 'oncology',
                          'carcinoma', 'melanoma', 'leukemia', 'lymphoma'],
            'tuva_concept': 'chronic_condition',
        },
        'mental health': {
            'icd10_prefixes': ['F20', 'F25', 'F30', 'F31', 'F32', 'F33',
                               'F34', 'F40', 'F41', 'F42', 'F43'],
            'hcc_categories': ['Mental Health'],
            'synonyms': ['depression', 'anxiety', 'bipolar', 'schizophrenia',
                          'mental illness', 'psychiatric', 'behavioral health',
                          'mood disorder', 'ptsd'],
            'tuva_concept': 'chronic_condition',
        },
        'covid': {
            'icd10_prefixes': ['U07'],
            'hcc_categories': [],
            'synonyms': ['covid-19', 'coronavirus', 'sars-cov-2', 'covid 19'],
            'tuva_concept': 'acute_condition',
        },
        'asthma': {
            'icd10_prefixes': ['J45'],
            'hcc_categories': [],
            'synonyms': ['bronchial asthma', 'reactive airway'],
            'tuva_concept': 'chronic_condition',
        },
        'obesity': {
            'icd10_prefixes': ['E66'],
            'hcc_categories': [],
            'synonyms': ['obese', 'morbid obesity', 'bmi', 'overweight'],
            'tuva_concept': 'chronic_condition',
        },

        'inpatient': {
            'visit_types': ['INPATIENT'],
            'synonyms': ['hospitalization', 'hospital stay', 'admission',
                          'admitted', 'hospital admission', 'ip'],
            'tuva_concept': 'encounter_type',
        },
        'outpatient': {
            'visit_types': ['OUTPATIENT'],
            'synonyms': ['office visit', 'clinic visit', 'ambulatory', 'op'],
            'tuva_concept': 'encounter_type',
        },
        'emergency': {
            'visit_types': ['EMERGENCY'],
            'synonyms': ['er', 'ed', 'emergency room', 'emergency department',
                          'emergency visit', 'ed visit'],
            'tuva_concept': 'encounter_type',
        },
        'telehealth': {
            'visit_types': ['TELEHEALTH'],
            'synonyms': ['telemedicine', 'virtual visit', 'video visit',
                          'remote visit', 'telecare'],
            'tuva_concept': 'encounter_type',
        },
    }

    QUALITY_MEASURES = {
        'a1c_testing': {
            'name': 'Comprehensive Diabetes Care: HbA1c Testing',
            'hedis_id': 'CDC',
            'description': 'Members with diabetes who had HbA1c testing',
            'denominator_concept': 'diabetes',
            'numerator_cpt': ['83036', '83037'],
            'age_range': (18, 75),
        },
        'breast_cancer_screening': {
            'name': 'Breast Cancer Screening',
            'hedis_id': 'BCS',
            'description': 'Women 50-74 who had a mammogram',
            'denominator_filter': "GENDER = 'F'",
            'numerator_cpt': ['77067', '77066', '77065'],
            'numerator_icd10_prefix': ['Z12.31'],
            'age_range': (50, 74),
        },
        'controlling_bp': {
            'name': 'Controlling High Blood Pressure',
            'hedis_id': 'CBP',
            'description': 'Members with hypertension whose BP is controlled',
            'denominator_concept': 'hypertension',
            'age_range': (18, 85),
        },
        'med_adherence': {
            'name': 'Medication Adherence',
            'hedis_id': 'SPD/SPC/SPM',
            'description': 'Adherence to statin/diabetes/RAS medications',
            'denominator_concept': 'diabetes',
            'age_range': (18, 999),
        },
        'readmission_rate': {
            'name': 'Plan All-Cause Readmissions',
            'hedis_id': 'PCR',
            'description': '30-day all-cause readmission rate',
            'denominator_filter': "VISIT_TYPE = 'INPATIENT'",
            'age_range': (18, 999),
        },
        'ed_utilization': {
            'name': 'Emergency Department Utilization',
            'hedis_id': 'EDU',
            'description': 'Rate of emergency department visits per 1000 members',
            'denominator_filter': "VISIT_TYPE = 'EMERGENCY'",
            'age_range': (18, 999),
        },
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.is_healthcare = False
        self._icd10_codes: Dict[str, str] = {}
        self._hcc_categories: Set[str] = set()
        self._cpt_codes: Dict[str, str] = {}
        self._visit_types: Set[str] = set()
        self._synonym_index: Dict[str, str] = {}

        if db_path:
            self._discover_from_data(db_path)
            self._build_synonym_index()

    def _discover_from_data(self, db_path: str):
        try:
            conn = sqlite3.connect(db_path)

            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            for table in table_names:
                columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
                col_names = [c[1].upper() for c in columns]

                for col in col_names:
                    if 'ICD' in col or 'DIAGNOSIS' in col.upper():
                        if 'CODE' in col or 'ICD' in col:
                            try:
                                codes = conn.execute(
                                    f"SELECT DISTINCT {col} FROM {table} "
                                    f"WHERE {col} IS NOT NULL LIMIT 500"
                                ).fetchall()
                                for (code,) in codes:
                                    if code and re.match(r'^[A-Z]\d{2}', str(code)):
                                        self._icd10_codes[str(code)] = ''
                                        self.is_healthcare = True
                            except Exception:
                                pass

                        if 'DESCRIPTION' in col or 'DESC' in col:
                            try:
                                descs = conn.execute(
                                    f"SELECT DISTINCT {col} FROM {table} "
                                    f"WHERE {col} IS NOT NULL LIMIT 500"
                                ).fetchall()
                                code_col = next(
                                    (c for c in col_names if 'ICD' in c and 'CODE' in c),
                                    None
                                )
                                if code_col:
                                    pairs = conn.execute(
                                        f"SELECT DISTINCT {code_col}, {col} FROM {table} "
                                        f"WHERE {code_col} IS NOT NULL LIMIT 500"
                                    ).fetchall()
                                    for code, desc in pairs:
                                        if code and desc:
                                            self._icd10_codes[str(code)] = str(desc)
                            except Exception:
                                pass

                for col in col_names:
                    if 'HCC' in col:
                        try:
                            cats = conn.execute(
                                f"SELECT DISTINCT {col} FROM {table} "
                                f"WHERE {col} IS NOT NULL LIMIT 100"
                            ).fetchall()
                            for (cat,) in cats:
                                if cat:
                                    self._hcc_categories.add(str(cat))
                                    self.is_healthcare = True
                        except Exception:
                            pass

                for col in col_names:
                    if 'CPT' in col:
                        try:
                            if 'CODE' in col:
                                codes = conn.execute(
                                    f"SELECT DISTINCT {col} FROM {table} "
                                    f"WHERE {col} IS NOT NULL LIMIT 500"
                                ).fetchall()
                                for (code,) in codes:
                                    if code:
                                        self._cpt_codes[str(code)] = ''
                                        self.is_healthcare = True
                        except Exception:
                            pass

                for col in col_names:
                    if 'VISIT' in col and 'TYPE' in col:
                        try:
                            types = conn.execute(
                                f"SELECT DISTINCT {col} FROM {table} "
                                f"WHERE {col} IS NOT NULL LIMIT 50"
                            ).fetchall()
                            for (vtype,) in types:
                                if vtype:
                                    self._visit_types.add(str(vtype))
                                    self.is_healthcare = True
                        except Exception:
                            pass

            conn.close()

            if self.is_healthcare:
                logger.info("Healthcare schema detected: %d ICD-10 codes, "
                            "%d HCC categories, %d CPT codes, %d visit types",
                            len(self._icd10_codes), len(self._hcc_categories),
                            len(self._cpt_codes), len(self._visit_types))
            else:
                logger.info("Non-healthcare schema — clinical layer inactive")

        except Exception as e:
            logger.warning("Clinical data discovery error: %s", e)

    def _build_synonym_index(self):
        for concept, info in self.CLINICAL_CONCEPTS.items():
            self._synonym_index[concept] = concept
            for syn in info.get('synonyms', []):
                self._synonym_index[syn.lower()] = concept

    def resolve_clinical_terms(self, question: str) -> List[Dict]:
        if not self.is_healthcare:
            return []

        q = question.lower()
        resolved = []

        seen_concepts = set()
        for term, concept_name in sorted(self._synonym_index.items(),
                                          key=lambda x: len(x[0]), reverse=True):
            if concept_name in seen_concepts:
                continue
            if len(term) <= 3:
                pattern = r'\b' + re.escape(term) + r'\b'
            else:
                pattern = re.escape(term)
            if not re.search(pattern, q):
                continue
            seen_concepts.add(concept_name)
            concept = self.CLINICAL_CONCEPTS[concept_name]
            result = {
                'concept': concept_name,
                'matched_term': term,
                'tuva_type': concept.get('tuva_concept', ''),
                'icd10_codes': [],
                'hcc_category': '',
                'visit_type': '',
                'filter_sql': '',
            }

            if 'icd10_prefixes' in concept:
                prefixes = concept['icd10_prefixes']
                matching_codes = []
                for code in self._icd10_codes:
                    for prefix in prefixes:
                        if code.startswith(prefix):
                            matching_codes.append(code)
                result['icd10_codes'] = matching_codes
                if matching_codes:
                    code_list = ', '.join(f"'{c}'" for c in matching_codes)
                    result['filter_sql'] = f"ICD10_CODE IN ({code_list})"

            if 'hcc_categories' in concept and concept['hcc_categories']:
                for hcc_cat in concept['hcc_categories']:
                    if hcc_cat in self._hcc_categories:
                        result['hcc_category'] = hcc_cat
                        result['filter_sql'] = f"HCC_CATEGORY = '{hcc_cat}'"
                        break

            if 'visit_types' in concept:
                for vtype in concept['visit_types']:
                    if vtype in self._visit_types:
                        result['visit_type'] = vtype
                        result['filter_sql'] = f"VISIT_TYPE = '{vtype}'"
                        break

            resolved.append(result)

        return resolved

    def resolve_quality_measure(self, question: str) -> Optional[Dict]:
        if not self.is_healthcare:
            return None

        q = question.lower()

        for measure_id, measure in self.QUALITY_MEASURES.items():
            if (measure['hedis_id'].lower() in q or
                    measure_id.replace('_', ' ') in q or
                    measure['name'].lower() in q):
                return {
                    'measure_id': measure_id,
                    'measure_name': measure['name'],
                    'hedis_id': measure['hedis_id'],
                    'description': measure['description'],
                    'age_range': measure.get('age_range'),
                }

            desc_words = set(measure['description'].lower().split())
            q_words = set(q.split())
            overlap = desc_words & q_words - {'with', 'who', 'had', 'a', 'the',
                                                'of', 'is', 'are', 'for', 'to'}
            if len(overlap) >= 3:
                return {
                    'measure_id': measure_id,
                    'measure_name': measure['name'],
                    'hedis_id': measure['hedis_id'],
                    'description': measure['description'],
                    'age_range': measure.get('age_range'),
                }

        return None

    def get_chronic_condition_filter(self, condition: str) -> Optional[str]:
        concept = self.CLINICAL_CONCEPTS.get(condition.lower())
        if not concept:
            concept_name = self._synonym_index.get(condition.lower())
            if concept_name:
                concept = self.CLINICAL_CONCEPTS[concept_name]

        if not concept:
            return None

        if 'hcc_categories' in concept:
            for hcc in concept['hcc_categories']:
                if hcc in self._hcc_categories:
                    return f"HCC_CATEGORY = '{hcc}'"

        if 'icd10_prefixes' in concept:
            matching = [c for c in self._icd10_codes
                        if any(c.startswith(p) for p in concept['icd10_prefixes'])]
            if matching:
                code_list = ', '.join(f"'{c}'" for c in matching)
                return f"ICD10_CODE IN ({code_list})"

        return None

    def enrich_question(self, question: str) -> Dict:
        terms = self.resolve_clinical_terms(question)
        measure = self.resolve_quality_measure(question)

        filters = [t['filter_sql'] for t in terms if t['filter_sql']]

        context_parts = []
        for t in terms:
            if t['icd10_codes']:
                context_parts.append(
                    f"<b>{t['concept']}</b>: {len(t['icd10_codes'])} ICD-10 codes matched"
                )
            elif t['hcc_category']:
                context_parts.append(
                    f"<b>{t['concept']}</b>: HCC category '{t['hcc_category']}'"
                )
            elif t['visit_type']:
                context_parts.append(
                    f"<b>{t['concept']}</b>: visit type '{t['visit_type']}'"
                )

        if measure:
            context_parts.append(
                f"<b>Quality Measure:</b> {measure['measure_name']} ({measure['hedis_id']})"
            )

        return {
            'clinical_terms': terms,
            'quality_measure': measure,
            'suggested_filters': filters,
            'clinical_context': '<br>'.join(context_parts),
            'is_clinical': bool(terms or measure),
        }


class TuvaClinicalEnricher:

    def __init__(self, db_path: str):
        self.registry = ClinicalConceptRegistry(db_path)

    @property
    def is_healthcare(self) -> bool:
        return self.registry.is_healthcare

    def enrich(self, question: str, engine_result: Dict) -> Dict:
        if not self.registry.is_healthcare:
            return engine_result

        enrichment = self.registry.enrich_question(question)

        engine_result['clinical_context'] = enrichment.get('clinical_context', '')
        engine_result['clinical_terms'] = enrichment.get('clinical_terms', [])
        engine_result['quality_measure'] = enrichment.get('quality_measure')

        if enrichment['suggested_filters'] and not enrichment.get('quality_measure'):
            existing_sql = engine_result.get('sql', '')
            for filt in enrichment['suggested_filters']:
                if filt.split('=')[0].strip() not in existing_sql:
                    if engine_result.get('data_suggestions') is None:
                        engine_result['data_suggestions'] = []
                    engine_result['data_suggestions'].append(
                        f"Consider adding clinical filter: {filt}"
                    )

        return engine_result


import os

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    hc_db = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_production.db')
    if os.path.exists(hc_db):
        print("=== HEALTHCARE DATABASE ===")
        registry = ClinicalConceptRegistry(hc_db)
        print(f"Healthcare detected: {registry.is_healthcare}")
        print(f"ICD-10 codes: {len(registry._icd10_codes)}")
        print(f"HCC categories: {registry._hcc_categories}")
        print(f"CPT codes: {len(registry._cpt_codes)}")
        print(f"Visit types: {registry._visit_types}")

        test_queries = [
            "diabetes claims by region",
            "heart failure hospitalization rate",
            "COVID emergency visits",
            "mental health encounters by provider",
            "breast cancer screening compliance",
            "readmission rate by facility",
        ]
        print()
        for q in test_queries:
            enrichment = registry.enrich_question(q)
            print(f"Q: {q}")
            if enrichment['clinical_terms']:
                for t in enrichment['clinical_terms']:
                    print(f"  → {t['concept']}: {t['filter_sql'][:80]}")
            if enrichment['quality_measure']:
                print(f"  → Measure: {enrichment['quality_measure']['measure_name']}")
            if not enrichment['is_clinical']:
                print(f"  → (no clinical terms detected)")
            print()

    ec_db = os.path.join(os.path.dirname(script_dir), 'data', 'ecommerce_test.db')
    if os.path.exists(ec_db):
        print("\n=== E-COMMERCE DATABASE ===")
        registry2 = ClinicalConceptRegistry(ec_db)
        print(f"Healthcare detected: {registry2.is_healthcare}")
        print(f"Clinical layer: {'ACTIVE' if registry2.is_healthcare else 'INACTIVE (correct)'}")
