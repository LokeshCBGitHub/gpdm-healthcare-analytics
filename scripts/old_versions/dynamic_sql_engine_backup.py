"""
Dynamic SQL Engine — Schema-Aware Natural Language to SQL
=========================================================

Replaces the hardcoded template SQL generation with a fully dynamic engine that:

1. Knows every column across all 7 tables (auto-discovered from semantic catalog)
2. Maps natural language words → columns via fuzzy matching + synonyms
3. Auto-discovers JOIN paths between tables using relationship map
4. Builds arbitrarily complex SQL: SELECT, FROM, JOIN, WHERE, GROUP BY,
   ORDER BY, HAVING — all driven by what the question references
5. Handles multi-table queries seamlessly (mentions column from claims +
   column from members → auto-joins them)

Zero external dependencies — pure Python.
"""

import re
import os
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set


# =============================================================================
# CENTRALIZED TYPO NORMALIZER — fuzzy matching for common healthcare terms
# =============================================================================
# Instead of maintaining individual typo maps in every method, this normalizer
# uses edit-distance matching to catch ANY misspelling of critical words.

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[len(b)]

# Critical words that users commonly misspell, with their canonical forms
# Format: canonical → (min_word_length, max_edit_distance)
# Short words (<=4 chars) use distance 1, longer words use distance 2
_CRITICAL_WORDS = {
    # Entity words (most commonly misspelled)
    # Format: canonical → (min_word_length_to_consider, max_edit_distance)
    # IMPORTANT: Use distance 1 for words <=6 chars to avoid false positives
    # (e.g., "class" → "claims" at dist 2, "under" → "gender" at dist 2)
    'members': (6, 2), 'member': (5, 2),
    'patients': (7, 2), 'patient': (6, 2),
    'providers': (8, 2), 'provider': (7, 2),
    'encounters': (9, 2), 'encounter': (8, 2),
    'claims': (5, 1), 'claim': (4, 1),
    'diagnoses': (8, 2), 'diagnosis': (8, 2),
    'prescriptions': (10, 2), 'prescription': (10, 2),
    'referrals': (7, 2), 'referral': (7, 2),
    'people': (5, 1), 'person': (5, 1),
    # Action words — longer words can tolerate distance 2
    'visited': (6, 2), 'emergency': (8, 2),
    'inpatient': (8, 2), 'outpatient': (9, 2),
    'telehealth': (9, 2), 'department': (9, 2),
    'facility': (7, 2), 'facilities': (9, 2),
    'specialty': (8, 2), 'specialties': (10, 2),
    # Aggregation words — keep short words at distance 1
    'average': (6, 2), 'total': (4, 1),
    'amount': (5, 1), 'billed': (5, 2),
    'approved': (7, 2), 'denied': (5, 1),
    'pending': (6, 2), 'cancelled': (8, 2),
    # Healthcare domain — longer words safe at distance 2
    'medication': (9, 2), 'chronic': (6, 1),
    'institutional': (12, 2), 'professional': (11, 2),
    'pharmacy': (7, 2), 'elective': (7, 2),
    # Geography / grouping — keep at distance 1 to avoid false positives
    'region': (5, 1), 'gender': (5, 1),
    'disposition': (10, 2),
    # Visit types
    'urgent': (5, 1),
}

# Words that should NEVER be corrected (common English words close to critical words)
_NORMALIZER_EXCLUSIONS = {
    'class', 'under', 'over', 'case', 'close', 'clear', 'place', 'since',
    'count', 'cost', 'costs', 'most', 'post', 'lost', 'last', 'list', 'best',
    'plan', 'plant', 'plans', 'plain', 'place',
    'state', 'states', 'stage', 'stable',
    'panel', 'model', 'level', 'label',
    'date', 'data', 'rate', 'rated', 'rates',
    'type', 'types', 'typed',
    'code', 'codes', 'coded',
    'name', 'named', 'names',
    'care', 'race', 'rare', 'core',
    'risk', 'disk',
    'stay', 'stat', 'star',
    'view', 'visit', 'visits',
    'show', 'shown', 'shows',
    'high', 'higher', 'highest',
    'paid', 'pair', 'pain',
    'aged', 'ages', 'aging',
    'male', 'female', 'scale',
    'many', 'money',
    'each', 'year', 'years',
    'this', 'that', 'than', 'then', 'them', 'they', 'their', 'there', 'these',
    'what', 'when', 'where', 'which', 'while', 'with', 'will',
    'from', 'have', 'been', 'more', 'some', 'time', 'like', 'just',
    'give', 'gave', 'gone', 'good', 'does', 'done',
    'test', 'text', 'term', 'terms',
    'drug', 'drugs',
    'fill', 'filled', 'first',
    'size', 'site', 'side',
    'days', 'daily',
    'old', 'new', 'all', 'per', 'for', 'the', 'and', 'not', 'are', 'was', 'how',
}

def normalize_typos(text: str) -> str:
    """
    Normalize common typos in a question string using fuzzy edit-distance matching.
    Replaces misspelled words with their canonical forms.

    This is the SINGLE entry point for typo correction — all methods should call this
    instead of maintaining their own typo maps.
    """
    words = text.lower().split()
    result = []
    for word in words:
        # Strip punctuation for matching but preserve it
        clean = word.strip('.,?!;:()[]{}"\'-')
        prefix = word[:len(word) - len(word.lstrip('.,?!;:()[]{}"\'-'))]
        suffix = word[len(clean) + len(prefix):]

        if len(clean) < 3:
            result.append(word)
            continue

        # Never correct known English words (prevents "class" → "claims", "under" → "gender")
        if clean in _NORMALIZER_EXCLUSIONS:
            result.append(word)
            continue

        best_match = None
        best_dist = float('inf')

        for canonical, (min_len, max_dist) in _CRITICAL_WORDS.items():
            # Only consider matches where word length is close to canonical length
            if abs(len(clean) - len(canonical)) > max_dist:
                continue
            # Skip if the word IS the canonical form (no fix needed)
            if clean == canonical:
                best_match = None
                break
            dist = _edit_distance(clean, canonical)
            if dist <= max_dist and dist < best_dist and dist > 0:
                best_dist = dist
                best_match = canonical

        if best_match:
            result.append(prefix + best_match + suffix)
        else:
            result.append(word)

    return ' '.join(result)


# =============================================================================
# COLUMN SYNONYM MAP — maps natural language terms → actual column names
# =============================================================================
# This is the key to understanding messy human questions. "Initiative start
# dates" doesn't appear in any column name, but "start" maps to date columns,
# "initiative" maps to program/plan columns, etc.

SYNONYMS = {
    # ---- Date columns ----
    'start date': ['ENROLLMENT_DATE', 'ADMIT_DATE', 'HIRE_DATE', 'PRESCRIPTION_DATE', 'REFERRAL_DATE', 'SERVICE_DATE'],
    'end date': ['DISENROLLMENT_DATE', 'DISCHARGE_DATE', 'RESOLVED_DATE'],
    'date': ['SERVICE_DATE', 'ADMIT_DATE', 'DISCHARGE_DATE', 'ENROLLMENT_DATE', 'DISENROLLMENT_DATE',
             'PRESCRIPTION_DATE', 'REFERRAL_DATE', 'DIAGNOSIS_DATE', 'FILL_DATE', 'SUBMITTED_DATE',
             'ADJUDICATED_DATE', 'APPOINTMENT_DATE', 'HIRE_DATE'],
    'when': ['SERVICE_DATE', 'ADMIT_DATE', 'ENROLLMENT_DATE', 'PRESCRIPTION_DATE', 'REFERRAL_DATE'],

    # ---- Member / Patient ----
    'patient': ['MEMBER_ID', 'MRN', 'FIRST_NAME', 'LAST_NAME'],
    'member': ['MEMBER_ID', 'MRN', 'FIRST_NAME', 'LAST_NAME'],
    'name': ['FIRST_NAME', 'LAST_NAME', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'MEDICATION_NAME'],
    'age': ['DATE_OF_BIRTH'],
    'gender': ['GENDER'],
    'race': ['RACE'],
    'language': ['LANGUAGE'],
    'address': ['ADDRESS', 'CITY', 'STATE', 'ZIP_CODE'],
    'phone': ['PHONE'],
    'email': ['EMAIL'],
    'risk': ['RISK_SCORE'],
    'chronic': ['CHRONIC_CONDITIONS', 'IS_CHRONIC'],
    'enrolled': ['ENROLLMENT_DATE'],
    'disenrolled': ['DISENROLLMENT_DATE'],

    # ---- Provider / Doctor ----
    'provider': ['NPI', 'RENDERING_NPI', 'BILLING_NPI', 'PRESCRIBING_NPI', 'REFERRING_NPI',
                 'REFERRED_TO_NPI', 'SUPERVISING_NPI', 'DIAGNOSING_NPI', 'PCP_NPI',
                 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME'],
    'doctor': ['NPI', 'RENDERING_NPI', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'SPECIALTY'],
    'npi': ['NPI', 'RENDERING_NPI', 'BILLING_NPI', 'PRESCRIBING_NPI'],
    'specialty': ['SPECIALTY'],
    'department': ['DEPARTMENT'],
    'panel': ['PANEL_SIZE'],

    # ---- Claims / Billing ----
    'claim': ['CLAIM_ID', 'CLAIM_STATUS', 'CLAIM_TYPE'],
    'billed': ['BILLED_AMOUNT'],
    'paid': ['PAID_AMOUNT'],
    'allowed': ['ALLOWED_AMOUNT'],
    'cost': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'COST'],
    'amount': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT', 'COST'],
    'money': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT'],
    'spend': ['BILLED_AMOUNT', 'PAID_AMOUNT'],
    'spending': ['BILLED_AMOUNT', 'PAID_AMOUNT'],
    'expense': ['BILLED_AMOUNT'],
    'charge': ['BILLED_AMOUNT'],
    'payment': ['PAID_AMOUNT'],
    'reimbursement': ['PAID_AMOUNT', 'ALLOWED_AMOUNT'],
    'copay': ['COPAY'],
    'coinsurance': ['COINSURANCE'],
    'deductible': ['DEDUCTIBLE'],
    'denied': ['CLAIM_STATUS', 'DENIAL_REASON'],
    'denial': ['CLAIM_STATUS', 'DENIAL_REASON'],
    'denial reason': ['DENIAL_REASON'],
    'status': ['CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'],
    'cpt': ['CPT_CODE', 'CPT_DESCRIPTION'],
    'procedure': ['CPT_CODE', 'CPT_DESCRIPTION'],

    # ---- Encounters / Visits ----
    'encounter': ['ENCOUNTER_ID', 'VISIT_TYPE', 'ENCOUNTER_STATUS'],
    'visit': ['VISIT_TYPE', 'ENCOUNTER_ID'],
    'admission': ['ADMIT_DATE', 'VISIT_TYPE'],
    'discharge': ['DISCHARGE_DATE', 'DISPOSITION'],
    'chief complaint': ['CHIEF_COMPLAINT'],
    'complaint': ['CHIEF_COMPLAINT'],
    'disposition': ['DISPOSITION'],
    'length of stay': ['LENGTH_OF_STAY'],
    'los': ['LENGTH_OF_STAY'],
    'inpatient': ['VISIT_TYPE'],
    'outpatient': ['VISIT_TYPE'],
    'emergency': ['VISIT_TYPE'],

    # ---- Diagnosis ----
    'diagnosis': ['ICD10_CODE', 'ICD10_DESCRIPTION', 'PRIMARY_DIAGNOSIS', 'DIAGNOSIS_DESCRIPTION'],
    'icd': ['ICD10_CODE', 'ICD10_DESCRIPTION'],
    'hcc': ['HCC_CODE', 'HCC_CATEGORY'],
    'severity': ['SEVERITY'],
    'condition': ['CHRONIC_CONDITIONS', 'ICD10_DESCRIPTION'],

    # ---- Prescription / Medication ----
    'prescription': ['RX_ID', 'MEDICATION_NAME', 'PRESCRIPTION_DATE'],
    'medication': ['MEDICATION_NAME', 'MEDICATION_CLASS'],
    'drug': ['MEDICATION_NAME', 'NDC_CODE'],
    'pharmacy': ['PHARMACY'],
    'refill': ['REFILLS_AUTHORIZED', 'REFILLS_USED'],
    'rx': ['RX_ID', 'MEDICATION_NAME'],

    # ---- Referral ----
    'referral': ['REFERRAL_ID', 'REFERRAL_REASON', 'REFERRAL_TYPE', 'REFERRAL_DATE'],
    'referred': ['REFERRING_NPI', 'REFERRED_TO_NPI'],
    'urgency': ['URGENCY'],
    'authorization': ['AUTHORIZATION_NUMBER'],

    # ---- Geography / Plan ----
    'region': ['KP_REGION'],
    'facility': ['FACILITY'],
    'plan': ['PLAN_TYPE'],
    'plan type': ['PLAN_TYPE'],
    'initiative': ['PLAN_TYPE', 'DEPARTMENT', 'MEDICATION_CLASS', 'HCC_CATEGORY'],
    'program': ['PLAN_TYPE', 'DEPARTMENT', 'MEDICATION_CLASS'],

    # ---- Aggregate concepts ----
    'volume': ['CLAIM_ID', 'ENCOUNTER_ID', 'RX_ID'],
    # NOTE: 'count' removed — it's an aggregation keyword, not a column synonym.
    # Mapping it to ENCOUNTER_ID etc. caused wrong tables to be pulled in.
    'revenue': ['PAID_AMOUNT', 'BILLED_AMOUNT'],
    'utilization': ['ENCOUNTER_ID', 'VISIT_TYPE'],
}


# =============================================================================
# DEEP SCHEMA KNOWLEDGE — complete column/value/relationship mapping
# =============================================================================
# This powers the "intelligent fallback" when interceptors don't match.
# Every column, every valid value, every relationship is mapped here.

SCHEMA_KNOWLEDGE = {
    'claims': {
        'row_count': 15000,
        'primary_key': 'CLAIM_ID',
        'columns': {
            'CLAIM_ID': {'type': 'id', 'desc': 'unique claim identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'ENCOUNTER_ID': {'type': 'fk', 'desc': 'encounter foreign key', 'joins_to': 'encounters.ENCOUNTER_ID', 'null_pct': 19.86},
            'SERVICE_DATE': {'type': 'date', 'desc': 'service delivery date', 'range': ['2023-01-01', '2025-12-31']},
            'RENDERING_NPI': {'type': 'fk', 'desc': 'rendering provider NPI', 'joins_to': 'providers.NPI'},
            'BILLING_NPI': {'type': 'fk', 'desc': 'billing provider NPI', 'joins_to': 'providers.NPI'},
            'CPT_CODE': {'type': 'code', 'desc': 'procedure code', 'joins_to': 'cpt_codes.CPT_CODE',
                         'values': [71046, 36415, 99223, 93000, 11102, 99214, 43239, 99283, 97110, 80053,
                                    99213, 99284, 99285, 90837, 90834, 99215, 99395, 99386, 27447, 99396]},
            'CPT_DESCRIPTION': {'type': 'text', 'desc': 'procedure description'},
            'ICD10_CODE': {'type': 'code', 'desc': 'diagnosis code on claim',
                          'values': ['M54.5', 'R51', 'I48.91', 'M17.11', 'J06.9', 'F32.1', 'G47.00', 'F41.1',
                                     'I25.10', 'E78.5', 'J45.20', 'K21.0', 'E11.65', 'N18.3', 'E66.01',
                                     'N39.0', 'G43.909', 'I10', 'J44.1', 'S82.001A', 'M79.3', 'R10.9',
                                     'J18.9', 'C50.911', 'Z96.641']},
            'ICD10_DESCRIPTION': {'type': 'text', 'desc': 'diagnosis description on claim'},
            'BILLED_AMOUNT': {'type': 'numeric', 'desc': 'amount billed', 'range': [50.02, 14999.14], 'mean': 7465.87},
            'ALLOWED_AMOUNT': {'type': 'numeric', 'desc': 'allowed amount', 'range': [23.54, 14085.0], 'mean': 5022.53},
            'PAID_AMOUNT': {'type': 'numeric', 'desc': 'amount paid', 'range': [0.0, 13729.83], 'mean': 2586.92},
            'MEMBER_RESPONSIBILITY': {'type': 'numeric', 'desc': 'member out of pocket'},
            'COPAY': {'type': 'numeric', 'desc': 'copayment amount', 'values': [0, 15, 20, 25, 30, 40, 50, 75]},
            'COINSURANCE': {'type': 'numeric', 'desc': 'coinsurance amount'},
            'DEDUCTIBLE': {'type': 'numeric', 'desc': 'deductible amount'},
            'CLAIM_STATUS': {'type': 'categorical', 'desc': 'claim processing status',
                            'values': ['PAID', 'PENDING', 'DENIED', 'ADJUSTED', 'APPEALED', 'VOIDED'],
                            'counts': {'PAID': 9063, 'PENDING': 2240, 'DENIED': 1487, 'ADJUSTED': 1209, 'APPEALED': 701, 'VOIDED': 300}},
            'CLAIM_TYPE': {'type': 'categorical', 'desc': 'claim type/category',
                          'values': ['PROFESSIONAL', 'INSTITUTIONAL', 'PHARMACY', 'DME'],
                          'counts': {'PROFESSIONAL': 7487, 'INSTITUTIONAL': 4501, 'PHARMACY': 2257, 'DME': 755}},
            'DENIAL_REASON': {'type': 'categorical', 'desc': 'reason for denial', 'null_pct': 94.61,
                             'values': ['Duplicate claim', 'Timely filing', 'Pre-auth required', 'Out of network',
                                       'Not medically necessary', 'Coding error']},
            'SUBMITTED_DATE': {'type': 'date', 'desc': 'date claim submitted'},
            'ADJUDICATED_DATE': {'type': 'date', 'desc': 'date claim adjudicated'},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP Healthcare region',
                         'values': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'GA', 'HI', 'MID'],
                         'aliases': {'northern california': 'NCAL', 'ncal': 'NCAL', 'southern california': 'SCAL',
                                    'scal': 'SCAL', 'northwest': 'NW', 'colorado': 'CO', 'mid-atlantic': 'MAS',
                                    'georgia': 'GA', 'hawaii': 'HI', 'midwest': 'MID',
                                    'baltimore': 'MAS', 'maryland': 'MAS', 'virginia': 'MAS', 'dc': 'MAS',
                                    'oakland': 'NCAL', 'san francisco': 'NCAL', 'los angeles': 'SCAL',
                                    'la': 'SCAL', 'portland': 'NW', 'oregon': 'NW', 'washington': 'NW',
                                    'denver': 'CO', 'honolulu': 'HI', 'atlanta': 'GA',
                                    'mas': 'MAS', 'ma': 'MAS', 'mid atlantic': 'MAS', 'midatlantic': 'MAS'}},
            'FACILITY': {'type': 'categorical', 'desc': 'facility name'},
            'PLAN_TYPE': {'type': 'categorical', 'desc': 'insurance plan type',
                         'values': ['EPO', 'HMO', 'PPO', 'HDHP', 'Medicaid', 'Medicare Advantage']},
        }
    },
    'members': {
        'row_count': 10000,
        'primary_key': 'MEMBER_ID',
        'columns': {
            'MRN': {'type': 'id', 'desc': 'medical record number'},
            'MEMBER_ID': {'type': 'id', 'desc': 'unique member identifier'},
            'FIRST_NAME': {'type': 'text', 'desc': 'first name', 'pii': True},
            'LAST_NAME': {'type': 'text', 'desc': 'last name', 'pii': True},
            'DATE_OF_BIRTH': {'type': 'date', 'desc': 'date of birth', 'pii': True},
            'GENDER': {'type': 'categorical', 'desc': 'gender', 'values': ['F', 'M'],
                      'counts': {'F': 5023, 'M': 4977}},
            'RACE': {'type': 'categorical', 'desc': 'race/ethnicity',
                    'values': ['Unknown', 'Asian', 'White', 'Hispanic', 'Other', 'Native Hawaiian', 'Black', 'American Indian']},
            'LANGUAGE': {'type': 'categorical', 'desc': 'primary language',
                        'values': ['English', 'Spanish', 'Chinese', 'Vietnamese', 'Other', 'Tagalog', 'Korean', 'Russian']},
            'ADDRESS': {'type': 'text', 'desc': 'street address', 'pii': True},
            'CITY': {'type': 'categorical', 'desc': 'city',
                    'values': ['Portland', 'Honolulu', 'Baltimore', 'Oakland', 'Tysons', 'Denver', 'Los Angeles', 'Atlanta']},
            'STATE': {'type': 'categorical', 'desc': 'state',
                     'values': ['WA', 'VA', 'DC', 'CO', 'HI', 'GA', 'CA', 'OR', 'MD']},
            'ZIP_CODE': {'type': 'numeric', 'desc': 'zip code'},
            'PHONE': {'type': 'text', 'desc': 'phone number', 'pii': True},
            'EMAIL': {'type': 'text', 'desc': 'email address', 'pii': True},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP region',
                         'values': ['NW', 'SCAL', 'MID', 'CO', 'HI', 'NCAL', 'GA', 'MAS']},
            'FACILITY': {'type': 'categorical', 'desc': 'home facility'},
            'PLAN_TYPE': {'type': 'categorical', 'desc': 'insurance plan type',
                         'values': ['EPO', 'HMO', 'PPO', 'HDHP', 'Medicaid', 'Medicare Advantage']},
            'ENROLLMENT_DATE': {'type': 'date', 'desc': 'enrollment date'},
            'DISENROLLMENT_DATE': {'type': 'date', 'desc': 'disenrollment date', 'null_pct': 91.63},
            'PCP_NPI': {'type': 'fk', 'desc': 'primary care provider NPI', 'joins_to': 'providers.NPI'},
            'RISK_SCORE': {'type': 'numeric', 'desc': 'health risk score', 'range': [0.2, 4.5]},
            'CHRONIC_CONDITIONS': {'type': 'numeric', 'desc': 'number of chronic conditions', 'range': [0, 6]},
        }
    },
    'encounters': {
        'row_count': 15068,
        'primary_key': 'ENCOUNTER_ID',
        'columns': {
            'ENCOUNTER_ID': {'type': 'id', 'desc': 'unique encounter identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'MRN': {'type': 'fk', 'desc': 'medical record number'},
            'SERVICE_DATE': {'type': 'date', 'desc': 'service date'},
            'ADMIT_DATE': {'type': 'date', 'desc': 'admission date', 'null_pct': 84.58},
            'DISCHARGE_DATE': {'type': 'date', 'desc': 'discharge date', 'null_pct': 84.58},
            'RENDERING_NPI': {'type': 'fk', 'desc': 'rendering provider', 'joins_to': 'providers.NPI'},
            'SUPERVISING_NPI': {'type': 'fk', 'desc': 'supervising provider', 'null_pct': 70.11},
            'VISIT_TYPE': {'type': 'categorical', 'desc': 'type of visit',
                          'values': ['OUTPATIENT', 'TELEHEALTH', 'INPATIENT', 'URGENT_CARE', 'EMERGENCY', 'HOME_HEALTH'],
                          'counts': {'OUTPATIENT': 5999, 'TELEHEALTH': 2925, 'INPATIENT': 2352, 'URGENT_CARE': 1525, 'EMERGENCY': 1501, 'HOME_HEALTH': 766}},
            'DEPARTMENT': {'type': 'categorical', 'desc': 'department name'},
            'FACILITY': {'type': 'categorical', 'desc': 'facility name'},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP region',
                         'values': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'GA', 'HI', 'MID']},
            'PRIMARY_DIAGNOSIS': {'type': 'code', 'desc': 'primary ICD-10 diagnosis code'},
            'DIAGNOSIS_DESCRIPTION': {'type': 'text', 'desc': 'diagnosis description'},
            'CHIEF_COMPLAINT': {'type': 'categorical', 'desc': 'chief complaint',
                               'values': ['Fever', 'Lab review', 'Joint pain', 'Eye irritation', 'Abdominal pain',
                                          'Skin rash', 'Shortness of breath', 'Nausea', 'Follow-up visit',
                                          'Routine checkup', 'Headache', 'Chest pain', 'Back pain', 'Cough',
                                          'Dizziness', 'Fatigue', 'Weight management', 'Anxiety']},
            'DISPOSITION': {'type': 'categorical', 'desc': 'patient disposition',
                           'values': ['Discharged', 'Observation', 'Admitted', 'Transferred', 'AMA', 'Expired']},
            'LENGTH_OF_STAY': {'type': 'numeric', 'desc': 'length of stay in days', 'range': [0, 14]},
            'ENCOUNTER_STATUS': {'type': 'categorical', 'desc': 'encounter status',
                                'values': ['COMPLETE', 'IN_PROGRESS', 'CANCELLED']},
        }
    },
    'providers': {
        'row_count': 1500,
        'primary_key': 'NPI',
        'columns': {
            'NPI': {'type': 'id', 'desc': 'National Provider Identifier'},
            'PROVIDER_FIRST_NAME': {'type': 'text', 'desc': 'provider first name'},
            'PROVIDER_LAST_NAME': {'type': 'text', 'desc': 'provider last name'},
            'SPECIALTY': {'type': 'categorical', 'desc': 'medical specialty',
                         'values': ['Cardiology', 'ENT', 'Emergency Medicine', 'Radiology', 'Rheumatology',
                                   'Orthopedics', 'Oncology', 'OB/GYN', 'Pediatrics', 'Surgery',
                                   'Internal Medicine', 'Neurology', 'Dermatology', 'Psychiatry',
                                   'Pulmonology', 'Nephrology', 'Gastroenterology', 'Urology',
                                   'Endocrinology', 'Physical Therapy'],
                         'aliases': {'heart': 'Cardiology', 'cardiac': 'Cardiology', 'cardiologist': 'Cardiology',
                                    'bone': 'Orthopedics', 'orthopedic': 'Orthopedics', 'ortho': 'Orthopedics',
                                    'skin': 'Dermatology', 'dermatologist': 'Dermatology',
                                    'mental': 'Psychiatry', 'psych': 'Psychiatry', 'psychiatrist': 'Psychiatry',
                                    'brain': 'Neurology', 'neurologist': 'Neurology', 'neuro': 'Neurology',
                                    'cancer': 'Oncology', 'oncologist': 'Oncology',
                                    'lung': 'Pulmonology', 'pulmonologist': 'Pulmonology',
                                    'kidney': 'Nephrology', 'nephrologist': 'Nephrology',
                                    'stomach': 'Gastroenterology', 'gi': 'Gastroenterology', 'gastro': 'Gastroenterology',
                                    'eye': 'ENT', 'ear': 'ENT',
                                    'women': 'OB/GYN', 'obgyn': 'OB/GYN', 'gynecology': 'OB/GYN',
                                    'child': 'Pediatrics', 'pediatrician': 'Pediatrics',
                                    'general': 'Internal Medicine', 'internist': 'Internal Medicine',
                                    'pt': 'Physical Therapy', 'physio': 'Physical Therapy',
                                    'surgeon': 'Surgery', 'surgical': 'Surgery',
                                    'er': 'Emergency Medicine', 'emergency': 'Emergency Medicine',
                                    'xray': 'Radiology', 'x-ray': 'Radiology', 'radiologist': 'Radiology',
                                    'joint': 'Rheumatology', 'rheumatologist': 'Rheumatology',
                                    'endo': 'Endocrinology', 'thyroid': 'Endocrinology', 'hormone': 'Endocrinology',
                                    'urine': 'Urology', 'urologist': 'Urology', 'bladder': 'Urology'}},
            'DEPARTMENT': {'type': 'categorical', 'desc': 'department'},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP region',
                         'values': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'GA', 'HI', 'MID']},
            'FACILITY': {'type': 'categorical', 'desc': 'facility name'},
            'PROVIDER_TYPE': {'type': 'categorical', 'desc': 'provider credential type',
                             'values': ['MD', 'NP', 'RN', 'DO', 'PA', 'LCSW', 'PhD'],
                             'counts': {'MD': 621, 'NP': 217, 'RN': 201, 'DO': 153, 'PA': 141, 'LCSW': 87, 'PhD': 80}},
            'DEA_NUMBER': {'type': 'id', 'desc': 'DEA registration number'},
            'LICENSE_STATE': {'type': 'categorical', 'desc': 'state of licensure',
                             'values': ['CA', 'VA', 'WA', 'GA', 'DC', 'MD', 'HI', 'CO', 'OR']},
            'HIRE_DATE': {'type': 'date', 'desc': 'hire date'},
            'STATUS': {'type': 'categorical', 'desc': 'provider status',
                      'values': ['ACTIVE', 'INACTIVE', 'ON_LEAVE'],
                      'counts': {'ACTIVE': 1297, 'INACTIVE': 139, 'ON_LEAVE': 64}},
            'PANEL_SIZE': {'type': 'numeric', 'desc': 'number of assigned patients', 'range': [101, 2498]},
            'ACCEPTS_NEW_PATIENTS': {'type': 'categorical', 'desc': 'accepting new patients', 'values': ['Y', 'N']},
        }
    },
    'diagnoses': {
        'row_count': 5068,
        'primary_key': 'DIAGNOSIS_ID',
        'columns': {
            'DIAGNOSIS_ID': {'type': 'id', 'desc': 'unique diagnosis identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'ENCOUNTER_ID': {'type': 'fk', 'desc': 'encounter foreign key', 'joins_to': 'encounters.ENCOUNTER_ID', 'null_pct': 30.16},
            'ICD10_CODE': {'type': 'code', 'desc': 'ICD-10 diagnosis code',
                          'values': ['M54.5', 'R51', 'I48.91', 'M17.11', 'J06.9', 'F32.1', 'G47.00', 'F41.1',
                                     'I25.10', 'E78.5', 'J45.20', 'K21.0', 'E11.65', 'N18.3', 'E66.01',
                                     'N39.0', 'G43.909', 'I10', 'J44.1', 'S82.001A', 'M79.3', 'R10.9',
                                     'J18.9', 'C50.911', 'Z96.641', 'U07.1']},
            'ICD10_DESCRIPTION': {'type': 'text', 'desc': 'diagnosis description'},
            'DIAGNOSIS_TYPE': {'type': 'categorical', 'desc': 'type of diagnosis',
                              'values': ['PRIMARY', 'DISCHARGE', 'ADMITTING', 'SECONDARY']},
            'DIAGNOSIS_DATE': {'type': 'date', 'desc': 'date of diagnosis'},
            'RESOLVED_DATE': {'type': 'date', 'desc': 'date resolved', 'null_pct': 61.42},
            'IS_CHRONIC': {'type': 'categorical', 'desc': 'is chronic condition', 'values': ['Y', 'N']},
            'SEVERITY': {'type': 'categorical', 'desc': 'diagnosis severity',
                        'values': ['MODERATE', 'MILD', 'SEVERE', 'CRITICAL']},
            'DIAGNOSING_NPI': {'type': 'fk', 'desc': 'diagnosing provider NPI', 'joins_to': 'providers.NPI'},
            'HCC_CODE': {'type': 'code', 'desc': 'HCC risk adjustment code', 'null_pct': 60.44},
            'HCC_CATEGORY': {'type': 'categorical', 'desc': 'HCC risk category',
                            'values': ['Mental Health', 'Cancer', 'CKD', 'Diabetes', 'COPD', 'Heart Failure', 'None', 'Respiratory']},
        }
    },
    'prescriptions': {
        'row_count': 3000,
        'primary_key': 'RX_ID',
        'columns': {
            'RX_ID': {'type': 'id', 'desc': 'prescription identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'PRESCRIBING_NPI': {'type': 'fk', 'desc': 'prescribing provider NPI', 'joins_to': 'providers.NPI'},
            'MEDICATION_NAME': {'type': 'categorical', 'desc': 'medication name',
                               'values': ['Azithromycin 250mg', 'Metoprolol 50mg', 'Lisinopril 10mg', 'Tramadol 50mg',
                                          'Atorvastatin 40mg', 'Amlodipine 5mg', 'Metformin 500mg', 'Omeprazole 20mg',
                                          'Sertraline 50mg', 'Levothyroxine 50mcg', 'Albuterol HFA', 'Gabapentin 300mg',
                                          'Losartan 50mg', 'Prednisone 10mg', 'Fluoxetine 20mg', 'Pantoprazole 40mg',
                                          'Hydrochlorothiazide 25mg', 'Cephalexin 500mg', 'Escitalopram 10mg', 'Alprazolam 0.5mg']},
            'MEDICATION_CLASS': {'type': 'categorical', 'desc': 'therapeutic drug class',
                                'values': ['Hypertension', 'Infection', 'Pain', 'Depression', 'GERD', 'Asthma',
                                          'Anxiety', 'Hypothyroid', 'Cholesterol', 'Diabetes'],
                                'aliases': {'blood pressure': 'Hypertension', 'bp': 'Hypertension', 'htn': 'Hypertension',
                                           'antibiotic': 'Infection', 'anti-infective': 'Infection',
                                           'painkiller': 'Pain', 'analgesic': 'Pain',
                                           'antidepressant': 'Depression', 'ssri': 'Depression',
                                           'acid reflux': 'GERD', 'heartburn': 'GERD', 'ppi': 'GERD',
                                           'inhaler': 'Asthma', 'bronchodilator': 'Asthma',
                                           'anti-anxiety': 'Anxiety', 'benzodiazepine': 'Anxiety', 'benzo': 'Anxiety',
                                           'thyroid': 'Hypothyroid', 'levothyroxine': 'Hypothyroid',
                                           'statin': 'Cholesterol', 'lipid': 'Cholesterol',
                                           'insulin': 'Diabetes', 'metformin': 'Diabetes', 'a1c': 'Diabetes'}},
            'NDC_CODE': {'type': 'code', 'desc': 'National Drug Code'},
            'QUANTITY': {'type': 'numeric', 'desc': 'quantity dispensed', 'values': [7, 14, 30, 60, 90, 120]},
            'DAYS_SUPPLY': {'type': 'numeric', 'desc': 'days supply', 'values': [7, 14, 30, 60, 90]},
            'REFILLS_AUTHORIZED': {'type': 'numeric', 'desc': 'refills authorized', 'range': [0, 11]},
            'REFILLS_USED': {'type': 'numeric', 'desc': 'refills used', 'range': [0, 11]},
            'PRESCRIPTION_DATE': {'type': 'date', 'desc': 'prescription date'},
            'FILL_DATE': {'type': 'date', 'desc': 'fill date', 'null_pct': 15.4},
            'PHARMACY': {'type': 'categorical', 'desc': 'dispensing pharmacy',
                        'values': ['KP Pharmacy LA', 'KP Mail Order', 'Walgreens', 'CVS', 'Rite Aid', 'KP Pharmacy Oakland']},
            'COST': {'type': 'numeric', 'desc': 'medication cost', 'range': [5.03, 799.75]},
            'COPAY': {'type': 'numeric', 'desc': 'prescription copay', 'values': [0, 5, 10, 15, 20, 25, 30, 50]},
            'STATUS': {'type': 'categorical', 'desc': 'prescription status',
                      'values': ['FILLED', 'PENDING', 'EXPIRED', 'TRANSFERRED', 'CANCELLED'],
                      'counts': {'FILLED': 2091, 'PENDING': 317, 'EXPIRED': 291}},
        }
    },
    'referrals': {
        'row_count': 1500,
        'primary_key': 'REFERRAL_ID',
        'columns': {
            'REFERRAL_ID': {'type': 'id', 'desc': 'unique referral identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'REFERRING_NPI': {'type': 'fk', 'desc': 'referring provider NPI', 'joins_to': 'providers.NPI'},
            'REFERRED_TO_NPI': {'type': 'fk', 'desc': 'receiving provider NPI', 'joins_to': 'providers.NPI'},
            'REFERRAL_DATE': {'type': 'date', 'desc': 'referral date'},
            'REFERRAL_REASON': {'type': 'categorical', 'desc': 'reason for referral',
                               'values': ['Specialist evaluation', 'Cancer screening', 'Post-op follow-up',
                                         'Cardiac evaluation', 'Pain management', 'Physical therapy',
                                         'Surgery consultation', 'Diagnostic workup', 'Chronic disease management',
                                         'Second opinion', 'Mental health evaluation']},
            'URGENCY': {'type': 'categorical', 'desc': 'referral urgency',
                       'values': ['ROUTINE', 'URGENT', 'ELECTIVE', 'STAT'],
                       'counts': {'ROUTINE': 750, 'URGENT': 391, 'ELECTIVE': 280, 'STAT': 79}},
            'REFERRAL_TYPE': {'type': 'categorical', 'desc': 'referral type',
                             'values': ['INTERNAL', 'EXTERNAL', 'SELF_REFERRAL']},
            'STATUS': {'type': 'categorical', 'desc': 'referral status',
                      'values': ['APPROVED', 'COMPLETED', 'PENDING', 'CANCELLED', 'DENIED', 'EXPIRED'],
                      'counts': {'APPROVED': 512, 'COMPLETED': 442, 'PENDING': 334}},
            'APPOINTMENT_DATE': {'type': 'date', 'desc': 'scheduled appointment date'},
            'SPECIALTY': {'type': 'categorical', 'desc': 'referred-to specialty',
                         'values': ['Urology', 'Pulmonology', 'Surgery', 'Cardiology', 'Nephrology', 'ENT',
                                   'Oncology', 'OB/GYN', 'Psychiatry', 'Rheumatology', 'Orthopedics',
                                   'Neurology', 'Dermatology', 'Gastroenterology', 'Physical Therapy',
                                   'Endocrinology', 'Emergency Medicine', 'Radiology', 'Internal Medicine', 'Pediatrics']},
            'AUTHORIZATION_NUMBER': {'type': 'id', 'desc': 'authorization number', 'null_pct': 30.67},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP region',
                         'values': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'GA', 'HI', 'MID']},
        }
    },
    'appointments': {
        'row_count': 2800,
        'primary_key': 'APPOINTMENT_ID',
        'columns': {
            'APPOINTMENT_ID': {'type': 'id', 'desc': 'unique appointment identifier'},
            'MEMBER_ID': {'type': 'fk', 'desc': 'member foreign key', 'joins_to': 'members.MEMBER_ID'},
            'PROVIDER_NPI': {'type': 'fk', 'desc': 'provider NPI', 'joins_to': 'providers.NPI'},
            'APPOINTMENT_DATE': {'type': 'date', 'desc': 'appointment date'},
            'APPOINTMENT_TIME': {'type': 'text', 'desc': 'appointment time HH:MM'},
            'APPOINTMENT_TYPE': {'type': 'categorical', 'desc': 'type of appointment',
                                'values': ['PCP_VISIT', 'SPECIALIST', 'FOLLOW_UP', 'ANNUAL_WELLNESS',
                                          'LAB_WORK', 'IMAGING', 'TELEHEALTH', 'URGENT']},
            'DEPARTMENT': {'type': 'categorical', 'desc': 'department'},
            'FACILITY': {'type': 'categorical', 'desc': 'facility name'},
            'KP_REGION': {'type': 'categorical', 'desc': 'KP region',
                         'values': ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'GA', 'HI', 'MID']},
            'STATUS': {'type': 'categorical', 'desc': 'appointment status',
                      'values': ['SCHEDULED', 'COMPLETED', 'CANCELLED', 'NO_SHOW', 'RESCHEDULED']},
            'REASON': {'type': 'text', 'desc': 'appointment reason'},
            'DURATION_MINUTES': {'type': 'numeric', 'desc': 'appointment duration in minutes'},
            'IS_PCP_VISIT': {'type': 'categorical', 'desc': 'is PCP visit', 'values': ['Y', 'N']},
        }
    },
    'cpt_codes': {
        'row_count': 81,
        'primary_key': 'CPT_CODE',
        'columns': {
            'CPT_CODE': {'type': 'id', 'desc': 'CPT procedure code'},
            'DESCRIPTION': {'type': 'text', 'desc': 'procedure description'},
            'CATEGORY': {'type': 'categorical', 'desc': 'CPT category',
                        'values': ['E&M', 'Preventive', 'Surgery', 'Lab', 'Radiology', 'Telehealth', 'Chronic Care']},
            'RVU': {'type': 'numeric', 'desc': 'Relative Value Units'},
        }
    },
}


class ValueResolver:
    """
    Resolves natural language value mentions to exact (table, column, value) tuples.

    When a user says "denied claims in NCAL", this resolver:
    - "denied" → (claims, CLAIM_STATUS, 'DENIED')
    - "NCAL" → (claims, KP_REGION, 'NCAL')

    This enables the engine to build correct WHERE clauses automatically.
    """

    _index = None  # Lazy-built reverse index

    @classmethod
    def _build_index(cls):
        """Build reverse index: lowercase_term → [(table, column, exact_value)]"""
        if cls._index is not None:
            return
        cls._index = {}

        for table_name, table_info in SCHEMA_KNOWLEDGE.items():
            for col_name, col_info in table_info['columns'].items():
                col_type = col_info.get('type', '')

                # Index categorical values
                if col_type == 'categorical' and 'values' in col_info:
                    for val in col_info['values']:
                        val_str = str(val)
                        # Index exact value (lowercase)
                        key = val_str.lower()
                        if key not in cls._index:
                            cls._index[key] = []
                        cls._index[key].append((table_name, col_name, val_str))

                        # Index without underscores: URGENT_CARE → "urgent care"
                        if '_' in val_str:
                            key2 = val_str.lower().replace('_', ' ')
                            if key2 not in cls._index:
                                cls._index[key2] = []
                            cls._index[key2].append((table_name, col_name, val_str))

                # Index aliases
                if 'aliases' in col_info:
                    for alias, exact_val in col_info['aliases'].items():
                        key = alias.lower()
                        if key not in cls._index:
                            cls._index[key] = []
                        cls._index[key].append((table_name, col_name, exact_val))

    @classmethod
    def resolve_values(cls, question: str) -> list:
        """
        Find all value mentions in a question.
        Returns: [(table, column, exact_value, matched_text), ...]

        Strategy:
        1. Check multi-word phrases first (longest match wins)
        2. Then check single words
        3. Filter out false positives (common English words that happen to match values)
        """
        cls._build_index()
        q_lower = question.lower()
        results = []
        matched_spans = []  # Track matched character positions to avoid overlaps

        # FALSE POSITIVE words — common English words that match column values
        # These should NOT trigger value resolution unless in clear context
        FALSE_POSITIVE_WORDS = {
            'do', 'no', 'or', 'a', 'i', 'in', 'on', 'to', 'is', 'it', 'at', 'as',
            'am', 'an', 'be', 'by', 'if', 'of', 'so', 'up', 'my', 'me', 'we', 'us',
            'y', 'n',  # Could match Y/N categorical values
        }

        # Sort index keys by length (longest first) for greedy matching
        sorted_keys = sorted(cls._index.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if len(key) < 2:
                continue
            if key in FALSE_POSITIVE_WORDS:
                continue

            # Check if key appears as a word boundary in the question
            pattern = r'\b' + re.escape(key) + r'\b'
            for m in re.finditer(pattern, q_lower):
                start, end = m.start(), m.end()
                # Skip if overlapping with already matched span
                if any(start < ms_end and end > ms_start for ms_start, ms_end in matched_spans):
                    continue

                matched_spans.append((start, end))
                for (table, column, exact_val) in cls._index[key]:
                    results.append((table, column, exact_val, key))

        return results

    @classmethod
    def get_table_scores_from_values(cls, question: str) -> dict:
        """
        Score tables based on value mentions in the question.
        Returns: {table_name: score}

        This supplements TableResolver's keyword matching with actual data value awareness.
        """
        value_matches = cls.resolve_values(question)
        scores = defaultdict(float)
        for table, column, exact_val, matched_text in value_matches:
            # Higher score for more specific matches
            if len(matched_text) >= 4:
                scores[table] += 3.0  # Strong evidence: a specific data value
            else:
                scores[table] += 1.5
        return dict(scores)

    @classmethod
    def get_auto_filters(cls, question: str, tables: list) -> list:
        """
        Generate automatic WHERE clause filters based on value mentions.
        Only generates filters for tables that are already selected.

        Returns: list of SQL condition strings
        """
        value_matches = cls.resolve_values(question)
        filters = []
        seen = set()

        q_lower = question.lower()

        for table, column, exact_val, matched_text in value_matches:
            if table not in tables:
                continue

            # Skip if this is being used as a column qualifier context
            # e.g., "paid amount" → "paid" should not trigger CLAIM_STATUS='PAID'
            qualifier_contexts = {
                'paid': ['paid amount', 'paid_amount', 'paid amt', 'avg paid', 'average paid',
                         'total paid', 'sum paid', 'max paid', 'min paid', 'paid vs'],
                'denied': ['denied amount', 'denied_amount', 'denied claims amount', 'denial rate'],
                'pending': ['pending amount', 'pending claims amount'],
            }
            skip = False
            for word, contexts in qualifier_contexts.items():
                if matched_text == word and any(ctx in q_lower for ctx in contexts):
                    skip = True
                    break
            if skip:
                continue

            # Skip if this value is part of a "breakdown" / "distribution" / "by" context
            # e.g., "breakdown of paid denied adjusted" → these are GROUP BY targets, not filters
            if re.search(r'\b(breakdown|distribution|split|ratio)\b', q_lower):
                # Check if the matched value is one of several status values mentioned together
                status_words_in_q = sum(1 for sv in ['paid', 'denied', 'adjusted', 'pending', 'appealed', 'voided']
                                       if sv in q_lower)
                if status_words_in_q >= 2 and matched_text in ['paid', 'denied', 'adjusted', 'pending', 'appealed', 'voided']:
                    continue

            # Deduplicate
            filter_key = (column, exact_val)
            if filter_key in seen:
                continue
            seen.add(filter_key)

            # Generate the SQL condition
            if isinstance(exact_val, (int, float)):
                filters.append(f"{column} = {exact_val}")
            else:
                filters.append(f"{column} = '{exact_val}'")

        return filters


def _fuzzy_match(term: str, candidate: str, max_dist: int = 2) -> bool:
    """Quick edit-distance check for typo tolerance (Levenshtein ≤ max_dist).
    Works for common misspellings like 'memeber'→'member', 'staus'→'status'."""
    a, b = term.lower(), candidate.lower()
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False
    # Simple DP Levenshtein — terms are short so this is fast
    la, lb = len(a), len(b)
    if la > 15 or lb > 15:  # skip for long strings
        return False
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    return prev[lb] <= max_dist


# =============================================================================
# SCHEMA REGISTRY — auto-discovered from catalog JSONs
# =============================================================================

class SchemaRegistry:
    """
    Maintains full knowledge of all tables, columns, types, and relationships.
    Auto-discovers from the semantic_catalog directory.
    """

    # ── Composite Join Keys ──
    # In KP production (Databricks), MEMBER_ID alone is NOT unique across regions.
    # Tables are partitioned by KP_REGION, so joins need MEMBER_ID + KP_REGION.
    # This mapping defines which secondary columns to add to joins for each primary key.
    # Format: primary_join_col -> [secondary_col1, secondary_col2, ...]
    # Only applied when BOTH tables in a join contain the secondary column.
    COMPOSITE_JOIN_KEYS = {
        'MEMBER_ID': ['KP_REGION'],       # KP: member + region = unique
        # 'ENCOUNTER_ID': ['KP_REGION'],  # Uncomment if encounters also need region
    }

    # Set to True for KP production / Databricks; False for local demo SQLite
    USE_COMPOSITE_JOINS = False  # Auto-detected or manually set

    def __init__(self, catalog_dir: str = None):
        self.tables: Dict[str, List[Dict]] = {}       # table -> [{col_name, data_type, ...}]
        self.column_to_tables: Dict[str, List[str]] = defaultdict(list)  # COL_NAME -> [table1, table2]
        self.relationships: List[Dict] = []             # [{source, target, join_col, type}]
        self.join_graph: Dict[str, Dict[str, str]] = defaultdict(dict)  # table1 -> {table2: join_col}

        if catalog_dir:
            self._load_from_catalog(catalog_dir)

    def _load_from_catalog(self, catalog_dir: str):
        """Load schema from semantic catalog JSON files."""
        tables_dir = os.path.join(catalog_dir, "tables")
        rels_dir = os.path.join(catalog_dir, "relationships")

        # Load tables
        if os.path.exists(tables_dir):
            for fname in os.listdir(tables_dir):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(tables_dir, fname), 'r') as f:
                        data = json.load(f)
                    table_name = data.get('table_name', fname.replace('.json', ''))
                    cols_raw = data.get('columns', {})
                    if isinstance(cols_raw, dict):
                        cols_raw = list(cols_raw.values())

                    cols = []
                    for c in cols_raw:
                        col_name = c.get('column_name', c.get('name', ''))
                        cols.append({
                            'name': col_name,
                            'data_type': c.get('data_type', c.get('semantic_type', 'text')),
                            'semantic_type': c.get('semantic_type', ''),
                            'healthcare_type': c.get('healthcare_type', ''),
                            'cardinality': c.get('cardinality', 0),
                            'null_pct': c.get('null_percentage', 0),
                            'top_values': c.get('top_values', []),
                        })
                        self.column_to_tables[col_name.upper()].append(table_name)

                    self.tables[table_name] = cols
                except Exception as e:
                    pass

        # Load relationships
        if os.path.exists(rels_dir):
            for fname in os.listdir(rels_dir):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(rels_dir, fname), 'r') as f:
                        data = json.load(f)
                    rels = data.get('relationships', data) if isinstance(data, dict) else data
                    if isinstance(rels, list):
                        self.relationships = rels
                except Exception:
                    pass

        # Build join graph — bidirectional
        self._build_join_graph()

    def _build_join_graph(self):
        """Build a graph of how tables can be joined."""
        # Priority join keys (most specific to least)
        PRIORITY_KEYS = {
            'encounter_id': 10,  # Very specific — 1:1 match
            'claim_id': 10,
            'diagnosis_id': 10,
            'referral_id': 10,
            'rx_id': 10,
            'member_id': 8,      # Common hub — good cardinality
            'mrn': 8,
            'rendering_npi': 7,
            'npi': 7,
            'icd10_code': 6,
            'kp_region': 1,      # LOW specificity — creates huge cross products!
            'facility': 1,       # LOW — same issue
            'department': 1,
            'status': 0,         # NEVER join on status — too generic
            'specialty': 1,
            'copay': 0,          # Never join on copay
        }

        # Collect all possible joins between each pair
        pair_joins = defaultdict(list)  # (t1, t2) -> [(join_col, priority)]

        for rel in self.relationships:
            src = rel.get('source_table', '')
            tgt = rel.get('target_table', '')
            jcol = rel.get('join_column', '')
            if src and tgt and jcol:
                priority = PRIORITY_KEYS.get(jcol.lower(), 5)
                if priority > 0:  # Skip zero-priority keys entirely
                    pair_joins[(src, tgt)].append((jcol, priority))
                    pair_joins[(tgt, src)].append((jcol, priority))

        # Add MANUAL high-quality joins that the relationship detector may have missed
        MANUAL_JOINS = [
            ('providers', 'claims', 'NPI', 'RENDERING_NPI', 9),
            ('providers', 'encounters', 'NPI', 'RENDERING_NPI', 9),
            ('providers', 'prescriptions', 'NPI', 'PRESCRIBING_NPI', 9),
            ('providers', 'referrals', 'NPI', 'REFERRING_NPI', 9),
            ('providers', 'diagnoses', 'NPI', 'DIAGNOSING_NPI', 9),
            ('claims', 'encounters', 'ENCOUNTER_ID', 'ENCOUNTER_ID', 10),
            ('diagnoses', 'encounters', 'ENCOUNTER_ID', 'ENCOUNTER_ID', 10),
        ]
        for t1, t2, col1, col2, pri in MANUAL_JOINS:
            pair_joins[(t1, t2)].append((f"{col1}={col2}", pri))
            pair_joins[(t2, t1)].append((f"{col2}={col1}", pri))

        # Keep only the best (highest priority) join for each pair
        for (t1, t2), joins in pair_joins.items():
            best_col, best_pri = max(joins, key=lambda x: x[1])
            # Only use joins with priority >= 2 (skip generic region/facility joins)
            if best_pri >= 2:
                self.join_graph[t1][t2] = best_col.upper()

    # Tables where MEMBER_ID should be a primary/unique key (dimension tables).
    # If MEMBER_ID is NOT unique in these tables, composite joins are needed.
    _MEMBER_DIMENSION_TABLES = {'members'}

    def detect_composite_join_need(self, db_path: str = None):
        """
        Auto-detect if composite joins are needed by checking if MEMBER_ID
        is truly unique in the DIMENSION tables (e.g., 'members').

        In KP production (Databricks), the members table is partitioned
        by KP_REGION, so the same MEMBER_ID can appear in multiple regions.
        In that case, MEMBER_ID alone is not unique → we need composite joins.

        For local demo SQLite where MEMBER_ID is unique in members, single joins are correct.
        """
        if not db_path:
            return
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            for table_name in self._MEMBER_DIMENSION_TABLES:
                if table_name not in self.tables:
                    continue
                col_names = [c['name'].upper() for c in self.tables[table_name]]
                if 'MEMBER_ID' not in col_names or 'KP_REGION' not in col_names:
                    continue

                total = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                unique_mid = conn.execute(f"SELECT COUNT(DISTINCT MEMBER_ID) FROM {table_name}").fetchone()[0]

                if total > unique_mid:
                    # MEMBER_ID is NOT unique in the dimension table → need composite joins
                    SchemaRegistry.USE_COMPOSITE_JOINS = True
                    conn.close()
                    return

            # MEMBER_ID is unique in all dimension tables → single joins are fine
            SchemaRegistry.USE_COMPOSITE_JOINS = False
            conn.close()
        except Exception:
            pass

    def get_composite_join_conditions(self, t1: str, t2: str, primary_col: str) -> List[str]:
        """
        Get additional join conditions for composite keys.
        Returns list of extra conditions like 'KP_REGION' if both tables have it.
        Only active when USE_COMPOSITE_JOINS is True.
        """
        if not SchemaRegistry.USE_COMPOSITE_JOINS:
            return []

        primary_upper = primary_col.upper().split('=')[0]  # Handle asymmetric like NPI=RENDERING_NPI
        secondary_cols = SchemaRegistry.COMPOSITE_JOIN_KEYS.get(primary_upper, [])

        extra = []
        t1_cols = {c['name'].upper() for c in self.tables.get(t1, [])}
        t2_cols = {c['name'].upper() for c in self.tables.get(t2, [])}

        for sec_col in secondary_cols:
            if sec_col in t1_cols and sec_col in t2_cols:
                extra.append(sec_col)

        return extra

    def get_columns(self, table: str) -> List[str]:
        """Get all column names for a table."""
        return [c['name'] for c in self.tables.get(table, [])]

    def get_column_info(self, table: str, col_name: str) -> Optional[Dict]:
        """Get metadata for a specific column."""
        for c in self.tables.get(table, []):
            if c['name'].upper() == col_name.upper():
                return c
        return None

    def find_join_path(self, tables: List[str]) -> List[Tuple[str, str, str]]:
        """
        Find the best JOIN path connecting all required tables.

        Returns list of (table1, table2, join_column) tuples.
        Uses BFS from the first table to find shortest paths.
        """
        if len(tables) <= 1:
            return []

        base = tables[0]
        joins = []
        connected = {base}

        for target in tables[1:]:
            if target in connected:
                continue

            # BFS from connected set to target
            path = self._bfs_path(base, target, connected)
            if path:
                for t1, t2, col in path:
                    if t2 not in connected:
                        joins.append((t1, t2, col))
                        connected.add(t2)
            else:
                # Direct join attempt via member_id (hub table)
                if 'members' not in connected and 'members' in self.join_graph:
                    if base in self.join_graph.get('members', {}):
                        joins.append(('members', base, self.join_graph['members'][base]))
                        connected.add('members')
                joins.append((base, target, 'MEMBER_ID'))  # fallback
                connected.add(target)

        return joins

    def _bfs_path(self, start: str, end: str, connected: Set[str]) -> List[Tuple[str, str, str]]:
        """BFS to find shortest join path from start to end."""
        if start == end:
            return []
        visited = set()
        queue = [(start, [])]
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor, join_col in self.join_graph.get(current, {}).items():
                if neighbor not in visited:
                    queue.append((neighbor, path + [(current, neighbor, join_col)]))
        return []

    def all_table_names(self) -> List[str]:
        return list(self.tables.keys())


# =============================================================================
# COLUMN RESOLVER — maps natural language → actual columns + tables
# =============================================================================

class ColumnResolver:
    """
    Resolves natural language terms to actual database columns.

    Strategy (in priority order):
    1. Exact column name match (case-insensitive)
    2. Synonym lookup
    3. Substring match against column names
    4. Fuzzy word overlap
    """

    def __init__(self, schema: SchemaRegistry):
        self.schema = schema
        # Build reverse index: column_name_lower -> [(table, col_name)]
        self._col_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for table, cols in schema.tables.items():
            for c in cols:
                cn = c['name'].upper()
                self._col_index[cn.lower()].append((table, cn))

    def resolve(self, question: str, hint_tables: List[str] = None) -> List[Dict]:
        """
        Extract column references from a natural language question.

        Returns list of:
            {'column': 'COL_NAME', 'table': 'table_name', 'match_type': 'exact|synonym|fuzzy',
             'role': 'select|filter|group|order', 'original_term': 'the matched word(s)'}
        """
        q = question.lower()
        words = re.findall(r'[a-z_]+', q)
        resolved = []
        seen = set()  # avoid duplicates

        # Phase 1: Multi-word synonym matches (longer phrases first)
        # Use word boundary matching to prevent "age" matching inside "average"
        # Also supports fuzzy matching for common typos (edit distance ≤ 2)
        sorted_syns = sorted(SYNONYMS.keys(), key=len, reverse=True)
        for phrase in sorted_syns:
            # Exact word boundary match first
            exact_match = bool(re.search(r'\b' + re.escape(phrase) + r'\b', q))
            # Fuzzy match: check each word against single-word synonyms
            fuzzy_match = False
            if not exact_match and ' ' not in phrase and len(phrase) >= 4:
                for word in words:
                    if _fuzzy_match(word, phrase):
                        fuzzy_match = True
                        break
            if exact_match or fuzzy_match:
                for col_name in SYNONYMS[phrase]:
                    tables_with_col = self.schema.column_to_tables.get(col_name, [])
                    if hint_tables:
                        # Prefer columns from hinted tables
                        for t in hint_tables:
                            if t in tables_with_col:
                                key = (col_name, t)
                                if key not in seen:
                                    resolved.append({
                                        'column': col_name, 'table': t,
                                        'match_type': 'synonym', 'original_term': phrase,
                                    })
                                    seen.add(key)
                                break
                        else:
                            if tables_with_col:
                                key = (col_name, tables_with_col[0])
                                if key not in seen:
                                    resolved.append({
                                        'column': col_name, 'table': tables_with_col[0],
                                        'match_type': 'synonym', 'original_term': phrase,
                                    })
                                    seen.add(key)
                    elif tables_with_col:
                        key = (col_name, tables_with_col[0])
                        if key not in seen:
                            resolved.append({
                                'column': col_name, 'table': tables_with_col[0],
                                'match_type': 'synonym', 'original_term': phrase,
                            })
                            seen.add(key)

        # Phase 2: Single word → exact column name match
        for word in words:
            upper = word.upper()
            if upper in self.schema.column_to_tables:
                for t in self.schema.column_to_tables[upper]:
                    key = (upper, t)
                    if key not in seen:
                        resolved.append({
                            'column': upper, 'table': t,
                            'match_type': 'exact', 'original_term': word,
                        })
                        seen.add(key)

        # Phase 3: Substring match against column names
        # Skip noise words that match column names by accident
        # e.g., "number" matching AUTHORIZATION_NUMBER, "count" matching ENCOUNTER_ID
        SUBSTRING_SKIP_WORDS = {
            'number', 'count', 'total', 'give', 'show', 'list', 'find', 'have',
            'highest', 'lowest', 'which', 'what', 'most', 'many', 'much', 'each',
            'every', 'with', 'from', 'that', 'this', 'type', 'time', 'first',
            'last', 'over', 'more', 'less', 'than', 'about', 'into', 'some',
        }
        for word in words:
            if len(word) < 4 or word in SUBSTRING_SKIP_WORDS:
                continue
            for col_lower, entries in self._col_index.items():
                if word in col_lower and (col_lower, entries[0][0]) not in seen:
                    for t, cn in entries:
                        key = (cn, t)
                        if key not in seen:
                            resolved.append({
                                'column': cn, 'table': t,
                                'match_type': 'substring', 'original_term': word,
                            })
                            seen.add(key)

        return resolved


# =============================================================================
# FILTER EXTRACTOR — extracts WHERE conditions from natural language
# =============================================================================

class FilterExtractor:
    """Extracts WHERE clause conditions from natural language."""

    # Value patterns
    # Words that suppress status filter when used as column qualifiers (Bug 4)
    COLUMN_QUALIFIER_CONTEXTS = {
        'paid': ['paid amount', 'paid_amount', 'paid amt', 'avg paid', 'average paid',
                 'total paid', 'sum paid', 'max paid', 'min paid'],
        'billed': ['billed amount', 'billed_amount', 'billed amt'],
        'allowed': ['allowed amount', 'allowed_amount'],
        'denied': ['denied amount', 'denied_amount', 'denied claims amount'],
    }

    STATUS_VALUES = {
        'denied': ('CLAIM_STATUS', '=', 'DENIED'),
        'rejected': ('CLAIM_STATUS', '=', 'DENIED'),
        'paid': ('CLAIM_STATUS', '=', 'PAID'),
        'approved': ('STATUS', '=', 'APPROVED'),
        'pending': ('CLAIM_STATUS', '=', 'PENDING'),
        'adjusted': ('CLAIM_STATUS', '=', 'ADJUSTED'),
        'appealed': ('CLAIM_STATUS', '=', 'APPEALED'),
        'active': ('STATUS', '=', 'ACTIVE'),
        'inactive': ('STATUS', '=', 'INACTIVE'),
        'on leave': ('STATUS', '=', 'ON_LEAVE'),
        'cancelled': ('STATUS', '=', 'CANCELLED'),
        'completed': ('STATUS', '=', 'COMPLETED'),
        'inpatient': ('VISIT_TYPE', '=', 'INPATIENT'),
        'outpatient': ('VISIT_TYPE', '=', 'OUTPATIENT'),
        'emergency': ('VISIT_TYPE', '=', 'EMERGENCY'),
        'telehealth': ('VISIT_TYPE', '=', 'TELEHEALTH'),
        'urgent': ('URGENCY', '=', 'URGENT'),
        'male': ('GENDER', '=', 'M'),
        'female': ('GENDER', '=', 'F'),
        'chronic': ('IS_CHRONIC', '=', 'Y'),
    }

    # Comparison patterns
    COMPARISON_RE = [
        # "more than $5000", "over 100", "greater than 50", "above 80"
        (r'(?:more than|over|greater than|above|exceeds?|>)\s*\$?([\d,]+\.?\d*)', '>', None),
        # "less than $1000", "under 50", "below 30"
        (r'(?:less than|under|below|fewer than|<)\s*\$?([\d,]+\.?\d*)', '<', None),
        # "at least 10", "minimum 5"
        (r'(?:at least|minimum|min)\s*\$?([\d,]+\.?\d*)', '>=', None),
        # "at most 100", "maximum 50"
        (r'(?:at most|maximum|max)\s*\$?([\d,]+\.?\d*)', '<=', None),
        # "between X and Y"
        (r'between\s*\$?([\d,]+\.?\d*)\s*and\s*\$?([\d,]+\.?\d*)', 'BETWEEN', None),
        # "equals 5", "equal to 10", "exactly 3"
        (r'(?:equals?|equal to|exactly)\s*\$?([\d,]+\.?\d*)', '=', None),
    ]

    # ── Domain Concept Map ──────────────────────────────────────────────
    # Maps healthcare domain terms to SQL conditions + required tables.
    # Each concept: {'conditions': [SQL WHERE clauses], 'tables': [required tables],
    #                'columns': [columns referenced]}
    # This enables questions like "foster care members on medicaid by region"
    # to produce: WHERE PLAN_TYPE = 'Medicaid' AND (date('now') - DATE_OF_BIRTH) < 21
    DOMAIN_CONCEPTS = {
        # ── Age/Demographic Concepts ──
        'newborn': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-2 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'infant': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-2 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'toddler': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-4 years') AND DATE_OF_BIRTH < date('now', '-2 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'pediatric': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-18 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'child': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-18 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'children': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-18 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'adolescent': {
            'conditions': ["DATE_OF_BIRTH BETWEEN date('now', '-18 years') AND date('now', '-12 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'teen': {
            'conditions': ["DATE_OF_BIRTH BETWEEN date('now', '-18 years') AND date('now', '-13 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'elderly': {
            'conditions': ["DATE_OF_BIRTH <= date('now', '-65 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'senior': {
            'conditions': ["DATE_OF_BIRTH <= date('now', '-65 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'geriatric': {
            'conditions': ["DATE_OF_BIRTH <= date('now', '-65 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'adult': {
            'conditions': ["DATE_OF_BIRTH <= date('now', '-18 years') AND DATE_OF_BIRTH > date('now', '-65 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'minor': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-18 years')"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH'],
        },
        'foster care': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-21 years')", "PLAN_TYPE = 'Medicaid'"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH', 'PLAN_TYPE'],
        },
        'foster': {
            'conditions': ["DATE_OF_BIRTH >= date('now', '-21 years')", "PLAN_TYPE = 'Medicaid'"],
            'tables': ['members'], 'columns': ['DATE_OF_BIRTH', 'PLAN_TYPE'],
        },

        # ── Insurance/Plan Concepts ──
        'medicaid': {
            'conditions': ["PLAN_TYPE = 'Medicaid'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'medicare': {
            'conditions': ["PLAN_TYPE = 'Medicare Advantage'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'commercial': {
            'conditions': ["PLAN_TYPE IN ('HMO', 'PPO', 'EPO', 'HDHP')"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'hmo': {
            'conditions': ["PLAN_TYPE = 'HMO'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'ppo': {
            'conditions': ["PLAN_TYPE = 'PPO'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'epo': {
            'conditions': ["PLAN_TYPE = 'EPO'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },
        'hdhp': {
            'conditions': ["PLAN_TYPE = 'HDHP'"],
            'tables': ['members'], 'columns': ['PLAN_TYPE'],
        },

        # ── Clinical Condition Concepts (ICD10-based) ──
        'diabetic': {
            'conditions': ["ICD10_CODE LIKE 'E11%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'diabetes': {
            'conditions': ["ICD10_CODE LIKE 'E11%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'hypertensive': {
            'conditions': ["ICD10_CODE LIKE 'I10%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'hypertension': {
            'conditions': ["ICD10_CODE LIKE 'I10%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'cardiac': {
            'conditions': ["ICD10_CODE LIKE 'I25%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'heart disease': {
            'conditions': ["ICD10_CODE LIKE 'I25%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'asthma': {
            'conditions': ["ICD10_CODE LIKE 'J45%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'copd': {
            'conditions': ["ICD10_CODE LIKE 'J44%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'depression': {
            'conditions': ["ICD10_CODE LIKE 'F32%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'anxiety': {
            'conditions': ["ICD10_CODE LIKE 'F41%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'ckd': {
            'conditions': ["ICD10_CODE LIKE 'N18%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'kidney disease': {
            'conditions': ["ICD10_CODE LIKE 'N18%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'obesity': {
            'conditions': ["ICD10_CODE LIKE 'E66%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'obese': {
            'conditions': ["ICD10_CODE LIKE 'E66%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'pneumonia': {
            'conditions': ["ICD10_CODE LIKE 'J18%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'uti': {
            'conditions': ["ICD10_CODE LIKE 'N39%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },
        'migraine': {
            'conditions': ["ICD10_CODE LIKE 'G43%'"],
            'tables': ['claims'], 'columns': ['ICD10_CODE'],
        },

        # ── HCC Risk Category Concepts ──
        'heart failure': {
            'conditions': ["HCC_CATEGORY = 'Heart Failure'"],
            'tables': ['diagnoses'], 'columns': ['HCC_CATEGORY'],
        },
        'cancer': {
            'conditions': ["HCC_CATEGORY = 'Cancer'"],
            'tables': ['diagnoses'], 'columns': ['HCC_CATEGORY'],
        },
        'mental health': {
            'conditions': ["HCC_CATEGORY = 'Mental Health'"],
            'tables': ['diagnoses'], 'columns': ['HCC_CATEGORY'],
        },
        'behavioral health': {
            'conditions': ["HCC_CATEGORY = 'Mental Health'"],
            'tables': ['diagnoses'], 'columns': ['HCC_CATEGORY'],
        },

        # ── Risk Stratification Concepts ──
        'high risk': {
            'conditions': ["CAST(RISK_SCORE AS REAL) >= 3.5"],
            'tables': ['members'], 'columns': ['RISK_SCORE'],
        },
        'low risk': {
            'conditions': ["CAST(RISK_SCORE AS REAL) < 2.0"],
            'tables': ['members'], 'columns': ['RISK_SCORE'],
        },
        'moderate risk': {
            'conditions': ["CAST(RISK_SCORE AS REAL) BETWEEN 2.0 AND 3.49"],
            'tables': ['members'], 'columns': ['RISK_SCORE'],
        },
        'rising risk': {
            'conditions': ["CAST(RISK_SCORE AS REAL) BETWEEN 3.0 AND 4.0"],
            'tables': ['members'], 'columns': ['RISK_SCORE'],
        },
        'complex care': {
            'conditions': ["CAST(RISK_SCORE AS REAL) >= 4.0", "CAST(CHRONIC_CONDITIONS AS INTEGER) >= 3"],
            'tables': ['members'], 'columns': ['RISK_SCORE', 'CHRONIC_CONDITIONS'],
        },
        'multi-morbid': {
            'conditions': ["CAST(CHRONIC_CONDITIONS AS INTEGER) >= 3"],
            'tables': ['members'], 'columns': ['CHRONIC_CONDITIONS'],
        },
        'comorbid': {
            'conditions': ["CAST(CHRONIC_CONDITIONS AS INTEGER) >= 2"],
            'tables': ['members'], 'columns': ['CHRONIC_CONDITIONS'],
        },

        # ── Visit Type Concepts ──
        'ed visit': {
            'conditions': ["VISIT_TYPE = 'EMERGENCY'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
        },
        'er visit': {
            'conditions': ["VISIT_TYPE = 'EMERGENCY'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
        },
        'virtual visit': {
            'conditions': ["VISIT_TYPE = 'TELEHEALTH'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
        },
        'telehealth': {
            'conditions': ["VISIT_TYPE = 'TELEHEALTH'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
        },
        'home health': {
            'conditions': ["VISIT_TYPE = 'HOME_HEALTH'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
        },
        'readmission': {
            'conditions': ["VISIT_TYPE = 'INPATIENT'", "DISPOSITION = 'Admitted'"],
            'tables': ['encounters'], 'columns': ['VISIT_TYPE', 'DISPOSITION'],
        },

        # ── Claim Outcome Concepts ──
        'clean claim': {
            'conditions': ["CLAIM_STATUS = 'PAID'", "DENIAL_REASON = ''"],
            'tables': ['claims'], 'columns': ['CLAIM_STATUS', 'DENIAL_REASON'],
        },
        'rejected claim': {
            'conditions': ["CLAIM_STATUS = 'DENIED'"],
            'tables': ['claims'], 'columns': ['CLAIM_STATUS'],
        },
        'pending claim': {
            'conditions': ["CLAIM_STATUS = 'PENDING'"],
            'tables': ['claims'], 'columns': ['CLAIM_STATUS'],
        },

        # ── Medication Concepts ──
        'polypharmacy': {
            'conditions': [],  # Handled via HAVING COUNT(DISTINCT MEDICATION_NAME) >= 5
            'tables': ['prescriptions'], 'columns': ['MEDICATION_NAME'],
        },
        'pain medication': {
            'conditions': ["MEDICATION_CLASS = 'Pain'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },
        'antidepressant': {
            'conditions': ["MEDICATION_CLASS = 'Depression'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },
        'blood pressure medication': {
            'conditions': ["MEDICATION_CLASS = 'Hypertension'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },
        'antibiotic': {
            'conditions': ["MEDICATION_CLASS = 'Infection'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },
        'statin': {
            'conditions': ["MEDICATION_CLASS = 'Cholesterol'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },
        'insulin': {
            'conditions': ["MEDICATION_CLASS = 'Diabetes'"],
            'tables': ['prescriptions'], 'columns': ['MEDICATION_CLASS'],
        },

        # ── Provider Concepts ──
        'specialist': {
            'conditions': ["SPECIALTY NOT IN ('Internal Medicine', 'Pediatrics')"],
            'tables': ['providers'], 'columns': ['SPECIALTY'],
        },
        'primary care': {
            'conditions': ["SPECIALTY IN ('Internal Medicine', 'Pediatrics')"],
            'tables': ['providers'], 'columns': ['SPECIALTY'],
        },
        'pcp': {
            'conditions': ["SPECIALTY IN ('Internal Medicine', 'Pediatrics')"],
            'tables': ['providers'], 'columns': ['SPECIALTY'],
        },
        'accepting new patients': {
            'conditions': ["ACCEPTS_NEW_PATIENTS = 'Y'"],
            'tables': ['providers'], 'columns': ['ACCEPTS_NEW_PATIENTS'],
        },
        'on leave': {
            'conditions': ["STATUS = 'ON_LEAVE'"],
            'tables': ['providers'], 'columns': ['STATUS'],
        },

        # ── Referral Concepts ──
        'stat referral': {
            'conditions': ["URGENCY = 'STAT'"],
            'tables': ['referrals'], 'columns': ['URGENCY'],
        },
        'internal referral': {
            'conditions': ["REFERRAL_TYPE = 'INTERNAL'"],
            'tables': ['referrals'], 'columns': ['REFERRAL_TYPE'],
        },
        'external referral': {
            'conditions': ["REFERRAL_TYPE = 'EXTERNAL'"],
            'tables': ['referrals'], 'columns': ['REFERRAL_TYPE'],
        },

        # ── Enrollment Concepts ──
        'new member': {
            'conditions': ["ENROLLMENT_DATE >= date((SELECT MAX(ENROLLMENT_DATE) FROM members), '-6 months')"],
            'tables': ['members'], 'columns': ['ENROLLMENT_DATE'],
        },
        'newly enrolled': {
            'conditions': ["ENROLLMENT_DATE >= date((SELECT MAX(ENROLLMENT_DATE) FROM members), '-6 months')"],
            'tables': ['members'], 'columns': ['ENROLLMENT_DATE'],
        },
        'disenrolled': {
            'conditions': ["DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''"],
            'tables': ['members'], 'columns': ['DISENROLLMENT_DATE'],
        },
        'active member': {
            'conditions': ["(DISENROLLMENT_DATE IS NULL OR DISENROLLMENT_DATE = '' OR DISENROLLMENT_DATE > date('now'))"],
            'tables': ['members'], 'columns': ['DISENROLLMENT_DATE'],
        },

        # ── Disposition/Outcome Concepts ──
        'discharged': {
            'conditions': ["DISPOSITION = 'Discharged'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'admitted': {
            'conditions': ["DISPOSITION = 'Admitted'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'transferred': {
            'conditions': ["DISPOSITION = 'Transferred'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'ama': {
            'conditions': ["DISPOSITION = 'AMA'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'left ama': {
            'conditions': ["DISPOSITION = 'AMA'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'expired': {
            'conditions': ["DISPOSITION = 'Expired'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },
        'mortality': {
            'conditions': ["DISPOSITION = 'Expired'"],
            'tables': ['encounters'], 'columns': ['DISPOSITION'],
        },

        # ── Claim Value Concepts ──
        'high-value claim': {
            'conditions': ["CAST(BILLED_AMOUNT AS REAL) > 10000"],
            'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
        },
        'high value claim': {
            'conditions': ["CAST(BILLED_AMOUNT AS REAL) > 10000"],
            'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
        },
        'high-value': {
            'conditions': ["CAST(BILLED_AMOUNT AS REAL) > 10000"],
            'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
        },
        'low-value claim': {
            'conditions': ["CAST(BILLED_AMOUNT AS REAL) < 1000"],
            'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
        },
        'expensive claim': {
            'conditions': ["CAST(BILLED_AMOUNT AS REAL) > 10000"],
            'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
        },

        # ── Processing Time Concepts ──
        'processing time': {
            'conditions': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"],
            'tables': ['claims'], 'columns': ['SUBMITTED_DATE', 'ADJUDICATED_DATE'],
        },
        'turnaround time': {
            'conditions': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"],
            'tables': ['claims'], 'columns': ['SUBMITTED_DATE', 'ADJUDICATED_DATE'],
        },
        'claim processing': {
            'conditions': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"],
            'tables': ['claims'], 'columns': ['SUBMITTED_DATE', 'ADJUDICATED_DATE'],
        },

        # ── Claim Type Concepts ──
        'professional claim': {
            'conditions': ["CLAIM_TYPE = 'PROFESSIONAL'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'institutional claim': {
            'conditions': ["CLAIM_TYPE = 'INSTITUTIONAL'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'pharmacy claim': {
            'conditions': ["CLAIM_TYPE = 'PHARMACY'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'dme claim': {
            'conditions': ["CLAIM_TYPE = 'DME'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'dme': {
            'conditions': ["CLAIM_TYPE = 'DME'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'institutional claims': {
            'conditions': ["CLAIM_TYPE = 'INSTITUTIONAL'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'institutional': {
            'conditions': ["CLAIM_TYPE = 'INSTITUTIONAL'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },
        'pharmacy claims': {
            'conditions': ["CLAIM_TYPE = 'PHARMACY'"],
            'tables': ['claims'], 'columns': ['CLAIM_TYPE'],
        },

        # ── Referral Urgency Concepts ──
        'elective': {
            'conditions': ["URGENCY = 'ELECTIVE'"],
            'tables': ['referrals'], 'columns': ['URGENCY'],
        },
        'stat': {
            'conditions': ["URGENCY = 'STAT'"],
            'tables': ['referrals'], 'columns': ['URGENCY'],
        },

        # ── Referral/General Approval Status ──
        'approved referral': {
            'conditions': ["STATUS = 'APPROVED'"],
            'tables': ['referrals'], 'columns': ['STATUS'],
        },
        'approved referrals': {
            'conditions': ["STATUS = 'APPROVED'"],
            'tables': ['referrals'], 'columns': ['STATUS'],
        },
        'completed referral': {
            'conditions': ["STATUS = 'COMPLETED'"],
            'tables': ['referrals'], 'columns': ['STATUS'],
        },
        'completed referrals': {
            'conditions': ["STATUS = 'COMPLETED'"],
            'tables': ['referrals'], 'columns': ['STATUS'],
        },

        # ── Prescription Status Concepts ──
        'cancelled': {
            'conditions': ["STATUS = 'CANCELLED'"],
            'tables': ['prescriptions'], 'columns': ['STATUS'],
        },
        'filled': {
            'conditions': ["STATUS = 'FILLED'"],
            'tables': ['prescriptions'], 'columns': ['STATUS'],
        },
        'expired prescription': {
            'conditions': ["STATUS = 'EXPIRED'"],
            'tables': ['prescriptions'], 'columns': ['STATUS'],
        },

        # ── Member PCP Concepts ──
        'without a pcp': {
            'conditions': ["(PCP_NPI IS NULL OR PCP_NPI = '')"],
            'tables': ['members'], 'columns': ['PCP_NPI'],
        },
        'no pcp': {
            'conditions': ["(PCP_NPI IS NULL OR PCP_NPI = '')"],
            'tables': ['members'], 'columns': ['PCP_NPI'],
        },

        # ── Severity Concepts ──
        'critical': {
            'conditions': ["SEVERITY = 'CRITICAL'"],
            'tables': ['diagnoses'], 'columns': ['SEVERITY'],
        },
        'severe': {
            'conditions': ["SEVERITY = 'SEVERE'"],
            'tables': ['diagnoses'], 'columns': ['SEVERITY'],
        },
    }

    # Typo map for domain concepts — common healthcare misspellings
    CONCEPT_TYPOS = {
        'meidcaid': 'medicaid', 'medicadi': 'medicaid', 'medcaid': 'medicaid',
        'medicar': 'medicare', 'meidcare': 'medicare',
        'diebetic': 'diabetic', 'diabtes': 'diabetes', 'dieabetes': 'diabetes',
        'newbron': 'newborn', 'newborns': 'newborn',
        'fostercare': 'foster care', 'fostr': 'foster',
        'pedeatric': 'pediatric', 'pediatirc': 'pediatric', 'peditric': 'pediatric',
        'hypertnesion': 'hypertension', 'hypertesnion': 'hypertension',
        'geriatircs': 'geriatric', 'eldery': 'elderly', 'senoir': 'senior',
        'adolecent': 'adolescent', 'adolescnet': 'adolescent',
        'depresion': 'depression', 'anxeity': 'anxiety',
        'obseity': 'obesity', 'obeese': 'obese',
        'pollypharmacy': 'polypharmacy',
        'tlehealth': 'telehealth', 'telemdicine': 'telehealth',
        'readmision': 'readmission', 'readmisions': 'readmission',
        'commerical': 'commercial', 'commercail': 'commercial',
    }

    # Time filters
    TIME_FILTERS = {
        'last 30 days': "SERVICE_DATE >= date('now', '-30 days')",
        'last month': "SERVICE_DATE >= date('now', '-1 month')",
        'last quarter': "SERVICE_DATE >= date('now', '-3 months')",
        'last 3 months': "SERVICE_DATE >= date('now', '-3 months')",
        'this year': "SERVICE_DATE >= date('now', 'start of year')",
        'last year': "SERVICE_DATE BETWEEN date('now', '-1 year', 'start of year') AND date('now', 'start of year', '-1 day')",
        'last week': "SERVICE_DATE >= date('now', '-7 days')",
        'last 7 days': "SERVICE_DATE >= date('now', '-7 days')",
        'last 6 months': "SERVICE_DATE >= date('now', '-6 months')",
        'last 90 days': "SERVICE_DATE >= date('now', '-90 days')",
    }

    @classmethod
    def extract(cls, question: str, resolved_cols: List[Dict], schema: SchemaRegistry) -> Tuple[List[str], str]:
        """Extract WHERE conditions and HAVING clause from the question.

        Returns:
            (conditions, having_clause) where having_clause is a string like "HAVING COUNT(*) > 5"
        """
        q = question.lower()
        conditions = []
        having_clause = ""

        # 0. Domain Concept Resolution — maps healthcare terms to WHERE clauses
        # MUST run before status/value filters to prevent duplicate conditions
        q_concept = q
        for typo, fix in cls.CONCEPT_TYPOS.items():
            q_concept = re.sub(r'\b' + re.escape(typo) + r'\b', fix, q_concept)

        concept_keys_sorted = sorted(cls.DOMAIN_CONCEPTS.keys(), key=len, reverse=True)
        matched_concepts = set()
        domain_tables_needed = []
        for concept in concept_keys_sorted:
            # BUG FIX: Use word-boundary matching for ALL concepts to avoid substring false positives
            # e.g., "uti" should not match inside "utilization"
            # Also "er visit" should not match inside "per visit" (the "er" in "per")
            if not re.search(r'\b' + re.escape(concept) + r'\b', q_concept):
                continue
            if concept not in matched_concepts:
                info = cls.DOMAIN_CONCEPTS[concept]
                if any(concept in mc and concept != mc for mc in matched_concepts):
                    continue
                for cond in info['conditions']:
                    if cond not in conditions:
                        conditions.append(cond)
                for tbl in info.get('tables', []):
                    if tbl not in domain_tables_needed:
                        domain_tables_needed.append(tbl)
                matched_concepts.add(concept)

        cls._domain_tables_needed = domain_tables_needed
        cls._matched_domain_concepts = matched_concepts
        domain_handled_cols = set()
        for concept in matched_concepts:
            for col in cls.DOMAIN_CONCEPTS[concept].get('columns', []):
                domain_handled_cols.add(col)
        cls._domain_handled_cols = domain_handled_cols

        # BUG 7 FIX: Check for "both A and B" pattern FIRST, before adding individual status filters
        # This prevents adding impossible WHERE conditions like "VISIT_TYPE = X AND VISIT_TYPE = Y"
        both_pattern = re.search(
            r'\b(\w+)\s+with\s+both\s+(\w+)\s+and\s+(\w+)',
            q, re.IGNORECASE
        )
        is_both_pattern = False
        if both_pattern:
            entity = both_pattern.group(1)  # members
            value1 = both_pattern.group(2).upper()  # EMERGENCY
            value2 = both_pattern.group(3).upper()  # INPATIENT
            # Mark that we have a both pattern so we skip individual status filters
            is_both_pattern = True

        # 1. Status/value filters (but NOT when used as column qualifiers like "paid amount")
        # BUG 1 FIX: Check if column actually exists in resolved tables before adding filter
        # BUG 9 FIX: Don't add WHERE filter if this status keyword is used in a "per X" or "by X" GROUP BY context
        for keyword, (col, op, val) in cls.STATUS_VALUES.items():
            # Skip if domain concepts already handle this column
            if col in domain_handled_cols:
                continue
            # BUG 7 FIX: Skip individual status filters if this is a "both A and B" pattern
            # The "both" pattern will add its own filter
            if is_both_pattern and col == 'VISIT_TYPE':
                continue
            if re.search(r'\b' + re.escape(keyword) + r'\b', q):
                # BUG 9 FIX: Check if this keyword is part of a GROUP BY dimension, not a filter
                # e.g., "average per visit type" should GROUP BY, not filter
                is_group_dimension = False
                if col == 'VISIT_TYPE' and any(phrase in q for phrase in ['per visit', 'by visit', 'per emergency', 'by emergency']):
                    # More specific check: is this "per visit type" or "by visit type"?
                    if re.search(r'\b(?:per|by)\s+(?:visit|emergency|outpatient|inpatient)', q):
                        is_group_dimension = True

                # Check if this is a column qualifier context
                is_qualifier = False
                for ctx_word, ctx_phrases in cls.COLUMN_QUALIFIER_CONTEXTS.items():
                    if keyword == ctx_word and any(p in q for p in ctx_phrases):
                        is_qualifier = True
                        break
                if not is_qualifier and not is_group_dimension:
                    # BUG 1 FIX: Find the right status column for the resolved tables
                    resolved_tables = set(rc['table'] for rc in resolved_cols)

                    # Status-like columns map differently across tables:
                    # claims→CLAIM_STATUS, referrals→STATUS, prescriptions→STATUS,
                    # encounters→ENCOUNTER_STATUS, providers→STATUS
                    # Apply this mapping for ANY keyword that targets a status column
                    STATUS_COL_MAP = {
                        'prescriptions': 'STATUS',
                        'referrals': 'STATUS',
                        'encounters': 'ENCOUNTER_STATUS',
                        'providers': 'STATUS',
                        'claims': 'CLAIM_STATUS',
                    }
                    STATUS_LIKE_COLS = {'CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'}

                    if col in STATUS_LIKE_COLS:
                        # Pick the right status column for the target table
                        # Priority: use the table explicitly mentioned in the question
                        # e.g., 'pending referrals' should use referrals.STATUS not claims.CLAIM_STATUS
                        status_col = None
                        # Build candidate tables: from resolved_cols + tables mentioned in question
                        candidate_tables = set(resolved_tables)
                        # Also check table names mentioned in question text
                        for tbl_name in STATUS_COL_MAP:
                            if tbl_name in q or tbl_name.rstrip('s') in q:
                                candidate_tables.add(tbl_name)
                        # First: prefer tables whose name (or singular) appears in the question
                        mentioned_tables = []
                        other_tables = []
                        for table_candidate in candidate_tables:
                            if table_candidate in q or table_candidate.rstrip('s') in q:
                                mentioned_tables.append(table_candidate)
                            else:
                                other_tables.append(table_candidate)
                        # Check mentioned tables first, then fall back to others
                        for table_candidate in mentioned_tables + other_tables:
                            if table_candidate in STATUS_COL_MAP:
                                status_col = STATUS_COL_MAP[table_candidate]
                                break
                        if status_col and status_col in schema.column_to_tables:
                            conditions.append(f"{status_col} {op} '{val}'")
                        elif col in schema.column_to_tables:
                            # Fall back to original if no table-specific mapping
                            conditions.append(f"{col} {op} '{val}'")
                    else:
                        # For non-status columns, use the original col
                        col_tables = schema.column_to_tables.get(col, [])
                        if col_tables:
                            conditions.append(f"{col} {op} '{val}'")

        # 1b. Detect "entity with more than N other_entity" → HAVING COUNT(*) > N
        # e.g., "members with more than 5 claims" → GROUP BY MEMBER_ID HAVING COUNT(*) > 5
        # Also handle "count where one member visited X more than N times" → nested subquery
        ENTITY_TABLES = {'members', 'member', 'patients', 'patient', 'providers', 'provider',
                         'doctors', 'doctor', 'claims', 'claim', 'encounters', 'encounter',
                         'diagnoses', 'diagnosis', 'prescriptions', 'prescription',
                         'referrals', 'referral', 'medications', 'medication', 'facilities', 'facility'}

        # BUG 1 FIX: Detect "count where one person visited X more than N times" pattern
        # This requires a nested subquery with outer COUNT
        nested_count_match = re.search(
            r'\b(?:count|how many)\b.*?\b(?:one |a )?(?:person|member|patient)\b.*?(?:visited|had|received)\s+(\w+)\s+(?:more than|over|at least|greater than|>)\s*(\d+)\s+(?:times?|visits?|admissions?)',
            q, re.IGNORECASE
        )
        if nested_count_match:
            visit_type = nested_count_match.group(1).upper()
            threshold = nested_count_match.group(2)
            # Flag this as needing nested subquery SQL generation (handled in _build_select)
            cls._needs_nested_subquery = {
                'visit_type': visit_type,
                'threshold': threshold
            }

        # BUG 7 FIX: Already handled above where we detected the both_pattern
        # Now add the filter and HAVING clause
        if is_both_pattern and both_pattern:
            value1 = both_pattern.group(2).upper()  # EMERGENCY
            value2 = both_pattern.group(3).upper()  # INPATIENT
            having_clause = "HAVING COUNT(DISTINCT VISIT_TYPE) = 2"
            conditions.append(f"VISIT_TYPE IN ('{value1}', '{value2}')")

        having_entity_match = re.search(
            r'\b(\w+)\s+(?:with|having|who have|that have|who had|who treated|who saw|who visited|who received|who seen|that had)\s+(?:more than|over|at least|greater than|>)\s*(\d+)\s+(\w+)',
            q
        )
        if having_entity_match:
            entity1 = having_entity_match.group(1)
            threshold = having_entity_match.group(2)
            entity2 = having_entity_match.group(3)
            # If entity2 looks like a table name, this is a HAVING query
            if entity2.rstrip('s') in {e.rstrip('s') for e in ENTITY_TABLES}:
                having_clause = f"HAVING COUNT(*) > {threshold}"

        # 2. Comparison filters (check if should be HAVING clause)
        # Detect count/aggregation keywords
        count_keywords = ['how many', 'count', 'number of', 'total number', '#', 'volume']
        has_count_context = any(kw in q for kw in count_keywords)

        for pattern, op, _ in cls.COMPARISON_RE:
            m = re.search(pattern, q)
            if m:
                val_str = m.group(1).replace(',', '')

                # Skip if this looks like a year (2020-2029)
                try:
                    val_num = float(val_str)
                    if 2020 <= val_num <= 2029:
                        continue
                except ValueError:
                    pass

                # Skip if HAVING clause already set by entity pattern (step 1b)
                if having_clause:
                    continue

                # BUG 2, 3, 4 FIX: Check if this comparison is about visit counts, not amounts
                # "more than 10 times", "more than 5 visits", "more than 3 admissions"
                # Should generate HAVING COUNT(*) > N, not BILLED_AMOUNT filter
                is_count_context = any(w in q for w in ['times', 'visits', 'admissions', 'encounters'])
                is_near_count = False
                if is_count_context:
                    # Check if this number is near visit/time keywords
                    pattern_pos = m.start()
                    # Look back/forward in question around this match
                    search_window = q[max(0, pattern_pos-100):min(len(q), pattern_pos+100)]
                    if any(w in search_window for w in ['times', 'visits', 'admissions', 'encounters']):
                        is_near_count = True

                # Also check: "X who treated more than N patients/members" pattern
                # The entity word after the number signals this is a count, not an amount
                if not is_near_count and m:
                    after_number = q[m.end():m.end()+30].strip().split()[0] if q[m.end():m.end()+30].strip() else ''
                    ENTITY_COUNT_WORDS = {'patients', 'members', 'people', 'providers', 'doctors',
                                          'claims', 'prescriptions', 'referrals', 'diagnoses', 'encounters'}
                    if after_number.rstrip('s').rstrip('e') in {w.rstrip('s').rstrip('e') for w in ENTITY_COUNT_WORDS}:
                        is_near_count = True

                # If question has count keywords, this comparison is a HAVING condition
                if has_count_context or is_near_count:
                    having_clause = f"HAVING COUNT(*) {op} {val_str}"
                    continue

                # AGE DETECTION: "members over 65", "patients under 18", "female members over 65"
                # When question is about members/patients and the number is a plausible age (1-120),
                # use DATE_OF_BIRTH age calculation instead of a numeric column filter
                is_age_context = False
                try:
                    age_val = float(val_str)
                    # Use word boundary matching to prevent "age" matching inside "average", "mileage", etc.
                    # Use prefix matching for member/patient/person to catch plurals (members, patients, persons)
                    _age_context_patterns = [r'\bmember', r'\bpatient', r'\bperson', r'\bpeople\b', r'\bfemale\b', r'\bmale\b', r'\bgender\b', r'\bage\b', r'\bold\b', r'\byear']
                    _has_age_word = any(re.search(p, q) for p in _age_context_patterns)
                    if 1 <= age_val <= 120 and _has_age_word:
                        # Make sure this isn't a dollar amount, claim count, or length-of-stay/days context
                        if not any(w in q for w in ['$', 'billed', 'paid', 'cost', 'amount', 'claim', 'score', 'length of stay', 'los', 'days', 'duration']):
                            is_age_context = True
                            age_int = int(age_val)
                            if op == '>':
                                conditions.append(f"DATE_OF_BIRTH <= date('now', '-{age_int} years')")
                            elif op == '<':
                                conditions.append(f"DATE_OF_BIRTH >= date('now', '-{age_int} years')")
                            elif op == '>=':
                                conditions.append(f"DATE_OF_BIRTH <= date('now', '-{age_int} years')")
                            elif op == '<=':
                                conditions.append(f"DATE_OF_BIRTH >= date('now', '-{age_int} years')")
                            elif op == 'BETWEEN':
                                val2 = m.group(2).replace(',', '') if m.lastindex >= 2 else str(age_int + 10)
                                conditions.append(f"DATE_OF_BIRTH BETWEEN date('now', '-{val2} years') AND date('now', '-{age_int} years')")
                            continue
                except (ValueError, IndexError):
                    pass

                if is_age_context:
                    continue

                # Find the best numeric column for this comparison
                num_col = None
                # First: infer from context keywords in the question
                # BUG 2, 3, 4 FIX: Exclude 'claim' from the BILLED_AMOUNT inference if we're talking about visit counts
                if not is_count_context and ('$' in q or 'claim' in q or 'billed' in q):
                    num_col = 'BILLED_AMOUNT'
                elif 'paid' in q and not is_count_context:
                    num_col = 'PAID_AMOUNT'
                elif 'cost' in q:
                    num_col = 'COST'
                elif 'score' in q or 'risk' in q:
                    num_col = 'RISK_SCORE'
                elif 'stay' in q or 'los' in q:
                    num_col = 'LENGTH_OF_STAY'
                else:
                    # Fall back to resolved columns
                    for rc in resolved_cols:
                        cn = rc['column'].lower()
                        if any(k in cn for k in ['amount', 'cost', 'paid', 'billed', 'score', 'count', 'size', 'stay']):
                            num_col = rc['column']
                            break

                if not num_col:
                    # Try to infer from context — be specific
                    if 'paid amount' in q or 'paid_amount' in q:
                        num_col = 'PAID_AMOUNT'
                    elif 'billed amount' in q or 'billed_amount' in q or ('billed' in q and not is_count_context):
                        num_col = 'BILLED_AMOUNT'
                    elif 'allowed' in q:
                        num_col = 'ALLOWED_AMOUNT'
                    elif 'score' in q or 'risk' in q:
                        num_col = 'RISK_SCORE'
                    elif 'stay' in q or 'los' in q:
                        num_col = 'LENGTH_OF_STAY'
                    elif '$' in q or ('amount' in q and not is_count_context) or 'cost' in q or ('claim' in q and not is_count_context):
                        # Dollar amounts almost always refer to billing amounts
                        num_col = 'BILLED_AMOUNT'

                if num_col:
                    if op == 'BETWEEN':
                        val1 = m.group(1).replace(',', '')
                        val2 = m.group(2).replace(',', '')
                        conditions.append(f"CAST({num_col} AS REAL) BETWEEN {val1} AND {val2}")
                    else:
                        conditions.append(f"CAST({num_col} AS REAL) {op} {val_str}")

        # 3. Year filter (Bug 2: "over 2024" should be date filter, not comparison)
        year_match = re.search(r'(?:in|over|during|for|from|year)?\s*(20[12]\d)\b', q)
        if year_match:
            year = year_match.group(1)
            date_col = 'SERVICE_DATE'
            for rc in resolved_cols:
                if 'DATE' in rc['column'].upper():
                    date_col = rc['column']
                    break
            conditions.append(f"{date_col} LIKE '{year}%'")

        # 4. Time filters
        # Table-specific default date columns
        TABLE_DATE_COL = {
            'claims': 'SERVICE_DATE', 'encounters': 'SERVICE_DATE',
            'diagnoses': 'DIAGNOSIS_DATE', 'prescriptions': 'PRESCRIPTION_DATE',
            'referrals': 'REFERRAL_DATE', 'members': 'ENROLLMENT_DATE',
            'providers': 'HIRE_DATE',
        }
        for phrase, sql_cond in cls.TIME_FILTERS.items():
            if phrase in q:
                # Try to find the right date column for the context
                date_col = 'SERVICE_DATE'
                # First check resolved columns
                for rc in resolved_cols:
                    if 'DATE' in rc['column'].upper():
                        date_col = rc['column']
                        break
                else:
                    # Fall back to table-specific defaults
                    for rc in resolved_cols:
                        tbl_date = TABLE_DATE_COL.get(rc['table'])
                        if tbl_date:
                            date_col = tbl_date
                            break
                conditions.append(sql_cond.replace('SERVICE_DATE', date_col))
                break

        # 5. LIKE pattern: "containing X", "with name X"
        # Exclude "look like" / "looks like" — those are natural language, not LIKE filters
        like_match = re.search(r'(?:containing|contains|named|called|with name)\s+["\']?(\w+)["\']?', q)
        if not like_match:
            # Only match standalone "like" when preceded by a column-like context, not "look(s) like"
            like_match2 = re.search(r'(?<!look\s)(?<!looks\s)\blike\s+["\']?(\w+)["\']?', q)
            if like_match2 and like_match2.group(1).lower() not in {'by', 'the', 'a', 'an', 'this', 'that', 'it'}:
                like_match = like_match2
        if like_match:
            val = like_match.group(1)
            # Find a text column to search
            for rc in resolved_cols:
                ci = schema.get_column_info(rc['table'], rc['column'])
                dt = (ci or {}).get('data_type', 'text').lower()
                if dt in ('text', 'string', 'category') and 'ID' not in rc['column'].upper():
                    conditions.append(f"{rc['column']} LIKE '%{val.upper()}%'")
                    break

        # 6. Specific value mentions: "in region EAST", "for plan HMO"
        region_match = re.search(r'(?:in |for |from )(?:region |kp_region )?(\w+)', q, re.IGNORECASE)
        if region_match and any(r in q for r in ['region', 'ncal', 'scal', 'norcal', 'socal']):
            region_val = region_match.group(1).upper()
            if region_val in ['NCAL', 'SCAL', 'NORCAL', 'SOCAL', 'EAST', 'WEST', 'CENTRAL']:
                conditions.append(f"KP_REGION = '{region_val}'")

        # BUG 2 FIX: Detect plan type values even without "plan" keyword
        # Skip if domain concepts already added PLAN_TYPE filters
        if 'PLAN_TYPE' not in domain_handled_cols:
            plan_match = re.search(r'(?:plan|type|for|in|enrolled)(?:\s+\w+)*\s+(hmo|ppo|epo|pos|medicare|medicaid)\b', q, re.IGNORECASE)
            if not plan_match:
                plan_match = re.search(r'\b(hmo|ppo|epo|pos|medicare|medicaid)\b', q, re.IGNORECASE)
            if plan_match:
                plan_val = plan_match.group(1) if plan_match.lastindex >= 1 else plan_match.group(0)
                conditions.append(f"PLAN_TYPE = '{plan_val.upper()}'")

        # BUG 2 FIX: Detect visit type values (telehealth, urgent_care, home_health)
        # Note: inpatient/outpatient/emergency/urgent already handled by KEYWORD_FILTERS above
        # Skip if domain concepts already added VISIT_TYPE filters
        visit_type_match = None
        if 'VISIT_TYPE' not in domain_handled_cols:
            visit_type_match = re.search(r'\b(telehealth|home\s*health|urgent\s*care|routine)\b', q, re.IGNORECASE)
        if visit_type_match:
            visit_val = visit_type_match.group(1).upper().replace(' ', '_')
            # Only add if not already covered by KEYWORD_FILTERS
            if not any("VISIT_TYPE" in c for c in conditions):
                conditions.append(f"VISIT_TYPE = '{visit_val}'")

        # BUG 2 FIX: Detect specialty values (cardiology, dermatology, etc.)
        specialty_match = re.search(r'\b(cardiology|dermatology|pediatrics|psychiatry|orthopedics|neurology|oncology|gastroenterology|pulmonology|rheumatology)\b', q, re.IGNORECASE)
        if specialty_match:
            conditions.append(f"SPECIALTY LIKE '%{specialty_match.group(1).upper()}%'")

        facility_match = re.search(r'(?:at |in |from )facility\s+(\w+)', q, re.IGNORECASE)
        if facility_match:
            conditions.append(f"FACILITY LIKE '%{facility_match.group(1).upper()}%'")

        return (conditions, having_clause)


# =============================================================================
# AGGREGATE DETECTOR — detects aggregation intent
# =============================================================================

class AggregateDetector:
    """Detects aggregation functions and GROUP BY columns from natural language."""

    @staticmethod
    def _singularize(word: str) -> str:
        """Convert plural to singular: specialties→specialty, facilities→facility, providers→provider."""
        w = word.lower()
        if w.endswith('ies') and len(w) > 4:
            return w[:-3] + 'y'  # specialties → specialty
        elif w.endswith('ses') or w.endswith('xes') or w.endswith('zes'):
            return w[:-2]  # diagnoses → diagnos... actually this is wrong for diagnoses
        elif w.endswith('es') and not w.endswith('ses'):
            return w[:-1]  # Not standard but handles "diagnoses" edge
        elif w.endswith('s') and not w.endswith('ss'):
            return w[:-1]  # providers → provider
        return w

    AGG_PATTERNS = {
        'COUNT': [r'\bhow many\b', r'\bcount\b', r'\btotal number\b', r'\bnumber of\b', r'\b# of\b', r'\bvolume\b',
                  r'\bmost\s+\w+ed\b', r'\bmost common\b', r'\bmost frequent\b', r'\brate\b', r'\bratio\b',
                  r'\bdistribut\w*\b', r'\bbreakdown\b', r'\bspread\b', r'\bsplit\b',
                  # Bug 9: "total prescriptions/claims/encounters" = COUNT (entity count, not SUM)
                  r'\btotal\s+(?:claims|encounters|members|providers|diagnoses|prescriptions|referrals|patients|doctors|visits)',
                  r'\bfrequen'],
        # Bug 9: "total prescriptions" = COUNT, "total billed amount" = SUM
        # Only match "total" when NOT followed by a table name (entity)
        'SUM': [r'\btotal\b(?! number)(?!\s+(?:claims|encounters|members|providers|diagnoses|prescriptions|referrals|patients|doctors|visits))',
                r'\bsum\b', r'\bcombined\b', r'\baggregate\b'],
        'AVG': [r'\baverage\b', r'\bavg\b', r'\bmean\b', r'\btypical\b'],
        'MAX': [r'\bmax\b', r'\bmaximum\b', r'\bhighest\b', r'\blargest\b', r'\bbiggest\b', r'\bmost expensive\b'],
        'MIN': [r'\bmin\b', r'\bminimum\b', r'\blowest\b', r'\bsmallest\b', r'\bcheapest\b'],
    }

    GROUP_PATTERNS = [
        r'\bby\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having|limit|with)|$)',  # removed 'and' as stop word to allow "by gender and region"
        r'\bper\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having|limit|with)|$)',  # BUG 4: capture multi-word terms
        r'\b(?:for|in) each\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|$)',
        r'\beach\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|$)',
        r'\bgrouped? by\s+([\w\s]+)',
        r'\bbroken? down by\s+([\w\s]+)',
        r'\bacross\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|$)',
    ]

    TOP_N_RE = re.compile(r'\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\bbottom\s+(\d+)\b')
    # Bug 8: "top 5 providers by total billed" → entity = providers
    # BUG 4 FIX: Also match "top departments" without number
    TOP_N_ENTITY_RE = re.compile(r'\b(?:top|first|bottom)(?:\s+\d+)?\s+(\w+)')
    LIMIT_RE = re.compile(r'\blimit\s+(\d+)\b')

    @classmethod
    def detect(cls, question: str) -> Dict[str, Any]:
        """
        Detect aggregation details from question.
        Returns: {'agg_func': 'AVG'|None, 'group_by_terms': [...], 'top_n': int|None, 'order': 'DESC'|'ASC'}
        """
        q = question.lower()
        result = {'agg_func': None, 'group_by_terms': [], 'top_n': None, 'order': 'DESC'}

        # Normalize common typos/misspellings in the query for pattern matching
        # BUG 6 FIX: Add common typos like 'wich' and 'facilty'
        # IMPORTANT: Use word boundaries to avoid replacing within words (e.g., 'which' -> 'whichh')
        _typo_map = {
            'higest': 'highest', 'hightest': 'highest', 'hihgest': 'highest',
            'lowets': 'lowest', 'lowset': 'lowest',
            'largset': 'largest', 'largets': 'largest',
            'smalest': 'smallest', 'smallets': 'smallest',
            'bigest': 'biggest', 'biggset': 'biggest',
            'fewset': 'fewest', 'mostt': 'most',
            'avergae': 'average', 'averge': 'average',
            'totla': 'total', 'toal': 'total',
            'facilty': 'facility', 'faciltiy': 'facility',  # BUG 6
            'amout': 'amount', 'amoutn': 'amount',  # BUG 6
            'reigon': 'region', 'recieved': 'received',  # BUG 6
        }
        q_norm = q
        for typo, fix in _typo_map.items():
            q_norm = q_norm.replace(typo, fix)
        # Word-boundary typo fixes (to avoid substring collisions like 'wich' inside 'which')
        for typo, fix in [('wich', 'which'), ('wihch', 'which'), ('whcih', 'which')]:
            q_norm = re.sub(r'\b' + typo + r'\b', fix, q_norm)
        # Continue with normalized q_norm below
        for typo, fix in {}.items():  # no-op (already applied above)
            q_norm = q_norm.replace(typo, fix)

        # Detect "which X has most Y" pattern FIRST (Bug 3) — before general agg patterns
        # BUG 5 FIX: Also capture multi-word terms like "plan type"
        # Also match "X with highest/most Y" pattern (e.g., "members with highest billed amount")
        which_match = re.search(r'\bwhich\s+([\w\s]+?)\s+(?:has|have|is|are|had|gets?|handles?|receives?|generates?|sees?|shows?)\s+(?:the\s+)?(?:most|highest|lowest|least|fewest|longest|shortest|largest|smallest|biggest)', q_norm)
        if not which_match:
            which_match = re.search(r'\b([\w\s]+?)\s+with\s+(?:the\s+)?(?:most|highest|lowest|least|fewest|longest|shortest|largest|smallest|biggest)\b', q_norm)
        # "most prescribed medication" → group by medication
        if not which_match:
            most_adj_match = re.search(r'\bmost\s+\w+(?:ed|ing)\s+(\w+)', q_norm)
            if most_adj_match:
                which_match = most_adj_match
        if which_match:
            group_term = which_match.group(1)
            result['group_by_terms'] = [group_term]
            # "which member" (singular) → top 1; "which members" (plural) → top 10
            result['top_n'] = 1 if not group_term.endswith('s') else 10
            result['order'] = 'DESC' if any(w in q_norm for w in ['most', 'highest', 'longest', 'largest', 'biggest']) else 'ASC'
            # Bug 7: "which X has highest AVERAGE cost" → AVG, not COUNT
            # Detect metric words that imply SUM rather than COUNT
            AMOUNT_WORDS = {'amount', 'cost', 'billed', 'paid', 'allowed', 'revenue',
                            'spend', 'spending', 'charge', 'expense', 'payment',
                            'copay', 'coinsurance', 'deductible', 'salary', 'price'}
            has_amount_word = any(w in q_norm for w in AMOUNT_WORDS)
            if 'average' in q_norm or 'avg' in q_norm:
                result['agg_func'] = 'AVG'
            elif 'total' in q_norm or 'sum' in q_norm:
                result['agg_func'] = 'SUM'
            elif has_amount_word:
                # "which X has highest billed amount" → SUM, not COUNT
                result['agg_func'] = 'SUM'
            else:
                result['agg_func'] = 'COUNT'

        # Detect aggregation function (unless already detected by which_match)
        if not result['agg_func']:
            # BUG 5 FIX: "top X" and "most common" without explicit metric should COUNT
            if 'top' in q_norm or 'bottom' in q_norm or 'first' in q_norm:
                # Check if there's already a metric word (amount, cost, billed, etc.)
                METRIC_WORDS = {'amount', 'cost', 'billed', 'paid', 'allowed', 'revenue', 'spend'}
                AVG_WORDS = {'average', 'avg'}
                SUM_WORDS = {'total', 'sum'}
                has_metric = any(m in q_norm for m in METRIC_WORDS)
                has_avg = any(m in q_norm for m in AVG_WORDS)
                has_sum = any(m in q_norm for m in SUM_WORDS)
                if has_avg:
                    result['agg_func'] = 'AVG'
                elif has_metric or has_sum:
                    result['agg_func'] = 'SUM'
                else:
                    result['agg_func'] = 'COUNT'

            if not result['agg_func']:
                for func, patterns in cls.AGG_PATTERNS.items():
                    for pat in patterns:
                        if re.search(pat, q_norm):
                            result['agg_func'] = func
                            break
                    if result['agg_func']:
                        break

        # Detect GROUP BY terms (unless already set by which_match)
        if not result['group_by_terms']:
            # Detect GROUP BY terms
            for pat in cls.GROUP_PATTERNS:
                m = re.search(pat, q_norm)
                if m:
                    raw = m.group(1).strip()
                    # Split on "and" to capture multi-dimension grouping: "gender and region"
                    # Then split each part on spaces for multi-word terms
                    parts = re.split(r'\band\b', raw)
                    terms = []
                    for part in parts:
                        words = part.strip().split()
                        noise = {'the', 'a', 'an', 'each', 'every', 'all', 'their', 'its', 'or', 'on', 'of', 'front', 'side', 'basis'}
                        words = [w for w in words if w.lower() not in noise and len(w) > 1]
                        terms.extend(words)
                    result['group_by_terms'] = terms
                    break

            # "X distribution/breakdown" pattern — group by X
            # e.g., "claim status distribution" → group by claim, status
            if not result['group_by_terms'] and result['agg_func'] == 'COUNT':
                dist_m = re.search(
                    r'([\w\s]+?)\s+(?:distribution|breakdown|spread|split)\b', q
                )
                if dist_m:
                    raw = dist_m.group(1).strip().split()
                    noise = {'the', 'a', 'an', 'show', 'get', 'give', 'me', 'display', 'list', 'what', 'is'}
                    terms = [t for t in raw if t.lower() not in noise and len(t) > 1]
                    if terms:
                        result['group_by_terms'] = terms

            # BUG 5 FIX: "most common ICD10 codes" → group by ICD10_CODE
            # "top denial reasons" → group by DENIAL_REASON
            if not result['group_by_terms'] and result['agg_func'] == 'COUNT':
                # Check for "most common X" or "top X reasons/codes/etc"
                most_common_m = re.search(r'(?:most|top)\s+(?:common|frequent)?\s+(\w+)\s+(?:reasons?|codes?|categories?|diagnoses?)', q, re.IGNORECASE)
                if most_common_m:
                    # For compound nouns like "denial reasons", also capture the second word
                    term = most_common_m.group(1)
                    second_word = most_common_m.group(0).split()[-1]  # e.g., "reasons"
                    # Map common suffixes to the actual column they represent
                    SUFFIX_MAP = {
                        'reasons': 'REASON', 'reason': 'REASON',
                        'codes': 'CODE', 'code': 'CODE',
                        'categories': 'CATEGORY', 'category': 'CATEGORY',
                    }
                    if second_word.lower() in SUFFIX_MAP:
                        # e.g., "denial" + "reasons" → "denial_reason"
                        result['group_by_terms'] = [f"{term}_{SUFFIX_MAP[second_word.lower()]}"]
                    else:
                        result['group_by_terms'] = [term]
                    if not result['top_n']:
                        result['top_n'] = 10

        # Detect TOP N
        m = cls.TOP_N_RE.search(q)
        if m:
            n = m.group(1) or m.group(2) or m.group(3)
            result['top_n'] = int(n)
            if m.group(3):  # "bottom N"
                result['order'] = 'ASC'

            # Bug 8: "top 5 providers by total billed" → entity = providers
            # The entity after "top N" is what we GROUP BY, not the "by X" part
            # The entity OVERRIDES any "by X" group terms since those describe the metric
            entity_m = cls.TOP_N_ENTITY_RE.search(q)
            if entity_m:
                entity = cls._singularize(entity_m.group(1))
                # Only override if entity is a real entity (not a metric word)
                METRIC_WORDS = {'total', 'average', 'count', 'sum', 'max', 'min', 'amount', 'cost', 'rate'}
                if entity.lower() not in METRIC_WORDS:
                    result['group_by_terms'] = [entity]
        else:
            # BUG 4 FIX: Also handle "top X" without a number, e.g., "top departments"
            entity_m = cls.TOP_N_ENTITY_RE.search(q)
            if entity_m and 'top' in q:
                entity = cls._singularize(entity_m.group(1))
                # BUG 5 FIX: Check for context words like "reasons", "codes", etc.
                # "top referral reasons" → group by "referral" + "reason"
                context_words = {'reasons': 'REASON', 'codes': 'CODE', 'categories': 'CATEGORY', 'diagnoses': 'DIAGNOSIS'}
                for suffix, col_suffix in context_words.items():
                    if suffix in q:
                        entity = f"{entity}_{col_suffix}"
                        break
                METRIC_WORDS = {'total', 'average', 'count', 'sum', 'max', 'min', 'amount', 'cost', 'rate'}
                if entity.lower() not in METRIC_WORDS and not entity.startswith('total'):
                    result['group_by_terms'] = [entity]
                    result['top_n'] = 10  # Default to top 10 if no number specified

        m2 = cls.LIMIT_RE.search(q)
        if m2 and not result['top_n']:
            result['top_n'] = int(m2.group(1))

        # Detect ordering
        if 'lowest' in q or 'smallest' in q or 'least' in q or 'cheapest' in q or 'fewest' in q:
            result['order'] = 'ASC'

        # BUG 8 FIX: "chronic vs non-chronic diagnoses" should GROUP BY IS_CHRONIC with COUNT
        if ('chronic' in q and 'vs' in q) or ('chronic' in q and 'versus' in q):
            if not result['agg_func']:
                result['agg_func'] = 'COUNT'
            if not result['group_by_terms']:
                result['group_by_terms'] = ['chronic']

        # If GROUP BY detected but no agg, infer COUNT when question contains
        # plural countable entities (e.g., "diagnoses by department" → COUNT)
        if result['group_by_terms'] and not result['agg_func']:
            COUNTABLE_PLURALS = {
                'claims', 'encounters', 'diagnoses', 'prescriptions', 'referrals',
                'members', 'patients', 'providers', 'visits', 'admissions',
            }
            if any(cp in q for cp in COUNTABLE_PLURALS):
                result['agg_func'] = 'COUNT'

        return result


# =============================================================================
# DYNAMIC SQL BUILDER — assembles SQL from resolved components
# =============================================================================

class DynamicSQLBuilder:
    """
    Assembles complete SQL statements from resolved columns, tables, joins,
    filters, and aggregations.
    """

    def __init__(self, schema: SchemaRegistry):
        self.schema = schema

    def build(self, question: str, resolved_cols: List[Dict], filters: List[str],
              agg_info: Dict, tables_needed: List[str], having_clause: str = "") -> str:
        """
        Build a complete SQL statement.

        Args:
            question: Original question
            resolved_cols: From ColumnResolver
            filters: From FilterExtractor
            agg_info: From AggregateDetector
            tables_needed: List of tables that need to be in the query
            having_clause: Optional HAVING clause string (Bug 1)

        Returns:
            Complete SQL string
        """
        if not tables_needed:
            tables_needed = ['claims']

        primary_table = tables_needed[0]

        # BUG 3 FIX: Validate resolved columns — only keep cols on tables we already need
        # Don't pull in new tables from loose synonym matches (causes table bloat)
        valid_resolved_cols = []
        # Track which original terms are already covered by existing tables
        covered_terms = set()
        for rc in resolved_cols:
            if rc['table'] in tables_needed:
                col_exists = rc['column'] in self.schema.get_columns(rc['table'])
                if col_exists:
                    valid_resolved_cols.append(rc)
                    if rc.get('original_term'):
                        covered_terms.add(rc['original_term'].lower())
        # Only add tables for exact matches if the term isn't already covered
        for rc in resolved_cols:
            if rc['table'] not in tables_needed and rc['match_type'] == 'exact':
                orig_term = (rc.get('original_term') or '').lower()
                if orig_term and orig_term in covered_terms:
                    continue  # Already have this concept via synonym on existing table
                col_exists = rc['column'] in self.schema.get_columns(rc['table'])
                if col_exists and rc['table'] not in tables_needed:
                    tables_needed.append(rc['table'])
                    valid_resolved_cols.append(rc)
                    if orig_term:
                        covered_terms.add(orig_term)

        resolved_cols = valid_resolved_cols

        # Determine SELECT columns, GROUP BY, ORDER BY
        agg_func = agg_info.get('agg_func')
        group_terms = agg_info.get('group_by_terms', [])
        top_n = agg_info.get('top_n')
        order = agg_info.get('order', 'DESC')

        # Resolve GROUP BY columns
        group_cols = self._resolve_group_columns(group_terms, resolved_cols, tables_needed)

        # If we have aggregation + top_n but no group columns, infer from question context
        if agg_func and top_n and not group_cols:
            group_cols = self._infer_group_column(question, tables_needed)

        # If we have "by X" in question and resolved it as agg but no group cols, infer
        q_lower = question.lower()
        if agg_func and not group_cols and ' by ' in q_lower:
            group_cols = self._infer_group_column(question, tables_needed)

        # Find numeric columns for aggregation
        num_col = self._find_best_numeric_col(question, resolved_cols, tables_needed)

        # Validate numeric column exists in available tables; if not, add the right table
        if num_col:
            num_col_in_tables = any(
                num_col in self.schema.get_columns(t) for t in tables_needed
            )
            if not num_col_in_tables:
                # Find which table has this column and add it
                for t_name in self.schema.tables.keys():
                    if num_col in self.schema.get_columns(t_name):
                        tables_needed.append(t_name)
                        break

        # Determine what to SELECT
        select_parts, needs_group_by = self._build_select(
            question, resolved_cols, agg_func, group_cols, num_col, tables_needed, filters
        )

        # Build FROM + JOINs (this sets self._current_aliases)
        from_clause = self._build_from(tables_needed)

        # Qualify columns if multi-table join
        if len(tables_needed) > 1:
            select_parts = [self._qualify_select_expr(sp, tables_needed) for sp in select_parts]
            filters = [self._qualify_filter(f, tables_needed) for f in filters]
            group_cols = [self._qualify_column(gc, tables_needed) for gc in group_cols]

        # Build WHERE
        where_clause = ""
        if filters:
            where_clause = " WHERE " + " AND ".join(filters)

        # Build GROUP BY
        group_clause = ""
        if needs_group_by and group_cols:
            group_clause = " GROUP BY " + ", ".join(group_cols)

        # Build HAVING clause (Bug 1)
        # HAVING requires GROUP BY — if no GROUP BY, skip HAVING
        having_sql = ""
        if having_clause and group_clause:
            having_sql = f" {having_clause}"

        # Build ORDER BY
        order_clause = ""
        if agg_func and needs_group_by:
            # Order by the aggregate
            agg_alias = select_parts[-1].split(' as ')[-1].strip() if ' as ' in select_parts[-1].lower() else None
            if agg_alias:
                order_clause = f" ORDER BY {agg_alias} {order}"
            else:
                order_clause = f" ORDER BY 2 {order}"
        elif not agg_func:
            # For lookups, order by date if available
            for rc in resolved_cols:
                if 'DATE' in rc['column'].upper() and rc['table'] in tables_needed:
                    order_clause = f" ORDER BY {rc['column']} DESC"
                    break

        # Build LIMIT
        limit_clause = ""
        if top_n:
            limit_clause = f" LIMIT {top_n}"
        elif not agg_func and not needs_group_by:
            limit_clause = " LIMIT 50"
        elif needs_group_by and not top_n:
            limit_clause = " LIMIT 30"

        # Assemble
        select_str = ", ".join(select_parts)
        sql = f"SELECT {select_str} FROM {from_clause}{where_clause}{group_clause}{having_sql}{order_clause}{limit_clause};"

        return sql

    @staticmethod
    def _singularize(term: str) -> str:
        """Basic plural → singular normalization for entity names."""
        t = term.lower()
        if t.endswith('ies') and len(t) > 4:       # facilities → facility
            return t[:-3] + 'y'
        if t.endswith('ses') or t.endswith('xes'):  # diagnoses → diagnosis (approx)
            return t[:-2]
        if t.endswith('s') and not t.endswith('ss') and len(t) > 3:  # members → member
            return t[:-1]
        return t

    def _resolve_group_columns(self, group_terms: List[str], resolved_cols: List[Dict],
                                tables: List[str]) -> List[str]:
        """Resolve natural language group-by terms to actual columns."""
        group_cols = []

        # Normalize plurals: "facilities" → "facility", "members" → "member"
        group_terms = [self._singularize(t) if t.lower() != self._singularize(t) and t.lower() not in SYNONYMS
                       else t for t in group_terms]

        # "by volume" / "by cost" / "by amount" are SORT targets, not GROUP BY targets
        # Filter these out — they indicate what to order by, not what to group by
        SORT_TERMS = {'volume', 'cost', 'amount', 'count', 'total', 'value', 'number', 'revenue', 'price'}
        group_terms = [t for t in group_terms if t.lower() not in SORT_TERMS]

        # Filter out domain concept terms — these are WHERE filters, not GROUP BY columns
        # e.g., "by region on medicaid" → group by region, filter by medicaid
        DOMAIN_FILTER_TERMS = {
            'medicaid', 'medicare', 'commercial', 'hmo', 'ppo', 'epo', 'hdhp',
            'diabetic', 'diabetes', 'hypertension', 'cardiac', 'asthma', 'copd',
            'depression', 'anxiety', 'obesity', 'obese', 'cancer',
            'newborn', 'infant', 'pediatric', 'elderly', 'senior', 'geriatric',
            'foster', 'adolescent', 'teen', 'adult', 'minor',
            'high', 'low', 'moderate', 'rising',  # risk levels
            'inpatient', 'outpatient', 'emergency', 'telehealth',
            'chronic', 'critical', 'severe',
            'on', 'front', 'side', 'basis', 'end', 'perspective',  # noise
        }
        group_terms = [t for t in group_terms if t.lower() not in DOMAIN_FILTER_TERMS]

        # BUG 4 FIX: Try to match multi-word columns (e.g., "claim type" → CLAIM_TYPE)
        # Check if consecutive terms match a column when combined
        combined_terms = []
        i = 0
        while i < len(group_terms):
            # Try combining current term with next term(s)
            matched = False
            for j in range(len(group_terms), i, -1):
                combined = '_'.join(group_terms[i:j])
                for t in tables:
                    for col in self.schema.get_columns(t):
                        if combined.upper() in col or combined.lower() == col.lower().replace('_', ''):
                            combined_terms.append(col)
                            i = j
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if not matched:
                combined_terms.append(group_terms[i])
                i += 1

        if len(combined_terms) < len(group_terms):
            # We found some multi-word matches, use them
            group_terms = combined_terms

        # If all group terms were sort terms, we need a real group column
        # Look for the main entity being counted (e.g., "top 10 providers" → group by provider)

        for term in group_terms:
            # Check if term directly matches a resolved column
            # BUG 1 FIX: Only check resolved columns that are in the target tables
            found = False
            for rc in resolved_cols:
                if rc['table'] not in tables:
                    continue  # Skip columns from tables we're not using
                if term.lower() in rc['column'].lower() or term.lower() in rc.get('original_term', '').lower():
                    if rc['column'] not in group_cols:
                        group_cols.append(rc['column'])
                    found = True
                    break

            if not found:
                # Try synonym lookup (exact + fuzzy for typo tolerance)
                # BUG 1 FIX: Prefer columns that exist in the resolved tables
                # BUG 4 FIX: Prefer descriptive columns (like CLAIM_TYPE) over ID columns (like CLAIM_ID)
                for phrase, cols in SYNONYMS.items():
                    if term in phrase or phrase in term or _fuzzy_match(term, phrase):
                        # Sort to prefer non-ID columns first, but ALSO prefer columns in our tables
                        cols_sorted = sorted(cols, key=lambda c: (not any(t in tables for t in self.schema.column_to_tables.get(c, [])), c.endswith('_ID'), c))
                        for c in cols_sorted:
                            c_tables = self.schema.column_to_tables.get(c, [])
                            if any(t in tables for t in c_tables):
                                if c not in group_cols:
                                    group_cols.append(c)
                                found = True
                                break
                    if found:
                        break

            if not found:
                # Try fuzzy match against all columns in used tables
                # BUG 4 FIX: Prefer non-ID columns over ID columns
                for t in tables:
                    # Collect matching columns with preference score
                    candidates = []
                    for col in self.schema.get_columns(t):
                        col_lower = col.lower()
                        term_lower = term.lower()
                        is_id_col = col_lower.endswith('_id')
                        # Exact substring match gets highest priority
                        if term_lower in col_lower:
                            candidates.append((col, 0, is_id_col))
                        else:
                            # Fuzzy match against column name without _ID suffix
                            col_base = col_lower.replace('_id', '').replace('_', '')
                            if _fuzzy_match(term_lower, col_base):
                                candidates.append((col, 1, is_id_col))

                    # Sort by priority: exact match first, then prefer non-ID
                    if candidates:
                        candidates.sort(key=lambda x: (x[1], x[2]))  # priority, then is_id_col
                        col = candidates[0][0]
                        if col not in group_cols:
                            group_cols.append(col)
                        found = True
                        break

        return group_cols

    def _infer_group_column(self, question: str, tables: List[str]) -> List[str]:
        """Infer the GROUP BY column from the question's subject entity."""
        q = question.lower()

        # BUG 4 FIX: Check for contextual suffixes like "reasons", "codes", etc.
        # "referral reasons" → REFERRAL_REASON, not REFERRAL_ID
        CONTEXT_SUFFIXES = {
            ('referral', 'reason'): 'REFERRAL_REASON',
            ('denial', 'reason'): 'DENIAL_REASON',
            ('hcc', 'category'): 'HCC_CATEGORY',
            ('icd', 'code'): 'ICD10_CODE',
        }
        for (entity, suffix), col in CONTEXT_SUFFIXES.items():
            if entity in q and suffix in q:
                for t in tables:
                    if col in [c['name'] for c in self.schema.tables.get(t, [])]:
                        return [col]

        # Map subject entities to their best grouping columns
        # BUG 4 FIX: Prefer descriptive columns (CLAIM_TYPE) over ID columns (CLAIM_ID)
        ENTITY_GROUP_COLS = {
            'provider': 'SPECIALTY', 'doctor': 'SPECIALTY', 'physician': 'SPECIALTY',  # BUG 4: specialty over NPI
            'member': 'MEMBER_ID', 'patient': 'MEMBER_ID',
            'region': 'KP_REGION', 'facility': 'FACILITY',
            'department': 'DEPARTMENT', 'specialty': 'SPECIALTY',
            'medication': 'MEDICATION_NAME', 'drug': 'MEDICATION_NAME',
            'diagnosis': 'ICD10_CODE', 'condition': 'ICD10_DESCRIPTION',
            'plan': 'PLAN_TYPE', 'visit': 'VISIT_TYPE',
            'claim': 'CLAIM_TYPE',  # BUG 4: CLAIM_TYPE instead of CLAIM_ID
            'status': 'CLAIM_STATUS', 'gender': 'GENDER',
            'chronic': 'IS_CHRONIC',  # BUG 8: chronic diagnoses
        }

        for entity, col in ENTITY_GROUP_COLS.items():
            if entity in q:
                # Verify the column exists in one of the tables
                for t in tables:
                    if col in [c['name'] for c in self.schema.tables.get(t, [])]:
                        return [col]
                # Try with provider name columns
                if col == 'NPI' and 'providers' in tables:
                    return ['NPI']

        # Default: first categorical column from primary table
        for t in tables:
            for c in self.schema.tables.get(t, []):
                dt = c.get('data_type', '').lower()
                cn = c['name'].lower()
                if dt in ('text', 'string', 'category') and \
                   not cn.endswith('_id') and 'date' not in cn and 'description' not in cn:
                    return [c['name']]

        return []

    def _find_best_numeric_col(self, question: str, resolved_cols: List[Dict],
                                tables: List[str]) -> Optional[str]:
        """Find the best numeric column for aggregation based on context."""
        q = question.lower()

        # Normalize common typos before checking keywords
        _typo_map = {
            'amout': 'amount', 'amoutn': 'amount', 'amont': 'amount',
            'bileld': 'billed', 'billd': 'billed',
            'reigon': 'region', 'reagon': 'region',
            'averge': 'average', 'avergae': 'average',
        }
        for typo, fix in _typo_map.items():
            q = q.replace(typo, fix)

        # PRIORITY 1: Context-based inference from explicit mentions
        # Check more specific phrases FIRST (Bug 4)
        if any(k in q for k in ['billed amount', 'billed_amount', 'billed charge']):
            return 'BILLED_AMOUNT'
        if any(k in q for k in ['billed', 'charge']) and 'paid' not in q:
            return 'BILLED_AMOUNT'
        if any(k in q for k in ['paid amount', 'paid_amount', 'average paid', 'avg paid', 'total paid']):
            return 'PAID_AMOUNT'
        if 'paid' in q and any(k in q for k in ['amount', 'average', 'avg', 'total', 'sum', 'max', 'min']) and 'billed' not in q:
            return 'PAID_AMOUNT'
        if any(k in q for k in ['payment', 'reimburs']):
            return 'PAID_AMOUNT'
        if 'cost' in q and 'prescriptions' in tables:
            return 'COST'
        if 'cost' in q and 'prescriptions' in tables:
            return 'COST'
        if 'cost' in q:
            return 'BILLED_AMOUNT'
        if any(k in q for k in ['allowed']):
            return 'ALLOWED_AMOUNT'
        # Generic "amount" without "billed"/"paid" qualifier → default to BILLED_AMOUNT
        if 'amount' in q and 'claims' in tables:
            return 'BILLED_AMOUNT'
        if any(k in q for k in ['risk', 'score']):
            return 'RISK_SCORE'
        if any(k in q for k in ['stay', 'los', 'length']):
            return 'LENGTH_OF_STAY'
        if any(k in q for k in ['copay']):
            return 'COPAY'
        if any(k in q for k in ['supply', 'days supply']):
            return 'DAYS_SUPPLY'

        # PRIORITY 2: Check resolved columns for numeric types
        for rc in resolved_cols:
            ci = self.schema.get_column_info(rc['table'], rc['column'])
            if ci:
                dt = ci.get('data_type', '').lower()
                cn = rc['column'].lower()
                if dt in ('currency', 'float', 'integer', 'numeric') or \
                   any(k in cn for k in ['amount', 'cost', 'paid', 'billed', 'score', 'size', 'stay',
                                          'quantity', 'refills', 'days', 'copay', 'coinsurance', 'deductible']):
                    return rc['column']

        # Default: first numeric column from primary table
        for t in tables:
            for c in self.schema.tables.get(t, []):
                dt = c.get('data_type', '').lower()
                cn = c['name'].lower()
                if dt in ('currency', 'float', 'integer', 'numeric') or \
                   any(k in cn for k in ['amount', 'cost', 'paid', 'billed']):
                    return c['name']

        return None

    def _build_select(self, question: str, resolved_cols: List[Dict],
                      agg_func: Optional[str], group_cols: List[str],
                      num_col: Optional[str], tables: List[str],
                      filters: List[str] = None) -> Tuple[List[str], bool]:
        """Build SELECT clause. Returns (select_parts, needs_group_by)."""
        q = question.lower()

        # Special case: trend/time-series queries (Bug 10)
        # "claims trend over 2024" → GROUP BY month, COUNT
        # Also handle "per month", "per year", "per day" as temporal grouping
        if any(k in q for k in ['trend', 'over time', 'month by month', 'monthly', 'time series',
                                 'per month', 'per year', 'per quarter', 'per day', 'per week',
                                 'growth rate', 'growth trend', 'growth over']):
            # Find a date column in the available tables
            # Priority: ENROLLMENT_DATE for member growth, SERVICE_DATE for claims,
            # then any date column as fallback
            date_col = None
            date_priority = []
            for t in tables:
                for c in self.schema.tables.get(t, []):
                    cn = c['name'].upper()
                    if 'DATE' not in cn:
                        continue
                    # Highest priority: matches query context
                    if 'enrollment' in q and 'ENROLLMENT' in cn:
                        date_col = c['name']
                        break
                    elif ('growth' in q or 'new member' in q) and 'ENROLLMENT' in cn:
                        date_col = c['name']
                        break
                    elif 'SERVICE' in cn:
                        date_priority.insert(0, c['name'])  # High priority
                    elif 'ADMIT' in cn or 'FILL' in cn or 'REFERRAL' in cn:
                        date_priority.insert(1, c['name'])  # Medium priority
                    elif 'BIRTH' not in cn:  # Skip DATE_OF_BIRTH for time-series
                        date_priority.append(c['name'])
                if date_col:
                    break
            if not date_col and date_priority:
                date_col = date_priority[0]
            if date_col:
                month_expr = f"SUBSTR({date_col}, 1, 7)"
                # Override group_cols to use month expression (so build() GROUPs BY it)
                group_cols.clear()
                group_cols.append(month_expr)
                return [f"{month_expr} as month", "COUNT(*) as count"], True

        # Special case: "processing time" → days between SUBMITTED_DATE and ADJUDICATED_DATE
        if 'processing time' in q or ('processing' in q and ('average' in q or 'avg' in q or 'mean' in q)):
            time_expr = "JULIANDAY(ADJUDICATED_DATE) - JULIANDAY(SUBMITTED_DATE)"
            if 'average' in q or 'avg' in q or 'mean' in q:
                agg_expr = f"ROUND(AVG({time_expr}), 2) as avg_processing_days"
            elif 'max' in q or 'longest' in q:
                agg_expr = f"ROUND(MAX({time_expr}), 2) as max_processing_days"
            elif 'min' in q or 'shortest' in q or 'fastest' in q:
                agg_expr = f"ROUND(MIN({time_expr}), 2) as min_processing_days"
            else:
                agg_expr = f"ROUND(AVG({time_expr}), 2) as avg_processing_days"
            if group_cols:
                return group_cols + [agg_expr, "COUNT(*) as claim_count"], True
            if 'detail' in q:
                return [
                    agg_expr,
                    f"ROUND(MIN({time_expr}), 2) as min_processing_days",
                    f"ROUND(MAX({time_expr}), 2) as max_processing_days",
                    "COUNT(*) as total_claims"
                ], False
            return [agg_expr, "COUNT(*) as total_claims"], False

        # Special case: "high-value claim rate" or similar concept-based rate queries
        # When domain filters define a subset and "rate" is asked, compute ratio
        _filters = filters or []
        if 'rate' in q and 'denial' not in q and 'deny' not in q:
            # Check if domain concepts added filters (e.g., high-value → BILLED_AMOUNT > 10000)
            domain_concepts = getattr(FilterExtractor, '_matched_domain_concepts', set())
            if domain_concepts and any(f for f in _filters if any(c in str(f).upper() for c in ['BILLED_AMOUNT', 'PAID_AMOUNT', 'RISK_SCORE'])):
                # Build a rate query: what % of claims match the domain filter?
                # Get the domain conditions to embed in CASE WHEN
                domain_conditions = []
                for concept in domain_concepts:
                    info = FilterExtractor.DOMAIN_CONCEPTS.get(concept, {})
                    domain_conditions.extend(info.get('conditions', []))

                if domain_conditions:
                    case_cond = ' AND '.join(domain_conditions)
                    rate_cols = [
                        "COUNT(*) as total_claims",
                        f"SUM(CASE WHEN {case_cond} THEN 1 ELSE 0 END) as matching_count",
                        f"ROUND(100.0 * SUM(CASE WHEN {case_cond} THEN 1 ELSE 0 END) / COUNT(*), 2) as rate_pct"
                    ]
                    # Clear the WHERE filters since we're computing rate via CASE
                    filters.clear()
                    if group_cols:
                        return group_cols + rate_cols, True
                    # If "details" in query, add a breakdown by status or region
                    if 'detail' in q:
                        return ['CLAIM_STATUS'] + rate_cols, True
                    return rate_cols, False

        # BUG 10 FIX: "readmission rate" special case
        # Requires self-join: encounters e1 LEFT JOIN encounters e2 to find readmissions within 30 days
        if 'readmission' in q and 'rate' in q:
            # This requires special handling at the table level (can't be done here)
            # Mark it so that _generate_single knows to build the complex query
            self._is_readmission_rate_query = True
            # Return a placeholder that will be replaced
            return [
                "COUNT(*) as total_discharges",
                "SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) as readmissions_30d",
                "ROUND(100.0 * SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) / COUNT(*), 2) as readmission_rate_pct"
            ], False

        # Special case: rate/ratio queries (Bug 5)
        if 'denial rate' in q or 'deny rate' in q:
            if group_cols:
                return group_cols + [
                    "COUNT(*) as total_claims",
                    "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_count",
                    "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) as denial_rate"
                ], True
            return [
                "COUNT(*) as total_claims",
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_count",
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) as denial_rate"
            ], False

        if 'fill rate' in q or 'approval rate' in q or 'completion rate' in q:
            if group_cols:
                return group_cols + [
                    "COUNT(*) as total",
                    "SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) as approved_count",
                    "ROUND(100.0 * SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) / COUNT(*), 2) as approval_rate"
                ], True
            return [
                "COUNT(*) as total",
                "SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) as approved_count",
                "ROUND(100.0 * SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) / COUNT(*), 2) as approval_rate"
            ], False

        # BUG 5 FIX: "what percentage/percent/pct of X are Y" → CASE WHEN ratio
        # e.g., "what percentage of claims are denied" → SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*)
        if any(p in q for p in ['percentage', 'percent', ' pct ', '%']) and ' of ' in q and ' are ' in q:
            # Extract what we're measuring and what value we're checking
            # "what percentage of claims are denied"
            denied_match = re.search(r'what\s+(?:percentage|percent|%|pct)\s+of\s+(\w+)\s+are\s+(\w+)', q)
            if denied_match:
                entity = denied_match.group(1)  # claims
                status = denied_match.group(2).upper()  # DENIED

                # Map entity to appropriate table/column
                # Also map certain values to the correct column (e.g., female→GENDER)
                PERCENT_VALUE_MAP = {
                    'FEMALE': ('GENDER', 'F'), 'MALE': ('GENDER', 'M'),
                    'CHRONIC': ('IS_CHRONIC', 'Y'),
                    'INPATIENT': ('VISIT_TYPE', 'INPATIENT'), 'OUTPATIENT': ('VISIT_TYPE', 'OUTPATIENT'),
                    'EMERGENCY': ('VISIT_TYPE', 'EMERGENCY'), 'TELEHEALTH': ('VISIT_TYPE', 'TELEHEALTH'),
                }
                if status in PERCENT_VALUE_MAP:
                    status_col, status = PERCENT_VALUE_MAP[status]
                    total_col = f'COUNT(*) as total_{entity}'
                elif 'claim' in entity:
                    total_col = 'COUNT(*) as total_claims'
                    status_col = 'CLAIM_STATUS'
                elif 'prescription' in entity:
                    total_col = 'COUNT(*) as total_prescriptions'
                    status_col = 'STATUS'
                elif 'referral' in entity:
                    total_col = 'COUNT(*) as total_referrals'
                    status_col = 'STATUS'
                elif 'encounter' in entity or 'visit' in entity:
                    total_col = f'COUNT(*) as total_{entity}'
                    status_col = 'VISIT_TYPE'
                elif 'diagnosis' in entity or 'diagnos' in entity:
                    total_col = f'COUNT(*) as total_{entity}'
                    status_col = 'SEVERITY'
                else:
                    total_col = f'COUNT(*) as total_{entity}'
                    status_col = 'STATUS'  # default

                # Clear filters since we embed the logic in CASE WHEN
                if filters:
                    filters.clear()

                # Build the percentage query
                label = status.lower().replace('_', ' ')
                if group_cols:
                    return group_cols + [
                        total_col,
                        f"SUM(CASE WHEN {status_col} = '{status}' THEN 1 ELSE 0 END) as {label}_count",
                        f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{status}' THEN 1 ELSE 0 END) / COUNT(*), 2) as {label}_pct"
                    ], True
                return [
                    total_col,
                    f"SUM(CASE WHEN {status_col} = '{status}' THEN 1 ELSE 0 END) as {label}_count",
                    f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{status}' THEN 1 ELSE 0 END) / COUNT(*), 2) as {label}_pct"
                ], False

        # BUG 11 FIX: COUNT(DISTINCT col) when "unique" or "distinct" keyword is present
        # "count of unique members per region" → COUNT(DISTINCT MEMBER_ID)
        if agg_func == 'COUNT' and ('unique' in q or 'distinct' in q):
            # Find the most appropriate column to count distinctly
            # Priority: MEMBER_ID, then any ID column in resolved_cols
            distinct_col = None
            if 'member' in q or 'patient' in q:
                distinct_col = 'MEMBER_ID'
            elif 'provider' in q or 'doctor' in q:
                distinct_col = 'NPI'
            elif 'claim' in q:
                distinct_col = 'CLAIM_ID'
            elif 'encounter' in q or 'visit' in q:
                distinct_col = 'ENCOUNTER_ID'
            else:
                # Use first ID column from resolved columns
                for rc in resolved_cols:
                    if 'ID' in rc['column'].upper():
                        distinct_col = rc['column']
                        break
                if not distinct_col and resolved_cols:
                    distinct_col = resolved_cols[0]['column']

            if distinct_col:
                if group_cols:
                    return group_cols + [f'COUNT(DISTINCT {distinct_col}) as unique_count'], True
                return [f'COUNT(DISTINCT {distinct_col}) as unique_count'], False

        # Case 1: COUNT — almost always means COUNT(*), not COUNT(some_col)
        if agg_func == 'COUNT':
            if group_cols:
                return group_cols + ['COUNT(*) as count'], True
            return ['COUNT(*) as total_count'], False

        # Case 2: SUM/AVG/MAX/MIN with numeric column
        if agg_func and num_col:
            agg_expr = f"{agg_func}(CAST({num_col} AS REAL)) as {agg_func.lower()}_{num_col.lower()}"
            if group_cols:
                return group_cols + [agg_expr], True
            return [agg_expr], False

        # Case 4: Group columns present but no explicit agg function
        # e.g., "compare billed vs paid by region" → auto-aggregate
        if group_cols and not agg_func:
            select_parts = list(group_cols)
            # Find numeric columns to aggregate — but only from tables in our query
            for rc in resolved_cols:
                cn = rc['column'].lower()
                # Verify column exists in one of the tables we're using
                col_exists = any(rc['column'].upper() in [c.upper() for c in self.schema.get_columns(t)] for t in tables)
                if not col_exists:
                    continue
                if any(k in cn for k in ['amount', 'cost', 'paid', 'billed', 'score', 'stay',
                                          'copay', 'coinsurance', 'deductible']):
                    select_parts.append(f"AVG(CAST({rc['column']} AS REAL)) as avg_{rc['column'].lower()}")
            if len(select_parts) == len(group_cols):
                # No numeric cols found, just count
                select_parts.append('COUNT(*) as count')
            return select_parts, True

        # Case 5: No aggregation — select specific columns
        select_cols = []
        seen_cols = set()

        # Add resolved columns that aren't IDs (unless specifically asked)
        for rc in resolved_cols:
            col = rc['column']
            if col not in seen_cols and rc['table'] in tables:
                select_cols.append(col)
                seen_cols.add(col)

        # If we have very few columns, add key columns from primary table
        if len(select_cols) < 3:
            key_cols = self._get_key_columns(tables[0])
            for kc in key_cols:
                if kc not in seen_cols:
                    select_cols.append(kc)
                    seen_cols.add(kc)
                    if len(select_cols) >= 8:
                        break

        # If still empty, select key columns
        if not select_cols:
            select_cols = self._get_key_columns(tables[0])

        return select_cols[:12], bool(group_cols)  # Cap at 12 columns

    def _get_key_columns(self, table: str) -> List[str]:
        """Get the most important columns for a table (for display)."""
        KEY_COLS = {
            'claims': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE', 'CPT_DESCRIPTION',
                       'BILLED_AMOUNT', 'PAID_AMOUNT', 'CLAIM_STATUS', 'DENIAL_REASON', 'KP_REGION'],
            'members': ['MEMBER_ID', 'FIRST_NAME', 'LAST_NAME', 'DATE_OF_BIRTH', 'GENDER',
                        'KP_REGION', 'PLAN_TYPE', 'ENROLLMENT_DATE', 'RISK_SCORE'],
            'providers': ['NPI', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'SPECIALTY',
                         'DEPARTMENT', 'KP_REGION', 'FACILITY'],
            'encounters': ['ENCOUNTER_ID', 'MEMBER_ID', 'SERVICE_DATE', 'VISIT_TYPE',
                          'DEPARTMENT', 'FACILITY', 'PRIMARY_DIAGNOSIS', 'LENGTH_OF_STAY'],
            'diagnoses': ['DIAGNOSIS_ID', 'MEMBER_ID', 'ICD10_CODE', 'ICD10_DESCRIPTION',
                         'DIAGNOSIS_TYPE', 'DIAGNOSIS_DATE', 'SEVERITY', 'IS_CHRONIC'],
            'prescriptions': ['RX_ID', 'MEMBER_ID', 'MEDICATION_NAME', 'MEDICATION_CLASS',
                            'PRESCRIPTION_DATE', 'COST', 'STATUS'],
            'referrals': ['REFERRAL_ID', 'MEMBER_ID', 'REFERRAL_REASON', 'REFERRAL_DATE',
                         'URGENCY', 'STATUS', 'SPECIALTY'],
            'appointments': ['APPOINTMENT_ID', 'MEMBER_ID', 'APPOINTMENT_DATE', 'APPOINTMENT_TIME',
                            'APPOINTMENT_TYPE', 'DEPARTMENT', 'FACILITY', 'KP_REGION', 'STATUS'],
            'cpt_codes': ['CPT_CODE', 'DESCRIPTION', 'CATEGORY', 'RVU'],
        }
        return KEY_COLS.get(table, self.schema.get_columns(table)[:8])

    def _build_from(self, tables: List[str]) -> str:
        """Build FROM clause with JOINs."""
        if len(tables) <= 1:
            return tables[0] if tables else 'claims'

        joins = self.schema.find_join_path(tables)
        base = tables[0]
        # Use table aliases: claims as c, members as m, etc.
        aliases = {}
        for t in tables:
            aliases[t] = t[0]  # first letter
            # Handle duplicates
            if sum(1 for tt in tables if tt[0] == t[0]) > 1:
                aliases[t] = t[:3]

        from_parts = [f"{base} {aliases[base]}"]
        joined = {base}
        for t1, t2, col in joins:
            if t2 not in joined:
                a1 = aliases.get(t1, t1)
                a2 = aliases.get(t2, t2)
                # Handle asymmetric joins like "NPI=RENDERING_NPI"
                if '=' in col:
                    left_col, right_col = col.split('=', 1)
                    conditions = [f"{a1}.{left_col} = {a2}.{right_col}"]
                else:
                    conditions = [f"{a1}.{col} = {a2}.{col}"]

                # Add composite join keys (e.g., KP_REGION) when configured
                extra_cols = self.schema.get_composite_join_conditions(t1, t2, col)
                for ec in extra_cols:
                    conditions.append(f"{a1}.{ec} = {a2}.{ec}")

                from_parts.append(f"JOIN {t2} {a2} ON {' AND '.join(conditions)}")
                joined.add(t2)

        # Store aliases for column qualification
        self._current_aliases = aliases
        return " ".join(from_parts)

    def _qualify_select_expr(self, expr: str, tables: List[str]) -> str:
        """Qualify column names within a SELECT expression (handles AGG(COL) as alias)."""
        # Match patterns like AVG(CAST(COL AS REAL)) or just COL_NAME
        import re
        # Find bare column references (not already qualified with alias.)
        def replace_col(m):
            col = m.group(0)
            if '.' in col or col.upper() in ('AS', 'REAL', 'TEXT', 'INTEGER', 'CAST', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN'):
                return col
            # Check if it's actually a column
            for t in tables:
                if col.upper() in [c.upper() for c in self.schema.get_columns(t)]:
                    return self._qualify_column(col, tables)
            return col

        # Only qualify the column part, not the alias
        if ' as ' in expr.lower():
            parts = expr.split(' as ', 1)
            qualified = re.sub(r'\b([A-Z_][A-Z_0-9]*)\b', replace_col, parts[0])
            return qualified + ' as ' + parts[1]
        return re.sub(r'\b([A-Z_][A-Z_0-9]*)\b', replace_col, expr)

    def _qualify_filter(self, filter_expr: str, tables: List[str]) -> str:
        """Qualify column names within a WHERE filter expression.

        BUG 8 FIX: Don't qualify text inside quoted strings (values).
        """
        import re
        # Split on quoted strings to avoid qualifying values inside quotes
        # This is a simple approach: track whether we're inside quotes
        parts = []
        in_quotes = False
        current = []
        i = 0
        while i < len(filter_expr):
            ch = filter_expr[i]
            if ch == "'":
                # Toggle quote state
                in_quotes = not in_quotes
                current.append(ch)
            elif not in_quotes:
                # Only qualify outside of quotes
                current.append(ch)
            else:
                # Inside quotes, don't touch
                current.append(ch)
            i += 1

        # Now apply qualification only to parts outside quotes
        def replace_col(m):
            col = m.group(0)
            if '.' in col or col.upper() in ('AND', 'OR', 'NOT', 'IS', 'NULL', 'LIKE', 'BETWEEN', 'CAST', 'AS', 'REAL', 'TEXT', 'INTEGER'):
                return col
            is_column = False
            for t in tables:
                if col.upper() in [c.upper() for c in self.schema.get_columns(t)]:
                    is_column = True
                    break
            if is_column:
                return self._qualify_column(col, tables)
            return col

        # Process: split by quotes, only apply regex to non-quoted parts
        result = []
        parts = filter_expr.split("'")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Outside quotes
                part = re.sub(r'\b([A-Z_][A-Z_0-9]*)\b', replace_col, part)
            result.append(part)
        return "'".join(result)

    def _qualify_column(self, col: str, tables: List[str]) -> str:
        """Qualify a column with table alias if it exists in multiple tables."""
        if not hasattr(self, '_current_aliases') or len(tables) <= 1:
            return col

        # Check which tables have this column
        tables_with_col = []
        for t in tables:
            if col.upper() in [c.upper() for c in self.schema.get_columns(t)]:
                tables_with_col.append(t)

        if len(tables_with_col) > 1:
            # Ambiguous — qualify with first table's alias
            alias = self._current_aliases.get(tables_with_col[0], tables_with_col[0])
            return f"{alias}.{col}"
        elif len(tables_with_col) == 1:
            alias = self._current_aliases.get(tables_with_col[0], tables_with_col[0])
            return f"{alias}.{col}"

        return col


# =============================================================================
# TABLE RESOLVER — determines which tables are needed
# =============================================================================

class TableResolver:
    """Determines which tables are needed for a query."""

    # Keywords that strongly indicate specific tables
    TABLE_KEYWORDS = {
        'claims': ['claim', 'billed', 'paid', 'denied', 'denial', 'cpt', 'allowed',
                    'copay', 'coinsurance', 'deductible', 'adjudicated', 'submitted',
                    'cost', 'spend', 'expense', 'charge', 'revenue'],
        'members': ['member', 'patient', 'enrolled', 'disenroll', 'mrn', 'age', 'gender',
                    'race', 'language', 'risk score', 'chronic condition', 'pcp',
                    'birth', 'dob', 'address', 'zip', 'phone', 'email'],
        'providers': ['provider', 'doctor', 'npi', 'specialty', 'department', 'panel', 'hire',
                      'dea', 'license', 'accepts new'],
        'encounters': ['encounter', 'visit', 'admission', 'admit', 'discharge',
                       'chief complaint', 'disposition', 'length of stay', 'los',
                       'stay', 'facility', 'utilization',
                       'inpatient', 'outpatient', 'emergency', 'supervising'],
        'diagnoses': ['diagnosis', 'diagnos', 'icd', 'hcc', 'severity', 'chronic',
                      'resolved', 'condition'],
        'prescriptions': ['prescription', 'medication', 'drug', 'pharmacy', 'rx',
                         'refill', 'ndc', 'days supply', 'fill date', 'med class'],
        'referrals': ['referral', 'referred', 'authorization', 'urgency',
                      'appointment', 'referring'],
        'appointments': ['appointment', 'schedule', 'upcoming', 'no show', 'no_show',
                        'reschedule', 'pcp visit', 'annual wellness', 'lab work', 'imaging'],
        'cpt_codes': ['cpt', 'procedure code', 'rvu', 'relative value'],
    }

    @classmethod
    def resolve(cls, question: str, resolved_cols: List[Dict]) -> List[str]:
        """
        Determine which tables are needed.

        Uses keyword matching primarily, with resolved columns as tiebreaker.
        Strict thresholds to avoid pulling in unnecessary tables.
        """
        q = question.lower()

        # Normalize common typos before keyword matching
        TYPO_FIXES = {
            'perscription': 'prescription', 'perscriptions': 'prescriptions',
            'prescripton': 'prescription', 'prescriptons': 'prescriptions',
            'medicaton': 'medication', 'medicatons': 'medications',
            'medcation': 'medication', 'mediction': 'medication',
            'diagnosis': 'diagnosis', 'dignosis': 'diagnosis',
            'diagonsis': 'diagnosis', 'diganosis': 'diagnosis',
            'encountrs': 'encounters', 'encountr': 'encounter',
            'speciality': 'specialty', 'specialties': 'specialty',
            'refferal': 'referral', 'referal': 'referral',
            'provder': 'provider', 'providor': 'provider',
        }
        for typo, fix in TYPO_FIXES.items():
            q = q.replace(typo, fix)

        keyword_scores = defaultdict(float)
        col_scores = defaultdict(float)

        # Score from keywords (strong signal)
        # Known false positives where substring match causes wrong table
        FALSE_POSITIVE_PAIRS = {
            # (keyword, container_word) → keyword should NOT match when inside container_word
            ('patient', 'inpatient'), ('patient', 'outpatient'),
            ('age', 'average'), ('age', 'usage'), ('age', 'package'), ('age', 'manage'),
            ('admit', 'admitted'),  # admit is encounters, admitted context may differ
            ('rx', 'proxy'),
        }
        fp_suppressed = set()
        for kw, container in FALSE_POSITIVE_PAIRS:
            if container in q and kw in container:
                fp_suppressed.add(kw)

        for table, keywords in cls.TABLE_KEYWORDS.items():
            for kw in keywords:
                if kw in fp_suppressed:
                    continue
                if kw in q:
                    keyword_scores[table] += 2.0

        # Score from resolved columns (weaker signal — only count unique columns,
        # not every synonym expansion)
        seen_cols_per_table = defaultdict(set)
        for rc in resolved_cols:
            col = rc['column']
            tbl = rc['table']
            if col not in seen_cols_per_table[tbl]:
                seen_cols_per_table[tbl].add(col)
                # Only count synonym/exact, not substring (too noisy)
                if rc['match_type'] in ('exact', 'synonym'):
                    col_scores[tbl] += 1.0

        # Combine scores
        scores = defaultdict(float)
        for t in set(list(keyword_scores.keys()) + list(col_scores.keys())):
            scores[t] = keyword_scores.get(t, 0) + col_scores.get(t, 0)

        # Score from value mentions (strong signal — actual data values like 'NCAL', 'DENIED', 'Cardiology')
        value_scores = ValueResolver.get_table_scores_from_values(question)
        for t, vs in value_scores.items():
            scores[t] += vs

        if not scores:
            return ['claims']  # Default

        # Sort by score, return tables with meaningful scores
        sorted_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_tables[0]
        result = [primary[0]]

        # Only add secondary tables if they have KEYWORD evidence (not just column matches)
        # A table must have at least one keyword match AND the score must be significant
        # relative to the primary table
        primary_score = primary[1]
        for table, score in sorted_tables[1:]:
            # Must have keyword evidence AND strong enough score
            if keyword_scores.get(table, 0) >= 2.0 and score >= primary_score * 0.4:
                result.append(table)

        return result[:3]  # Max 3-way join


# =============================================================================
# MAIN ENGINE — the public API
# =============================================================================

class DynamicSQLEngine:
    """
    Main entry point for dynamic SQL generation.

    Usage:
        engine = DynamicSQLEngine(catalog_dir='semantic_catalog/')
        result = engine.generate("give me initiative start dates")
        # result = {'sql': 'SELECT ...', 'tables': [...], 'confidence': 0.85, ...}
    """

    def __init__(self, catalog_dir: str = None, db_path: str = None,
                 composite_joins: bool = None):
        """
        Args:
            catalog_dir: Path to semantic catalog directory
            db_path: Path to the database (for auto-detecting composite join needs)
            composite_joins: Force composite joins on/off.
                None = auto-detect from database.
                True = always use composite joins (KP production / Databricks).
                False = single-key joins only (local demo).
        """
        if catalog_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            catalog_dir = os.path.join(os.path.dirname(script_dir), 'semantic_catalog')

        self.schema = SchemaRegistry(catalog_dir)

        # Configure composite joins
        if composite_joins is not None:
            SchemaRegistry.USE_COMPOSITE_JOINS = composite_joins
        elif db_path:
            self.schema.detect_composite_join_need(db_path)
        else:
            # Try auto-detect with default db path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_db = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_production.db')
            if os.path.exists(default_db):
                self.schema.detect_composite_join_need(default_db)

        self.resolver = ColumnResolver(self.schema)
        self.builder = DynamicSQLBuilder(self.schema)

    def _split_multi_question(self, question: str) -> List[str]:
        """
        Detect and split multi-question inputs.
        "Which diagnosis has most claims, which regions have most visits" → 2 questions.
        Returns list of individual questions (possibly just [question] if single).
        """
        q = question.strip()
        # Split on comma/semicolon followed by a question word
        # Pattern: ", which", "; how", ", what", etc.
        parts = re.split(
            r'[,;]\s*(?=(?:which|what|how|who|where|when|why|give|show|list|get|find|tell|count|total)\b)',
            q, flags=re.IGNORECASE
        )
        # Also split on period followed by a question word
        if len(parts) == 1:
            parts = re.split(
                r'\.\s*(?=(?:which|what|how|who|where|when|why|give|show|list|get|find|tell|count|total)\b)',
                q, flags=re.IGNORECASE
            )
        # Also split on " and " when followed by question words (but not "and region", "and plan")
        if len(parts) == 1:
            parts = re.split(
                r'\s+and\s+(?=(?:which|what|how|who|also|then)\b)',
                q, flags=re.IGNORECASE
            )
        # Filter empty parts
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        return parts if len(parts) > 1 else [question]

    # ── RATIO PATTERN MAPS ──
    # Maps natural-language terms to (table, column, value) for ratio queries.
    # "emergency to outpatient ratio" → encounters.VISIT_TYPE EMERGENCY vs OUTPATIENT
    RATIO_VALUE_MAP = {
        # Visit / encounter types
        'emergency':   ('encounters', 'VISIT_TYPE', 'EMERGENCY'),
        'outpatient':  ('encounters', 'VISIT_TYPE', 'OUTPATIENT'),
        'inpatient':   ('encounters', 'VISIT_TYPE', 'INPATIENT'),
        'telehealth':  ('encounters', 'VISIT_TYPE', 'TELEHEALTH'),
        'urgent care': ('encounters', 'VISIT_TYPE', 'URGENT_CARE'),
        'urgent_care': ('encounters', 'VISIT_TYPE', 'URGENT_CARE'),
        'home health': ('encounters', 'VISIT_TYPE', 'HOME_HEALTH'),
        'home_health': ('encounters', 'VISIT_TYPE', 'HOME_HEALTH'),
        # Claim types
        'professional':  ('claims', 'CLAIM_TYPE', 'PROFESSIONAL'),
        'institutional': ('claims', 'CLAIM_TYPE', 'INSTITUTIONAL'),
        'pharmacy':      ('claims', 'CLAIM_TYPE', 'PHARMACY'),
        'dme':           ('claims', 'CLAIM_TYPE', 'DME'),
        # Claim status
        'denied':   ('claims', 'CLAIM_STATUS', 'DENIED'),
        'approved':  ('claims', 'CLAIM_STATUS', 'APPROVED'),
        'paid':      ('claims', 'CLAIM_STATUS', 'PAID'),
        'pending':   ('claims', 'CLAIM_STATUS', 'PENDING'),
        # Referral status
        'completed': ('referrals', 'REFERRAL_STATUS', 'COMPLETED'),
        'open':      ('referrals', 'REFERRAL_STATUS', 'OPEN'),
        # Gender
        'male':   ('members', 'GENDER', 'M'),
        'female': ('members', 'GENDER', 'F'),
    }

    _RATIO_RE = re.compile(
        r'(?:ratio|comparison|compare)\s+(?:of\s+)?(.+?)\s+(?:to|vs\.?|versus|and)\s+(.+?)(?:\s+(?:ratio|details?|breakdown|by\s+.+))?$'
        r'|'
        r'(.+?)\s+(?:to|vs\.?|versus)\s+(.+?)\s+(?:ratio|proportion|comparison|details?|breakdown)',
        re.IGNORECASE
    )

    def _try_cost_comparison(self, question: str, tables: List[str],
                             resolved_cols: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Detect "compare X vs Y costs/amounts" and generate proper cost aggregation.
        Returns a complete result dict if matched, else None.

        Examples:
          - "compare inpatient vs outpatient costs"
          - "compare emergency and outpatient billed amounts"
        """
        q = question.lower().strip()

        # Must have compare/vs + cost keywords
        has_compare = any(kw in q for kw in ('compare', 'vs', 'versus', 'comparison'))
        has_cost = any(kw in q for kw in ('cost', 'amount', 'billed', 'paid', 'allowed', 'spend', 'expense', 'charge'))
        if not has_compare or not has_cost:
            return None

        # Find two known visit/claim type terms
        found_terms = []
        for term in sorted(self.RATIO_VALUE_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                info = self.RATIO_VALUE_MAP[term]
                found_terms.append((term, info))

        if len(found_terms) < 2:
            return None

        (term_a, info_a), (term_b, info_b) = found_terms[0], found_terms[1]
        table_a, col_a, val_a = info_a
        table_b, col_b, val_b = info_b

        # Both must be from the same column (e.g., both VISIT_TYPE)
        if col_a != col_b:
            return None

        column = col_a
        table = table_a

        # Determine the cost column and join table
        # If visit types → need encounters + claims join
        cost_col = 'BILLED_AMOUNT'
        if 'paid' in q:
            cost_col = 'PAID_AMOUNT'
        elif 'allowed' in q:
            cost_col = 'ALLOWED_AMOUNT'

        if table == 'encounters':
            # Need claims for amounts
            sql = (
                f"SELECT e.{column}, "
                f"COUNT(*) as visit_count, "
                f"AVG(CAST(c.{cost_col} AS REAL)) as avg_{cost_col.lower()}, "
                f"SUM(CAST(c.{cost_col} AS REAL)) as total_{cost_col.lower()}, "
                f"MIN(CAST(c.{cost_col} AS REAL)) as min_{cost_col.lower()}, "
                f"MAX(CAST(c.{cost_col} AS REAL)) as max_{cost_col.lower()} "
                f"FROM encounters e "
                f"JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                f"WHERE e.{column} IN ('{val_a}', '{val_b}') "
                f"GROUP BY e.{column};"
            )
            used_tables = ['encounters', 'claims']
        else:
            sql = (
                f"SELECT {column}, "
                f"COUNT(*) as record_count, "
                f"AVG(CAST({cost_col} AS REAL)) as avg_{cost_col.lower()}, "
                f"SUM(CAST({cost_col} AS REAL)) as total_{cost_col.lower()} "
                f"FROM {table} "
                f"WHERE {column} IN ('{val_a}', '{val_b}') "
                f"GROUP BY {column};"
            )
            used_tables = [table]

        return {
            'sql': sql,
            'tables_used': used_tables,
            'columns_resolved': resolved_cols,
            'filters': [],
            'agg_info': {'agg_func': 'AVG', 'group_by_terms': [column], 'top_n': None, 'order': 'DESC'},
            'confidence': 0.95,
            'explanation': f"Comparing {cost_col} between {val_a} and {val_b} by grouping on {column}. "
                          f"Shows count, average, total, min, and max for each type.",
        }

    def _try_nested_count(self, question: str, tables: List[str],
                          resolved_cols: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Detect "count where one person visited X more than N times" patterns.
        Generates a nested subquery: COUNT(*) FROM (... GROUP BY MEMBER_ID HAVING COUNT(*) > N).

        Examples:
          - "what is the count where one person visited emergency more than 10 times"
          - "how many people went to the ER more than 5 times"
        """
        q = question.lower().strip()

        # Normalize common typos for member/patient/person
        _member_typos = {
            'memebrs': 'members', 'memebers': 'members', 'membrs': 'members',
            'membes': 'members', 'memmbers': 'members', 'mebers': 'members',
            'memebers': 'members', 'memers': 'members', 'memberes': 'members',
            'patinets': 'patients', 'paitents': 'patients', 'patietns': 'patients',
            'pateints': 'patients', 'perosn': 'person', 'persn': 'person',
        }
        for typo, fix in _member_typos.items():
            q = q.replace(typo, fix)

        # Pattern: count/how many + person/member/patient + visited/went/had + type + more than N + times
        m = re.search(
            r'\b(?:count|how many|number of)\b.*?'
            r'\b(?:person|people|member|members|patient|patients|individual|individuals)\b.*?'
            r'\b(?:visited|went to|had|received|been to|went|seen)\s+'
            r'(?:the\s+)?(\w+)\s+'
            r'(?:more than|over|at least|greater than|>=?)\s*(\d+)\s*'
            r'(?:times?|visits?|admissions?|encounters?)',
            q, re.IGNORECASE
        )
        if not m:
            # Also try: "count where one person visited X more than N times"
            m = re.search(
                r'\b(?:count|how many)\b.*?'
                r'\b(?:one\s+)?(?:person|member|patient)\b.*?'
                r'\b(?:visited|went|had)\s+(\w+)\s+'
                r'(?:more than|over|at least|greater than|>)\s*(\d+)\s*'
                r'(?:times?|visits?)',
                q, re.IGNORECASE
            )
        if not m:
            return None

        visit_type_word = m.group(1).upper()
        threshold = m.group(2)

        # Map common words to actual VISIT_TYPE values
        VISIT_TYPE_MAP = {
            'EMERGENCY': 'EMERGENCY', 'ER': 'EMERGENCY', 'ED': 'EMERGENCY',
            'INPATIENT': 'INPATIENT', 'HOSPITAL': 'INPATIENT',
            'OUTPATIENT': 'OUTPATIENT', 'OFFICE': 'OUTPATIENT',
            'TELEHEALTH': 'TELEHEALTH', 'VIRTUAL': 'TELEHEALTH',
            'URGENT': 'URGENT_CARE',
        }
        visit_type = VISIT_TYPE_MAP.get(visit_type_word, visit_type_word)

        sql = (
            f"SELECT COUNT(*) as member_count FROM ("
            f"SELECT MEMBER_ID FROM encounters "
            f"WHERE VISIT_TYPE = '{visit_type}' "
            f"GROUP BY MEMBER_ID "
            f"HAVING COUNT(*) > {threshold}"
            f") sub;"
        )

        return {
            'sql': sql,
            'tables_used': ['encounters'],
            'columns_resolved': resolved_cols,
            'filters': [],
            'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['MEMBER_ID'], 'top_n': None, 'order': 'DESC'},
            'confidence': 0.95,
            'explanation': f"Counting members who visited {visit_type} more than {threshold} times. "
                          f"Uses a subquery: first groups encounters by member with a HAVING threshold, "
                          f"then counts how many members meet the criteria.",
        }

    def _try_ratio_pattern(self, question: str, tables: List[str],
                           resolved_cols: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Detect "X to Y ratio" patterns and generate proper CASE-WHEN SQL.

        Returns a complete result dict if a ratio pattern is found, else None.
        Examples handled:
          - "show me emergency to outpatient ratio details"
          - "emergency vs inpatient ratio by region"
          - "compare denied to approved claims"

        BUG 6 FIX: If question contains cost/amount/billed/paid keywords, generate
        cost comparison instead of count ratio.
        """
        q = question.lower().strip()

        # Quick gate — must contain ratio-related keywords
        if not any(kw in q for kw in ('ratio', 'vs', 'versus', 'compare', 'comparison', 'proportion')):
            return None

        # BUG 6 FIX: If this is a cost comparison (not a count ratio), handle differently
        cost_keywords = {'cost', 'amount', 'billed', 'paid', 'allowed', 'spend', 'expense', 'charge'}
        if any(kw in q for kw in cost_keywords):
            # This is asking for cost comparison: "compare inpatient vs outpatient costs"
            # Should return AVG(BILLED_AMOUNT) or SUM(BILLED_AMOUNT) grouped by visit type
            # Let the normal flow handle this, return None to skip ratio pattern
            return None

        # Strategy: scan the question for any two known RATIO_VALUE_MAP terms
        # connected by "to", "vs", "versus", or appearing when "ratio/compare" is present
        found_terms = []
        for term in sorted(self.RATIO_VALUE_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                found_terms.append(term)

        # Need exactly 2 (or at least 2) terms from the map
        if len(found_terms) < 2:
            return None

        # Pick the first two terms (longest matches first, so multi-word wins)
        term_a, term_b = found_terms[0], found_terms[1]

        # Verify they appear in a comparison context (A to B, A vs B, compare A and B)
        # Build a loose check: both terms present + a ratio/comparison keyword
        has_connector = any(kw in q for kw in ('ratio', 'vs', 'versus', 'compare', 'comparison', 'proportion'))
        if not has_connector:
            return None

        # Look up both terms
        info_a = self.RATIO_VALUE_MAP.get(term_a)
        info_b = self.RATIO_VALUE_MAP.get(term_b)

        if not info_a or not info_b:
            return None

        table_a, col_a, val_a = info_a
        table_b, col_b, val_b = info_b

        # Both should reference the same column for a meaningful ratio
        if col_a != col_b:
            return None

        column = col_a
        table = table_a  # same table

        # Detect optional GROUP BY from "by region", "by department", etc.
        group_col = None
        group_match = re.search(r'\bby\s+(\w[\w\s]*?)(?:\s*$)', q)
        if group_match:
            group_term = group_match.group(1).strip().upper().replace(' ', '_')
            # Resolve to actual column name
            table_cols = [c.upper() for c in self.schema.get_columns(table)]
            if group_term in table_cols:
                group_col = group_term
            elif group_term + '_ID' in table_cols:
                group_col = group_term + '_ID'
            elif 'KP_' + group_term in table_cols:
                group_col = 'KP_' + group_term

        # Build the SQL
        label_a = val_a.lower().replace('_', ' ')
        label_b = val_b.lower().replace('_', ' ')

        select_parts = []
        if group_col:
            select_parts.append(group_col)

        select_parts.extend([
            f"SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) as {label_a.replace(' ', '_')}_count",
            f"SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END) as {label_b.replace(' ', '_')}_count",
            "COUNT(*) as total_count",
            f"ROUND(100.0 * SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) / COUNT(*), 2) as {label_a.replace(' ', '_')}_pct",
            f"ROUND(100.0 * SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END) / COUNT(*), 2) as {label_b.replace(' ', '_')}_pct",
            f"ROUND(CAST(SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) AS REAL) "
            f"/ NULLIF(SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END), 0), 4) as {label_a.replace(' ', '_')}_to_{label_b.replace(' ', '_')}_ratio",
        ])

        sql = f"SELECT {', '.join(select_parts)} FROM {table}"
        if group_col:
            sql += f" GROUP BY {group_col} ORDER BY {label_a.replace(' ', '_')}_to_{label_b.replace(' ', '_')}_ratio DESC"
        sql += " LIMIT 50;"

        return {
            'sql': sql,
            'tables_used': [table],
            'columns_resolved': resolved_cols,
            'filters': [],
            'agg_info': {'agg_func': 'RATIO', 'group_by_terms': [group_col] if group_col else [], 'top_n': None, 'order': 'DESC'},
            'confidence': 0.95,
            'explanation': f"Computing the ratio of {label_a} to {label_b} in {table} using CASE WHEN expressions. "
                          f"Shows counts, percentages, and the direct ratio."
                          + (f" Grouped by {group_col}." if group_col else ""),
        }

    def generate(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL from a natural language question.
        If the input contains multiple questions, generates SQL for the LAST question
        (most recent intent) and notes that multiple questions were detected.

        Returns:
            {
                'sql': str,              # The generated SQL
                'tables_used': [str],    # Tables involved
                'columns_resolved': [...], # Which columns were identified
                'filters': [str],        # WHERE conditions
                'agg_info': {...},       # Aggregation details
                'confidence': float,     # 0.0-1.0 confidence score
                'explanation': str,      # Human-readable explanation of what it's doing
            }
        """
        # Step 0a: CENTRALIZED TYPO NORMALIZATION
        # Fix all common misspellings ONCE at the entry point so every downstream
        # method (resolver, filter extractor, aggregate detector, interceptors) works
        # with clean text. This replaces scattered typo maps throughout the codebase.
        question = normalize_typos(question)

        # Step 0b: Multi-question detection
        # If user typed multiple questions, generate SQL for each separately
        parts = self._split_multi_question(question)
        if len(parts) > 1:
            # Generate SQL for each sub-question independently
            results = []
            for part in parts:
                results.append(self._generate_single(part))
            # Return a combined result with multiple SQLs
            combined = results[-1].copy()  # Use last question as primary
            combined['multi_query'] = True
            combined['all_results'] = results
            combined['sql'] = ' UNION_SPLIT '.join(r['sql'] for r in results)
            return combined

        return self._generate_single(question)

    def _generate_single(self, question: str) -> Dict[str, Any]:
        """Generate SQL for a single question."""
        # Step 1: Resolve columns from question
        resolved_cols = self.resolver.resolve(question)

        # Step 2: Determine tables needed
        tables = TableResolver.resolve(question, resolved_cols)

        # Refine column resolution with table hints
        resolved_cols = self.resolver.resolve(question, hint_tables=tables)

        # Step 3: Detect aggregation intent
        agg_info = AggregateDetector.detect(question)

        # Step 3b: RATIO PATTERN — "X to Y ratio" or "X vs Y ratio"
        # Detects questions like "emergency to outpatient ratio" and generates
        # proper CASE WHEN ratio SQL instead of impossible WHERE col='A' AND col='B'
        ratio_sql = self._try_ratio_pattern(question, tables, resolved_cols)
        if ratio_sql:
            return ratio_sql

        # Step 3c: BUG 10 FIX - READMISSION RATE special case
        # "what is the readmission rate" → complex self-join calculation
        if 'readmission' in question.lower() and 'rate' in question.lower():
            sql = """SELECT COUNT(*) as total_discharges,
  SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) as readmissions_30d,
  ROUND(100.0 * SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) / COUNT(*), 2) as readmission_rate_pct
FROM (
  SELECT e1.ENCOUNTER_ID, e1.MEMBER_ID, e1.DISCHARGE_DATE,
    MIN(JULIANDAY(e2.ADMIT_DATE) - JULIANDAY(e1.DISCHARGE_DATE)) as days_to_readmit
  FROM encounters e1
  LEFT JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID
    AND e2.ADMIT_DATE > e1.DISCHARGE_DATE
    AND e2.VISIT_TYPE = 'INPATIENT'
  WHERE e1.VISIT_TYPE = 'INPATIENT' AND e1.DISCHARGE_DATE != ''
  GROUP BY e1.ENCOUNTER_ID
) sub;"""
            return {
                'sql': sql,
                'tables_used': ['encounters'],
                'columns_resolved': resolved_cols,
                'filters': [],
                'agg_info': {'agg_func': 'RATE', 'group_by_terms': [], 'top_n': None, 'order': 'DESC'},
                'confidence': 0.95,
                'explanation': "Computing 30-day readmission rate: percentage of inpatient discharges followed by readmission within 30 days.",
            }

        # Step 3d: COST COMPARISON — "compare X vs Y costs/amounts"
        # When two visit/claim types + cost keywords are present, generate aggregation by type
        cost_comparison = self._try_cost_comparison(question, tables, resolved_cols)
        if cost_comparison:
            return cost_comparison

        # Step 3e: NESTED COUNT — "count where one person visited X more than N times"
        # Needs a subquery: SELECT COUNT(*) FROM (SELECT MEMBER_ID ... HAVING COUNT(*) > N)
        nested_subquery = self._try_nested_count(question, tables, resolved_cols)
        if nested_subquery:
            return nested_subquery

        # Step 3f: SMART PATTERN INTERCEPTORS — handle patterns that the general pipeline struggles with
        q_lower = question.lower()

        # 3f-1: "members older than N" / "patients older than N" / "members younger than N"
        # Age detection via "older/younger" phrasing (the FilterExtractor handles "over/under N" for ages,
        # but "older than" is a different lexical form)
        older_m = re.search(r'\b(?:older|elder)\s+than\s+(\d+)', q_lower)
        younger_m = re.search(r'\b(?:younger)\s+than\s+(\d+)', q_lower)
        if older_m and any(w in q_lower for w in ['member', 'patient', 'person', 'people']):
            age_int = int(older_m.group(1))
            if 'members' not in tables:
                tables.append('members')
            # Let FilterExtractor run but inject the age condition afterward
            filters_pre, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)
            age_cond = f"DATE_OF_BIRTH <= date('now', '-{age_int} years')"
            if age_cond not in filters_pre:
                filters_pre.append(age_cond)
            # Build SQL with injected filter
            domain_tables = getattr(FilterExtractor, '_domain_tables_needed', [])
            for dt in domain_tables:
                if dt not in tables:
                    tables.append(dt)
            sql = self.builder.build(question, resolved_cols, filters_pre, agg_info, tables, having_clause)
            confidence = self._compute_confidence(resolved_cols, tables, filters_pre, agg_info)
            explanation = self._explain(question, resolved_cols, tables, filters_pre, agg_info)
            return {'sql': sql, 'tables_used': tables, 'columns_resolved': resolved_cols,
                    'filters': filters_pre, 'agg_info': agg_info, 'confidence': confidence, 'explanation': explanation}
        if younger_m and any(w in q_lower for w in ['member', 'patient', 'person', 'people']):
            age_int = int(younger_m.group(1))
            if 'members' not in tables:
                tables.append('members')
            filters_pre, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)
            age_cond = f"DATE_OF_BIRTH >= date('now', '-{age_int} years')"
            if age_cond not in filters_pre:
                filters_pre.append(age_cond)
            domain_tables = getattr(FilterExtractor, '_domain_tables_needed', [])
            for dt in domain_tables:
                if dt not in tables:
                    tables.append(dt)
            sql = self.builder.build(question, resolved_cols, filters_pre, agg_info, tables, having_clause)
            confidence = self._compute_confidence(resolved_cols, tables, filters_pre, agg_info)
            explanation = self._explain(question, resolved_cols, tables, filters_pre, agg_info)
            return {'sql': sql, 'tables_used': tables, 'columns_resolved': resolved_cols,
                    'filters': filters_pre, 'agg_info': agg_info, 'confidence': confidence, 'explanation': explanation}

        # 3f-2: "youngest/oldest members" — ORDER BY DATE_OF_BIRTH
        if re.search(r'\b(?:youngest|oldest)\s+(?:member|patient|person|people)', q_lower):
            if 'members' not in tables:
                tables = ['members']
            order_dir = 'DESC' if 'youngest' in q_lower else 'ASC'
            limit = agg_info.get('top_n') or 10
            sql = f"SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER, KP_REGION FROM members ORDER BY DATE_OF_BIRTH {order_dir} LIMIT {limit};"
            return {'sql': sql, 'tables_used': ['members'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.9,
                    'explanation': f"Retrieving the {'youngest' if 'youngest' in q_lower else 'oldest'} members by DATE_OF_BIRTH."}

        # 3f-3: "busiest facilities/departments" — COUNT + GROUP BY + ORDER BY DESC
        busiest_m = re.search(r'\b(?:busiest|most active|highest volume)\s+(\w+)', q_lower)
        if busiest_m:
            entity = busiest_m.group(1).upper()
            col_map = {'FACILITIES': 'FACILITY', 'FACILITY': 'FACILITY', 'DEPARTMENTS': 'DEPARTMENT',
                       'DEPARTMENT': 'DEPARTMENT', 'PROVIDERS': 'NPI', 'REGIONS': 'KP_REGION'}
            col = col_map.get(entity, entity.rstrip('S'))
            table = 'encounters' if col in ('FACILITY', 'DEPARTMENT') else 'claims'
            limit = agg_info.get('top_n') or 10
            sql = f"SELECT {col}, COUNT(*) as total_count FROM {table} GROUP BY {col} ORDER BY total_count DESC LIMIT {limit};"
            return {'sql': sql, 'tables_used': [table], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.9,
                    'explanation': f"Counting records per {col} to find the busiest."}

        # 3f-4: "members with multiple encounters/claims" — GROUP BY MEMBER_ID HAVING COUNT > 1
        multiple_m = re.search(r'\b(?:member|patient)s?\s+with\s+(?:multiple|several|many|more than \d+)\s+(\w+)', q_lower)
        if multiple_m:
            entity = multiple_m.group(1).lower()
            fact_table = 'encounters' if 'encounter' in entity or 'visit' in entity else 'claims'
            threshold_m = re.search(r'more than\s+(\d+)', q_lower)
            threshold = int(threshold_m.group(1)) if threshold_m else 1
            sql = (f"SELECT m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, COUNT(*) as record_count "
                   f"FROM members m JOIN {fact_table} f ON m.MEMBER_ID = f.MEMBER_ID "
                   f"GROUP BY m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME "
                   f"HAVING COUNT(*) > {threshold} ORDER BY record_count DESC LIMIT 50;")
            return {'sql': sql, 'tables_used': ['members', fact_table], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.9,
                    'explanation': f"Finding members with more than {threshold} {entity}."}

        # 3f-5: "claims for member XXXXX" / "encounters for member XXXXX" — extract specific ID
        id_m = re.search(r'\b(?:for|of|from)\s+(?:member|patient)\s+(\d+)\b', q_lower)
        if not id_m:
            id_m = re.search(r'\bmember\s+(?:id\s+)?(\d{3,})\b', q_lower)
        if id_m:
            member_id = id_m.group(1)
            primary_table = tables[0] if tables else 'claims'
            # Build with the member filter injected
            filters_pre, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)
            member_cond = f"MEMBER_ID = '{member_id}'"
            if not any(member_id in f for f in filters_pre):
                filters_pre.append(member_cond)
            domain_tables = getattr(FilterExtractor, '_domain_tables_needed', [])
            for dt in domain_tables:
                if dt not in tables:
                    tables.append(dt)
            sql = self.builder.build(question, resolved_cols, filters_pre, agg_info, tables, having_clause)
            confidence = self._compute_confidence(resolved_cols, tables, filters_pre, agg_info)
            explanation = self._explain(question, resolved_cols, tables, filters_pre, agg_info)
            return {'sql': sql, 'tables_used': tables, 'columns_resolved': resolved_cols,
                    'filters': filters_pre, 'agg_info': agg_info, 'confidence': confidence, 'explanation': explanation}

        # 3f-6: "total claims amount this year" → SUM(BILLED_AMOUNT) not COUNT
        # Fix: when "total" + "amount/cost/billed/paid" → force SUM
        if agg_info.get('agg_func') == 'COUNT' and any(w in q_lower for w in ['amount', 'cost', 'billed', 'paid', 'spend']):
            if 'total' in q_lower:
                agg_info['agg_func'] = 'SUM'

        # 3f-7: "average billed vs paid amount" → ensure both columns are selected
        # This is a comparison pattern handled by cost_comparison for visit types,
        # but for column comparisons we need both AVG columns
        if agg_info.get('agg_func') == 'AVG' and 'vs' in q_lower:
            amount_cols = []
            for w, col in [('billed', 'BILLED_AMOUNT'), ('paid', 'PAID_AMOUNT'), ('allowed', 'ALLOWED_AMOUNT')]:
                if w in q_lower:
                    amount_cols.append(col)
            if len(amount_cols) >= 2:
                select_parts = ', '.join(f"AVG(CAST({c} AS REAL)) as avg_{c.lower()}" for c in amount_cols)
                table = 'claims'
                filters_pre, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)
                where = f" WHERE {' AND '.join(filters_pre)}" if filters_pre else ""
                sql = f"SELECT {select_parts} FROM {table}{where};"
                return {'sql': sql, 'tables_used': [table], 'columns_resolved': resolved_cols,
                        'filters': filters_pre, 'agg_info': agg_info, 'confidence': 0.9,
                        'explanation': f"Comparing averages of {' vs '.join(amount_cols)}."}

        # 3f-8: "most common diagnoses" → GROUP BY ICD10_CODE + COUNT + ORDER BY DESC
        if re.search(r'\bmost common\s+(?:diagnos|icd)', q_lower):
            sql = "SELECT ICD10_CODE, COUNT(*) as frequency FROM diagnoses GROUP BY ICD10_CODE ORDER BY frequency DESC LIMIT 10;"
            return {'sql': sql, 'tables_used': ['diagnoses'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['ICD10_CODE'], 'top_n': 10, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': "Counting diagnoses by ICD10 code and ranking by frequency."}

        # 3f-9: "encounters with length of stay over N" → WHERE LENGTH_OF_STAY > N
        los_m = re.search(r'\blength of stay\s+(?:over|above|greater than|more than|exceeding)\s+(\d+)', q_lower)
        if los_m:
            los_val = los_m.group(1)
            filters_pre, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)
            los_cond = f"CAST(LENGTH_OF_STAY AS REAL) > {los_val}"
            if not any('LENGTH_OF_STAY' in f for f in filters_pre):
                filters_pre.append(los_cond)
            if 'encounters' not in tables:
                tables = ['encounters'] + tables
            domain_tables = getattr(FilterExtractor, '_domain_tables_needed', [])
            for dt in domain_tables:
                if dt not in tables:
                    tables.append(dt)
            sql = self.builder.build(question, resolved_cols, filters_pre, agg_info, tables, having_clause)
            confidence = self._compute_confidence(resolved_cols, tables, filters_pre, agg_info)
            explanation = self._explain(question, resolved_cols, tables, filters_pre, agg_info)
            return {'sql': sql, 'tables_used': tables, 'columns_resolved': resolved_cols,
                    'filters': filters_pre, 'agg_info': agg_info, 'confidence': confidence, 'explanation': explanation}

        # 3f-10: "HCC categories count" → GROUP BY HCC_CATEGORY + COUNT
        if re.search(r'\bhcc\b', q_lower) and ('count' in q_lower or 'categor' in q_lower):
            sql = "SELECT HCC_CATEGORY, COUNT(*) as count FROM diagnoses WHERE HCC_CATEGORY IS NOT NULL AND HCC_CATEGORY != '' GROUP BY HCC_CATEGORY ORDER BY count DESC;"
            return {'sql': sql, 'tables_used': ['diagnoses'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['HCC_CATEGORY'], 'top_n': None, 'order': 'DESC'},
                    'confidence': 0.9, 'explanation': "Counting diagnoses by HCC category."}

        # 3f-11: "upcoming pcp appointments for [location]" → appointments table
        if re.search(r'\bappointment|pcp\b.*\bappointment|upcoming\b.*\bpcp\b', q_lower):
            region_map = {
                'baltimore': 'MAS', 'maryland': 'MAS', 'mid-atlantic': 'MAS', 'virginia': 'MAS', 'dc': 'MAS',
                'oakland': 'NCAL', 'san francisco': 'NCAL', 'santa clara': 'NCAL', 'ncal': 'NCAL', 'northern california': 'NCAL',
                'los angeles': 'SCAL', 'san diego': 'SCAL', 'fontana': 'SCAL', 'scal': 'SCAL', 'southern california': 'SCAL',
                'portland': 'NW', 'seattle': 'NW', 'northwest': 'NW',
                'denver': 'CO', 'aurora': 'CO', 'colorado': 'CO',
                'atlanta': 'GA', 'georgia': 'GA',
                'honolulu': 'HI', 'hawaii': 'HI',
                'kansas city': 'MID', 'st. louis': 'MID', 'midwest': 'MID',
            }
            region_filter = ''
            for loc, reg in region_map.items():
                if loc in q_lower:
                    region_filter = f" AND a.KP_REGION = '{reg}'"
                    break
            pcp_filter = ''
            if 'pcp' in q_lower or 'primary care' in q_lower:
                pcp_filter = " AND a.IS_PCP_VISIT = 'Y'"
            upcoming = " AND a.APPOINTMENT_DATE >= date('now')" if 'upcoming' in q_lower or 'future' in q_lower or 'next' in q_lower else ''
            sql = f"""SELECT a.APPOINTMENT_DATE, a.APPOINTMENT_TIME, a.APPOINTMENT_TYPE, a.DEPARTMENT, a.FACILITY, a.KP_REGION, a.STATUS, a.REASON, m.FIRST_NAME || ' ' || m.LAST_NAME as MEMBER_NAME, p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as PROVIDER_NAME
FROM appointments a
LEFT JOIN members m ON a.MEMBER_ID = m.MEMBER_ID
LEFT JOIN providers p ON a.PROVIDER_NPI = p.NPI
WHERE a.STATUS = 'SCHEDULED'{pcp_filter}{region_filter}{upcoming}
ORDER BY a.APPOINTMENT_DATE LIMIT 50;"""
            return {'sql': sql, 'tables_used': ['appointments', 'members', 'providers'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.95,
                    'explanation': f"Showing upcoming PCP appointments{' for ' + region_filter.split('=')[1].strip().strip(chr(39)) if region_filter else ''}."}

        # 3f-12: "most occurring/common cpt code [in YEAR]" → claims GROUP BY CPT_CODE (MUST be before description lookup)
        if re.search(r'\bcpt\b', q_lower) and re.search(r'\bmost\b.*\b(common|frequent|occurring|popular)\b|\btop\b.*\bcpt\b|\b(common|frequent|occurring|popular)\b.*\bcpt\b', q_lower):
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" WHERE SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY, COUNT(*) as occurrence_count
FROM claims c LEFT JOIN cpt_codes cpt ON c.CPT_CODE = cpt.CPT_CODE{year_filter}
GROUP BY c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY ORDER BY occurrence_count DESC LIMIT 20;"""
            return {'sql': sql, 'tables_used': ['claims', 'cpt_codes'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['CPT_CODE'], 'top_n': 20, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': f"Top CPT codes by frequency{' in ' + year_match.group(1) if year_match else ''}."}

        # 3f-13: "description of cpt code XXXXX" or "what is cpt code" → cpt_codes table
        _cpt_match = re.search(r'\bcpt\b.*?\bcode\b\s*(\d{4,5})?', q_lower) or re.search(r'\bcpt\b\s*(\d{4,5})', q_lower)
        if _cpt_match and ('description' in q_lower or 'what is' in q_lower or 'describe' in q_lower or 'meaning' in q_lower):
            cpt_val = _cpt_match.group(1)
            if cpt_val:
                sql = f"SELECT CPT_CODE, DESCRIPTION, CATEGORY, RVU FROM cpt_codes WHERE CPT_CODE = '{cpt_val}';"
                expl = f"Looking up CPT code {cpt_val} description."
            else:
                sql = "SELECT CPT_CODE, DESCRIPTION, CATEGORY, RVU FROM cpt_codes ORDER BY CPT_CODE LIMIT 50;"
                expl = "Listing all available CPT codes with descriptions."
            return {'sql': sql, 'tables_used': ['cpt_codes'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.95, 'explanation': expl}

        # 3f-14: "breakdown of claims per claim categories/types [in YEAR]" → GROUP BY CLAIM_TYPE
        if re.search(r'\b(breakdown|distribution|split)\b.*\b(claim|claims)\b.*\b(categor|type|class)', q_lower) or \
           re.search(r'\bclaim\b.*\b(categor|type)\b.*\b(breakdown|distribution|split)\b', q_lower):
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" WHERE SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT CLAIM_TYPE, COUNT(*) as claim_count, ROUND(SUM(BILLED_AMOUNT),2) as total_billed, ROUND(SUM(PAID_AMOUNT),2) as total_paid, ROUND(AVG(BILLED_AMOUNT),2) as avg_billed
FROM claims{year_filter}
GROUP BY CLAIM_TYPE ORDER BY claim_count DESC;"""
            return {'sql': sql, 'tables_used': ['claims'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['CLAIM_TYPE'], 'top_n': None, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': f"Claims breakdown by category/type{' in ' + year_match.group(1) if year_match else ''}."}

        # 3f-15: "processing time" → claims SUBMITTED_DATE and ADJUDICATED_DATE
        if re.search(r'\bprocessing\s*time\b|\badjudication\s*time\b|\btime\s*to\s*(process|adjudicate)\b', q_lower):
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" AND SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT CLAIM_TYPE, COUNT(*) as claim_count, ROUND(AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as avg_processing_days,
ROUND(MIN(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as min_days, ROUND(MAX(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as max_days
FROM claims WHERE SUBMITTED_DATE IS NOT NULL AND ADJUDICATED_DATE IS NOT NULL{year_filter}
GROUP BY CLAIM_TYPE ORDER BY avg_processing_days DESC;"""
            return {'sql': sql, 'tables_used': ['claims'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'AVG', 'group_by_terms': ['CLAIM_TYPE'], 'top_n': None, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': "Average claim processing time (submitted to adjudicated) by claim type."}

        # 3f-16: "claims and physician names for [specialty] in [area]" → claims JOIN providers (BEFORE provider-only)
        if ('claim' in q_lower) and re.search(r'\b(physician|doctor|provider)', q_lower):
            spec_filter = ''
            if any(w in q_lower for w in ['general', 'primary care', 'pcp', 'family', 'internal medicine']):
                spec_filter = " AND (p.SPECIALTY IN ('Internal Medicine','Family Medicine','General Practice'))"
            region_map = {'baltimore': 'MAS', 'maryland': 'MAS', 'ma ': 'MAS', 'mas': 'MAS', 'ncal': 'NCAL', 'scal': 'SCAL', 'nw': 'NW', 'co': 'CO', 'ga': 'GA', 'hi': 'HI'}
            region_filter = ''
            for loc, reg in region_map.items():
                if loc in q_lower:
                    region_filter = f" AND c.KP_REGION = '{reg}'"
                    break
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" AND c.SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT c.CLAIM_ID, c.SERVICE_DATE, c.CLAIM_TYPE, c.BILLED_AMOUNT, c.PAID_AMOUNT, c.CLAIM_STATUS,
p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as PHYSICIAN_NAME, p.SPECIALTY, p.FACILITY, c.KP_REGION
FROM claims c LEFT JOIN providers p ON c.RENDERING_NPI = p.NPI
WHERE 1=1{spec_filter}{region_filter}{year_filter}
ORDER BY c.SERVICE_DATE DESC LIMIT 50;"""
            return {'sql': sql, 'tables_used': ['claims', 'providers'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.95,
                    'explanation': "Claims with physician names."}

        # 3f-17: "general physicians in [location]" / "physicians in [area]" → providers table (provider-only, no claims)
        if re.search(r'\b(physician|doctor|provider|pcp)', q_lower) and re.search(r'\b(in|at|from|near|list|show|find)\b', q_lower):
            spec_filter = ''
            if any(w in q_lower for w in ['general', 'primary care', 'pcp', 'family', 'internal medicine']):
                spec_filter = " AND (SPECIALTY IN ('Internal Medicine','Family Medicine','General Practice') OR DEPARTMENT LIKE '%Primary%')"
            elif 'cardiolog' in q_lower:
                spec_filter = " AND SPECIALTY = 'Cardiology'"
            elif 'dermatolog' in q_lower:
                spec_filter = " AND SPECIALTY = 'Dermatology'"
            elif 'orthop' in q_lower:
                spec_filter = " AND SPECIALTY = 'Orthopedics'"
            region_map = {
                'baltimore': 'MAS', 'maryland': 'MAS', 'mid-atlantic': 'MAS', 'ma ': 'MAS', 'mas': 'MAS', 'virginia': 'MAS',
                'ncal': 'NCAL', 'northern california': 'NCAL', 'oakland': 'NCAL', 'san francisco': 'NCAL',
                'scal': 'SCAL', 'southern california': 'SCAL', 'los angeles': 'SCAL',
                'portland': 'NW', 'seattle': 'NW', 'northwest': 'NW',
                'denver': 'CO', 'colorado': 'CO',
                'atlanta': 'GA', 'georgia': 'GA',
                'honolulu': 'HI', 'hawaii': 'HI',
            }
            region_filter = ''
            for loc, reg in region_map.items():
                if loc in q_lower:
                    region_filter = f" AND KP_REGION = '{reg}'"
                    break
            sql = f"""SELECT NPI, PROVIDER_FIRST_NAME || ' ' || PROVIDER_LAST_NAME as PROVIDER_NAME, SPECIALTY, DEPARTMENT, FACILITY, KP_REGION, STATUS, PANEL_SIZE, ACCEPTS_NEW_PATIENTS
FROM providers WHERE STATUS = 'ACTIVE'{spec_filter}{region_filter}
ORDER BY PROVIDER_LAST_NAME LIMIT 50;"""
            return {'sql': sql, 'tables_used': ['providers'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.95,
                    'explanation': f"Listing providers{' with specialty filter' if spec_filter else ''}{' in ' + region_filter.split('=')[1].strip().strip(chr(39)) if region_filter else ''}."}

        # 3f-18: "covid hospitalization in [region] for [year]" → encounters with U07.1
        if re.search(r'\bcovid\b', q_lower) and re.search(r'\b(hospital|admission|inpatient|encounter|visit)', q_lower):
            region_map = {'ncal': 'NCAL', 'northern california': 'NCAL', 'scal': 'SCAL', 'southern california': 'SCAL',
                          'mas': 'MAS', 'baltimore': 'MAS', 'maryland': 'MAS', 'nw': 'NW', 'co': 'CO', 'ga': 'GA', 'hi': 'HI', 'mid': 'MID'}
            region_filter = ''
            for loc, reg in region_map.items():
                if loc in q_lower:
                    region_filter = f" AND e.KP_REGION = '{reg}'"
                    break
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" AND e.SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT e.ENCOUNTER_ID, e.SERVICE_DATE, e.VISIT_TYPE, e.DEPARTMENT, e.LENGTH_OF_STAY, e.DISPOSITION, e.FACILITY, e.KP_REGION,
m.FIRST_NAME || ' ' || m.LAST_NAME as PATIENT_NAME, m.DATE_OF_BIRTH, m.GENDER
FROM encounters e LEFT JOIN members m ON e.MEMBER_ID = m.MEMBER_ID
WHERE e.PRIMARY_DIAGNOSIS = 'U07.1'{region_filter}{year_filter}
ORDER BY e.SERVICE_DATE DESC LIMIT 50;"""
            return {'sql': sql, 'tables_used': ['encounters', 'members'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': agg_info, 'confidence': 0.95,
                    'explanation': f"COVID-19 hospitalizations{' in ' + region_filter.split('=')[1].strip().strip(chr(39)) if region_filter else ''}{' for ' + year_match.group(1) if year_match else ''}."}

        # 3f-19: "breakdown of claim paid denied adjusted [in YEAR]" → claims GROUP BY CLAIM_STATUS
        if re.search(r'\b(breakdown|distribution|split|ratio)\b', q_lower) and re.search(r'\b(paid|denied|adjusted|status)\b', q_lower) and 'claim' in q_lower:
            year_match = re.search(r'\b(202[0-9])\b', q_lower)
            year_filter = f" WHERE SERVICE_DATE LIKE '{year_match.group(1)}%'" if year_match else ''
            sql = f"""SELECT CLAIM_STATUS, COUNT(*) as claim_count, ROUND(SUM(BILLED_AMOUNT),2) as total_billed, ROUND(SUM(PAID_AMOUNT),2) as total_paid,
ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims{year_filter}), 1) as percentage
FROM claims{year_filter}
GROUP BY CLAIM_STATUS ORDER BY claim_count DESC;"""
            return {'sql': sql, 'tables_used': ['claims'], 'columns_resolved': resolved_cols,
                    'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['CLAIM_STATUS'], 'top_n': None, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': f"Claim status breakdown (paid/denied/adjusted){' in ' + year_match.group(1) if year_match else ''}."}

        # Step 4: Extract filters and having clause (Bug 1)
        filters, having_clause = FilterExtractor.extract(question, resolved_cols, self.schema)

        # Step 4-values: Auto-generate filters from value mentions in the question
        auto_filters = ValueResolver.get_auto_filters(question, tables)
        for af in auto_filters:
            # Don't duplicate filters that FilterExtractor already found
            if not any(af.upper() in existing.upper() for existing in filters):
                filters.append(af)

        # Step 4-domain: Add tables required by domain concept filters
        domain_tables = getattr(FilterExtractor, '_domain_tables_needed', [])
        for dt in domain_tables:
            if dt not in tables:
                tables.append(dt)

        # Step 4a-having: If HAVING clause detected but no agg_func, force COUNT mode
        # "members with more than 5 claims" → needs GROUP BY member_id + COUNT + HAVING
        if having_clause and not agg_info.get('agg_func'):
            agg_info['agg_func'] = 'COUNT'
        # Infer group entity from the subject when HAVING clause present and no group_by_terms
        if having_clause and not agg_info.get('group_by_terms'):
            q_lower = question.lower()
            HAVING_ENTITY_MAP = {
                'member': 'MEMBER_ID', 'patient': 'MEMBER_ID',
                'provider': 'NPI', 'doctor': 'NPI',
                'facility': 'FACILITY', 'department': 'DEPARTMENT',
                'region': 'KP_REGION', 'specialty': 'SPECIALTY',
            }
            # Get the part before "with/having" to find the subject entity
            subject_part = q_lower.split(' with ')[0] if ' with ' in q_lower else q_lower.split(' having ')[0] if ' having ' in q_lower else q_lower
            for entity, col in HAVING_ENTITY_MAP.items():
                # Handle plurals: facility→facilities, member→members, etc.
                plural_forms = [entity, entity + 's']
                if entity.endswith('y'):
                    plural_forms.append(entity[:-1] + 'ies')  # facility → facilities
                if any(re.search(r'\b' + pf + r'\b', subject_part) for pf in plural_forms):
                    agg_info['group_by_terms'] = [entity]
                    break

        # Step 4a-fix: If aggregating on a dimension table, add a fact table
        # e.g., "top 10 providers by volume" → providers needs claims to count
        # BUT: "count of medicaid members by region" should NOT add claims —
        # we're counting the members themselves, not a secondary entity
        DIMENSION_TABLES = {'providers', 'members'}
        FACT_TABLE_FOR = {
            'providers': 'claims',    # Count claims per provider
            'members': 'claims',      # Count claims per member
        }
        agg_func = agg_info.get('agg_func')
        if agg_func and len(tables) == 1 and tables[0] in DIMENSION_TABLES:
            q_lower = question.lower()
            dim_table = tables[0]
            # Check if the question is counting the dimension entity itself
            # e.g., "count of members" vs "claims per member"
            counting_self = False
            entity_words = {'member': 'members', 'provider': 'providers', 'patient': 'members'}
            for ew, tbl in entity_words.items():
                if tbl == dim_table and (f'count of {ew}' in q_lower or f'many {ew}' in q_lower
                                          or f'number of {ew}' in q_lower or f'total {ew}' in q_lower
                                          or q_lower.startswith(f'{ew}') or f'count {ew}' in q_lower):
                    counting_self = True
                    break
            # Also check if domain concepts already added filters — that's a sign we're counting members
            if domain_tables and dim_table in domain_tables:
                counting_self = True
            if not counting_self:
                fact_table = FACT_TABLE_FOR.get(dim_table)
                if fact_table and fact_table not in tables:
                    tables.append(fact_table)

        # Step 4b: Ensure group-by columns' tables are included (Bug 6)
        # But only add a NEW table if the group column isn't already in existing tables
        group_terms = agg_info.get('group_by_terms', [])
        for term in group_terms:
            term_lower = term.lower()
            # Check if any existing table already has a matching column
            already_covered = False
            for existing_table in tables:
                for col in self.schema.get_columns(existing_table):
                    if term_lower in col.lower() or col.lower() in term_lower:
                        already_covered = True
                        break
                if already_covered:
                    break
            # Only search for new tables if the column isn't in any existing table
            if not already_covered:
                for table_name in self.schema.tables.keys():
                    if table_name in tables:
                        continue
                    for col in self.schema.get_columns(table_name):
                        if term_lower in col.lower() or col.lower() in term_lower:
                            tables.append(table_name)
                            break
                    else:
                        continue
                    break  # Only add the first matching table

        # Step 4c: Prune unnecessary secondary tables to avoid join-inflated counts
        # Only prune a table if ALL these conditions are true:
        # 1) It has no unique columns needed by filters or domain concepts
        # 2) All its relevant columns (ICD10, KP_REGION, etc.) exist in the primary table
        # 3) It was added only by keyword similarity, not by explicit query reference
        if len(tables) > 1:
            primary = tables[0]
            primary_cols = set(c.upper() for c in self.schema.get_columns(primary))
            q_lower = question.lower()

            # Collect columns referenced by filters (including domain concept filters)
            filter_cols = set()
            for f in filters:
                for col_name in self.schema.column_to_tables.keys():
                    if col_name.upper() in f.upper():
                        filter_cols.add(col_name.upper())

            # Collect columns needed by GROUP BY terms
            # Use strict matching: term must match column name closely
            # e.g., "specialty" → SPECIALTY, "visit type" → VISIT_TYPE
            # But "member" should NOT match MEMBER_RESPONSIBILITY
            group_col_names = set()
            for term in group_terms:
                term_upper = term.upper().replace(' ', '_')
                for col_name in self.schema.column_to_tables.keys():
                    col_upper = col_name.upper()
                    # Exact match or term matches the column's base name
                    if col_upper == term_upper or col_upper == term_upper + '_ID':
                        group_col_names.add(col_upper)
                    elif '_' in col_upper:
                        # Multi-word column: check if term matches a meaningful segment
                        # e.g., "specialty" matches "SPECIALTY", "visit" matches "VISIT_TYPE"
                        parts = col_upper.split('_')
                        if term_upper in parts or term_upper == '_'.join(parts[:-1]):
                            group_col_names.add(col_upper)

            pruned = [primary]
            for t in tables[1:]:
                t_cols = set(c.upper() for c in self.schema.get_columns(t))
                unique_cols = t_cols - primary_cols

                # Keep this table if:
                keep = False
                # a) Domain concepts require it
                if t in domain_tables:
                    keep = True
                # b) Filters reference columns unique to this table
                elif filter_cols & unique_cols:
                    keep = True
                # f) GROUP BY columns exist on this table but not primary
                elif group_col_names & unique_cols:
                    keep = True
                # c) Special query patterns that need specific tables
                elif 'denial' in q_lower and t == 'claims':
                    keep = True  # denial rate needs CLAIM_STATUS from claims
                elif any(w in q_lower for w in ['billed', 'paid', 'allowed', 'cost', 'amount', 'spend']) and t == 'claims':
                    # Don't force claims for "cost" when prescriptions is already in the table list
                    # (prescriptions has its own COST column)
                    if not ('medication' in q_lower and 'prescriptions' in tables):
                        keep = True  # financial queries need claims
                # d) The question explicitly mentions the table's entity
                # Handle plural forms: diagnoses→diagnosis, encounters→encounter, etc.
                elif t in q_lower or t.rstrip('s') in q_lower:
                    keep = True
                elif t.endswith('es') and t[:-2] + 'is' in q_lower:
                    keep = True  # diagnoses → diagnosis
                elif t.endswith('s') and t[:-1] in q_lower:
                    keep = True  # claims → claim
                # g) Table's keywords appear in the question (it was selected by resolver for a reason)
                elif any(kw in q_lower for kw in TableResolver.TABLE_KEYWORDS.get(t, [])):
                    keep = True
                # e) HAVING clause present — keep fact tables for counting
                elif having_clause and t in ('claims', 'encounters', 'prescriptions', 'referrals'):
                    keep = True

                if keep:
                    pruned.append(t)
            tables = pruned

        # Step 5: Build SQL
        sql = self.builder.build(question, resolved_cols, filters, agg_info, tables, having_clause)

        # Step 6: Compute confidence
        confidence = self._compute_confidence(resolved_cols, tables, filters, agg_info)

        # Step 7: Generate explanation
        explanation = self._explain(question, resolved_cols, tables, filters, agg_info)

        return {
            'sql': sql,
            'tables_used': tables,
            'columns_resolved': resolved_cols,
            'filters': filters,
            'agg_info': agg_info,
            'confidence': confidence,
            'explanation': explanation,
        }

    def _compute_confidence(self, resolved_cols, tables, filters, agg_info) -> float:
        """Estimate confidence in the generated SQL."""
        score = 0.3  # Base

        if resolved_cols:
            score += min(0.3, 0.1 * len(resolved_cols))
            # Bonus for exact matches
            exact = sum(1 for rc in resolved_cols if rc['match_type'] == 'exact')
            score += min(0.1, 0.05 * exact)

        if tables:
            score += 0.1

        if filters:
            score += min(0.1, 0.05 * len(filters))

        if agg_info.get('agg_func'):
            score += 0.1

        return min(1.0, score)

    def _explain(self, question, resolved_cols, tables, filters, agg_info) -> str:
        """
        Generate a detailed, structured SQL explanation covering:
        1. What the question is asking (interpretation)
        2. Why specific tables were chosen
        3. Why JOINs are needed (or not)
        4. How the query ensures correct results
        5. What each major clause does
        """
        lines = []

        # ------ 1. INTERPRETATION ------
        agg_func = agg_info.get('agg_func')
        group_terms = agg_info.get('group_by_terms', [])
        top_n = agg_info.get('top_n')

        if agg_func == 'COUNT':
            action = "counting records"
        elif agg_func == 'AVG':
            action = "calculating the average"
        elif agg_func == 'SUM':
            action = "summing up totals"
        elif agg_func == 'MAX':
            action = "finding the maximum value"
        elif agg_func == 'MIN':
            action = "finding the minimum value"
        elif top_n:
            action = f"ranking the top {top_n}"
        else:
            action = "retrieving records"

        lines.append(f"<b>Interpretation:</b> You asked \"{question}\" — so we are {action}.")

        # ------ 2. TABLE SELECTION ------
        TABLE_PURPOSES = {
            'claims': 'billing & payment data (billed amounts, paid amounts, denial status, CPT codes)',
            'members': 'patient demographics (name, age, gender, enrollment, risk scores, chronic conditions)',
            'providers': 'physician/facility info (NPI, specialty, department, panel size)',
            'encounters': 'visit records (admission, discharge, visit type, length of stay, chief complaint)',
            'diagnoses': 'clinical diagnoses (ICD-10 codes, HCC categories, severity, chronic flags)',
            'prescriptions': 'medication data (drug names, classes, costs, refills, pharmacy)',
            'referrals': 'referral records (referring/referred providers, urgency, authorization)',
        }
        if len(tables) == 1:
            why = TABLE_PURPOSES.get(tables[0], 'relevant data')
            lines.append(f"<b>Table chosen:</b> <code>{tables[0]}</code> — this table holds {why}. "
                        f"No other tables are needed because all requested data lives here.")
        else:
            table_reasons = []
            for t in tables:
                why = TABLE_PURPOSES.get(t, 'supporting data')
                table_reasons.append(f"<code>{t}</code> for {why}")
            lines.append(f"<b>Tables chosen ({len(tables)}):</b> {'; '.join(table_reasons)}.")

            # Explain JOINs
            join_path = self.schema.find_join_path(tables)
            if join_path:
                join_descs = []
                for t1, t2, col in join_path:
                    if '=' in col:
                        left, right = col.split('=', 1)
                        join_descs.append(f"<code>{t1}.{left}</code> = <code>{t2}.{right}</code>")
                    else:
                        join_descs.append(f"<code>{t1}.{col}</code> = <code>{t2}.{col}</code>")
                lines.append(f"<b>JOINs:</b> Tables are linked via {', '.join(join_descs)}. "
                            f"This ensures every row in the result connects related records across tables — "
                            f"no orphaned or duplicate data.")

        # ------ 3. COLUMN MAPPING ------
        if resolved_cols:
            # Show how user's words mapped to actual columns
            mappings = []
            seen = set()
            for rc in resolved_cols[:6]:
                key = rc['column']
                if key in seen:
                    continue
                seen.add(key)
                term = rc.get('original_term', '')
                mt = rc['match_type']
                if mt == 'synonym' and term:
                    mappings.append(f"\"{term}\" → <code>{rc['table']}.{rc['column']}</code>")
                elif mt == 'exact':
                    mappings.append(f"<code>{rc['column']}</code> (exact match in <code>{rc['table']}</code>)")
            if mappings:
                lines.append(f"<b>Column mapping:</b> {'; '.join(mappings)}.")

        # ------ 4. FILTERS ------
        if filters:
            filter_descs = []
            for f in filters:
                if 'DENIED' in f:
                    filter_descs.append("only DENIED claims (filtering out Paid, Pending, etc.)")
                elif 'PAID' in f and '=' in f:
                    filter_descs.append("only PAID/approved claims")
                elif "> " in f:
                    filter_descs.append(f"values above the threshold ({f.split('> ')[-1].rstrip(')')})")
                elif "< " in f:
                    filter_descs.append(f"values below the threshold ({f.split('< ')[-1].rstrip(')')})")
                elif "date(" in f.lower():
                    filter_descs.append("restricted to the specified time window")
                elif "LIKE" in f:
                    filter_descs.append(f"matching the pattern {f.split('LIKE ')[-1]}")
                elif "IS_CHRONIC" in f:
                    filter_descs.append("only patients with chronic conditions")
                elif "VISIT_TYPE" in f:
                    vt = f.split("'")[-2] if "'" in f else "specified"
                    filter_descs.append(f"only {vt} visits")
                else:
                    filter_descs.append(f"{f}")
            lines.append(f"<b>Filters applied:</b> {'; '.join(filter_descs)}. "
                        f"This narrows the dataset to exactly what you asked about.")

        # ------ 5. AGGREGATION / GROUPING ------
        if agg_func and group_terms:
            lines.append(f"<b>Aggregation:</b> Computing {agg_func} and grouping by "
                        f"{', '.join(group_terms)}. Each row in the result represents one "
                        f"distinct {'/'.join(group_terms)} value with its computed metric.")
        elif agg_func:
            lines.append(f"<b>Aggregation:</b> Computing a single {agg_func} across all matching records.")

        if top_n:
            order_word = "highest" if agg_info.get('order') == 'DESC' else "lowest"
            lines.append(f"<b>Ranking:</b> Results are sorted and limited to the {order_word} {top_n}.")

        # ------ 6. CORRECTNESS GUARANTEE ------
        guarantees = []
        if agg_func:
            guarantees.append("aggregation operates on all matching rows, not a sample")
        if filters:
            guarantees.append("WHERE clause precisely filters before aggregation")
        if len(tables) > 1:
            guarantees.append("JOINs use high-specificity keys (IDs, not region/facility) to avoid row duplication")
        if not filters and not agg_func:
            guarantees.append("LIMIT clause prevents overwhelming results while showing a representative sample")
        if guarantees:
            lines.append(f"<b>Correctness:</b> {'; '.join(guarantees)}.")

        return "<br>".join(lines)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == '__main__':
    engine = DynamicSQLEngine()

    test_questions = [
        "give me initiative start dates",
        "show me denied claims",
        "average paid amount by region",
        "how many claims",
        "top 10 providers by volume",
        "show me members who enrolled last year",
        "what medications are prescribed for patients with chronic conditions",
        "average length of stay by facility",
        "denied claims over $5000 in the last 90 days",
        "which providers have the highest denial rate",
        "compare billed vs paid amounts by region",
        "show me emergency visits with high severity diagnoses",
        "how many referrals per specialty last quarter",
        "patient journey from admission to discharge",
        "top 5 most prescribed medications by cost",
        "members with risk score above 3",
        "claims trend over time by status",
    ]

    for q in test_questions:
        result = engine.generate(q)
        print(f"\nQ: {q}")
        print(f"   SQL: {result['sql']}")
        print(f"   Tables: {result['tables_used']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Explanation: {result['explanation']}")
