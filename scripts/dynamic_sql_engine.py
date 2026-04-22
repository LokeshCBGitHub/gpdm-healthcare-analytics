import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[len(b)]


_CRITICAL_WORDS = {
    'members': (6, 2), 'member': (5, 2), 'patients': (7, 2), 'patient': (6, 2),
    'providers': (8, 2), 'provider': (7, 2), 'encounters': (9, 2), 'encounter': (8, 2),
    'claims': (5, 2), 'claim': (4, 1), 'diagnoses': (8, 2), 'diagnosis': (8, 2),
    'prescriptions': (10, 2), 'prescription': (10, 2), 'referrals': (7, 2), 'referral': (7, 2),
    'people': (5, 1), 'person': (5, 1), 'visited': (6, 2), 'emergency': (8, 2),
    'inpatient': (8, 2), 'outpatient': (9, 2), 'telehealth': (9, 2),
    'department': (9, 2), 'facility': (7, 2), 'facilities': (9, 2),
    'specialty': (8, 2), 'specialties': (10, 2), 'average': (6, 2), 'total': (4, 1),
    'amount': (5, 1), 'billed': (5, 2), 'approved': (7, 2), 'denied': (5, 1),
    'pending': (6, 2), 'cancelled': (8, 2), 'medication': (9, 2), 'chronic': (6, 1),
    'institutional': (12, 2), 'professional': (11, 2), 'pharmacy': (7, 2),
    'elective': (7, 2), 'region': (5, 1), 'gender': (5, 1), 'disposition': (10, 2),
    'urgent': (5, 1),
    'authorization': (13, 2), 'dollar': (5, 1), 'dollars': (6, 1),
    'expensive': (8, 2), 'reimbursement': (13, 2), 'enrollment': (9, 2),
    'utilization': (11, 2), 'readmission': (11, 2), 'demographic': (10, 2),
    'percentage': (10, 2), 'appointment': (10, 2), 'appointments': (11, 2),
    'hypertension': (12, 2), 'diabetes': (7, 2), 'diabetic': (7, 2),
    'depression': (9, 2), 'anxiety': (6, 2), 'obesity': (6, 2),
    'pneumonia': (8, 2), 'asthma': (5, 1), 'copd': (4, 1),
    'schizophrenia': (13, 2), 'alzheimer': (8, 2), 'epilepsy': (7, 2),
    'cirrhosis': (8, 2), 'osteoarthritis': (14, 2), 'osteoporosis': (12, 2),
    'sepsis': (5, 1), 'anemia': (5, 1),
    'cardiology': (10, 2), 'oncology': (8, 2), 'neurology': (9, 2),
    'orthopedics': (11, 2), 'psychiatry': (10, 2), 'dermatology': (11, 2),
    'endocrinology': (13, 2), 'gastroenterology': (16, 2), 'nephrology': (10, 2),
    'pulmonology': (11, 2), 'rheumatology': (12, 2), 'ophthalmology': (13, 2),
    'pediatrics': (10, 2), 'radiology': (9, 2), 'urology': (7, 2),
    'adjudication': (12, 2), 'coinsurance': (11, 2), 'deductible': (10, 2),
    'copayment': (9, 2), 'beneficiary': (11, 2), 'eligibility': (11, 2),
    'formulary': (9, 2), 'adherence': (9, 2), 'polypharmacy': (12, 2),
    'pressure': (8, 2), 'procedure': (9, 2), 'treatment': (9, 2),
    'discharge': (9, 2), 'admission': (9, 2), 'readmission': (11, 2),
    'denial': (6, 2), 'denials': (7, 2), 'revenue': (7, 2),
    'insurance': (9, 2), 'coverage': (8, 2), 'premium': (7, 2),
    'hospital': (8, 2), 'physician': (9, 2), 'clinician': (9, 2),
    'compliance': (10, 2), 'population': (10, 2), 'management': (10, 2),
    'coordination': (12, 2), 'vaccination': (11, 2),
    'performance': (11, 2), 'forecasting': (11, 2), 'dashboard': (9, 2),
    'amnt': (4, 1), 'amt': (3, 1), 'amout': (5, 1),
    'memebers': (8, 2), 'memebrs': (7, 2), 'membrs': (6, 2),
    'specialites': (11, 2), 'speciality': (10, 2), 'specilaity': (10, 2),
    'clam': (4, 1), 'clams': (5, 1), 'calim': (5, 1), 'calims': (6, 1),
    'encunters': (9, 2), 'encountes': (9, 2),
    'presciptions': (12, 2), 'perscriptions': (13, 2),
    'refarrals': (9, 2), 'refferals': (9, 2),
    'diagnosies': (10, 2), 'diagnsis': (8, 2),
    'provders': (8, 2), 'providrs': (8, 2),
}

_NORMALIZER_EXCLUSIONS = {
    'class', 'under', 'over', 'case', 'close', 'clear', 'place', 'since',
    'count', 'cost', 'costs', 'most', 'post', 'lost', 'last', 'list', 'best',
    'plan', 'plant', 'plans', 'plain', 'state', 'states', 'stage', 'stable',
    'panel', 'model', 'level', 'label', 'date', 'data', 'rate', 'rated', 'rates',
    'type', 'types', 'typed', 'code', 'codes', 'coded', 'name', 'named', 'names',
    'care', 'race', 'rare', 'core', 'risk', 'disk', 'stay', 'stat', 'star',
    'view', 'visit', 'visits', 'show', 'shown', 'shows', 'high', 'higher', 'highest',
    'paid', 'pair', 'pain', 'aged', 'ages', 'aging', 'male', 'female', 'scale',
    'many', 'money', 'each', 'year', 'years',
    'this', 'that', 'than', 'then', 'them', 'they', 'their', 'there', 'these',
    'what', 'when', 'where', 'which', 'while', 'with', 'will',
    'from', 'have', 'been', 'more', 'some', 'time', 'like', 'just',
    'give', 'gave', 'gone', 'good', 'does', 'done',
    'test', 'text', 'term', 'terms', 'drug', 'drugs',
    'fill', 'filled', 'first', 'size', 'site', 'side', 'days', 'daily',
    'old', 'new', 'all', 'per', 'for', 'the', 'and', 'not', 'are', 'was', 'how',
    'description', 'descriptions', 'describe', 'detail', 'details', 'detailed',
    'upcoming', 'breakdown', 'breakdowns', 'distribution', 'processing',
    'general', 'specific', 'particular', 'occurring', 'percentage', 'proportion',
    'cad', 'chf', 'ckd', 'aki', 'bph', 'pvd', 'cva', 'tia', 'vte', 'sud',
    'afib', 'gerd', 'esrd', 'ptsd', 'hiv', 'snf', 'asc', 'aco', 'mlr',
    'pmpm', 'alos', 'lwbs', 'dur', 'ndc', 'rvu', 'cob', 'drg', 'mdc',
    'ent', 'hmo', 'ppo', 'epo', 'hdhp', 'raf', 'hcc', 'ccsr',
    'mild', 'trend', 'across', 'compare', 'volume', 'changed', 'impact',
    'affect', 'number', 'prior', 'order', 'admit', 'reason', 'result',
    'month', 'quarter', 'annual', 'report', 'group', 'score', 'tier',
    'brand', 'generic', 'supply', 'dose', 'refill', 'audit', 'measure',
}


_DIRECT_TYPO_MAP = {
    'amnt': 'amount', 'amt': 'amount', 'amout': 'amount',
    'memebers': 'members', 'memebrs': 'members', 'membrs': 'members',
    'specialites': 'specialties', 'specilaity': 'specialty', 'speciality': 'specialty',
    'clam': 'claim', 'clams': 'claims', 'calim': 'claim', 'calims': 'claims',
    'encunters': 'encounters', 'encountes': 'encounters',
    'presciptions': 'prescriptions', 'perscriptions': 'prescriptions',
    'refarrals': 'referrals', 'refferals': 'referrals',
    'diagnosies': 'diagnoses', 'diagnsis': 'diagnosis',
    'provders': 'providers', 'providrs': 'providers',
    'pateints': 'patients', 'patiens': 'patients',
    'whats': "what is", 'howmany': 'how many',
}


def normalize_typos(text: str) -> str:
    words = text.lower().split()
    mapped = []
    for w in words:
        clean = w.strip('.,?!;:()[]{}"\'-')
        if clean in _DIRECT_TYPO_MAP:
            mapped.append(w.replace(clean, _DIRECT_TYPO_MAP[clean]))
        else:
            mapped.append(w)
    text = ' '.join(mapped)
    words = text.lower().split()
    result = []
    for word in words:
        clean = word.strip('.,?!;:()[]{}"\'-')
        prefix = word[:len(word) - len(word.lstrip('.,?!;:()[]{}"\'-'))]
        suffix = word[len(clean) + len(prefix):]
        if len(clean) < 3 or clean in _NORMALIZER_EXCLUSIONS:
            result.append(word)
            continue
        best_match, best_dist = None, float('inf')
        for canonical, (min_len, max_dist) in _CRITICAL_WORDS.items():
            if abs(len(clean) - len(canonical)) > max_dist:
                continue
            if clean == canonical:
                best_match = None
                break
            dist = _edit_distance(clean, canonical)
            if 0 < dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_match = canonical
        result.append(prefix + best_match + suffix if best_match else word)
    return ' '.join(result)


SYNONYMS = {
    'start date': ['ENROLLMENT_DATE', 'ADMIT_DATE', 'HIRE_DATE', 'PRESCRIPTION_DATE', 'REFERRAL_DATE', 'SERVICE_DATE'],
    'end date': ['DISENROLLMENT_DATE', 'DISCHARGE_DATE', 'RESOLVED_DATE'],
    'date': ['SERVICE_DATE', 'ADMIT_DATE', 'DISCHARGE_DATE', 'ENROLLMENT_DATE', 'DISENROLLMENT_DATE',
             'PRESCRIPTION_DATE', 'REFERRAL_DATE', 'DIAGNOSIS_DATE', 'FILL_DATE', 'SUBMITTED_DATE',
             'ADJUDICATED_DATE', 'APPOINTMENT_DATE', 'HIRE_DATE'],
    'when': ['SERVICE_DATE', 'ADMIT_DATE', 'ENROLLMENT_DATE', 'PRESCRIPTION_DATE', 'REFERRAL_DATE'],
    'service date': ['SERVICE_DATE'], 'admit date': ['ADMIT_DATE'], 'discharge date': ['DISCHARGE_DATE'],
    'fill date': ['FILL_DATE'], 'submitted date': ['SUBMITTED_DATE'],
    'adjudicated date': ['ADJUDICATED_DATE'], 'paid date': ['ADJUDICATED_DATE'],
    'enrollment date': ['ENROLLMENT_DATE'], 'disenrollment date': ['DISENROLLMENT_DATE'],

    'patient': ['MEMBER_ID', 'MRN', 'FIRST_NAME', 'LAST_NAME'],
    'member': ['MEMBER_ID', 'MRN', 'FIRST_NAME', 'LAST_NAME'],
    'subscriber': ['MEMBER_ID', 'MRN'], 'person': ['MEMBER_ID', 'MRN'],
    'beneficiary': ['MEMBER_ID', 'MRN'],
    'enrollee': ['MEMBER_ID', 'MRN'],
    'name': ['FIRST_NAME', 'LAST_NAME', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'MEDICATION_NAME'],

    'age': ['DATE_OF_BIRTH'], 'gender': ['GENDER'], 'sex': ['GENDER'],
    'race': ['RACE'], 'ethnicity': ['RACE'], 'language': ['LANGUAGE'],
    'address': ['ADDRESS', 'CITY', 'STATE', 'ZIP_CODE'], 'zip': ['ZIP_CODE'], 'zip code': ['ZIP_CODE'],
    'city': ['CITY'], 'state': ['STATE'], 'location': ['CITY', 'STATE', 'KP_REGION', 'FACILITY'],
    'geography': ['KP_REGION', 'STATE', 'CITY', 'ZIP_CODE'],
    'phone': ['PHONE'], 'email': ['EMAIL'],
    'date of birth': ['DATE_OF_BIRTH'], 'dob': ['DATE_OF_BIRTH'], 'birthday': ['DATE_OF_BIRTH'],

    'risk': ['RISK_SCORE'], 'risk score': ['RISK_SCORE'], 'raf': ['RISK_SCORE'],
    'risk adjustment': ['RISK_SCORE'], 'risk factor': ['RISK_SCORE'],
    'chronic': ['CHRONIC_CONDITIONS', 'IS_CHRONIC'], 'chronic condition': ['CHRONIC_CONDITIONS', 'IS_CHRONIC'],
    'comorbidity': ['CHRONIC_CONDITIONS'], 'comorbidities': ['CHRONIC_CONDITIONS'],
    'disease': ['ICD10_DESCRIPTION', 'HCC_CATEGORY', 'CHRONIC_CONDITIONS'],

    'enrolled': ['ENROLLMENT_DATE'], 'disenrolled': ['DISENROLLMENT_DATE'],
    'eligibility': ['ENROLLMENT_DATE', 'DISENROLLMENT_DATE'],
    'coverage': ['PLAN_TYPE', 'ENROLLMENT_DATE', 'DISENROLLMENT_DATE'],
    'coverage type': ['PLAN_TYPE'],
    'member status': ['DISENROLLMENT_DATE'],

    'provider': ['NPI', 'RENDERING_NPI', 'BILLING_NPI', 'PRESCRIBING_NPI', 'REFERRING_NPI',
                 'REFERRED_TO_NPI', 'SUPERVISING_NPI', 'DIAGNOSING_NPI', 'PCP_NPI',
                 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME'],
    'doctor': ['NPI', 'RENDERING_NPI', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'SPECIALTY'],
    'physician': ['NPI', 'RENDERING_NPI', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'SPECIALTY'],
    'clinician': ['NPI', 'RENDERING_NPI', 'SPECIALTY'],
    'practitioner': ['NPI', 'RENDERING_NPI', 'PROVIDER_TYPE'],
    'npi': ['NPI', 'RENDERING_NPI', 'BILLING_NPI', 'PRESCRIBING_NPI'],
    'rendering provider': ['RENDERING_NPI'], 'billing provider': ['BILLING_NPI'],
    'prescribing provider': ['PRESCRIBING_NPI'], 'referring provider': ['REFERRING_NPI'],
    'supervising provider': ['SUPERVISING_NPI'], 'attending provider': ['RENDERING_NPI'],
    'pcp': ['PCP_NPI'], 'primary care physician': ['PCP_NPI'],
    'specialty': ['SPECIALTY'], 'department': ['DEPARTMENT'], 'panel': ['PANEL_SIZE'],
    'panel size': ['PANEL_SIZE'], 'provider type': ['PROVIDER_TYPE'],
    'credentials': ['PROVIDER_TYPE'], 'license': ['LICENSE_STATE'],
    'dea': ['DEA_NUMBER'], 'accepting patients': ['ACCEPTS_NEW_PATIENTS'],

    'claim': ['CLAIM_ID', 'CLAIM_STATUS', 'CLAIM_TYPE'],
    'claim type': ['CLAIM_TYPE'], 'claim status': ['CLAIM_STATUS'],
    'billed': ['BILLED_AMOUNT'], 'billed amount': ['BILLED_AMOUNT'],
    'paid': ['PAID_AMOUNT'], 'paid amount': ['PAID_AMOUNT'],
    'allowed': ['ALLOWED_AMOUNT'], 'allowed amount': ['ALLOWED_AMOUNT'],
    'submitted amount': ['BILLED_AMOUNT'],
    'cost': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'COST'],
    'amount': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT', 'COST'],
    'money': ['BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT'],
    'spend': ['BILLED_AMOUNT', 'PAID_AMOUNT'], 'spending': ['BILLED_AMOUNT', 'PAID_AMOUNT'],
    'expense': ['BILLED_AMOUNT'], 'charge': ['BILLED_AMOUNT'], 'charges': ['BILLED_AMOUNT'],
    'payment': ['PAID_AMOUNT'], 'reimbursement': ['PAID_AMOUNT', 'ALLOWED_AMOUNT'],
    'pmpm': ['PAID_AMOUNT'],
    'per member per month': ['PAID_AMOUNT'],
    'member responsibility': ['MEMBER_RESPONSIBILITY', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE'],
    'out of pocket': ['MEMBER_RESPONSIBILITY', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE'],
    'cost sharing': ['COPAY', 'COINSURANCE', 'DEDUCTIBLE'],
    'copay': ['COPAY'], 'copayment': ['COPAY'],
    'coinsurance': ['COINSURANCE'], 'deductible': ['DEDUCTIBLE'],
    'denied': ['CLAIM_STATUS', 'DENIAL_REASON'], 'denial': ['CLAIM_STATUS', 'DENIAL_REASON'],
    'denial reason': ['DENIAL_REASON'], 'denial code': ['DENIAL_REASON'],
    'adjudication': ['CLAIM_STATUS', 'ADJUDICATED_DATE'],
    'remittance': ['PAID_AMOUNT', 'ADJUDICATED_DATE'],
    'status': ['CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'],

    'cpt': ['CPT_CODE', 'CPT_DESCRIPTION'], 'cpt code': ['CPT_CODE'],
    'procedure': ['CPT_CODE', 'CPT_DESCRIPTION'],
    'procedure code': ['CPT_CODE'], 'hcpcs': ['CPT_CODE'],
    'service': ['CPT_DESCRIPTION', 'CPT_CODE'], 'service code': ['CPT_CODE'],
    'rvu': ['RVU'],

    'encounter': ['ENCOUNTER_ID', 'VISIT_TYPE', 'ENCOUNTER_STATUS'],
    'visit': ['VISIT_TYPE', 'ENCOUNTER_ID'], 'visit type': ['VISIT_TYPE'],
    'encounter type': ['VISIT_TYPE'], 'care setting': ['VISIT_TYPE', 'DEPARTMENT'],
    'admission': ['ADMIT_DATE', 'VISIT_TYPE'], 'admit': ['ADMIT_DATE'],
    'discharge': ['DISCHARGE_DATE', 'DISPOSITION'], 'discharge disposition': ['DISPOSITION'],
    'chief complaint': ['CHIEF_COMPLAINT'], 'complaint': ['CHIEF_COMPLAINT'],
    'reason for visit': ['CHIEF_COMPLAINT', 'REASON'],
    'disposition': ['DISPOSITION'],
    'length of stay': ['LENGTH_OF_STAY'], 'los': ['LENGTH_OF_STAY'],
    'alos': ['LENGTH_OF_STAY'],
    'bed days': ['LENGTH_OF_STAY'],
    'inpatient': ['VISIT_TYPE'], 'outpatient': ['VISIT_TYPE'], 'emergency': ['VISIT_TYPE'],
    'encounter status': ['ENCOUNTER_STATUS'],

    'diagnosis': ['ICD10_CODE', 'ICD10_DESCRIPTION', 'PRIMARY_DIAGNOSIS', 'DIAGNOSIS_DESCRIPTION'],
    'diagnoses': ['ICD10_CODE', 'ICD10_DESCRIPTION'],
    'dx': ['ICD10_CODE', 'ICD10_DESCRIPTION'],
    'principal diagnosis': ['PRIMARY_DIAGNOSIS'],
    'primary diagnosis': ['PRIMARY_DIAGNOSIS', 'ICD10_CODE'],
    'secondary diagnosis': ['ICD10_CODE'],
    'icd': ['ICD10_CODE', 'ICD10_DESCRIPTION'], 'icd10': ['ICD10_CODE', 'ICD10_DESCRIPTION'],
    'icd code': ['ICD10_CODE'], 'diagnosis code': ['ICD10_CODE'],
    'hcc': ['HCC_CODE', 'HCC_CATEGORY'], 'hcc code': ['HCC_CODE'],
    'hcc category': ['HCC_CATEGORY'], 'condition category': ['HCC_CATEGORY'],
    'severity': ['SEVERITY'], 'diagnosis type': ['DIAGNOSIS_TYPE'],
    'condition': ['CHRONIC_CONDITIONS', 'ICD10_DESCRIPTION', 'HCC_CATEGORY'],

    'prescription': ['RX_ID', 'MEDICATION_NAME', 'PRESCRIPTION_DATE'],
    'medication': ['MEDICATION_NAME', 'MEDICATION_CLASS'],
    'drug': ['MEDICATION_NAME', 'NDC_CODE'], 'drug name': ['MEDICATION_NAME'],
    'drug class': ['MEDICATION_CLASS'], 'therapeutic class': ['MEDICATION_CLASS'],
    'medication class': ['MEDICATION_CLASS'],
    'pharmacy': ['PHARMACY'], 'pharmacy name': ['PHARMACY'],
    'refill': ['REFILLS_AUTHORIZED', 'REFILLS_USED'],
    'refills': ['REFILLS_AUTHORIZED', 'REFILLS_USED'],
    'rx': ['RX_ID', 'MEDICATION_NAME'],
    'ndc': ['NDC_CODE'], 'ndc code': ['NDC_CODE'],
    'days supply': ['DAYS_SUPPLY'], 'quantity': ['QUANTITY'],
    'formulary': ['MEDICATION_CLASS'],
    'adherence': ['REFILLS_AUTHORIZED', 'REFILLS_USED'],
    'generic': ['MEDICATION_NAME'],
    'brand': ['MEDICATION_NAME'],

    'referral': ['REFERRAL_ID', 'REFERRAL_REASON', 'REFERRAL_TYPE', 'REFERRAL_DATE'],
    'referred': ['REFERRING_NPI', 'REFERRED_TO_NPI'],
    'referral reason': ['REFERRAL_REASON'], 'referral type': ['REFERRAL_TYPE'],
    'urgency': ['URGENCY'], 'authorization': ['AUTHORIZATION_NUMBER'],
    'auth': ['AUTHORIZATION_NUMBER'], 'auth number': ['AUTHORIZATION_NUMBER'],

    'appointment': ['APPOINTMENT_ID', 'APPOINTMENT_DATE', 'APPOINTMENT_TYPE'],
    'appointment type': ['APPOINTMENT_TYPE'], 'appointment status': ['STATUS'],
    'scheduled': ['STATUS'], 'check in': ['CHECK_IN_TIME'], 'check out': ['CHECK_OUT_TIME'],
    'duration': ['DURATION_MINUTES'], 'wait time': ['DURATION_MINUTES'],
    'reason': ['REASON'],

    'region': ['KP_REGION'], 'facility': ['FACILITY'], 'site': ['FACILITY'],
    'market': ['KP_REGION'], 'service area': ['KP_REGION'],
    'plan': ['PLAN_TYPE'], 'plan type': ['PLAN_TYPE'],
    'health plan': ['PLAN_TYPE'], 'insurance': ['PLAN_TYPE'],
    'payer': ['PLAN_TYPE'],
    'line of business': ['PLAN_TYPE'],
    'initiative': ['PLAN_TYPE', 'DEPARTMENT', 'MEDICATION_CLASS', 'HCC_CATEGORY'],
    'program': ['PLAN_TYPE', 'DEPARTMENT', 'MEDICATION_CLASS'],

    'volume': ['CLAIM_ID', 'ENCOUNTER_ID', 'RX_ID'],
    'count': ['CLAIM_ID', 'ENCOUNTER_ID', 'MEMBER_ID'],
    'frequency': ['CLAIM_ID', 'ENCOUNTER_ID'],
    'revenue': ['PAID_AMOUNT', 'BILLED_AMOUNT'],
    'utilization': ['ENCOUNTER_ID', 'VISIT_TYPE'],
    'readmission': ['ENCOUNTER_ID', 'VISIT_TYPE'],
    'admission rate': ['ENCOUNTER_ID', 'VISIT_TYPE'],
}

TABLE_KEYWORDS = {
    'claims': [
        'claim', 'claims', 'billed', 'paid', 'allowed', 'denied', 'denial', 'copay', 'coinsurance',
        'deductible', 'adjudicated', 'submitted', 'cpt', 'icd10', 'reimbursement',
        'cost', 'billing', 'revenue', 'spend', 'expense', 'charge', 'charges',
        'professional', 'institutional', 'pharmacy', 'dme',
        'adjudication', 'remittance', 'reversal', 'adjustment', 'adjusted',
        'claim header', 'claim line', 'service line',
        'medical claim', 'pharmacy claim', 'professional claim', 'institutional claim',
        'coordination of benefits', 'cob', 'primary payer', 'secondary payer',
        'pmpm', 'per member per month', 'medical loss ratio', 'mlr',
        'member responsibility', 'out of pocket', 'cost sharing',
        'submitted amount', 'paid amount', 'billed amount', 'allowed amount',
        'bill type', 'revenue code', 'place of service', 'modifier',
        'units of service', 'claim cost',
        'service category', 'service type',
    ],
    'members': [
        'member', 'patient', 'enrolled', 'enrollment', 'disenroll', 'risk score',
        'chronic', 'gender', 'race', 'language', 'pcp', 'plan type', 'zip',
        'demographic', 'population', 'cohort', 'birth', 'age',
        'eligibility', 'coverage', 'subscriber', 'dependent', 'beneficiary', 'enrollee',
        'coverage type', 'member status', 'active member', 'termed', 'terminated',
        'plan year', 'renewal', 'coverage gap', 'exchange', 'marketplace',
        'member months', 'partial eligibility',
        'person', 'sex', 'dob', 'date of birth', 'ethnicity', 'geography',
        'zip code', 'city', 'state', 'address',
        'risk adjustment', 'risk score', 'raf', 'risk stratification',
        'risk tier', 'high risk', 'low risk', 'rising risk',
        'disease count', 'multi-morbid', 'comorbid',
        'population health', 'health plan', 'payer', 'line of business',
        'attribution', 'attributed',
    ],
    'encounters': [
        'encounter', 'visit', 'admit', 'discharge', 'inpatient', 'outpatient',
        'emergency', 'telehealth', 'urgent care', 'home health', 'length of stay',
        'los', 'chief complaint', 'disposition', 'department', 'readmission',
        'acute inpatient', 'skilled nursing', 'snf', 'ed visit', 'er visit',
        'office visit', 'ambulatory', 'virtual visit', 'remote',
        'radiology', 'lab', 'ambulatory surgery', 'asc',
        'admit type', 'admit source', 'discharge disposition',
        'ama', 'against medical advice', 'transferred', 'expired',
        'left without being seen', 'lwbs', 'observation',
        'encounter type', 'care setting', 'visit type',
        'alos', 'average length of stay', 'bed days', 'inpatient days',
        'admission rate', 'hospital readmission', '30-day readmission',
        'potentially preventable', 'avoidable hospitalization',
        'ed classification', 'ed crowding', 'preventable ed',
    ],
    'providers': [
        'provider', 'doctor', 'physician', 'npi', 'specialty', 'panel size',
        'accepting', 'hire', 'license', 'dea',
        'clinician', 'practitioner', 'surgeon', 'specialist',
        'nurse practitioner', 'physician assistant', 'np', 'pa',
        'rendering provider', 'billing provider', 'supervising provider',
        'attending provider', 'operating provider',
        'taxonomy', 'taxonomy code', 'provider type', 'credentials',
        'primary care', 'cardiology', 'oncology', 'orthopedics',
        'psychiatry', 'behavioral health', 'emergency medicine',
        'pediatrics', 'internal medicine', 'surgery',
        'dermatology', 'neurology', 'endocrinology', 'gastroenterology',
        'nephrology', 'pulmonology', 'rheumatology', 'urology',
        'group practice', 'solo practice', 'aco',
        'accountable care', 'medical home',
    ],
    'diagnoses': [
        'diagnosis', 'diagnoses', 'icd', 'hcc', 'severity', 'chronic condition',
        'comorbid', 'resolv',
        'icd10', 'icd-10', 'icd10-cm', 'diagnosis code', 'dx',
        'principal diagnosis', 'primary diagnosis', 'secondary diagnosis',
        'admitting diagnosis', 'discharge diagnosis',
        'diagnosis type', 'condition',
        'hcc code', 'hcc category', 'condition category',
        'hierarchical condition', 'risk adjustment',
        'drg', 'ms-drg', 'apr-drg', 'mdc', 'major diagnostic category',
        'ccsr', 'clinical classification',
        'chronic disease', 'comorbidity', 'comorbidities',
        'diabetes', 'hypertension', 'heart failure', 'copd', 'asthma',
        'ckd', 'kidney disease', 'obesity', 'depression', 'anxiety',
        'cancer', 'mental health',
    ],
    'prescriptions': [
        'prescription', 'medication', 'drug', 'rx', 'pharmacy', 'refill',
        'ndc', 'days supply', 'quantity', 'fill date', 'medication class',
        'ndc code', 'national drug code', 'rxnorm', 'drug code',
        'drug name', 'drug class', 'therapeutic class',
        'generic', 'brand', 'specialty drug', 'biologic',
        'retail pharmacy', 'mail order', 'specialty pharmacy',
        'dispensing', 'dispensed', 'fill', 'filled',
        'days supply', 'quantity dispensed', 'refills remaining',
        'ingredient cost', 'dispensing fee',
        'formulary', 'prior authorization', 'step therapy',
        'quantity limits', 'therapeutic substitution',
        'drug utilization review', 'dur',
        'generic available', 'brand-generic',
        'drug cost', 'generic penetration', 'average cost per claim',
        'medication adherence', 'adherence', 'non-adherent',
        'polypharmacy',
    ],
    'referrals': [
        'referral', 'referred', 'authorization', 'urgency', 'stat referral',
        'internal referral', 'external referral', 'second opinion',
        'prior authorization', 'prior auth', 'pre-authorization', 'preauthorization',
        'auth number', 'authorization number',
        'referring provider', 'referred to',
        'clinical indication', 'referral reason', 'referral status',
        'network leakage', 'out of network', 'in network',
        'in-network', 'out-of-network',
    ],
    'appointments': [
        'appointment', 'scheduled', 'no-show', 'no show', 'rescheduled',
        'upcoming', 'pcp visit', 'annual wellness',
        'check in', 'check out', 'checked in', 'wait time', 'duration',
        'office visit', 'wellness check', 'imaging', 'lab appointment',
        'vaccine', 'vaccination', 'immunization',
        'procedure appointment', 'telehealth appointment',
        'missed appointment', 'cancelled appointment',
        'access to care', 'scheduling', 'availability',
    ],
    'cpt_codes': ['cpt code', 'rvu', 'procedure code', 'hcpcs', 'service code'],
}

STATUS_VALUES = {
    'denied': ('CLAIM_STATUS', '=', 'DENIED'), 'rejected': ('CLAIM_STATUS', '=', 'DENIED'),
    'paid': ('CLAIM_STATUS', '=', 'PAID'), 'approved': ('STATUS', '=', 'APPROVED'),
    'pending': ('CLAIM_STATUS', '=', 'PENDING'), 'adjusted': ('CLAIM_STATUS', '=', 'ADJUSTED'),
    'appealed': ('CLAIM_STATUS', '=', 'APPEALED'), 'voided': ('CLAIM_STATUS', '=', 'VOIDED'),
    'active': ('STATUS', '=', 'ACTIVE'), 'inactive': ('STATUS', '=', 'INACTIVE'),
    'on leave': ('STATUS', '=', 'ON_LEAVE'), 'cancelled': ('STATUS', '=', 'CANCELLED'),
    'completed': ('STATUS', '=', 'COMPLETED'),
    'inpatient': ('VISIT_TYPE', '=', 'INPATIENT'), 'outpatient': ('VISIT_TYPE', '=', 'OUTPATIENT'),
    'emergency': ('VISIT_TYPE', '=', 'EMERGENCY'), 'telehealth': ('VISIT_TYPE', '=', 'TELEHEALTH'),
    'urgent care': ('VISIT_TYPE', '=', 'URGENT_CARE'), 'home health': ('VISIT_TYPE', '=', 'HOME_HEALTH'),
    'urgent': ('URGENCY', '=', 'URGENT'), 'routine': ('URGENCY', '=', 'ROUTINE'),
    'male': ('GENDER', '=', 'M'), 'female': ('GENDER', '=', 'F'),
    'chronic': ('IS_CHRONIC', '=', 'Y'),
    'scheduled': ('STATUS', '=', 'SCHEDULED'), 'checked in': ('STATUS', '=', 'CHECKED_IN'),
    'no show': ('STATUS', '=', 'NO_SHOW'), 'rescheduled': ('STATUS', '=', 'RESCHEDULED'),
    'filled': ('STATUS', '=', 'FILLED'), 'expired': ('STATUS', '=', 'EXPIRED'),
    'transferred': ('STATUS', '=', 'TRANSFERRED'),
    'primary': ('DIAGNOSIS_TYPE', '=', 'PRIMARY'), 'secondary': ('DIAGNOSIS_TYPE', '=', 'SECONDARY'),
    'mild': ('SEVERITY', '=', 'MILD'), 'moderate': ('SEVERITY', '=', 'MODERATE'),
    'severe': ('SEVERITY', '=', 'SEVERE'), 'critical': ('SEVERITY', '=', 'CRITICAL'),
    'discharged': ('DISPOSITION', '=', 'Discharged'), 'admitted': ('DISPOSITION', '=', 'Admitted'),
    'observation': ('DISPOSITION', '=', 'Observation'),
    'ama': ('DISPOSITION', '=', 'AMA'),
}

STATUS_COL_MAP = {
    'claims': 'CLAIM_STATUS', 'encounters': 'ENCOUNTER_STATUS',
    'prescriptions': 'STATUS', 'referrals': 'STATUS', 'providers': 'STATUS',
    'appointments': 'STATUS',
}

COLUMN_QUALIFIER_CONTEXTS = {
    'paid': ['paid amount', 'paid_amount', 'avg paid', 'average paid', 'total paid', 'sum paid', 'max paid', 'min paid', 'paid vs'],
    'billed': ['billed amount', 'billed_amount', 'billed amt'],
    'allowed': ['allowed amount', 'allowed_amount'],
    'denied': ['denied amount', 'denied_amount', 'denial rate'],
}

DOMAIN_CONCEPTS = {
    'newborn': {'conds': ["DATE_OF_BIRTH >= date('now', '-2 years')"], 'tables': ['members']},
    'infant': {'conds': ["DATE_OF_BIRTH >= date('now', '-2 years')"], 'tables': ['members']},
    'toddler': {'conds': ["DATE_OF_BIRTH >= date('now', '-4 years') AND DATE_OF_BIRTH < date('now', '-2 years')"], 'tables': ['members']},
    'pediatric': {'conds': ["DATE_OF_BIRTH >= date('now', '-18 years')"], 'tables': ['members']},
    'child': {'conds': ["DATE_OF_BIRTH >= date('now', '-18 years')"], 'tables': ['members']},
    'children': {'conds': ["DATE_OF_BIRTH >= date('now', '-18 years')"], 'tables': ['members']},
    'adolescent': {'conds': ["DATE_OF_BIRTH BETWEEN date('now', '-18 years') AND date('now', '-12 years')"], 'tables': ['members']},
    'teen': {'conds': ["DATE_OF_BIRTH BETWEEN date('now', '-18 years') AND date('now', '-13 years')"], 'tables': ['members']},
    'elderly': {'conds': ["DATE_OF_BIRTH <= date('now', '-65 years')"], 'tables': ['members']},
    'senior': {'conds': ["DATE_OF_BIRTH <= date('now', '-65 years')"], 'tables': ['members']},
    'geriatric': {'conds': ["DATE_OF_BIRTH <= date('now', '-65 years')"], 'tables': ['members']},
    'adult': {'conds': ["DATE_OF_BIRTH <= date('now', '-18 years') AND DATE_OF_BIRTH > date('now', '-65 years')"], 'tables': ['members']},
    'minor': {'conds': ["DATE_OF_BIRTH >= date('now', '-18 years')"], 'tables': ['members']},
    'foster care': {'conds': ["DATE_OF_BIRTH >= date('now', '-21 years')", "PLAN_TYPE = 'Medicaid'"], 'tables': ['members']},
    'foster': {'conds': ["DATE_OF_BIRTH >= date('now', '-21 years')", "PLAN_TYPE = 'Medicaid'"], 'tables': ['members']},
    'medicaid': {'conds': ["PLAN_TYPE = 'Medicaid'"], 'tables': ['members']},
    'medicare': {'conds': ["PLAN_TYPE = 'Medicare Advantage'"], 'tables': ['members']},
    'commercial': {'conds': ["PLAN_TYPE IN ('HMO', 'PPO', 'EPO', 'HDHP')"], 'tables': ['members']},
    'hmo': {'conds': ["PLAN_TYPE = 'HMO'"], 'tables': ['members']},
    'ppo': {'conds': ["PLAN_TYPE = 'PPO'"], 'tables': ['members']},
    'epo': {'conds': ["PLAN_TYPE = 'EPO'"], 'tables': ['members']},
    'hdhp': {'conds': ["PLAN_TYPE = 'HDHP'"], 'tables': ['members']},
    'diabetic': {'conds': ["ICD10_CODE LIKE 'E11%'"], 'tables': ['claims']},
    'diabetes': {'conds': ["ICD10_CODE LIKE 'E11%'"], 'tables': ['claims']},
    'hypertensive': {'conds': ["ICD10_CODE LIKE 'I10%'"], 'tables': ['claims']},
    'hypertension': {'conds': ["ICD10_CODE LIKE 'I10%'"], 'tables': ['claims']},
    'cardiac': {'conds': ["ICD10_CODE LIKE 'I25%'"], 'tables': ['claims']},
    'heart disease': {'conds': ["ICD10_CODE LIKE 'I25%'"], 'tables': ['claims']},
    'asthma': {'conds': ["ICD10_CODE LIKE 'J45%'"], 'tables': ['claims']},
    'copd': {'conds': ["ICD10_CODE LIKE 'J44%'"], 'tables': ['claims']},
    'depression': {'conds': ["ICD10_CODE LIKE 'F32%'"], 'tables': ['claims']},
    'anxiety': {'conds': ["ICD10_CODE LIKE 'F41%'"], 'tables': ['claims']},
    'ckd': {'conds': ["ICD10_CODE LIKE 'N18%'"], 'tables': ['claims']},
    'kidney disease': {'conds': ["ICD10_CODE LIKE 'N18%'"], 'tables': ['claims']},
    'obesity': {'conds': ["ICD10_CODE LIKE 'E66%'"], 'tables': ['claims']},
    'obese': {'conds': ["ICD10_CODE LIKE 'E66%'"], 'tables': ['claims']},
    'pneumonia': {'conds': ["ICD10_CODE LIKE 'J18%'"], 'tables': ['claims']},
    'uti': {'conds': ["ICD10_CODE LIKE 'N39%'"], 'tables': ['claims']},
    'migraine': {'conds': ["ICD10_CODE LIKE 'G43%'"], 'tables': ['claims']},
    'heart failure': {'conds': ["HCC_CATEGORY = 'Heart Failure'"], 'tables': ['diagnoses']},
    'cancer': {'conds': ["HCC_CATEGORY = 'Cancer'"], 'tables': ['diagnoses']},
    'mental health': {'conds': ["HCC_CATEGORY = 'Mental Health'"], 'tables': ['diagnoses']},
    'behavioral health': {'conds': ["HCC_CATEGORY = 'Mental Health'"], 'tables': ['diagnoses']},
    'high risk': {'conds': ["CAST(RISK_SCORE AS REAL) >= 3.5"], 'tables': ['members']},
    'low risk': {'conds': ["CAST(RISK_SCORE AS REAL) < 2.0"], 'tables': ['members']},
    'moderate risk': {'conds': ["CAST(RISK_SCORE AS REAL) BETWEEN 2.0 AND 3.49"], 'tables': ['members']},
    'rising risk': {'conds': ["CAST(RISK_SCORE AS REAL) BETWEEN 3.0 AND 4.0"], 'tables': ['members']},
    'complex care': {'conds': ["CAST(RISK_SCORE AS REAL) >= 4.0", "CAST(CHRONIC_CONDITIONS AS INTEGER) >= 3"], 'tables': ['members']},
    'multi-morbid': {'conds': ["CAST(CHRONIC_CONDITIONS AS INTEGER) >= 3"], 'tables': ['members']},
    'comorbid': {'conds': ["CAST(CHRONIC_CONDITIONS AS INTEGER) >= 2"], 'tables': ['members']},
    'ed visit': {'conds': ["VISIT_TYPE = 'EMERGENCY'"], 'tables': ['encounters']},
    'er visit': {'conds': ["VISIT_TYPE = 'EMERGENCY'"], 'tables': ['encounters']},
    'virtual visit': {'conds': ["VISIT_TYPE = 'TELEHEALTH'"], 'tables': ['encounters']},
    'home health': {'conds': ["VISIT_TYPE = 'HOME_HEALTH'"], 'tables': ['encounters']},
    'readmission': {'conds': ["VISIT_TYPE = 'INPATIENT'", "DISPOSITION = 'Admitted'"], 'tables': ['encounters']},
    'clean claim': {'conds': ["CLAIM_STATUS = 'PAID'", "DENIAL_REASON = ''"], 'tables': ['claims']},
    'rejected claim': {'conds': ["CLAIM_STATUS = 'DENIED'"], 'tables': ['claims']},
    'pending claim': {'conds': ["CLAIM_STATUS = 'PENDING'"], 'tables': ['claims']},
    'high-value claim': {'conds': ["CAST(BILLED_AMOUNT AS REAL) > 10000"], 'tables': ['claims']},
    'high value claim': {'conds': ["CAST(BILLED_AMOUNT AS REAL) > 10000"], 'tables': ['claims']},
    'high-value': {'conds': ["CAST(BILLED_AMOUNT AS REAL) > 10000"], 'tables': ['claims']},
    'low-value claim': {'conds': ["CAST(BILLED_AMOUNT AS REAL) < 1000"], 'tables': ['claims']},
    'expensive claim': {'conds': ["CAST(BILLED_AMOUNT AS REAL) > 10000"], 'tables': ['claims']},
    'professional claim': {'conds': ["CLAIM_TYPE = 'PROFESSIONAL'"], 'tables': ['claims']},
    'institutional claim': {'conds': ["CLAIM_TYPE = 'INSTITUTIONAL'"], 'tables': ['claims']},
    'institutional claims': {'conds': ["CLAIM_TYPE = 'INSTITUTIONAL'"], 'tables': ['claims']},
    'institutional': {'conds': ["CLAIM_TYPE = 'INSTITUTIONAL'"], 'tables': ['claims']},
    'pharmacy claim': {'conds': ["CLAIM_TYPE = 'PHARMACY'"], 'tables': ['claims']},
    'pharmacy claims': {'conds': ["CLAIM_TYPE = 'PHARMACY'"], 'tables': ['claims']},
    'dme claim': {'conds': ["CLAIM_TYPE = 'DME'"], 'tables': ['claims']},
    'dme': {'conds': ["CLAIM_TYPE = 'DME'"], 'tables': ['claims']},
    'polypharmacy': {'conds': [], 'tables': ['prescriptions']},
    'pain medication': {'conds': ["MEDICATION_CLASS = 'Pain'"], 'tables': ['prescriptions']},
    'antidepressant': {'conds': ["MEDICATION_CLASS = 'Depression'"], 'tables': ['prescriptions']},
    'blood pressure medication': {'conds': ["MEDICATION_CLASS = 'Hypertension'"], 'tables': ['prescriptions']},
    'antibiotic': {'conds': ["MEDICATION_CLASS = 'Infection'"], 'tables': ['prescriptions']},
    'statin': {'conds': ["MEDICATION_CLASS = 'Cholesterol'"], 'tables': ['prescriptions']},
    'insulin': {'conds': ["MEDICATION_CLASS = 'Diabetes'"], 'tables': ['prescriptions']},
    'specialist': {'conds': ["SPECIALTY NOT IN ('Internal Medicine', 'Pediatrics')"], 'tables': ['providers']},
    'primary care': {'conds': ["SPECIALTY IN ('Internal Medicine', 'Pediatrics')"], 'tables': ['providers']},
    'pcp': {'conds': ["SPECIALTY IN ('Internal Medicine', 'Pediatrics')"], 'tables': ['providers']},
    'accepting new patients': {'conds': ["ACCEPTS_NEW_PATIENTS = 'Y'"], 'tables': ['providers']},
    'stat referral': {'conds': ["URGENCY = 'STAT'"], 'tables': ['referrals']},
    'internal referral': {'conds': ["REFERRAL_TYPE = 'INTERNAL'"], 'tables': ['referrals']},
    'external referral': {'conds': ["REFERRAL_TYPE = 'EXTERNAL'"], 'tables': ['referrals']},
    'elective': {'conds': ["URGENCY = 'ELECTIVE'"], 'tables': ['referrals']},
    'stat': {'conds': ["URGENCY = 'STAT'"], 'tables': ['referrals']},
    'new member': {'conds': ["ENROLLMENT_DATE >= date((SELECT MAX(ENROLLMENT_DATE) FROM members), '-6 months')"], 'tables': ['members']},
    'newly enrolled': {'conds': ["ENROLLMENT_DATE >= date((SELECT MAX(ENROLLMENT_DATE) FROM members), '-6 months')"], 'tables': ['members']},
    'disenrolled': {'conds': ["DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''"], 'tables': ['members']},
    'active member': {'conds': ["(DISENROLLMENT_DATE IS NULL OR DISENROLLMENT_DATE = '' OR DISENROLLMENT_DATE > date('now'))"], 'tables': ['members']},
    'discharged': {'conds': ["DISPOSITION = 'Discharged'"], 'tables': ['encounters']},
    'admitted': {'conds': ["DISPOSITION = 'Admitted'"], 'tables': ['encounters']},
    'transferred': {'conds': ["DISPOSITION = 'Transferred'"], 'tables': ['encounters']},
    'ama': {'conds': ["DISPOSITION = 'AMA'"], 'tables': ['encounters']},
    'left ama': {'conds': ["DISPOSITION = 'AMA'"], 'tables': ['encounters']},
    'expired': {'conds': ["DISPOSITION = 'Expired'"], 'tables': ['encounters']},
    'mortality': {'conds': ["DISPOSITION = 'Expired'"], 'tables': ['encounters']},
    'processing time': {'conds': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"], 'tables': ['claims']},
    'turnaround time': {'conds': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"], 'tables': ['claims']},
    'claim processing': {'conds': ["ADJUDICATED_DATE IS NOT NULL AND ADJUDICATED_DATE != ''"], 'tables': ['claims']},
    'critical': {'conds': ["SEVERITY = 'CRITICAL'"], 'tables': ['diagnoses']},
    'severe': {'conds': ["SEVERITY = 'SEVERE'"], 'tables': ['diagnoses']},
    'filled': {'conds': ["STATUS = 'FILLED'"], 'tables': ['prescriptions']},
    'expired prescription': {'conds': ["STATUS = 'EXPIRED'"], 'tables': ['prescriptions']},
    'without a pcp': {'conds': ["(PCP_NPI IS NULL OR PCP_NPI = '')"], 'tables': ['members']},
    'no pcp': {'conds': ["(PCP_NPI IS NULL OR PCP_NPI = '')"], 'tables': ['members']},
    'prior authorization': {'conds': ["AUTHORIZATION_NUMBER IS NOT NULL AND AUTHORIZATION_NUMBER != ''"], 'tables': ['referrals']},
    'prior auth': {'conds': ["AUTHORIZATION_NUMBER IS NOT NULL AND AUTHORIZATION_NUMBER != ''"], 'tables': ['referrals']},
    'pre-authorization': {'conds': ["AUTHORIZATION_NUMBER IS NOT NULL AND AUTHORIZATION_NUMBER != ''"], 'tables': ['referrals']},
    'preauthorization': {'conds': ["AUTHORIZATION_NUMBER IS NOT NULL AND AUTHORIZATION_NUMBER != ''"], 'tables': ['referrals']},
    'authorized': {'conds': ["AUTHORIZATION_NUMBER IS NOT NULL AND AUTHORIZATION_NUMBER != ''"], 'tables': ['referrals']},
    'high dollar': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 1000"], 'tables': ['claims']},
    'high cost': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 1000"], 'tables': ['claims']},
    'expensive': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 1000"], 'tables': ['claims']},
    'costly': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 1000"], 'tables': ['claims']},
    'no show': {'conds': ["STATUS = 'NO_SHOW'"], 'tables': ['appointments']},
    'no-show': {'conds': ["STATUS = 'NO_SHOW'"], 'tables': ['appointments']},
    'missed appointment': {'conds': ["STATUS = 'NO_SHOW'"], 'tables': ['appointments']},
    'rescheduled': {'conds': ["STATUS = 'RESCHEDULED'"], 'tables': ['appointments']},
    'network leakage': {'conds': ["REFERRAL_TYPE = 'EXTERNAL'"], 'tables': ['referrals']},
    'out of network': {'conds': ["REFERRAL_TYPE = 'EXTERNAL'"], 'tables': ['referrals']},
    'in network': {'conds': ["REFERRAL_TYPE = 'INTERNAL'"], 'tables': ['referrals']},
    'unfilled': {'conds': ["STATUS != 'FILLED'"], 'tables': ['prescriptions']},
    'non-adherent': {'conds': ["CAST(REFILLS_USED AS REAL) < CAST(REFILLS_AUTHORIZED AS REAL) * 0.5"], 'tables': ['prescriptions']},
    'adherent': {'conds': ["CAST(REFILLS_USED AS REAL) >= CAST(REFILLS_AUTHORIZED AS REAL) * 0.8"], 'tables': ['prescriptions']},
    'observation': {'conds': ["DISPOSITION = 'Observation'"], 'tables': ['encounters']},
    'against medical advice': {'conds': ["DISPOSITION = 'AMA'"], 'tables': ['encounters']},


    'coronary artery disease': {'conds': ["ICD10_CODE LIKE 'I25%'"], 'tables': ['claims', 'diagnoses']},
    'cad': {'conds': ["ICD10_CODE LIKE 'I25%'"], 'tables': ['claims', 'diagnoses']},
    'congestive heart failure': {'conds': ["ICD10_CODE LIKE 'I50%'"], 'tables': ['claims', 'diagnoses']},
    'chf': {'conds': ["ICD10_CODE LIKE 'I50%'"], 'tables': ['claims', 'diagnoses']},
    'atrial fibrillation': {'conds': ["ICD10_CODE LIKE 'I48%'"], 'tables': ['claims', 'diagnoses']},
    'afib': {'conds': ["ICD10_CODE LIKE 'I48%'"], 'tables': ['claims', 'diagnoses']},
    'myocardial infarction': {'conds': ["ICD10_CODE LIKE 'I21%'"], 'tables': ['claims', 'diagnoses']},
    'heart attack': {'conds': ["ICD10_CODE LIKE 'I21%'"], 'tables': ['claims', 'diagnoses']},
    'stroke': {'conds': ["ICD10_CODE LIKE 'I63%'"], 'tables': ['claims', 'diagnoses']},
    'cerebrovascular': {'conds': ["ICD10_CODE LIKE 'I6%'"], 'tables': ['claims', 'diagnoses']},
    'cva': {'conds': ["ICD10_CODE LIKE 'I63%'"], 'tables': ['claims', 'diagnoses']},
    'tia': {'conds': ["ICD10_CODE LIKE 'G45%'"], 'tables': ['claims', 'diagnoses']},
    'peripheral vascular': {'conds': ["ICD10_CODE LIKE 'I73%'"], 'tables': ['claims', 'diagnoses']},
    'pvd': {'conds': ["ICD10_CODE LIKE 'I73%'"], 'tables': ['claims', 'diagnoses']},
    'arrhythmia': {'conds': ["ICD10_CODE LIKE 'I49%'"], 'tables': ['claims', 'diagnoses']},
    'vte': {'conds': ["ICD10_CODE LIKE 'I82%'"], 'tables': ['claims', 'diagnoses']},

    'type 1 diabetes': {'conds': ["ICD10_CODE LIKE 'E10%'"], 'tables': ['claims', 'diagnoses']},
    'type 2 diabetes': {'conds': ["ICD10_CODE LIKE 'E11%'"], 'tables': ['claims', 'diagnoses']},
    'gestational diabetes': {'conds': ["ICD10_CODE LIKE 'O24%'"], 'tables': ['claims', 'diagnoses']},
    'hyperlipidemia': {'conds': ["ICD10_CODE LIKE 'E78%'"], 'tables': ['claims', 'diagnoses']},
    'dyslipidemia': {'conds': ["ICD10_CODE LIKE 'E78%'"], 'tables': ['claims', 'diagnoses']},
    'high cholesterol': {'conds': ["ICD10_CODE LIKE 'E78%'"], 'tables': ['claims', 'diagnoses']},
    'metabolic syndrome': {'conds': ["ICD10_CODE LIKE 'E88%'"], 'tables': ['claims', 'diagnoses']},
    'thyroid': {'conds': ["ICD10_CODE LIKE 'E0%'"], 'tables': ['claims', 'diagnoses']},
    'hypothyroid': {'conds': ["ICD10_CODE LIKE 'E03%'"], 'tables': ['claims', 'diagnoses']},
    'hyperthyroid': {'conds': ["ICD10_CODE LIKE 'E05%'"], 'tables': ['claims', 'diagnoses']},

    'bronchitis': {'conds': ["ICD10_CODE LIKE 'J20%' OR ICD10_CODE LIKE 'J40%'"], 'tables': ['claims', 'diagnoses']},
    'emphysema': {'conds': ["ICD10_CODE LIKE 'J43%'"], 'tables': ['claims', 'diagnoses']},
    'sleep apnea': {'conds': ["ICD10_CODE LIKE 'G47.3%'"], 'tables': ['claims', 'diagnoses']},
    'pulmonary fibrosis': {'conds': ["ICD10_CODE LIKE 'J84%'"], 'tables': ['claims', 'diagnoses']},
    'tuberculosis': {'conds': ["ICD10_CODE LIKE 'A15%'"], 'tables': ['claims', 'diagnoses']},

    'esrd': {'conds': ["ICD10_CODE LIKE 'N18.6%'"], 'tables': ['claims', 'diagnoses']},
    'end stage renal': {'conds': ["ICD10_CODE LIKE 'N18.6%'"], 'tables': ['claims', 'diagnoses']},
    'acute kidney injury': {'conds': ["ICD10_CODE LIKE 'N17%'"], 'tables': ['claims', 'diagnoses']},
    'aki': {'conds': ["ICD10_CODE LIKE 'N17%'"], 'tables': ['claims', 'diagnoses']},
    'bph': {'conds': ["ICD10_CODE LIKE 'N40%'"], 'tables': ['claims', 'diagnoses']},

    'osteoarthritis': {'conds': ["ICD10_CODE LIKE 'M15%' OR ICD10_CODE LIKE 'M16%' OR ICD10_CODE LIKE 'M17%'"], 'tables': ['claims', 'diagnoses']},
    'rheumatoid arthritis': {'conds': ["ICD10_CODE LIKE 'M05%' OR ICD10_CODE LIKE 'M06%'"], 'tables': ['claims', 'diagnoses']},
    'osteoporosis': {'conds': ["ICD10_CODE LIKE 'M80%' OR ICD10_CODE LIKE 'M81%'"], 'tables': ['claims', 'diagnoses']},
    'back pain': {'conds': ["ICD10_CODE LIKE 'M54%'"], 'tables': ['claims', 'diagnoses']},

    'epilepsy': {'conds': ["ICD10_CODE LIKE 'G40%'"], 'tables': ['claims', 'diagnoses']},
    'parkinson': {'conds': ["ICD10_CODE LIKE 'G20%'"], 'tables': ['claims', 'diagnoses']},
    'multiple sclerosis': {'conds': ["ICD10_CODE LIKE 'G35%'"], 'tables': ['claims', 'diagnoses']},
    'alzheimer': {'conds': ["ICD10_CODE LIKE 'G30%'"], 'tables': ['claims', 'diagnoses']},
    'dementia': {'conds': ["ICD10_CODE LIKE 'F01%' OR ICD10_CODE LIKE 'F02%' OR ICD10_CODE LIKE 'F03%'"], 'tables': ['claims', 'diagnoses']},

    'bipolar': {'conds': ["ICD10_CODE LIKE 'F31%'"], 'tables': ['claims', 'diagnoses']},
    'schizophrenia': {'conds': ["ICD10_CODE LIKE 'F20%'"], 'tables': ['claims', 'diagnoses']},
    'ptsd': {'conds': ["ICD10_CODE LIKE 'F43.1%'"], 'tables': ['claims', 'diagnoses']},
    'substance abuse': {'conds': ["ICD10_CODE LIKE 'F1%'"], 'tables': ['claims', 'diagnoses']},
    'substance use disorder': {'conds': ["ICD10_CODE LIKE 'F1%'"], 'tables': ['claims', 'diagnoses']},
    'sud': {'conds': ["ICD10_CODE LIKE 'F1%'"], 'tables': ['claims', 'diagnoses']},
    'opioid use': {'conds': ["ICD10_CODE LIKE 'F11%'"], 'tables': ['claims', 'diagnoses']},
    'alcohol use': {'conds': ["ICD10_CODE LIKE 'F10%'"], 'tables': ['claims', 'diagnoses']},

    'gerd': {'conds': ["ICD10_CODE LIKE 'K21%'"], 'tables': ['claims', 'diagnoses']},
    'crohn': {'conds': ["ICD10_CODE LIKE 'K50%'"], 'tables': ['claims', 'diagnoses']},
    'ulcerative colitis': {'conds': ["ICD10_CODE LIKE 'K51%'"], 'tables': ['claims', 'diagnoses']},
    'cirrhosis': {'conds': ["ICD10_CODE LIKE 'K74%'"], 'tables': ['claims', 'diagnoses']},
    'nafld': {'conds': ["ICD10_CODE LIKE 'K76.0%'"], 'tables': ['claims', 'diagnoses']},

    'sepsis': {'conds': ["ICD10_CODE LIKE 'A41%'"], 'tables': ['claims', 'diagnoses']},
    'anemia': {'conds': ["ICD10_CODE LIKE 'D6%'"], 'tables': ['claims', 'diagnoses']},
    'hiv': {'conds': ["ICD10_CODE LIKE 'B20%'"], 'tables': ['claims', 'diagnoses']},
    'covid': {'conds': ["ICD10_CODE LIKE 'U07%'"], 'tables': ['claims', 'diagnoses']},

    'acute inpatient': {'conds': ["VISIT_TYPE = 'INPATIENT'"], 'tables': ['encounters']},
    'skilled nursing': {'conds': ["DEPARTMENT LIKE '%Nursing%' OR VISIT_TYPE = 'HOME_HEALTH'"], 'tables': ['encounters']},
    'snf': {'conds': ["DEPARTMENT LIKE '%Nursing%'"], 'tables': ['encounters']},
    'urgent care': {'conds': ["VISIT_TYPE = 'URGENT_CARE'"], 'tables': ['encounters']},
    'office visit': {'conds': ["VISIT_TYPE = 'OUTPATIENT'"], 'tables': ['encounters']},
    'ambulatory': {'conds': ["VISIT_TYPE IN ('OUTPATIENT', 'URGENT_CARE')"], 'tables': ['encounters']},
    'long stay': {'conds': ["CAST(LENGTH_OF_STAY AS INTEGER) > 7"], 'tables': ['encounters']},
    'short stay': {'conds': ["CAST(LENGTH_OF_STAY AS INTEGER) <= 2"], 'tables': ['encounters']},
    'extended stay': {'conds': ["CAST(LENGTH_OF_STAY AS INTEGER) > 14"], 'tables': ['encounters']},
    'lwbs': {'conds': ["DISPOSITION = 'AMA'"], 'tables': ['encounters']},

    'cardiologist': {'conds': ["SPECIALTY = 'Cardiology'"], 'tables': ['providers']},
    'oncologist': {'conds': ["SPECIALTY = 'Oncology'"], 'tables': ['providers']},
    'dermatologist': {'conds': ["SPECIALTY = 'Dermatology'"], 'tables': ['providers']},
    'neurologist': {'conds': ["SPECIALTY = 'Neurology'"], 'tables': ['providers']},
    'orthopedist': {'conds': ["SPECIALTY = 'Orthopedics'"], 'tables': ['providers']},
    'psychiatrist': {'conds': ["SPECIALTY = 'Psychiatry'"], 'tables': ['providers']},
    'pediatrician': {'conds': ["SPECIALTY = 'Pediatrics'"], 'tables': ['providers']},
    'internist': {'conds': ["SPECIALTY = 'Internal Medicine'"], 'tables': ['providers']},
    'surgeon': {'conds': ["SPECIALTY = 'Surgery'"], 'tables': ['providers']},
    'endocrinologist': {'conds': ["SPECIALTY = 'Endocrinology'"], 'tables': ['providers']},
    'gastroenterologist': {'conds': ["SPECIALTY = 'Gastroenterology'"], 'tables': ['providers']},
    'nephrologist': {'conds': ["SPECIALTY = 'Nephrology'"], 'tables': ['providers']},
    'pulmonologist': {'conds': ["SPECIALTY = 'Pulmonology'"], 'tables': ['providers']},
    'rheumatologist': {'conds': ["SPECIALTY = 'Rheumatology'"], 'tables': ['providers']},
    'urologist': {'conds': ["SPECIALTY = 'Urology'"], 'tables': ['providers']},
    'radiologist': {'conds': ["SPECIALTY = 'Radiology'"], 'tables': ['providers']},
    'ophthalmologist': {'conds': ["SPECIALTY = 'Ophthalmology'"], 'tables': ['providers']},
    'ent': {'conds': ["SPECIALTY = 'ENT'"], 'tables': ['providers']},
    'ob/gyn': {'conds': ["SPECIALTY = 'OB/GYN'"], 'tables': ['providers']},
    'obgyn': {'conds': ["SPECIALTY = 'OB/GYN'"], 'tables': ['providers']},

    'md': {'conds': ["PROVIDER_TYPE = 'MD'"], 'tables': ['providers']},
    'do': {'conds': ["PROVIDER_TYPE = 'DO'"], 'tables': ['providers']},
    'nurse practitioner': {'conds': ["PROVIDER_TYPE = 'NP'"], 'tables': ['providers']},
    'physician assistant': {'conds': ["PROVIDER_TYPE = 'PA'"], 'tables': ['providers']},
    'mid-level': {'conds': ["PROVIDER_TYPE IN ('NP', 'PA')"], 'tables': ['providers']},

    'fee for service': {'conds': [], 'tables': ['claims']},
    'capitation': {'conds': [], 'tables': ['claims']},
    'value based': {'conds': [], 'tables': ['claims']},
    'dual eligible': {'conds': ["PLAN_TYPE IN ('Medicaid', 'Medicare Advantage')"], 'tables': ['members']},
    'exchange': {'conds': ["PLAN_TYPE IN ('HMO', 'PPO', 'EPO')"], 'tables': ['members']},
    'marketplace': {'conds': ["PLAN_TYPE IN ('HMO', 'PPO', 'EPO')"], 'tables': ['members']},
    'self insured': {'conds': [], 'tables': ['members']},

    'wellness check': {'conds': ["APPOINTMENT_TYPE = 'WELLNESS_CHECK'"], 'tables': ['appointments']},
    'annual wellness': {'conds': ["APPOINTMENT_TYPE = 'WELLNESS_CHECK'"], 'tables': ['appointments']},
    'preventive visit': {'conds': ["APPOINTMENT_TYPE = 'WELLNESS_CHECK'"], 'tables': ['appointments']},
    'lab appointment': {'conds': ["APPOINTMENT_TYPE = 'LAB'"], 'tables': ['appointments']},
    'imaging appointment': {'conds': ["APPOINTMENT_TYPE = 'IMAGING'"], 'tables': ['appointments']},
    'vaccine appointment': {'conds': ["APPOINTMENT_TYPE = 'VACCINE'"], 'tables': ['appointments']},
    'procedure appointment': {'conds': ["APPOINTMENT_TYPE = 'PROCEDURE'"], 'tables': ['appointments']},
    'telehealth appointment': {'conds': ["APPOINTMENT_TYPE = 'TELEHEALTH'"], 'tables': ['appointments']},

    'high dollar claim': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 1000"], 'tables': ['claims']},
    'catastrophic claim': {'conds': ["CAST(PAID_AMOUNT AS REAL) > 50000"], 'tables': ['claims']},
    'low dollar claim': {'conds': ["CAST(PAID_AMOUNT AS REAL) < 100"], 'tables': ['claims']},
    'zero pay': {'conds': ["CAST(PAID_AMOUNT AS REAL) = 0"], 'tables': ['claims']},
    'zero paid': {'conds': ["CAST(PAID_AMOUNT AS REAL) = 0"], 'tables': ['claims']},
    'underpaid': {'conds': ["CAST(PAID_AMOUNT AS REAL) < CAST(ALLOWED_AMOUNT AS REAL) * 0.8"], 'tables': ['claims']},
    'overpaid': {'conds': ["CAST(PAID_AMOUNT AS REAL) > CAST(ALLOWED_AMOUNT AS REAL) * 1.2"], 'tables': ['claims']},

    'primary diagnosis': {'conds': ["DIAGNOSIS_TYPE = 'PRIMARY'"], 'tables': ['diagnoses']},
    'secondary diagnosis': {'conds': ["DIAGNOSIS_TYPE = 'SECONDARY'"], 'tables': ['diagnoses']},
    'admitting diagnosis': {'conds': ["DIAGNOSIS_TYPE = 'ADMITTING'"], 'tables': ['diagnoses']},
    'discharge diagnosis': {'conds': ["DIAGNOSIS_TYPE = 'DISCHARGE'"], 'tables': ['diagnoses']},
    'mild': {'conds': ["SEVERITY = 'MILD'"], 'tables': ['diagnoses']},
    'moderate': {'conds': ["SEVERITY = 'MODERATE'"], 'tables': ['diagnoses']},

    'anxiety medication': {'conds': ["MEDICATION_CLASS = 'Anxiety'"], 'tables': ['prescriptions']},
    'asthma medication': {'conds': ["MEDICATION_CLASS = 'Asthma'"], 'tables': ['prescriptions']},
    'cholesterol medication': {'conds': ["MEDICATION_CLASS = 'Cholesterol'"], 'tables': ['prescriptions']},
    'diabetes medication': {'conds': ["MEDICATION_CLASS = 'Diabetes'"], 'tables': ['prescriptions']},
    'gerd medication': {'conds': ["MEDICATION_CLASS = 'GERD'"], 'tables': ['prescriptions']},
    'hypertension medication': {'conds': ["MEDICATION_CLASS = 'Hypertension'"], 'tables': ['prescriptions']},
    'hypothyroid medication': {'conds': ["MEDICATION_CLASS = 'Hypothyroid'"], 'tables': ['prescriptions']},
    'infection medication': {'conds': ["MEDICATION_CLASS = 'Infection'"], 'tables': ['prescriptions']},
    'inflammation medication': {'conds': ["MEDICATION_CLASS = 'Inflammation'"], 'tables': ['prescriptions']},
    'neuropathy medication': {'conds': ["MEDICATION_CLASS = 'Neuropathy'"], 'tables': ['prescriptions']},
    'opioid': {'conds': ["MEDICATION_CLASS = 'Pain'"], 'tables': ['prescriptions']},
    'painkiller': {'conds': ["MEDICATION_CLASS = 'Pain'"], 'tables': ['prescriptions']},
    'analgesic': {'conds': ["MEDICATION_CLASS = 'Pain'"], 'tables': ['prescriptions']},

    'voided': {'conds': ["CLAIM_STATUS = 'VOIDED'"], 'tables': ['claims']},
    'voided claim': {'conds': ["CLAIM_STATUS = 'VOIDED'"], 'tables': ['claims']},
    'appealed claim': {'conds': ["CLAIM_STATUS = 'APPEALED'"], 'tables': ['claims']},
    'open claim': {'conds': ["CLAIM_STATUS IN ('PENDING', 'APPEALED')"], 'tables': ['claims']},
    'closed claim': {'conds': ["CLAIM_STATUS IN ('PAID', 'DENIED', 'VOIDED', 'ADJUSTED')"], 'tables': ['claims']},
    'finalized': {'conds': ["CLAIM_STATUS IN ('PAID', 'DENIED', 'VOIDED', 'ADJUSTED')"], 'tables': ['claims']},
    'denied': {'conds': ["CLAIM_STATUS = 'DENIED'"], 'tables': ['claims']},
    'denial': {'conds': ["CLAIM_STATUS = 'DENIED'"], 'tables': ['claims']},
    'approved': {'conds': ["CLAIM_STATUS = 'PAID'"], 'tables': ['claims']},
    'approval': {'conds': ["CLAIM_STATUS = 'PAID'"], 'tables': ['claims']},
    'paid claim': {'conds': ["CLAIM_STATUS = 'PAID'"], 'tables': ['claims']},
    'pending claim': {'conds': ["CLAIM_STATUS = 'PENDING'"], 'tables': ['claims']},

    'approved referral': {'conds': ["STATUS = 'APPROVED'"], 'tables': ['referrals']},
    'denied referral': {'conds': ["STATUS = 'DENIED'"], 'tables': ['referrals']},
    'pending referral': {'conds': ["STATUS = 'PENDING'"], 'tables': ['referrals']},
    'completed referral': {'conds': ["STATUS = 'COMPLETED'"], 'tables': ['referrals']},
    'expired referral': {'conds': ["STATUS = 'EXPIRED'"], 'tables': ['referrals']},
    'self referral': {'conds': ["REFERRAL_TYPE = 'SELF_REFERRAL'"], 'tables': ['referrals']},
}

COMPUTED_COLUMNS = {
    'age': {
        'expr': "CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INTEGER)",
        'alias': 'age',
        'table': 'members',
        'description': 'Patient age calculated from date of birth',
    },
    'age_group': {
        'expr': """CASE
            WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 18 THEN '0-17'
            WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 35 THEN '18-34'
            WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 50 THEN '35-49'
            WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 65 THEN '50-64'
            ELSE '65+' END""",
        'alias': 'age_group',
        'table': 'members',
        'description': 'Tuva-style age group buckets',
    },
    'processing_days': {
        'expr': "ROUND(julianday(c.ADJUDICATED_DATE) - julianday(c.SUBMITTED_DATE), 1)",
        'alias': 'processing_days',
        'table': 'claims',
        'description': 'Days between submission and adjudication',
    },
    'member_tenure_months': {
        'expr': "CAST((julianday(COALESCE(NULLIF(m.DISENROLLMENT_DATE, ''), date('now'))) - julianday(m.ENROLLMENT_DATE)) / 30.44 AS INTEGER)",
        'alias': 'tenure_months',
        'table': 'members',
        'description': 'Months since enrollment',
    },
    'cost_per_encounter': {
        'expr': "ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2)",
        'alias': 'avg_cost_per_encounter',
        'table': 'claims',
        'description': 'Tuva: Average cost per encounter/claim',
    },
    'denial_rate': {
        'expr': "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
        'alias': 'denial_rate_pct',
        'table': 'claims',
        'description': 'Tuva: Percentage of claims denied',
    },
    'clean_claim_rate': {
        'expr': "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'PAID' AND (c.DENIAL_REASON = '' OR c.DENIAL_REASON IS NULL) THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)",
        'alias': 'clean_claim_rate_pct',
        'table': 'claims',
        'description': 'Tuva: Percentage of claims paid without denial',
    },
    'risk_tier': {
        'expr': """CASE
            WHEN CAST(m.RISK_SCORE AS REAL) >= 4.0 THEN 'Very High Risk'
            WHEN CAST(m.RISK_SCORE AS REAL) >= 3.0 THEN 'High Risk'
            WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 'Moderate Risk'
            ELSE 'Low Risk' END""",
        'alias': 'risk_tier',
        'table': 'members',
        'description': 'Tuva: Risk stratification tier',
    },
}

DEMOGRAPHICS_EXPANSION = {
    'demographics': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE'],
    'member demographics': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE'],
    'patient demographics': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE'],
    'population demographics': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE'],
    'patient profile': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE', 'KP_REGION'],
    'member profile': ['GENDER', 'age_group', 'RACE', 'PLAN_TYPE', 'KP_REGION'],
}

CONCEPT_TYPOS = {
    'meidcaid': 'medicaid', 'medicadi': 'medicaid', 'medcaid': 'medicaid',
    'medicar': 'medicare', 'meidcare': 'medicare',
    'commerical': 'commercial', 'commercail': 'commercial',
    'diebetic': 'diabetic', 'diabtes': 'diabetes', 'dieabetes': 'diabetes',
    'hypertnesion': 'hypertension', 'hypertesnion': 'hypertension',
    'depresion': 'depression', 'anxeity': 'anxiety',
    'obseity': 'obesity', 'obeese': 'obese',
    'pnemonia': 'pneumonia', 'pnuemonia': 'pneumonia',
    'schizophrnia': 'schizophrenia', 'schizofrenia': 'schizophrenia',
    'alzhiemers': 'alzheimer', 'alzeheimers': 'alzheimer',
    'athritis': 'osteoarthritis', 'arthritus': 'osteoarthritis',
    'epilepsey': 'epilepsy', 'parkinsons': 'parkinson',
    'cirrohsis': 'cirrhosis', 'cirrosis': 'cirrhosis',
    'fibrilation': 'atrial fibrillation', 'afibb': 'afib',
    'hyperlipidimia': 'hyperlipidemia', 'dyslipedemia': 'dyslipidemia',
    'newbron': 'newborn', 'fostercare': 'foster care', 'fostr': 'foster',
    'pedeatric': 'pediatric', 'pediatirc': 'pediatric', 'peditric': 'pediatric',
    'geriatircs': 'geriatric', 'eldery': 'elderly', 'senoir': 'senior',
    'adolecent': 'adolescent', 'adolescnet': 'adolescent',
    'pollypharmacy': 'polypharmacy',
    'tlehealth': 'telehealth', 'telemdicine': 'telehealth',
    'readmision': 'readmission', 'readmisions': 'readmission',
    'authrorization': 'authorization', 'authoirzation': 'authorization',
    'referal': 'referral', 'referall': 'referral', 'refferal': 'referral',
    'perscription': 'prescription', 'presciption': 'prescription',
    'medicaton': 'medication', 'medcation': 'medication',
    'diagnois': 'diagnosis', 'diagnossis': 'diagnosis',
    'encoutner': 'encounter', 'enounter': 'encounter',
    'adjudicaton': 'adjudication', 'adjudacation': 'adjudication',
    'reimbursment': 'reimbursement', 'reiumbursement': 'reimbursement',
    'utilzation': 'utilization', 'utilizaton': 'utilization',
    'cardiolgy': 'cardiology', 'cardiologst': 'cardiologist',
    'nurology': 'neurology', 'nuerologist': 'neurologist',
    'orthopaedics': 'orthopedics', 'opthalmology': 'ophthalmology',
    'psychatry': 'psychiatry', 'pschiatrist': 'psychiatrist',
    'pulmunology': 'pulmonology', 'gastroenterolgy': 'gastroenterology',
    'dermatolgy': 'dermatology', 'endocrinolgy': 'endocrinology',
}

TIME_FILTERS = {
    'last 30 days': "date('now', '-30 days')", 'last month': "date('now', '-1 month')",
    'last quarter': "date('now', '-3 months')", 'last 3 months': "date('now', '-3 months')",
    'this year': "date('now', 'start of year')",
    'last year': ("date('now', '-1 year', 'start of year')", "date('now', 'start of year', '-1 day')"),
    'last week': "date('now', '-7 days')", 'last 7 days': "date('now', '-7 days')",
    'last 6 months': "date('now', '-6 months')", 'last 90 days': "date('now', '-90 days')",
}

TABLE_DATE_COL = {
    'claims': 'SERVICE_DATE', 'encounters': 'SERVICE_DATE', 'diagnoses': 'DIAGNOSIS_DATE',
    'prescriptions': 'PRESCRIPTION_DATE', 'referrals': 'REFERRAL_DATE',
    'members': 'ENROLLMENT_DATE', 'providers': 'HIRE_DATE', 'appointments': 'APPOINTMENT_DATE',
}

RATIO_VALUE_MAP = {
    'emergency': ('encounters', 'VISIT_TYPE', 'EMERGENCY'),
    'outpatient': ('encounters', 'VISIT_TYPE', 'OUTPATIENT'),
    'inpatient': ('encounters', 'VISIT_TYPE', 'INPATIENT'),
    'telehealth': ('encounters', 'VISIT_TYPE', 'TELEHEALTH'),
    'urgent care': ('encounters', 'VISIT_TYPE', 'URGENT_CARE'),
    'urgent_care': ('encounters', 'VISIT_TYPE', 'URGENT_CARE'),
    'home health': ('encounters', 'VISIT_TYPE', 'HOME_HEALTH'),
    'home_health': ('encounters', 'VISIT_TYPE', 'HOME_HEALTH'),
    'professional': ('claims', 'CLAIM_TYPE', 'PROFESSIONAL'),
    'institutional': ('claims', 'CLAIM_TYPE', 'INSTITUTIONAL'),
    'pharmacy': ('claims', 'CLAIM_TYPE', 'PHARMACY'),
    'dme': ('claims', 'CLAIM_TYPE', 'DME'),
    'denied': ('claims', 'CLAIM_STATUS', 'DENIED'),
    'paid': ('claims', 'CLAIM_STATUS', 'PAID'),
    'pending': ('claims', 'CLAIM_STATUS', 'PENDING'),
    'male': ('members', 'GENDER', 'M'), 'female': ('members', 'GENDER', 'F'),
}

KPI_TEMPLATES = {
    'denial_rate': {
        'triggers': [r'denial\s+rate', r'deny\s+rate'],
        'sql': "SELECT {group}COUNT(*) as total_claims, SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_count, ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate FROM claims{where}{group_by}{order} LIMIT 50;",
        'tables': ['claims'],
    },
    'readmission_rate': {
        'triggers': [r'readmission\s+rate'],
        'sql': """SELECT COUNT(*) as total_discharges,
  SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) as readmissions_30d,
  ROUND(100.0 * SUM(CASE WHEN days_to_readmit <= 30 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as readmission_rate_pct
FROM (
  SELECT e1.ENCOUNTER_ID, e1.MEMBER_ID, e1.DISCHARGE_DATE,
    MIN(JULIANDAY(e2.ADMIT_DATE) - JULIANDAY(e1.DISCHARGE_DATE)) as days_to_readmit
  FROM encounters e1
  LEFT JOIN encounters e2 ON e1.MEMBER_ID = e2.MEMBER_ID
    AND e2.ADMIT_DATE > e1.DISCHARGE_DATE AND e2.VISIT_TYPE = 'INPATIENT'
  WHERE e1.VISIT_TYPE = 'INPATIENT' AND e1.DISCHARGE_DATE != ''
  GROUP BY e1.ENCOUNTER_ID
) sub;""",
        'tables': ['encounters'], 'static': True,
    },
    'processing_time': {
        'triggers': [r'processing\s*time', r'adjudication\s*time', r'time\s*to\s*(?:process|adjudicate)'],
        'sql': """SELECT {group_col}, COUNT(*) as claim_count, ROUND(AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as avg_processing_days,
ROUND(MIN(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as min_days, ROUND(MAX(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)),1) as max_days
FROM claims WHERE SUBMITTED_DATE IS NOT NULL AND ADJUDICATED_DATE IS NOT NULL{year_filter}
GROUP BY {group_col} ORDER BY avg_processing_days DESC;""",
        'tables': ['claims'],
    },
    'fill_rate': {
        'triggers': [r'fill\s+rate', r'approval\s+rate', r'completion\s+rate'],
        'sql': "SELECT {group}COUNT(*) as total, SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) as approved_count, ROUND(100.0 * SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as approval_rate FROM prescriptions{where}{group_by}{order} LIMIT 50;",
        'tables': ['prescriptions'],
    },
    'no_show_rate': {
        'triggers': [r'no[\s-]?show\s+rate'],
        'sql': "SELECT {group}COUNT(*) as total_appointments, SUM(CASE WHEN STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) as no_shows, ROUND(100.0 * SUM(CASE WHEN STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as no_show_rate FROM appointments{where}{group_by}{order} LIMIT 50;",
        'tables': ['appointments'],
    },
    'alos': {
        'triggers': [r'average\s+length\s+of\s+stay', r'\balos\b', r'avg\s+los'],
        'sql': "SELECT {group}ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)), 2) as avg_los, COUNT(*) as encounter_count FROM encounters WHERE {visit_type_filter}{extra_where}{group_by}{order} LIMIT 50;",
        'tables': ['encounters'],
    },
    'pmpm': {
        'triggers': [r'\bpmpm\b', r'per\s+member\s+per\s+month'],
        'sql': """SELECT SUBSTR(c.SERVICE_DATE, 1, 7) as month, COUNT(DISTINCT c.MEMBER_ID) as members,
ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT c.MEMBER_ID), 0), 2) as pmpm
FROM claims c{where} GROUP BY month ORDER BY month LIMIT 50;""",
        'tables': ['claims'],
    },
}

REGION_MAP = {
    'baltimore': 'MAS', 'maryland': 'MAS', 'mid-atlantic': 'MAS', 'virginia': 'MAS', 'dc': 'MAS', 'mas': 'MAS',
    'oakland': 'NCAL', 'san francisco': 'NCAL', 'santa clara': 'NCAL', 'ncal': 'NCAL', 'northern california': 'NCAL',
    'los angeles': 'SCAL', 'san diego': 'SCAL', 'fontana': 'SCAL', 'scal': 'SCAL', 'southern california': 'SCAL',
    'portland': 'NW', 'seattle': 'NW', 'northwest': 'NW', 'nw': 'NW',
    'denver': 'CO', 'aurora': 'CO', 'colorado': 'CO', 'co': 'CO',
    'atlanta': 'GA', 'georgia': 'GA', 'ga': 'GA',
    'honolulu': 'HI', 'hawaii': 'HI', 'hi': 'HI',
    'kansas city': 'MID', 'st. louis': 'MID', 'midwest': 'MID', 'mid': 'MID',
}

NUMERIC_COL_MAP = [
    (['billed amount', 'billed_amount', 'billed charge'], 'BILLED_AMOUNT'),
    (['paid amount', 'paid_amount', 'average paid', 'avg paid', 'total paid'], 'PAID_AMOUNT'),
    (['allowed amount', 'allowed_amount'], 'ALLOWED_AMOUNT'),
    (['risk', 'score'], 'RISK_SCORE'),
    (['length of stay', 'stay', 'los'], 'LENGTH_OF_STAY'),
    (['copay'], 'COPAY'),
    (['days supply', 'supply'], 'DAYS_SUPPLY'),
    (['panel'], 'PANEL_SIZE'),
    (['rvu'], 'RVU'),
]

ENTITY_GROUP_COLS = {
    'provider': 'SPECIALTY', 'doctor': 'SPECIALTY', 'physician': 'SPECIALTY',
    'member': 'MEMBER_ID', 'patient': 'MEMBER_ID',
    'region': 'KP_REGION', 'facility': 'FACILITY', 'department': 'DEPARTMENT',
    'specialty': 'SPECIALTY', 'medication': 'MEDICATION_NAME', 'drug': 'MEDICATION_NAME',
    'diagnosis': 'ICD10_CODE', 'condition': 'ICD10_DESCRIPTION',
    'plan': 'PLAN_TYPE', 'visit': 'VISIT_TYPE', 'claim': 'CLAIM_TYPE',
    'status': 'CLAIM_STATUS', 'gender': 'GENDER', 'chronic': 'IS_CHRONIC',
}


class SchemaRegistry:

    COMPOSITE_JOIN_KEYS = {'MEMBER_ID': ['KP_REGION']}
    USE_COMPOSITE_JOINS = False

    def __init__(self, catalog_dir: str = None):
        self.tables: Dict[str, List[Dict]] = {}
        self.column_to_tables: Dict[str, List[str]] = defaultdict(list)
        self.relationships: List[Dict] = []
        self.join_graph: Dict[str, Dict[str, str]] = defaultdict(dict)
        if catalog_dir:
            self._load(catalog_dir)

    def _load(self, catalog_dir: str):
        tables_dir = os.path.join(catalog_dir, "tables")
        rels_dir = os.path.join(catalog_dir, "relationships")
        if os.path.exists(tables_dir):
            for fname in os.listdir(tables_dir):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(tables_dir, fname), 'r') as f:
                        data = json.load(f)
                    tname = data.get('table_name', fname.replace('.json', ''))
                    cols_raw = data.get('columns', {})
                    if isinstance(cols_raw, dict):
                        cols_raw = [{'column_name': k, **v} for k, v in cols_raw.items()]
                    cols = []
                    for c in cols_raw:
                        cn = c.get('column_name', c.get('name', ''))
                        cols.append({
                            'name': cn,
                            'data_type': c.get('data_type', c.get('semantic_type', 'text')),
                            'semantic_type': c.get('semantic_type', ''),
                            'cardinality': c.get('cardinality', 0),
                            'null_pct': c.get('null_percentage', 0),
                            'top_values': c.get('top_values', []),
                        })
                        self.column_to_tables[cn.upper()].append(tname)
                    self.tables[tname] = cols
                except Exception:
                    pass
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
        self._build_join_graph()

    def _build_join_graph(self):
        PRIORITY = {
            'encounter_id': 10, 'claim_id': 10, 'diagnosis_id': 10, 'referral_id': 10, 'rx_id': 10,
            'member_id': 8, 'mrn': 8, 'rendering_npi': 7, 'npi': 7, 'icd10_code': 6,
            'kp_region': 1, 'facility': 1, 'department': 1, 'status': 0, 'specialty': 1, 'copay': 0,
        }
        pair_joins = defaultdict(list)
        for rel in self.relationships:
            src, tgt, jcol = rel.get('source_table', ''), rel.get('target_table', ''), rel.get('join_column', '')
            if src and tgt and jcol:
                pri = PRIORITY.get(jcol.lower(), 5)
                if pri > 0:
                    pair_joins[(src, tgt)].append((jcol, pri))
                    pair_joins[(tgt, src)].append((jcol, pri))
        MANUAL = [
            ('providers', 'claims', 'NPI', 'RENDERING_NPI', 9),
            ('providers', 'encounters', 'NPI', 'RENDERING_NPI', 9),
            ('providers', 'prescriptions', 'NPI', 'PRESCRIBING_NPI', 9),
            ('providers', 'referrals', 'NPI', 'REFERRING_NPI', 9),
            ('providers', 'diagnoses', 'NPI', 'DIAGNOSING_NPI', 9),
            ('claims', 'encounters', 'ENCOUNTER_ID', 'ENCOUNTER_ID', 10),
            ('diagnoses', 'encounters', 'ENCOUNTER_ID', 'ENCOUNTER_ID', 10),
        ]
        for t1, t2, c1, c2, pri in MANUAL:
            pair_joins[(t1, t2)].append((f"{c1}={c2}", pri))
            pair_joins[(t2, t1)].append((f"{c2}={c1}", pri))
        for (t1, t2), joins in pair_joins.items():
            best_col, best_pri = max(joins, key=lambda x: x[1])
            if best_pri >= 2:
                self.join_graph[t1][t2] = best_col.upper()

    def detect_composite_join_need(self, db_path: str):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            total = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
            unique = conn.execute("SELECT COUNT(DISTINCT MEMBER_ID) FROM members").fetchone()[0]
            SchemaRegistry.USE_COMPOSITE_JOINS = total > unique
            conn.close()
        except Exception:
            pass

    def get_columns(self, table: str) -> List[str]:
        return [c['name'] for c in self.tables.get(table, [])]

    def get_column_info(self, table: str, col_name: str) -> Optional[Dict]:
        for c in self.tables.get(table, []):
            if c['name'].upper() == col_name.upper():
                return c
        return None

    def find_join_path(self, tables: List[str]) -> List[Tuple[str, str, str]]:
        if len(tables) <= 1:
            return []
        base = tables[0]
        joins, connected = [], {base}
        for target in tables[1:]:
            if target in connected:
                continue
            path = self._bfs(base, target)
            if path:
                for t1, t2, col in path:
                    if t2 not in connected:
                        joins.append((t1, t2, col))
                        connected.add(t2)
            else:
                joins.append((base, target, 'MEMBER_ID'))
                connected.add(target)
        return joins

    def _bfs(self, start: str, end: str) -> List[Tuple[str, str, str]]:
        if start == end:
            return []
        visited, queue = set(), [(start, [])]
        while queue:
            cur, path = queue.pop(0)
            if cur == end:
                return path
            if cur in visited:
                continue
            visited.add(cur)
            for nb, jcol in self.join_graph.get(cur, {}).items():
                if nb not in visited:
                    queue.append((nb, path + [(cur, nb, jcol)]))
        return []

    def get_composite_conditions(self, t1: str, t2: str, primary_col: str) -> List[str]:
        if not SchemaRegistry.USE_COMPOSITE_JOINS:
            return []
        pk = primary_col.upper().split('=')[0]
        extras = []
        t1_cols = {c['name'].upper() for c in self.tables.get(t1, [])}
        t2_cols = {c['name'].upper() for c in self.tables.get(t2, [])}
        for sec in self.COMPOSITE_JOIN_KEYS.get(pk, []):
            if sec in t1_cols and sec in t2_cols:
                extras.append(sec)
        return extras


def _fuzzy_match(a: str, b: str, max_dist: int = 2) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist or len(a) > 15 or len(b) > 15:
        return False
    return _edit_distance(a, b) <= max_dist


class ColumnResolver:

    def __init__(self, schema: SchemaRegistry):
        self.schema = schema
        self._idx: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for table, cols in schema.tables.items():
            for c in cols:
                self._idx[c['name'].upper().lower()].append((table, c['name'].upper()))

    def resolve(self, question: str, hint_tables: List[str] = None) -> List[Dict]:
        q = question.lower()
        words = re.findall(r'[a-z_]+', q)
        resolved, seen = [], set()

        for phrase in sorted(SYNONYMS.keys(), key=len, reverse=True):
            exact = bool(re.search(r'\b' + re.escape(phrase) + r'\b', q))
            fuzzy = False
            if not exact and ' ' not in phrase and len(phrase) >= 4:
                fuzzy = any(_fuzzy_match(w, phrase) for w in words)
            if exact or fuzzy:
                for col in SYNONYMS[phrase]:
                    tables_with = self.schema.column_to_tables.get(col, [])
                    if hint_tables:
                        for t in hint_tables:
                            if t in tables_with:
                                key = (col, t)
                                if key not in seen:
                                    resolved.append({'column': col, 'table': t, 'match_type': 'synonym', 'original_term': phrase})
                                    seen.add(key)
                                break
                        else:
                            if tables_with:
                                key = (col, tables_with[0])
                                if key not in seen:
                                    resolved.append({'column': col, 'table': tables_with[0], 'match_type': 'synonym', 'original_term': phrase})
                                    seen.add(key)
                    elif tables_with:
                        key = (col, tables_with[0])
                        if key not in seen:
                            resolved.append({'column': col, 'table': tables_with[0], 'match_type': 'synonym', 'original_term': phrase})
                            seen.add(key)

        for word in words:
            upper = word.upper()
            if upper in self.schema.column_to_tables:
                for t in self.schema.column_to_tables[upper]:
                    key = (upper, t)
                    if key not in seen:
                        resolved.append({'column': upper, 'table': t, 'match_type': 'exact', 'original_term': word})
                        seen.add(key)

        SKIP = {'number', 'count', 'total', 'give', 'show', 'list', 'find', 'have', 'highest', 'lowest',
                'which', 'what', 'most', 'many', 'much', 'each', 'every', 'with', 'from', 'that', 'this',
                'type', 'time', 'first', 'last', 'over', 'more', 'less', 'than', 'about', 'into', 'some'}
        for word in words:
            if len(word) < 4 or word in SKIP:
                continue
            for cl, entries in self._idx.items():
                if word in cl and (cl, entries[0][0]) not in seen:
                    for t, cn in entries:
                        key = (cn, t)
                        if key not in seen:
                            resolved.append({'column': cn, 'table': t, 'match_type': 'substring', 'original_term': word})
                            seen.add(key)
        return resolved


class QueryParser:

    AGG_PATTERNS = {
        'COUNT': [r'\bhow many\b', r'\bcount\b', r'\btotal number\b', r'\bnumber of\b', r'\bvolume\b',
                  r'\bmost common\b', r'\bmost frequent\b', r'\brate\b', r'\bratio\b',
                  r'\bdistribut\w*\b', r'\bbreakdown\b', r'\bsplit\b',
                  r'\btotal\s+(?:claims|encounters|members|providers|diagnoses|prescriptions|referrals|patients|visits)',
                  r'\bfrequen'],
        'SUM': [r'\btotal\b(?! number)(?!\s+(?:claims|encounters|members|providers|diagnoses|prescriptions|referrals|patients|visits))',
                r'\bsum\b', r'\bcombined\b', r'\baggregate\b'],
        'AVG': [r'\baverage\b', r'\bavg\b', r'\bmean\b', r'\btypical\b'],
        'MAX': [r'\bmax\b', r'\bmaximum\b', r'\bhighest\b', r'\blargest\b', r'\bbiggest\b', r'\bmost expensive\b'],
        'MIN': [r'\bmin\b', r'\bminimum\b', r'\blowest\b', r'\bsmallest\b', r'\bcheapest\b'],
    }

    GROUP_RE = [
        r'\bby\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having|limit|with)|[?.!;,]?\s*$)',
        r'\bper\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having|limit|with)|[?.!;,]?\s*$)',
        r'\b(?:for|in) each\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|[?.!;,]?\s*$)',
        r'\beach\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|[?.!;,]?\s*$)',
        r'\bgrouped? by\s+([\w\s]+?)(?:[?.!;,]?\s*$)', r'\bbroken? down by\s+([\w\s]+?)(?:[?.!;,]?\s*$)',
        r'\bacross\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having)|[?.!;,]?\s*$)',
    ]

    COMPARISON_RE = [
        (r'(?:more than|over|greater than|above|exceeds?|>)\s*\$?([\d,]+\.?\d*)', '>'),
        (r'(?:less than|under|below|fewer than|<)\s*\$?([\d,]+\.?\d*)', '<'),
        (r'(?:at least|minimum|min)\s*\$?([\d,]+\.?\d*)', '>='),
        (r'(?:at most|maximum|max)\s*\$?([\d,]+\.?\d*)', '<='),
        (r'between\s*\$?([\d,]+\.?\d*)\s*and\s*\$?([\d,]+\.?\d*)', 'BETWEEN'),
        (r'(?:equals?|equal to|exactly)\s*\$?([\d,]+\.?\d*)', '='),
    ]

    @classmethod
    def detect_agg(cls, question: str) -> Dict[str, Any]:
        q = question.lower()
        result = {'agg_func': None, 'group_by_terms': [], 'top_n': None, 'order': 'DESC'}

        AMOUNT_WORDS = {'amount', 'cost', 'billed', 'paid', 'allowed', 'revenue', 'spend', 'spending', 'charge', 'expense', 'payment'}

        wm = re.search(r'\bwhich\s+([\w\s]+?)\s+(?:has|have|is|are|had|gets?|handles?|receives?|generates?|sees?|shows?)\s+(?:the\s+)?(?:most|highest|lowest|least|fewest|longest|shortest|largest|smallest|biggest)', q)
        if not wm:
            wm = re.search(r'\b([\w\s]+?)\s+with\s+(?:the\s+)?(?:most|highest|lowest|least|fewest|longest|shortest|largest|smallest|biggest)\b', q)
        if not wm:
            mm = re.search(r'\bmost\s+\w+(?:ed|ing)\s+(\w+)', q)
            if mm:
                wm = mm
        if wm:
            gt = wm.group(1)
            result['group_by_terms'] = [gt]
            result['top_n'] = 1 if not gt.endswith('s') else 10
            result['order'] = 'DESC' if any(w in q for w in ['most', 'highest', 'longest', 'largest', 'biggest']) else 'ASC'
            if 'average' in q or 'avg' in q:
                result['agg_func'] = 'AVG'
            elif 'total' in q or 'sum' in q:
                result['agg_func'] = 'SUM'
            elif any(w in q for w in AMOUNT_WORDS):
                result['agg_func'] = 'SUM'
            else:
                result['agg_func'] = 'COUNT'

        top_m = re.search(r'\b(?:top|first|bottom)(?:\s+(\d+))?\s+(\w+)', q)
        if top_m:
            n = int(top_m.group(1)) if top_m.group(1) else 10
            entity = top_m.group(2)
            METRIC = {'total', 'average', 'count', 'sum', 'max', 'min', 'amount', 'cost', 'rate'}
            if entity.lower() not in METRIC:
                result['group_by_terms'] = [_singularize(entity)]
                result['top_n'] = n
            if 'bottom' in q:
                result['order'] = 'ASC'
            if not result['agg_func']:
                if any(m in q for m in ['average', 'avg']):
                    result['agg_func'] = 'AVG'
                elif any(m in q for m in AMOUNT_WORDS) or 'total' in q:
                    result['agg_func'] = 'SUM'
                else:
                    result['agg_func'] = 'COUNT'

        if not result['agg_func']:
            if 'top' in q or 'bottom' in q or 'first' in q:
                result['agg_func'] = 'COUNT'
            else:
                for func, pats in cls.AGG_PATTERNS.items():
                    if any(re.search(p, q) for p in pats):
                        result['agg_func'] = func
                        break

        if not result['group_by_terms']:
            for pat in cls.GROUP_RE:
                m = re.search(pat, q)
                if m:
                    raw = m.group(1).strip()
                    parts = re.split(r'\band\b', raw)
                    noise = {'the', 'a', 'an', 'each', 'every', 'all', 'their', 'its', 'or', 'on', 'of', 'front', 'side', 'basis'}
                    terms = []
                    for part in parts:
                        words = [w for w in part.strip().split() if w.lower() not in noise and len(w) > 1]
                        terms.extend(words)
                    result['group_by_terms'] = terms
                    break
            if not result['group_by_terms'] and result['agg_func'] == 'COUNT':
                dm = re.search(r'([\w\s]+?)\s+(?:distribution|breakdown|spread|split)\b', q)
                if dm:
                    raw = dm.group(1).strip().split()
                    noise = {'the', 'a', 'an', 'show', 'get', 'give', 'me', 'display', 'list', 'what', 'is'}
                    terms = [t for t in raw if t.lower() not in noise and len(t) > 1]
                    if terms:
                        result['group_by_terms'] = terms

        if result['group_by_terms'] and not result['agg_func']:
            COUNTABLE = {'claims', 'encounters', 'diagnoses', 'prescriptions', 'referrals', 'members', 'patients', 'providers', 'visits', 'admissions'}
            if any(cp in q for cp in COUNTABLE):
                result['agg_func'] = 'COUNT'

        if 'chronic' in q and ('vs' in q or 'versus' in q):
            result['agg_func'] = result['agg_func'] or 'COUNT'
            result['group_by_terms'] = result['group_by_terms'] or ['chronic']

        if result['agg_func'] == 'COUNT' and 'total' in q and any(w in q for w in ['amount', 'cost', 'billed', 'paid', 'spend']):
            result['agg_func'] = 'SUM'

        if any(w in q for w in ['lowest', 'smallest', 'least', 'cheapest', 'fewest']):
            result['order'] = 'ASC'

        lm = re.search(r'\blimit\s+(\d+)\b', q)
        if lm and not result['top_n']:
            result['top_n'] = int(lm.group(1))

        return result

    @classmethod
    def resolve_tables(cls, question: str, resolved_cols: List[Dict]) -> List[str]:
        q = question.lower()
        FP_PAIRS = [('patient', 'inpatient'), ('patient', 'outpatient'), ('age', 'average'), ('age', 'usage'), ('rx', 'proxy')]
        suppressed = set()
        for kw, container in FP_PAIRS:
            if container in q and kw in container:
                suppressed.add(kw)

        kw_scores = defaultdict(float)
        for table, keywords in TABLE_KEYWORDS.items():
            for kw in keywords:
                if kw in suppressed:
                    continue
                if kw in q:
                    kw_scores[table] += 2.0

        col_scores = defaultdict(float)
        seen_per = defaultdict(set)
        for rc in resolved_cols:
            if rc['column'] not in seen_per[rc['table']] and rc['match_type'] in ('exact', 'synonym'):
                seen_per[rc['table']].add(rc['column'])
                col_scores[rc['table']] += 1.0

        scores = defaultdict(float)
        for t in set(list(kw_scores) + list(col_scores)):
            scores[t] = kw_scores.get(t, 0) + col_scores.get(t, 0)

        if not scores:
            return ['claims']

        sorted_t = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_t[0]
        result = [primary[0]]
        for table, score in sorted_t[1:]:
            if kw_scores.get(table, 0) >= 2.0 and score >= primary[1] * 0.4:
                result.append(table)
        return result[:3]

    @classmethod
    def extract_filters(cls, question: str, resolved_cols: List[Dict], schema: SchemaRegistry,
                        tables: List[str]) -> Tuple[List[str], str, List[str]]:
        q = question.lower()
        conditions, having = [], ""
        domain_tables = []

        q_c = q
        for typo, fix in CONCEPT_TYPOS.items():
            q_c = re.sub(r'\b' + re.escape(typo) + r'\b', fix, q_c)
        matched_concepts, handled_cols = set(), set()
        for concept in sorted(DOMAIN_CONCEPTS.keys(), key=len, reverse=True):
            if not re.search(r'\b' + re.escape(concept) + r'\b', q_c):
                continue
            if any(concept in mc and concept != mc for mc in matched_concepts):
                continue
            info = DOMAIN_CONCEPTS[concept]
            for cond in info['conds']:
                if cond not in conditions:
                    conditions.append(cond)
            for t in info['tables']:
                if t not in domain_tables:
                    domain_tables.append(t)
            matched_concepts.add(concept)

        for concept in matched_concepts:
            for cond in DOMAIN_CONCEPTS[concept]['conds']:
                for col in schema.column_to_tables:
                    if col.upper() in cond.upper():
                        handled_cols.add(col.upper())

        both_m = re.search(r'\b(\w+)\s+with\s+both\s+(\w+)\s+and\s+(\w+)', q)
        is_both = bool(both_m)
        if is_both:
            v1, v2 = both_m.group(2).upper(), both_m.group(3).upper()
            having = "HAVING COUNT(DISTINCT VISIT_TYPE) = 2"
            conditions.append(f"VISIT_TYPE IN ('{v1}', '{v2}')")

        for kw, (col, op, val) in STATUS_VALUES.items():
            if col in handled_cols:
                continue
            if is_both and col == 'VISIT_TYPE':
                continue
            if not re.search(r'\b' + re.escape(kw) + r'\b', q):
                continue
            if any(kw == cw and any(p in q for p in ps) for cw, ps in COLUMN_QUALIFIER_CONTEXTS.items()):
                continue
            if col == 'VISIT_TYPE' and re.search(r'\b(?:per|by)\s+(?:visit|emergency|outpatient|inpatient)', q):
                continue
            STATUS_LIKE = {'CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'}
            if col in STATUS_LIKE:
                if kw in ('active', 'inactive') and 'members' in tables:
                    if kw == 'active':
                        conditions.append("(DISENROLLMENT_DATE IS NULL OR DISENROLLMENT_DATE = '')")
                    else:
                        conditions.append("(DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != '')")
                    continue
                actual_col = col
                for t in tables:
                    if t in q or t.rstrip('s') in q:
                        if t in STATUS_COL_MAP:
                            actual_col = STATUS_COL_MAP[t]
                            break
                else:
                    for t in tables:
                        if t in STATUS_COL_MAP:
                            actual_col = STATUS_COL_MAP[t]
                            break
                if actual_col in schema.column_to_tables:
                    conditions.append(f"{actual_col} {op} '{val}'")
            else:
                col_tables = schema.column_to_tables.get(col, [])
                if col_tables:
                    col_in_current = any(t in tables for t in col_tables)
                    if col_in_current:
                        conditions.append(f"{col} {op} '{val}'")
                    else:
                        domain_tables.append(col_tables[0])
                        conditions.append(f"{col} {op} '{val}'")

        hm = re.search(r'\b(\w+)\s+(?:with|having|who have|that have)\s+(?:more than|over|at least|greater than|>)\s*(\d+)\s+(\w+)', q)
        if hm and not having:
            entity2 = hm.group(3)
            ENTITY_NAMES = {'members', 'member', 'patients', 'patient', 'providers', 'claims', 'encounters', 'diagnoses', 'prescriptions', 'referrals'}
            if entity2.rstrip('s') in {e.rstrip('s') for e in ENTITY_NAMES}:
                having = f"HAVING COUNT(*) > {hm.group(2)}"

        count_ctx = any(kw in q for kw in ['how many', 'count', 'number of', 'total number'])
        for pat, op in cls.COMPARISON_RE:
            m = re.search(pat, q)
            if m:
                vs = m.group(1).replace(',', '')
                try:
                    vn = float(vs)
                    if 2020 <= vn <= 2029:
                        continue
                except ValueError:
                    pass
                if having:
                    continue
                is_count = any(w in q for w in ['times', 'visits', 'admissions', 'encounters'])
                if not is_count and m:
                    after = q[m.end():m.end()+30].strip().split()[0] if q[m.end():m.end()+30].strip() else ''
                    ENTITY_COUNT = {'patients', 'members', 'people', 'providers', 'claims', 'prescriptions', 'referrals', 'diagnoses', 'encounters'}
                    if after.rstrip('s').rstrip('e') in {w.rstrip('s').rstrip('e') for w in ENTITY_COUNT}:
                        is_count = True
                if count_ctx or is_count:
                    having = f"HAVING COUNT(*) {op} {vs}"
                    continue

                try:
                    av = float(vs)
                    age_words = [r'\bmember', r'\bpatient', r'\bperson', r'\bpeople\b', r'\bfemale\b', r'\bmale\b', r'\bage\b', r'\bold\b', r'\byear']
                    if 1 <= av <= 120 and any(re.search(p, q) for p in age_words):
                        if not any(w in q for w in ['$', 'billed', 'paid', 'cost', 'amount', 'claim', 'score', 'los', 'days', 'duration']):
                            ai = int(av)
                            if op in ('>', '>='):
                                conditions.append(f"DATE_OF_BIRTH <= date('now', '-{ai} years')")
                            elif op in ('<', '<='):
                                conditions.append(f"DATE_OF_BIRTH >= date('now', '-{ai} years')")
                            elif op == 'BETWEEN':
                                v2 = m.group(2).replace(',', '') if m.lastindex >= 2 else str(ai + 10)
                                conditions.append(f"DATE_OF_BIRTH BETWEEN date('now', '-{v2} years') AND date('now', '-{ai} years')")
                            continue
                except (ValueError, IndexError):
                    pass

                num_col = cls._infer_numeric_col(q, resolved_cols, tables, is_count)
                if num_col:
                    if op == 'BETWEEN':
                        v2 = m.group(2).replace(',', '')
                        conditions.append(f"CAST({num_col} AS REAL) BETWEEN {vs} AND {v2}")
                    else:
                        conditions.append(f"CAST({num_col} AS REAL) {op} {vs}")

        ym = re.search(r'(?:in|over|during|for|from|year)?\s*(20[12]\d)\b', q)
        if ym:
            yr = ym.group(1)
            dc = TABLE_DATE_COL.get(tables[0], 'SERVICE_DATE') if tables else 'SERVICE_DATE'
            for rc in resolved_cols:
                if 'DATE' in rc['column'].upper():
                    dc = rc['column']
                    break
            conditions.append(f"{dc} LIKE '{yr}%'")

        for phrase, expr in TIME_FILTERS.items():
            if phrase in q:
                dc = TABLE_DATE_COL.get(tables[0], 'SERVICE_DATE') if tables else 'SERVICE_DATE'
                for rc in resolved_cols:
                    if 'DATE' in rc['column'].upper():
                        dc = rc['column']
                        break
                if isinstance(expr, tuple):
                    conditions.append(f"{dc} BETWEEN {expr[0]} AND {expr[1]}")
                else:
                    conditions.append(f"{dc} >= {expr}")
                break

        spec_m = re.search(r'\b(cardiology|dermatology|pediatrics|psychiatry|orthopedics|neurology|oncology|gastroenterology|pulmonology|rheumatology)\b', q, re.I)
        if spec_m:
            conditions.append(f"SPECIALTY LIKE '%{spec_m.group(1).upper()}%'")

        for loc, reg in sorted(REGION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(r'\b' + re.escape(loc) + r'\b', q):
                conditions.append(f"KP_REGION = '{reg}'")
                break

        if 'PLAN_TYPE' not in handled_cols:
            pm = re.search(r'\b(hmo|ppo|epo|pos|medicare|medicaid)\b', q, re.I)
            if pm:
                conditions.append(f"PLAN_TYPE = '{pm.group(1).upper()}'")

        return conditions, having, domain_tables

    @classmethod
    def _infer_numeric_col(cls, q: str, resolved_cols: List[Dict], tables: List[str], is_count: bool) -> Optional[str]:
        if is_count:
            return None
        for phrases, col in NUMERIC_COL_MAP:
            if any(p in q for p in phrases):
                return col
        if '$' in q or ('billed' in q and 'claim' not in q.split('billed')[0][-10:]):
            return 'BILLED_AMOUNT'
        if 'paid' in q and any(k in q for k in ['amount', 'average', 'avg', 'total', 'sum']):
            return 'PAID_AMOUNT'
        if 'cost' in q:
            return 'COST' if any(t == 'prescriptions' for t in tables) else 'BILLED_AMOUNT'
        for rc in resolved_cols:
            cn = rc['column'].lower()
            if any(k in cn for k in ['amount', 'cost', 'paid', 'billed', 'score', 'size', 'stay']):
                return rc['column']
        return None


def _singularize(word: str) -> str:
    w = word.lower()
    IRREGULAR = {'diagnoses': 'diagnosis', 'analyses': 'analysis', 'crises': 'crisis',
                 'theses': 'thesis', 'bases': 'basis', 'axes': 'axis'}
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w.endswith(('is', 'us', 'ss')):
        return w
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('ses') and len(w) > 4:
        return w[:-2]
    if w.endswith('es') and len(w) > 3:
        return w[:-1]
    if w.endswith('s') and len(w) > 3:
        return w[:-1]
    return w


class SQLComposer:

    def __init__(self, schema: SchemaRegistry):
        self.schema = schema

    def compose(self, question: str, resolved_cols: List[Dict], filters: List[str],
                agg_info: Dict, tables: List[str], having: str = "") -> str:
        if not tables:
            tables = ['claims']
        primary = tables[0]

        valid_cols = [rc for rc in resolved_cols if rc['table'] in tables and rc['column'] in self.schema.get_columns(rc['table'])]
        resolved_cols = valid_cols

        agg_func = agg_info.get('agg_func')
        group_terms = agg_info.get('group_by_terms', [])
        top_n = agg_info.get('top_n')
        order = agg_info.get('order', 'DESC')

        group_cols = self._resolve_groups(group_terms, resolved_cols, tables, question)
        if agg_func and top_n and not group_cols:
            group_cols = self._infer_group(question, tables)
        if agg_func and not group_cols and ' by ' in question.lower():
            group_cols = self._infer_group(question, tables)

        num_col = self._find_numeric(question, resolved_cols, tables)
        if num_col and not any(num_col in self.schema.get_columns(t) for t in tables):
            for tn in self.schema.tables:
                if num_col in self.schema.get_columns(tn):
                    tables.append(tn)
                    break

        select_parts, needs_gb = self._build_select(question, resolved_cols, agg_func, group_cols, num_col, tables, filters)

        from_clause = self._build_from(tables)

        if len(tables) > 1:
            select_parts = [self._qualify_expr(sp, tables) for sp in select_parts]
            filters = [self._qualify_expr(f, tables) for f in filters]
            group_cols = [self._qualify_col(gc, tables) for gc in group_cols]

        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        group_clause = f" GROUP BY {', '.join(group_cols)}" if needs_gb and group_cols else ""
        having_sql = f" {having}" if having and group_clause else ""

        order_clause = ""
        if agg_func and needs_gb:
            alias = select_parts[-1].split(' as ')[-1].strip() if ' as ' in select_parts[-1].lower() else None
            order_clause = f" ORDER BY {alias} {order}" if alias else f" ORDER BY 2 {order}"
        elif not agg_func:
            for rc in resolved_cols:
                if 'DATE' in rc['column'].upper() and rc['table'] in tables:
                    order_clause = f" ORDER BY {rc['column']} DESC"
                    break

        limit = ""
        if top_n:
            limit = f" LIMIT {top_n}"
        elif not agg_func and not needs_gb:
            limit = " LIMIT 50"
        elif needs_gb:
            limit = " LIMIT 30"

        return f"SELECT {', '.join(select_parts)} FROM {from_clause}{where}{group_clause}{having_sql}{order_clause}{limit};"

    def _resolve_groups(self, terms: List[str], cols: List[Dict], tables: List[str], question: str) -> List[str]:
        terms = [_singularize(t) if t.lower() != _singularize(t) and t.lower() not in SYNONYMS else t for t in terms]
        SORT_TERMS = {'volume', 'cost', 'amount', 'count', 'total', 'value', 'number', 'revenue', 'price'}
        FILTER_TERMS = {'medicaid', 'medicare', 'commercial', 'hmo', 'ppo', 'epo', 'hdhp',
                        'diabetic', 'diabetes', 'hypertension', 'cardiac', 'asthma', 'copd', 'depression', 'anxiety',
                        'obesity', 'cancer', 'newborn', 'infant', 'pediatric', 'elderly', 'senior', 'geriatric',
                        'foster', 'adolescent', 'teen', 'adult', 'minor', 'high', 'low', 'moderate', 'rising',
                        'inpatient', 'outpatient', 'emergency', 'telehealth', 'chronic', 'critical', 'severe',
                        'on', 'front', 'side', 'basis', 'end', 'perspective'}
        terms = [t for t in terms if t.lower() not in SORT_TERMS and t.lower() not in FILTER_TERMS]

        COMPOUND_MAP = {
            'plan type': 'PLAN_TYPE', 'claim type': 'CLAIM_TYPE', 'visit type': 'VISIT_TYPE',
            'referral type': 'REFERRAL_TYPE', 'appointment type': 'APPOINTMENT_TYPE',
            'diagnosis type': 'DIAGNOSIS_TYPE', 'medication class': 'MEDICATION_CLASS',
            'icd code': 'ICD10_CODE', 'icd10 code': 'ICD10_CODE', 'cpt code': 'CPT_CODE',
            'hcc category': 'HCC_CATEGORY', 'denial reason': 'DENIAL_REASON',
            'referral reason': 'REFERRAL_REASON', 'plan': 'PLAN_TYPE',
        }
        joined = ' '.join(t.lower() for t in terms)
        for phrase, col in sorted(COMPOUND_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if phrase in joined:
                for t in tables:
                    if col in self.schema.get_columns(t):
                        return [col]

        combined, i = [], 0
        while i < len(terms):
            found = False
            for j in range(len(terms), i, -1):
                comb = '_'.join(terms[i:j])
                for t in tables:
                    for col in self.schema.get_columns(t):
                        if comb.upper() in col or comb.lower() == col.lower().replace('_', ''):
                            combined.append(col)
                            i = j
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                combined.append(terms[i])
                i += 1
        if len(combined) < len(terms):
            terms = combined

        result = []
        for term in terms:
            found = False
            for rc in cols:
                if rc['table'] not in tables:
                    continue
                if term.lower() in rc['column'].lower() or term.lower() in rc.get('original_term', '').lower():
                    if rc['column'] not in result:
                        result.append(rc['column'])
                    found = True
                    break
            if found:
                continue
            for phrase, syn_cols in SYNONYMS.items():
                if term in phrase or phrase in term or _fuzzy_match(term, phrase):
                    sorted_cols = sorted(syn_cols, key=lambda c: (not any(t in tables for t in self.schema.column_to_tables.get(c, [])), c.endswith('_ID')))
                    for c in sorted_cols:
                        if any(t in tables for t in self.schema.column_to_tables.get(c, [])):
                            if c not in result:
                                result.append(c)
                            found = True
                            break
                if found:
                    break
            if found:
                continue
            for t in tables:
                candidates = []
                for col in self.schema.get_columns(t):
                    cl = col.lower()
                    tl = term.lower()
                    is_id = cl.endswith('_id')
                    if tl in cl:
                        candidates.append((col, 0, is_id))
                    elif _fuzzy_match(tl, cl.replace('_id', '').replace('_', '')):
                        candidates.append((col, 1, is_id))
                if candidates:
                    candidates.sort(key=lambda x: (x[1], x[2]))
                    if candidates[0][0] not in result:
                        result.append(candidates[0][0])
                    break
        return result

    def _infer_group(self, question: str, tables: List[str]) -> List[str]:
        q = question.lower()
        CONTEXT = {('referral', 'reason'): 'REFERRAL_REASON', ('denial', 'reason'): 'DENIAL_REASON',
                   ('hcc', 'category'): 'HCC_CATEGORY', ('icd', 'code'): 'ICD10_CODE'}
        for (e, s), col in CONTEXT.items():
            if e in q and s in q:
                for t in tables:
                    if col in self.schema.get_columns(t):
                        return [col]

        if re.search(r'\bby\s+status\b', q) or (re.search(r'\bstatus\b', q) and 'by' in q):
            for t in tables:
                scol = STATUS_COL_MAP.get(t)
                if scol and scol in self.schema.get_columns(t):
                    return [scol]

        if re.search(r'\bby\s+type\b', q):
            TYPE_MAP = {'claims': 'CLAIM_TYPE', 'encounters': 'VISIT_TYPE', 'referrals': 'REFERRAL_TYPE',
                        'appointments': 'APPOINTMENT_TYPE', 'diagnoses': 'DIAGNOSIS_TYPE'}
            for t in tables:
                tcol = TYPE_MAP.get(t)
                if tcol and tcol in self.schema.get_columns(t):
                    return [tcol]

        for entity, col in ENTITY_GROUP_COLS.items():
            if entity in q:
                for t in tables:
                    if col in self.schema.get_columns(t):
                        return [col]
        for t in tables:
            for c in self.schema.tables.get(t, []):
                dt = c.get('data_type', '').lower()
                cn = c['name'].lower()
                if dt in ('text', 'string', 'category') and not cn.endswith('_id') and 'date' not in cn and 'description' not in cn:
                    return [c['name']]
        return []

    def _find_numeric(self, question: str, cols: List[Dict], tables: List[str]) -> Optional[str]:
        q = question.lower()
        for phrases, col in NUMERIC_COL_MAP:
            if any(p in q for p in phrases):
                return col
        if any(k in q for k in ['billed', 'charge']) and 'paid' not in q:
            return 'BILLED_AMOUNT'
        if 'paid' in q and any(k in q for k in ['amount', 'average', 'avg', 'total', 'sum', 'max', 'min']) and 'billed' not in q:
            return 'PAID_AMOUNT'
        if any(k in q for k in ['payment', 'reimburs']):
            return 'PAID_AMOUNT'
        if 'cost' in q:
            return 'COST' if 'prescriptions' in tables else 'BILLED_AMOUNT'
        if 'allowed' in q:
            return 'ALLOWED_AMOUNT'
        if 'amount' in q and 'claims' in tables:
            return 'BILLED_AMOUNT'
        for rc in cols:
            ci = self.schema.get_column_info(rc['table'], rc['column'])
            if ci and ci.get('data_type', '').lower() in ('currency', 'float', 'integer', 'numeric'):
                return rc['column']
        for t in tables:
            for c in self.schema.tables.get(t, []):
                if c.get('data_type', '').lower() in ('currency', 'float'):
                    return c['name']
        return None

    def _build_select(self, question: str, cols: List[Dict], agg_func: Optional[str],
                      group_cols: List[str], num_col: Optional[str],
                      tables: List[str], filters: List[str]) -> Tuple[List[str], bool]:
        q = question.lower()

        if any(k in q for k in ['trend', 'over time', 'month by month', 'monthly', 'time series',
                                 'per month', 'per year', 'per quarter', 'per day', 'per week', 'growth']):
            date_col = self._find_date_col(q, tables)
            if date_col:
                month_expr = f"SUBSTR({date_col}, 1, 7)"
                group_cols.clear()
                group_cols.append(month_expr)
                return [f"{month_expr} as month", "COUNT(*) as count"], True

        if 'processing time' in q or ('processing' in q and any(w in q for w in ['average', 'avg', 'mean'])):
            te = "JULIANDAY(ADJUDICATED_DATE) - JULIANDAY(SUBMITTED_DATE)"
            agg = f"ROUND(AVG({te}), 2) as avg_processing_days"
            if group_cols:
                return group_cols + [agg, "COUNT(*) as claim_count"], True
            return [agg, "COUNT(*) as total_claims"], False

        if 'denial rate' in q or 'deny rate' in q:
            rate_cols = [
                "COUNT(*) as total_claims",
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_count",
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate"
            ]
            if group_cols:
                return group_cols + rate_cols, True
            return rate_cols, False

        if any(r in q for r in ['fill rate', 'approval rate', 'completion rate']):
            rate_cols = [
                "COUNT(*) as total",
                "SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) as approved_count",
                "ROUND(100.0 * SUM(CASE WHEN STATUS IN ('APPROVED', 'FILLED', 'COMPLETED') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as approval_rate"
            ]
            if group_cols:
                return group_cols + rate_cols, True
            return rate_cols, False

        pct_m = re.search(r'what\s+(?:percentage|percent|%|pct)\s+of\s+(\w+)\s+are\s+(\w+)', q)
        if pct_m:
            entity, status = pct_m.group(1), pct_m.group(2).upper()
            PCTMAP = {'FEMALE': ('GENDER', 'F'), 'MALE': ('GENDER', 'M'), 'CHRONIC': ('IS_CHRONIC', 'Y'),
                      'INPATIENT': ('VISIT_TYPE', 'INPATIENT'), 'OUTPATIENT': ('VISIT_TYPE', 'OUTPATIENT'),
                      'EMERGENCY': ('VISIT_TYPE', 'EMERGENCY'), 'TELEHEALTH': ('VISIT_TYPE', 'TELEHEALTH')}
            if status in PCTMAP:
                scol, sval = PCTMAP[status]
            elif 'claim' in entity:
                scol, sval = 'CLAIM_STATUS', status
            elif 'prescription' in entity:
                scol, sval = 'STATUS', status
            else:
                scol, sval = 'STATUS', status
            return [
                f"COUNT(*) as total",
                f"SUM(CASE WHEN {scol} = '{sval}' THEN 1 ELSE 0 END) as matching",
                f"ROUND(100.0 * SUM(CASE WHEN {scol} = '{sval}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as percentage"
            ], False

        if agg_func:
            if agg_func == 'COUNT':
                if group_cols:
                    if any(k in q for k in ['%', 'percent', 'percentage', 'proportion', 'share']):
                        return group_cols + ["COUNT(*) as count", "ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct"], True
                    return group_cols + ["COUNT(*) as count"], True
                return ["COUNT(*) as total_count"], False
            elif agg_func in ('AVG', 'SUM', 'MAX', 'MIN') and num_col:
                alias = f"{agg_func.lower()}_{num_col.lower()}"
                expr = f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {alias}"
                if group_cols:
                    return group_cols + [expr, "COUNT(*) as count"], True
                return [expr, "COUNT(*) as total_records"], False
            elif agg_func in ('AVG', 'SUM', 'MAX', 'MIN'):
                if group_cols:
                    return group_cols + ["COUNT(*) as count"], True
                return ["COUNT(*) as total_count"], False

        key_cols = self._get_key_columns(tables[0] if tables else 'claims')
        return key_cols, False

    def _find_date_col(self, q: str, tables: List[str]) -> Optional[str]:
        priority = []
        for t in tables:
            for c in self.schema.tables.get(t, []):
                cn = c['name'].upper()
                if 'DATE' not in cn or 'BIRTH' in cn:
                    continue
                if ('enrollment' in q or 'growth' in q or 'new member' in q) and 'ENROLLMENT' in cn:
                    return c['name']
                if 'SERVICE' in cn:
                    priority.insert(0, c['name'])
                else:
                    priority.append(c['name'])
        return priority[0] if priority else None

    def _get_key_columns(self, table: str) -> List[str]:
        cols = self.schema.get_columns(table)
        if not cols:
            return ['*']
        result = []
        for c in cols:
            ci = self.schema.get_column_info(table, c)
            dt = (ci or {}).get('data_type', 'text').lower()
            if len(result) >= 8:
                break
            result.append(c)
        return result if result else ['*']

    def _build_from(self, tables: List[str]) -> str:
        if len(tables) == 1:
            return tables[0]
        joins = self.schema.find_join_path(tables)
        base = tables[0]
        parts = [base]
        for t1, t2, col in joins:
            if '=' in col:
                left, right = col.split('=', 1)
                cond = f"{t1}.{left} = {t2}.{right}"
            else:
                cond = f"{t1}.{col} = {t2}.{col}"
            extras = self.schema.get_composite_conditions(t1, t2, col)
            for ec in extras:
                cond += f" AND {t1}.{ec} = {t2}.{ec}"
            parts.append(f"JOIN {t2} ON {cond}")
        return ' '.join(parts)

    def _qualify_col(self, col: str, tables: List[str]) -> str:
        if '(' in col or '.' in col:
            return col
        for t in tables:
            if col in self.schema.get_columns(t):
                return f"{t}.{col}"
        return col

    def _qualify_expr(self, expr: str, tables: List[str]) -> str:
        if not tables or len(tables) <= 1:
            return expr
        all_cols = set()
        for t in tables:
            for c in self.schema.get_columns(t):
                all_cols.add(c.upper())

        def repl(m):
            word = m.group(0)
            if word.upper() in all_cols:
                for t in tables:
                    if word.upper() in {c.upper() for c in self.schema.get_columns(t)}:
                        return f"{t}.{word}"
            return word

        result = re.sub(r'\b([A-Z_][A-Z0-9_]*)\b', repl, expr)
        return result


class DynamicSQLEngine:

    def __init__(self, catalog_dir: str = None, db_path: str = None, composite_joins: bool = None):
        if catalog_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            catalog_dir = os.path.join(os.path.dirname(script_dir), 'semantic_catalog')
        self.schema = SchemaRegistry(catalog_dir)
        if composite_joins is not None:
            SchemaRegistry.USE_COMPOSITE_JOINS = composite_joins
        elif db_path:
            self.schema.detect_composite_join_need(db_path)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_db = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_production.db')
            if os.path.exists(default_db):
                self.schema.detect_composite_join_need(default_db)
        self.resolver = ColumnResolver(self.schema)
        self.composer = SQLComposer(self.schema)

    def generate(self, question: str) -> Dict[str, Any]:
        question = normalize_typos(question)
        parts = self._split(question)
        if len(parts) > 1:
            results = [self._generate_single(p) for p in parts]
            combined = results[-1].copy()
            combined['multi_query'] = True
            combined['all_results'] = results
            combined['sql'] = ' UNION_SPLIT '.join(r['sql'] for r in results)
            return combined
        return self._generate_single(question)

    def _split(self, question: str) -> List[str]:
        q = question.strip()
        parts = re.split(r'[,;]\s*(?=(?:which|what|how|who|where|when|why|give|show|list|get|find|tell|count|total)\b)', q, flags=re.I)
        if len(parts) == 1:
            parts = re.split(r'\.\s*(?=(?:which|what|how|who|where|when|why|give|show|list|get|find|tell|count|total)\b)', q, flags=re.I)
        if len(parts) == 1:
            parts = re.split(r'\s+and\s+(?=(?:which|what|how|who|also|then)\b)', q, flags=re.I)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        return parts if len(parts) > 1 else [question]

    def _generate_single(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        resolved_cols = self.resolver.resolve(question)
        tables = QueryParser.resolve_tables(question, resolved_cols)
        resolved_cols = self.resolver.resolve(question, hint_tables=tables)
        agg_info = QueryParser.detect_agg(question)

        for kpi_name, kpi in KPI_TEMPLATES.items():
            if any(re.search(trigger, q) for trigger in kpi['triggers']):
                return self._render_kpi(kpi_name, kpi, question, resolved_cols, agg_info, tables)

        ratio = self._try_ratio(question, tables, resolved_cols)
        if ratio:
            return ratio

        cost_cmp = self._try_cost_comparison(question, tables, resolved_cols)
        if cost_cmp:
            return cost_cmp

        nested = self._try_nested_count(question)
        if nested:
            return nested

        special = self._try_special_patterns(question, tables, resolved_cols, agg_info)
        if special:
            return special

        filters, having, domain_tables = QueryParser.extract_filters(question, resolved_cols, self.schema, tables)
        for dt in domain_tables:
            if dt not in tables:
                tables.append(dt)

        if having and not agg_info.get('agg_func'):
            agg_info['agg_func'] = 'COUNT'
        if having and not agg_info.get('group_by_terms'):
            subject = q.split(' with ')[0] if ' with ' in q else q.split(' having ')[0] if ' having ' in q else q
            HAVING_MAP = {'member': 'MEMBER_ID', 'patient': 'MEMBER_ID', 'provider': 'NPI', 'doctor': 'NPI',
                          'facility': 'FACILITY', 'department': 'DEPARTMENT', 'region': 'KP_REGION', 'specialty': 'SPECIALTY'}
            for entity, col in HAVING_MAP.items():
                if entity in subject or entity + 's' in subject:
                    agg_info['group_by_terms'] = [entity]
                    break

        DIM_TABLES = {'providers': 'claims', 'members': 'claims'}
        if agg_info.get('agg_func') and len(tables) == 1 and tables[0] in DIM_TABLES:
            counting_self = any(f'count of {e}' in q or f'many {e}' in q or f'number of {e}' in q or f'total {e}' in q
                                for e, tbl in {'member': 'members', 'patient': 'members', 'provider': 'providers'}.items()
                                if tbl == tables[0])
            if domain_tables and tables[0] in domain_tables:
                counting_self = True
            if not counting_self:
                ft = DIM_TABLES[tables[0]]
                if ft not in tables:
                    tables.append(ft)

        tables = self._prune_tables(tables, filters, agg_info, domain_tables, q)

        sql = self.composer.compose(question, resolved_cols, filters, agg_info, tables, having)
        confidence = self._confidence(resolved_cols, tables, filters, agg_info)
        explanation = self._explain(question, resolved_cols, tables, filters, agg_info)
        return {'sql': sql, 'tables_used': tables, 'columns_resolved': resolved_cols,
                'filters': filters, 'agg_info': agg_info, 'confidence': confidence, 'explanation': explanation}

    def _render_kpi(self, name: str, kpi: Dict, question: str, cols: List[Dict],
                    agg_info: Dict, tables: List[str]) -> Dict[str, Any]:
        q = question.lower()
        kpi_tables = kpi['tables'] or tables

        if kpi.get('static'):
            return {'sql': kpi['sql'], 'tables_used': kpi_tables, 'columns_resolved': cols,
                    'filters': [], 'agg_info': {'agg_func': 'RATE', 'group_by_terms': [], 'top_n': None, 'order': 'DESC'},
                    'confidence': 0.95, 'explanation': f"Computing {name.replace('_', ' ')} KPI."}

        group_terms = agg_info.get('group_by_terms', [])
        composer = SQLComposer(self.schema)
        group_cols = composer._resolve_groups(group_terms, cols, kpi_tables, question)

        if not group_cols and group_terms:
            group_cols = composer._infer_group(question, kpi_tables)

        if any(p in q for p in ['over time', 'trend', 'month by month', 'monthly', 'by month', 'per month']):
            date_col = composer._find_date_col(q, kpi_tables)
            if date_col:
                group_cols = [f"SUBSTR({date_col}, 1, 7)"]
        elif any(p in q for p in ['by quarter', 'quarterly', 'per quarter']):
            date_col = composer._find_date_col(q, kpi_tables)
            if date_col:
                group_cols = [f"SUBSTR({date_col}, 1, 4) || '-Q' || ((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)"]
        elif 'seasonal' in q or 'by season' in q:
            date_col = composer._find_date_col(q, kpi_tables)
            if date_col:
                group_cols = [f"SUBSTR({date_col}, 6, 2)"]

        group_select = ', '.join(g + (' as period' if 'SUBSTR' in g else '') for g in group_cols) + ', ' if group_cols else ''
        group_by_clause = ' GROUP BY ' + ', '.join(group_cols) if group_cols else ''
        order_clause = f" ORDER BY {group_cols[0]}" if group_cols else ''

        filters, _, domain_tables = QueryParser.extract_filters(question, cols, self.schema, kpi_tables)

        visit_type_filter = "VISIT_TYPE = 'INPATIENT'"
        if name == 'alos':
            VT_MAP = {'emergency': 'EMERGENCY', 'outpatient': 'OUTPATIENT', 'telehealth': 'TELEHEALTH',
                       'urgent': 'URGENT_CARE', 'home health': 'HOME_HEALTH'}
            for word, vt in VT_MAP.items():
                if word in q:
                    visit_type_filter = f"VISIT_TYPE = '{vt}'"
                    break
            filters = [f for f in filters if 'VISIT_TYPE' not in f]

        where = ' WHERE ' + ' AND '.join(filters) if filters else ''

        year_m = re.search(r'\b(202[0-9])\b', q)
        year_filter = f" AND SERVICE_DATE LIKE '{year_m.group(1)}%'" if year_m else ''

        group_col = 'CLAIM_TYPE'
        if name == 'processing_time' and group_cols:
            group_col = group_cols[0]
        elif name == 'processing_time':
            for entity, col in ENTITY_GROUP_COLS.items():
                if entity in q and col in self.schema.get_columns('claims'):
                    group_col = col
                    break

        table = kpi_tables[0] if kpi_tables else 'claims'
        sql = kpi['sql'].format(
            group=group_select, where=where, group_by=group_by_clause, order=order_clause,
            table=table, year_filter=year_filter,
            extra_where=where.replace(' WHERE ', ' AND ') if where else '',
            visit_type_filter=visit_type_filter,
            group_col=group_col
        )
        return {'sql': sql, 'tables_used': kpi_tables, 'columns_resolved': cols,
                'filters': filters, 'agg_info': agg_info, 'confidence': 0.95,
                'explanation': f"Computing {name.replace('_', ' ')} KPI.{' Grouped by ' + ', '.join(group_cols) + '.' if group_cols else ''}"}

    def _try_ratio(self, question: str, tables: List[str], cols: List[Dict]) -> Optional[Dict]:
        q = question.lower()
        if not any(kw in q for kw in ('ratio', 'vs', 'versus', 'compare', 'comparison', 'proportion')):
            return None
        if any(kw in q for kw in ('cost', 'amount', 'billed', 'paid', 'allowed', 'spend', 'expense', 'charge')):
            return None

        found = []
        for term in sorted(RATIO_VALUE_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                found.append(term)
        if len(found) < 2:
            return None

        info_a, info_b = RATIO_VALUE_MAP[found[0]], RATIO_VALUE_MAP[found[1]]
        if info_a[1] != info_b[1]:
            return None

        table, column = info_a[0], info_a[1]
        val_a, val_b = info_a[2], info_b[2]
        la, lb = val_a.lower().replace('_', ' '), val_b.lower().replace('_', ' ')

        gc = None
        gm = re.search(r'\bby\s+(\w[\w\s]*?)(?:\s*$)', q)
        if gm:
            gt = gm.group(1).strip().upper().replace(' ', '_')
            t_cols = [c.upper() for c in self.schema.get_columns(table)]
            gc = gt if gt in t_cols else ('KP_' + gt if 'KP_' + gt in t_cols else None)

        sel = []
        if gc:
            sel.append(gc)
        sel.extend([
            f"SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) as {la.replace(' ', '_')}_count",
            f"SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END) as {lb.replace(' ', '_')}_count",
            "COUNT(*) as total_count",
            f"ROUND(100.0 * SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {la.replace(' ', '_')}_pct",
            f"ROUND(100.0 * SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {lb.replace(' ', '_')}_pct",
            f"ROUND(CAST(SUM(CASE WHEN {column} = '{val_a}' THEN 1 ELSE 0 END) AS REAL) / NULLIF(SUM(CASE WHEN {column} = '{val_b}' THEN 1 ELSE 0 END), 0), 4) as ratio",
        ])
        sql = f"SELECT {', '.join(sel)} FROM {table}"
        if gc:
            sql += f" GROUP BY {gc} ORDER BY ratio DESC"
        sql += " LIMIT 50;"
        return {'sql': sql, 'tables_used': [table], 'columns_resolved': cols, 'filters': [],
                'agg_info': {'agg_func': 'RATIO', 'group_by_terms': [gc] if gc else [], 'top_n': None, 'order': 'DESC'},
                'confidence': 0.95, 'explanation': f"Computing ratio of {la} to {lb}.{' Grouped by ' + gc + '.' if gc else ''}"}

    def _try_cost_comparison(self, question: str, tables: List[str], cols: List[Dict]) -> Optional[Dict]:
        q = question.lower()
        if not (any(kw in q for kw in ('compare', 'vs', 'versus', 'comparison')) and
                any(kw in q for kw in ('cost', 'amount', 'billed', 'paid', 'allowed', 'spend', 'expense', 'charge'))):
            return None

        found = []
        for term in sorted(RATIO_VALUE_MAP.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                found.append((term, RATIO_VALUE_MAP[term]))
        if len(found) < 2 or found[0][1][1] != found[1][1][1]:
            return None

        table, column = found[0][1][0], found[0][1][1]
        val_a, val_b = found[0][1][2], found[1][1][2]
        cost_col = 'PAID_AMOUNT' if 'paid' in q else 'ALLOWED_AMOUNT' if 'allowed' in q else 'BILLED_AMOUNT'

        if table == 'encounters':
            time_bucket = ''
            time_group = ''
            if 'quarter' in q:
                time_bucket = f"SUBSTR(c.SERVICE_DATE, 1, 4) || '-Q' || ((CAST(SUBSTR(c.SERVICE_DATE, 6, 2) AS INTEGER) - 1) / 3 + 1) as quarter, "
                time_group = f", SUBSTR(c.SERVICE_DATE, 1, 4) || '-Q' || ((CAST(SUBSTR(c.SERVICE_DATE, 6, 2) AS INTEGER) - 1) / 3 + 1)"
            elif any(w in q for w in ['month', 'monthly']):
                time_bucket = "SUBSTR(c.SERVICE_DATE, 1, 7) as month, "
                time_group = ", SUBSTR(c.SERVICE_DATE, 1, 7)"
            sql = (f"SELECT {time_bucket}e.{column}, COUNT(*) as visit_count, "
                   f"ROUND(AVG(CAST(c.{cost_col} AS REAL)), 2) as avg_{cost_col.lower()}, "
                   f"ROUND(SUM(CAST(c.{cost_col} AS REAL)), 2) as total_{cost_col.lower()} "
                   f"FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                   f"WHERE e.{column} IN ('{val_a.replace(chr(39), chr(39)*2)}', '{val_b.replace(chr(39), chr(39)*2)}') GROUP BY e.{column}{time_group} ORDER BY 1;")
            used = ['encounters', 'claims']
        else:
            sql = (f"SELECT {column}, COUNT(*) as record_count, "
                   f"AVG(CAST({cost_col} AS REAL)) as avg_{cost_col.lower()}, "
                   f"SUM(CAST({cost_col} AS REAL)) as total_{cost_col.lower()} "
                   f"FROM {table} WHERE {column} IN ('{val_a.replace(chr(39), chr(39)*2)}', '{val_b.replace(chr(39), chr(39)*2)}') GROUP BY {column};")
            used = [table]
        return {'sql': sql, 'tables_used': used, 'columns_resolved': cols, 'filters': [],
                'agg_info': {'agg_func': 'AVG', 'group_by_terms': [column], 'top_n': None, 'order': 'DESC'},
                'confidence': 0.95, 'explanation': f"Comparing {cost_col} between {val_a} and {val_b}."}

    def _try_nested_count(self, question: str) -> Optional[Dict]:
        q = question.lower()
        m = re.search(
            r'\b(?:count|how many|number of)\b.*?'
            r'\b(?:person|people|member|members|patient|patients)\b.*?'
            r'\b(?:visited|went to|had|received|been to|went|seen)\s+(?:the\s+)?(\w+)\s+'
            r'(?:more than|over|at least|greater than|>=?)\s*(\d+)\s*(?:times?|visits?|admissions?|encounters?)',
            q, re.I)
        if not m:
            m = re.search(r'\b(?:count|how many)\b.*?\b(?:one\s+)?(?:person|member|patient)\b.*?\b(?:visited|went|had)\s+(\w+)\s+(?:more than|over|at least|>)\s*(\d+)\s*(?:times?|visits?)', q, re.I)
        if not m:
            return None

        vt_word = m.group(1).upper()
        threshold = m.group(2)
        VT_MAP = {'EMERGENCY': 'EMERGENCY', 'ER': 'EMERGENCY', 'ED': 'EMERGENCY',
                   'INPATIENT': 'INPATIENT', 'HOSPITAL': 'INPATIENT', 'OUTPATIENT': 'OUTPATIENT',
                   'TELEHEALTH': 'TELEHEALTH', 'VIRTUAL': 'TELEHEALTH', 'URGENT': 'URGENT_CARE'}
        vt = VT_MAP.get(vt_word, vt_word)
        sql = (f"SELECT COUNT(*) as member_count FROM (SELECT MEMBER_ID FROM encounters "
               f"WHERE VISIT_TYPE = '{vt}' GROUP BY MEMBER_ID HAVING COUNT(*) > {threshold}) sub;")
        return {'sql': sql, 'tables_used': ['encounters'], 'columns_resolved': [],
                'filters': [], 'agg_info': {'agg_func': 'COUNT', 'group_by_terms': ['MEMBER_ID'], 'top_n': None, 'order': 'DESC'},
                'confidence': 0.95, 'explanation': f"Counting members who visited {vt} more than {threshold} times."}

    def _try_special_patterns(self, question: str, tables: List[str], cols: List[Dict], agg_info: Dict) -> Optional[Dict]:
        q = question.lower()

        older = re.search(r'\b(?:older|elder)\s+than\s+(\d+)', q)
        younger = re.search(r'\b(?:younger)\s+than\s+(\d+)', q)
        if (older or younger) and any(w in q for w in ['member', 'patient', 'person', 'people']):
            if 'members' not in tables:
                tables.append('members')
            age = int((older or younger).group(1))
            op = '<=' if older else '>='
            filters, having, dt = QueryParser.extract_filters(question, cols, self.schema, tables)
            cond = f"DATE_OF_BIRTH {op} date('now', '-{age} years')"
            if cond not in filters:
                filters.append(cond)
            for t in dt:
                if t not in tables:
                    tables.append(t)
            sql = self.composer.compose(question, cols, filters, agg_info, tables, having)
            return self._result(sql, tables, cols, filters, agg_info, question)

        if re.search(r'\b(?:youngest|oldest)\s+(?:member|patient|person|people)', q):
            od = 'DESC' if 'youngest' in q else 'ASC'
            limit = agg_info.get('top_n') or 10
            sql = f"SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER, KP_REGION FROM members ORDER BY DATE_OF_BIRTH {od} LIMIT {limit};"
            return self._result(sql, ['members'], cols, [], agg_info, question)

        if re.search(r'\bage\b', q) and any(w in q for w in ['average', 'avg', 'mean', 'median']):
            if 'members' not in tables:
                tables.append('members')
            filters, having, dt = QueryParser.extract_filters(question, cols, self.schema, tables)
            for t in dt:
                if t not in tables:
                    tables.append(t)
            where = f" WHERE {' AND '.join(filters)}" if filters else ""
            sql = (f"SELECT ROUND(AVG((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25), 1) as avg_age, "
                   f"COUNT(*) as member_count FROM members{where};")
            return self._result(sql, ['members'], cols, filters, agg_info, question)

        top_by_m = re.search(r'\btop\s+(\d+)\s+(\w+)\s+by\s+(\w+)\s+(?:count|volume|number)', q)
        if top_by_m:
            limit = int(top_by_m.group(1))
            entity = top_by_m.group(2).lower().rstrip('s')
            metric_table = top_by_m.group(3).lower().rstrip('s')
            ENTITY_COL = {'provider': 'RENDERING_NPI', 'doctor': 'RENDERING_NPI', 'physician': 'RENDERING_NPI',
                          'member': 'MEMBER_ID', 'patient': 'MEMBER_ID',
                          'facility': 'FACILITY', 'department': 'DEPARTMENT',
                          'specialty': 'SPECIALTY', 'region': 'KP_REGION',
                          'medication': 'MEDICATION_NAME', 'drug': 'MEDICATION_NAME'}
            ENTITY_COL_ALT = {'provider': [('claims', 'RENDERING_NPI'), ('encounters', 'RENDERING_NPI'), ('providers', 'NPI')]}
            TABLE_MAP = {'claim': 'claims', 'encounter': 'encounters', 'prescription': 'prescriptions',
                         'referral': 'referrals', 'diagnosis': 'diagnoses', 'appointment': 'appointments'}
            gcol = ENTITY_COL.get(entity)
            mtable = TABLE_MAP.get(metric_table, 'claims')
            if gcol and gcol in self.schema.get_columns(mtable):
                sql = f"SELECT {gcol}, COUNT(*) as total_count FROM {mtable} GROUP BY {gcol} ORDER BY total_count DESC LIMIT {limit};"
                return self._result(sql, [mtable], cols, [], agg_info, question)
            for alt_table, alt_col in ENTITY_COL_ALT.get(entity, []):
                if mtable == alt_table and alt_col in self.schema.get_columns(mtable):
                    sql = f"SELECT {alt_col}, COUNT(*) as total_count FROM {mtable} GROUP BY {alt_col} ORDER BY total_count DESC LIMIT {limit};"
                    return self._result(sql, [mtable], cols, [], agg_info, question)

        bm = re.search(r'\b(?:busiest|most active|highest volume)\s+(\w+)', q)
        if bm:
            entity = bm.group(1).upper()
            MAP = {'FACILITIES': 'FACILITY', 'FACILITY': 'FACILITY', 'DEPARTMENTS': 'DEPARTMENT',
                   'DEPARTMENT': 'DEPARTMENT', 'PROVIDERS': 'RENDERING_NPI', 'REGIONS': 'KP_REGION'}
            col = MAP.get(entity, entity.rstrip('S'))
            table = 'encounters' if col in ('FACILITY', 'DEPARTMENT') else 'claims'
            limit = agg_info.get('top_n') or 10
            sql = f"SELECT {col}, COUNT(*) as total_count FROM {table} GROUP BY {col} ORDER BY total_count DESC LIMIT {limit};"
            return self._result(sql, [table], cols, [], agg_info, question)

        mm = re.search(r'\b(?:member|patient)s?\s+with\s+(?:multiple|several|many|more than \d+)\s+(\w+)', q)
        if mm:
            entity = mm.group(1).lower()
            fact = 'encounters' if 'encounter' in entity or 'visit' in entity else 'claims'
            th_m = re.search(r'more than\s+(\d+)', q)
            th = int(th_m.group(1)) if th_m else 1
            sql = (f"SELECT m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, COUNT(*) as record_count "
                   f"FROM members m JOIN {fact} f ON m.MEMBER_ID = f.MEMBER_ID "
                   f"GROUP BY m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME HAVING COUNT(*) > {th} ORDER BY record_count DESC LIMIT 50;")
            return self._result(sql, ['members', fact], cols, [], agg_info, question)

        having_m = re.search(r'\b(\w+)\s+with\s+(?:more than|over|at least|greater than|above)\s+(\d+)\s+(\w+)', q)
        if having_m and having_m.group(1).lower() not in ('member', 'patient', 'members', 'patients'):
            entity = having_m.group(1).lower().rstrip('s')
            threshold = int(having_m.group(2))
            fact_word = having_m.group(3).lower().rstrip('s')
            ENTITY_COL_H = {'provider': 'RENDERING_NPI', 'doctor': 'RENDERING_NPI', 'physician': 'RENDERING_NPI',
                            'facility': 'FACILITY', 'department': 'DEPARTMENT', 'specialty': 'SPECIALTY',
                            'region': 'KP_REGION', 'medication': 'MEDICATION_NAME'}
            TABLE_MAP_H = {'claim': 'claims', 'encounter': 'encounters', 'prescription': 'prescriptions',
                           'referral': 'referrals', 'diagnosis': 'diagnoses', 'appointment': 'appointments'}
            gcol_h = ENTITY_COL_H.get(entity)
            mtable_h = TABLE_MAP_H.get(fact_word, 'claims')
            if gcol_h and gcol_h in self.schema.get_columns(mtable_h):
                sql = (f"SELECT {gcol_h}, COUNT(*) as total_count FROM {mtable_h} "
                       f"GROUP BY {gcol_h} HAVING COUNT(*) > {threshold} ORDER BY total_count DESC LIMIT 50;")
                return self._result(sql, [mtable_h], cols, [], agg_info, question)

        rank_m = re.search(r'\b(?:which|what)\s+(\w+)\s+(?:have|has|show|with)\s+the\s+(?:highest|lowest|most|least|biggest|smallest)\s+(?:average\s+|avg\s+|total\s+|mean\s+)?(\w+)', q)
        if rank_m:
            entity_w = _singularize(rank_m.group(1).lower())
            metric_w = rank_m.group(2).lower()
            order_dir = 'ASC' if any(w in q for w in ['lowest', 'least', 'smallest', 'cheapest']) else 'DESC'
            agg_fn = 'SUM' if 'total' in q else 'AVG'
            limit = agg_info.get('top_n') or 10
            RANK_ENTITY = {'specialty': ('SPECIALTY', 'providers'), 'department': ('DEPARTMENT', 'encounters'),
                           'facility': ('FACILITY', 'claims'), 'region': ('KP_REGION', 'claims'),
                           'provider': ('RENDERING_NPI', 'claims'), 'medication': ('MEDICATION_NAME', 'prescriptions'),
                           'diagnosis': ('ICD10_CODE', 'diagnoses'), 'plan': ('PLAN_TYPE', 'claims')}
            RANK_METRIC = {'billed': 'BILLED_AMOUNT', 'paid': 'PAID_AMOUNT', 'allowed': 'ALLOWED_AMOUNT',
                           'cost': 'BILLED_AMOUNT', 'amount': 'BILLED_AMOUNT', 'spend': 'BILLED_AMOUNT',
                           'stay': 'LENGTH_OF_STAY', 'los': 'LENGTH_OF_STAY'}
            entity_info = RANK_ENTITY.get(entity_w)
            metric_col = RANK_METRIC.get(metric_w)
            if entity_info and metric_col:
                gcol_r, table_r = entity_info
                if metric_col in self.schema.get_columns(table_r):
                    sql = (f"SELECT {gcol_r}, ROUND({agg_fn}(CAST({metric_col} AS REAL)), 2) as {agg_fn.lower()}_{metric_col.lower()}, "
                           f"COUNT(*) as record_count FROM {table_r} GROUP BY {gcol_r} "
                           f"ORDER BY {agg_fn.lower()}_{metric_col.lower()} {order_dir} LIMIT {limit};")
                    return self._result(sql, [table_r], cols, [], agg_info, question)
                for join_table in ['claims', 'encounters']:
                    if metric_col in self.schema.get_columns(join_table) and join_table != table_r:
                        join_path = self.schema.find_join_path([table_r, join_table])
                        if join_path:
                            jcol = join_path[0][2]
                            on_clause = f"a.{jcol} = b.{jcol}"
                            if table_r == 'providers' and join_table == 'claims':
                                on_clause = "a.NPI = b.RENDERING_NPI"
                            sql = (f"SELECT a.{gcol_r}, ROUND({agg_fn}(CAST(b.{metric_col} AS REAL)), 2) as {agg_fn.lower()}_{metric_col.lower()}, "
                                   f"COUNT(*) as record_count FROM {table_r} a JOIN {join_table} b ON {on_clause} "
                                   f"GROUP BY a.{gcol_r} ORDER BY {agg_fn.lower()}_{metric_col.lower()} {order_dir} LIMIT {limit};")
                            return self._result(sql, [table_r, join_table], cols, [], agg_info, question)

        expensive_m = re.search(r'\btop\s+(\d+)\s+(?:most\s+)?(?:expensive|costly|high.cost)\s+(\w+)', q)
        if expensive_m:
            limit = int(expensive_m.group(1))
            entity_w = _singularize(expensive_m.group(2).lower())
            EXPENSIVE_MAP = {'diagnosis': ('ICD10_CODE', 'claims'), 'icd': ('ICD10_CODE', 'claims'),
                             'procedure': ('CPT_CODE', 'claims'), 'cpt': ('CPT_CODE', 'claims'),
                             'medication': ('MEDICATION_NAME', 'prescriptions'), 'drug': ('MEDICATION_NAME', 'prescriptions'),
                             'provider': ('RENDERING_NPI', 'claims'), 'facility': ('FACILITY', 'claims')}
            info = EXPENSIVE_MAP.get(entity_w)
            if info:
                gcol_e, table_e = info
                cost_col = 'COST' if table_e == 'prescriptions' else 'BILLED_AMOUNT'
                sql = (f"SELECT {gcol_e}, ROUND(SUM(CAST({cost_col} AS REAL)), 2) as total_cost, "
                       f"ROUND(AVG(CAST({cost_col} AS REAL)), 2) as avg_cost, COUNT(*) as record_count "
                       f"FROM {table_e} GROUP BY {gcol_e} ORDER BY total_cost DESC LIMIT {limit};")
                return self._result(sql, [table_e], cols, [], agg_info, question)

        if re.search(r'\b(?:seasonal|pattern|trend|monthly|quarterly)\b', q) and not any(kw in q for kw in ['denial rate', 'fill rate', 'no-show', 'processing time', 'readmission', 'pmpm']):
            table_for_trend = tables[0] if tables else 'claims'
            date_col = self.composer._find_date_col(q, tables)
            if date_col:
                filters, _, _ = QueryParser.extract_filters(question, cols, self.schema, tables)
                where = ' WHERE ' + ' AND '.join(filters) if filters else ''
                if 'quarter' in q:
                    bucket = f"SUBSTR({date_col}, 1, 4) || '-Q' || ((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)"
                else:
                    bucket = f"SUBSTR({date_col}, 1, 7)"
                sql = f"SELECT {bucket} as period, COUNT(*) as count FROM {table_for_trend}{where} GROUP BY period ORDER BY period LIMIT 50;"
                return self._result(sql, tables, cols, filters, agg_info, question)

        id_m = (re.search(r'\b(?:for|of|from)\s+(?:member|patient)\s+([A-Za-z0-9_]+)\b', question, re.I) or
                re.search(r'\bmember\s+(?:id\s+)?([A-Za-z]*\d{2,}[A-Za-z0-9]*)\b', question, re.I) or
                re.search(r'\b(?:look\s*up|find|search|get)\s+member\s+(?:id\s+)?([A-Za-z0-9_]+)\b', question, re.I))
        if id_m:
            mid = id_m.group(1).upper()
            filters, having, dt = QueryParser.extract_filters(question, cols, self.schema, tables)
            if not any(mid in f for f in filters):
                filters.append(f"MEMBER_ID = '{mid.replace(chr(39), chr(39)*2)}'")
            for t in dt:
                if t not in tables:
                    tables.append(t)
            sql = self.composer.compose(question, cols, filters, agg_info, tables, having)
            return self._result(sql, tables, cols, filters, agg_info, question)

        if re.search(r'\b(?:by\s+)?age\s+(?:group|bucket|bracket|bin|distribution|breakdown)', q) or \
           (re.search(r'\bage\b', q) and re.search(r'\b(?:by|distribution|breakdown|group)\b', q)):
            if 'members' not in tables:
                tables.append('members')
            filters, having, dt = QueryParser.extract_filters(question, cols, self.schema, tables)
            for t in dt:
                if t not in tables:
                    tables.append(t)
            where = f" WHERE {' AND '.join(filters)}" if filters else ""
            age_group_expr = """CASE
                WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 18 THEN '0-17'
                WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 35 THEN '18-34'
                WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 50 THEN '35-49'
                WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 65 THEN '50-64'
                ELSE '65+' END"""
            sql = f"""SELECT {age_group_expr} as age_group, COUNT(*) as member_count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members{where.replace(' WHERE ', ' WHERE ')}), 2) as percentage
FROM members{where} GROUP BY age_group ORDER BY CASE age_group WHEN '0-17' THEN 1 WHEN '18-34' THEN 2 WHEN '35-49' THEN 3 WHEN '50-64' THEN 4 ELSE 5 END;"""
            return self._result(sql, ['members'], cols, filters, agg_info, question)

        if re.search(r'\b(?:claims|members|patients|people|encounters)\b.*\bby\b.*\b(?:demographics|demographic)\b', q) or \
           re.search(r'\b(?:claims|members|patients|people|encounters)\s*%\s*by\s+(?:demographics|demographic)', q):
            if any(t in q for t in ['claim', 'member', 'patient', 'encounter']):
                fact_table = 'claims' if 'claim' in q else ('encounters' if 'encounter' in q else 'members')
                if fact_table == 'members':
                    sql = f"""SELECT GENDER, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 2) as pct FROM members GROUP BY GENDER UNION ALL
SELECT age_group, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 2) as pct FROM (SELECT CASE WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 18 THEN '0-17' WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 35 THEN '18-34' WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 50 THEN '35-49' WHEN (julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 < 65 THEN '50-64' ELSE '65+' END as age_group FROM members) GROUP BY age_group UNION ALL
SELECT RACE, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 2) as pct FROM members WHERE RACE IS NOT NULL AND RACE != '' GROUP BY RACE UNION ALL
SELECT PLAN_TYPE, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 2) as pct FROM members GROUP BY PLAN_TYPE;"""
                    return self._result(sql, ['members'], cols, [], agg_info, question)
                else:
                    join_col = 'MEMBER_ID' if fact_table == 'encounters' else 'MEMBER_ID'
                    sql = f"""SELECT m.GENDER, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {fact_table}), 2) as pct FROM {fact_table} f JOIN members m ON f.MEMBER_ID = m.MEMBER_ID GROUP BY m.GENDER UNION ALL
SELECT age_group, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {fact_table}), 2) as pct FROM (SELECT f.*, CASE WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 18 THEN '0-17' WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 35 THEN '18-34' WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 50 THEN '35-49' WHEN (julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 < 65 THEN '50-64' ELSE '65+' END as age_group FROM {fact_table} f JOIN members m ON f.MEMBER_ID = m.MEMBER_ID) GROUP BY age_group UNION ALL
SELECT m.RACE, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {fact_table}), 2) as pct FROM {fact_table} f JOIN members m ON f.MEMBER_ID = m.MEMBER_ID WHERE m.RACE IS NOT NULL AND m.RACE != '' GROUP BY m.RACE UNION ALL
SELECT m.PLAN_TYPE, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {fact_table}), 2) as pct FROM {fact_table} f JOIN members m ON f.MEMBER_ID = m.MEMBER_ID GROUP BY m.PLAN_TYPE;"""
                    return self._result(sql, [fact_table, 'members'], cols, [], agg_info, question)

        if agg_info.get('agg_func') == 'AVG' and 'vs' in q:
            amount_cols = [col for w, col in [('billed', 'BILLED_AMOUNT'), ('paid', 'PAID_AMOUNT'), ('allowed', 'ALLOWED_AMOUNT')] if w in q]
            if len(amount_cols) >= 2:
                sel = ', '.join(f"AVG(CAST({c} AS REAL)) as avg_{c.lower()}" for c in amount_cols)
                filters, _, _ = QueryParser.extract_filters(question, cols, self.schema, tables)
                where = f" WHERE {' AND '.join(filters)}" if filters else ""
                sql = f"SELECT {sel} FROM claims{where};"
                return self._result(sql, ['claims'], cols, filters, agg_info, question)

        if re.search(r'\bmost common\s+(?:diagnos|icd)', q):
            sql = "SELECT ICD10_CODE, COUNT(*) as frequency FROM diagnoses GROUP BY ICD10_CODE ORDER BY frequency DESC LIMIT 10;"
            return self._result(sql, ['diagnoses'], cols, [], {'agg_func': 'COUNT', 'group_by_terms': ['ICD10_CODE'], 'top_n': 10, 'order': 'DESC'}, question)

        los_m = re.search(r'\blength of stay\s+(?:over|above|greater than|more than|exceeding)\s+(\d+)', q)
        if los_m:
            filters, having, dt = QueryParser.extract_filters(question, cols, self.schema, tables)
            filters.append(f"CAST(LENGTH_OF_STAY AS REAL) > {los_m.group(1)}")
            if 'encounters' not in tables:
                tables.insert(0, 'encounters')
            sql = self.composer.compose(question, cols, filters, agg_info, tables, having)
            return self._result(sql, tables, cols, filters, agg_info, question)

        if re.search(r'\bhcc\b', q) and ('count' in q or 'categor' in q):
            sql = "SELECT HCC_CATEGORY, COUNT(*) as count FROM diagnoses WHERE HCC_CATEGORY IS NOT NULL AND HCC_CATEGORY != '' GROUP BY HCC_CATEGORY ORDER BY count DESC;"
            return self._result(sql, ['diagnoses'], cols, [], agg_info, question)

        if re.search(r'\bappointment|pcp\b.*\bappointment|upcoming\b.*\bpcp\b', q):
            region_filter = ''
            for loc, reg in REGION_MAP.items():
                if loc in q:
                    region_filter = f" AND a.KP_REGION = '{reg}'"
                    break
            pcp = " AND a.IS_PCP_VISIT = 'Y'" if 'pcp' in q or 'primary care' in q else ''
            upcoming = " AND a.APPOINTMENT_DATE >= date('now')" if any(w in q for w in ['upcoming', 'future', 'next']) else ''
            sql = f"""SELECT a.APPOINTMENT_DATE, a.APPOINTMENT_TIME, a.APPOINTMENT_TYPE, a.DEPARTMENT, a.FACILITY, a.KP_REGION, a.STATUS, a.REASON, m.FIRST_NAME || ' ' || m.LAST_NAME as MEMBER_NAME, p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as PROVIDER_NAME
FROM appointments a LEFT JOIN members m ON a.MEMBER_ID = m.MEMBER_ID LEFT JOIN providers p ON a.PROVIDER_NPI = p.NPI
WHERE a.STATUS = 'SCHEDULED'{pcp}{region_filter}{upcoming} ORDER BY a.APPOINTMENT_DATE LIMIT 50;"""
            return self._result(sql, ['appointments', 'members', 'providers'], cols, [], agg_info, question)

        if re.search(r'\bcpt\b', q) and re.search(r'\bmost\b.*\b(common|frequent|occurring|popular)\b|\btop\b.*\bcpt\b', q):
            ym = re.search(r'\b(202[0-9])\b', q)
            yf = f" WHERE SERVICE_DATE LIKE '{ym.group(1)}%'" if ym else ''
            sql = f"""SELECT c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY, COUNT(*) as occurrence_count
FROM claims c LEFT JOIN cpt_codes cpt ON c.CPT_CODE = cpt.CPT_CODE{yf}
GROUP BY c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY ORDER BY occurrence_count DESC LIMIT 20;"""
            return self._result(sql, ['claims', 'cpt_codes'], cols, [], agg_info, question)

        cpt_m = re.search(r'\bcpt\b.*?\bcode\b\s*(\d{4,5})?', q) or re.search(r'\bcpt\b\s*(\d{4,5})', q)
        if cpt_m and any(w in q for w in ['description', 'what is', 'describe', 'meaning']):
            cv = cpt_m.group(1)
            if cv:
                sql = f"SELECT CPT_CODE, DESCRIPTION, CATEGORY, RVU FROM cpt_codes WHERE CPT_CODE = '{cv}';"
            else:
                sql = "SELECT CPT_CODE, DESCRIPTION, CATEGORY, RVU FROM cpt_codes ORDER BY CPT_CODE LIMIT 50;"
            return self._result(sql, ['cpt_codes'], cols, [], agg_info, question)

        if re.search(r'\b(breakdown|distribution|split)\b.*\b(claim|claims)\b.*\b(categor|type|class)', q) or \
           re.search(r'\bclaim\b.*\b(categor|type)\b.*\b(breakdown|distribution|split)\b', q):
            ym = re.search(r'\b(202[0-9])\b', q)
            yf = f" WHERE SERVICE_DATE LIKE '{ym.group(1)}%'" if ym else ''
            sql = f"""SELECT CLAIM_TYPE, COUNT(*) as claim_count, ROUND(SUM(BILLED_AMOUNT),2) as total_billed, ROUND(SUM(PAID_AMOUNT),2) as total_paid, ROUND(AVG(BILLED_AMOUNT),2) as avg_billed
FROM claims{yf} GROUP BY CLAIM_TYPE ORDER BY claim_count DESC;"""
            return self._result(sql, ['claims'], cols, [], agg_info, question)

        if 'claim' in q and re.search(r'\b(physician|doctor|provider)', q) and not re.search(r'\btop\s+\d+\s+\w+\s+by\s+\w+\s+(?:count|volume|number)', q):
            spec = ''
            if any(w in q for w in ['general', 'general physician', 'primary care', 'pcp', 'family', 'internal medicine']):
                spec = " AND (p.SPECIALTY IN ('Internal Medicine','Family Medicine','General Practice'))"
            rf = ''
            for loc, reg in REGION_MAP.items():
                if re.search(r'\b' + re.escape(loc) + r'\b', q):
                    rf = f" AND c.KP_REGION = '{reg}'"
                    break
            ym = re.search(r'\b(202[0-9])\b', q)
            yf = f" AND c.SERVICE_DATE LIKE '{ym.group(1)}%'" if ym else ''
            sql = f"""SELECT c.CLAIM_ID, c.SERVICE_DATE, c.CLAIM_TYPE, c.BILLED_AMOUNT, c.PAID_AMOUNT, c.CLAIM_STATUS,
p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as PHYSICIAN_NAME, p.SPECIALTY, p.FACILITY, c.KP_REGION
FROM claims c LEFT JOIN providers p ON c.RENDERING_NPI = p.NPI WHERE 1=1{spec}{rf}{yf} ORDER BY c.SERVICE_DATE DESC LIMIT 50;"""
            return self._result(sql, ['claims', 'providers'], cols, [], agg_info, question)

        if re.search(r'\b(physician|doctor|provider|pcp)', q) and re.search(r'\b(in|at|from|near|list|show|find)\b', q):
            spec = ''
            if any(w in q for w in ['general', 'general physician', 'primary care', 'pcp', 'family', 'internal medicine']):
                spec = " AND (SPECIALTY IN ('Internal Medicine','Family Medicine','General Practice') OR DEPARTMENT LIKE '%Primary%')"
            elif 'cardiolog' in q:
                spec = " AND SPECIALTY = 'Cardiology'"
            elif 'dermatolog' in q:
                spec = " AND SPECIALTY = 'Dermatology'"
            elif 'orthop' in q:
                spec = " AND SPECIALTY = 'Orthopedics'"
            rf = ''
            for loc, reg in REGION_MAP.items():
                if re.search(r'\b' + re.escape(loc) + r'\b', q):
                    rf = f" AND KP_REGION = '{reg}'"
                    break
            sql = f"""SELECT NPI, PROVIDER_FIRST_NAME || ' ' || PROVIDER_LAST_NAME as PROVIDER_NAME, SPECIALTY, DEPARTMENT, FACILITY, KP_REGION, STATUS, PANEL_SIZE, ACCEPTS_NEW_PATIENTS
FROM providers WHERE STATUS = 'ACTIVE'{spec}{rf} ORDER BY PROVIDER_LAST_NAME LIMIT 50;"""
            return self._result(sql, ['providers'], cols, [], agg_info, question)

        if re.search(r'\bcovid\b', q) and re.search(r'\b(hospital|admission|inpatient|encounter|visit)', q):
            rf = ''
            for loc, reg in REGION_MAP.items():
                if re.search(r'\b' + re.escape(loc) + r'\b', q):
                    rf = f" AND e.KP_REGION = '{reg}'"
                    break
            ym = re.search(r'\b(202[0-9])\b', q)
            yf = f" AND e.SERVICE_DATE LIKE '{ym.group(1)}%'" if ym else ''
            sql = f"""SELECT e.ENCOUNTER_ID, e.SERVICE_DATE, e.VISIT_TYPE, e.DEPARTMENT, e.LENGTH_OF_STAY, e.DISPOSITION, e.FACILITY, e.KP_REGION,
m.FIRST_NAME || ' ' || m.LAST_NAME as PATIENT_NAME, m.DATE_OF_BIRTH, m.GENDER
FROM encounters e LEFT JOIN members m ON e.MEMBER_ID = m.MEMBER_ID
WHERE e.PRIMARY_DIAGNOSIS = 'U07.1'{rf}{yf} ORDER BY e.SERVICE_DATE DESC LIMIT 50;"""
            return self._result(sql, ['encounters', 'members'], cols, [], agg_info, question)

        if re.search(r'\b(breakdown|distribution|split|ratio)\b', q) and re.search(r'\b(paid|denied|adjusted|status)\b', q) and 'claim' in q:
            ym = re.search(r'\b(202[0-9])\b', q)
            yf = f" WHERE SERVICE_DATE LIKE '{ym.group(1)}%'" if ym else ''
            sql = f"""SELECT CLAIM_STATUS, COUNT(*) as claim_count, ROUND(SUM(BILLED_AMOUNT),2) as total_billed, ROUND(SUM(PAID_AMOUNT),2) as total_paid,
ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims{yf}), 1) as percentage
FROM claims{yf} GROUP BY CLAIM_STATUS ORDER BY claim_count DESC;"""
            return self._result(sql, ['claims'], cols, [], agg_info, question)

        return None

    def _prune_tables(self, tables: List[str], filters: List[str], agg_info: Dict,
                      domain_tables: List[str], q: str) -> List[str]:
        if len(tables) <= 1:
            return tables
        primary = tables[0]
        primary_cols = {c.upper() for c in self.schema.get_columns(primary)}
        filter_cols = set()
        for f in filters:
            for cn in self.schema.column_to_tables:
                if cn.upper() in f.upper():
                    filter_cols.add(cn.upper())
        group_cols = set()
        for term in agg_info.get('group_by_terms', []):
            tu = term.upper().replace(' ', '_')
            for cn in self.schema.column_to_tables:
                if cn.upper() == tu or cn.upper() == tu + '_ID':
                    group_cols.add(cn.upper())

        pruned = [primary]
        for t in tables[1:]:
            t_cols = {c.upper() for c in self.schema.get_columns(t)}
            unique = t_cols - primary_cols
            keep = (t in domain_tables or
                    bool(filter_cols & unique) or bool(group_cols & unique) or
                    t in q or t.rstrip('s') in q or
                    any(kw in q for kw in TABLE_KEYWORDS.get(t, [])) or
                    ('denial' in q and t == 'claims') or
                    (any(w in q for w in ['billed', 'paid', 'allowed', 'cost', 'amount', 'spend']) and t == 'claims'))
            if keep:
                pruned.append(t)
        return pruned

    def _result(self, sql: str, tables: List[str], cols: List[Dict], filters: List[str],
                agg_info: Dict, question: str) -> Dict[str, Any]:
        return {'sql': sql, 'tables_used': tables, 'columns_resolved': cols,
                'filters': filters, 'agg_info': agg_info,
                'confidence': self._confidence(cols, tables, filters, agg_info),
                'explanation': self._explain(question, cols, tables, filters, agg_info)}

    def _confidence(self, cols, tables, filters, agg_info) -> float:
        s = 0.3
        if cols:
            s += min(0.3, 0.1 * len(cols))
            s += min(0.1, 0.05 * sum(1 for c in cols if c['match_type'] == 'exact'))
        if tables:
            s += 0.1
        if filters:
            s += min(0.1, 0.05 * len(filters))
        if agg_info.get('agg_func'):
            s += 0.1
        return min(1.0, s)

    def _validate_data_exists(self, question: str, tables: List[str]) -> Optional[str]:
        q = question.lower()

        if re.search(r'\bcovid\b', q):
            if 'diagnoses' not in tables and 'encounters' not in tables:
                return "COVID data is stored in diagnoses/encounters tables with ICD code U07.1. Query may return no results if no COVID records exist."

        year_m = re.search(r'\b(202[0-9])\b', q)
        if year_m:
            year = int(year_m.group(1))
            current_year = 2026
            if year > current_year:
                return f"Query asks for year {year}, which is in the future. Data may not exist yet."

        all_cols = set()
        for t in tables:
            for c in self.schema.get_columns(t):
                all_cols.add(c.upper())

        WARN_COLS = {
            'age': 'Age is calculated from DATE_OF_BIRTH; use "members by age group" for age-bucketed analysis',
            'processing_days': 'Processing time is calculated from SUBMITTED_DATE and ADJUDICATED_DATE',
            'tenure': 'Member tenure is calculated from ENROLLMENT_DATE and DISENROLLMENT_DATE',
        }
        for col, msg in WARN_COLS.items():
            if col.upper() in q and col.upper() not in all_cols:
                return msg

        return None

    def _explain(self, question, cols, tables, filters, agg_info) -> str:
        lines = []
        af = agg_info.get('agg_func')
        gt = agg_info.get('group_by_terms', [])
        tn = agg_info.get('top_n')

        ACTION = {'COUNT': 'counting records', 'AVG': 'calculating the average', 'SUM': 'summing up totals',
                  'MAX': 'finding the maximum value', 'MIN': 'finding the minimum value'}
        action = ACTION.get(af, f"ranking the top {tn}" if tn else 'retrieving records')
        lines.append(f"<b>Interpretation:</b> You asked \"{question}\" — so we are {action}.")

        TABLE_DESC = {
            'claims': 'billing & payment data', 'members': 'patient demographics',
            'providers': 'physician/facility info', 'encounters': 'visit records',
            'diagnoses': 'clinical diagnoses', 'prescriptions': 'medication data',
            'referrals': 'referral records', 'appointments': 'appointment data', 'cpt_codes': 'procedure codes',
        }
        if len(tables) == 1:
            lines.append(f"<b>Table chosen:</b> <code>{tables[0]}</code> — {TABLE_DESC.get(tables[0], 'data')}.")
        else:
            trs = [f"<code>{t}</code> for {TABLE_DESC.get(t, 'data')}" for t in tables]
            lines.append(f"<b>Tables chosen ({len(tables)}):</b> {'; '.join(trs)}.")
            jp = self.schema.find_join_path(tables)
            if jp:
                jd = []
                for t1, t2, col in jp:
                    if '=' in col:
                        l, r = col.split('=', 1)
                        jd.append(f"<code>{t1}.{l}</code> = <code>{t2}.{r}</code>")
                    else:
                        jd.append(f"<code>{t1}.{col}</code> = <code>{t2}.{col}</code>")
                lines.append(f"<b>JOINs:</b> {', '.join(jd)}.")

        if cols:
            mappings = []
            seen = set()
            for rc in cols[:6]:
                if rc['column'] in seen:
                    continue
                seen.add(rc['column'])
                if rc['match_type'] == 'synonym' and rc.get('original_term'):
                    mappings.append(f"\"{rc['original_term']}\" → <code>{rc['table']}.{rc['column']}</code>")
                elif rc['match_type'] == 'exact':
                    mappings.append(f"<code>{rc['column']}</code>")
            if mappings:
                lines.append(f"<b>Column mapping:</b> {'; '.join(mappings)}.")

        if filters:
            lines.append(f"<b>Filters applied:</b> {len(filters)} condition(s) narrowing the dataset.")
        if af and gt:
            lines.append(f"<b>Aggregation:</b> Computing {af} grouped by {', '.join(gt)}.")
        elif af:
            lines.append(f"<b>Aggregation:</b> Computing {af} across all matching records.")
        if tn:
            lines.append(f"<b>Ranking:</b> Top {tn} results.")
        return "<br>".join(lines)


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
