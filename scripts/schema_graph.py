import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger('gpdm.schema_graph')


@dataclass
class SemanticType:
    category: str
    subcategory: str
    aggregatable: bool
    groupable: bool
    filterable: bool
    is_identifier: bool = False
    is_date: bool = False
    is_money: bool = False


SEMANTIC_TYPES: Dict[str, SemanticType] = {
    'PAID_AMOUNT':           SemanticType('money', 'cost',       True, False, True,  is_money=True),
    'BILLED_AMOUNT':         SemanticType('money', 'charge',     True, False, True,  is_money=True),
    'ALLOWED_AMOUNT':        SemanticType('money', 'allowed',    True, False, True,  is_money=True),
    'MEMBER_RESPONSIBILITY': SemanticType('money', 'liability',  True, False, True,  is_money=True),
    'COPAY':                 SemanticType('money', 'copay',      True, False, True,  is_money=True),
    'COINSURANCE':           SemanticType('money', 'coinsurance',True, False, True,  is_money=True),
    'DEDUCTIBLE':            SemanticType('money', 'deductible', True, False, True,  is_money=True),
    'COST':                  SemanticType('money', 'cost',       True, False, True,  is_money=True),
    'RVU':                   SemanticType('money', 'rvu',        True, False, True,  is_money=True),

    'RISK_SCORE':            SemanticType('measure', 'risk',     True, False, True),
    'LENGTH_OF_STAY':        SemanticType('measure', 'duration', True, False, True),
    'PANEL_SIZE':            SemanticType('measure', 'capacity', True, False, True),
    'DURATION_MINUTES':      SemanticType('measure', 'duration', True, False, True),
    'QUANTITY':              SemanticType('measure', 'quantity',  True, False, True),
    'DAYS_SUPPLY':           SemanticType('measure', 'duration', True, False, True),
    'REFILLS_AUTHORIZED':    SemanticType('measure', 'count',    True, False, True),
    'REFILLS_USED':          SemanticType('measure', 'count',    True, False, True),
    'CHRONIC_CONDITIONS':    SemanticType('measure', 'count',    True, False, True),

    'MEMBER_ID':     SemanticType('identifier', 'member',    False, True, True, is_identifier=True),
    'CLAIM_ID':      SemanticType('identifier', 'claim',     False, False, True, is_identifier=True),
    'ENCOUNTER_ID':  SemanticType('identifier', 'encounter', False, False, True, is_identifier=True),
    'NPI':           SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'RENDERING_NPI': SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'BILLING_NPI':   SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'PRESCRIBING_NPI':SemanticType('identifier', 'provider', False, True, True, is_identifier=True),
    'SUPERVISING_NPI':SemanticType('identifier', 'provider', False, True, True, is_identifier=True),
    'REFERRING_NPI': SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'REFERRED_TO_NPI':SemanticType('identifier', 'provider', False, True, True, is_identifier=True),
    'PROVIDER_NPI':  SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'PCP_NPI':       SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'DIAGNOSING_NPI':SemanticType('identifier', 'provider',  False, True, True, is_identifier=True),
    'MRN':           SemanticType('identifier', 'member',    False, False, True, is_identifier=True),
    'RX_ID':         SemanticType('identifier', 'prescription',False,False,True, is_identifier=True),
    'REFERRAL_ID':   SemanticType('identifier', 'referral',  False, False, True, is_identifier=True),
    'DIAGNOSIS_ID':  SemanticType('identifier', 'diagnosis', False, False, True, is_identifier=True),
    'APPOINTMENT_ID':SemanticType('identifier', 'appointment',False,False,True, is_identifier=True),
    'CPT_CODE':      SemanticType('identifier', 'procedure', False, True, True, is_identifier=True),
    'ICD10_CODE':    SemanticType('identifier', 'diagnosis', False, True, True, is_identifier=True),
    'NDC_CODE':      SemanticType('identifier', 'drug',      False, True, True, is_identifier=True),
    'HCC_CODE':      SemanticType('identifier', 'hcc',       False, True, True, is_identifier=True),
    'AUTHORIZATION_NUMBER': SemanticType('identifier', 'auth',False,False,True, is_identifier=True),
    'DEA_NUMBER':    SemanticType('identifier', 'dea',       False, False, True, is_identifier=True),

    'SERVICE_DATE':      SemanticType('time', 'service_date',   False, True, True, is_date=True),
    'ADMIT_DATE':        SemanticType('time', 'admit_date',     False, True, True, is_date=True),
    'DISCHARGE_DATE':    SemanticType('time', 'discharge_date', False, True, True, is_date=True),
    'SUBMITTED_DATE':    SemanticType('time', 'submit_date',    False, True, True, is_date=True),
    'ADJUDICATED_DATE':  SemanticType('time', 'adjudicate_date',False, True, True, is_date=True),
    'ENROLLMENT_DATE':   SemanticType('time', 'enroll_date',    False, True, True, is_date=True),
    'DISENROLLMENT_DATE':SemanticType('time', 'disenroll_date', False, True, True, is_date=True),
    'DATE_OF_BIRTH':     SemanticType('time', 'birth_date',     False, True, True, is_date=True),
    'APPOINTMENT_DATE':  SemanticType('time', 'appt_date',      False, True, True, is_date=True),
    'PRESCRIPTION_DATE': SemanticType('time', 'rx_date',        False, True, True, is_date=True),
    'FILL_DATE':         SemanticType('time', 'fill_date',      False, True, True, is_date=True),
    'DIAGNOSIS_DATE':    SemanticType('time', 'dx_date',        False, True, True, is_date=True),
    'RESOLVED_DATE':     SemanticType('time', 'resolve_date',   False, True, True, is_date=True),
    'REFERRAL_DATE':     SemanticType('time', 'referral_date',  False, True, True, is_date=True),
    'HIRE_DATE':         SemanticType('time', 'hire_date',      False, True, True, is_date=True),
    'APPOINTMENT_TIME':  SemanticType('time', 'time',           False, False, True, is_date=True),
    'CHECK_IN_TIME':     SemanticType('time', 'time',           False, False, True, is_date=True),
    'CHECK_OUT_TIME':    SemanticType('time', 'time',           False, False, True, is_date=True),

    'KP_REGION':  SemanticType('geography', 'region',   False, True, True),
    'FACILITY':   SemanticType('geography', 'facility', False, True, True),
    'CITY':       SemanticType('geography', 'city',     False, True, True),
    'STATE':      SemanticType('geography', 'state',    False, True, True),
    'ZIP_CODE':   SemanticType('geography', 'zip',      False, True, True),
    'ADDRESS':    SemanticType('geography', 'address',  False, False, False),
    'PHARMACY':   SemanticType('geography', 'pharmacy', False, True, True),
    'LICENSE_STATE': SemanticType('geography', 'state', False, True, True),

    'PLAN_TYPE':         SemanticType('category', 'plan',       False, True, True),
    'VISIT_TYPE':        SemanticType('category', 'visit',      False, True, True),
    'CLAIM_STATUS':      SemanticType('category', 'status',     False, True, True),
    'CLAIM_TYPE':        SemanticType('category', 'claim_type', False, True, True),
    'DEPARTMENT':        SemanticType('category', 'department', False, True, True),
    'SPECIALTY':         SemanticType('category', 'specialty',  False, True, True),
    'PROVIDER_TYPE':     SemanticType('category', 'prov_type',  False, True, True),
    'GENDER':            SemanticType('category', 'gender',     False, True, True),
    'RACE':              SemanticType('category', 'race',       False, True, True),
    'LANGUAGE':          SemanticType('category', 'language',   False, True, True),
    'DIAGNOSIS_TYPE':    SemanticType('category', 'dx_type',    False, True, True),
    'IS_CHRONIC':        SemanticType('category', 'chronic',    False, True, True),
    'SEVERITY':          SemanticType('category', 'severity',   False, True, True),
    'HCC_CATEGORY':      SemanticType('category', 'hcc_cat',   False, True, True),
    'MEDICATION_CLASS':  SemanticType('category', 'drug_class', False, True, True),
    'MEDICATION_NAME':   SemanticType('category', 'drug_name',  False, True, True),
    'APPOINTMENT_TYPE':  SemanticType('category', 'appt_type',  False, True, True),
    'URGENCY':           SemanticType('category', 'urgency',    False, True, True),
    'REFERRAL_TYPE':     SemanticType('category', 'ref_type',   False, True, True),
    'REFERRAL_REASON':   SemanticType('category', 'reason',     False, True, True),
    'DENIAL_REASON':     SemanticType('category', 'denial',     False, True, True),
    'DISPOSITION':       SemanticType('category', 'disposition', False, True, True),
    'ENCOUNTER_STATUS':  SemanticType('category', 'status',     False, True, True),
    'CATEGORY':          SemanticType('category', 'cpt_cat',    False, True, True),
    'IS_PCP_VISIT':      SemanticType('category', 'boolean',    False, True, True),
    'ACCEPTS_NEW_PATIENTS': SemanticType('category', 'boolean', False, True, True),
    'STATUS':            SemanticType('category', 'status',     False, True, True),
    'REASON':            SemanticType('category', 'reason',     False, True, True),
    'CHIEF_COMPLAINT':   SemanticType('category', 'complaint',  False, True, True),

    'ICD10_DESCRIPTION':    SemanticType('text', 'dx_desc',    False, True, True),
    'CPT_DESCRIPTION':      SemanticType('text', 'proc_desc',  False, True, True),
    'DIAGNOSIS_DESCRIPTION':SemanticType('text', 'dx_desc',    False, True, True),
    'DESCRIPTION':          SemanticType('text', 'desc',       False, True, True),
    'FIRST_NAME':           SemanticType('text', 'name',       False, False, True),
    'LAST_NAME':            SemanticType('text', 'name',       False, False, True),
    'PROVIDER_FIRST_NAME':  SemanticType('text', 'name',       False, False, True),
    'PROVIDER_LAST_NAME':   SemanticType('text', 'name',       False, False, True),
    'EMAIL':                SemanticType('text', 'contact',    False, False, False),
    'PHONE':                SemanticType('text', 'contact',    False, False, False),
    'PRIMARY_DIAGNOSIS':    SemanticType('text', 'dx_code',    False, True, True),
}


@dataclass
class TableConcept:
    name: str
    concept: str
    description: str
    synonyms: List[str]
    primary_key: str
    primary_date: str
    primary_measure: str
    count_column: str


TABLE_CONCEPTS: Dict[str, TableConcept] = {
    'members': TableConcept(
        'members', 'member', 'Health plan members/patients/enrollees',
        ['member', 'patient', 'enrollee', 'person', 'people', 'population',
         'beneficiary', 'subscriber'],
        'MEMBER_ID', 'ENROLLMENT_DATE', 'RISK_SCORE', 'MEMBER_ID',
    ),
    'claims': TableConcept(
        'claims', 'claim', 'Insurance claims for medical services',
        ['claim', 'bill', 'charge', 'payment', 'reimbursement', 'cost',
         'spend', 'expense', 'financial', 'money', 'dollar', 'amount',
         'billed', 'paid', 'denied', 'denial', 'copay'],
        'CLAIM_ID', 'SERVICE_DATE', 'PAID_AMOUNT', 'CLAIM_ID',
    ),
    'encounters': TableConcept(
        'encounters', 'encounter', 'Patient visits/encounters with providers',
        ['encounter', 'visit', 'admission', 'admissions', 'stay', 'hospitalization',
         'er', 'emergency', 'inpatient', 'outpatient', 'telehealth',
         'appointment', 'discharge', 'hospital'],
        'ENCOUNTER_ID', 'SERVICE_DATE', 'LENGTH_OF_STAY', 'ENCOUNTER_ID',
    ),
    'providers': TableConcept(
        'providers', 'provider', 'Healthcare providers/doctors/physicians',
        ['provider', 'doctor', 'physician', 'specialist', 'clinician',
         'practitioner', 'npi', 'surgeon', 'doc', 'docs', 'panel'],
        'NPI', 'HIRE_DATE', 'PANEL_SIZE', 'NPI',
    ),
    'prescriptions': TableConcept(
        'prescriptions', 'prescription', 'Medication prescriptions',
        ['prescription', 'medication', 'drug', 'pharmacy', 'rx',
         'medicine', 'pharmaceutical', 'pill', 'refill',
         'script', 'scripts', 'therapeutic'],
        'RX_ID', 'PRESCRIPTION_DATE', 'COST', 'RX_ID',
    ),
    'diagnoses': TableConcept(
        'diagnoses', 'diagnosis', 'Patient diagnoses and conditions',
        ['diagnosis', 'condition', 'disease', 'illness', 'icd',
         'chronic', 'comorbidity', 'hcc', 'diabetes', 'hypertension',
         'asthma', 'copd', 'cancer', 'heart', 'renal', 'mental'],
        'DIAGNOSIS_ID', 'DIAGNOSIS_DATE', 'SEVERITY', 'DIAGNOSIS_ID',
    ),
    'referrals': TableConcept(
        'referrals', 'referral', 'Patient referrals between providers',
        ['referral', 'refer', 'referred', 'authorization',
         'authorizations', 'auth', 'auths', 'prior auth'],
        'REFERRAL_ID', 'REFERRAL_DATE', 'URGENCY', 'REFERRAL_ID',
    ),
    'appointments': TableConcept(
        'appointments', 'appointment', 'Scheduled appointments',
        ['appointment', 'schedule', 'no-show', 'noshow', 'cancelled',
         'booking'],
        'APPOINTMENT_ID', 'APPOINTMENT_DATE', 'DURATION_MINUTES', 'APPOINTMENT_ID',
    ),
    'cpt_codes': TableConcept(
        'cpt_codes', 'procedure', 'CPT procedure codes and descriptions',
        ['cpt', 'procedure', 'service code', 'rvu'],
        'CPT_CODE', '', 'RVU', 'CPT_CODE',
    ),
}


@dataclass
class JoinPath:
    from_table: str
    to_table: str
    from_col: str
    to_col: str
    relationship: str
    join_type: str
    priority: int


JOIN_PATHS: List[JoinPath] = [
    JoinPath('members', 'claims',       'MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),
    JoinPath('members', 'encounters',   'MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),
    JoinPath('members', 'prescriptions','MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),
    JoinPath('members', 'diagnoses',    'MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),
    JoinPath('members', 'referrals',    'MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),
    JoinPath('members', 'appointments', 'MEMBER_ID', 'MEMBER_ID',      'has',          'LEFT', 1),

    JoinPath('claims', 'encounters',    'ENCOUNTER_ID', 'ENCOUNTER_ID','generated_by', 'INNER', 1),
    JoinPath('claims', 'members',       'MEMBER_ID', 'MEMBER_ID',      'belongs_to',   'INNER', 1),
    JoinPath('claims', 'providers',     'RENDERING_NPI', 'NPI',        'performed_by', 'INNER', 2),
    JoinPath('claims', 'providers',     'BILLING_NPI', 'NPI',          'billed_by',    'INNER', 3),
    JoinPath('claims', 'diagnoses',     'MEMBER_ID', 'MEMBER_ID',      'related_to',   'INNER', 2),
    JoinPath('claims', 'cpt_codes',     'CPT_CODE', 'CPT_CODE',       'coded_as',     'LEFT',  2),

    JoinPath('encounters', 'claims',    'ENCOUNTER_ID', 'ENCOUNTER_ID','generates',    'LEFT',  1),
    JoinPath('encounters', 'members',   'MEMBER_ID', 'MEMBER_ID',      'belongs_to',   'INNER', 1),
    JoinPath('encounters', 'providers', 'RENDERING_NPI', 'NPI',        'performed_by', 'INNER', 2),
    JoinPath('encounters', 'diagnoses', 'ENCOUNTER_ID', 'ENCOUNTER_ID','results_in',   'LEFT',  1),

    JoinPath('diagnoses', 'claims',     'MEMBER_ID', 'MEMBER_ID',      'drives_cost',  'INNER', 2),
    JoinPath('diagnoses', 'encounters', 'ENCOUNTER_ID', 'ENCOUNTER_ID','diagnosed_at', 'INNER', 1),
    JoinPath('diagnoses', 'members',    'MEMBER_ID', 'MEMBER_ID',      'belongs_to',   'INNER', 1),

    JoinPath('prescriptions', 'members',   'MEMBER_ID', 'MEMBER_ID',   'belongs_to',   'INNER', 1),
    JoinPath('prescriptions', 'providers', 'PRESCRIBING_NPI', 'NPI',   'prescribed_by','INNER', 2),

    JoinPath('providers', 'claims',     'NPI', 'RENDERING_NPI',        'filed',        'LEFT',  2),
    JoinPath('providers', 'encounters', 'NPI', 'RENDERING_NPI',        'conducted',    'LEFT',  2),

    JoinPath('referrals', 'members',    'MEMBER_ID', 'MEMBER_ID',      'belongs_to',   'INNER', 1),
    JoinPath('referrals', 'providers',  'REFERRING_NPI', 'NPI',        'referred_by',  'INNER', 2),
    JoinPath('referrals', 'providers',  'REFERRED_TO_NPI', 'NPI',     'referred_to',  'INNER', 2),

    JoinPath('appointments', 'members', 'MEMBER_ID', 'MEMBER_ID',      'belongs_to',   'INNER', 1),
    JoinPath('appointments', 'providers','PROVIDER_NPI', 'NPI',        'with_provider','INNER', 2),
]


WORD_TO_COLUMN: Dict[str, List[Tuple[str, str, float]]] = {
    'cost':         [('claims', 'PAID_AMOUNT', 0.9), ('prescriptions', 'COST', 0.7)],
    'spend':        [('claims', 'PAID_AMOUNT', 0.9)],
    'spending':     [('claims', 'PAID_AMOUNT', 0.9)],
    'expense':      [('claims', 'PAID_AMOUNT', 0.9)],
    'expensive':    [('claims', 'PAID_AMOUNT', 0.9)],
    'paid':         [('claims', 'PAID_AMOUNT', 0.95)],
    'payment':      [('claims', 'PAID_AMOUNT', 0.9)],
    'billed':       [('claims', 'BILLED_AMOUNT', 0.95)],
    'charge':       [('claims', 'BILLED_AMOUNT', 0.85)],
    'revenue':      [('claims', 'PAID_AMOUNT', 0.8)],
    'copay':        [('claims', 'COPAY', 0.95), ('prescriptions', 'COPAY', 0.7)],
    'coinsurance':  [('claims', 'COINSURANCE', 0.95)],
    'deductible':   [('claims', 'DEDUCTIBLE', 0.95)],
    'allowed':      [('claims', 'ALLOWED_AMOUNT', 0.95)],
    'drug cost':    [('prescriptions', 'COST', 0.95)],
    'medication cost': [('prescriptions', 'COST', 0.95)],
    'pmpm':         [('claims', 'PAID_AMOUNT', 0.9)],

    'diagnosis':    [('diagnoses', 'ICD10_DESCRIPTION', 0.9)],
    'condition':    [('diagnoses', 'ICD10_DESCRIPTION', 0.9)],
    'chronic':      [('diagnoses', 'IS_CHRONIC', 0.95)],
    'disease':      [('diagnoses', 'ICD10_DESCRIPTION', 0.85)],
    'diabetes':     [('diagnoses', 'ICD10_DESCRIPTION', 0.95)],
    'hypertension': [('diagnoses', 'ICD10_DESCRIPTION', 0.95)],
    'cancer':       [('diagnoses', 'ICD10_DESCRIPTION', 0.95)],
    'asthma':       [('diagnoses', 'ICD10_DESCRIPTION', 0.95)],
    'heart failure':[('diagnoses', 'ICD10_DESCRIPTION', 0.95)],
    'severity':     [('diagnoses', 'SEVERITY', 0.95)],
    'hcc':          [('diagnoses', 'HCC_CATEGORY', 0.9)],

    'visit':        [('encounters', 'VISIT_TYPE', 0.85)],
    'er':           [('encounters', 'VISIT_TYPE', 0.9)],
    'emergency':    [('encounters', 'VISIT_TYPE', 0.9)],
    'inpatient':    [('encounters', 'VISIT_TYPE', 0.9)],
    'outpatient':   [('encounters', 'VISIT_TYPE', 0.9)],
    'telehealth':   [('encounters', 'VISIT_TYPE', 0.9)],
    'hospital stay':[('encounters', 'LENGTH_OF_STAY', 0.9)],
    'length of stay': [('encounters', 'LENGTH_OF_STAY', 0.95)],
    'los':          [('encounters', 'LENGTH_OF_STAY', 0.95)],
    'admission':    [('encounters', 'ADMIT_DATE', 0.9)],
    'discharge':    [('encounters', 'DISCHARGE_DATE', 0.9)],
    'readmission':  [('encounters', 'ENCOUNTER_ID', 0.85)],

    'doctor':       [('providers', 'NPI', 0.9)],
    'provider':     [('providers', 'NPI', 0.9)],
    'physician':    [('providers', 'NPI', 0.9)],
    'specialist':   [('providers', 'SPECIALTY', 0.9)],
    'specialty':    [('providers', 'SPECIALTY', 0.95)],
    'specialties':  [('providers', 'SPECIALTY', 0.95)],
    'department':   [('encounters', 'DEPARTMENT', 0.8), ('providers', 'DEPARTMENT', 0.7)],

    'denied':       [('claims', 'CLAIM_STATUS', 0.95)],
    'denial':       [('claims', 'CLAIM_STATUS', 0.95)],
    'pending':      [('claims', 'CLAIM_STATUS', 0.9)],
    'approved':     [('claims', 'CLAIM_STATUS', 0.9)],
    'no-show':      [('appointments', 'STATUS', 0.95)],
    'noshow':       [('appointments', 'STATUS', 0.95)],
    'cancelled':    [('appointments', 'STATUS', 0.85)],

    'region':       [('claims', 'KP_REGION', 0.9), ('members', 'KP_REGION', 0.8)],
    'ncal':         [('claims', 'KP_REGION', 0.95)],
    'scal':         [('claims', 'KP_REGION', 0.95)],
    'facility':     [('encounters', 'FACILITY', 0.85), ('claims', 'FACILITY', 0.8)],

    'medication':   [('prescriptions', 'MEDICATION_NAME', 0.95)],
    'drug':         [('prescriptions', 'MEDICATION_NAME', 0.9)],
    'prescription': [('prescriptions', 'RX_ID', 0.9)],
    'pharmacy':     [('prescriptions', 'PHARMACY', 0.9)],
    'refill':       [('prescriptions', 'REFILLS_USED', 0.9)],
    'days supply':  [('prescriptions', 'DAYS_SUPPLY', 0.98)],
    'day supply':   [('prescriptions', 'DAYS_SUPPLY', 0.95)],
    'supply':       [('prescriptions', 'DAYS_SUPPLY', 0.7)],

    'rvu':          [('cpt_codes', 'RVU', 0.98)],
    'rvus':         [('cpt_codes', 'RVU', 0.98)],
    'relative value': [('cpt_codes', 'RVU', 0.95)],
    'cpt':          [('cpt_codes', 'CPT_CODE', 0.95)],
    'cpt code':     [('cpt_codes', 'CPT_CODE', 0.98)],
    'procedure code': [('cpt_codes', 'CPT_CODE', 0.9)],
    'category':     [('cpt_codes', 'CATEGORY', 0.85)],

    'risk':         [('members', 'RISK_SCORE', 0.95)],
    'risk score':   [('members', 'RISK_SCORE', 0.98)],
    'sick':         [('members', 'RISK_SCORE', 0.85)],
    'sickest':      [('members', 'RISK_SCORE', 0.9)],
    'high risk':    [('members', 'RISK_SCORE', 0.9)],
    'claim type':   [('claims', 'CLAIM_TYPE', 0.95)],
    'claim types':  [('claims', 'CLAIM_TYPE', 0.95)],
    'claim status': [('claims', 'CLAIM_STATUS', 0.95)],
    'panel size':   [('providers', 'PANEL_SIZE', 0.95)],
    'panel':        [('providers', 'PANEL_SIZE', 0.85)],
    'plan type':    [('claims', 'PLAN_TYPE', 0.9), ('members', 'PLAN_TYPE', 0.8)],
    'hmo':          [('claims', 'PLAN_TYPE', 0.9), ('members', 'PLAN_TYPE', 0.8)],
    'ppo':          [('claims', 'PLAN_TYPE', 0.9), ('members', 'PLAN_TYPE', 0.8)],
    'enrolled':     [('members', 'ENROLLMENT_DATE', 0.85)],
    'disenrolled':  [('members', 'DISENROLLMENT_DATE', 0.9)],
    'gender':       [('members', 'GENDER', 0.95)],
    'female':       [('members', 'GENDER', 0.9)],
    'male':         [('members', 'GENDER', 0.9)],
    'age':          [('members', 'DATE_OF_BIRTH', 0.85)],
    'race':         [('members', 'RACE', 0.95)],

    'referral':     [('referrals', 'REFERRAL_ID', 0.9)],
    'referred':     [('referrals', 'REFERRAL_ID', 0.85)],
}


class SemanticSchemaGraph:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables: Dict[str, TableConcept] = dict(TABLE_CONCEPTS)
        self.columns: Dict[str, Dict[str, SemanticType]] = {}
        self.join_paths: List[JoinPath] = list(JOIN_PATHS)
        self.word_map = dict(WORD_TO_COLUMN)

        self._build_column_registry()

        self.column_to_tables: Dict[str, List[str]] = defaultdict(list)
        for table, cols in self.columns.items():
            for col in cols:
                self.column_to_tables[col].append(table)

        self.join_index: Dict[Tuple[str, str], JoinPath] = {}
        for jp in sorted(self.join_paths, key=lambda j: j.priority):
            key = (jp.from_table, jp.to_table)
            rev_key = (jp.to_table, jp.from_table)
            if key not in self.join_index:
                self.join_index[key] = jp
            if rev_key not in self.join_index:
                self.join_index[rev_key] = JoinPath(
                    jp.to_table, jp.from_table, jp.to_col, jp.from_col,
                    f'reverse_{jp.relationship}', jp.join_type, jp.priority
                )

        logger.info("SemanticSchemaGraph: %d tables, %d columns, %d join paths, %d word mappings",
                     len(self.tables), sum(len(c) for c in self.columns.values()),
                     len(self.join_paths), len(self.word_map))

    def _build_column_registry(self):
        try:
            conn = sqlite3.connect(self.db_path)
            for table_name in self.tables:
                cols = {}
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall():
                    col_name = row[1]
                    sem_type = SEMANTIC_TYPES.get(col_name,
                        SemanticType('unknown', 'unknown', False, False, False))
                    cols[col_name] = sem_type
                self.columns[table_name] = cols
            conn.close()
        except Exception as e:
            logger.error("Failed to build column registry: %s", e)

    def find_tables_for_question(self, words: List[str], raw_question: str = '') -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        q_lower = raw_question.lower() if raw_question else ' '.join(w.lower() for w in words)

        for key in self.word_map:
            if ' ' in key and key in q_lower:
                for table, col, conf in self.word_map[key]:
                    scores[table] += conf + 0.1

        q_words = set(q_lower.split())
        MEASURE_WORDS = {'cost', 'spend', 'expense', 'payment', 'amount', 'billed',
                         'paid', 'money', 'dollar', 'financial', 'copay', 'charge',
                         'reimbursement', 'bill', 'denied', 'denial'}
        for table_name, concept in self.tables.items():
            name_variants = ({table_name, concept.concept} | set(concept.synonyms)) - MEASURE_WORDS
            for variant in name_variants:
                if len(variant) >= 3 and variant.lower() in q_words:
                    scores[table_name] += 1.0
                elif len(variant) >= 3 and variant.lower() in q_lower:
                    scores[table_name] += 0.8

        for word in words:
            word_lower = word.lower()

            for table_name, concept in self.tables.items():
                for syn in concept.synonyms:
                    if syn in word_lower or word_lower in syn:
                        scores[table_name] += 0.8
                    if len(syn) >= 4 and len(word_lower) >= 4 and syn[:4] == word_lower[:4]:
                        scores[table_name] += 0.3

            if word_lower in self.word_map:
                for table, col, conf in self.word_map[word_lower]:
                    scores[table] += conf

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(t, s) for t, s in ranked if s > 0.2]

    def find_columns_for_words(self, words: List[str], target_tables: List[str] = None,
                               raw_question: str = ''
                               ) -> List[Tuple[str, str, SemanticType, float]]:
        results = []
        seen = set()
        q_lower = raw_question.lower() if raw_question else ' '.join(w.lower() for w in words)

        for key in self.word_map:
            if ' ' in key and key in q_lower:
                for table, col, conf in self.word_map[key]:
                    if target_tables and table not in target_tables:
                        continue
                    if (table, col) not in seen:
                        sem_type = self.columns.get(table, {}).get(col)
                        if sem_type:
                            results.append((table, col, sem_type, conf + 0.1))
                            seen.add((table, col))

        for word in words:
            word_lower = word.lower()
            if word_lower in self.word_map:
                for table, col, conf in self.word_map[word_lower]:
                    if target_tables and table not in target_tables:
                        continue
                    if (table, col) not in seen:
                        sem_type = self.columns.get(table, {}).get(col)
                        if sem_type:
                            results.append((table, col, sem_type, conf))
                            seen.add((table, col))

        for table_name, cols in self.columns.items():
            if target_tables and table_name not in target_tables:
                continue
            for col_name, sem_type in cols.items():
                if (table_name, col_name) in seen:
                    continue
                col_words = col_name.lower().replace('_', ' ').split()
                match_count = sum(1 for w in col_words if len(w) >= 3 and w in q_lower)
                if match_count > 0:
                    confidence = 0.5 + (0.4 * match_count / max(len(col_words), 1))
                    results.append((table_name, col_name, sem_type, confidence))
                    seen.add((table_name, col_name))

        for word in words:
            word_lower = word.lower()
            col_normalized = word_lower.replace(' ', '_').upper()
            for table_name, cols in self.columns.items():
                if target_tables and table_name not in target_tables:
                    continue
                for col_name, sem_type in cols.items():
                    if (table_name, col_name) in seen:
                        continue
                    if col_name == col_normalized:
                        results.append((table_name, col_name, sem_type, 0.95))
                        seen.add((table_name, col_name))
                    elif len(word_lower) >= 4 and word_lower.upper() in col_name:
                        results.append((table_name, col_name, sem_type, 0.6))
                        seen.add((table_name, col_name))

        results.sort(key=lambda x: -x[3])
        return results

    def find_join_path(self, from_table: str, to_table: str) -> Optional[JoinPath]:
        if from_table == to_table:
            return None
        key = (from_table, to_table)
        return self.join_index.get(key)

    def find_multi_table_join(self, tables: List[str]) -> List[JoinPath]:
        if len(tables) <= 1:
            return []

        joins = []
        connected = {tables[0]}
        remaining = set(tables[1:])

        while remaining:
            best_join = None
            best_target = None
            best_priority = 999

            for src in connected:
                for tgt in remaining:
                    jp = self.find_join_path(src, tgt)
                    if jp and jp.priority < best_priority:
                        best_join = jp
                        best_target = tgt
                        best_priority = jp.priority

            if best_join:
                joins.append(best_join)
                connected.add(best_target)
                remaining.remove(best_target)
            else:
                for tgt in list(remaining):
                    for bridge in self.tables:
                        if bridge in connected or bridge in remaining:
                            continue
                        jp1 = self.find_join_path(list(connected)[0], bridge)
                        jp2 = self.find_join_path(bridge, tgt)
                        if jp1 and jp2:
                            joins.extend([jp1, jp2])
                            connected.add(bridge)
                            connected.add(tgt)
                            remaining.discard(tgt)
                            break
                    else:
                        continue
                    break
                else:
                    break

        return joins

    def get_money_columns(self, table: str) -> List[str]:
        return [col for col, st in self.columns.get(table, {}).items() if st.is_money]

    def get_date_columns(self, table: str) -> List[str]:
        return [col for col, st in self.columns.get(table, {}).items() if st.is_date]

    def get_groupable_columns(self, table: str) -> List[str]:
        return [col for col, st in self.columns.get(table, {}).items() if st.groupable]

    def get_aggregatable_columns(self, table: str) -> List[str]:
        return [col for col, st in self.columns.get(table, {}).items() if st.aggregatable]

    def get_table_columns(self, table: str) -> List[str]:
        return list(self.columns.get(table, {}).keys())
