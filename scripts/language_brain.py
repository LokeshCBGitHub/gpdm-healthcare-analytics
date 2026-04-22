import math
import re
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.language_brain')


INTENT_TRAINING_DATA = {
    'count': {
        'seeds': [
            'how many claims are there',
            'total number of members',
            'count of providers',
            'how many encounters',
            'number of prescriptions',
            'how many referrals do we have',
            'what is the total number of unique providers',
            'how many denied claims',
            'count of approved referrals',
            'how many patients do we have',
        ],
        'paraphrases': [
            'give me the count of claims',
            'whats the member count',
            'tell me how many doctors we have',
            'total visits in the system',
            'show me the number of rx',
            'how many referrals exist',
            'unique provider count',
            'number of rejected claims',
            'approved referral count',
            'patient population size',
            'what is our total patient volume',
            'can you tell me how many members enrolled',
            'i need the claim volume',
            'how many scripts were written',
            'provider headcount',
            'show total enrollees',
            'number of active patients',
            'what does our membership look like in terms of size',
            'do we know how many people are covered',
            'give me the number of unique NPIs',
        ],
    },
    'aggregate': {
        'seeds': [
            'what is the average copay per claim',
            'total paid amount',
            'average risk score',
            'sum of deductibles',
            'maximum billed amount',
            'average length of stay',
            'average coinsurance amount',
            'average panel size',
            'total cost of claims',
            'what is the average RVU',
        ],
        'paraphrases': [
            'how much does a typical member pay out of pocket on each visit',
            'mean cost per encounter',
            'whats the avg paid per claim',
            'overall spending total',
            'typical risk level across our population',
            'add up all the deductible amounts',
            'highest single charge we have seen',
            'on average how long are people staying',
            'typical coinsurance payment',
            'what does an average provider panel look like',
            'how much have we paid out total',
            'what is our average reimbursement',
            'tell me the mean copayment',
            'aggregate cost across all claims',
            'how much are members paying on average',
            'combined deductible burden',
            'peak billed charge',
            'mean inpatient stay duration',
            'average out of pocket expense',
            'how much does each visit cost on average',
            'what are we spending per claim typically',
            'run me the totals on paid amounts',
            'avg member responsibility per encounter',
            'whats the going rate for copays',
        ],
    },
    'rank': {
        'seeds': [
            'top 10 diagnoses by cost',
            'which provider has the most claims',
            'top medications by prescription count',
            'highest cost specialties',
            'most common visit types',
            'which region has the most members',
            'top diagnoses by claim cost',
            'most prescribed medication classes',
            'providers with most patients',
            'leading causes of denial',
        ],
        'paraphrases': [
            'what are the costliest diagnoses',
            'which doctor sees the most volume',
            'most popular drugs',
            'where are we spending the most by specialty',
            'rank encounter types by frequency',
            'biggest region by membership',
            'conditions driving the most cost',
            'most frequently prescribed drug categories',
            'busiest providers by panel',
            'primary reasons claims get rejected',
            'show me the top cost drivers',
            'what conditions are the most expensive to treat',
            'who are our highest volume doctors',
            'rank the drug classes by usage',
            'which departments generate the most claims',
            'what are the top reasons for denial',
            'give me a ranking of specialties by spend',
            'show the biggest diagnoses by dollar amount',
        ],
    },
    'rate': {
        'seeds': [
            'what is the denial rate',
            'claim approval percentage',
            'what percentage of claims are denied',
            'denial rate by provider',
            'denial rate by payer',
            'no show rate for appointments',
            'what fraction of encounters are emergency',
            'proportion of chronic conditions',
        ],
        'paraphrases': [
            'how often do claims get denied',
            'what share of submissions are approved',
            'what percent get rejected',
            'per provider rejection frequency',
            'plan level denial percentages',
            'appointment no show frequency',
            'how many visits end up in the ER as a percentage',
            'chronic condition prevalence rate',
            'claim denial frequency',
            'whats our clean claim ratio',
            'how frequently are we denying claims',
            'give me the approval rate',
            'rejection rate across the network',
            'payer-level denial metrics',
        ],
    },
    'trend': {
        'seeds': [
            'claims over time',
            'monthly cost trend',
            'quarterly enrollment by region',
            'year over year spending',
            'how has volume changed by month',
        ],
        'paraphrases': [
            'show me the claim trajectory',
            'cost changes month to month',
            'membership growth per quarter per area',
            'annual spend comparison',
            'visit volume patterns over the past year',
            'trending claims data',
            'spending patterns by period',
            'how is utilization changing over time',
            'month over month cost analysis',
        ],
    },
    'compare': {
        'seeds': [
            'compare HMO vs PPO costs',
            'inpatient vs outpatient spending',
            'drugs vs hospital costs',
            'denied vs approved claim amounts',
        ],
        'paraphrases': [
            'how do HMO and PPO plans stack up on cost',
            'whats the difference in spending inpatient versus outpatient',
            'pharmacy spend compared to facility spend',
            'cost comparison of rejected and approved claims',
            'side by side plan type analysis',
            'contrast ED visits with office visits by cost',
        ],
    },
    'list': {
        'seeds': [
            'show me claims by region',
            'list members with diabetes',
            'show me encounters by visit type',
            'show prescriptions by medication class',
        ],
        'paraphrases': [
            'break down claims geographically',
            'which members have a diabetes diagnosis',
            'display visits grouped by type',
            'medications organized by class',
            'give me the claims data by area',
            'find patients with chronic kidney disease',
        ],
    },
}


PHRASE_TO_COLUMN = {
    'out of pocket': ('claims', 'COPAY'),
    'member cost share': ('claims', 'COPAY'),
    'copayment': ('claims', 'COPAY'),
    'copay': ('claims', 'COPAY'),
    'co-pay': ('claims', 'COPAY'),
    'what members pay': ('claims', 'COPAY'),
    'patient pays': ('claims', 'COPAY'),

    'reimbursement': ('claims', 'PAID_AMOUNT'),
    'payment': ('claims', 'PAID_AMOUNT'),
    'paid': ('claims', 'PAID_AMOUNT'),
    'cost': ('claims', 'PAID_AMOUNT'),
    'spend': ('claims', 'PAID_AMOUNT'),
    'spending': ('claims', 'PAID_AMOUNT'),
    'expense': ('claims', 'PAID_AMOUNT'),
    'expenditure': ('claims', 'PAID_AMOUNT'),
    'what we pay': ('claims', 'PAID_AMOUNT'),
    'how much we paid': ('claims', 'PAID_AMOUNT'),

    'charge': ('claims', 'BILLED_AMOUNT'),
    'billed': ('claims', 'BILLED_AMOUNT'),
    'submitted amount': ('claims', 'BILLED_AMOUNT'),
    'charges': ('claims', 'BILLED_AMOUNT'),

    'contracted rate': ('claims', 'ALLOWED_AMOUNT'),
    'allowed': ('claims', 'ALLOWED_AMOUNT'),
    'negotiated rate': ('claims', 'ALLOWED_AMOUNT'),

    'cost sharing': ('claims', 'COINSURANCE'),
    'coinsurance': ('claims', 'COINSURANCE'),
    'co-insurance': ('claims', 'COINSURANCE'),
    'member share': ('claims', 'COINSURANCE'),

    'deductible': ('claims', 'DEDUCTIBLE'),
    'annual deductible': ('claims', 'DEDUCTIBLE'),
    'out of pocket max': ('claims', 'DEDUCTIBLE'),

    'member responsibility': ('claims', 'MEMBER_RESPONSIBILITY'),
    'oop': ('claims', 'MEMBER_RESPONSIBILITY'),
    'patient responsibility': ('claims', 'MEMBER_RESPONSIBILITY'),

    'stay duration': ('encounters', 'LENGTH_OF_STAY'),
    'length of stay': ('encounters', 'LENGTH_OF_STAY'),
    'los': ('encounters', 'LENGTH_OF_STAY'),
    'days admitted': ('encounters', 'LENGTH_OF_STAY'),
    'how long they stayed': ('encounters', 'LENGTH_OF_STAY'),
    'inpatient days': ('encounters', 'LENGTH_OF_STAY'),

    'risk level': ('members', 'RISK_SCORE'),
    'risk score': ('members', 'RISK_SCORE'),
    'acuity': ('members', 'RISK_SCORE'),
    'hcc score': ('members', 'RISK_SCORE'),
    'risk adjustment': ('members', 'RISK_SCORE'),
    'patient complexity': ('members', 'RISK_SCORE'),

    'patient panel': ('providers', 'PANEL_SIZE'),
    'panel size': ('providers', 'PANEL_SIZE'),
    'physician panel': ('providers', 'PANEL_SIZE'),
    'caseload': ('providers', 'PANEL_SIZE'),
    'attributed members': ('providers', 'PANEL_SIZE'),
    'number of patients per doctor': ('providers', 'PANEL_SIZE'),

    'drug class': ('prescriptions', 'MEDICATION_CLASS'),
    'medication class': ('prescriptions', 'MEDICATION_CLASS'),
    'therapeutic class': ('prescriptions', 'MEDICATION_CLASS'),
    'drug category': ('prescriptions', 'MEDICATION_CLASS'),

    'rvu': ('cpt_codes', 'RVU'),
    'relative value': ('cpt_codes', 'RVU'),
    'work rvu': ('cpt_codes', 'RVU'),

    'revenue': ('claims', 'PAID_AMOUNT'),
    'reimbursing': ('claims', 'PAID_AMOUNT'),
    'paying out': ('claims', 'PAID_AMOUNT'),
    'pay out': ('claims', 'PAID_AMOUNT'),
    'paid out': ('claims', 'PAID_AMOUNT'),

    'medication name': ('prescriptions', 'MEDICATION_NAME'),
    'drug name': ('prescriptions', 'MEDICATION_NAME'),

    'drug spend': ('prescriptions', 'COST'),
    'prescription cost': ('prescriptions', 'COST'),
    'cost of prescription': ('prescriptions', 'COST'),
    'cost per prescription': ('prescriptions', 'COST'),
    'cost of prescriptions': ('prescriptions', 'COST'),
    'medication cost': ('prescriptions', 'COST'),
    'drug cost': ('prescriptions', 'COST'),
    'pharmacy cost': ('prescriptions', 'COST'),
    'rx cost': ('prescriptions', 'COST'),
    'expensive medication': ('prescriptions', 'COST'),
    'most expensive medication': ('prescriptions', 'COST'),
    'costliest medication': ('prescriptions', 'COST'),
    'prescription copay': ('prescriptions', 'COPAY'),
    'rx copay': ('prescriptions', 'COPAY'),

    'days supply': ('prescriptions', 'DAYS_SUPPLY'),
    'day supply': ('prescriptions', 'DAYS_SUPPLY'),
    'supply days': ('prescriptions', 'DAYS_SUPPLY'),
    'prescription duration': ('prescriptions', 'DAYS_SUPPLY'),
    'rx duration': ('prescriptions', 'DAYS_SUPPLY'),
    'fill duration': ('prescriptions', 'DAYS_SUPPLY'),
    'quantity': ('prescriptions', 'QUANTITY'),
    'pills': ('prescriptions', 'QUANTITY'),
    'refill': ('prescriptions', 'REFILLS_USED'),
    'refills': ('prescriptions', 'REFILLS_USED'),
    'refills authorized': ('prescriptions', 'REFILLS_AUTHORIZED'),
    'pharmacy': ('prescriptions', 'PHARMACY'),
    'ndc': ('prescriptions', 'NDC_CODE'),
    'ndc code': ('prescriptions', 'NDC_CODE'),
    'medication': ('prescriptions', 'MEDICATION_NAME'),

    'denial': ('claims', 'CLAIM_STATUS'),
    'denied': ('claims', 'CLAIM_STATUS'),
    'rejected': ('claims', 'CLAIM_STATUS'),
    'rejection': ('claims', 'CLAIM_STATUS'),
    'approval': ('claims', 'CLAIM_STATUS'),
    'approved': ('claims', 'CLAIM_STATUS'),
    'clean claim': ('claims', 'CLAIM_STATUS'),

    'region': ('members', 'KP_REGION'),
    'geography': ('members', 'KP_REGION'),
    'area': ('members', 'KP_REGION'),
    'geographic': ('members', 'KP_REGION'),

    'age group': ('members', 'AGE_GROUP'),
    'gender': ('members', 'GENDER'),
    'race': ('members', 'RACE'),
    'ethnicity': ('members', 'RACE'),

    'plan type': ('claims', 'PLAN_TYPE'),
    'insurance plan': ('claims', 'PLAN_TYPE'),
    'plan': ('claims', 'PLAN_TYPE'),
    'payer': ('claims', 'PLAN_TYPE'),
    'hmo': ('claims', 'PLAN_TYPE'),
    'ppo': ('claims', 'PLAN_TYPE'),

    'visit type': ('encounters', 'VISIT_TYPE'),
    'encounter type': ('encounters', 'VISIT_TYPE'),
    'department': ('encounters', 'DEPARTMENT'),

    'specialty': ('providers', 'SPECIALTY'),

    'diagnosis': ('diagnoses', 'ICD10_DESCRIPTION'),
    'diagnoses': ('diagnoses', 'ICD10_DESCRIPTION'),
    'diagnosed': ('diagnoses', 'ICD10_DESCRIPTION'),
    'condition': ('diagnoses', 'ICD10_DESCRIPTION'),
    'conditions': ('diagnoses', 'ICD10_DESCRIPTION'),
    'disease': ('diagnoses', 'ICD10_DESCRIPTION'),
    'diseases': ('diagnoses', 'ICD10_DESCRIPTION'),
    'icd code': ('diagnoses', 'ICD10_CODE'),
    'icd10': ('diagnoses', 'ICD10_CODE'),
    'diagnosis code': ('diagnoses', 'ICD10_CODE'),
    'clinical condition': ('diagnoses', 'ICD10_DESCRIPTION'),

    'procedure': ('cpt_codes', 'CPT_DESCRIPTION'),
    'procedures': ('cpt_codes', 'CPT_DESCRIPTION'),
    'cpt code': ('cpt_codes', 'CPT_CODE'),
    'cpt': ('cpt_codes', 'CPT_CODE'),
    'surgery': ('cpt_codes', 'CPT_DESCRIPTION'),
    'operation': ('cpt_codes', 'CPT_DESCRIPTION'),

    'referral': ('referrals', 'REFERRAL_REASON'),
    'referred': ('referrals', 'REFERRAL_REASON'),
    'referral reason': ('referrals', 'REFERRAL_REASON'),

    'appointment': ('appointments', 'APPOINTMENT_TYPE'),
    'appointments': ('appointments', 'APPOINTMENT_TYPE'),
    'scheduled': ('appointments', 'STATUS'),
    'no show': ('appointments', 'STATUS'),
    'wait time': ('appointments', 'WAIT_DAYS'),

    'encounter': ('encounters', 'ENCOUNTER_ID'),
    'visit': ('encounters', 'VISIT_TYPE'),
    'admission': ('encounters', 'VISIT_TYPE'),
    'discharge': ('encounters', 'DISCHARGE_DISPOSITION'),
    'readmission': ('encounters', 'READMISSION_FLAG'),
    'readmit': ('encounters', 'READMISSION_FLAG'),

    'provider': ('providers', 'PROVIDER_NAME'),
    'doctor': ('providers', 'PROVIDER_NAME'),
    'physician': ('providers', 'PROVIDER_NAME'),
    'npi': ('providers', 'NPI'),

    'member': ('members', 'MEMBER_ID'),
    'patient': ('members', 'MEMBER_ID'),
    'enrollee': ('members', 'MEMBER_ID'),
    'beneficiary': ('members', 'MEMBER_ID'),
    'subscriber': ('members', 'MEMBER_ID'),

    'facility': ('claims', 'FACILITY'),
    'hospital': ('claims', 'FACILITY'),
    'clinic': ('claims', 'FACILITY'),
    'location': ('claims', 'FACILITY'),

    'claim type': ('claims', 'CLAIM_TYPE'),
    'claim status': ('claims', 'CLAIM_STATUS'),
    'service date': ('claims', 'SERVICE_DATE'),
    'date of service': ('claims', 'SERVICE_DATE'),

    'processing time': ('claims', 'ADJUDICATED_DATE'),
    'turnaround': ('claims', 'ADJUDICATED_DATE'),
    'adjudication': ('claims', 'ADJUDICATED_DATE'),
    'submitted': ('claims', 'SUBMITTED_DATE'),
}

AGG_SYNONYMS = {
    'AVG': ['average', 'avg', 'mean', 'typical', 'normally', 'usually',
            'on average', 'per claim', 'per encounter', 'per visit',
            'going rate', 'typical amount', 'expected', 'anticipated',
            'what does', 'looking like', 'how much does', 'how much are',
            'standard', 'per member', 'per patient'],
    'SUM': ['total', 'sum', 'combined', 'aggregate', 'altogether',
            'add up', 'tally up', 'tally', 'overall', 'cumulative',
            'grand total', 'all together', 'combined total',
            'how much total', 'totals', 'run me the totals',
            'sum total', 'combined amount'],
    'MAX': ['maximum', 'max', 'highest', 'peak', 'largest', 'biggest',
            'most expensive', 'greatest', 'ceiling', 'upper bound',
            'longest', 'highest single'],
    'MIN': ['minimum', 'min', 'lowest', 'smallest', 'cheapest',
            'least expensive', 'bottom', 'floor', 'lower bound',
            'shortest', 'lowest single'],
    'COUNT': ['count', 'how many', 'number of', 'total number',
              'headcount', 'head count',
              'tally of', 'give me the number'],
}

import re as _re_agg
_EXPLICIT_AGG_OVERRIDES_LIST = [
    ('total number', 'COUNT'),
    ('number of', 'COUNT'),
    ('how many', 'COUNT'),
    ('sum of', 'SUM'),
    ('sum total', 'SUM'),
    ('tally up', 'SUM'),
    ('grand total', 'SUM'),
    ('add up', 'SUM'),
    ('cumulative', 'SUM'),
    ('average', 'AVG'),
    ('avg', 'AVG'),
    ('mean', 'AVG'),
    ('maximum', 'MAX'),
    ('minimum', 'MIN'),
]
_EXPLICIT_AGG_WORD_BOUNDARY = [
    (r'\bcount\b', 'COUNT'),
    (r'\bmean\b', 'AVG'),
]

ENTITY_SYNONYMS = {
    'claims': ['claim', 'claims', 'billing', 'submission', 'submissions',
               'charges', 'reimbursement', 'spending', 'spend', 'cost',
               'expense', 'expenditure'],
    'members': ['member', 'members', 'patient', 'patients', 'enrollee',
                'enrollees', 'people', 'population', 'covered lives',
                'lives', 'membership', 'enrolled', 'beneficiary',
                'beneficiaries', 'subscriber', 'region', 'regions'],
    'providers': ['provider', 'providers', 'doctor', 'doctors', 'physician',
                  'physicians', 'clinician', 'clinicians', 'npi', 'specialist',
                  'specialists', 'specialty', 'specialties', 'doc ', 'docs',
                  'practitioner'],
    'encounters': ['encounter', 'encounters', 'visit', 'visits', 'admission',
                   'admissions', 'stay', 'stays', 'hospitalization',
                   'utilization', 'inpatient', 'outpatient', 'emergency',
                   'department', 'departments'],
    'diagnoses': ['diagnosis', 'diagnoses', 'condition', 'conditions',
                  'disease', 'diseases', 'icd', 'comorbidity', 'comorbidities',
                  'diagnosed', 'diagnostic', 'morbidity'],
    'prescriptions': ['prescription', 'prescriptions', 'medication', 'medications',
                      'drug', 'drugs', 'rx', 'script', 'scripts', 'pharmacy',
                      'pharmaceutical', 'therapeutic class', 'drug class',
                      'medication class'],
    'referrals': ['referral', 'referrals', 'authorization', 'authorizations',
                  'auth', 'auths', 'prior auth'],
    'appointments': ['appointment', 'appointments', 'scheduling', 'slot',
                     'booking', 'bookings', 'no show', 'no-show'],
}


class LanguageBrain:

    def __init__(self, schema_graph, embedder=None):
        self.graph = schema_graph
        self._embedder = embedder
        self._intent_centroids = {}
        self._trained = False
        self._phrase_column_index = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from semantic_embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def train(self):
        if self._trained:
            return

        logger.info("LanguageBrain training on %d intent categories...",
                    len(INTENT_TRAINING_DATA))

        for intent_name, data in INTENT_TRAINING_DATA.items():
            all_examples = data['seeds'] + data['paraphrases']
            embeddings = [self.embedder.embed_question(self._normalize_text(ex)) for ex in all_examples]

            if embeddings:
                dim = embeddings[0].dim
                centroid_data = [0.0] * dim
                for emb in embeddings:
                    for j in range(dim):
                        centroid_data[j] += emb.data[j]
                n = len(embeddings)
                centroid_data = [v / n for v in centroid_data]
                from semantic_embedder import EmbeddingVector
                self._intent_centroids[intent_name] = EmbeddingVector(centroid_data).normalize()

            logger.debug("  %s: %d examples → centroid computed", intent_name, len(all_examples))

        for phrase, (table, col) in PHRASE_TO_COLUMN.items():
            emb = self.embedder.embed_question(self._normalize_text(phrase))
            self._phrase_column_index[phrase] = {
                'embedding': emb,
                'table': table,
                'column': col,
            }

        self._trained = True
        logger.info("LanguageBrain trained: %d intent centroids, %d phrase→column mappings",
                    len(self._intent_centroids), len(self._phrase_column_index))


    INTENT_KEYWORD_SIGNALS: Dict[str, Dict] = {
        'count': {
            'positive': [
                r'\bhow many\b', r'\bnumber of\b', r'\btotal number\b',
                r'\bcount\b', r'\bheadcount\b', r'\btally\b',
                r'\bpopulation size\b', r'\bvolume\b', r'\benrolled\b',
                r'\bhow many\b.*\bdo we\b',
            ],
            'negative': [],
            'boost': 0.25,
        },
        'aggregate': {
            'positive': [
                r'\baverage\b', r'\bavg\b', r'\bmean\b', r'\btypical\b',
                r'\btotal\s+(?:paid|cost|spend|billed|charge|amount)',
                r'\bsum\s+of\b', r'\badd up\b', r'\bcombined\b',
                r'\bper\s+(?:claim|encounter|visit|member)\b',
                r'\bhow much\b', r'\boverall\s+(?:cost|spend)',
                r'\bon average\b', r'\btally up\b', r'\bgrand total\b',
                r'\baggregate\b', r'\bcumulative\b',
                r'\bgoing rate\b',
                r'\bfloor\b', r'\bceiling\b', r'\bpeak\b',
                r'\bexpected\b', r'\banticipated\b',
                r'\bmaximum\b', r'\bminimum\b',
                r'\bmax\s+(?:of|for)\b', r'\bmin\s+(?:of|for)\b',
                r'\bhow\s+(?:long|big|much|high|low)\b',
                r'\blongest\b', r'\bshortest\b',
            ],
            'negative': [
                r'\bhow much\b.*\bhow many\b',
            ],
            'boost': 0.25,
        },
        'rate': {
            'positive': [
                r'\brate\b', r'\bpercentage\b', r'\bpercent\b',
                r'\bhow often\b', r'\bfrequency\b', r'\bproportion\b',
                r'\bfraction\b', r'\bratio\b', r'\bprevalence\b',
                r'\b%\s+of\b', r'\bshare\b',
                r'\bclean claim\b', r'\bdenial\s+frequency\b',
                r'\brejection\s+rate\b', r'\bfrequently\b',
                r'\bhow\s+(?:frequently|often)\b',
            ],
            'negative': [
                r'\bgoing rate\b',
                r'\breimbursement rate\b',
                r'\brate for\b',
            ],
            'boost': 0.25,
        },
        'rank': {
            'positive': [
                r'\btop\s+\d+\b', r'\bhighest\b', r'\blowest\b',
                r'\bmost\b', r'\bleast\b', r'\bbiggest\b', r'\bsmallest\b',
                r'\brank\b', r'\bleading\b', r'\bcostliest\b',
                r'\bwhich\s+\w+\s+has\s+the\b', r'\bmost\s+(?:expensive|common|frequent)',
            ],
            'negative': [],
            'boost': 0.15,
        },
        'trend': {
            'positive': [
                r'\bover time\b', r'\btrend\b', r'\bmonthly\b',
                r'\bquarterly\b', r'\byear over year\b', r'\bby month\b',
                r'\bquarter over quarter\b', r'\btrajectory\b',
                r'\btime series\b', r'\bchanged?\s+over\b',
            ],
            'negative': [],
            'boost': 0.15,
        },
        'compare': {
            'positive': [
                r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b',
                r'\bcomparison\b', r'\bside by side\b', r'\bcontrast\b',
                r'\bstack up\b', r'\bdifference\s+(?:between|in)\b',
            ],
            'negative': [],
            'boost': 0.15,
        },
        'list': {
            'positive': [
                r'\blist\b', r'\bshow me\b', r'\bdisplay\b',
                r'\bbreak down\b', r'\borganized by\b',
                r'\bgrouped by\b', r'\bgive me the\b.*\bdata\b',
            ],
            'negative': [],
            'boost': 0.10,
        },
    }

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("'", "").replace("\u2019", "").replace("\u2018", "")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def classify_intent(self, question: str) -> Tuple[str, float]:
        if not self._trained:
            self.train()

        q_normalized = self._normalize_text(question)
        q_emb = self.embedder.embed_question(q_normalized)
        q_lower = question.lower()

        all_sims = {}
        for intent_name, centroid in self._intent_centroids.items():
            all_sims[intent_name] = q_emb.cosine(centroid)

        boosted_sims = dict(all_sims)
        for intent_name, signals in self.INTENT_KEYWORD_SIGNALS.items():
            if intent_name not in boosted_sims:
                continue

            suppressed = False
            for neg_pat in signals.get('negative', []):
                if re.search(neg_pat, q_lower):
                    suppressed = True
                    break

            if suppressed:
                continue

            for pos_pat in signals['positive']:
                if re.search(pos_pat, q_lower):
                    boosted_sims[intent_name] += signals['boost']
                    break

        best_intent = max(boosted_sims, key=boosted_sims.get)
        best_sim = boosted_sims[best_intent]

        sorted_sims = sorted(boosted_sims.values(), reverse=True)
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]

        confidence = min(best_sim * (1.0 + margin * 2), 1.0)

        return best_intent, confidence

    def resolve_agg_function(self, question: str) -> Optional[str]:
        q_lower = question.lower()

        for phrase, fn in _EXPLICIT_AGG_OVERRIDES_LIST:
            if phrase in q_lower:
                return fn
        for pattern, fn in _EXPLICIT_AGG_WORD_BOUNDARY:
            if re.search(pattern, q_lower):
                return fn

        best_fn = None
        best_score = 0

        for fn, synonyms in AGG_SYNONYMS.items():
            for syn in synonyms:
                if syn in q_lower:
                    score = len(syn)
                    if score > best_score:
                        best_score = score
                        best_fn = fn

        return best_fn

    def resolve_column(self, question: str, table_hint: str = '') -> Optional[Tuple[str, str]]:
        if not self._trained:
            self.train()

        q_lower = question.lower()

        _entity_context = set()
        for table_name, synonyms in ENTITY_SYNONYMS.items():
            for syn in synonyms:
                if syn in q_lower:
                    _entity_context.add(table_name)
                    break

        sorted_phrases = sorted(PHRASE_TO_COLUMN.keys(), key=len, reverse=True)

        if _entity_context:
            for phrase in sorted_phrases:
                if phrase in q_lower:
                    table, col = PHRASE_TO_COLUMN[phrase]
                    if table in _entity_context:
                        if not table_hint or table == table_hint:
                            return (table, col)

        for phrase in sorted_phrases:
            if phrase in q_lower:
                table, col = PHRASE_TO_COLUMN[phrase]
                if not table_hint or table == table_hint:
                    return (table, col)

        q_emb = self.embedder.embed_question(self._normalize_text(question))
        best_match = None
        best_sim = 0.3

        for phrase, info in self._phrase_column_index.items():
            sim = q_emb.cosine(info['embedding'])
            effective_sim = sim
            if info['table'] in _entity_context:
                effective_sim += 0.15
            if effective_sim > best_sim:
                if not table_hint or info['table'] == table_hint:
                    best_sim = effective_sim
                    best_match = (info['table'], info['column'])

        return best_match

    def resolve_entity(self, question: str) -> List[str]:
        q_lower = question.lower()
        matched_tables = []

        for table_name, synonyms in ENTITY_SYNONYMS.items():
            for syn in synonyms:
                if syn in q_lower:
                    if table_name not in matched_tables:
                        matched_tables.append(table_name)
                    break

        return matched_tables

    def understand(self, question: str) -> Dict[str, Any]:
        if not self._trained:
            self.train()

        intent, intent_confidence = self.classify_intent(question)

        _rate_col_phrases = ['contracted rate', 'negotiated rate', 'reimbursement rate',
                             'going rate', 'rate per claim', 'rate per visit']
        if intent == 'rate' and any(p in question.lower() for p in _rate_col_phrases):
            intent = 'aggregate'
            intent_confidence = max(intent_confidence, 0.70)

        agg_fn = self.resolve_agg_function(question)

        column_match = self.resolve_column(question)

        tables = self.resolve_entity(question)

        understanding = {
            'intent': intent,
            'intent_confidence': intent_confidence,
            'agg_function': agg_fn,
            'target_column': column_match,
            'tables': tables,
            'confident': intent_confidence >= 0.60,
        }

        q_lower = question.lower()

        if re.search(r'\bper\s+(?:member|patient|enrollee)\b', q_lower):
            understanding['sub_intent'] = 'per_unit'
            if 'members' not in tables:
                understanding['tables'].append('members')

        by_match = re.search(r'\bby\s+(\w+(?:\s+\w+){0,2})', q_lower)
        if by_match:
            by_term = by_match.group(1)
            group_col = self._resolve_group_by(by_term)
            if group_col:
                understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            bd_match = re.search(r'(?:break\s+down|broken\s+(?:down|out))\b.*?\bby\s+(\w+(?:\s+\w+){0,2})', q_lower)
            if bd_match:
                group_col = self._resolve_group_by(bd_match.group(1))
                if group_col:
                    understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            strat_match = re.search(r'(?:stratified|segmented|grouped|categorized|bucketed)\s+by\s+(\w+(?:\s+\w+){0,2})', q_lower)
            if strat_match:
                group_col = self._resolve_group_by(strat_match.group(1))
                if group_col:
                    understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            across_match = re.search(r'\bacross\s+(?:different\s+|our\s+|the\s+)?(\w+(?:\s+\w+)?)', q_lower)
            if across_match:
                group_col = self._resolve_group_by(across_match.group(1))
                if group_col:
                    understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            per_match = re.search(r'\bper\s+(\w+(?:\s+\w+)?)', q_lower)
            if per_match:
                per_term = per_match.group(1)
                per_unit_terms = {'claim', 'visit', 'encounter', 'member', 'patient', 'enrollee', 'capita'}
                if per_term not in per_unit_terms:
                    group_col = self._resolve_group_by(per_term)
                    if group_col:
                        understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            count_per_match = re.search(r'count\s+per\s+(\w+(?:\s+\w+)?)', q_lower)
            if count_per_match:
                group_col = self._resolve_group_by(count_per_match.group(1))
                if group_col:
                    understanding['group_by'] = group_col

        if not understanding.get('group_by'):
            vary_match = re.search(r'(?:var(?:y|ies)|compare|differ)\s+by\s+(\w+(?:\s+\w+)?)', q_lower)
            if vary_match:
                group_col = self._resolve_group_by(vary_match.group(1))
                if group_col:
                    understanding['group_by'] = group_col

        _superlative_words = ['highest', 'lowest', 'most', 'least', 'top',
                              'best', 'worst', 'peak', 'biggest', 'smallest',
                              'largest', 'longest', 'shortest', 'costliest',
                              'cheapest', 'busiest', 'greatest', 'primary',
                              'most commonly', 'most frequently', 'most expensive']
        has_superlative = any(w in q_lower for w in _superlative_words)
        if has_superlative:
            non_groupable = {'claim', 'claims', 'spending', 'spend', 'cost',
                            'expense', 'charge', 'charges', 'billing',
                            'submission', 'submissions', 'reimbursement',
                            'data', 'amount', 'amounts', 'payment', 'payments'}
            groupable = any(syn in q_lower for syns in ENTITY_SYNONYMS.values()
                          for syn in syns if syn not in non_groupable)
            if not groupable and intent == 'rank':
                understanding['intent'] = 'aggregate'
                _min_words = ['lowest', 'least', 'smallest', 'cheapest',
                              'shortest', 'floor', 'minimum', 'min']
                if any(w in q_lower for w in _min_words):
                    understanding['agg_function'] = 'MIN'
                else:
                    understanding['agg_function'] = 'MAX'
            elif groupable:
                understanding['intent'] = 'rank'
                understanding['order_direction'] = 'DESC'
                _min_rank_words = ['lowest', 'least', 'smallest', 'cheapest', 'shortest']
                if any(w in q_lower for w in _min_rank_words):
                    understanding['order_direction'] = 'ASC'
                if not understanding.get('group_by'):
                    _RANK_GROUP_MAP = [
                        (['condition', 'diagnosis', 'diagnosed', 'disease'], ('diagnoses', 'ICD10_DESCRIPTION')),
                        (['provider', 'doctor', 'physician', 'clinician', 'docs', 'doc '], ('claims', 'RENDERING_NPI')),
                        (['specialty', 'specialties'], ('providers', 'SPECIALTY')),
                        (['region', 'area', 'geographic'], ('members', 'KP_REGION')),
                        (['medication', 'drug', 'rx'], ('prescriptions', 'MEDICATION_NAME')),
                        (['plan', 'insurance', 'payer'], ('claims', 'PLAN_TYPE')),
                        (['department', 'facility'], ('encounters', 'DEPARTMENT')),
                        (['visit type', 'visit types'], ('encounters', 'VISIT_TYPE')),
                        (['patient', 'member', 'enrollee'], ('claims', 'MEMBER_ID')),
                    ]
                    for keywords, group_col in _RANK_GROUP_MAP:
                        if any(kw in q_lower for kw in keywords):
                            understanding['group_by'] = group_col
                            if group_col[0] not in understanding['tables']:
                                understanding['tables'].append(group_col[0])
                            break

        if agg_fn in ('AVG', 'SUM', 'MAX', 'MIN') and understanding.get('intent') in ('count', 'rank'):
            has_group_signal = bool(understanding.get('group_by'))
            has_which = bool(re.search(r'\b(?:which|what)\s+\w+', q_lower))
            if not has_group_signal and not has_which:
                understanding['intent'] = 'aggregate'
                if agg_fn == 'AVG':
                    understanding['sub_intent'] = 'avg'
                elif agg_fn == 'SUM':
                    understanding['sub_intent'] = 'sum'

        if understanding.get('intent') == 'rank' and not understanding.get('agg_function'):
            _cost_words = ['expensive', 'costliest', 'costlier', 'revenue', 'spend']
            if any(w in q_lower for w in _cost_words):
                understanding['agg_function'] = 'SUM'
                if not column_match:
                    understanding['target_column'] = ('claims', 'PAID_AMOUNT')
                    if 'claims' not in understanding['tables']:
                        understanding['tables'].append('claims')
            else:
                understanding['agg_function'] = 'COUNT'

        if understanding.get('group_by'):
            grp_table = understanding['group_by'][0]
            if grp_table not in understanding.get('tables', []):
                understanding['tables'].append(grp_table)


        _tc = understanding.get('target_column')
        _gb = understanding.get('group_by')
        if _tc and _gb and _tc[1] == _gb[1]:
            _is_rx_context = 'prescriptions' in understanding.get('tables', [])
            _cost_col_words = {
                'paid': ('claims', 'PAID_AMOUNT'), 'reimbursement': ('claims', 'PAID_AMOUNT'),
                'cost': ('prescriptions', 'COST') if _is_rx_context else ('claims', 'PAID_AMOUNT'),
                'spend': ('prescriptions', 'COST') if _is_rx_context else ('claims', 'PAID_AMOUNT'),
                'billed': ('claims', 'BILLED_AMOUNT'), 'charge': ('claims', 'BILLED_AMOUNT'),
                'allowed': ('claims', 'ALLOWED_AMOUNT'), 'copay': ('claims', 'COPAY'),
                'out of pocket': ('claims', 'COPAY'), 'coinsurance': ('claims', 'COINSURANCE'),
                'deductible': ('claims', 'DEDUCTIBLE'),
                'risk score': ('members', 'RISK_SCORE'), 'acuity': ('members', 'RISK_SCORE'),
                'length of stay': ('encounters', 'LENGTH_OF_STAY'), 'stay': ('encounters', 'LENGTH_OF_STAY'),
                'panel size': ('providers', 'PANEL_SIZE'),
                'days supply': ('prescriptions', 'DAYS_SUPPLY'),
            }
            new_target = None
            for kw, col_tup in _cost_col_words.items():
                if kw in q_lower:
                    new_target = col_tup
                    break
            if new_target:
                understanding['target_column'] = new_target
                if new_target[0] not in understanding['tables']:
                    understanding['tables'].append(new_target[0])
            elif understanding.get('agg_function') in ('SUM', 'AVG') and understanding['intent'] in ('aggregate', 'rank'):
                if _is_rx_context:
                    understanding['target_column'] = ('prescriptions', 'COST')
                else:
                    understanding['target_column'] = ('claims', 'PAID_AMOUNT')
                    if 'claims' not in understanding['tables']:
                        understanding['tables'].append('claims')

        if understanding.get('intent') == 'rank' and understanding.get('group_by'):
            _cost_rank_words = ['expensive', 'costliest', 'costlier', 'revenue', 'spend', 'cost']
            if any(w in q_lower for w in _cost_rank_words):
                understanding['agg_function'] = 'SUM'
                if not understanding.get('target_column') or understanding['target_column'][1] in ('AGE_GROUP', 'CLAIM_STATUS', 'MEDICATION_NAME'):
                    if 'prescriptions' in understanding.get('tables', []):
                        understanding['target_column'] = ('prescriptions', 'COST')
                    else:
                        understanding['target_column'] = ('claims', 'PAID_AMOUNT')
                        if 'claims' not in understanding['tables']:
                            understanding['tables'].append('claims')

        if any(w in q_lower for w in ['stays', 'length of stay', 'los ']):
            if understanding.get('target_column') and understanding['target_column'][1] != 'LENGTH_OF_STAY':
                understanding['target_column'] = ('encounters', 'LENGTH_OF_STAY')
                if 'encounters' not in understanding['tables']:
                    understanding['tables'].append('encounters')

        if 'distribution' in q_lower and understanding.get('target_column'):
            tc_table, tc_col = understanding['target_column']
            if not understanding.get('group_by'):
                understanding['group_by'] = (tc_table, tc_col)
                understanding['intent'] = 'aggregate'
                if not understanding.get('agg_function'):
                    understanding['agg_function'] = 'COUNT'

        if any(w in q_lower for w in ['unique', 'distinct']):
            understanding['distinct'] = True

        if 'compare' in q_lower or 'comparison' in q_lower:
            understanding['intent'] = 'compare'
            _compare_cols = []
            for phrase, (t, c) in PHRASE_TO_COLUMN.items():
                if phrase in q_lower:
                    _compare_cols.append((t, c))
            if len(_compare_cols) >= 2:
                understanding['compare_columns'] = _compare_cols[:2]

        if understanding.get('intent') == 'rate' or any(w in q_lower for w in ['ratio', 'rate', 'frequency']):
            if any(w in q_lower for w in ['denied', 'denial', 'reject', 'clean claim']):
                understanding['intent'] = 'rate'
                understanding['rate_type'] = 'denial'
                if not understanding.get('target_column'):
                    understanding['target_column'] = ('claims', 'CLAIM_STATUS')
                if 'claims' not in understanding['tables']:
                    understanding['tables'].append('claims')

        if 'loss ratio' in q_lower:
            understanding['intent'] = 'compare'
            understanding['target_column'] = ('claims', 'PAID_AMOUNT')
            understanding['compare_columns'] = [('claims', 'PAID_AMOUNT'), ('claims', 'BILLED_AMOUNT')]
            if 'claims' not in understanding['tables']:
                understanding['tables'].append('claims')

        if understanding.get('group_by') and understanding.get('intent') == 'list':
            if any(w in q_lower for w in ['cost', 'spend', 'paid', 'billed', 'charge']):
                understanding['intent'] = 'aggregate'
                understanding['agg_function'] = 'SUM'
                if not understanding.get('target_column') or understanding['target_column'] == understanding['group_by']:
                    understanding['target_column'] = ('claims', 'PAID_AMOUNT')
                    if 'claims' not in understanding['tables']:
                        understanding['tables'].append('claims')

        if 'contracted rate' in q_lower or 'negotiated rate' in q_lower:
            understanding['intent'] = 'aggregate'
            understanding['target_column'] = ('claims', 'ALLOWED_AMOUNT')
            if not understanding.get('agg_function'):
                understanding['agg_function'] = 'AVG'
            if 'claims' not in understanding['tables']:
                understanding['tables'].append('claims')

        _rate_denial_phrases = ['denial frequency', 'claim denial', 'clean claim ratio',
                                'denial rate', 'getting rejected', 'denial problem',
                                'how frequently', 'how often', 'what share',
                                'what percentage', 'what percent']
        if any(p in q_lower for p in _rate_denial_phrases):
            if understanding.get('intent') not in ('rate',):
                understanding['intent'] = 'rate'
                understanding['rate_type'] = 'denial'
                if 'claims' not in understanding['tables']:
                    understanding['tables'].append('claims')
        elif any(w in q_lower for w in ['denied', 'rejected', 'rejection']):
            if understanding.get('intent') == 'rank' or any(w in q_lower for w in ['most', 'primary', 'top', 'biggest']):
                from intent_parser import ParsedFilter
                understanding['_denial_filter'] = True
                if 'claims' not in understanding['tables']:
                    understanding['tables'].append('claims')

        if any(w in q_lower for w in ['diabetic', 'diabetes', 'hypertension', 'chronic']):
            if 'diagnoses' not in understanding.get('tables', []):
                understanding['tables'].append('diagnoses')
        if any(w in q_lower for w in ['specialty', 'specialties', 'provider specialty']):
            if 'providers' not in understanding.get('tables', []):
                understanding['tables'].append('providers')
            if not understanding.get('group_by'):
                understanding['group_by'] = ('providers', 'SPECIALTY')

        if 'copay' in q_lower and 'paid' in q_lower:
            understanding['target_column'] = ('claims', 'COPAY')
            if 'claims' not in understanding['tables']:
                understanding['tables'].append('claims')

        if 'out of pocket' in q_lower:
            understanding['target_column'] = ('claims', 'COPAY')
            if not understanding.get('agg_function'):
                understanding['agg_function'] = 'AVG'
            if 'claims' not in understanding['tables']:
                understanding['tables'].append('claims')

        if understanding.get('sub_intent') == 'per_unit' and understanding.get('group_by'):
            understanding['sub_intent'] = 'avg'
            if not understanding.get('agg_function'):
                understanding['agg_function'] = 'AVG'

        if 'plan type' in q_lower and understanding.get('intent') == 'rate':
            if not understanding.get('group_by'):
                understanding['group_by'] = ('claims', 'PLAN_TYPE')
                if 'claims' not in understanding['tables']:
                    understanding['tables'].append('claims')

        if understanding.get('intent') == 'rank':
            if any(w in q_lower for w in ['diagnos', 'condition', 'icd']):
                if 'diagnoses' not in understanding.get('tables', []):
                    understanding['tables'].append('diagnoses')
                if not understanding.get('group_by'):
                    understanding['group_by'] = ('diagnoses', 'ICD10_DESCRIPTION')

        _rx_context = 'prescriptions' in understanding.get('tables', [])
        if _rx_context:
            _tc = understanding.get('target_column')
            if _tc and _tc == ('claims', 'PAID_AMOUNT'):
                if any(w in q_lower for w in ['cost', 'spend', 'expensive', 'price', 'expense']):
                    understanding['target_column'] = ('prescriptions', 'COST')
            if any(w in q_lower for w in ['expensive', 'costliest', 'cheapest', 'priciest']):
                if any(w in q_lower for w in ['medication', 'drug', 'rx', 'prescription']):
                    understanding['target_column'] = ('prescriptions', 'COST')
                    if understanding.get('intent') in ('rank', 'aggregate'):
                        if not understanding.get('group_by'):
                            understanding['group_by'] = ('prescriptions', 'MEDICATION_NAME')
                    if understanding.get('intent') == 'aggregate' and not understanding.get('group_by'):
                        understanding['agg_function'] = 'MAX'
            if any(w in q_lower for w in ['days supply', 'day supply', 'supply days']):
                understanding['target_column'] = ('prescriptions', 'DAYS_SUPPLY')
            if not understanding.get('target_column') and understanding.get('agg_function') in ('SUM', 'AVG', 'MAX', 'MIN'):
                _rx_financial_words = ['cost', 'spend', 'expensive', 'price', 'charge',
                                       'copay', 'paid', 'billed', 'reimburse']
                if any(w in q_lower for w in _rx_financial_words):
                    understanding['target_column'] = ('prescriptions', 'COST')

        if understanding.get('intent') == 'aggregate' and understanding.get('group_by'):
            _superlative_rank_words = ['most expensive', 'costliest', 'cheapest', 'busiest', 'most common']
            if any(w in q_lower for w in _superlative_rank_words):
                understanding['intent'] = 'rank'
                if understanding.get('agg_function') == 'MAX':
                    understanding['agg_function'] = 'SUM'

        logger.debug("LanguageBrain: %s → %s (conf=%.2f, agg=%s, col=%s, group=%s)",
                     question[:60], understanding.get('intent'), intent_confidence,
                     understanding.get('agg_function'), understanding.get('target_column'),
                     understanding.get('group_by'))

        return understanding

    def _resolve_group_by(self, term: str) -> Optional[Tuple[str, str]]:
        term = term.lower().strip()

        GROUP_TERM_MAP = {
            'provider': ('claims', 'RENDERING_NPI'),
            'doctor': ('claims', 'RENDERING_NPI'),
            'physician': ('claims', 'RENDERING_NPI'),
            'npi': ('claims', 'RENDERING_NPI'),
            'region': ('members', 'KP_REGION'),
            'area': ('members', 'KP_REGION'),
            'geography': ('members', 'KP_REGION'),
            'geographic region': ('members', 'KP_REGION'),
            'location': ('members', 'KP_REGION'),
            'plan': ('claims', 'PLAN_TYPE'),
            'plan type': ('claims', 'PLAN_TYPE'),
            'payer': ('claims', 'PLAN_TYPE'),
            'insurance': ('claims', 'PLAN_TYPE'),
            'insurance plan': ('claims', 'PLAN_TYPE'),
            'specialty': ('providers', 'SPECIALTY'),
            'specialty area': ('providers', 'SPECIALTY'),
            'provider specialty': ('providers', 'SPECIALTY'),
            'department': ('encounters', 'DEPARTMENT'),
            'facility': ('claims', 'FACILITY'),
            'gender': ('members', 'GENDER'),
            'age': ('members', 'AGE_GROUP'),
            'age group': ('members', 'AGE_GROUP'),
            'age groups': ('members', 'AGE_GROUP'),
            'visit type': ('encounters', 'VISIT_TYPE'),
            'encounter type': ('encounters', 'VISIT_TYPE'),
            'diagnosis': ('diagnoses', 'ICD10_DESCRIPTION'),
            'diagnosis category': ('diagnoses', 'ICD10_DESCRIPTION'),
            'condition': ('diagnoses', 'ICD10_DESCRIPTION'),
            'medication class': ('prescriptions', 'MEDICATION_CLASS'),
            'therapeutic class': ('prescriptions', 'MEDICATION_CLASS'),
            'drug class': ('prescriptions', 'MEDICATION_CLASS'),
            'medication': ('prescriptions', 'MEDICATION_NAME'),
            'drug': ('prescriptions', 'MEDICATION_NAME'),
            'claim type': ('claims', 'CLAIM_TYPE'),
            'race': ('members', 'RACE'),
            'ethnicity': ('members', 'RACE'),
            'status': ('claims', 'CLAIM_STATUS'),
        }

        if term in GROUP_TERM_MAP:
            return GROUP_TERM_MAP[term]

        for key, val in GROUP_TERM_MAP.items():
            if key in term or term in key:
                return val

        return None


class IntelligentIntentParser:

    def __init__(self, schema_graph, embedder=None):
        self.graph = schema_graph
        self.brain = LanguageBrain(schema_graph, embedder=embedder)
        self.brain.train()

        from intent_parser import IntentParser
        self.regex_parser = IntentParser(schema_graph)

    def parse(self, question: str) -> 'ParsedIntent':
        understanding = self.brain.understand(question)

        intent = self.regex_parser.parse(question)

        if understanding['confident']:
            brain_intent = understanding['intent']
            brain_agg = understanding.get('agg_function')
            brain_col = understanding.get('target_column')
            brain_tables = understanding.get('tables', [])

            if brain_intent != intent.intent:
                logger.info("LanguageBrain override intent: %s → %s (conf=%.2f)",
                           intent.intent, brain_intent, understanding['intent_confidence'])
                intent.intent = brain_intent

            if brain_agg:
                if brain_agg != intent.agg_function:
                    logger.info("LanguageBrain override agg: %s → %s",
                               intent.agg_function, brain_agg)
                intent.agg_function = brain_agg
                _sub_map = {'AVG': 'avg', 'SUM': 'sum', 'MAX': 'max', 'MIN': 'min'}
                if brain_agg in _sub_map:
                    intent.sub_intent = _sub_map[brain_agg]

            if brain_col:
                brain_table, brain_column = brain_col
                intent.agg_table = brain_table
                intent.agg_column = brain_column
                if brain_table not in intent.tables:
                    intent.tables.append(brain_table)

            for t in brain_tables:
                if t not in intent.tables:
                    intent.tables.append(t)

            brain_group = understanding.get('group_by')
            if brain_group:
                if not intent.group_by:
                    intent.group_by.append(brain_group)
                elif brain_intent == 'rank' and brain_group not in intent.group_by:
                    intent.group_by = [brain_group]
                if brain_group[0] not in intent.tables:
                    intent.tables.append(brain_group[0])

            if understanding.get('sub_intent') and not intent.sub_intent:
                intent.sub_intent = understanding['sub_intent']

            if understanding.get('order_direction') and not intent.order_by:
                intent.order_by = understanding['order_direction'].lower()
            elif understanding.get('intent') == 'rank' and not intent.order_by:
                intent.order_by = 'desc'

            if understanding.get('distinct'):
                intent.distinct = True
            if understanding.get('compare_columns'):
                intent.compare_columns = understanding['compare_columns']
            if understanding.get('rate_type'):
                intent.rate_type = understanding['rate_type']

            if understanding.get('_denial_filter'):
                from intent_parser import ParsedFilter
                has_denial = any(f.column == 'CLAIM_STATUS' for f in intent.filters)
                if not has_denial:
                    intent.filters.append(ParsedFilter(
                        column='CLAIM_STATUS', operator='=', value='DENIED',
                        table_hint='claims', confidence=0.90
                    ))

        else:
            brain_col = understanding.get('target_column')
            brain_agg = understanding.get('agg_function')
            brain_group = understanding.get('group_by')
            brain_tables = understanding.get('tables', [])
            brain_intent = understanding.get('intent')

            q_lower = question.lower()

            if brain_col:
                brain_table, brain_column = brain_col
                if not intent.agg_column:
                    intent.agg_table = brain_table
                    intent.agg_column = brain_column
                    if brain_table not in intent.tables:
                        intent.tables.append(brain_table)
                    logger.info("Brain soft-override column: %s.%s", brain_table, brain_column)

            if brain_agg and not intent.agg_function:
                _agg_words = {'AVG': ['average', 'avg', 'mean', 'typical', 'per '],
                              'SUM': ['total', 'sum', 'cumulative', 'combined', 'tally'],
                              'MAX': ['max', 'highest', 'peak', 'largest', 'longest'],
                              'MIN': ['min', 'lowest', 'smallest', 'floor', 'least']}
                if brain_agg in _agg_words:
                    if any(w in q_lower for w in _agg_words[brain_agg]):
                        intent.agg_function = brain_agg
                        _sub_map = {'AVG': 'avg', 'SUM': 'sum', 'MAX': 'max', 'MIN': 'min'}
                        intent.sub_intent = _sub_map.get(brain_agg, intent.sub_intent)
                        logger.info("Brain soft-override agg: %s", brain_agg)

            if brain_group and not intent.group_by:
                intent.group_by.append(brain_group)
                if brain_group[0] not in intent.tables:
                    intent.tables.append(brain_group[0])
                logger.info("Brain soft-override group_by: %s.%s", brain_group[0], brain_group[1])

            _rank_indicators = ['most', 'top', 'busiest', 'highest', 'longest',
                                'costliest', 'biggest', 'primary', 'commonly']
            if brain_intent == 'rank' and any(w in q_lower for w in _rank_indicators):
                if intent.intent not in ('rank',):
                    intent.intent = 'rank'
                    if not intent.order_by:
                        intent.order_by = 'desc'
                    logger.info("Brain soft-override intent to rank")

            _agg_indicators = ['average', 'avg', 'mean', 'total', 'sum', 'break down',
                               'broken down', 'stratified', 'per ']
            if brain_intent == 'aggregate' and any(w in q_lower for w in _agg_indicators):
                if intent.intent not in ('aggregate', 'rank'):
                    intent.intent = 'aggregate'
                    logger.info("Brain soft-override intent to aggregate")

            for t in brain_tables:
                if t not in intent.tables:
                    intent.tables.append(t)

            if understanding.get('distinct'):
                intent.distinct = True
            if understanding.get('compare_columns'):
                intent.compare_columns = understanding['compare_columns']
            if understanding.get('rate_type'):
                intent.rate_type = understanding['rate_type']

        return intent


_GLOBAL_BRAIN: Optional[LanguageBrain] = None


def get_brain(schema_graph) -> LanguageBrain:
    global _GLOBAL_BRAIN
    if _GLOBAL_BRAIN is None:
        _GLOBAL_BRAIN = LanguageBrain(schema_graph)
        _GLOBAL_BRAIN.train()
    return _GLOBAL_BRAIN


def create_intelligent_parser(schema_graph, embedder=None) -> IntelligentIntentParser:
    return IntelligentIntentParser(schema_graph, embedder=embedder)
