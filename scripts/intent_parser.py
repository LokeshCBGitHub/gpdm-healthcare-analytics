import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.intent_parser')


TYPO_MAP = {
    'memebers': 'members', 'memebrs': 'members', 'memebrrs': 'members',
    'patints': 'patients', 'paitents': 'patients', 'patiens': 'patients',
    'clams': 'claims', 'cliam': 'claims', 'calims': 'claims',
    'specialites': 'specialties', 'specialites': 'specialties', 'specailties': 'specialties',
    'speciality': 'specialty', 'spcialty': 'specialty',
    'encountrs': 'encounters', 'encountr': 'encounter',
    'perscriptions': 'prescriptions', 'presriptions': 'prescriptions',
    'diagnosies': 'diagnoses', 'diagnois': 'diagnosis',
    'provders': 'providers', 'providrs': 'providers',
    'refferals': 'referrals', 'refferal': 'referral',
    'amnt': 'amount', 'amout': 'amount', 'ammount': 'amount',
    'averge': 'average', 'avarage': 'average', 'avgerage': 'average',
    'toatl': 'total', 'totla': 'total',
    'billd': 'billed', 'billled': 'billed',
    'deniied': 'denied', 'dennied': 'denied',
    'facilty': 'facility', 'faclity': 'facility',
    'reigon': 'region', 'regon': 'region',
    'medcation': 'medication', 'medicaton': 'medication',
    'phamacy': 'pharmacy', 'pharamcy': 'pharmacy',
    'clam': 'claim',
    'whats': "what's", 'hwo': 'how', 'waht': 'what',
}

def normalize_typos(text: str) -> str:
    words = text.split()
    result = []
    for w in words:
        lower = w.lower()
        if lower in TYPO_MAP:
            result.append(TYPO_MAP[lower])
        else:
            result.append(w)
    return ' '.join(result)


@dataclass
class ParsedFilter:
    column: str
    operator: str
    value: Any
    table_hint: str = ''
    confidence: float = 0.8


@dataclass
class ParsedIntent:
    original_question: str
    normalized_question: str

    intent: str
    sub_intent: str = ''

    tables: List[str] = field(default_factory=list)
    columns: List[Tuple[str, str]] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)

    filters: List[ParsedFilter] = field(default_factory=list)
    group_by: List[Tuple[str, str]] = field(default_factory=list)
    order_by: str = ''
    limit: Optional[int] = None
    distinct: bool = False

    temporal: bool = False
    time_granularity: str = ''
    time_range: Optional[str] = None
    trend: bool = False

    agg_function: str = ''
    agg_column: str = ''
    agg_table: str = ''

    comparison: bool = False
    compare_dimension: str = ''
    compare_values: List[str] = field(default_factory=list)

    negation: bool = False

    confidence: float = 0.5
    ambiguous: bool = False


class IntentParser:

    def __init__(self, schema_graph):
        self.graph = schema_graph

        self.value_map = {
            'hmo': ('PLAN_TYPE', 'HMO'),
            'ppo': ('PLAN_TYPE', 'PPO'),
            'epo': ('PLAN_TYPE', 'EPO'),
            'hdhp': ('PLAN_TYPE', 'HDHP'),
            'denied': ('CLAIM_STATUS', 'DENIED'),
            'denial': ('CLAIM_STATUS', 'DENIED'),
            'pending': ('CLAIM_STATUS', 'PENDING'),
            'approved': ('CLAIM_STATUS', 'APPROVED'),
            'approval': ('CLAIM_STATUS', 'APPROVED'),
            'appealed': ('CLAIM_STATUS', 'APPEALED'),
            'adjusted': ('CLAIM_STATUS', 'ADJUSTED'),
            'emergency': ('VISIT_TYPE', 'EMERGENCY'),
            'inpatient': ('VISIT_TYPE', 'INPATIENT'),
            'outpatient': ('VISIT_TYPE', 'OUTPATIENT'),
            'telehealth': ('VISIT_TYPE', 'TELEHEALTH'),
            'home health': ('VISIT_TYPE', 'HOME_HEALTH'),
            'office visit': ('VISIT_TYPE', 'OFFICE_VISIT'),
            'male': ('GENDER', 'M'),
            'female': ('GENDER', 'F'),
            'active': ('STATUS', 'ACTIVE'),
            'inactive': ('STATUS', 'INACTIVE'),
            'ncal': ('KP_REGION', 'NCAL'),
            'scal': ('KP_REGION', 'SCAL'),
            'colorado': ('KP_REGION', 'CO'),
            'co': ('KP_REGION', 'CO'),
            'hawaii': ('KP_REGION', 'HI'),
            'hi': ('KP_REGION', 'HI'),
            'northwest': ('KP_REGION', 'NW'),
            'nw': ('KP_REGION', 'NW'),
            'georgia': ('KP_REGION', 'GA'),
            'ga': ('KP_REGION', 'GA'),
            'mid-atlantic': ('KP_REGION', 'MAS'),
            'mas': ('KP_REGION', 'MAS'),
            'mid': ('KP_REGION', 'MID'),
            'chronic': ('IS_CHRONIC', 'Y'),
            'filled': ('STATUS', 'FILLED'),
            'cancelled': ('STATUS', 'CANCELLED'),
            'no-show': ('STATUS', 'NO_SHOW'),
            'noshow': ('STATUS', 'NO_SHOW'),
            'primary care': ('VISIT_TYPE', 'OFFICE_VISIT'),
            'professional': ('CLAIM_TYPE', 'PROFESSIONAL'),
            'institutional': ('CLAIM_TYPE', 'INSTITUTIONAL'),
            'urgent care': ('VISIT_TYPE', 'URGENT_CARE'),
            'specialist': ('VISIT_TYPE', 'SPECIALIST'),
            'surgical': ('VISIT_TYPE', 'SURGICAL'),
            'pediatric': ('SPECIALTY', 'PEDIATRICS'),
            'cardiology': ('SPECIALTY', 'CARDIOLOGY'),
            'cardiologist': ('SPECIALTY', 'CARDIOLOGY'),
            'orthopedic': ('SPECIALTY', 'ORTHOPEDICS'),
            'dermatology': ('SPECIALTY', 'DERMATOLOGY'),
            'neurology': ('SPECIALTY', 'NEUROLOGY'),
            'oncology specialty': ('SPECIALTY', 'ONCOLOGY'),
            'psychiatry': ('SPECIALTY', 'PSYCHIATRY'),
            'endocrinology': ('SPECIALTY', 'ENDOCRINOLOGY'),
            'pulmonology': ('SPECIALTY', 'PULMONOLOGY'),
            'gastroenterology': ('SPECIALTY', 'GASTROENTEROLOGY'),
            'nephrology': ('SPECIALTY', 'NEPHROLOGY'),
            'high risk': ('RISK_SCORE', '>3.0'),
            'low risk': ('RISK_SCORE', '<1.5'),
            'elderly': ('AGE_GROUP', '65+'),
            'senior': ('AGE_GROUP', '65+'),
            'pediatric patient': ('AGE_GROUP', '0-17'),
            'young adult': ('AGE_GROUP', '18-34'),
        }

        self.disease_patterns = {
            'diabetes': 'diabetes',
            'diabetic': 'diabetes',
            'type 2 diabetes': 'diabetes',
            'type 1 diabetes': 'diabetes',
            't2dm': 'diabetes',
            't1dm': 'diabetes',
            'hypertension': 'hypertension',
            'hypertensive': 'hypertension',
            'high blood pressure': 'hypertension',
            'htn': 'hypertension',
            'asthma': 'asthma',
            'asthmatic': 'asthma',
            'copd': 'copd',
            'chronic obstructive': 'copd',
            'emphysema': 'emphysema',
            'cancer': 'cancer',
            'malignant': 'malignant',
            'neoplasm': 'neoplasm',
            'oncology': 'cancer',
            'tumor': 'tumor',
            'carcinoma': 'carcinoma',
            'heart failure': 'heart failure',
            'chf': 'heart failure',
            'congestive': 'heart failure',
            'cardiac failure': 'heart failure',
            'coronary': 'coronary',
            'cad': 'coronary',
            'heart disease': 'heart',
            'cardiovascular': 'cardio',
            'atrial fibrillation': 'atrial fibrillation',
            'afib': 'atrial fibrillation',
            'a-fib': 'atrial fibrillation',
            'stroke': 'stroke',
            'cva': 'cerebrovascular',
            'cerebrovascular': 'cerebrovascular',
            'depression': 'depress',
            'depressive': 'depress',
            'depressed': 'depress',
            'major depressive': 'depress',
            'mdd': 'depress',
            'anxiety': 'anxiety',
            'anxious': 'anxiety',
            'gad': 'anxiety',
            'bipolar': 'bipolar',
            'schizophrenia': 'schizophreni',
            'ptsd': 'stress disorder',
            'obesity': 'obesi',
            'obese': 'obesi',
            'overweight': 'obesi',
            'bmi': 'obesi',
            'ckd': 'kidney',
            'kidney disease': 'kidney',
            'renal': 'renal',
            'chronic kidney': 'kidney',
            'esrd': 'kidney',
            'end stage renal': 'renal',
            'dialysis': 'dialysis',
            'arthritis': 'arthritis',
            'rheumatoid': 'rheumatoid',
            'osteoarthritis': 'osteoarthritis',
            'joint pain': 'arthritis',
            'pneumonia': 'pneumonia',
            'respiratory infection': 'respiratory',
            'bronchitis': 'bronchitis',
            'influenza': 'influenza',
            'flu': 'influenza',
            'covid': 'covid',
            'sepsis': 'sepsis',
            'septic': 'sepsis',
            'urinary tract': 'urinary',
            'uti': 'urinary',
            'dvt': 'thrombosis',
            'thrombosis': 'thrombosis',
            'pulmonary embolism': 'pulmonary embolism',
            'pe': 'pulmonary embolism',
            'anemia': 'anemia',
            'thyroid': 'thyroid',
            'hypothyroid': 'thyroid',
            'hyperthyroid': 'thyroid',
            'dementia': 'dementia',
            'alzheimer': 'alzheimer',
            'epilepsy': 'epilep',
            'seizure': 'epilep',
            'migraine': 'migraine',
            'headache': 'headache',
            'back pain': 'back pain',
            'low back': 'back pain',
            'fracture': 'fracture',
            'osteoporosis': 'osteoporosis',
            'cellulitis': 'cellulitis',
            'skin infection': 'skin infection',
            'pancreatitis': 'pancreatit',
            'liver disease': 'liver',
            'hepatitis': 'hepatit',
            'cirrhosis': 'cirrhosis',
            'gerd': 'reflux',
            'acid reflux': 'reflux',
            'ulcer': 'ulcer',
            'colitis': 'colitis',
            'crohns': 'crohn',
            'ibs': 'irritable bowel',
            'sleep apnea': 'sleep apnea',
            'insomnia': 'insomnia',
            'substance abuse': 'substance',
            'alcohol': 'alcohol',
            'opioid': 'opioid',
            'overdose': 'overdose',
        }

    def parse(self, question: str) -> ParsedIntent:
        normalized = normalize_typos(question.lower().strip())

        intent = ParsedIntent(
            original_question=question,
            normalized_question=normalized,
            intent='unknown',
        )

        words = re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', normalized)

        self._detect_intent(intent, normalized, words)

        self._extract_entities(intent, normalized, words)

        self._extract_filters(intent, normalized, words)

        self._extract_temporal(intent, normalized, words)

        self._extract_comparison(intent, normalized, words)

        self._extract_aggregation(intent, normalized, words)

        self._extract_ordering(intent, normalized, words)

        self._extract_negation(intent, normalized, words)

        self._infer_cross_table_needs(intent, normalized, words)

        intent.confidence = self._calculate_confidence(intent)

        return intent

    def _detect_intent(self, intent: ParsedIntent, q: str, words: List[str]):
        _has_count_kw = any(p in q for p in ['how many', 'count of', 'number of', 'total number',
                                              'how much', 'count the'])
        if not _has_count_kw and re.search(r'\bcount\b', q):
            _has_count_kw = True
        if _has_count_kw:
            intent.intent = 'count'
            if 'how much' in q and any(w in q for w in ['cost', 'spend', 'paid', 'billed']):
                intent.intent = 'aggregate'
                intent.sub_intent = 'sum'
            return

        _has_rate_kw = any(p in q for p in ['percentage', 'percent', 'rate', '% of', 'proportion',
                                             'what fraction', 'ratio'])
        if _has_rate_kw:
            _has_superlative_ctx = any(p in q for p in ['highest ', 'lowest ', 'most ', 'least ',
                                                         'biggest ', 'smallest ', 'top '])
            _has_which_what_ctx = bool(re.search(r'\b(?:which|what|who)\b', q))
            if _has_superlative_ctx or (_has_which_what_ctx and _has_superlative_ctx):
                pass
            else:
                intent.intent = 'rate'
                intent.sub_intent = 'percentage'
                return

        if any(p in q for p in ['trend', 'over time', 'month over month', 'quarter over quarter',
                                 'year over year', 'growth', 'trajectory', 'change over',
                                 'per month', 'per quarter', 'per year', 'per day',
                                 'by month', 'by quarter', 'by year',
                                 'each month', 'each quarter', 'each year',
                                 'monthly', 'quarterly', 'yearly', 'annually']):
            intent.intent = 'trend'
            intent.temporal = True
            return

        if any(p in q for p in [' vs ', ' versus ', 'compare', 'comparison', 'compared to',
                                 'difference between', 'higher than', 'lower than',
                                 'more than', 'less than', 'spending more']):
            intent.intent = 'compare'
            return


        superlative_words = ['highest ', 'lowest ', 'most ', 'least ',
                             'biggest ', 'smallest ', 'largest ']
        has_superlative = any(p in q for p in superlative_words)
        has_top_n = bool(re.search(r'top\s+\d+', q))
        has_top_bare = bool(re.search(r'\btop\s+(?!of\b)\w+', q)) and not has_top_n
        has_rank_kw = any(p in q for p in ['best ', 'worst ', 'leading ', 'rank', 'outlier'])
        has_which_what = bool(re.search(r'\b(?:which|what|who)\b.*\b(?:the\s+)?(?:most|highest|lowest|least)\b', q))

        if has_superlative or has_top_n or has_top_bare or has_rank_kw or has_which_what:

            groupable_entities = [
                'provider', 'doctor', 'physician', 'specialist',
                'region', 'facility', 'department', 'specialty',
                'payer', 'plan', 'diagnosis', 'diagnoses', 'medication',
                'member', 'patient', 'hospital', 'clinic', 'insurer',
                'category', 'type', 'group', 'age group', 'gender',
                'state', 'county', 'zip', 'city',
            ]
            has_groupable = any(ge in q for ge in groupable_entities)
            has_by_pattern = bool(re.search(r'\bby\s+\w+', q))

            should_rank = False
            if has_top_n:
                should_rank = True
            elif has_top_bare:
                should_rank = True
            elif has_rank_kw:
                should_rank = True
            elif has_which_what or has_superlative:
                if has_groupable or has_by_pattern:
                    should_rank = True

            if not should_rank and (has_superlative or has_which_what):
                intent.intent = 'aggregate'
                if any(w in q for w in ['lowest', 'least', 'smallest', 'fewest']):
                    intent.sub_intent = 'min'
                    intent.agg_function = 'MIN'
                else:
                    intent.sub_intent = 'max'
                    intent.agg_function = 'MAX'
                intent.order_by = 'desc' if intent.sub_intent == 'max' else 'asc'
                return

            if should_rank:
                intent.intent = 'rank'
                intent.order_by = 'desc'
                if any(w in q for w in ['lowest', 'worst', 'smallest', 'least', 'fewest']):
                    intent.order_by = 'asc'
                limit_match = re.search(r'top\s+(\d+)', q)
                if limit_match:
                    intent.limit = int(limit_match.group(1))
                elif has_which_what and has_groupable:
                    intent.limit = 1
                else:
                    intent.limit = 10
                return

        if any(p in q for p in ['correlation', 'correlate', 'relationship between',
                                 'associated with', 'impact of', 'effect of',
                                 'drive', 'drives', 'driven by', 'influence']):
            intent.intent = 'correlate'
            return

        if any(p in q for p in ['average', 'avg', 'mean', 'total', 'sum', 'maximum', 'max ',
                                 'minimum', 'min ', 'median']):
            intent.intent = 'aggregate'
            if any(w in q for w in ['average', 'avg', 'mean']):
                intent.sub_intent = 'avg'
                intent.agg_function = 'AVG'
            elif any(w in q for w in ['total', 'sum']):
                intent.sub_intent = 'sum'
                intent.agg_function = 'SUM'
            elif any(w in q for w in ['maximum', 'max ']):
                intent.sub_intent = 'max'
                intent.agg_function = 'MAX'
            elif any(w in q for w in ['minimum', 'min ']):
                intent.sub_intent = 'min'
                intent.agg_function = 'MIN'
            elif 'median' in q:
                intent.sub_intent = 'median'
                intent.agg_function = 'MEDIAN'
            return

        if any(p in q for p in ['are there', 'is there', 'find ', 'show me',
                                 'list ', 'which ', 'who ', 'what are',
                                 'members with', 'members who', 'patients with',
                                 'providers who', 'claims that']):
            intent.intent = 'list'
            if any(p in q for p in ['are there any', 'is there any', 'do we have any']):
                intent.intent = 'exists'
            if any(p in q for p in ['switched', 'changed', 'moved', 'transferred']):
                intent.sub_intent = 'transition'
            return

        if any(p in q for p in ['summary', 'overview', 'everything about', 'profile', 'snapshot']):
            intent.intent = 'summary'
            return

        if any(p in q for p in ['break down', 'breakdown', 'distribution']):
            intent.intent = 'aggregate'
            intent.sub_intent = 'count'
            return

        if any(p in q for p in ['by ', 'per ', 'for each ', 'in each ', 'breakdown', 'grouped by']):
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
                intent.sub_intent = 'count'
            return

        intent.intent = 'count'
        intent.ambiguous = True

    def _extract_entities(self, intent: ParsedIntent, q: str, words: List[str]):

        table_matches = self.graph.find_tables_for_question(words, raw_question=q)
        intent.tables = [t for t, s in table_matches[:3]]

        col_matches = self.graph.find_columns_for_words(
            words, intent.tables or None, raw_question=q
        )
        intent.columns = [(t, c) for t, c, st, conf in col_matches[:10]]

        if not intent.tables:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            table_matches = self.graph.find_tables_for_question(bigrams, raw_question=q)
            intent.tables = [t for t, s in table_matches[:3]]

        if not intent.tables:
            intent.tables = ['claims']
            intent.ambiguous = True

    def _extract_filters(self, intent: ParsedIntent, q: str, words: List[str]):

        for keyword, (col, val) in self.value_map.items():
            if not re.search(r'(?<![a-z])' + re.escape(keyword) + r'(?![a-z])', q):
                continue

            tables_with_col = self.graph.column_to_tables.get(col, [])
            table_hint = ''
            if tables_with_col:
                for t in tables_with_col:
                    tc = self.graph.tables.get(t)
                    if tc:
                        names = {t, tc.concept} | set(tc.synonyms)
                        if any(n in q for n in names if len(n) >= 3):
                            table_hint = t
                            break
                if not table_hint:
                    for t in intent.tables:
                        if t in tables_with_col:
                            table_hint = t
                            break
                if not table_hint:
                    table_hint = tables_with_col[0]

            intent.filters.append(ParsedFilter(
                column=col, operator='=', value=val,
                table_hint=table_hint, confidence=0.9
            ))

        for disease_name, pattern in self.disease_patterns.items():
            if disease_name in q:
                intent.filters.append(ParsedFilter(
                    column='ICD10_DESCRIPTION', operator='LIKE',
                    value=f'%{pattern}%',
                    table_hint='diagnoses', confidence=0.85
                ))
                if 'diagnoses' not in intent.tables:
                    intent.tables.append('diagnoses')

        num_patterns = [
            (r'(?:above|over|greater than|more than|higher than|exceeds?|>=?)\s+([\d.]+)', '>'),
            (r'(?:below|under|less than|lower than|fewer than|<=?)\s+([\d.]+)', '<'),
            (r'(?:equal to|equals?|exactly|==?)\s+([\d.]+)', '='),
            (r'(?:between)\s+([\d.]+)\s+(?:and)\s+([\d.]+)', 'BETWEEN'),
        ]
        for pattern, op in num_patterns:
            match = re.search(pattern, q)
            if match:
                pre_text = q[:match.start()]
                col_matches = self.graph.find_columns_for_words(
                    re.findall(r'[a-z]+', pre_text), intent.tables
                )
                agg_cols = [(t,c) for t,c,st,conf in col_matches if st.aggregatable]
                if agg_cols:
                    table, col = agg_cols[0]
                    if op == 'BETWEEN':
                        intent.filters.append(ParsedFilter(
                            column=col, operator='BETWEEN',
                            value=(float(match.group(1)), float(match.group(2))),
                            table_hint=table, confidence=0.8
                        ))
                    else:
                        intent.filters.append(ParsedFilter(
                            column=col, operator=op,
                            value=float(match.group(1)),
                            table_hint=table, confidence=0.8
                        ))

        year_match = re.search(r'\b(?:in|during|for|of)\s+(20\d{2})\b', q)
        if year_match:
            intent.time_range = year_match.group(1)

        if 'past year' in q or 'last year' in q:
            intent.time_range = 'last_year'
        elif 'this year' in q:
            intent.time_range = 'this_year'

    def _extract_temporal(self, intent: ParsedIntent, q: str, words: List[str]):
        if any(p in q for p in ['per month', 'by month', 'each month', 'monthly']):
            intent.temporal = True
            intent.time_granularity = 'month'
        elif any(p in q for p in ['per quarter', 'by quarter', 'quarterly', 'quarter over quarter']):
            intent.temporal = True
            intent.time_granularity = 'quarter'
        elif any(p in q for p in ['per year', 'by year', 'annually', 'yearly', 'year over year']):
            intent.temporal = True
            intent.time_granularity = 'year'
        elif any(p in q for p in ['per day', 'by day', 'daily']):
            intent.temporal = True
            intent.time_granularity = 'day'
        elif any(p in q for p in ['over time', 'trend', 'trajectory']):
            intent.temporal = True
            intent.time_granularity = 'month'
            intent.trend = True

    def _extract_comparison(self, intent: ParsedIntent, q: str, words: List[str]):
        concept_vs = re.search(
            r'(\w+)\s+(?:spending|costs?|expenses?|payments?|amounts?)\s+(?:vs|versus|compared to|or)\s+(\w+)',
            q
        )
        if concept_vs:
            intent.comparison = True
            intent.compare_values = [concept_vs.group(1), concept_vs.group(2)]
        else:
            vs_match = re.search(r'(\w+)\s+(?:vs|versus|compared to|or)\s+(\w+)', q)
            if vs_match:
                intent.comparison = True
                intent.compare_values = [vs_match.group(1), vs_match.group(2)]

        spending_match = re.search(r'(?:more|spending)\s+(?:on|for)\s+(\w+)\s+or\s+(\w+)', q)
        if spending_match:
            intent.comparison = True
            intent.compare_values = [spending_match.group(1), spending_match.group(2)]

        group_patterns = [
            (r'(?:by|per|for each|in each|across)\s+(region)', 'KP_REGION'),
            (r'(?:by|per|for each|in each|across)\s+(plan\s*type)', 'PLAN_TYPE'),
            (r'(?:by|per|for each|in each|across)\s+(specialty|specialties)', 'SPECIALTY'),
            (r'(?:by|per|for each|in each|across)\s+(facility|facilities)', 'FACILITY'),
            (r'(?:by|per|for each|in each|across)\s+(department)', 'DEPARTMENT'),
            (r'(?:by|per|for each|in each|across)\s+(visit\s*type)', 'VISIT_TYPE'),
            (r'(?:by|per|for each|in each|across)\s+(medication\s*class)', 'MEDICATION_CLASS'),
            (r'(?:by|per|for each|in each|across)\s+(gender)', 'GENDER'),
            (r'(?:by|per|for each|in each|across)\s+(race(?:/ethnicity)?)', 'RACE'),
            (r'(?:by|per|for each|in each|across)\s+(ethnicity)', 'RACE'),
            (r'(?:by|per|for each|in each|across)\s+(provider)', 'RENDERING_NPI'),
            (r'(?:by|per|for each|in each|across)\s+(member)', 'MEMBER_ID'),
            (r'(?:by|per|for each|in each|across)\s+(status)', 'STATUS'),
            (r'(?:by|per|for each|in each|across)\s+(claim\s*status)', 'CLAIM_STATUS'),
            (r'(?:by|per|for each|in each|across)\s+(claim\s*type)', 'CLAIM_TYPE'),
            (r'(?:by|per|for each|in each|across)\s+(severity)', 'SEVERITY'),
            (r'(?:by|per|for each|in each|across)\s+(urgency)', 'URGENCY'),
            (r'(?:by|per|for each|in each|across)\s+(diagnosis\s*type)', 'DIAGNOSIS_TYPE'),
            (r'(?:by|per|for each|in each|across)\s+(category)', 'CATEGORY'),
            (r'(?:by|per|for each|in each|across)\s+(drug\s*class)', 'DRUG_CLASS'),
            (r'(?:by|per|for each|in each|across)\s+(payer)', 'PLAN_TYPE'),
            (r'(status)\s+(?:breakdown|distribution)', 'STATUS'),
            (r'(claim\s*status)\s+(?:breakdown|distribution)', 'CLAIM_STATUS'),
            (r'(claim\s*type)\s+(?:breakdown|distribution)', 'CLAIM_TYPE'),
            (r'(visit\s*type)\s+(?:breakdown|distribution)', 'VISIT_TYPE'),
            (r'(specialty)\s+(?:breakdown|distribution)', 'SPECIALTY'),
            (r'(region)\s+(?:breakdown|distribution)', 'KP_REGION'),
            (r'(plan\s*type)\s+(?:breakdown|distribution)', 'PLAN_TYPE'),
            (r'(gender)\s+(?:breakdown|distribution)', 'GENDER'),
            (r'(severity)\s+(?:breakdown|distribution)', 'SEVERITY'),
            (r'(urgency)\s+(?:breakdown|distribution)', 'URGENCY'),
            (r'(department)\s+(?:breakdown|distribution)', 'DEPARTMENT'),
            (r'(medication\s*class)\s+(?:breakdown|distribution)', 'MEDICATION_CLASS'),
            (r'(denial\s*reason)\s+(?:breakdown|distribution)', 'DENIAL_REASON'),
        ]
        for pattern, col in group_patterns:
            if re.search(pattern, q):
                tables_with_col = self.graph.column_to_tables.get(col, [])

                if not tables_with_col or col == 'STATUS':
                    PREFIXED_VARIANTS = {
                        'STATUS': ['CLAIM_STATUS', 'ENCOUNTER_STATUS', 'STATUS'],
                    }
                    for variant in PREFIXED_VARIANTS.get(col, [col]):
                        variant_tables = self.graph.column_to_tables.get(variant, [])
                        for t in intent.tables:
                            if t in variant_tables:
                                col = variant
                                tables_with_col = variant_tables
                                break
                        if tables_with_col:
                            break

                table = intent.tables[0] if intent.tables else 'claims'
                if tables_with_col:
                    for t in intent.tables:
                        if t in tables_with_col:
                            table = t
                            break
                    else:
                        table = tables_with_col[0]
                intent.group_by.append((table, col))

    def _extract_aggregation(self, intent: ParsedIntent, q: str, words: List[str]):
        if intent.intent in ('aggregate', 'rank', 'correlate', 'compare'):
            MEASURE_WORDS = {'cost', 'spend', 'expense', 'payment', 'amount', 'billed',
                             'paid', 'money', 'dollar', 'financial', 'copay', 'charge',
                             'reimbursement', 'bill', 'denied', 'denial'}
            target_table = None
            for t, tc in self.graph.tables.items():
                name_variants = ({t, tc.concept} | set(tc.synonyms)) - MEASURE_WORDS
                for variant in name_variants:
                    if len(variant) >= 3 and re.search(r'(?<![a-z])' + re.escape(variant) + r'(?![a-z])', q):
                        target_table = t
                        break
                if target_table:
                    break

            candidates = []
            for key in self.graph.word_map:
                if ' ' in key and key in q:
                    for table, col, conf in self.graph.word_map[key]:
                        sem_type = self.graph.columns.get(table, {}).get(col)
                        if sem_type and sem_type.aggregatable:
                            candidates.append((table, col, conf + 0.1))
            for word in words:
                if word in self.graph.word_map:
                    for table, col, conf in self.graph.word_map[word]:
                        sem_type = self.graph.columns.get(table, {}).get(col)
                        if sem_type and sem_type.aggregatable:
                            candidates.append((table, col, conf))
            candidates.sort(key=lambda x: -x[2])

            if candidates:
                if target_table:
                    target_matches = [(t, c, s) for t, c, s in candidates if t == target_table]
                    if target_matches:
                        intent.agg_table, intent.agg_column, _ = target_matches[0]
                    else:
                        intent.agg_table, intent.agg_column, _ = candidates[0]
                else:
                    intent.agg_table, intent.agg_column, _ = candidates[0]
                if intent.agg_table not in intent.tables:
                    intent.tables.append(intent.agg_table)

            if not intent.sub_intent:
                intent.sub_intent = 'count'
            if not intent.agg_function:
                intent.agg_function = {
                    'avg': 'AVG', 'sum': 'SUM', 'max': 'MAX',
                    'min': 'MIN', 'count': 'COUNT',
                }.get(intent.sub_intent, 'COUNT')

        if 'pmpm' in q or 'per member per month' in q:
            intent.intent = 'aggregate'
            intent.sub_intent = 'per_unit'
            intent.agg_column = 'PAID_AMOUNT'
            intent.agg_table = 'claims'
            if 'claims' not in intent.tables:
                intent.tables.append('claims')
            if 'members' not in intent.tables:
                intent.tables.append('members')

    def _extract_ordering(self, intent: ParsedIntent, q: str, words: List[str]):
        top_match = re.search(r'top\s+(\d+)', q)
        if top_match:
            intent.limit = int(top_match.group(1))
            if not intent.order_by:
                intent.order_by = 'desc'

        if any(w in q for w in ['highest', 'most', 'largest', 'biggest', 'best', 'maximum']):
            intent.order_by = 'desc'

        if any(w in q for w in ['lowest', 'least', 'smallest', 'fewest', 'worst', 'minimum']):
            intent.order_by = 'asc'

    def _extract_negation(self, intent: ParsedIntent, q: str, words: List[str]):
        neg_patterns = ['not ', "don't", "doesn't", 'without', 'excluding',
                        'no claims', 'no visits', 'no encounters', 'never',
                        'zero ', 'none ']
        if any(p in q for p in neg_patterns):
            intent.negation = True

    def _infer_cross_table_needs(self, intent: ParsedIntent, q: str, words: List[str]):
        if intent.agg_table and intent.group_by:
            for grp_table, grp_col in intent.group_by:
                if grp_table != intent.agg_table:
                    if grp_table not in intent.tables:
                        intent.tables.append(grp_table)
                        logger.debug("Cross-table inference: added group table %s", grp_table)
                    if intent.agg_table not in intent.tables:
                        intent.tables.append(intent.agg_table)
                        logger.debug("Cross-table inference: added agg table %s", intent.agg_table)

        if re.search(r'\bper\s+(?:member|patient|enrollee)\b', q):
            if 'members' not in intent.tables:
                intent.tables.append('members')
                logger.debug("Cross-table inference: added members for 'per member' pattern")

        if re.search(r'\bby\s+age\b', q):
            if 'members' not in intent.tables:
                intent.tables.append('members')
                logger.debug("Cross-table inference: added members for 'by age' pattern")

        if intent.intent in ('count', 'aggregate'):
            raw_q = getattr(intent, 'original_question', '') or q
            q_lower = raw_q.lower()
            if re.search(r'\b(?:unique|distinct|total)\s+(?:\w+\s+)*providers?\b', q_lower) or \
               re.search(r'\bproviders?\s+(?:\w+\s+)*(?:unique|distinct|total)\b', q_lower):
                if 'providers' not in intent.tables:
                    intent.tables.append('providers')
                    logger.debug("Cross-table inference: added providers for 'unique providers' pattern")
                intent.intent = 'count'
                intent.agg_column = 'NPI'
                intent.agg_table = 'providers'
                intent.distinct = True
                if 'providers' in intent.tables:
                    intent.tables.remove('providers')
                    intent.tables.insert(0, 'providers')
                logger.debug("Cross-table inference: unique providers → COUNT(DISTINCT NPI) FROM providers")

        if any(w in q for w in ['deductible', 'deductibles']):
            if intent.intent == 'aggregate' and not intent.agg_column:
                intent.agg_column = 'DEDUCTIBLE'
                intent.agg_table = 'claims'
                if 'claims' not in intent.tables:
                    intent.tables.append('claims')
                logger.debug("Cross-table inference: resolved deductible → claims.DEDUCTIBLE")

        CROSS_TABLE_CONCEPTS = {
            'diagnos': ('diagnoses', 'ICD10_DESCRIPTION', 'claims', 'PAID_AMOUNT', 'SUM'),
            'diagnosis': ('diagnoses', 'ICD10_DESCRIPTION', 'claims', 'PAID_AMOUNT', 'SUM'),
        }

        raw_q = getattr(intent, 'original_question', '') or q
        q_lower = raw_q.lower()

        if intent.intent == 'rank' and not intent.group_by:
            for concept_key, (grp_tbl, grp_col, agg_tbl, agg_col, agg_fn) in CROSS_TABLE_CONCEPTS.items():
                if concept_key in q_lower and any(w in q_lower for w in ['cost', 'amount', 'paid', 'spend']):
                    intent.group_by.append((grp_tbl, grp_col))
                    if not intent.agg_column:
                        intent.agg_column = agg_col
                        intent.agg_table = agg_tbl
                        intent.agg_function = agg_fn
                    if grp_tbl not in intent.tables:
                        intent.tables.append(grp_tbl)
                    if agg_tbl not in intent.tables:
                        intent.tables.append(agg_tbl)
                    logger.debug("Cross-table concept: %s by cost → %s.%s grouped by %s.%s",
                                concept_key, agg_tbl, agg_col, grp_tbl, grp_col)
                    break

        if re.search(r'\bcost\s+per\s+member\b', q_lower):
            if not intent.agg_column:
                intent.agg_column = 'PAID_AMOUNT'
                intent.agg_table = 'claims'
                intent.agg_function = 'SUM'
            if 'claims' not in intent.tables:
                intent.tables.append('claims')
            if 'members' not in intent.tables:
                intent.tables.append('members')
            if re.search(r'\bby\s+age\s+group\b', q_lower):
                if not intent.group_by:
                    intent.group_by.append(('members', 'AGE_GROUP'))
                intent.sub_intent = 'per_unit'

        if intent.intent in ('aggregate', 'rank') and not intent.agg_column:
            MEASURE_TO_COL = {
                'cost': ('claims', 'PAID_AMOUNT'),
                'paid': ('claims', 'PAID_AMOUNT'),
                'spend': ('claims', 'PAID_AMOUNT'),
                'billed': ('claims', 'BILLED_AMOUNT'),
                'copay': ('claims', 'COPAY'),
                'coinsurance': ('claims', 'COINSURANCE'),
                'deductible': ('claims', 'DEDUCTIBLE'),
                'allowed': ('claims', 'ALLOWED_AMOUNT'),
                'risk': ('members', 'RISK_SCORE'),
                'length of stay': ('encounters', 'LENGTH_OF_STAY'),
                'los': ('encounters', 'LENGTH_OF_STAY'),
                'rvu': ('cpt_codes', 'RVU'),
                'panel': ('providers', 'PANEL_SIZE'),
                'days supply': ('prescriptions', 'DAYS_SUPPLY'),
            }
            for measure_word, (m_table, m_col) in MEASURE_TO_COL.items():
                if measure_word in q_lower:
                    intent.agg_column = m_col
                    intent.agg_table = m_table
                    if m_table not in intent.tables:
                        intent.tables.append(m_table)
                    logger.debug("Fallback measure resolution: %s → %s.%s", measure_word, m_table, m_col)
                    break

        if 'provider' in q_lower and any(w in q_lower for w in ['denial', 'denied', 'reject']):
            if intent.intent in ('rank', 'rate', 'aggregate') and not intent.group_by:
                intent.group_by.append(('claims', 'RENDERING_NPI'))
                if 'claims' not in intent.tables:
                    intent.tables.append('claims')
                logger.debug("Cross-table inference: providers + denial → GROUP BY RENDERING_NPI")
            if not any(f.column == 'CLAIM_STATUS' for f in intent.filters):
                intent.intent = 'rate'
                intent.sub_intent = 'percentage'
                from intent_parser import ParsedFilter
                intent.filters.append(ParsedFilter(
                    column='CLAIM_STATUS', operator='=', value='DENIED',
                    table_hint='claims', confidence=0.9
                ))

        if re.search(r'\b(?:which|what)\s+region\b', q_lower) or re.search(r'\bby\s+region\b', q_lower):
            correct_group = ('members', 'KP_REGION')
            if any(gt == 'members' and gc == 'MEMBER_ID' for gt, gc in intent.group_by):
                intent.group_by = [correct_group]
            elif not any(gc == 'KP_REGION' for _, gc in intent.group_by):
                intent.group_by = [correct_group]
            if 'members' not in intent.tables:
                intent.tables.append('members')
            if 'cost' in q_lower and 'member' in q_lower:
                intent.sub_intent = 'per_unit'
                if not intent.agg_column:
                    intent.agg_column = 'PAID_AMOUNT'
                    intent.agg_table = 'claims'
                if 'claims' not in intent.tables:
                    intent.tables.append('claims')
            logger.debug("Cross-table inference: region → GROUP BY members.KP_REGION")

        if re.search(r'\bchronic\s+condition\s+burden\b', q_lower):
            if 'diagnoses' not in intent.tables:
                intent.tables.append('diagnoses')
            if 'members' not in intent.tables:
                intent.tables.append('members')
            intent.filters.append(ParsedFilter(
                column='IS_CHRONIC', operator='=', value='Y',
                table_hint='diagnoses', confidence=0.95
            ))
            if not intent.agg_column:
                intent.agg_column = 'CHRONIC_CONDITIONS'
                intent.agg_table = 'members'
                intent.agg_function = 'AVG'
                intent.sub_intent = 'avg'
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
            logger.debug("Concept expansion: chronic condition burden → DIAGNOSES + IS_CHRONIC=Y + AVG(CHRONIC_CONDITIONS)")

        if re.search(r'\bhigh[\s-]*risk\s+(?:member|patient)', q_lower):
            if 'members' not in intent.tables:
                intent.tables.append('members')
            intent.filters.append(ParsedFilter(
                column='RISK_SCORE', operator='>=', value='2.0',
                table_hint='members', confidence=0.90
            ))
            if not intent.agg_column:
                intent.agg_column = 'RISK_SCORE'
                intent.agg_table = 'members'
                intent.agg_function = 'AVG'
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
            logger.debug("Concept expansion: high risk members → MEMBERS WHERE RISK_SCORE >= 2.0")

        if re.search(r'\butilization\s+pattern', q_lower) or re.search(r'\bhealthcare\s+utilization\b', q_lower):
            if 'encounters' not in intent.tables:
                intent.tables.append('encounters')
            if 'members' not in intent.tables:
                intent.tables.append('members')
            if not intent.agg_column:
                intent.agg_column = 'ENCOUNTER_ID'
                intent.agg_table = 'encounters'
                intent.agg_function = 'COUNT'
                intent.sub_intent = 'count'
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
            logger.debug("Concept expansion: utilization patterns → ENCOUNTERS + MEMBERS + COUNT(ENCOUNTER_ID)")

        if re.search(r'\bhealth\s+equity\s+scorecard\b', q_lower):
            for tbl in ['members', 'claims', 'encounters']:
                if tbl not in intent.tables:
                    intent.tables.append(tbl)
            if not intent.group_by:
                intent.group_by.append(('members', 'RACE'))
            if not intent.agg_column:
                intent.agg_column = 'PAID_AMOUNT'
                intent.agg_table = 'claims'
                intent.agg_function = 'AVG'
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
            logger.debug("Concept expansion: health equity scorecard → MEMBERS + CLAIMS + ENCOUNTERS GROUP BY RACE")

        if re.search(r'\bcost\s*(?:&|and)\s*risk\b', q_lower):
            if 'members' not in intent.tables:
                intent.tables.append('members')
            if 'claims' not in intent.tables:
                intent.tables.append('claims')
            if not intent.agg_column:
                intent.agg_column = 'PAID_AMOUNT'
                intent.agg_table = 'claims'
                intent.agg_function = 'AVG'
            if intent.intent == 'unknown':
                intent.intent = 'aggregate'
            logger.debug("Concept expansion: cost & risk → MEMBERS + CLAIMS AVG(PAID_AMOUNT) + RISK_SCORE")

    def _calculate_confidence(self, intent: ParsedIntent) -> float:
        score = 0.5

        if intent.intent != 'unknown':
            score += 0.15
        if intent.tables:
            score += 0.1
        if intent.columns:
            score += 0.1
        if intent.filters:
            score += 0.05
        if intent.group_by:
            score += 0.05
        if intent.agg_column:
            score += 0.05
        if intent.ambiguous:
            score -= 0.2

        return min(max(score, 0.1), 1.0)
