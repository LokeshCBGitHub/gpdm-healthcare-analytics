import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    TREND = "trend"
    BREAKDOWN = "breakdown"
    LOOKUP = "lookup"
    RANKING = "ranking"
    CORRELATION = "correlation"
    FORECAST = "forecast"
    WHAT_IF = "what_if"
    UNKNOWN = "unknown"


class EntityType(Enum):
    REGION = "region"
    PLAN_TYPE = "plan_type"
    DIAGNOSIS = "diagnosis"
    CPT_CODE = "cpt_code"
    PROVIDER_SPECIALTY = "provider_specialty"
    DATE_RANGE = "date_range"
    METRIC = "metric"
    DEMOGRAPHIC = "demographic"


@dataclass
class Entity:
    type: EntityType
    value: str
    original_text: str
    confidence: float = 0.8
    normalized_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'value': self.value,
            'original_text': self.original_text,
            'confidence': self.confidence,
            'normalized_value': self.normalized_value
        }


@dataclass
class TemporalInfo:
    period_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    relative_period: Optional[str] = None
    granularity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SQLHints:
    columns: List[str] = field(default_factory=list)
    aggregation_function: Optional[str] = None
    aggregation_column: Optional[str] = None
    group_by: List[str] = field(default_factory=list)
    order_by: List[Tuple[str, str]] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    joins_required: List[str] = field(default_factory=list)
    having_clauses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'columns': self.columns,
            'aggregation_function': self.aggregation_function,
            'aggregation_column': self.aggregation_column,
            'group_by': self.group_by,
            'order_by': self.order_by,
            'filters': self.filters,
            'joins_required': self.joins_required,
            'having_clauses': self.having_clauses
        }


@dataclass
class QueryAnalysis:
    intent: IntentType
    entities: List[Entity]
    sql_hints: SQLHints
    temporal: TemporalInfo
    aggregation_type: Optional[str] = None
    comparison_axes: List[str] = field(default_factory=list)
    is_follow_up: bool = False
    follow_up_type: Optional[str] = None
    confidence: float = 0.8
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'intent': self.intent.value,
            'entities': [e.to_dict() for e in self.entities],
            'sql_hints': self.sql_hints.to_dict(),
            'temporal': self.temporal.to_dict(),
            'aggregation_type': self.aggregation_type,
            'comparison_axes': self.comparison_axes,
            'is_follow_up': self.is_follow_up,
            'follow_up_type': self.follow_up_type,
            'confidence': self.confidence,
            'explanation': self.explanation
        }


class DeepQueryUnderstanding:

    REGIONS = {
        'ncal': ['northern california', 'ncal', 'north california', 'nor cal'],
        'scal': ['southern california', 'scal', 'south california', 'sou cal'],
        'mas': ['massachusetts', 'mas', 'boston'],
        'mid_atlantic': ['mid-atlantic', 'mid atlantic', 'philadelphia'],
        'georgia': ['georgia', 'ga'],
        'florida': ['florida', 'fl'],
        'colorado': ['colorado', 'co']
    }

    PLAN_TYPES = {
        'hmo': ['hmo', 'health maintenance'],
        'ppo': ['ppo', 'preferred provider'],
        'ma': ['ma plan', 'medicare advantage', 'ma'],
        'hra': ['hra', 'health reimbursement'],
        'epo': ['epo', 'exclusive provider']
    }

    PROVIDER_SPECIALTIES = {
        'cardiology': ['cardiology', 'cardiologist', 'cardiac', 'heart'],
        'orthopedics': ['orthopedic', 'orthopedics', 'bone', 'joint'],
        'primary_care': ['primary care', 'primary', 'family medicine', 'internist'],
        'psychiatry': ['psychiatry', 'psychiatric', 'mental health', 'psych'],
        'neurology': ['neurology', 'neurologist', 'neuro'],
        'oncology': ['oncology', 'cancer', 'tumor'],
        'pediatrics': ['pediatrics', 'pediatric', 'children'],
        'surgery': ['surgery', 'surgical', 'surgeon'],
        'emergency': ['emergency', 'er', 'e.d.', 'emergency department'],
        'radiology': ['radiology', 'x-ray', 'imaging', 'mri', 'ct'],
    }

    DIAGNOSIS_PATTERNS = {
        'diabetes': [r'\bdiabetes\b', r'\bdm\b', r'\btype\s*[12]\s*diabetes'],
        'hypertension': [r'\bhypertension\b', r'\bhtn\b', r'\bhigh\s*blood\s*pressure'],
        'heart_disease': [r'\bchd\b', r'\bcoronary\b', r'\bheart\s*disease', r'\bcardiac'],
        'depression': [r'\bdepression\b', r'\bmdd\b', r'\bdepressive'],
        'asthma': [r'\basthma\b'],
        'copd': [r'\bcopd\b', r'\bchronic\s*obstructive\b'],
        'ckd': [r'\bckd\b', r'\bkidney\s*disease'],
        'obesity': [r'\bobesity\b', r'\bobese\b'],
    }

    AGGREGATION_PATTERNS = {
        'count': [
            r'\bhow\s+many\b', r'\bcount\b', r'\bnumber\s+of\b',
            r'\btotal\s+members\b', r'\btotal\s+claims\b'
        ],
        'sum': [
            r'\btotal\s+(?:spending|cost|paid|charged)',
            r'\bsum\s+(?:spending|cost|paid|charged)',
            r'\baggregate\s+(?:spending|cost)',
            r'\btotal\s+(?:amount|payments)'
        ],
        'avg': [
            r'\baverage\s+(?:cost|spending|paid)',
            r'\bmean\s+(?:cost|spending)',
            r'\bper\s+member\s+per\s+month\b',
            r'\bpmpm\b',
            r'\bper\s+claim\b'
        ],
        'min': [
            r'\bcheapest\b', r'\blowest\s+(?:cost|price)',
            r'\bminimum\b', r'\bmin\s+(?:cost|spending)'
        ],
        'max': [
            r'\bmost\s+expensive\b', r'\bhighest\s+(?:cost|price)',
            r'\bmaximum\b', r'\bmax\s+(?:cost|spending)',
            r'\btop\s+cost'
        ]
    }

    COST_METRICS = {
        'paid_amount': ['paid', 'actual', 'amount paid'],
        'billed_amount': ['billed', 'charged', 'list price'],
        'allowed_amount': ['allowed', 'negotiated'],
        'cost': ['cost', 'expense', 'spending']
    }

    COMPARISON_PATTERNS = [
        r'\bcompare\b', r'\bvs\.?\b', r'\bversus\b',
        r'\bdifference\s+between\b', r'\bhow\s+(?:different|much)',
        r'\b(?:more|less|higher|lower)\s+than\b'
    ]

    TREND_PATTERNS = [
        r'\b(?:trend|trajectory|over\s+time)\b',
        r'\b(?:monthly|quarterly|annual|yearly)\s+(?:trend|change)',
        r'\b(?:growth|decline|increase|decrease)\b',
        r'\byear\s+over\s+year\b', r'\byoy\b',
        r'\bprevious\s+(?:month|quarter|year)\b'
    ]

    RANKING_PATTERNS = [
        r'\b(?:top|bottom)\s+\d+\b',
        r'\b(?:highest|lowest)\b',
        r'\brank(?:ing)?\b',
        r'\blead(?:ing)?\b',
        r'\bmost.*(?:expensive|common|frequent|utilized)'
    ]

    BREAKDOWN_PATTERNS = [
        r'\bbreak\s+down\b', r'\bbreakdown\b',
        r'\bby\s+(?:region|plan|specialty|diagnosis)',
        r'\bsegment\s+(?:by|across)\b',
        r'\b(?:distribution|composition)\b'
    ]

    CORRELATION_PATTERNS = [
        r'\bcorrelat(?:e|ion)\b',
        r'\brelat(?:ed|ionship)\b',
        r'\bassociat(?:ed|ion)\b',
        r'\binfluence\b', r'\bimpact\b',
        r'\beffect\s+of\b'
    ]

    FORECAST_PATTERNS = [
        r'\bforecast\b', r'\bpredict\b', r'\bproject\b',
        r'\bexpect(?:ed)?\b', r'\bnext\s+(?:month|quarter|year)',
        r'\btrend\s+(?:forward|projection)'
    ]

    WHAT_IF_PATTERNS = [
        r'\bwhat\s+if\b', r'\bscenario\b',
        r'\bif\s+(?:we|they|members)\b',
        r'\bassum(?:e|ing)\b'
    ]

    LOOKUP_PATTERNS = [
        r'\b(?:show|list|get|find|fetch|retrieve)\b',
        r'\bdetails?\b', r'\binformation\b',
        r'\b(?:what|which|who)\s+(?:is|are)\b'
    ]

    TEMPORAL_PATTERNS = {
        'last_quarter': [r'\blast\s+quarter\b', r'\bq[1-4]\s+(?:of\s+)?this\s+year'],
        'this_quarter': [r'\bthis\s+quarter\b'],
        'ytd': [r'\bytd\b', r'\byear\s+to\s+date\b'],
        'last_year': [r'\blast\s+year\b', r'\bprevious\s+year\b'],
        'this_year': [r'\bthis\s+year\b'],
        'last_month': [r'\blast\s+month\b'],
        'this_month': [r'\bthis\s+month\b'],
        'last_day': [r'\blast\s+day\b', r'\byesterday\b'],
        'last_week': [r'\blast\s+week\b'],
        'this_week': [r'\bthis\s+week\b'],
        'date_range': [r'\b(?:from|between)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}']
    }

    def __init__(self):
        logger.info("Initializing DeepQueryUnderstanding engine")
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.compiled_patterns = {}

        for agg_type, patterns in self.AGGREGATION_PATTERNS.items():
            self.compiled_patterns[f'agg_{agg_type}'] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        intent_patterns = {
            'comparison': self.COMPARISON_PATTERNS,
            'trend': self.TREND_PATTERNS,
            'ranking': self.RANKING_PATTERNS,
            'breakdown': self.BREAKDOWN_PATTERNS,
            'correlation': self.CORRELATION_PATTERNS,
            'forecast': self.FORECAST_PATTERNS,
            'what_if': self.WHAT_IF_PATTERNS,
            'lookup': self.LOOKUP_PATTERNS
        }

        for intent, patterns in intent_patterns.items():
            self.compiled_patterns[f'intent_{intent}'] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        for temporal_type, patterns in self.TEMPORAL_PATTERNS.items():
            self.compiled_patterns[f'temporal_{temporal_type}'] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def analyze(self, question: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        logger.info(f"Analyzing query: {question}")

        normalized_question = question.lower().strip()

        intent, intent_confidence = self._classify_intent(normalized_question)

        entities = self._extract_entities(normalized_question)

        temporal = self._extract_temporal(normalized_question)

        cost_metric = self._detect_cost_metric(normalized_question, context)

        sql_hints = self._generate_sql_hints(
            intent, entities, normalized_question, cost_metric, context
        )

        is_follow_up, follow_up_type = self._detect_follow_up(
            normalized_question, context
        )

        aggregation_type = self._detect_aggregation_type(normalized_question)

        comparison_axes = self._extract_comparison_axes(entities)

        analysis = QueryAnalysis(
            intent=intent,
            entities=entities,
            sql_hints=sql_hints,
            temporal=temporal,
            aggregation_type=aggregation_type,
            comparison_axes=comparison_axes,
            is_follow_up=is_follow_up,
            follow_up_type=follow_up_type,
            confidence=intent_confidence,
            explanation=self._generate_explanation(
                intent, entities, temporal, cost_metric
            )
        )

        logger.info(f"Analysis complete: intent={intent.value}, confidence={intent_confidence:.2f}")
        return analysis

    def _classify_intent(self, question: str) -> Tuple[IntentType, float]:
        intent_scores = {}

        for agg_type in ['count', 'sum', 'avg', 'min', 'max']:
            patterns = self.compiled_patterns.get(f'agg_{agg_type}', [])
            if any(p.search(question) for p in patterns):
                intent_scores[IntentType.AGGREGATION] = 0.95
                break

        patterns = self.compiled_patterns.get('intent_comparison', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.COMPARISON] = 0.9

        patterns = self.compiled_patterns.get('intent_trend', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.TREND] = 0.9

        patterns = self.compiled_patterns.get('intent_ranking', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.RANKING] = 0.9

        patterns = self.compiled_patterns.get('intent_breakdown', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.BREAKDOWN] = 0.9

        patterns = self.compiled_patterns.get('intent_correlation', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.CORRELATION] = 0.85

        patterns = self.compiled_patterns.get('intent_forecast', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.FORECAST] = 0.85

        patterns = self.compiled_patterns.get('intent_what_if', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.WHAT_IF] = 0.85

        patterns = self.compiled_patterns.get('intent_lookup', [])
        if any(p.search(question) for p in patterns):
            intent_scores[IntentType.LOOKUP] = 0.8

        if intent_scores:
            intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[intent]
        else:
            intent = IntentType.UNKNOWN
            confidence = 0.3

        return intent, confidence

    def _extract_entities(self, question: str) -> List[Entity]:
        entities = []

        for region_code, keywords in self.REGIONS.items():
            for keyword in keywords:
                if keyword in question:
                    entities.append(Entity(
                        type=EntityType.REGION,
                        value=region_code,
                        original_text=keyword,
                        confidence=0.95,
                        normalized_value=region_code
                    ))
                    break

        for plan_code, keywords in self.PLAN_TYPES.items():
            for keyword in keywords:
                if keyword in question:
                    entities.append(Entity(
                        type=EntityType.PLAN_TYPE,
                        value=plan_code,
                        original_text=keyword,
                        confidence=0.95,
                        normalized_value=plan_code
                    ))
                    break

        for specialty_code, keywords in self.PROVIDER_SPECIALTIES.items():
            for keyword in keywords:
                if keyword in question:
                    entities.append(Entity(
                        type=EntityType.PROVIDER_SPECIALTY,
                        value=specialty_code,
                        original_text=keyword,
                        confidence=0.9,
                        normalized_value=specialty_code
                    ))
                    break

        for diagnosis_code, patterns in self.DIAGNOSIS_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    entities.append(Entity(
                        type=EntityType.DIAGNOSIS,
                        value=diagnosis_code,
                        original_text=match.group(0),
                        confidence=0.85,
                        normalized_value=diagnosis_code
                    ))
                    break

        if re.search(r'\bage\b.*\d+', question, re.IGNORECASE):
            age_match = re.search(r'age[:\s]+(\d+)', question, re.IGNORECASE)
            if age_match:
                entities.append(Entity(
                    type=EntityType.DEMOGRAPHIC,
                    value='age',
                    original_text=age_match.group(0),
                    confidence=0.9
                ))

        if re.search(r'\b(?:male|female|gender|men|women)\b', question, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.DEMOGRAPHIC,
                value='gender',
                original_text='gender/sex',
                confidence=0.9
            ))

        cpt_match = re.findall(r'\b\d{5}(?:\-\d{2})?\b', question)
        for code in cpt_match:
            entities.append(Entity(
                type=EntityType.CPT_CODE,
                value=code,
                original_text=code,
                confidence=0.95
            ))

        date_match = re.search(
            r'\b(?:from|between)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:to|and)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            question,
            re.IGNORECASE
        )
        if date_match:
            entities.append(Entity(
                type=EntityType.DATE_RANGE,
                value=f"{date_match.group(1)} to {date_match.group(2)}",
                original_text=date_match.group(0),
                confidence=0.95
            ))

        logger.info(f"Extracted {len(entities)} entities")
        return entities

    def _extract_temporal(self, question: str) -> TemporalInfo:
        temporal = TemporalInfo()

        temporal_matches = {
            'last_quarter': ('quarter', 'last'),
            'this_quarter': ('quarter', 'this'),
            'ytd': ('year', 'ytd'),
            'last_year': ('year', 'last'),
            'this_year': ('year', 'this'),
            'last_month': ('month', 'last'),
            'this_month': ('month', 'this'),
            'last_week': ('week', 'last'),
            'this_week': ('week', 'this'),
        }

        for temporal_key, (period, rel_period) in temporal_matches.items():
            patterns = self.compiled_patterns.get(f'temporal_{temporal_key}', [])
            if any(p.search(question) for p in patterns):
                temporal.period_type = period
                temporal.relative_period = rel_period if rel_period != 'ytd' else 'ytd'
                break

        if re.search(r'\b(?:daily|per\s+day)\b', question, re.IGNORECASE):
            temporal.granularity = 'daily'
        elif re.search(r'\b(?:weekly|per\s+week)\b', question, re.IGNORECASE):
            temporal.granularity = 'weekly'
        elif re.search(r'\b(?:monthly|per\s+month)\b', question, re.IGNORECASE):
            temporal.granularity = 'monthly'
        elif re.search(r'\b(?:quarterly|per\s+quarter)\b', question, re.IGNORECASE):
            temporal.granularity = 'quarterly'
        elif re.search(r'\b(?:annually|per\s+year|yearly)\b', question, re.IGNORECASE):
            temporal.granularity = 'yearly'

        return temporal

    def _detect_cost_metric(
        self, question: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        question_lower = question.lower()

        if re.search(r'\b(?:paid|actual|reimbursed)\b', question_lower):
            return 'PAID_AMOUNT'
        if re.search(r'\b(?:billed|charged|list\s+price)\b', question_lower):
            return 'BILLED_AMOUNT'
        if re.search(r'\b(?:allowed|negotiated|contracted)\b', question_lower):
            return 'ALLOWED_AMOUNT'

        if context and 'preferred_cost_metric' in context:
            return context['preferred_cost_metric']

        return 'PAID_AMOUNT'

    def _generate_sql_hints(
        self,
        intent: IntentType,
        entities: List[Entity],
        question: str,
        cost_metric: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SQLHints:
        hints = SQLHints()

        base_table = self._determine_base_table(question)

        if intent == IntentType.AGGREGATION:
            hints.columns = [cost_metric] if base_table == 'claims' else ['*']
            agg_type = self._detect_aggregation_type(question)
            if agg_type == 'count':
                hints.aggregation_function = 'COUNT'
                hints.aggregation_column = '*'
            elif agg_type == 'sum':
                hints.aggregation_function = 'SUM'
                hints.aggregation_column = cost_metric
            elif agg_type == 'avg':
                hints.aggregation_function = 'AVG'
                hints.aggregation_column = cost_metric
            elif agg_type == 'min':
                hints.aggregation_function = 'MIN'
                hints.aggregation_column = cost_metric
            elif agg_type == 'max':
                hints.aggregation_function = 'MAX'
                hints.aggregation_column = cost_metric

        elif intent == IntentType.BREAKDOWN:
            group_by_cols = []
            for entity in entities:
                if entity.type == EntityType.REGION:
                    group_by_cols.append('KP_REGION')
                elif entity.type == EntityType.PLAN_TYPE:
                    group_by_cols.append('PLAN_TYPE')
                elif entity.type == EntityType.PROVIDER_SPECIALTY:
                    group_by_cols.append('PROVIDER_SPECIALTY')
                elif entity.type == EntityType.DIAGNOSIS:
                    group_by_cols.append('DIAGNOSIS_CODE')

            if not group_by_cols:
                if re.search(r'\bby\s+region\b', question, re.IGNORECASE):
                    group_by_cols.append('KP_REGION')
                elif re.search(r'\bby\s+plan\b', question, re.IGNORECASE):
                    group_by_cols.append('PLAN_TYPE')
                elif re.search(r'\bby\s+specialty\b', question, re.IGNORECASE):
                    group_by_cols.append('PROVIDER_SPECIALTY')
                elif re.search(r'\bby\s+diagnosis\b', question, re.IGNORECASE):
                    group_by_cols.append('DIAGNOSIS_CODE')

            hints.group_by = group_by_cols or ['KP_REGION']
            hints.aggregation_function = 'SUM'
            hints.aggregation_column = cost_metric

        elif intent == IntentType.TREND:
            hints.group_by = ['DATE_TRUNC(SERVICE_DATE, MONTH)']
            hints.aggregation_function = 'SUM'
            hints.aggregation_column = cost_metric
            hints.order_by = [('DATE_TRUNC(SERVICE_DATE, MONTH)', 'ASC')]

        elif intent == IntentType.RANKING:
            hints.aggregation_function = 'SUM'
            hints.aggregation_column = cost_metric

            if 'specialty' in question.lower():
                hints.group_by = ['PROVIDER_SPECIALTY']
            elif 'provider' in question.lower():
                hints.group_by = ['PROVIDER_ID']
            elif 'diagnosis' in question.lower():
                hints.group_by = ['DIAGNOSIS_CODE']
            elif 'region' in question.lower():
                hints.group_by = ['KP_REGION']

            if re.search(r'\b(?:top|highest|most)\b', question, re.IGNORECASE):
                hints.order_by = [(hints.aggregation_column, 'DESC')]
            else:
                hints.order_by = [(hints.aggregation_column, 'ASC')]

            limit_match = re.search(r'\btop\s+(\d+)\b', question, re.IGNORECASE)
            if limit_match:
                hints.having_clauses.append(f"LIMIT {limit_match.group(1)}")

        elif intent == IntentType.COMPARISON:
            hints.aggregation_function = 'SUM'
            hints.aggregation_column = cost_metric

            if any(e.type == EntityType.REGION for e in entities):
                hints.group_by = ['KP_REGION']
            elif any(e.type == EntityType.PLAN_TYPE for e in entities):
                hints.group_by = ['PLAN_TYPE']

        filters = self._generate_filters(entities, context)
        hints.filters = filters

        hints.joins_required = self._determine_joins(base_table, entities)

        logger.info(f"Generated SQL hints: agg={hints.aggregation_function}, "
                   f"group_by={hints.group_by}")
        return hints

    def _determine_base_table(self, question: str) -> str:
        if any(word in question.lower() for word in ['medication', 'pharmacy', 'drug']):
            return 'pharmacy_claims'
        elif any(word in question.lower() for word in ['visit', 'encounter']):
            return 'encounters'
        elif any(word in question.lower() for word in ['provider', 'referral']):
            return 'providers'
        elif any(word in question.lower() for word in ['grievance', 'complaint']):
            return 'grievances'
        elif any(word in question.lower() for word in ['quality', 'measure']):
            return 'quality_measures'
        elif any(word in question.lower() for word in ['authorization', 'approval']):
            return 'authorizations'
        else:
            return 'claims'

    def _generate_filters(
        self, entities: List[Entity], context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        filters = []

        for entity in entities:
            if entity.type == EntityType.REGION:
                filters.append(f"KP_REGION = '{entity.normalized_value.upper()}'")
            elif entity.type == EntityType.PLAN_TYPE:
                filters.append(f"PLAN_TYPE = '{entity.normalized_value.upper()}'")
            elif entity.type == EntityType.DIAGNOSIS:
                filters.append(f"DIAGNOSIS_CODE LIKE '%{entity.value}%'")
            elif entity.type == EntityType.CPT_CODE:
                filters.append(f"CPT_CODE = '{entity.value}'")

        if context:
            if 'selected_region' in context:
                filters.append(f"KP_REGION = '{context['selected_region']}'")
            if 'date_range' in context:
                start, end = context['date_range']
                filters.append(f"SERVICE_DATE BETWEEN '{start}' AND '{end}'")

        return filters

    def _determine_joins(self, base_table: str, entities: List[Entity]) -> List[str]:
        joins = []

        if base_table == 'claims':
            if any(e.type == EntityType.PROVIDER_SPECIALTY for e in entities):
                joins.append('LEFT JOIN providers ON claims.PROVIDER_ID = providers.PROVIDER_ID')
            if any(e.type == EntityType.REGION for e in entities):
                joins.append('LEFT JOIN members ON claims.MEMBER_ID = members.MEMBER_ID')

        return joins

    def _detect_aggregation_type(self, question: str) -> Optional[str]:
        for agg_type, patterns in self.AGGREGATION_PATTERNS.items():
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            if any(p.search(question) for p in compiled):
                return agg_type
        return None

    def _extract_comparison_axes(self, entities: List[Entity]) -> List[str]:
        axes = []

        entity_type_to_axis = {
            EntityType.REGION: 'region',
            EntityType.PLAN_TYPE: 'plan_type',
            EntityType.DIAGNOSIS: 'diagnosis',
            EntityType.PROVIDER_SPECIALTY: 'specialty',
            EntityType.DEMOGRAPHIC: 'demographic'
        }

        for entity in entities:
            if entity.type in entity_type_to_axis:
                axis = entity_type_to_axis[entity.type]
                if axis not in axes:
                    axes.append(axis)

        return axes

    def _detect_follow_up(
        self, question: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        if not context or 'previous_question' not in context:
            return False, None

        previous = context['previous_question'].lower()
        current = question.lower()

        if re.search(r'\b(?:now|then|also|additionally|furthermore)\b', current):
            return True, 'explicit'

        if re.search(r'\bbreak\s+(?:that|it)\s+down\b', current):
            return True, 'breakdown'

        if re.search(r'\bby\s+(?:region|plan|specialty|diagnosis)\b', current) and \
           not re.search(r'\b(?:cost|spending|claims|members)\b', current):
            return True, 'dimensional_drill_down'

        if re.search(r'\bcompared\s+to\b', current) and \
           not re.search(r'\b(?:cost|spending|members)\b', current):
            return True, 'comparison'

        if re.search(r'\b(?:more|less|higher|lower)\b', current) and \
           not re.search(r'\b(?:cost|spending|claims)\b', current):
            return True, 'comparison'

        return False, None

    def _generate_explanation(
        self, intent: IntentType, entities: List[Entity],
        temporal: TemporalInfo, cost_metric: str
    ) -> str:
        parts = []

        intent_descriptions = {
            IntentType.AGGREGATION: "Calculate aggregate metrics",
            IntentType.COMPARISON: "Compare metrics across dimensions",
            IntentType.TREND: "Analyze trend over time",
            IntentType.BREAKDOWN: "Break down metrics by dimension",
            IntentType.RANKING: "Rank items by metric",
            IntentType.LOOKUP: "Look up specific information",
            IntentType.CORRELATION: "Find correlations",
            IntentType.FORECAST: "Forecast future values",
            IntentType.WHAT_IF: "Analyze hypothetical scenarios"
        }

        parts.append(intent_descriptions.get(intent, "Analyze query"))

        if entities:
            entity_strs = [f"{e.type.value}={e.value}" for e in entities[:3]]
            parts.append(f"Entities: {', '.join(entity_strs)}")

        if temporal.relative_period:
            parts.append(f"Time: {temporal.relative_period}")

        return " | ".join(parts)

    def get_sql_hints(self, question: str) -> Dict[str, Any]:
        analysis = self.analyze(question)
        return analysis.sql_hints.to_dict()

    def detect_follow_up(
        self, question: str, previous_question: str
    ) -> Dict[str, Any]:
        context = {'previous_question': previous_question}
        is_follow_up, follow_up_type = self._detect_follow_up(question, context)

        return {
            'is_follow_up': is_follow_up,
            'follow_up_type': follow_up_type,
            'previous_question': previous_question,
            'current_question': question
        }


if __name__ == '__main__':
    engine = DeepQueryUnderstanding()

    test_queries = [
        "What is the average cost per member in Northern California?",
        "Show me the top 10 most expensive procedures by specialty",
        "Compare HMO vs PPO spending trends over the last quarter",
        "Break down total claims by region and plan type",
        "How many members have diabetes diagnosed?",
        "What's the trend in cardiac procedures year over year?",
        "Now break that down by age group",
        "Which providers have the highest utilization rates?",
    ]

    print("=" * 80)
    print("Deep Query Understanding Engine - Example Analysis")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        analysis = engine.analyze(query)
        result = analysis.to_dict()

        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Entities: {len(result['entities'])} found")
        for entity in result['entities'][:3]:
            print(f"  - {entity['type']}: {entity['value']}")
        print(f"Aggregation: {result['aggregation_type']}")
        print(f"Temporal: {result['temporal']['relative_period']}")
        print(f"SQL Hints:")
        hints = result['sql_hints']
        if hints['aggregation_function']:
            print(f"  - Function: {hints['aggregation_function']}")
        if hints['group_by']:
            print(f"  - Group By: {hints['group_by']}")
        if hints['order_by']:
            print(f"  - Order By: {hints['order_by']}")
